# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Tiny A/B regression for selective activation checkpointing.

This is a trainer-outside probe. It compares no checkpointing, full block
checkpointing and selective checkpointing on small Anima/Newbie-like block
loops, then reports output/gradient parity and CUDA peak memory when available.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.checkpoint_policy import build_selective_checkpoint_context_fn  # noqa: E402


@dataclass
class CaseResult:
    route: str
    mode: str
    success: bool
    failed_reason: str = ""
    output_sum: float = 0.0
    loss: float = 0.0
    forward_backward_ms: float = 0.0
    peak_allocated_mb: float = 0.0
    max_abs_output_diff_vs_off: float | None = None
    max_abs_grad_diff_vs_off: float | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(float(torch.cuda.max_memory_allocated(device)) / 1024.0 / 1024.0, 3)


class _TinyAnimaBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor, adaln_lora: Any = None) -> torch.Tensor:
        del adaln_lora
        cond = emb.unsqueeze(1) * 0.01 + context.mean(dim=1, keepdim=True) * 0.01
        h = self.linear(x + cond)
        return self.norm(x + self.ff(h))


class _TinyAnimaDiT(nn.Module):
    def __init__(self, blocks: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_TinyAnimaBlock(dim) for _ in range(blocks)])
        self.anima_block_checkpointing = False
        self.anima_block_checkpointing_mode = "off"

    def set_anima_block_checkpointing(self, enabled: bool, mode: str = "block") -> None:
        normalized = str(mode or "block").strip().lower().replace("-", "_")
        self.anima_block_checkpointing = bool(enabled) and normalized in {"block", "selective"}
        self.anima_block_checkpointing_mode = normalized if self.anima_block_checkpointing else "off"

    def _checkpoint_block(self, block: nn.Module, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor, adaln_lora=None):
        from torch.utils.checkpoint import checkpoint

        kwargs: Dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": False}
        if self.anima_block_checkpointing_mode == "selective":
            context_fn = build_selective_checkpoint_context_fn("balanced")
            if context_fn is not None:
                kwargs["context_fn"] = context_fn

        def block_forward(x_arg: torch.Tensor, emb_arg: torch.Tensor, context_arg: torch.Tensor) -> torch.Tensor:
            return block(x_arg, emb_arg, context_arg, None)

        return checkpoint(block_forward, x, emb, context, **kwargs)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        use_checkpoint = self.anima_block_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.net.blocks:
            if use_checkpoint:
                x = self._checkpoint_block(block, x, emb, context)
            else:
                x = block(x, emb, context, None)
        return x


class _TinyNewbieBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.linear(x + t_emb.unsqueeze(1) * 0.01)
        return self.norm(x + torch.nn.functional.silu(h) * torch.sigmoid(self.gate(h)))


class _TinyNewbieDiT(nn.Module):
    def __init__(self, blocks: int, dim: int) -> None:
        super().__init__()
        self._block_modules = nn.ModuleList([_TinyNewbieBlock(dim) for _ in range(blocks)])
        self._gradient_checkpointing = False
        self._newbie_block_checkpointing_mode = "off"

    def set_newbie_block_checkpointing(self, enabled: bool, mode: str = "block") -> None:
        normalized = str(mode or "block").strip().lower().replace("-", "_")
        self._gradient_checkpointing = bool(enabled) and normalized in {"", "block", "selective"}
        self._newbie_block_checkpointing_mode = "selective" if self._gradient_checkpointing and normalized == "selective" else "block" if self._gradient_checkpointing else "off"

    def _run_dit_block(self, block: nn.Module, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return block(x, t_emb)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self._block_modules:
            if self._gradient_checkpointing and self.training and torch.is_grad_enabled():
                import torch.utils.checkpoint as checkpoint_mod

                kwargs: Dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": False}
                if self._newbie_block_checkpointing_mode == "selective":
                    context_fn = build_selective_checkpoint_context_fn("balanced")
                    if context_fn is not None:
                        kwargs["context_fn"] = context_fn
                x = checkpoint_mod.checkpoint(self._run_dit_block, block, x, t_emb, **kwargs)
            else:
                x = self._run_dit_block(block, x, t_emb)
        return x


def _make_model(route: str, blocks: int, dim: int) -> nn.Module:
    return _TinyAnimaDiT(blocks, dim) if route == "anima" else _TinyNewbieDiT(blocks, dim)


def _set_mode(model: nn.Module, route: str, mode: str) -> None:
    if route == "anima":
        model.set_anima_block_checkpointing(mode != "off", "selective" if mode == "selective" else "block")
    else:
        model.set_newbie_block_checkpointing(mode != "off", "selective" if mode == "selective" else "block")


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    grads: List[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().float().reshape(-1).cpu())
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)


def _run_case(route: str, mode: str, base: nn.Module, inputs: tuple[torch.Tensor, ...], target: torch.Tensor, device: torch.device) -> tuple[CaseResult, torch.Tensor | None, torch.Tensor | None]:
    model = copy.deepcopy(base).to(device).train()
    _set_mode(model, route, mode)
    local_inputs = tuple(item.detach().clone().to(device).requires_grad_(True) for item in inputs)
    target = target.detach().clone().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        _sync(device)
        started = time.perf_counter()
        output = model(*local_inputs)
        loss = torch.nn.functional.mse_loss(output.float(), target.float())
        loss.backward()
        _sync(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        return (
            CaseResult(
                route=route,
                mode=mode,
                success=True,
                output_sum=float(output.detach().float().sum().cpu().item()),
                loss=float(loss.detach().float().cpu().item()),
                forward_backward_ms=round(elapsed, 3),
                peak_allocated_mb=_peak_mb(device),
            ),
            output.detach().float().cpu(),
            _flatten_grads(model),
        )
    except Exception as exc:
        return CaseResult(route=route, mode=mode, success=False, failed_reason=f"{type(exc).__name__}: {exc}"), None, None


def run_selective_checkpoint_regression(
    *,
    device: str = "auto",
    routes: Iterable[str] = ("anima", "newbie"),
    blocks: int = 3,
    dim: int = 32,
    batch: int = 2,
    tokens: int = 8,
    seed: int = 1234,
) -> Dict[str, Any]:
    requested = str(device or "auto").lower()
    torch_device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    torch.manual_seed(seed)
    payload: Dict[str, Any] = {
        "probe": "selective_checkpoint_regression",
        "device": str(torch_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": {"blocks": int(blocks), "dim": int(dim), "batch": int(batch), "tokens": int(tokens), "seed": int(seed)},
        "routes": {},
    }
    for route in routes:
        route = str(route).strip().lower()
        if route not in {"anima", "newbie"}:
            continue
        torch.manual_seed(seed)
        base = _make_model(route, blocks, dim)
        x = torch.randn(batch, tokens, dim)
        cond = torch.randn(batch, dim)
        if route == "anima":
            context = torch.randn(batch, tokens, dim)
            inputs = (x, cond, context)
        else:
            inputs = (x, cond)
        with torch.no_grad():
            target = base(*inputs).detach() + 0.01 * torch.randn(batch, tokens, dim)
        route_results: List[CaseResult] = []
        reference_out: torch.Tensor | None = None
        reference_grad: torch.Tensor | None = None
        for mode in ("off", "block", "selective"):
            result, out, grad = _run_case(route, mode, base, inputs, target, torch_device)
            if mode == "off" and out is not None and grad is not None:
                reference_out = out
                reference_grad = grad
            elif result.success and reference_out is not None and reference_grad is not None and out is not None and grad is not None:
                result.max_abs_output_diff_vs_off = float((out - reference_out).abs().max().item())
                result.max_abs_grad_diff_vs_off = float((grad - reference_grad).abs().max().item())
            route_results.append(result)
        payload["routes"][route] = [asdict(item) for item in route_results]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--routes", default="anima,newbie")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="temp/selective_checkpoint_regression.json")
    args = parser.parse_args()
    payload = run_selective_checkpoint_regression(
        device=args.device,
        routes=[item for item in args.routes.split(",") if item.strip()],
        blocks=args.blocks,
        dim=args.dim,
        batch=args.batch,
        tokens=args.tokens,
        seed=args.seed,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

