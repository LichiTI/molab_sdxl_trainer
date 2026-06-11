# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Research probe for Anima DiT rematerialization candidates.

This probe compares current safe rematerialization-adjacent techniques on a
small DiT-like block stack: eager execution, block checkpointing, selective
checkpointing, saved-activation compression, and their combination. It is not a
true reversible-block implementation and does not change trainer defaults.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.activation_compression import ActivationCompressionContext  # noqa: E402
from core.lulynx_trainer.checkpoint_policy import build_selective_checkpoint_context_fn  # noqa: E402


MODES = (
    "eager",
    "block_checkpoint",
    "selective_checkpoint",
    "activation_compression",
    "selective_plus_activation_compression",
)


@dataclass
class RematProbeResult:
    mode: str
    success: bool
    failed_reason: str = ""
    loss: float = 0.0
    forward_backward_ms: float = 0.0
    peak_allocated_mb: float = 0.0
    max_abs_output_diff_vs_eager: float | None = None
    max_abs_grad_diff_vs_eager: float | None = None
    activation_compression: dict[str, Any] | None = None
    reversible_kernel_present: bool = False


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"} and device.type != "cpu":
        return torch.float16
    return torch.float32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(float(torch.cuda.max_memory_allocated(device)) / 1024.0 / 1024.0, 3)


class _TinyDiTBlock(nn.Module):
    def __init__(self, width: int, context_width: int, mlp_width: int) -> None:
        super().__init__()
        self.self_q = nn.Linear(width, width, bias=False)
        self.self_k = nn.Linear(width, width, bias=False)
        self.self_v = nn.Linear(width, width, bias=False)
        self.self_o = nn.Linear(width, width, bias=False)
        self.cross_k = nn.Linear(context_width, width, bias=False)
        self.cross_v = nn.Linear(context_width, width, bias=False)
        self.cross_o = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, mlp_width, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_width, width, bias=False),
        )
        self.cond = nn.Linear(width, width * 3, bias=False)
        self.norm = nn.LayerNorm(width)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = max(q.shape[-1], 1) ** -0.5
        weights = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        return weights @ v

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.cond(emb).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + self.self_o(self._attention(self.self_q(h), self.self_k(h), self.self_v(h)))
        cross = self._attention(self.self_q(self.norm(x)), self.cross_k(context), self.cross_v(context))
        x = x + self.cross_o(cross)
        return x + self.mlp(x) * torch.tanh(gate).unsqueeze(1)


class _TinyRematDiT(nn.Module):
    def __init__(self, *, blocks: int, width: int, context_width: int, mlp_width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TinyDiTBlock(width, context_width, mlp_width) for _ in range(max(int(blocks), 1))]
        )
        self.final_norm = nn.LayerNorm(width)
        self.final = nn.Linear(width, width, bias=False)
        self.checkpoint_mode = "eager"

    def set_checkpoint_mode(self, mode: str) -> None:
        self.checkpoint_mode = str(mode or "eager").strip().lower()

    def _checkpoint_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        import torch.utils.checkpoint as checkpoint_mod

        kwargs: dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": False}
        if self.checkpoint_mode in {"selective_checkpoint", "selective_plus_activation_compression"}:
            context_fn = build_selective_checkpoint_context_fn("balanced")
            if context_fn is not None:
                kwargs["context_fn"] = context_fn

        def block_forward(x_arg: torch.Tensor, emb_arg: torch.Tensor, context_arg: torch.Tensor) -> torch.Tensor:
            return block(x_arg, emb_arg, context_arg)

        return checkpoint_mod.checkpoint(block_forward, x, emb, context, **kwargs)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        use_checkpoint = self.checkpoint_mode in {
            "block_checkpoint",
            "selective_checkpoint",
            "selective_plus_activation_compression",
        }
        for block in self.blocks:
            if use_checkpoint and self.training and torch.is_grad_enabled():
                x = self._checkpoint_block(block, x, emb, context)
            else:
                x = block(x, emb, context)
        return self.final(self.final_norm(x))


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    grads: list[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().float().reshape(-1).cpu())
    return torch.cat(grads) if grads else torch.empty(0)


def _make_inputs(
    *,
    batch: int,
    tokens: int,
    width: int,
    context_tokens: int,
    context_width: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.randn(batch, tokens, width, generator=generator)
    emb = torch.randn(batch, width, generator=generator)
    context = torch.randn(batch, context_tokens, context_width, generator=generator)
    target = torch.randn(batch, tokens, width, generator=generator)
    return x, emb, context, target


def _run_case(
    *,
    mode: str,
    base: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    activation_dtype: str,
    min_tensor_bytes: int,
) -> tuple[RematProbeResult, torch.Tensor | None, torch.Tensor | None]:
    model = copy.deepcopy(base).to(device=device, dtype=dtype).train()
    model.set_checkpoint_mode(mode)
    local_inputs = tuple(item.detach().clone().to(device=device, dtype=dtype).requires_grad_(True) for item in inputs)
    local_target = target.detach().clone().to(device=device, dtype=dtype)
    compression_enabled = mode in {"activation_compression", "selective_plus_activation_compression"}
    compression = ActivationCompressionContext(
        enabled=compression_enabled,
        storage_dtype=activation_dtype,
        min_tensor_bytes=int(min_tensor_bytes),
    )
    context = compression.context() if compression_enabled else nullcontext()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        _sync(device)
        started = time.perf_counter()
        with context:
            output = model(*local_inputs)
            loss = torch.nn.functional.mse_loss(output.float(), local_target.float())
        loss.backward()
        _sync(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        return (
            RematProbeResult(
                mode=mode,
                success=True,
                loss=float(loss.detach().float().cpu().item()),
                forward_backward_ms=round(elapsed, 3),
                peak_allocated_mb=_peak_mb(device),
                activation_compression=compression.as_dict() if compression_enabled else None,
                reversible_kernel_present=False,
            ),
            output.detach().float().cpu(),
            _flatten_grads(model),
        )
    except Exception as exc:
        return (
            RematProbeResult(
                mode=mode,
                success=False,
                failed_reason=f"{type(exc).__name__}: {exc}",
                activation_compression=compression.as_dict() if compression_enabled else None,
                reversible_kernel_present=False,
            ),
            None,
            None,
        )


def run_rematerialization_probe(
    *,
    device: str = "auto",
    dtype: str = "float32",
    blocks: int = 3,
    width: int = 64,
    context_width: int = 64,
    mlp_width: int = 128,
    batch: int = 2,
    tokens: int = 16,
    context_tokens: int = 16,
    seed: int = 1234,
    modes: Iterable[str] = MODES,
    activation_dtype: str = "fp16",
    min_tensor_bytes: int = 0,
) -> dict[str, Any]:
    torch_device = _device(device)
    torch_dtype = _dtype(dtype, torch_device)
    torch.manual_seed(int(seed))
    base = _TinyRematDiT(blocks=blocks, width=width, context_width=context_width, mlp_width=mlp_width)
    x, emb, context, target = _make_inputs(
        batch=batch,
        tokens=tokens,
        width=width,
        context_tokens=context_tokens,
        context_width=context_width,
        seed=seed + 77,
    )
    selected = [str(mode).strip().lower() for mode in modes if str(mode).strip()]
    if not selected:
        selected = list(MODES)

    results: list[RematProbeResult] = []
    eager_output: torch.Tensor | None = None
    eager_grad: torch.Tensor | None = None
    for mode in selected:
        if mode not in MODES:
            results.append(RematProbeResult(mode=mode, success=False, failed_reason="unknown mode"))
            continue
        result, output, grad = _run_case(
            mode=mode,
            base=base,
            inputs=(x, emb, context),
            target=target,
            device=torch_device,
            dtype=torch_dtype,
            activation_dtype=activation_dtype,
            min_tensor_bytes=min_tensor_bytes,
        )
        if mode == "eager" and result.success and output is not None and grad is not None:
            eager_output = output
            eager_grad = grad
        elif result.success and eager_output is not None and eager_grad is not None and output is not None and grad is not None:
            result.max_abs_output_diff_vs_eager = float((output - eager_output).abs().max().item())
            result.max_abs_grad_diff_vs_eager = float((grad - eager_grad).abs().max().item())
        results.append(result)

    successful = [item for item in results if item.success]
    best_memory = min(successful, key=lambda item: item.peak_allocated_mb).mode if successful and torch_device.type == "cuda" else ""
    best_time = min(successful, key=lambda item: item.forward_backward_ms).mode if successful else ""
    return {
        "schema_version": 1,
        "probe": "anima_rematerialization_probe",
        "device": str(torch_device),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": {
            "blocks": int(blocks),
            "width": int(width),
            "context_width": int(context_width),
            "mlp_width": int(mlp_width),
            "batch": int(batch),
            "tokens": int(tokens),
            "context_tokens": int(context_tokens),
            "seed": int(seed),
        },
        "summary": {
            "best_time_mode": best_time,
            "best_peak_memory_mode": best_memory,
            "true_reversible_kernel_present": False,
            "status": "checkpoint_and_saved_tensor_remat_probe_only",
            "next_gate": "custom autograd or native reversible block must match eager gradients before runtime dispatch",
        },
        "results": [asdict(item) for item in results],
        "notes": [
            "block/selective checkpointing recomputes forward activations; it is the safe current approximation for rematerialization.",
            "activation compression changes saved tensor storage dtype and should be gated by parity and stability checks.",
            "A true reversible DiT block is not implemented by this probe.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--context-width", type=int, default=64)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--activation-dtype", default="fp16")
    parser.add_argument("--min-tensor-bytes", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = run_rematerialization_probe(
        device=args.device,
        dtype=args.dtype,
        blocks=args.blocks,
        width=args.width,
        context_width=args.context_width,
        mlp_width=args.mlp_width,
        batch=args.batch,
        tokens=args.tokens,
        context_tokens=args.context_tokens,
        seed=args.seed,
        modes=[part for part in str(args.modes).split(",") if part.strip()],
        activation_dtype=args.activation_dtype,
        min_tensor_bytes=args.min_tensor_bytes,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
