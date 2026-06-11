"""Real matrix smoke for the experimental LAB LoRA distiller.

This script loads the real SDXL topology, teacher LoRA, semantic manager, and
sidecar modules, then inspects a tiny number of K/V regression matrices.  It is
intentionally narrower than full distillation so we can validate tensor shapes,
finite values, distributions, and one optimizer step without a long training run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prepare_path() -> None:
    backend_root = _backend_root()
    project_root = backend_root.parent
    for path in (str(backend_root), str(project_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_prepare_path()

try:
    from core.tools.lulynx_lab.distiller import LoRADistiller, _align_last_dim
except ImportError:
    from backend.core.tools.lulynx_lab.distiller import LoRADistiller, _align_last_dim


DEFAULT_PROMPTS = [
    "masterpiece, best quality, ultra detailed",
    "red hair, blue eyes, sunset lighting",
]


def _dtype(value: str, device: str) -> torch.dtype:
    key = str(value or "auto").strip().lower()
    if key == "auto":
        return torch.float16 if str(device).startswith("cuda") and torch.cuda.is_available() else torch.float32
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _finite_ratio(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 1.0
    return float(torch.isfinite(tensor).float().mean().item())


def _stats(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float().flatten()
    finite = torch.isfinite(data)
    finite_data = data[finite]
    if finite_data.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite_ratio": _finite_ratio(tensor),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "abs_p50": None,
            "abs_p95": None,
            "abs_p99": None,
        }
    abs_data = finite_data.abs()
    quantiles = torch.quantile(
        abs_data,
        torch.tensor([0.5, 0.95, 0.99], device=abs_data.device),
    )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_ratio": _finite_ratio(tensor),
        "mean": float(finite_data.mean().item()),
        "std": float(finite_data.std(unbiased=False).item()) if finite_data.numel() > 1 else 0.0,
        "min": float(finite_data.min().item()),
        "max": float(finite_data.max().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p95": float(quantiles[1].item()),
        "abs_p99": float(quantiles[2].item()),
    }


def _print_record(kind: str, **payload: Any) -> None:
    print(json.dumps({"kind": kind, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _iter_layer_items(distiller: LoRADistiller, max_layers: int) -> Iterable[tuple[str, dict[str, torch.Tensor]]]:
    count = 0
    for layer_name, targets in distiller.teacher_deltas.items():
        if layer_name not in distiller.sidecar_net.neuro_modules:
            continue
        yield layer_name, targets
        count += 1
        if count >= max_layers:
            return


def _build_embedding_cache(distiller: LoRADistiller, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    cache_c = []
    cache_l = []
    with torch.no_grad():
        for prompt in prompts:
            c_emb = distiller.semantic_manager.encode_ghost_branch([prompt])
            if c_emb is None:
                raise RuntimeError("Ghost CLIP teacher returned no embeddings")
            l_emb = distiller.semantic_manager.encode_main_branch([prompt])
            cache_c.append(c_emb.to(distiller.device).to(distiller.dtype))
            cache_l.append(l_emb.to(distiller.device).to(distiller.dtype))
    return torch.cat(cache_c, dim=0), torch.cat(cache_l, dim=0)


def _layer_loss_and_stats(
    distiller: LoRADistiller,
    layer_name: str,
    targets: dict[str, torch.Tensor],
    c_emb: torch.Tensor,
    l_emb: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    student_mod = distiller.sidecar_net.neuro_modules[layer_name]
    losses = []
    records: list[dict[str, Any]] = []

    for lane, proj_name in (("k", "to_k"), ("v", "to_v")):
        if lane not in targets:
            continue
        delta = targets[lane].to(distiller.device)
        c_aligned = _align_last_dim(c_emb, int(delta.shape[1]))
        target_act = c_aligned @ delta.t()
        # Match the LAB matrix-regression objective: raw sidecar projection
        # should approximate raw LoRA delta activations. Runtime LaneNorm/gate
        # are fusion-time stabilizers and are reported separately when needed.
        pred_act = student_mod[proj_name](l_emb)
        target_pool = target_act.mean(dim=1)
        pred_pool = pred_act.mean(dim=1)
        loss = F.mse_loss(pred_pool.float(), target_pool.float())
        losses.append(loss)
        records.append({
            "layer": layer_name,
            "lane": lane,
            "delta": _stats(delta),
            "c_aligned": _stats(c_aligned),
            "target_act": _stats(target_act),
            "pred_act": _stats(pred_act),
            "pooled_loss": float(loss.detach().float().item()),
            "shape_ok": (
                c_aligned.shape[-1] == delta.shape[1]
                and target_act.shape[-1] == delta.shape[0]
                and pred_act.shape[0] == target_act.shape[0]
                and pred_act.shape[-1] == target_act.shape[-1]
            ),
        })

    if not losses:
        return torch.zeros((), device=distiller.device, dtype=torch.float32), records
    return torch.stack([loss.float() for loss in losses]).sum(), records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real LAB distiller matrix smoke.")
    parser.add_argument("--unet-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--projector-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--allow-tokenizer-only-clip", action="store_true")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = _dtype(args.dtype, device)
    max_layers = max(int(args.max_layers), 1)
    optimizer_steps = max(int(args.optimizer_steps), 0)
    prompts = args.prompts or DEFAULT_PROMPTS

    _print_record(
        "smoke_start",
        device=device,
        dtype=str(dtype),
        max_layers=max_layers,
        optimizer_steps=optimizer_steps,
        prompt_count=len(prompts),
    )

    distiller = LoRADistiller(
        unet_path=args.unet_path,
        lora_path=args.lora_path,
        llm_path=args.llm_path,
        projector_path=args.projector_path or None,
        teacher_path=args.teacher_path,
        allow_tokenizer_only_clip=bool(args.allow_tokenizer_only_clip),
        device=device,
        dtype=dtype,
    )

    _print_record(
        "loaded",
        teacher_layer_count=len(distiller.teacher_deltas),
        sidecar_layer_count=len(distiller.sidecar_net.neuro_modules),
        clip_model_kind=distiller.clip_model_kind,
        learning_rate=distiller.optimizer.param_groups[0].get("lr"),
    )

    c_emb, l_emb = _build_embedding_cache(distiller, prompts)
    _print_record("embedding", branch="clip_teacher", stats=_stats(c_emb))
    _print_record("embedding", branch="llm_projected", stats=_stats(l_emb))

    selected_layers = list(_iter_layer_items(distiller, max_layers))
    if not selected_layers:
        raise RuntimeError("No compatible teacher delta layers were found.")
    _print_record("selected_layers", layers=[name for name, _ in selected_layers])

    with torch.no_grad():
        initial_losses = []
        for layer_name, targets in selected_layers:
            loss, records = _layer_loss_and_stats(distiller, layer_name, targets, c_emb, l_emb)
            initial_losses.append(float(loss.detach().float().item()))
            for record in records:
                _print_record("matrix_initial", **record)
        _print_record(
            "loss_initial",
            total=sum(initial_losses),
            finite=all(math.isfinite(value) for value in initial_losses),
        )

    for step in range(optimizer_steps):
        distiller.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        for layer_name, targets in selected_layers:
            loss, _ = _layer_loss_and_stats(distiller, layer_name, targets, c_emb, l_emb)
            total_loss = total_loss + loss.float()
        total_loss.backward()
        grad_norm_sq = 0.0
        grad_finite = True
        for param in distiller.sidecar_net.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            grad_finite = grad_finite and bool(torch.isfinite(grad).all().item())
            grad_norm_sq += float(torch.sum(grad * grad).item())
        grad_norm = math.sqrt(max(grad_norm_sq, 0.0))
        distiller.optimizer.step()
        _print_record(
            "optimizer_step",
            step=step + 1,
            loss=float(total_loss.detach().float().item()),
            grad_norm=grad_norm,
            grad_finite=grad_finite,
        )

    with torch.no_grad():
        final_losses = []
        for layer_name, targets in selected_layers:
            loss, records = _layer_loss_and_stats(distiller, layer_name, targets, c_emb, l_emb)
            final_losses.append(float(loss.detach().float().item()))
            for record in records:
                _print_record("matrix_final", **record)
        _print_record(
            "loss_final",
            total=sum(final_losses),
            finite=all(math.isfinite(value) for value in final_losses),
        )

    if device.startswith("cuda"):
        _print_record(
            "cuda_memory",
            allocated=torch.cuda.memory_allocated(device),
            reserved=torch.cuda.memory_reserved(device),
            max_allocated=torch.cuda.max_memory_allocated(device),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
