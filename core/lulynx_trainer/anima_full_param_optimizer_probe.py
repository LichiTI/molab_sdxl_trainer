# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Research probe for Anima full-parameter optimizer backends.

This is a trainer-outside development probe. It does not enable a native
optimizer in the runtime path. The goal is to size full-parameter DiT-like
optimizer state, compare current PyTorch/Lulynx candidates, and make the gate
for a future TurboCore native optimizer explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.anima_factored_optimizer import AnimaFactoredAdamW  # noqa: E402
from core.lulynx_trainer.fused_adamw import FusedAdamW  # noqa: E402
from core.lulynx_trainer.turbocore_optimizer_performance_gate import evaluate_optimizer_performance_gate  # noqa: E402
from core.turbocore_parity import check_stateful_native_optimizer_parity  # noqa: E402


ShapeList = list[tuple[int, ...]]
OptimizerFactory = Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer]


PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"blocks": 2, "hidden": 64, "mlp": 128},
    "dit_block_short": {"blocks": 2, "hidden": 256, "mlp": 1024},
    "anima_16g_micro": {"blocks": 4, "hidden": 512, "mlp": 2048},
}


@dataclass
class OptimizerProbeResult:
    optimizer: str
    success: bool
    failed_reason: str = ""
    parameter_tensors: int = 0
    parameter_count: int = 0
    parameter_mb: float = 0.0
    state_mb: float = 0.0
    state_to_parameter_ratio: float = 0.0
    step_ms: float = 0.0
    parity_max_abs_diff: float | None = None
    parity_max_rel_diff: float | None = None
    exact_adamw_candidate: bool = False
    native_kernel_present: bool = False
    profile: dict[str, Any] | None = None


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


def _dit_like_shapes(*, blocks: int, hidden: int, mlp: int) -> ShapeList:
    shapes: ShapeList = []
    shapes.extend([(hidden, hidden), (hidden,), (hidden * 4, hidden), (hidden, hidden * 4)])
    for _ in range(max(int(blocks), 1)):
        for _attn in range(2):
            shapes.extend([(hidden, hidden), (hidden, hidden), (hidden, hidden), (hidden, hidden)])
        shapes.extend([(mlp, hidden), (hidden, mlp)])
        shapes.extend([(hidden * 3, hidden), (hidden * 3, hidden), (hidden * 3, hidden)])
        shapes.extend([(hidden,), (hidden,)])
    shapes.extend([(hidden * 2, hidden), (hidden, hidden), (hidden,)])
    return shapes


def _make_params(
    shapes: Iterable[tuple[int, ...]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    params: list[torch.nn.Parameter] = []
    for shape in shapes:
        value = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.01
        params.append(torch.nn.Parameter(value.to(device=device, dtype=dtype)))
    return params


def _clone_params(params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _assign_grads(params: Iterable[torch.nn.Parameter], *, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    grads: list[torch.Tensor] = []
    for param in params:
        grad = torch.randn(tuple(param.shape), generator=generator, dtype=torch.float32)
        grad = grad.to(device=param.device, dtype=param.dtype)
        param.grad = grad
        grads.append(grad.detach().clone())
    return grads


def _assign_existing_grads(params: Iterable[torch.nn.Parameter], grads: Iterable[torch.Tensor]) -> None:
    for param, grad in zip(params, grads):
        param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _parameter_bytes(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(_tensor_bytes(param.detach()) for param in params)


def _state_bytes(optimizer: torch.optim.Optimizer) -> int:
    seen: set[int] = set()
    total = 0
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if isinstance(value, torch.Tensor) and id(value) not in seen:
                seen.add(id(value))
                total += _tensor_bytes(value)
    return total


def _profile_for(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    getter = getattr(optimizer, "get_profile", None)
    if callable(getter):
        try:
            payload = getter()
            if isinstance(payload, dict):
                return dict(payload)
        except Exception as exc:  # pragma: no cover - best-effort probe field
            return {"profile_error": f"{type(exc).__name__}: {exc}"}
    return {}


def _time_optimizer(
    factory: OptimizerFactory,
    shapes: ShapeList,
    *,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    seed: int,
) -> tuple[float, float, float, dict[str, Any], int, int]:
    params = _make_params(shapes, dtype=dtype, device=device, seed=seed)
    optimizer = factory(params)
    _assign_grads(params, seed=seed + 1000)
    optimizer.step()
    for _ in range(max(int(warmup), 0)):
        optimizer.step()

    _sync(device)
    started = time.perf_counter()
    for _ in range(max(int(iters), 1)):
        optimizer.step()
    _sync(device)

    param_bytes = _parameter_bytes(params)
    state_bytes = _state_bytes(optimizer)
    step_ms = (time.perf_counter() - started) * 1000.0 / max(int(iters), 1)
    return step_ms, param_bytes / 1024 / 1024, state_bytes / 1024 / 1024, _profile_for(optimizer), len(params), sum(p.numel() for p in params)


def _parity_against_adamw(
    factory: OptimizerFactory,
    shapes: ShapeList,
    *,
    dtype: torch.dtype,
    device: torch.device,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[float, float]:
    ref_params = _make_params(shapes, dtype=dtype, device=device, seed=seed)
    cand_params = [torch.nn.Parameter(value.detach().clone()) for value in _clone_params(ref_params)]
    grads = _assign_grads(ref_params, seed=seed + 5000)
    _assign_existing_grads(cand_params, grads)

    ref = torch.optim.AdamW(ref_params, lr=lr, weight_decay=weight_decay)
    cand = factory(cand_params)
    ref.step()
    cand.step()
    _sync(device)

    max_abs = 0.0
    max_ref = 0.0
    for ref_param, cand_param in zip(ref_params, cand_params):
        diff = (ref_param.detach().float() - cand_param.detach().float()).abs()
        max_abs = max(max_abs, float(diff.max().cpu().item()))
        max_ref = max(max_ref, float(ref_param.detach().float().abs().max().cpu().item()))
    return max_abs, max_abs / max(max_ref, 1e-12)


def _optimizer_factories(
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    factored_min_dim: int,
    factored_min_numel: int,
) -> list[tuple[str, bool, OptimizerFactory | None, str]]:
    factories: list[tuple[str, bool, OptimizerFactory | None, str]] = [
        (
            "torch_adamw",
            True,
            lambda params: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay),
            "",
        ),
        (
            "lulynx_python_fused_adamw",
            True,
            lambda params: FusedAdamW(params, lr=lr, weight_decay=weight_decay),
            "",
        ),
        (
            "anima_factored_adamw",
            False,
            lambda params: AnimaFactoredAdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                min_dim=factored_min_dim,
                min_numel=factored_min_numel,
            ),
            "",
        ),
    ]
    if device.type == "cuda":
        factories.insert(
            1,
            (
                "torch_adamw_fused",
                True,
                lambda params: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True),
                "",
            ),
        )
    else:
        factories.insert(1, ("torch_adamw_fused", True, None, "torch fused AdamW is only probed on CUDA"))
    return factories


def run_full_param_optimizer_probe(
    *,
    preset: str = "tiny",
    device: str = "auto",
    dtype: str = "float32",
    iters: int = 5,
    warmup: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 1234,
    optimizers: Iterable[str] | None = None,
    factored_min_dim: int = 128,
    factored_min_numel: int = 65536,
    performance_min_speedup_ratio: float = 1.10,
    performance_promotion_speedup_ratio: float = 1.20,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; available={sorted(PRESETS)}")
    torch_device = _device(device)
    torch_dtype = _dtype(dtype, torch_device)
    shape_cfg = PRESETS[preset]
    shapes = _dit_like_shapes(**shape_cfg)
    selected = {str(item).strip().lower() for item in optimizers or [] if str(item).strip()}
    results: list[OptimizerProbeResult] = []
    stateful_gate = check_stateful_native_optimizer_parity(
        layers=2,
        in_features=min(int(shape_cfg["hidden"]), 64),
        out_features=min(int(shape_cfg["hidden"]), 64),
        rank=4,
        dtype=torch.float32 if torch_device.type == "cpu" else torch_dtype,
        device=torch_device,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        steps=3,
    ).as_dict()
    for name, exact, factory, skip_reason in _optimizer_factories(
        lr=lr,
        weight_decay=weight_decay,
        device=torch_device,
        factored_min_dim=factored_min_dim,
        factored_min_numel=factored_min_numel,
    ):
        if selected and name.lower() not in selected:
            continue
        if factory is None:
            results.append(OptimizerProbeResult(optimizer=name, success=False, failed_reason=skip_reason, exact_adamw_candidate=exact))
            continue
        try:
            step_ms, param_mb, state_mb, profile, tensor_count, param_count = _time_optimizer(
                factory,
                shapes,
                dtype=torch_dtype,
                device=torch_device,
                iters=max(int(iters), 1),
                warmup=max(int(warmup), 0),
                seed=int(seed),
            )
            parity_abs, parity_rel = _parity_against_adamw(
                factory,
                shapes,
                dtype=torch_dtype,
                device=torch_device,
                lr=lr,
                weight_decay=weight_decay,
                seed=int(seed) + 41,
            )
            results.append(
                OptimizerProbeResult(
                    optimizer=name,
                    success=True,
                    parameter_tensors=int(tensor_count),
                    parameter_count=int(param_count),
                    parameter_mb=round(float(param_mb), 3),
                    state_mb=round(float(state_mb), 3),
                    state_to_parameter_ratio=round(float(state_mb / max(param_mb, 1e-12)), 4),
                    step_ms=round(float(step_ms), 4),
                    parity_max_abs_diff=float(parity_abs),
                    parity_max_rel_diff=float(parity_rel),
                    exact_adamw_candidate=exact,
                    native_kernel_present=False,
                    profile=profile,
                )
            )
        except Exception as exc:
            results.append(
                OptimizerProbeResult(
                    optimizer=name,
                    success=False,
                    failed_reason=f"{type(exc).__name__}: {exc}",
                    exact_adamw_candidate=exact,
                    native_kernel_present=False,
                )
            )

    successful = [item for item in results if item.success]
    best_step = min(successful, key=lambda item: item.step_ms).optimizer if successful else ""
    lowest_state = min(successful, key=lambda item: item.state_mb).optimizer if successful else ""
    payload = {
        "schema_version": 1,
        "probe": "anima_full_param_optimizer_probe",
        "preset": preset,
        "shape_config": dict(shape_cfg),
        "device": str(torch_device),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "iters": int(max(int(iters), 1)),
        "warmup": int(max(int(warmup), 0)),
        "native_kernel_present": False,
        "summary": {
            "best_step_optimizer": best_step,
            "lowest_state_optimizer": lowest_state,
            "native_full_param_optimizer_status": "research_probe_only",
            "next_gate": "stateful native ABI plus repeated-step parity and performance win before runtime dispatch",
            "stateful_abi_gate_ok": bool(stateful_gate.get("ok", False)),
        },
        "stateful_abi_gate": stateful_gate,
        "results": [asdict(item) for item in results],
        "notes": [
            "step_ms measures optimizer.step() after state initialization with fixed synthetic gradients already resident on the parameter device.",
            "lulynx_python_fused_adamw is a Python/PyTorch dispatch reduction path, not a Rust/CUDA kernel.",
            "anima_factored_adamw intentionally changes second-moment representation, so exact AdamW parity is not expected.",
            "CUDA/native exact AdamW candidates must beat the PyTorch fused AdamW baseline by the configured performance gate.",
            "No optimizer result from this probe is automatically selected by the training runtime.",
        ],
    }
    performance_gate = evaluate_optimizer_performance_gate(
        payload,
        min_speedup_ratio=float(performance_min_speedup_ratio),
        promotion_speedup_ratio=float(performance_promotion_speedup_ratio),
    )
    payload["performance_gate"] = performance_gate
    payload["summary"].update(
        {
            "performance_gate_ok": bool(performance_gate.get("ok", False)),
            "performance_gate_status": performance_gate.get("status", "unknown"),
            "performance_baseline_optimizer": performance_gate.get("baseline_optimizer"),
            "best_native_performance_candidate": (performance_gate.get("best_candidate") or {}).get("optimizer"),
            "required_native_speedup_vs_baseline": performance_gate.get("required_speedup_vs_baseline"),
            "promotion_native_speedup_vs_baseline": performance_gate.get("promotion_speedup_vs_baseline"),
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimizers", default="")
    parser.add_argument("--factored-min-dim", type=int, default=128)
    parser.add_argument("--factored-min-numel", type=int, default=65536)
    parser.add_argument("--performance-min-speedup-ratio", type=float, default=1.10)
    parser.add_argument("--performance-promotion-speedup-ratio", type=float, default=1.20)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = run_full_param_optimizer_probe(
        preset=args.preset,
        device=args.device,
        dtype=args.dtype,
        iters=args.iters,
        warmup=args.warmup,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        optimizers=[part for part in str(args.optimizers).split(",") if part.strip()],
        factored_min_dim=args.factored_min_dim,
        factored_min_numel=args.factored_min_numel,
        performance_min_speedup_ratio=args.performance_min_speedup_ratio,
        performance_promotion_speedup_ratio=args.performance_promotion_speedup_ratio,
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
