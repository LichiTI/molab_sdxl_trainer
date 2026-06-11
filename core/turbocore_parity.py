"""Parity anchors for future TurboCore native implementations.

The functions in this module compare today's PyTorch reference behavior against
candidate callables.  Until native kernels exist, the default candidates are
PyTorch-equivalent implementations; future Rust/CUDA bridge calls can be passed
in without changing the test contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from core.turbocore_candidates import get_turbocore_candidate
from core.turbocore_optimizer_abi import (
    AdamWStatefulOptimizerConfig,
    PyTorchStatefulAdamWBackend,
)


@dataclass(frozen=True)
class TurboCoreParityResult:
    name: str
    ok: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    tolerance_abs: float = 1e-5
    tolerance_rel: float = 1e-4
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("details") is None:
            payload["details"] = {}
        return payload


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_f.abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return max_abs, max_rel


def _allclose(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> tuple[bool, float, float]:
    max_abs, max_rel = _max_errors(actual, expected)
    return bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)), max_abs, max_rel


def check_lora_delta_parity(
    *,
    batch: int = 2,
    tokens: int = 64,
    in_features: int = 320,
    out_features: int = 320,
    rank: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    scale: float | None = None,
    candidate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None,
    candidate_name: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> TurboCoreParityResult:
    """Check LoRA delta/add parity for one representative shape."""

    device = torch.device(device)
    torch.manual_seed(113)
    rank = max(int(rank), 1)
    scale_value = float(scale if scale is not None else 1.0 / rank)
    x = torch.randn(batch, tokens, in_features, dtype=dtype, device=device)
    down = torch.randn(rank, in_features, dtype=dtype, device=device)
    up = torch.randn(out_features, rank, dtype=dtype, device=device)
    base = torch.randn(batch, tokens, out_features, dtype=dtype, device=device)

    expected = base + F.linear(F.linear(x, down), up) * scale_value
    candidate_label = candidate_name or "pytorch_explicit"
    registered = None
    if candidate is None:
        registered = get_turbocore_candidate("lora_fused", candidate_name)
        if registered is not None:
            candidate = registered.callable  # type: ignore[assignment]
            candidate_label = registered.name
    if candidate is None:
        actual = base + F.linear(F.linear(x, down), up) * scale_value
    else:
        actual = candidate(x, down, up, base, scale_value)
        if candidate_name is None and registered is None:
            candidate_label = getattr(candidate, "__name__", candidate_label)
    ok, max_abs, max_rel = _allclose(actual, expected, atol=atol, rtol=rtol)
    return TurboCoreParityResult(
        name="lora_fused_delta",
        ok=ok,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        tolerance_abs=atol,
        tolerance_rel=rtol,
        details={
            "candidate": candidate_label,
            "shape": [int(batch), int(tokens), int(in_features), int(out_features)],
            "rank": rank,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
        },
    )


def _make_param_pair(shapes: Iterable[tuple[int, ...]], *, dtype: torch.dtype, device: torch.device) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    refs: list[torch.nn.Parameter] = []
    cands: list[torch.nn.Parameter] = []
    for shape in shapes:
        value = torch.randn(*shape, dtype=dtype, device=device) * 0.01
        grad = torch.randn_like(value)
        ref = torch.nn.Parameter(value.clone())
        cand = torch.nn.Parameter(value.clone())
        ref.grad = grad.clone()
        cand.grad = grad.clone()
        refs.append(ref)
        cands.append(cand)
    return refs, cands


def _make_param_sets(
    shapes: Iterable[tuple[int, ...]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    copies: int,
) -> list[list[torch.nn.Parameter]]:
    torch.manual_seed(int(seed))
    values = [torch.randn(*shape, dtype=dtype, device=device) * 0.01 for shape in shapes]
    return [[torch.nn.Parameter(value.detach().clone()) for value in values] for _ in range(max(int(copies), 1))]


def _assign_seeded_grads(params: Iterable[torch.nn.Parameter], *, seed: int, scale: float = 1.0) -> None:
    torch.manual_seed(int(seed))
    for param in params:
        param.grad = torch.randn_like(param) * float(scale)


def _max_param_errors(actual_params: Iterable[torch.nn.Parameter], expected_params: Iterable[torch.nn.Parameter]) -> tuple[bool, float, float]:
    ok = True
    max_abs = 0.0
    max_rel = 0.0
    for actual, expected in zip(actual_params, expected_params):
        item_ok, item_abs, item_rel = _allclose(actual.detach(), expected.detach(), atol=1e-6, rtol=1e-5)
        ok = ok and item_ok
        max_abs = max(max_abs, item_abs)
        max_rel = max(max_rel, item_rel)
    return ok, max_abs, max_rel


def _max_param_errors_tol(
    actual_params: Iterable[torch.nn.Parameter],
    expected_params: Iterable[torch.nn.Parameter],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    ok = True
    max_abs = 0.0
    max_rel = 0.0
    for actual, expected in zip(actual_params, expected_params):
        item_ok, item_abs, item_rel = _allclose(actual.detach(), expected.detach(), atol=atol, rtol=rtol)
        ok = ok and item_ok
        max_abs = max(max_abs, item_abs)
        max_rel = max(max_rel, item_rel)
    return ok, max_abs, max_rel


def _adamw_candidate_step(
    params: list[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
) -> None:
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    optimizer.step()


def check_native_optimizer_parity(
    *,
    layers: int = 4,
    in_features: int = 320,
    out_features: int = 320,
    rank: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    candidate: Callable[[list[torch.nn.Parameter], float, float, float], None] | None = None,
    candidate_name: str | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> TurboCoreParityResult:
    """Check one-step LoRA-sized AdamW/update parity."""

    device = torch.device(device)
    torch.manual_seed(211)
    rank = max(int(rank), 1)
    shapes: list[tuple[int, ...]] = []
    for _ in range(max(int(layers), 1)):
        shapes.append((rank, int(in_features)))
        shapes.append((int(out_features), rank))
    ref_params, cand_params = _make_param_pair(shapes, dtype=dtype, device=device)

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(ref_params, max_grad_norm)
    ref_optimizer = torch.optim.AdamW(ref_params, lr=lr, weight_decay=weight_decay)
    ref_optimizer.step()

    candidate_label = candidate_name or "pytorch_adamw"
    registered = None
    if candidate is None:
        registered = get_turbocore_candidate("native_optimizer", candidate_name)
        if registered is not None:
            candidate = registered.callable  # type: ignore[assignment]
            candidate_label = registered.name
    if candidate is None:
        _adamw_candidate_step(cand_params, lr=lr, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
    else:
        candidate(cand_params, lr, weight_decay, max_grad_norm)
        if candidate_name is None and registered is None:
            candidate_label = getattr(candidate, "__name__", candidate_label)

    max_abs = 0.0
    max_rel = 0.0
    ok = True
    for actual, expected in zip(cand_params, ref_params):
        item_ok, item_abs, item_rel = _allclose(actual.detach(), expected.detach(), atol=atol, rtol=rtol)
        ok = ok and item_ok
        max_abs = max(max_abs, item_abs)
        max_rel = max(max_rel, item_rel)

    return TurboCoreParityResult(
        name="native_optimizer_adamw",
        ok=ok,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        tolerance_abs=atol,
        tolerance_rel=rtol,
        details={
            "candidate": candidate_label,
            "layers": int(layers),
            "rank": rank,
            "parameter_tensors": len(ref_params),
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "max_grad_norm": float(max_grad_norm),
        },
    )


def check_stateful_native_optimizer_parity(
    *,
    layers: int = 3,
    in_features: int = 64,
    out_features: int = 64,
    rank: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    steps: int = 3,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> TurboCoreParityResult:
    """Check stateful AdamW lifecycle parity for future native optimizers."""

    device = torch.device(device)
    rank = max(int(rank), 1)
    shapes: list[tuple[int, ...]] = []
    for _ in range(max(int(layers), 1)):
        shapes.append((rank, int(in_features)))
        shapes.append((int(out_features), rank))

    ref_params, cand_params = _make_param_sets(
        shapes,
        dtype=dtype,
        device=device,
        seed=311,
        copies=2,
    )
    config = AdamWStatefulOptimizerConfig(
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        finite_check=True,
        set_to_none=True,
    )
    ref = PyTorchStatefulAdamWBackend(ref_params, config, backend_name="reference_stateful_adamw")
    cand = PyTorchStatefulAdamWBackend(cand_params, config, backend_name="candidate_stateful_adamw")

    reports: list[dict[str, Any]] = []
    resume = None
    resume_params: list[torch.nn.Parameter] | None = None
    restore_step = max(int(steps), 1) // 2
    for step_index in range(max(int(steps), 1)):
        _assign_seeded_grads(ref_params, seed=9000 + step_index, scale=2.0)
        _assign_seeded_grads(cand_params, seed=9000 + step_index, scale=2.0)
        if resume is not None and resume_params is not None:
            _assign_seeded_grads(resume_params, seed=9000 + step_index, scale=2.0)
        ref_report = ref.step()
        cand_report = cand.step()
        resume_report = resume.step() if resume is not None else None
        reports.append(cand_report.as_dict())
        if step_index == restore_step:
            saved_state = cand.state_dict()
            resume_params = [torch.nn.Parameter(param.detach().clone()) for param in cand_params]
            resume = PyTorchStatefulAdamWBackend(resume_params, config, backend_name="restored_stateful_adamw")
            resume.load_state_dict(saved_state)
        ref.zero_grad()
        cand.zero_grad()
        if resume is not None:
            resume.zero_grad()
        if any(param.grad is not None for param in cand_params):
            return TurboCoreParityResult(
                name="native_optimizer_adamw_stateful",
                ok=False,
                details={"reason": "zero_grad_did_not_clear_gradients"},
            )
        if resume_params is not None and any(param.grad is not None for param in resume_params):
            return TurboCoreParityResult(
                name="native_optimizer_adamw_stateful",
                ok=False,
                details={"reason": "resume_zero_grad_did_not_clear_gradients"},
            )
        if ref_report.skipped or cand_report.skipped or (resume_report is not None and resume_report.skipped):
            return TurboCoreParityResult(
                name="native_optimizer_adamw_stateful",
                ok=False,
                details={"reason": "unexpected_skip", "candidate_report": cand_report.as_dict()},
            )

    ok, max_abs, max_rel = _max_param_errors_tol(cand_params, ref_params, atol=atol, rtol=rtol)
    restore_ok = False
    restore_abs = 0.0
    restore_rel = 0.0
    if resume_params is not None:
        restore_ok, restore_abs, restore_rel = _max_param_errors_tol(resume_params, cand_params, atol=atol, rtol=rtol)
        ok = ok and restore_ok
        max_abs = max(max_abs, restore_abs)
        max_rel = max(max_rel, restore_rel)

    before_nonfinite = [param.detach().clone() for param in cand_params]
    _assign_seeded_grads(cand_params, seed=9911, scale=1.0)
    if cand_params and cand_params[0].grad is not None:
        cand_params[0].grad.flatten()[0] = float("nan")
    nonfinite_report = cand.step()
    nonfinite_unchanged = all(
        torch.equal(param.detach(), before)
        for param, before in zip(cand_params, before_nonfinite)
    )
    ok = ok and bool(nonfinite_report.skipped) and nonfinite_unchanged

    return TurboCoreParityResult(
        name="native_optimizer_adamw_stateful",
        ok=ok,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        tolerance_abs=atol,
        tolerance_rel=rtol,
        details={
            "candidate": "pytorch_stateful_adamw_reference",
            "steps": int(max(int(steps), 1)),
            "layers": int(layers),
            "rank": rank,
            "parameter_tensors": len(ref_params),
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "max_grad_norm": float(max_grad_norm),
            "reports": reports,
            "restore_ok": restore_ok,
            "restore_max_abs_error": restore_abs,
            "restore_max_rel_error": restore_rel,
            "nonfinite_skip_ok": bool(nonfinite_report.skipped),
            "nonfinite_params_unchanged": bool(nonfinite_unchanged),
            "snapshot": cand.snapshot(),
        },
    )


def build_turbocore_parity_report(*, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Run the default small parity anchors and return a report."""

    results = [
        check_lora_delta_parity(device=device, dtype=dtype),
        check_native_optimizer_parity(device=device, dtype=dtype),
        check_stateful_native_optimizer_parity(device=device, dtype=dtype),
    ]
    return {
        "schema_version": 1,
        "summary": {
            "ok": all(result.ok for result in results),
            "native_kernel_present": False,
            "purpose": "reference parity anchors for future TurboCore native candidates",
        },
        "results": [result.as_dict() for result in results],
    }


__all__ = [
    "TurboCoreParityResult",
    "build_turbocore_parity_report",
    "check_lora_delta_parity",
    "check_native_optimizer_parity",
    "check_stateful_native_optimizer_parity",
]
