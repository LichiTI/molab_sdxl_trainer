"""Default-off live tensor-binding canary for built-in Muon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_muon_native_scratch_kernel_scorecard import (
    build_muon_native_scratch_kernel_scorecard,
)


ENTRYPOINT = "probe_muon_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 2.0e-5


def build_muon_training_tensor_binding_canary_scorecard(
    *,
    native_scratch_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Run a Muon live tensor-binding canary without enabling dispatch."""

    scratch = _as_dict(native_scratch_report or build_muon_native_scratch_kernel_scorecard())
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch)
    ready = _ready(scratch, live_probe)
    blockers = _blockers(scratch, live_probe)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_training_tensor_binding_canary_scorecard_v0",
        "gate": "muon_model_shape_aware_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "training_tensor_binding_probe_ready": live_probe.get("status") == "passed",
        "native_scratch_kernel_ready": scratch.get("native_scratch_kernel_ready") is True,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "entrypoint": ENTRYPOINT,
        "optimizer_type": "Muon",
        "native_kernel_name": "muon_flat_fp32_cuda_v0",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "live_probe": live_probe,
        "native_scratch_summary": _as_dict(scratch.get("summary")),
        "summary": {
            "optimizer_count": 1,
            "training_tensor_binding_canary_ready_count": 1 if ready else 0,
            "training_tensor_binding_parity_ready_count": 1
            if live_probe.get("training_tensor_binding_parity_passed") is True
            else 0,
            "kernel_executed_count": 1 if live_probe.get("kernel_executed_case_count") == 1 else 0,
            "training_parameters_mutated_count": 1
            if live_probe.get("training_parameters_mutated_count") == 1
            else 0,
            "native_scratch_kernel_ready_count": 1
            if scratch.get("native_scratch_kernel_ready") is True
            else 0,
            "runtime_canary_ready_count": 0,
            "runtime_canary_hit_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "muon_training_loop_canary_missing",
                "muon_runtime_dispatch_shadow_missing",
                "muon_owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Muon TrainingLoop canary with dispatch still default-off"
            if ready
            else "fix Muon live tensor-binding canary blockers"
        ),
        "notes": [
            "This canary binds isolated toy CUDA tensors to the Muon native kernel.",
            "It mutates only those toy tensors, not product training parameters.",
            "It does not enable runtime/native/product dispatch.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _failed("cuda_unavailable", ["muon_training_tensor_binding_cuda_unavailable"])
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed("lulynx_native_entrypoint_missing", ["muon_training_tensor_binding_entrypoint_missing"])
    case = _run_case(native, workspace_root=workspace_root, arch=arch)
    passed = case.get("ok") is True
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "probe_kind": "muon_training_tensor_binding_canary_v0",
        "case_count": 1,
        "passed_case_count": 1 if passed else 0,
        "kernel_executed_case_count": 1 if case.get("kernel_executed") is True else 0,
        "training_parameters_mutated_count": 1 if case.get("training_parameters_mutated") is True else 0,
        "training_tensor_binding_parity_passed": passed,
        "cases": [case],
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "max_param_diff": float(case.get("param_max_abs_diff", 0.0) or 0.0),
        "max_momentum_diff": float(case.get("momentum_buffer_max_abs_diff", 0.0) or 0.0),
        "blocked_reasons": _strings(case.get("blocked_reasons")),
    }


def _run_case(native: Any, *, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    rows = 4
    cols = 4
    numel = rows * cols
    param = torch.tensor(_param_values(numel), device="cuda", dtype=torch.float32).contiguous()
    grad = torch.tensor(_grad_values(numel), device="cuda", dtype=torch.float32).contiguous()
    momentum_buffer = torch.tensor(_momentum_values(numel), device="cuda", dtype=torch.float32).contiguous()
    ref_param = param.detach().clone()
    ref_momentum = momentum_buffer.detach().clone()
    lr = 1.0e-2
    momentum = 0.95
    ns_steps = 5
    nesterov = True
    _reference_muon(ref_param, grad, ref_momentum, rows, cols, lr, momentum, ns_steps, nesterov)
    try:
        launch = dict(
            getattr(native, ENTRYPOINT)(
                param,
                grad,
                momentum_buffer,
                rows,
                cols,
                lr,
                momentum,
                ns_steps,
                nesterov,
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or _cuda_arch(param.device),
            )
        )
    except Exception as exc:  # pragma: no cover - native/CUDA dependent
        return _case_failed(f"muon_training_tensor_binding_call_failed:{type(exc).__name__}: {exc}")
    param_diff = _max_abs_diff(ref_param, param)
    momentum_diff = _max_abs_diff(ref_momentum, momentum_buffer)
    finite = all(torch.isfinite(t).all().item() for t in (param, momentum_buffer))
    ok = bool(
        launch.get("ok") is True
        and launch.get("kernel_executed") is True
        and launch.get("native_live_tensor_binding") is True
        and launch.get("training_dispatch") is False
        and launch.get("training_path_enabled") is False
        and finite
        and param_diff <= TOLERANCE
        and momentum_diff <= TOLERANCE
    )
    launch.update(
        {
            "ok": ok,
            "case": "muon_4x4_live_tensor_binding",
            "param_max_abs_diff": param_diff,
            "momentum_buffer_max_abs_diff": momentum_diff,
            "max_abs_diff": max(param_diff, momentum_diff),
            "tolerance": TOLERANCE,
            "losses_finite": finite,
            "blocked_reasons": [] if ok else _case_blockers(launch, param_diff, momentum_diff, finite),
        }
    )
    return launch


def _reference_muon(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    rows: int,
    cols: int,
    lr: float,
    momentum: float,
    ns_steps: int,
    nesterov: bool,
) -> None:
    next_buffer = momentum_buffer * momentum + grad * (1.0 - momentum)
    momentum_buffer.copy_(next_buffer)
    update = grad * (1.0 - momentum) + next_buffer * momentum if nesterov else next_buffer
    ortho = _zero_power_newton_schulz(update.reshape(rows, cols), ns_steps).reshape(-1)
    param.add_(ortho, alpha=-lr)


def _zero_power_newton_schulz(source: torch.Tensor, steps: int) -> torch.Tensor:
    x = source / torch.sqrt(torch.sum(source * source).clamp_min(1.0e-14))
    w0 = 3.4445
    w1 = -4.7750
    w2 = 2.0315
    for _ in range(steps):
        a = x @ x.t()
        aa = a @ a
        b = w1 * a + w2 * aa
        x = w0 * x + b @ x
    return x


def _ready(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> bool:
    return bool(
        scratch.get("native_scratch_kernel_ready") is True
        and live_probe.get("status") == "passed"
        and live_probe.get("training_tensor_binding_parity_passed") is True
        and live_probe.get("training_dispatch") is False
        and live_probe.get("training_path_enabled") is False
        and live_probe.get("native_dispatch_allowed") is False
    )


def _blockers(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if scratch.get("native_scratch_kernel_ready") is not True:
        blockers.append("muon_native_scratch_kernel_missing")
    if live_probe.get("status") != "passed":
        blockers.extend(_strings(live_probe.get("blocked_reasons")) or ["muon_training_tensor_binding_canary_failed"])
    if live_probe.get("training_dispatch") is True:
        blockers.append("muon_training_tensor_binding_training_dispatch_enabled")
    if live_probe.get("training_path_enabled") is True:
        blockers.append("muon_training_tensor_binding_training_path_enabled")
    if live_probe.get("native_dispatch_allowed") is True:
        blockers.append("muon_training_tensor_binding_native_dispatch_allowed")
    return _dedupe(blockers)


def _case_blockers(
    launch: Mapping[str, Any],
    param_diff: float,
    momentum_diff: float,
    finite: bool,
) -> list[str]:
    blockers = _strings(launch.get("blocked_reasons"))
    if not blockers and launch.get("reason"):
        blockers.append(str(launch.get("reason")))
    if launch.get("kernel_executed") is not True:
        blockers.append("muon_training_tensor_binding_kernel_not_executed")
    if launch.get("native_live_tensor_binding") is not True:
        blockers.append("muon_training_tensor_binding_missing")
    if launch.get("training_dispatch") is True:
        blockers.append("muon_training_tensor_binding_training_dispatch_enabled")
    if launch.get("training_path_enabled") is True:
        blockers.append("muon_training_tensor_binding_training_path_enabled")
    if param_diff > TOLERANCE:
        blockers.append("muon_training_tensor_binding_param_parity_failed")
    if momentum_diff > TOLERANCE:
        blockers.append("muon_training_tensor_binding_momentum_parity_failed")
    if not finite:
        blockers.append("muon_training_tensor_binding_non_finite_tensor")
    return _dedupe(blockers)


def _case_failed(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "case": "muon_4x4_live_tensor_binding",
        "reason": reason,
        "kernel_executed": False,
        "native_live_tensor_binding": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [reason],
    }


def _failed(reason: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "kernel_executed_case_count": 0,
        "training_parameters_mutated_count": 0,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
    }


def _param_values(numel: int) -> list[float]:
    return [((1.0 if index % 2 == 0 else -1.0) * (0.12 + index * 0.011)) for index in range(numel)]


def _grad_values(numel: int) -> list[float]:
    return [((-1.0 if index % 3 == 0 else 1.0) * (0.02 + index * 0.0013)) for index in range(numel)]


def _momentum_values(numel: int) -> list[float]:
    return [((-1.0 if index % 5 == 0 else 1.0) * (0.004 + index * 0.0007)) for index in range(numel)]


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu().item())


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_training_tensor_binding_canary_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "build_muon_training_tensor_binding_canary_scorecard"]
