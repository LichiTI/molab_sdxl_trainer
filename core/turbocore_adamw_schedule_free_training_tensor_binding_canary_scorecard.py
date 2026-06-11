"""Native live-tensor binding canary for AdamWScheduleFree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints
from core.turbocore_adamw_schedule_free_training_executor import ENTRYPOINT


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adamw_schedule_free_training_tensor_binding_canary_scorecard() -> dict[str, Any]:
    """Run a default-off live tensor binding probe against the native kernel."""

    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_adamw_schedule_free_training_tensor_binding_canary")
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _blocked("adamw_schedule_free_live_tensor_entrypoint_missing")
    case = _run_case(native)
    ready = bool(case.get("ok", False))
    blockers = _strings(case.get("blocked_reasons"))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_training_tensor_binding_canary_scorecard_v0",
        "gate": "adamw_schedule_free_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_ready": ready,
        "native_live_entrypoint_ready": True,
        "runtime_dispatch_ready": False,
        "training_loop_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoint": ENTRYPOINT,
        "optimizer_kind": "adamw_schedule_free",
        "case": case,
        "summary": {
            "case_count": 1,
            "passed_case_count": 1 if ready else 0,
            "kernel_executed": case.get("kernel_executed") is True,
            "native_live_tensor_binding": case.get("native_live_tensor_binding") is True,
            "training_parameters_mutated": case.get("training_parameters_mutated") is True,
            "training_dispatch": case.get("training_dispatch") is True,
            "training_path_enabled": case.get("training_path_enabled") is True,
            "max_abs_diff": float(case.get("max_abs_diff", 0.0) or 0.0),
            "k_after": int(case.get("k_after", 0) or 0),
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_schedule_free_training_loop_native_dispatch_missing",
                "product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire AdamWScheduleFree TrainingLoop dispatch canary with native tensor binding proven"
            if ready
            else "fix AdamWScheduleFree native live tensor binding canary blockers"
        ),
        "notes": [
            "This probe mutates only toy CUDA tensors and keeps training dispatch disabled.",
            "It proves live param/grad/z/exp_avg_sq binding against the native kernel.",
            "It does not enable TrainingLoop or product native dispatch.",
        ],
    }


def _run_case(native: Any) -> dict[str, Any]:
    param = torch.linspace(-0.25, 0.35, steps=64, device="cuda", dtype=torch.float32)
    grad = torch.linspace(0.01, 0.08, steps=64, device="cuda", dtype=torch.float32)
    z = param.detach().clone()
    exp_avg_sq = torch.zeros_like(param)
    config = {
        "lr": 1.0e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "warmup_steps": 4,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "k": 0,
        "weight_sum": 0.0,
        "lr_max": 0.0,
        "scheduled_lr": 0.0,
        "max_numel": int(param.numel()),
        "canary_probe_only": True,
        "training_tensor_binding": True,
        "training_dispatch": False,
        "training_path_enabled": False,
    }
    try:
        launch = dict(
            getattr(native, ENTRYPOINT)(
                param,
                grad,
                z,
                exp_avg_sq,
                json.dumps(config),
                str(REPO_ROOT.resolve()),
                _cuda_arch(param.device),
            )
        )
    except Exception as exc:  # pragma: no cover - native/CUDA dependent
        return {
            "schema_version": 1,
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "kernel_executed": False,
            "native_live_tensor_binding": False,
            "training_dispatch": False,
            "training_path_enabled": False,
            "blocked_reasons": ["adamw_schedule_free_live_tensor_binding_call_failed"],
        }
    ok = bool(
        launch.get("ok", False)
        and launch.get("kernel_executed") is True
        and launch.get("native_live_tensor_binding") is True
        and launch.get("training_dispatch") is False
        and launch.get("training_path_enabled") is False
        and int(launch.get("k_after", 0) or 0) == 1
    )
    launch["ok"] = ok
    launch["blocked_reasons"] = [] if ok else _case_blockers(launch)
    return launch


def _case_blockers(launch: Mapping[str, Any]) -> list[str]:
    blockers = _strings(launch.get("blocked_reasons"))
    if not blockers and launch.get("reason"):
        blockers.append(str(launch.get("reason")))
    if launch.get("kernel_executed") is not True:
        blockers.append("adamw_schedule_free_live_tensor_kernel_not_executed")
    if launch.get("native_live_tensor_binding") is not True:
        blockers.append("adamw_schedule_free_live_tensor_binding_missing")
    if int(launch.get("k_after", 0) or 0) != 1:
        blockers.append("adamw_schedule_free_live_tensor_step_not_recorded")
    return _dedupe(blockers)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_training_tensor_binding_canary_scorecard_v0",
        "gate": "adamw_schedule_free_training_tensor_binding_canary",
        "ok": False,
        "promotion_ready": False,
        "training_tensor_binding_ready": False,
        "native_live_entrypoint_ready": False,
        "runtime_dispatch_ready": False,
        "training_loop_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoint": ENTRYPOINT,
        "optimizer_kind": "adamw_schedule_free",
        "case": {},
        "summary": {"case_count": 0, "passed_case_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "build AdamWScheduleFree native live tensor entrypoint before TrainingLoop canary",
    }


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


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


__all__ = ["build_adamw_schedule_free_training_tensor_binding_canary_scorecard"]
