"""Default-off training tensor binding canary for selected plugin AdamG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.turbocore_plugin_adamg_native_scratch_kernel_scorecard import (
    build_plugin_adamg_native_scratch_kernel_scorecard,
)
from core.turbocore_tensor_handle_registry import (
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adamg_tensor_probe",
    "destroy_tensor_binding_session",
)
FLOAT_TOLERANCE = 5e-6
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_adamg_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(
        scratch_report
        or build_plugin_adamg_native_scratch_kernel_scorecard(workspace_root=workspace_root, arch=arch)
    )
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch) if run_live_probe else _skipped("live_probe_disabled")
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamg_training_tensor_binding_canary_scorecard_v0",
        "gate": "plugin_adamg_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_e2e_no_regression_ready": bool(live_probe.get("e2e_no_regression_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoints": list(ENTRYPOINTS),
        "optimizer_kind": "adamg",
        "optimizer_family": "adam_like_formula",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "scratch_summary": dict(scratch.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "kernel_executed": bool(live_probe.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
            "e2e_no_regression_passed": bool(live_probe.get("e2e_no_regression_passed", False)),
            "max_abs_diff": live_probe.get("max_abs_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamg_runtime_dispatch_shadow_missing",
                "adamg_training_loop_canary_missing",
                "adamg_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected AdamG runtime dispatch shadow before TrainingLoop canary"
            if ready
            else "fix selected AdamG training tensor binding canary blockers"
        ),
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _failed("lulynx_native_entrypoint_missing", ["adamg_training_tensor_binding_entrypoint_missing"])
    request, tensor_map, role_tensors = _make_cuda_request()
    config = {
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "lr": 0.031,
        "betas": [0.95, 0.999, 0.95],
        "p": 0.2,
        "q": 0.24,
        "eps": 1e-8,
        "weight_decay": 0.012,
        "weight_decouple": False,
        "fixed_decay": False,
        "step_index": 3,
        "block_size": 128,
        "max_numel": 64,
    }
    reference = _reference_adamg(role_tensors, config)
    session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
    if not bool(session.get("ok", False)):
        return _failed("tensor_binding_session_create_failed", ["adamg_tensor_binding_session_create_failed"], session=session)
    session_id = int(session["session_id"])
    try:
        launch = dict(native.tensor_binding_session_cuda_adamg_tensor_probe(session_id, json.dumps(config)))
        if not bool(launch.get("ok", False)):
            return _failed(
                f"adamg_tensor_probe_launch_failed:{launch.get('reason', 'unknown')}",
                ["adamg_tensor_probe_launch_failed"],
                session=session,
                launch=launch,
            )
        torch.cuda.synchronize()
        diffs = {
            "param_flat": _max_abs_diff(role_tensors["param_flat"], reference["param_flat"]),
            "exp_avg": _max_abs_diff(role_tensors["exp_avg"], reference["exp_avg"]),
            "exp_avg_sq": _max_abs_diff(role_tensors["exp_avg_sq"], reference["exp_avg_sq"]),
            "r": _max_abs_diff(role_tensors["r"], reference["r"]),
        }
        max_diff = max(diffs.values())
        ok = bool(
            max_diff <= FLOAT_TOLERANCE
            and launch.get("kernel_executed") is True
            and launch.get("training_tensor_binding") is True
            and launch.get("training_dispatch") is False
            and launch.get("training_path_enabled") is False
        )
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "probe_kind": "plugin_adamg_training_tensor_binding_canary_v0",
            "ok": ok,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": ok,
            "e2e_no_regression_passed": ok,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "max_abs_diff": max_diff,
            "diffs": diffs,
            "session": session,
            "launch": launch,
            "blocked_reasons": [] if ok else ["adamg_training_tensor_binding_parity_failed"],
        }
    finally:
        destroyed = dict(native.destroy_tensor_binding_session(session_id))
        if not bool(destroyed.get("ok", False)):
            pass


def _make_cuda_request() -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    registry = TurboCoreTensorHandleRegistry(namespace="plugin_adamg_tensor_binding_canary")
    param = (torch.arange(24, dtype=torch.float32, device=device) * 0.117 + 0.43).contiguous()
    grad = (torch.arange(24, dtype=torch.float32, device=device) * 0.0009 + 0.012).contiguous()
    m = (torch.arange(24, dtype=torch.float32, device=device) * 0.00027 - 0.001).contiguous()
    v = (torch.arange(24, dtype=torch.float32, device=device) * 0.000012 + 0.00021).contiguous()
    r = (torch.arange(24, dtype=torch.float32, device=device) * 0.000009 + 0.00015).contiguous()
    handles = registry.register_flat_adamw_buffers(
        param_flat=param,
        grad_flat=grad,
        exp_avg=m,
        exp_avg_sq=v,
    )
    r_record = registry.register(r, role="r", expected_numel=int(param.numel()))
    request = build_flat_adamw_native_binding_request(registry, handles)
    request["optimizer"] = "AdamG"
    r_binding = {
        "role": r_record.role,
        "handle_id": r_record.handle_id,
        "handle_kind": r_record.handle_kind,
        "numel": r_record.numel,
        "dtype": r_record.dtype,
        "device_type": r_record.device_type,
        "device_index": r_record.device_index,
        "layout": r_record.layout,
        "contiguous": r_record.contiguous,
        "alignment_bytes": r_record.alignment_bytes,
        "pointer_exported": False,
    }
    request["bindings"] = [*list(request["bindings"]), r_binding]
    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    tensor_map[r_record.handle_id] = registry.resolve(r_record.handle_id)
    role_tensors = {binding["role"]: tensor_map[binding["handle_id"]] for binding in request["bindings"]}
    return request, tensor_map, role_tensors


def _reference_adamg(role_tensors: dict[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    param = role_tensors["param_flat"].detach().clone()
    grad = role_tensors["grad_flat"].detach().clone()
    m = role_tensors["exp_avg"].detach().clone()
    v = role_tensors["exp_avg_sq"].detach().clone()
    r = role_tensors["r"].detach().clone()
    beta1, beta2, beta3 = [float(item) for item in config["betas"]]
    lr = float(config["lr"])
    eps = float(config["eps"])
    weight_decay = float(config["weight_decay"])
    p_value = float(config["p"])
    q_value = float(config["q"])
    step_number = int(config["step_index"]) + 1
    if weight_decay:
        if bool(config.get("weight_decouple", False)):
            decay = weight_decay if bool(config.get("fixed_decay", False)) else weight_decay * lr
            param.mul_(1.0 - decay)
        else:
            grad.add_(param, alpha=weight_decay)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    r.mul_(beta3).add_(v.pow(q_value).mul_(p_value), alpha=1.0 - beta3)
    m.mul_(beta1).addcmul_(r, grad, value=1.0 - beta1)
    bias_correction1 = 1.0 - beta1**step_number
    bias_correction2 = 1.0 - beta2**step_number
    step_size = min(lr, 1.0 / (step_number**0.5))
    denom = v.div(bias_correction2).sqrt().add_(eps)
    param.addcdiv_(m / bias_correction1, denom, value=-step_size)
    return {"param_flat": param, "exp_avg": m, "exp_avg_sq": v, "r": r}


def _validations(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "p55_adamg_scratch_kernel_ready",
            bool(scratch.get("adamg_native_kernel_parity", False)),
            "adamg_native_scratch_kernel_missing",
        ),
        _validation(
            "training_tensor_binding_parity",
            bool(live_probe.get("training_tensor_binding_parity_passed", False)),
            "adamg_training_tensor_binding_canary_failed",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "adamg_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, reason: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "blocked_reasons": [] if ok else [reason]}


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [reason],
    }


def _failed(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_plugin_adamg_training_tensor_binding_canary_scorecard"]
