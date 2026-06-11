"""Default-off training tensor binding canary for selected plugin adamod."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.turbocore_plugin_adamod_native_scratch_kernel_scorecard import (
    build_plugin_adamod_native_scratch_kernel_scorecard,
)
from core.turbocore_tensor_handle_registry import (
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adamod_tensor_probe",
    "destroy_tensor_binding_session",
)
FLOAT_TOLERANCE = 5e-6
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_adamod_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(
        scratch_report
        or build_plugin_adamod_native_scratch_kernel_scorecard(workspace_root=workspace_root, arch=arch)
    )
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch) if run_live_probe else _skipped("live_probe_disabled")
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamod_training_tensor_binding_canary_scorecard_v0",
        "gate": "plugin_adamod_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_e2e_no_regression_ready": bool(live_probe.get("e2e_no_regression_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoints": list(ENTRYPOINTS),
        "optimizer_kind": "adamod",
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
                "adamod_runtime_dispatch_shadow_missing",
                "adamod_training_loop_canary_missing",
                "adamod_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected adamod runtime dispatch shadow before TrainingLoop canary"
            if ready
            else "fix selected adamod training tensor binding canary blockers"
        ),
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _failed("lulynx_native_entrypoint_missing", ["adamod_training_tensor_binding_entrypoint_missing"])
    request, tensor_map, role_tensors = _make_cuda_request()
    config = {
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "lr": 0.031,
        "betas": [0.9, 0.99, 0.9999],
        "eps": 1e-8,
        "weight_decay": 0.012,
        "weight_decouple": True,
        "fixed_decay": False,
        "step_index": 3,
        "block_size": 128,
        "max_numel": 64,
    }
    reference = _reference_adamod(role_tensors, config)
    session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
    if not bool(session.get("ok", False)):
        return _failed("tensor_binding_session_create_failed", ["adamod_tensor_binding_session_create_failed"], session=session)
    session_id = int(session["session_id"])
    try:
        launch = dict(native.tensor_binding_session_cuda_adamod_tensor_probe(session_id, json.dumps(config)))
        if not bool(launch.get("ok", False)):
            return _failed(
                f"adamod_tensor_probe_launch_failed:{launch.get('reason', 'unknown')}",
                ["adamod_tensor_probe_launch_failed"],
                session=session,
                launch=launch,
            )
        torch.cuda.synchronize()
        diffs = {
            "param_flat": _max_abs_diff(role_tensors["param_flat"], reference["param_flat"]),
            "exp_avg": _max_abs_diff(role_tensors["exp_avg"], reference["exp_avg"]),
            "exp_avg_sq": _max_abs_diff(role_tensors["exp_avg_sq"], reference["exp_avg_sq"]),
            "exp_avg_lr": _max_abs_diff(role_tensors["exp_avg_lr"], reference["exp_avg_lr"]),
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
            "probe_kind": "plugin_adamod_training_tensor_binding_canary_v0",
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
            "blocked_reasons": [] if ok else ["adamod_training_tensor_binding_parity_failed"],
        }
    finally:
        destroyed = dict(native.destroy_tensor_binding_session(session_id))
        if not bool(destroyed.get("ok", False)):
            pass


def _make_cuda_request() -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    registry = TurboCoreTensorHandleRegistry(namespace="plugin_adamod_tensor_binding_canary")
    param = (torch.arange(24, dtype=torch.float32, device=device) * 0.117 + 0.43).contiguous()
    grad = (torch.arange(24, dtype=torch.float32, device=device) * 0.0009 + 0.012).contiguous()
    m = (torch.arange(24, dtype=torch.float32, device=device) * 0.00027 - 0.001).contiguous()
    v = (torch.arange(24, dtype=torch.float32, device=device) * 0.000012 + 0.00021).contiguous()
    exp_avg_lr = (torch.arange(24, dtype=torch.float32, device=device) * 0.000009 + 0.00015).contiguous()
    handles = registry.register_flat_adamw_buffers(
        param_flat=param,
        grad_flat=grad,
        exp_avg=m,
        exp_avg_sq=v,
    )
    exp_avg_lr_record = registry.register(exp_avg_lr, role="exp_avg_lr", expected_numel=int(param.numel()))
    request = build_flat_adamw_native_binding_request(registry, handles)
    request["optimizer"] = "AdaMod"
    exp_avg_lr_binding = {
        "role": exp_avg_lr_record.role,
        "handle_id": exp_avg_lr_record.handle_id,
        "handle_kind": exp_avg_lr_record.handle_kind,
        "numel": exp_avg_lr_record.numel,
        "dtype": exp_avg_lr_record.dtype,
        "device_type": exp_avg_lr_record.device_type,
        "device_index": exp_avg_lr_record.device_index,
        "layout": exp_avg_lr_record.layout,
        "contiguous": exp_avg_lr_record.contiguous,
        "alignment_bytes": exp_avg_lr_record.alignment_bytes,
        "pointer_exported": False,
    }
    request["bindings"] = [*list(request["bindings"]), exp_avg_lr_binding]
    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    tensor_map[exp_avg_lr_record.handle_id] = registry.resolve(exp_avg_lr_record.handle_id)
    role_tensors = {binding["role"]: tensor_map[binding["handle_id"]] for binding in request["bindings"]}
    return request, tensor_map, role_tensors


def _reference_adamod(role_tensors: dict[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    param = role_tensors["param_flat"].detach().clone()
    grad = role_tensors["grad_flat"].detach().clone()
    m = role_tensors["exp_avg"].detach().clone()
    v = role_tensors["exp_avg_sq"].detach().clone()
    exp_avg_lr = role_tensors["exp_avg_lr"].detach().clone()
    beta1, beta2, beta3 = [float(item) for item in config["betas"]]
    lr = float(config["lr"])
    eps = float(config["eps"])
    weight_decay = float(config["weight_decay"])
    step_number = int(config["step_index"]) + 1
    if weight_decay:
        if bool(config.get("weight_decouple", False)):
            decay = weight_decay if bool(config.get("fixed_decay", False)) else weight_decay * lr
            param.mul_(1.0 - decay)
        else:
            grad.add_(param, alpha=weight_decay)
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step_number
    bias_correction2_sqrt = (1.0 - beta2**step_number) ** 0.5
    step_size = lr * bias_correction2_sqrt / bias_correction1
    update_lr = torch.full_like(v, fill_value=step_size).div_(v.sqrt().add_(eps))
    exp_avg_lr.mul_(beta3).add_(update_lr, alpha=1.0 - beta3)
    param.add_(torch.minimum(update_lr, exp_avg_lr).mul_(m), alpha=-1.0)
    return {"param_flat": param, "exp_avg": m, "exp_avg_sq": v, "exp_avg_lr": exp_avg_lr}


def _validations(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "P59_adamod_scratch_kernel_ready",
            bool(scratch.get("adamod_native_kernel_parity", False)),
            "adamod_native_scratch_kernel_missing",
        ),
        _validation(
            "training_tensor_binding_parity",
            bool(live_probe.get("training_tensor_binding_parity_passed", False)),
            "adamod_training_tensor_binding_canary_failed",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "adamod_training_tensor_binding_enabled_dispatch",
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


__all__ = ["build_plugin_adamod_training_tensor_binding_canary_scorecard"]
