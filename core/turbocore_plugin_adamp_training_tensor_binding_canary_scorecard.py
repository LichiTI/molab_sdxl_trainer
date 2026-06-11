"""Default-off training tensor binding canary for selected plugin adamp."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.turbocore_plugin_adamp_native_scratch_kernel_scorecard import (
    build_plugin_adamp_native_scratch_kernel_scorecard,
)
from core.turbocore_tensor_handle_registry import (
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adamp_tensor_probe",
    "destroy_tensor_binding_session",
)
FLOAT_TOLERANCE = 5e-6
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_adamp_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(
        scratch_report
        or build_plugin_adamp_native_scratch_kernel_scorecard(workspace_root=workspace_root, arch=arch)
    )
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch) if run_live_probe else _skipped("live_probe_disabled")
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamp_training_tensor_binding_canary_scorecard_v0",
        "gate": "plugin_adamp_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_e2e_no_regression_ready": bool(live_probe.get("e2e_no_regression_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoints": list(ENTRYPOINTS),
        "optimizer_kind": "adamp",
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
                "adamp_runtime_dispatch_shadow_missing",
                "adamp_training_loop_canary_missing",
                "adamp_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected adamp runtime dispatch shadow before TrainingLoop canary"
            if ready
            else "fix selected adamp training tensor binding canary blockers"
        ),
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _failed("lulynx_native_entrypoint_missing", ["adamp_training_tensor_binding_entrypoint_missing"])
    request, tensor_map, role_tensors = _make_cuda_request()
    config = {
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "rows": 4,
        "cols": 8,
        "lr": 0.021,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.017,
        "weight_decouple": True,
        "fixed_decay": False,
        "delta": 0.1,
        "wd_ratio": 0.1,
        "nesterov": False,
        "adam_debias": False,
        "step_index": 2,
        "block_size": 128,
        "max_numel": 64,
    }
    reference = _reference_adamp(role_tensors, config)
    session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
    if not bool(session.get("ok", False)):
        return _failed("tensor_binding_session_create_failed", ["adamp_tensor_binding_session_create_failed"], session=session)
    session_id = int(session["session_id"])
    try:
        launch = dict(native.tensor_binding_session_cuda_adamp_tensor_probe(session_id, json.dumps(config)))
        if not bool(launch.get("ok", False)):
            return _failed(
                f"adamp_tensor_probe_launch_failed:{launch.get('reason', 'unknown')}",
                ["adamp_tensor_probe_launch_failed"],
                session=session,
                launch=launch,
            )
        torch.cuda.synchronize()
        diffs = {
            "param_flat": _max_abs_diff(role_tensors["param_flat"], reference["param_flat"]),
            "exp_avg": _max_abs_diff(role_tensors["exp_avg"], reference["exp_avg"]),
            "exp_avg_sq": _max_abs_diff(role_tensors["exp_avg_sq"], reference["exp_avg_sq"]),
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
            "probe_kind": "plugin_adamp_training_tensor_binding_canary_v0",
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
            "blocked_reasons": [] if ok else ["adamp_training_tensor_binding_parity_failed"],
        }
    finally:
        destroyed = dict(native.destroy_tensor_binding_session(session_id))
        if not bool(destroyed.get("ok", False)):
            pass


def _make_cuda_request() -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    rows, cols = 4, 8
    numel = rows * cols
    registry = TurboCoreTensorHandleRegistry(namespace="plugin_adamp_tensor_binding_canary")
    param = (
        torch.arange(numel, dtype=torch.float32, device=device).reshape(rows, cols) * 0.011 + 0.14
    ).contiguous().flatten()
    signs = torch.where(torch.arange(numel, device=device) % 2 == 0, 1.0, -1.0)
    grad = (torch.arange(numel, dtype=torch.float32, device=device) * 0.0007 + 0.008).mul(signs).contiguous()
    m = (torch.arange(numel, dtype=torch.float32, device=device) * 0.00031 - 0.0015).contiguous()
    v = (torch.arange(numel, dtype=torch.float32, device=device) * 0.000011 + 0.00019).contiguous()
    handles = registry.register_flat_adamw_buffers(
        param_flat=param,
        grad_flat=grad,
        exp_avg=m,
        exp_avg_sq=v,
    )
    request = build_flat_adamw_native_binding_request(registry, handles)
    request["optimizer"] = "AdamP"
    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    role_tensors = {binding["role"]: tensor_map[binding["handle_id"]] for binding in request["bindings"]}
    return request, tensor_map, role_tensors


def _reference_adamp(role_tensors: dict[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    param = role_tensors["param_flat"].detach().clone()
    grad = role_tensors["grad_flat"].detach().clone()
    m = role_tensors["exp_avg"].detach().clone()
    v = role_tensors["exp_avg_sq"].detach().clone()
    beta1, beta2 = [float(item) for item in config["betas"]]
    lr = float(config["lr"])
    eps = float(config["eps"])
    step_number = int(config["step_index"]) + 1
    weight_decay = float(config["weight_decay"])
    weight_decouple = bool(config.get("weight_decouple", True))
    fixed_decay = bool(config.get("fixed_decay", False))
    delta = float(config["delta"])
    wd_ratio = float(config["wd_ratio"])
    nesterov = bool(config.get("nesterov", False))
    adam_debias = bool(config.get("adam_debias", False))
    rows = int(config["rows"])
    cols = int(config["cols"])

    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step_number
    bias_correction2_sqrt = (1.0 - beta2**step_number) ** 0.5
    inv_denom = bias_correction2_sqrt / (v.sqrt().add(eps))
    if nesterov:
        perturb = beta1 * m + (1.0 - beta1) * grad * inv_denom
    else:
        perturb = m * inv_denom
    perturb, projection_ratio = _project_perturb(param, grad, perturb, rows, cols, delta, wd_ratio, eps)
    if weight_decay and weight_decouple:
        decay = weight_decay * (1.0 if fixed_decay else lr) * projection_ratio
        param.mul_(1.0 - decay)
    step_size = lr if adam_debias else lr / bias_correction1
    param.add_(perturb, alpha=-step_size)
    return {"param_flat": param, "exp_avg": m, "exp_avg_sq": v}


def _project_perturb(
    param: torch.Tensor,
    grad: torch.Tensor,
    perturb: torch.Tensor,
    rows: int,
    cols: int,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> tuple[torch.Tensor, float]:
    if rows <= 1 or rows * cols != int(param.numel()):
        return perturb, 1.0
    p2 = param.reshape(rows, cols)
    g2 = grad.reshape(rows, cols)
    u2 = perturb.reshape(rows, cols)
    row_cos = _cosine_abs(g2, p2, dim=1, eps=eps).max()
    if bool(row_cos < delta * (cols**-0.5)):
        param_norm = p2.norm(dim=1, keepdim=True).add(eps)
        param_unit = p2 / param_norm
        projected = u2 - param_unit * (param_unit * u2).sum(dim=1, keepdim=True)
        return projected.reshape_as(perturb), wd_ratio
    layer_cos = _cosine_abs(grad, param, dim=0, eps=eps)
    if bool(layer_cos < delta * (int(param.numel()) ** -0.5)):
        param_norm = param.norm().add(eps)
        param_unit = param / param_norm
        return perturb - param_unit * torch.dot(param_unit, perturb), wd_ratio
    return perturb, 1.0


def _cosine_abs(left: torch.Tensor, right: torch.Tensor, *, dim: int, eps: float) -> torch.Tensor:
    dot = (left * right).sum(dim=dim).abs()
    denom = left.norm(dim=dim) * right.norm(dim=dim)
    return torch.where(denom > eps, dot / denom, torch.zeros_like(dot))


def _validations(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "P63_adamp_scratch_kernel_ready",
            bool(scratch.get("adamp_native_kernel_parity", False)),
            "adamp_native_scratch_kernel_missing",
        ),
        _validation(
            "training_tensor_binding_parity",
            bool(live_probe.get("training_tensor_binding_parity_passed", False)),
            "adamp_training_tensor_binding_canary_failed",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "adamp_training_tensor_binding_enabled_dispatch",
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


__all__ = ["build_plugin_adamp_training_tensor_binding_canary_scorecard"]
