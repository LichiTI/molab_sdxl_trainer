"""Checkpoint/resume contract helpers for TurboCore update prototypes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

from core.turbocore_flat_adamw_state import PersistentFlatAdamW


def sync_flat_owner_state_from_optimizer(
    owner: Any,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
) -> dict[str, Any]:
    """Mirror PyTorch AdamW moment state into a PersistentFlatAdamW-like owner."""

    param_list = [param for param in params if isinstance(param, torch.nn.Parameter)]
    exp_avg_parts: list[torch.Tensor] = []
    exp_avg_sq_parts: list[torch.Tensor] = []
    step_values: list[int] = []
    missing = 0
    for param in param_list:
        state = optimizer.state.get(param, {}) if optimizer is not None else {}
        exp_avg = state.get("exp_avg") if isinstance(state, dict) else None
        exp_avg_sq = state.get("exp_avg_sq") if isinstance(state, dict) else None
        if isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor):
            exp_avg_parts.append(exp_avg.detach().float().reshape(-1).to(device=owner.exp_avg.device))
            exp_avg_sq_parts.append(exp_avg_sq.detach().float().reshape(-1).to(device=owner.exp_avg_sq.device))
        else:
            exp_avg_parts.append(torch.zeros(int(param.numel()), device=owner.exp_avg.device, dtype=owner.exp_avg.dtype))
            exp_avg_sq_parts.append(torch.zeros(int(param.numel()), device=owner.exp_avg_sq.device, dtype=owner.exp_avg_sq.dtype))
            missing += 1
        step_values.append(_step_to_int(state.get("step") if isinstance(state, dict) else None))
    if exp_avg_parts:
        owner.exp_avg.copy_(torch.cat(exp_avg_parts).to(dtype=owner.exp_avg.dtype))
        owner.exp_avg_sq.copy_(torch.cat(exp_avg_sq_parts).to(dtype=owner.exp_avg_sq.dtype))
    owner.step_index = max(step_values) if step_values else int(getattr(owner, "step_index", 0) or 0)
    return {
        "schema_version": 1,
        "synced": True,
        "state_tensors": len(param_list) - int(missing),
        "missing_state_tensors": int(missing),
        "step_index": int(owner.step_index),
        "training_path_enabled": False,
    }


def sync_optimizer_state_from_flat_owner(
    owner: Any,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
) -> dict[str, Any]:
    """Mirror a PersistentFlatAdamW-like owner state back into PyTorch AdamW.

    Native update dispatch skips ``optimizer.step()`` on the Python optimizer,
    so fallback/resume/checkpoint safety requires the authoritative PyTorch
    optimizer object to receive the same moment tensors and step index.
    """

    param_list = [param for param in params if isinstance(param, torch.nn.Parameter)]
    offset = 0
    synced = 0
    for param in param_list:
        count = int(param.numel())
        state = optimizer.state.setdefault(param, {}) if optimizer is not None else {}
        if not isinstance(state, dict):
            offset += count
            continue
        exp_avg = owner.exp_avg.narrow(0, offset, count).view_as(param).detach()
        exp_avg_sq = owner.exp_avg_sq.narrow(0, offset, count).view_as(param).detach()
        state["exp_avg"] = exp_avg.to(device=param.device, dtype=param.dtype).clone()
        state["exp_avg_sq"] = exp_avg_sq.to(device=param.device, dtype=param.dtype).clone()
        state["step"] = _step_tensor(int(getattr(owner, "step_index", 0) or 0), param)
        synced += 1
        offset += count
    return {
        "schema_version": 1,
        "synced": True,
        "direction": "flat_owner_to_pytorch_optimizer",
        "state_tensors": synced,
        "parameter_tensors": len(param_list),
        "step_index": int(getattr(owner, "step_index", 0) or 0),
        "training_path_enabled": True,
    }


def build_flat_adamw_checkpoint_contract(
    owner: Any,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
    run_roundtrip: bool = False,
    trainer_state_metadata_integrated: bool = False,
    trainer_state_save_sync_verified: bool = False,
    resume_owner_state_guard_verified: bool = False,
) -> dict[str, Any]:
    """Describe checkpoint readiness without wiring it into trainer saves."""

    param_list = [param for param in (params or []) if isinstance(param, torch.nn.Parameter)]
    blocked: list[str] = []
    state_dict_available = callable(getattr(owner, "state_dict", None))
    load_state_dict_available = callable(getattr(owner, "load_state_dict", None))
    if not state_dict_available:
        blocked.append("owner_state_dict_missing")
    if not load_state_dict_available:
        blocked.append("owner_load_state_dict_missing")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "turbocore_flat_adamw_checkpoint_contract_v0",
        "state_dict_available": state_dict_available,
        "load_state_dict_available": load_state_dict_available,
        "trainer_state_metadata_integrated": bool(trainer_state_metadata_integrated),
        "trainer_state_save_sync_verified": bool(trainer_state_save_sync_verified),
        "resume_owner_state_guard_verified": bool(resume_owner_state_guard_verified),
        "trainer_checkpoint_integration": bool(
            trainer_state_metadata_integrated
            and trainer_state_save_sync_verified
            and resume_owner_state_guard_verified
        ),
        "training_path_enabled": False,
        "native_kernel_present": False,
        "roundtrip_checked": False,
        "roundtrip_ok": False,
        "blocked_reasons": [],
    }
    state: Mapping[str, Any] = {}
    if state_dict_available:
        state = owner.state_dict()
        payload.update(
            {
                "state_schema_version": int(state.get("schema_version", 0) or 0),
                "step_index": int(state.get("step_index", 0) or 0),
                "layout_total_numel": int((state.get("layout") or {}).get("total_numel", 0) or 0),
                "state_tensor_roles": [
                    role
                    for role in ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")
                    if isinstance(state.get(role), torch.Tensor)
                ],
            }
        )
    if optimizer is not None and param_list:
        payload["pytorch_optimizer_state"] = _optimizer_state_summary(optimizer, param_list)
    if run_roundtrip and state_dict_available and load_state_dict_available:
        payload.update(_roundtrip_owner_state(state))
    if not payload.get("roundtrip_ok", False):
        blocked.append("roundtrip_not_verified" if not run_roundtrip else "roundtrip_failed")
    if not bool(trainer_state_metadata_integrated):
        blocked.append("trainer_checkpoint_integration_missing")
    if not bool(trainer_state_save_sync_verified):
        blocked.append("trainer_state_save_sync_guard_missing")
    if not bool(resume_owner_state_guard_verified):
        blocked.append("trainer_resume_owner_state_guard_missing")
    payload["ok"] = bool(state_dict_available and load_state_dict_available and payload.get("roundtrip_ok", False))
    payload["blocked_reasons"] = _dedupe(blocked)
    return payload


def _optimizer_state_summary(optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter]) -> dict[str, Any]:
    with_moments = 0
    missing_moments = 0
    steps: list[int] = []
    for param in params:
        state = optimizer.state.get(param, {})
        if isinstance(state, dict) and isinstance(state.get("exp_avg"), torch.Tensor) and isinstance(state.get("exp_avg_sq"), torch.Tensor):
            with_moments += 1
        else:
            missing_moments += 1
        steps.append(_step_to_int(state.get("step") if isinstance(state, dict) else None))
    return {
        "parameter_tensors": len(params),
        "parameter_numel": int(sum(param.numel() for param in params)),
        "state_tensors_with_moments": with_moments,
        "missing_moment_tensors": missing_moments,
        "max_step": max(steps) if steps else 0,
    }


def _roundtrip_owner_state(state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        restored = PersistentFlatAdamW.from_state_dict(dict(state))
        restored_state = restored.state_dict()
        diffs = {
            role: _max_abs_diff(state.get(role), restored_state.get(role))
            for role in ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")
        }
        max_diff = max(diffs.values()) if diffs else 0.0
        return {
            "roundtrip_checked": True,
            "roundtrip_ok": bool(max_diff == 0.0),
            "roundtrip_max_abs_diff": max_diff,
            "roundtrip_role_diffs": diffs,
        }
    except Exception as exc:  # pragma: no cover - defensive report
        return {
            "roundtrip_checked": True,
            "roundtrip_ok": False,
            "roundtrip_error": f"{type(exc).__name__}: {exc}",
        }


def _max_abs_diff(left: Any, right: Any) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return float("inf")
    if left.numel() != right.numel():
        return float("inf")
    return float((left.detach().float() - right.detach().float()).abs().max().cpu().item()) if left.numel() else 0.0


def _step_to_int(step: Any) -> int:
    if isinstance(step, torch.Tensor):
        if step.numel() == 0:
            return 0
        return int(step.detach().reshape(-1)[0].cpu().item())
    try:
        return int(step or 0)
    except (TypeError, ValueError):
        return 0


def _step_tensor(step_index: int, param: torch.nn.Parameter) -> torch.Tensor:
    device = param.device if isinstance(param, torch.Tensor) else torch.device("cpu")
    return torch.tensor(float(step_index), dtype=torch.float32, device=device)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "build_flat_adamw_checkpoint_contract",
    "sync_flat_owner_state_from_optimizer",
    "sync_optimizer_state_from_flat_owner",
]
