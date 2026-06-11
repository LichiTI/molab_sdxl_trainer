"""Report-only state-layout scorecard for factored/custom optimizers.

Adafactor, Automagic++, and AnimaFactoredAdamW have optimizer-owned state
layouts that are useful for memory pressure, but their layouts are not AdamW
compatible.  This scorecard records layout and quality guards before any native
kernel or runtime dispatch work is considered.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.configs import OptimizerType
from core.lulynx_trainer.anima_factored_optimizer import AnimaFactoredAdamW
from core.lulynx_trainer.automagic_plus_plus_optimizer import AutomagicPlusPlus
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report


FACTORED_CUSTOM_OPTIMIZERS = (
    OptimizerType.ADAFACTOR,
    OptimizerType.AUTOMAGIC_PLUS_PLUS,
    OptimizerType.ANIMA_FACTORED_ADAMW,
)


def build_factored_custom_optimizer_state_layout_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Validate factored/custom optimizer layout contracts without dispatch."""

    rows = _contract_rows()
    cases = [
        _automagic_layout_roundtrip_case(),
        _anima_factored_layout_memory_case(),
        _adamw_reuse_guard_case(rows),
    ]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    expected = {optimizer.value for optimizer in FACTORED_CUSTOM_OPTIMIZERS}
    present = {str(row.get("optimizer_type", "")) for row in rows}
    missing = sorted(expected - present)
    classified = not missing and all(bool(row.get("custom_state_layout_required", False)) for row in rows)
    quality_guarded = all(bool(row.get("quality_guard_required", False)) for row in rows)
    adamw_reuse_blocked = all(not bool(row.get("adamw_kernel_compatible", True)) for row in rows)
    ready = not failed and classified and quality_guarded and adamw_reuse_blocked
    if missing:
        blockers.extend(f"missing_factored_custom_contract:{name}" for name in missing)

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_factored_custom_optimizer_state_layout_scorecard_v0",
        "gate": "factored_custom_optimizer_state_layout_reference",
        "ok": ready,
        "promotion_ready": False,
        "state_layout_reference_ready": ready,
        "factored_custom_family_classified": classified,
        "quality_guard_documented": quality_guarded,
        "adamw_kernel_reuse_blocked": adamw_reuse_blocked,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_group": "factored_custom_memory_optimizers",
        "request_contract": {
            "optimizer_types": sorted(expected),
            "native_route_policy": "no_dispatch_until_layout_quality_and_resume_review",
            "runtime_authority": "existing_python_or_third_party_optimizer",
            "full_finetune_quality_matrix_required": True,
            "checkpoint_resume_required": True,
        },
        "state_contract": {
            "requires_custom_state_layout": True,
            "requires_shape_and_dtype_validation": True,
            "requires_resume_shape_validation": True,
            "requires_update_rms_or_clip_guard": True,
            "adamw_state_schema_compatible": False,
        },
        "rows": rows,
        "cases": cases,
        "summary": {
            "optimizer_count": len(rows),
            "expected_optimizer_count": len(expected),
            "classified_count": sum(1 for row in rows if bool(row.get("custom_state_layout_required", False))),
            "local_live_reference_count": sum(1 for case in cases if case.get("uses_local_optimizer") is True),
            "memory_saving_candidate_count": sum(1 for row in rows if bool(row.get("memory_saving_candidate", False))),
            "passed_case_count": len(cases) - len(failed),
            "required_case_count": len(cases),
        },
        "promotion_blockers": blockers
        + ["native_layout_abi_missing", "full_finetune_quality_matrix_missing", "owner_release_hold_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "advance factored/custom owner/release hold and request/schema/UI non-exposure evidence"
            if ready
            else "fix factored/custom optimizer state-layout blockers"
        ),
        "notes": [
            "This scorecard is report-only and does not replace the active Python optimizer path.",
            "Adafactor is kept as an external conceptual contract; local live layout cases use owned implementations.",
            "Factored/custom optimizers must not share the exact AdamW native kernel.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _contract_rows() -> list[dict[str, Any]]:
    capabilities = optimizer_capability_report(FACTORED_CUSTOM_OPTIMIZERS).get("optimizers", [])
    by_name = {str(item.get("optimizer_type", "")): item for item in capabilities if isinstance(item, Mapping)}
    return [_contract_row(optimizer, by_name.get(optimizer.value, {})) for optimizer in FACTORED_CUSTOM_OPTIMIZERS]


def _contract_row(optimizer: OptimizerType, capability: Mapping[str, Any]) -> dict[str, Any]:
    layout = _state_layout(optimizer)
    return {
        "optimizer_type": optimizer.value,
        "implementation": str(capability.get("implementation", "") or ""),
        "dependency": str(capability.get("dependency", "") or ""),
        "dependency_available": bool(capability.get("dependency_available", False)),
        "fallback_optimizer": str(capability.get("fallback_optimizer", "") or ""),
        "current_scheduler_policy": str(capability.get("scheduler_policy", "standard") or "standard"),
        "state_layout_kind": layout["kind"],
        "state_schema_level": layout["level"],
        "state_tensors": layout["state_tensors"],
        "quality_guards": layout["quality_guards"],
        "memory_saving_candidate": bool(layout["memory_saving_candidate"]),
        "custom_state_layout_required": True,
        "quality_guard_required": True,
        "checkpoint_resume_required": True,
        "adamw_kernel_compatible": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "notes": layout["notes"],
    }


def _state_layout(optimizer: OptimizerType) -> dict[str, Any]:
    if optimizer == OptimizerType.AUTOMAGIC_PLUS_PLUS:
        return {
            "kind": "factored_preconditioner_with_local_lr_mask",
            "level": "observed_local_state_dict",
            "state_tensors": ["local_lr", "prev_sign", "row_var_or_col_var_or_full_var", "avg_lr", "momentum_optional"],
            "quality_guards": ["min_lr_max_lr_clamp", "clip_threshold", "max_update_rms_ratio", "finite_update_required"],
            "memory_saving_candidate": True,
            "notes": ["Local optimizer; live case validates state shape and resume."],
        }
    if optimizer == OptimizerType.ANIMA_FACTORED_ADAMW:
        return {
            "kind": "full_first_moment_factored_second_moment",
            "level": "observed_local_state_dict",
            "state_tensors": ["exp_avg", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq_for_small_tensors"],
            "quality_guards": ["min_dim", "min_numel", "factored_eps", "full_finetune_short_matrix_required"],
            "memory_saving_candidate": True,
            "notes": ["Local full-finetune optimizer; large 2D tensors factor second moment only."],
        }
    return {
        "kind": "external_adafactor_factored_second_moment",
        "level": "conceptual_external_contract",
        "state_tensors": ["row_factor", "col_factor", "full_second_moment_for_small_or_unfactored_tensors"],
        "quality_guards": ["clip_threshold", "relative_step_policy", "scale_parameter_policy", "resume_shape_validation"],
        "memory_saving_candidate": True,
        "notes": ["Third-party implementation; exact state names are not copied into this scorecard."],
    }


def _automagic_layout_roundtrip_case() -> dict[str, Any]:
    value = torch.linspace(-0.2, 0.3, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.16, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.05, 0.04, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    param = torch.nn.Parameter(value.clone())
    optimizer = AutomagicPlusPlus([param], lr=1e-4, beta1=0.9, max_update_rms_ratio=None)

    _step(param, optimizer, grad1)
    after_step = _automagic_state_contract(optimizer.state_dict())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = param.detach().clone()
    restored_param = torch.nn.Parameter(saved_param.clone())
    restored = AutomagicPlusPlus([restored_param], lr=1e-4, beta1=0.9, max_update_rms_ratio=None)
    restored.load_state_dict(saved_state)
    _step(param, optimizer, grad2)
    _step(restored_param, restored, grad2)
    diff = _max_abs(param.detach(), restored_param.detach())
    ok = after_step["has_required_state"] and after_step["local_lr_shape"] == [4, 4] and diff <= 1e-6
    return {
        "schema_version": 1,
        "case": "automagic_layout_roundtrip",
        "ok": ok,
        "uses_local_optimizer": True,
        "covers_resume": True,
        "after_step": after_step,
        "max_resume_diff": diff,
        "tolerance": 1e-6,
        "blocked_reasons": [] if ok else ["automagic_layout_roundtrip_failed"],
    }


def _anima_factored_layout_memory_case() -> dict[str, Any]:
    value = torch.linspace(-0.1, 0.1, steps=256 * 256, dtype=torch.float32).reshape(256, 256)
    grad1 = torch.linspace(0.001, 0.01, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.004, 0.006, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    param = torch.nn.Parameter(value.clone())
    optimizer = AnimaFactoredAdamW([param], lr=1e-4, min_dim=128, min_numel=65536)

    _step(param, optimizer, grad1)
    after_step = _anima_state_contract(optimizer.state_dict(), optimizer.get_profile())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = param.detach().clone()
    restored_param = torch.nn.Parameter(saved_param.clone())
    restored = AnimaFactoredAdamW([restored_param], lr=1e-4, min_dim=128, min_numel=65536)
    restored.load_state_dict(saved_state)
    _step(param, optimizer, grad2)
    _step(restored_param, restored, grad2)
    diff = _max_abs(param.detach(), restored_param.detach())
    ok = after_step["is_factored"] and after_step["estimated_second_moment_saved_mb"] > 0 and diff <= 1e-6
    return {
        "schema_version": 1,
        "case": "anima_factored_layout_memory",
        "ok": ok,
        "uses_local_optimizer": True,
        "covers_resume": True,
        "covers_memory_saving": True,
        "after_step": after_step,
        "max_resume_diff": diff,
        "tolerance": 1e-6,
        "blocked_reasons": [] if ok else ["anima_factored_layout_memory_failed"],
    }


def _adamw_reuse_guard_case(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    incompatible = [str(row.get("optimizer_type", "")) for row in rows if bool(row.get("adamw_kernel_compatible", True))]
    missing_layout = [str(row.get("optimizer_type", "")) for row in rows if not bool(row.get("custom_state_layout_required", False))]
    ok = not incompatible and not missing_layout
    return {
        "schema_version": 1,
        "case": "adamw_kernel_reuse_guard",
        "ok": ok,
        "incompatible_with_exact_adamw_kernel_count": len(rows) - len(incompatible),
        "custom_state_layout_required_count": len(rows) - len(missing_layout),
        "blocked_reasons": [] if ok else ["factored_custom_adamw_reuse_guard_failed"],
    }


def _automagic_state_contract(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = _first_state(state_dict)
    keys = sorted(str(key) for key in state.keys())
    local_lr = state.get("local_lr")
    row_var = state.get("row_var")
    col_var = state.get("col_var")
    prev_sign = state.get("prev_sign")
    return {
        "param_state_keys": keys,
        "has_required_state": {"step", "local_lr", "prev_sign", "row_var", "col_var", "avg_lr"}.issubset(set(keys)),
        "local_lr_shape": _shape(local_lr),
        "row_var_shape": _shape(row_var),
        "col_var_shape": _shape(col_var),
        "prev_sign_dtype": str(prev_sign.dtype) if torch.is_tensor(prev_sign) else "",
    }


def _anima_state_contract(state_dict: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    state = _first_state(state_dict)
    keys = sorted(str(key) for key in state.keys())
    row = state.get("exp_avg_sq_row")
    col = state.get("exp_avg_sq_col")
    return {
        "param_state_keys": keys,
        "is_factored": bool(state.get("factored", False)),
        "has_required_factored_state": {"step", "exp_avg", "exp_avg_sq_row", "exp_avg_sq_col", "factored"}.issubset(set(keys)),
        "row_shape": _shape(row),
        "col_shape": _shape(col),
        "estimated_second_moment_saved_mb": float(profile.get("estimated_second_moment_saved_mb", 0.0) or 0.0),
        "factored_numel": int(profile.get("factored_numel", 0) or 0),
    }


def _first_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    first = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    return dict(first) if isinstance(first, Mapping) else {}


def _shape(value: Any) -> list[int]:
    return list(value.shape) if torch.is_tensor(value) else []


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_factored_custom_optimizer_state_layout_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["FACTORED_CUSTOM_OPTIMIZERS", "build_factored_custom_optimizer_state_layout_scorecard"]
