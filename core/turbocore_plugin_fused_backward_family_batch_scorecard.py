"""Report-only batch scorecard for selected plugin fused-backward optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_fused_backward_execution_matrix import (
    build_fused_backward_execution_matrix,
    promote_fused_backward_row,
)


TARGET_PLUGIN_OPTIMIZERS: tuple[str, ...] = ("adalomo", "lomo")
FUSED_BACKWARD_FAMILY = "fused_backward"

UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "product_native_dispatch_ready",
    "product_native_ready",
    "native_kernel_ready",
)

_ABI_REQUIREMENTS_BY_OPTIMIZER: dict[str, dict[str, bool]] = {
    "adalomo": {
        "requires_fused_backward": True,
        "requires_backward_hook": True,
        "requires_gradient_ownership_abi": True,
        "requires_skip_step_contract": True,
        "requires_loss_scale_boundary": True,
    },
    "lomo": {
        "requires_fused_backward": True,
        "requires_backward_hook": True,
        "requires_gradient_ownership_abi": True,
        "requires_skip_step_contract": True,
        "requires_loss_scale_boundary": True,
    },
}

_ABI_SPEC_BY_OPTIMIZER: dict[str, dict[str, Any]] = {
    "adalomo": {
        "optimizer_step_policy": "forbid_public_optimizer_step_call",
        "backward_hook": {
            "attachment_point": "loss_backward_owner",
            "expected_hook_order": ["loss.backward", "gradient_ready_hook", "native_update"],
            "requires_grad_callback_per_parameter": True,
            "requires_backward_completion_barrier": True,
        },
        "gradient_ownership": {
            "gradient_owner": "fused_backward_hook",
            "optimizer_step_reads_grad": False,
            "native_update_consumes_gradient_once": True,
            "zero_grad_boundary": "after_native_update_ack",
        },
        "skip_optimizer_step": {
            "training_loop_action": "skip_optimizer_step_after_fused_backward_update",
            "forbidden_call": "optimizer.step",
            "requires_explicit_skip_evidence": True,
        },
        "loss_backward_call_site": {
            "authoritative_call_site": "training_loop_loss_backward",
            "requires_loss_tensor_before_backward": True,
            "requires_backward_invocation_token": True,
            "mixed_precision_boundary": "loss_scaler_unscale_before_native_update",
        },
        "state_resume_scope": {
            "owned_state": ["step", "exp_avg_sq_or_lomo_state", "param_group_hparams"],
            "resume_adapter": "plugin_state_dict_plus_fused_backward_owner_state",
            "requires_mid_backward_resume_block": True,
        },
        "native_kernel_preconditions": {
            "requires_flat_param_buffer": True,
            "requires_flat_grad_buffer": True,
            "requires_backward_hook_ownership_token": True,
            "requires_loss_scale_metadata": True,
            "requires_stream_ordering_token": True,
        },
    },
    "lomo": {
        "optimizer_step_policy": "forbid_public_optimizer_step_call",
        "backward_hook": {
            "attachment_point": "loss_backward_owner",
            "expected_hook_order": ["loss.backward", "gradient_ready_hook", "native_update"],
            "requires_grad_callback_per_parameter": True,
            "requires_backward_completion_barrier": True,
        },
        "gradient_ownership": {
            "gradient_owner": "fused_backward_hook",
            "optimizer_step_reads_grad": False,
            "native_update_consumes_gradient_once": True,
            "zero_grad_boundary": "after_native_update_ack",
        },
        "skip_optimizer_step": {
            "training_loop_action": "skip_optimizer_step_after_fused_backward_update",
            "forbidden_call": "optimizer.step",
            "requires_explicit_skip_evidence": True,
        },
        "loss_backward_call_site": {
            "authoritative_call_site": "training_loop_loss_backward",
            "requires_loss_tensor_before_backward": True,
            "requires_backward_invocation_token": True,
            "mixed_precision_boundary": "loss_scaler_unscale_before_native_update",
        },
        "state_resume_scope": {
            "owned_state": ["step", "lomo_state", "param_group_hparams"],
            "resume_adapter": "plugin_state_dict_plus_fused_backward_owner_state",
            "requires_mid_backward_resume_block": True,
        },
        "native_kernel_preconditions": {
            "requires_flat_param_buffer": True,
            "requires_flat_grad_buffer": True,
            "requires_backward_hook_ownership_token": True,
            "requires_loss_scale_metadata": True,
            "requires_stream_ordering_token": True,
        },
    },
}


def build_plugin_fused_backward_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected plugin fused-backward status without dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    selector_rows = _selector_rows(selector)
    selected_names = tuple(sorted(selector_rows))
    rows = [_row(name, selector_rows.get(name, {})) for name in selected_names]
    execution_matrix = build_fused_backward_execution_matrix(rows)
    rows = [promote_fused_backward_row(row, execution_matrix) for row in rows]
    missing = [name for name in TARGET_PLUGIN_OPTIMIZERS if name not in selector_rows]
    unexpected = [name for name in selected_names if name not in TARGET_PLUGIN_OPTIMIZERS]
    unsafe = _unsafe_claims({"selector": selector}, rows)
    selector_count = int(_as_dict(_summary(selector).get("route_family_counts")).get(FUSED_BACKWARD_FAMILY, 0) or 0)
    ready = selector.get("ok") is True and execution_matrix.get("ok") is True and not missing and not unexpected and not unsafe

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_fused_backward_family_batch_scorecard_v0",
        "gate": "plugin_fused_backward_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_fused_backward_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": FUSED_BACKWARD_FAMILY,
        "selector_scorecard": _compact_selector(selector),
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(selected_names),
            "expected_selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "runtime_authority": "existing_pytorch_optimizer_plugin",
            "native_route_policy": "blocked_until_fused_backward_hook_gradient_ownership_abi",
        },
        "fused_backward_abi": {
            "requires_fused_backward": True,
            "requires_backward_hook_contract": True,
            "requires_gradient_ownership_abi": True,
            "requires_skip_step_contract": True,
            "requires_training_loop_boundary_review": True,
            "per_optimizer_abi_spec_ready": ready,
            "abi_implementation_ready": execution_matrix.get("ok") is True,
            "adamw_step_kernel_compatible": False,
            "simple_formula_kernel_compatible": False,
            "can_reuse_exact_adamw_native_dispatch": False,
        },
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selector_fused_backward_count": selector_count,
            "selector_classified_count": sum(1 for row in rows if row["selector_classified"] is True),
            "fused_backward_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_fused_backward"] is True
            ),
            "backward_hook_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_backward_hook"] is True
            ),
            "gradient_ownership_abi_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_gradient_ownership_abi"] is True
            ),
            "skip_step_contract_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_skip_step_contract"] is True
            ),
            "per_optimizer_abi_spec_ready_count": sum(
                1 for row in rows if row["fused_backward_abi_spec_ready"] is True
            ),
            "backward_hook_spec_ready_count": sum(1 for row in rows if row["abi_spec"]["backward_hook"]),
            "gradient_ownership_spec_ready_count": sum(
                1 for row in rows if row["abi_spec"]["gradient_ownership"]
            ),
            "skip_optimizer_step_spec_ready_count": sum(
                1 for row in rows if row["abi_spec"]["skip_optimizer_step"]
            ),
            "loss_backward_call_site_spec_ready_count": sum(
                1 for row in rows if row["abi_spec"]["loss_backward_call_site"]
            ),
            "state_resume_scope_spec_ready_count": sum(
                1 for row in rows if row["abi_spec"]["state_resume_scope"]
            ),
            "native_kernel_preconditions_spec_ready_count": sum(
                1 for row in rows if row["abi_spec"]["native_kernel_preconditions"]
            ),
            "resume_parity_matrix_spec_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix"]["matrix_spec_ready"] is True
            ),
            "resume_parity_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get("resume_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "fused_backward_replay_case_planned_count": sum(
                len(row["resume_parity_matrix"]["planned_cases"]) for row in rows
            ),
            "loss_scale_boundary_case_planned_count": sum(
                1
                for row in rows
                for case in row["resume_parity_matrix"]["planned_cases"]
                if case["case_group"] == "loss_scale_boundary"
            ),
            "fused_backward_replay_case_implementation_ready_count": int(
                _summary(execution_matrix).get("fused_backward_replay_case_implementation_ready_count", 0) or 0
            ),
            "fused_backward_abi_implementation_ready_count": int(
                _summary(execution_matrix).get("fused_backward_abi_implementation_ready_count", 0) or 0
            ),
            "native_kernel_preconditions_implementation_ready_count": int(
                _summary(execution_matrix).get("native_kernel_preconditions_implementation_ready_count", 0) or 0
            ),
            "adamw_kernel_compatible_count": 0,
            "simple_kernel_compatible_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "product_native_dispatch_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "plugin_selected_native_ready_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "missing_expected_selector_count": len(missing),
            "unexpected_selector_count": len(unexpected),
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [f"selector_fused_backward_missing:{name}" for name in missing]
            + [f"selector_fused_backward_unexpected:{name}" for name in unexpected]
            + [
                "adamw_native_simple_kernel_not_reusable",
                "native_kernel_implementation_missing",
                "runtime_dispatch_shadow_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(
            unsafe
            + [f"selector_fused_backward_missing:{name}" for name in missing]
            + [f"selector_fused_backward_unexpected:{name}" for name in unexpected]
        ),
        "recommended_next_step": (
            "owner/release hold for implementation-ready fused-backward ABI matrices with dispatch default-off"
            if ready
            else "fix selector fused-backward blockers before ABI drafting"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Fused-backward plugins depend on backward hooks and gradient ownership inside the training loop.",
            "Resume/parity matrices are specified per optimizer but are not implemented or dispatched.",
            "These optimizers cannot reuse exact AdamW or simple-formula native kernels.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _call_selector() -> dict[str, Any]:
    try:
        return dict(build_plugin_optimizer_selector_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_optimizer_selector_scorecard", exc)


def _selector_rows(selector: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = selector.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("optimizer_name", "")).strip().lower(): row
        for row in rows
        if isinstance(row, Mapping) and row.get("native_route_family") == FUSED_BACKWARD_FAMILY
    }


def _row(name: str, selector_row: Mapping[str, Any]) -> dict[str, Any]:
    requirements = dict(_ABI_REQUIREMENTS_BY_OPTIMIZER.get(name, _default_abi_requirements()))
    abi_spec = _abi_spec(name)
    classified = selector_row.get("native_route_family") == FUSED_BACKWARD_FAMILY
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": str(selector_row.get("selector", OptimizerType.PYTORCH_OPTIMIZER.value)),
        "native_route_family": FUSED_BACKWARD_FAMILY,
        "selector_classified": classified,
        "resume_proven": selector_row.get("resume_proven") is True,
        "batch_status": "abi_required_report_only" if classified else "selector_classification_pending",
        "abi_requirements": requirements,
        "abi_spec": abi_spec,
        "resume_parity_matrix": _resume_parity_matrix(name),
        "fused_backward_abi_spec_ready": classified and bool(abi_spec),
        "fused_backward_abi_implementation_ready": False,
        "native_kernel_preconditions_implementation_ready": False,
        "runtime_authority": "existing_pytorch_optimizer_plugin",
        "native_route": "none_report_only",
        "adamw_kernel_compatible": False,
        "simple_formula_kernel_compatible": False,
        "can_reuse_exact_adamw_native_dispatch": False,
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "selected_plugin_fused_backward_hook_gradient_ownership_abi",
        "blocked_reasons": [
            "selected_plugin_fused_backward_hook_abi_missing",
            "selected_plugin_gradient_ownership_abi_missing",
            "selected_plugin_skip_step_training_loop_contract_missing",
            "selected_plugin_loss_backward_call_site_implementation_missing",
            "selected_plugin_state_resume_adapter_implementation_missing",
            "selected_plugin_fused_backward_resume_parity_matrix_implementation_missing",
            "selected_plugin_native_kernel_preconditions_implementation_missing",
            "adamw_native_simple_kernel_not_reusable",
            "native_dispatch_gate_not_requested",
        ],
    }


def _abi_spec(name: str) -> dict[str, Any]:
    spec = _ABI_SPEC_BY_OPTIMIZER.get(name)
    if isinstance(spec, Mapping):
        return dict(spec)
    return {
        "optimizer_step_policy": "forbid_public_optimizer_step_call",
        "backward_hook": {"attachment_point": "loss_backward_owner"},
        "gradient_ownership": {"gradient_owner": "fused_backward_hook"},
        "skip_optimizer_step": {"forbidden_call": "optimizer.step"},
        "loss_backward_call_site": {"authoritative_call_site": "training_loop_loss_backward"},
        "state_resume_scope": {"resume_adapter": "plugin_state_dict_plus_fused_backward_owner_state"},
        "native_kernel_preconditions": {
            "requires_flat_param_buffer": True,
            "requires_flat_grad_buffer": True,
            "requires_backward_hook_ownership_token": True,
        },
    }


def _default_abi_requirements() -> dict[str, bool]:
    return {
        "requires_fused_backward": True,
        "requires_backward_hook": True,
        "requires_gradient_ownership_abi": True,
        "requires_skip_step_contract": True,
        "requires_loss_scale_boundary": True,
    }


def _resume_parity_matrix(name: str) -> dict[str, Any]:
    cases = [
        ("baseline_resume", "single_step_then_resume"),
        ("baseline_resume", "gradient_accumulation_boundary"),
        ("loss_scale_boundary", "amp_loss_scale_before_native_update"),
        ("loss_scale_boundary", "unscale_then_backward_hook_order"),
        ("state_ownership", "zero_grad_after_native_update_ack"),
    ]
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "matrix_spec_ready": True,
        "matrix_implementation_ready": False,
        "planned_cases": [
            {
                "case_id": f"{name}:{case_id}",
                "case_group": group,
                "requires_backward_hook_ownership_token": True,
                "requires_skip_public_optimizer_step": True,
                "requires_state_dict_roundtrip": True,
                "native_dispatch_allowed": False,
            }
            for group, case_id in cases
        ],
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "fused_backward_count": int(counts.get(FUSED_BACKWARD_FAMILY, 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
    }


def _compact_execution_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "execution_matrix_ready": report.get("execution_matrix_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "fused_backward_abi_implementation_ready_count": int(
            summary.get("fused_backward_abi_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "fused_backward_replay_case_implementation_ready_count": int(
            summary.get("fused_backward_replay_case_implementation_ready_count", 0) or 0
        ),
        "blocked_reasons": [str(reason) for reason in report.get("blocked_reasons", []) or [] if str(reason)],
    }


def _unsafe_claims(
    reports: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[str]:
    out: list[str] = []
    for name, report in reports.items():
        scorecard = str(report.get("scorecard", name))
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                out.append(f"unsafe_plugin_fused_backward_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_fused_backward_row:{selected}:{field}")
        if row.get("adamw_kernel_compatible") is True or row.get("simple_formula_kernel_compatible") is True:
            out.append(f"unsafe_plugin_fused_backward_row:{selected}:kernel_compatible")
    return _dedupe(out)


def _failed_report(builder_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "scorecard": builder_name,
        "error": f"{type(exc).__name__}: {exc}",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "native_kernel_ready": False,
        "blocked_reasons": [f"builder_failed:{builder_name}"],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_fused_backward_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "FUSED_BACKWARD_FAMILY",
    "TARGET_PLUGIN_OPTIMIZERS",
    "build_plugin_fused_backward_family_batch_scorecard",
]
