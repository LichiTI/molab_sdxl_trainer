"""Report-only parity matrix plans for custom-formula plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping


def build_custom_formula_parity_plan_artifacts(
    name: str,
    artifacts: Mapping[str, str],
    formula_spec_artifact: Mapping[str, Any] | None,
    state_inventory_artifact: Mapping[str, Any] | None,
    quality_guard_artifact: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not formula_spec_artifact or not state_inventory_artifact or not quality_guard_artifact:
        return None, None
    formula_cases = _formula_cases(name, formula_spec_artifact, quality_guard_artifact)
    resume_cases = _resume_cases(name, state_inventory_artifact)
    source_class = str(state_inventory_artifact.get("source_class", ""))
    source_file = str(state_inventory_artifact.get("source_file", ""))
    formula = {
        "schema_version": 1,
        "artifact": str(artifacts["formula_parity_matrix"]),
        "status": "planned",
        "report_only": True,
        "source_review_target": f"pytorch_optimizer:{name}",
        "source_class": source_class,
        "source_file": source_file,
        "formula_family": str(formula_spec_artifact.get("formula_family", "")),
        "case_count": len(formula_cases),
        "cases": formula_cases,
        "implementation_ready": False,
        "native_kernel_ready": False,
    }
    resume = {
        "schema_version": 1,
        "artifact": str(artifacts["resume_parity_matrix"]),
        "status": "planned",
        "report_only": True,
        "source_review_target": f"pytorch_optimizer:{name}",
        "source_class": source_class,
        "source_file": source_file,
        "state_dict_key_inventory": list(state_inventory_artifact.get("state_dict_key_inventory", [])),
        "case_count": len(resume_cases),
        "cases": resume_cases,
        "implementation_ready": False,
        "native_kernel_ready": False,
    }
    return formula, resume


def promote_custom_formula_parity_artifacts(
    formula_artifact: Mapping[str, Any] | None,
    resume_artifact: Mapping[str, Any] | None,
    execution_row: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Mark report-only parity matrices ready when plugin execution evidence passed."""

    if not formula_artifact or not resume_artifact or not execution_row:
        return _copy_or_none(formula_artifact), _copy_or_none(resume_artifact)
    formula_ready = execution_row.get("formula_parity_implementation_ready") is True
    resume_ready = execution_row.get("resume_parity_implementation_ready") is True
    formula = _promote(formula_artifact, execution_row, formula_ready, "formula_step_execution")
    resume = _promote(resume_artifact, execution_row, resume_ready, "resume_next_step_replay")
    return formula, resume


def _promote(
    artifact: Mapping[str, Any],
    execution_row: Mapping[str, Any],
    ready: bool,
    proof_kind: str,
) -> dict[str, Any]:
    out = dict(artifact)
    if ready:
        out.update(
            {
                "status": "implementation_ready",
                "implementation_ready": True,
                "implementation_evidence": {
                    "schema_version": 1,
                    "proof_kind": proof_kind,
                    "execution_reference": str(execution_row.get("execution_reference", "")),
                    "source_file": str(execution_row.get("source_file", "")),
                    "source_class": str(execution_row.get("source_class", "")),
                    "formula_step_execution_ready": execution_row.get("formula_step_execution_ready") is True,
                    "resume_next_step_replay_ready": execution_row.get("resume_next_step_replay_ready") is True,
                    "state_dict_state_count": int(execution_row.get("state_dict_state_count", 0) or 0),
                },
                "native_kernel_ready": False,
                "product_native_ready": False,
                "runtime_dispatch_ready": False,
                "native_dispatch_allowed": False,
            }
        )
    return out


def _copy_or_none(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(value) if value else None


def _formula_cases(
    name: str,
    formula_spec_artifact: Mapping[str, Any],
    quality_guard_artifact: Mapping[str, Any],
) -> list[dict[str, str]]:
    family = str(formula_spec_artifact.get("formula_family", "custom_formula"))
    guard_count = int(quality_guard_artifact.get("guard_case_count", 0) or 0)
    return [
        _case(name, "dense_fp32_reference_step", f"{family}: compare one dense fp32 step against plugin"),
        _case(name, "none_grad_skip", "parameter with grad=None must keep parameter and state unchanged"),
        _case(name, "weight_decay_branch", "cover coupled or decoupled weight decay placement"),
        _case(name, "hparam_boundary", "cover epsilon, beta, clamp, bound, phase, or threshold edge values"),
        _case(name, "quality_guard_boundaries", f"execute {guard_count} planned quality guard boundaries"),
        _case(name, "dtype_device_boundary", "cover fp32 CUDA tensor and reject unsupported dtype/device paths"),
    ]


def _resume_cases(name: str, state_inventory_artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    state_count = len(list(state_inventory_artifact.get("state_dict_key_inventory", [])))
    return [
        _case(name, "state_dict_round_trip", f"round-trip {state_count} state_dict keys without loss"),
        _case(name, "param_group_step_restore", "restore param-group step and hparam surface before next step"),
        _case(name, "dtype_device_restore", "restore state tensors onto the expected dtype/device"),
        _case(name, "next_step_parity", "after load_state_dict, next plugin step matches uninterrupted training"),
    ]


def _case(name: str, suffix: str, expectation: str) -> dict[str, str]:
    return {
        "case_id": f"{name}:{suffix}",
        "expectation": expectation,
    }


__all__ = [
    "build_custom_formula_parity_plan_artifacts",
    "promote_custom_formula_parity_artifacts",
]
