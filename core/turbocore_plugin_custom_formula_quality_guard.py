"""Report-only quality guard matrix for custom-formula plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping


def build_custom_formula_quality_guard_artifact(
    name: str,
    artifacts: Mapping[str, str],
    formula_spec_artifact: Mapping[str, Any] | None,
    state_inventory_artifact: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not formula_spec_artifact or not state_inventory_artifact:
        return None
    guard_names = _guard_names(formula_spec_artifact)
    source_class = str(state_inventory_artifact.get("source_class", ""))
    source_file = str(state_inventory_artifact.get("source_file", ""))
    cases = _guard_cases(name, guard_names, state_inventory_artifact)
    return {
        "schema_version": 1,
        "artifact": str(artifacts["quality_guard_matrix"]),
        "status": "ready",
        "report_only": True,
        "source_review_target": f"pytorch_optimizer:{name}",
        "source_class": source_class,
        "source_file": source_file,
        "guard_source": str(formula_spec_artifact.get("artifact", "")),
        "state_inventory_source": str(state_inventory_artifact.get("artifact", "")),
        "guard_case_count": len(cases),
        "guard_cases": cases,
        "sparse_gradient_policy": state_inventory_artifact.get("sparse_gradient_policy"),
        "complex_tensor_policy": state_inventory_artifact.get("complex_tensor_policy"),
        "hparam_surface_keys": list(state_inventory_artifact.get("hparam_surface_keys", [])),
        "native_kernel_ready": False,
        "formula_parity_status": "pending",
        "resume_parity_status": "pending",
    }


def custom_formula_quality_guard_ready(
    formula_spec_artifact: Mapping[str, Any] | None,
    state_inventory_artifact: Mapping[str, Any] | None,
) -> bool:
    return bool(formula_spec_artifact) and bool(state_inventory_artifact)


def _guard_names(formula_spec_artifact: Mapping[str, Any]) -> list[str]:
    names = formula_spec_artifact.get("quality_guard_skeleton", [])
    return [str(name) for name in names if str(name).strip()]


def _guard_cases(
    name: str,
    guard_names: list[str],
    state_inventory_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = [
        {
            "case_id": f"{name}:finite_update",
            "kind": "finite_update",
            "expectation": "reject or bound non-finite update before native promotion",
        },
        {
            "case_id": f"{name}:sparse_gradient_policy",
            "kind": "sparse_gradient",
            "expectation": str(state_inventory_artifact.get("sparse_gradient_policy", "")),
        },
        {
            "case_id": f"{name}:complex_tensor_policy",
            "kind": "complex_tensor",
            "expectation": str(state_inventory_artifact.get("complex_tensor_policy", "")),
        },
    ]
    cases.extend(
        {
            "case_id": f"{name}:{guard}",
            "kind": "formula_specific_guard",
            "expectation": guard,
        }
        for guard in guard_names
    )
    hparams = list(state_inventory_artifact.get("hparam_surface_keys", []))
    if hparams:
        cases.append(
            {
                "case_id": f"{name}:hparam_surface_boundaries",
                "kind": "hparam_surface",
                "expectation": "boundary coverage required for " + ",".join(str(item) for item in hparams),
            }
        )
    return cases


__all__ = [
    "build_custom_formula_quality_guard_artifact",
    "custom_formula_quality_guard_ready",
]
