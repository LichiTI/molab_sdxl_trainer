"""Representative product-training canary package for fp32 simple optimizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.turbocore_simple_optimizer_family_batch_scorecard import (
    build_simple_optimizer_family_batch_scorecard,
)


TARGET_OPTIMIZERS = ("Lion", "SGDNesterov")
REQUIRED_STAGES = (
    "training_executor",
    "dispatch_runtime",
    "training_loop_canary",
    "e2e_no_regression",
)


def build_simple_optimizer_product_training_canary_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Package representative product-training evidence without enabling dispatch."""

    family = _as_dict(
        family_batch_report
        or build_simple_optimizer_family_batch_scorecard(workspace_root=workspace_root)
    )
    rows = [_case_row(name, family) for name in TARGET_OPTIMIZERS]
    validations = _validations(family, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_product_training_canary_scorecard_v0",
        "gate": "simple_formula_representative_product_training_canary",
        "ok": ready,
        "promotion_ready": False,
        "representative_product_training_canary_ready": ready,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_dispatch_ready": False,
        "product_native_ready_count": 0,
        "target_optimizer_types": list(TARGET_OPTIMIZERS),
        "rows": rows,
        "family_batch_summary": _as_dict(family.get("summary")),
        "validations": validations,
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "representative_product_training_canary_ready_count": len(rows) if ready else 0,
            "required_stage_count": len(REQUIRED_STAGES),
            "ready_required_stage_count": _ready_required_stage_count(family),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "simple_formula_owner_approval_missing",
                "simple_formula_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit owner/release approval before any fp32 simple optimizer product dispatch wiring"
            if ready
            else "fix fp32 simple optimizer representative product-training canary blockers"
        ),
        "notes": [
            "This package consumes family batch evidence for fp32 Lion and SGDNesterov.",
            "It is representative product-training evidence, not a product native-dispatch approval.",
            "Request, schema, UI, runtime dispatch, and training defaults remain unchanged.",
        ],
    }


def _case_row(optimizer_type: str, family: Mapping[str, Any]) -> dict[str, Any]:
    source = _family_row(optimizer_type, family)
    stage_ready = _as_dict(source.get("stage_ready"))
    ready = (
        source.get("batch_status") == "simple_formula_native_batch_canary_ready"
        and source.get("native_kernel_ready") is True
        and source.get("runtime_canary_ready") is True
        and source.get("training_loop_canary_ready") is True
        and all(stage_ready.get(stage) is True for stage in REQUIRED_STAGES)
    )
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "optimizer_family": "simple_formula",
        "canary_status": (
            "representative_product_training_canary_ready"
            if ready
            else "representative_product_training_canary_blocked"
        ),
        "native_route": str(source.get("native_route") or "rust_cuda_simple_formula_runtime_v0"),
        "native_kernel_ready": source.get("native_kernel_ready") is True,
        "runtime_canary_ready": source.get("runtime_canary_ready") is True,
        "training_loop_canary_ready": source.get("training_loop_canary_ready") is True,
        "required_stage_ready": {stage: stage_ready.get(stage) is True for stage in REQUIRED_STAGES},
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "simple_formula_explicit_owner_release_review",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_representative_product_training_canary_missing"],
    }


def _validations(family: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _validation(
            "family_batch_ready",
            family.get("simple_formula_native_batch_canary_ready") is True,
            "simple_formula_family_batch_canary_missing",
        ),
        _validation(
            "target_rows_ready",
            len(rows) == len(TARGET_OPTIMIZERS)
            and all(row.get("canary_status") == "representative_product_training_canary_ready" for row in rows),
            "simple_formula_product_training_canary_rows_incomplete",
        ),
        _validation(
            "required_stages_ready",
            _ready_required_stage_count(family) == len(REQUIRED_STAGES),
            "simple_formula_product_training_required_stage_missing",
        ),
        _validation(
            "product_boundaries_default_off",
            family.get("training_path_enabled") is False
            and family.get("native_dispatch_allowed") is False
            and family.get("runtime_dispatch_ready") is False
            and family.get("default_behavior_changed") is False,
            "simple_formula_product_training_canary_enabled_dispatch",
        ),
        _validation(
            "product_readiness_not_claimed",
            _as_dict(family.get("summary")).get("product_native_ready_count") == 0,
            "simple_formula_product_training_canary_claimed_product_ready",
        ),
    ]


def _family_row(optimizer_type: str, family: Mapping[str, Any]) -> dict[str, Any]:
    for row in family.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer_type:
            return dict(row)
    return {}


def _ready_required_stage_count(family: Mapping[str, Any]) -> int:
    stage_rows = {
        str(row.get("stage") or ""): row
        for row in family.get("stage_rows", [])
        if isinstance(row, Mapping)
    }
    return sum(1 for stage in REQUIRED_STAGES if _as_dict(stage_rows.get(stage)).get("ready") is True)


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "REQUIRED_STAGES",
    "TARGET_OPTIMIZERS",
    "build_simple_optimizer_product_training_canary_scorecard",
]
