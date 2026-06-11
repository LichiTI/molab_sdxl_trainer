"""Representative product-training canary package for AdamW variants.

This gate packages already-generated AdamW variant native canary evidence while
keeping the real product training path and native dispatch disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.turbocore_adamw_variant_family_batch_scorecard import (
    TARGET_OPTIMIZERS,
    build_adamw_variant_family_batch_scorecard,
)


REQUIRED_FAMILY_GATES = (
    "native_canary_stage_evidence",
    "e2e_shadow_matrix",
    "canary_rollout_policy",
    "dispatch_integration_review",
)


def build_adamw_variant_product_training_canary_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Package representative AdamW variant training evidence without dispatch."""

    del workspace_root
    family = _as_dict(
        family_batch_report
        or build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=True)
    )
    rows = [_case_row(str(optimizer.value), family) for optimizer in TARGET_OPTIMIZERS]
    validations = _validations(family, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_variant_product_training_canary_scorecard_v0",
        "gate": "adamw_variant_representative_product_training_canary",
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
        "target_optimizer_types": [str(optimizer.value) for optimizer in TARGET_OPTIMIZERS],
        "rows": rows,
        "family_batch_summary": _as_dict(family.get("summary")),
        "validations": validations,
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "representative_product_training_canary_ready_count": len(rows) if ready else 0,
            "required_family_gate_count": len(REQUIRED_FAMILY_GATES),
            "ready_required_family_gate_count": _ready_required_family_gate_count(family),
            "native_canary_stage_evidence_ready_count": int(
                _as_dict(family.get("summary")).get("native_canary_stage_evidence_ready_count", 0) or 0
            ),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_variant_owner_approval_missing",
                "adamw_variant_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record explicit owner/release approval before any AdamW variant product dispatch wiring"
            if ready
            else "fix AdamW variant representative product-training canary blockers"
        ),
        "notes": [
            "This package consumes AdamW variant family batch evidence for six built-in variants.",
            "It is representative product-training evidence, not product native-dispatch approval.",
            "Request, schema, UI, runtime dispatch, and training defaults remain unchanged.",
        ],
    }


def _case_row(optimizer_type: str, family: Mapping[str, Any]) -> dict[str, Any]:
    source = _family_row(optimizer_type, family)
    ready = (
        source.get("batch_status") == "native_canary_ready"
        and source.get("native_ready") is True
        and source.get("state_reference_ready") is True
        and source.get("native_canary_manifest_ready") is True
        and source.get("training_loop_canary_ready") is True
        and _boundary_default_off(source)
    )
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "optimizer_family": str(source.get("optimizer_family") or "adamw_variant"),
        "canary_status": (
            "representative_product_training_canary_ready"
            if ready
            else "representative_product_training_canary_blocked"
        ),
        "native_route": "adamw_variant_dedicated_kernel_required",
        "state_reference_ready": source.get("state_reference_ready") is True,
        "native_canary_manifest_ready": source.get("native_canary_manifest_ready") is True,
        "training_loop_canary_ready": source.get("training_loop_canary_ready") is True,
        "native_canary_stage_ready": source.get("native_ready") is True,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "adamw_variant_explicit_owner_release_review",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_representative_product_training_canary_missing"],
    }


def _validations(family: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _validation(
            "family_batch_ready",
            family.get("ok") is True
            and _as_dict(family.get("summary")).get("pending_count") == 0
            and _as_dict(family.get("summary")).get("native_ready_count") == len(TARGET_OPTIMIZERS),
            "adamw_variant_family_batch_canary_missing",
        ),
        _validation(
            "target_rows_ready",
            len(rows) == len(TARGET_OPTIMIZERS)
            and all(row.get("canary_status") == "representative_product_training_canary_ready" for row in rows),
            "adamw_variant_product_training_canary_rows_incomplete",
        ),
        _validation(
            "required_family_gates_ready",
            _ready_required_family_gate_count(family) == len(REQUIRED_FAMILY_GATES),
            "adamw_variant_product_training_required_gate_missing",
        ),
        _validation(
            "product_boundaries_default_off",
            family.get("training_path_enabled") is False
            and family.get("native_dispatch_allowed") is False
            and family.get("runtime_dispatch_ready") is False
            and family.get("default_behavior_changed") is False,
            "adamw_variant_product_training_canary_enabled_dispatch",
        ),
        _validation(
            "product_readiness_not_claimed",
            _as_dict(family.get("summary")).get("product_native_ready_count") == 0,
            "adamw_variant_product_training_canary_claimed_product_ready",
        ),
    ]


def _family_row(optimizer_type: str, family: Mapping[str, Any]) -> dict[str, Any]:
    for row in family.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer_type:
            return dict(row)
    return {}


def _ready_required_family_gate_count(family: Mapping[str, Any]) -> int:
    summary = _as_dict(family.get("summary"))
    ready = 0
    if int(summary.get("native_canary_stage_evidence_ready_count", 0) or 0) == len(TARGET_OPTIMIZERS):
        ready += 1
    if summary.get("e2e_shadow_matrix_ready") is True:
        ready += 1
    if summary.get("canary_rollout_policy_ready") is True:
        ready += 1
    if summary.get("dispatch_integration_review_ready") is True:
        ready += 1
    return ready


def _boundary_default_off(row: Mapping[str, Any]) -> bool:
    return (
        row.get("training_path_enabled") is False
        and row.get("native_dispatch_allowed") is False
        and row.get("default_behavior_changed") is False
        and row.get("product_native_dispatch_ready") is False
    )


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
    "REQUIRED_FAMILY_GATES",
    "build_adamw_variant_product_training_canary_scorecard",
]
