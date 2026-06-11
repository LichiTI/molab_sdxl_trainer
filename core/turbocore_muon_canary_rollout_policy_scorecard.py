"""Default-off canary rollout policy for built-in Muon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_muon_e2e_shadow_matrix_scorecard import (
    build_muon_e2e_shadow_matrix_scorecard,
)


POLICY_KIND = "muon_model_shape_aware_canary_rollout_policy_v0"
FALLBACK_BACKEND = "python_muon_optimizer"
OPTIMIZER_FAMILY = "built_in_muon_model_shape_aware"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_muon_canary_rollout_policy_scorecard(
    *,
    shadow_matrix_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Build a report-only Muon rollout policy without enabling dispatch."""

    shadow = _as_dict(shadow_matrix_report or build_muon_e2e_shadow_matrix_scorecard())
    rows = [_policy_row(row) for row in shadow.get("rows", []) if isinstance(row, Mapping)]
    policy = _policy(rows)
    validations = _validations(shadow, rows, policy)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_canary_rollout_policy_scorecard_v0",
        "gate": "muon_model_shape_aware_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "canary_rollout_policy_ready": ready,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "policy_kind": POLICY_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "optimizer_family": OPTIMIZER_FAMILY,
        "policy": policy,
        "rows": rows,
        "shadow_matrix_summary": _as_dict(shadow.get("summary")),
        "validations": validations,
        "summary": {
            "optimizer_count": len(rows),
            "canary_rollout_policy_ready": ready,
            "canary_rollout_policy_ready_count": sum(
                1 for row in rows if row.get("canary_rollout_policy_ready") is True
            ),
            "manual_review_required": True,
            "canary_auto_enabled": False,
            "canary_enabled_by_default": False,
            "explicit_opt_in_required": True,
            "max_canary_fraction_default": 0.0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
            "fallback_backend_authoritative": True,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "muon_runtime_dispatch_disabled_pending_review",
                "muon_real_dispatch_wiring_missing",
                "muon_owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit owner/release review before wiring Muon canary dispatch"
            if ready
            else "fix Muon canary rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "Python Muon remains authoritative until explicit owner/release review.",
            "Actual product dispatch wiring remains a separate roadmap gate.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_type = str(row.get("optimizer_type") or "Muon")
    ready = row.get("e2e_shadow_matrix_ready") is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": str(row.get("family") or OPTIMIZER_FAMILY),
        "canary_rollout_policy_ready": ready,
        "e2e_shadow_matrix_ready": ready,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "canary_enabled_by_default": False,
        "explicit_opt_in_required": True,
        "max_canary_fraction_default": 0.0,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "next_gate": "muon_owner_release_hold_default_off",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_muon_e2e_shadow_matrix_missing"],
    }


def _policy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ready_names = [
        str(row.get("optimizer_type"))
        for row in rows
        if row.get("canary_rollout_policy_ready") is True
    ]
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "optimizer_types": ready_names,
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "muon_e2e_shadow_matrix",
            "fallback_backend_authoritative",
            "native_shadow_does_not_mutate_authority",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
            "manual_review_approval_recorded",
        ],
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_shape_partition_mismatch": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
    }


def _validations(
    shadow: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rollback = _as_dict(policy.get("rollback_policy"))
    return [
        _validation(
            "e2e_shadow_matrix_ready",
            shadow.get("e2e_shadow_matrix_ready") is True
            and all(row.get("e2e_shadow_matrix_ready") is True for row in rows),
            "muon_e2e_shadow_matrix_missing",
        ),
        _validation(
            "policy_default_off",
            policy.get("canary_enabled_by_default") is False
            and policy.get("canary_auto_enabled") is False
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "muon_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_and_manual_review_required",
            policy.get("explicit_opt_in_required") is True and policy.get("manual_review_required") is True,
            "muon_policy_missing_manual_review",
        ),
        _validation(
            "fallback_rollback_ready",
            rollback.get("fallback_authoritative") is True
            and rollback.get("rollback_on_nonfinite") is True
            and rollback.get("rollback_on_parity_failure") is True
            and rollback.get("rollback_on_shape_partition_mismatch") is True,
            "muon_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            policy.get("runtime_dispatch_ready") is False
            and policy.get("native_dispatch_allowed") is False
            and policy.get("training_path_enabled") is False
            and shadow.get("runtime_dispatch_ready") is False
            and shadow.get("native_dispatch_allowed") is False
            and shadow.get("training_path_enabled") is False,
            "muon_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            policy.get("default_behavior_changed") is False
            and shadow.get("default_behavior_changed") is False,
            "muon_policy_changed_default_behavior",
        ),
    ]


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_canary_rollout_policy_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


__all__ = ["POLICY_KIND", "build_muon_canary_rollout_policy_scorecard"]
