"""Smoke checks for V5-P36 post-archive rollout scope classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_archive_integrity_ledger import build_v5_archive_integrity_ledger  # noqa: E402
from core.turbocore_v5_archive_replay_verifier import build_v5_archive_replay_verifier  # noqa: E402
from core.turbocore_v5_post_archive_rollout_scope import build_v5_post_archive_rollout_scope  # noqa: E402
from lulynx_trainer.turbocore_v5_archive_integrity_ledger_smoke import _p33_signoff  # noqa: E402
from lulynx_trainer.turbocore_v5_owner_archive_signoff_smoke import _p32_package  # noqa: E402


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p35_ready = _p35_ready()
    ready_pending_review = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(),
    )
    assert ready_pending_review["ok"] is True, ready_pending_review
    assert ready_pending_review["decision"] == "post_archive_rollout_scope_classified_default_off", ready_pending_review
    assert ready_pending_review["scope_classification_ready"] is True, ready_pending_review
    assert ready_pending_review["controlled_rollout_policy_recorded"] is False, ready_pending_review
    assert ready_pending_review["rollout_authorization_allowed"] is False, ready_pending_review
    assert ready_pending_review["scope_classification"]["internal_canary_scope"]["state"] == "ready", ready_pending_review
    assert ready_pending_review["scope_classification"]["broader_real_route_coverage"]["state"] == "blocked", ready_pending_review
    assert ready_pending_review["scope_classification"]["controlled_rollout_policy"]["state"] == "pending_review", ready_pending_review
    _assert_default_off(ready_pending_review)

    fully_classified = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(broader=True, packaging=True),
        rollout_policy_review=_scope_review(),
    )
    assert fully_classified["ok"] is True, fully_classified
    assert fully_classified["controlled_rollout_policy_recorded"] is True, fully_classified
    assert fully_classified["broader_route_evidence_claim_ready"] is True, fully_classified
    assert fully_classified["broader_rollout_claim_ready"] is False, fully_classified
    assert fully_classified["rollout_authorization_allowed"] is False, fully_classified
    assert fully_classified["ui_exposure_allowed"] is False, fully_classified
    assert fully_classified["default_behavior_changed"] is False, fully_classified
    assert fully_classified["manual_review_required"] is True, fully_classified
    _assert_default_off(fully_classified)

    audit_only = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        roadmap_audit=_roadmap_audit(),
    )
    assert audit_only["ok"] is True, audit_only
    assert audit_only["scope_classification"]["internal_canary_scope"]["state"] == "ready", audit_only
    _assert_default_off(audit_only)

    missing_p35 = build_v5_post_archive_rollout_scope(readiness_report=_readiness_report())
    _assert_blocked(missing_p35, "p35", "missing")

    p35_blocked = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification={**p35_ready, "archived_ledger_matches_replay": False},
        readiness_report=_readiness_report(),
    )
    _assert_blocked(p35_blocked, "p35", "not_ready")

    unsafe_review = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(),
        rollout_policy_review={**_scope_review(), "approve_ui_exposure_allowed": True},
    )
    _assert_blocked(unsafe_review, "unsafe", "ui")
    assert unsafe_review["allowed_next_actions"] == [
        "repair_scope_classification_review_without_rollout_or_ui_approval"
    ], unsafe_review

    unsafe_rollout_review = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(),
        rollout_policy_review={**_scope_review(), "approve_default_rollout_allowed": True},
    )
    _assert_blocked(unsafe_rollout_review, "unsafe", "default")

    ui_claim = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(ready_for_ui=True),
    )
    _assert_blocked(ui_claim, "ui", "policy")
    assert ui_claim["allowed_next_actions"] == [
        "remove_premature_ui_readiness_claim_before_scope_classification"
    ], ui_claim

    bad_readiness = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(summary_ok=False, broader=True, packaging=True),
    )
    _assert_blocked(bad_readiness, "readiness", "not_ok")
    assert bad_readiness["scope_classification"]["broader_real_route_coverage"]["state"] == "blocked", bad_readiness
    assert bad_readiness["allowed_next_actions"] == [
        "repair_readiness_evidence_before_scope_classification"
    ], bad_readiness

    readiness_blockers = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(blockers=["native_dispatch_training_path_default_off"]),
    )
    _assert_blocked(readiness_blockers, "readiness", "primary")

    unlocked_readiness = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        readiness_report=_readiness_report(native_training_path_locked=False),
    )
    _assert_blocked(unlocked_readiness, "training_path", "locked")

    bad_audit = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=p35_ready,
        roadmap_audit=_roadmap_audit(ok=False, remaining_blockers=["coverage_identity_mismatch"]),
    )
    _assert_blocked(bad_audit, "roadmap", "not_ok")
    assert bad_audit["allowed_next_actions"] == [
        "repair_roadmap_audit_evidence_before_scope_classification"
    ], bad_audit

    p35_dirty = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification={**p35_ready, "default_behavior_changed": True},
        readiness_report=_readiness_report(),
    )
    _assert_blocked(p35_dirty, "p35", "not_ready")

    p35_ui_dirty = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification={**p35_ready, "ui_exposure_allowed": True},
        readiness_report=_readiness_report(),
    )
    _assert_blocked(p35_ui_dirty, "p35", "not_ready")

    p35_blockers = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification={**p35_ready, "blocked_reasons": ["tampered_blocker"]},
        readiness_report=_readiness_report(),
    )
    _assert_blocked(p35_blockers, "tampered")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p36_post_archive_rollout_scope_smoke",
        "ok": True,
        "ready_pending_review": _summary(ready_pending_review),
        "fully_classified": _summary(fully_classified),
        "audit_only": _summary(audit_only),
        "missing_p35": _summary(missing_p35),
        "p35_blocked": _summary(p35_blocked),
        "unsafe_review": _summary(unsafe_review),
        "unsafe_rollout_review": _summary(unsafe_rollout_review),
        "ui_claim": _summary(ui_claim),
        "bad_readiness": _summary(bad_readiness),
        "readiness_blockers": _summary(readiness_blockers),
        "unlocked_readiness": _summary(unlocked_readiness),
        "bad_audit": _summary(bad_audit),
        "p35_dirty": _summary(p35_dirty),
        "p35_ui_dirty": _summary(p35_ui_dirty),
        "p35_blockers": _summary(p35_blockers),
    }


def _p35_ready() -> dict[str, Any]:
    p32_ready = _p32_package()
    p33_ready = _p33_signoff(p32_ready)
    ledger = build_v5_archive_integrity_ledger(
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    return build_v5_archive_replay_verifier(
        archived_archive_ledger=ledger,
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )


def _readiness_report(
    *,
    broader: bool = False,
    packaging: bool = False,
    ready_for_ui: bool = False,
    summary_ok: bool = True,
    native_training_path_locked: bool = True,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "fixture_turbocore_readiness",
        "summary": {
            "ok": bool(summary_ok),
            "ready_for_ui": bool(ready_for_ui),
            "native_training_path_locked": bool(native_training_path_locked),
            "native_update_promotion_ready": True,
            "lora_promotion_ready": True,
            "native_stub_schema_complete": True,
            "workspace_data_pipeline_lifecycle_ok": True,
            "broader_real_route_coverage_ready": bool(broader),
            "packaging_observability_ready": bool(packaging),
            "native_update_promotion_blockers": list(blockers or []),
        },
    }


def _roadmap_audit(
    *,
    ok: bool = True,
    post_p5: bool = True,
    remaining_blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit": "fixture_native_training_performance_roadmap_v2_audit",
        "ok": bool(ok),
        "roadmap_completed": True,
        "required_promotions": {
            "lora_mixed_precision_dispatch": True,
            "optimizer_multitensor_update": True,
            "native_data_prefetch_batch_path": True,
            "runtime_native_router_canary": True,
            "e2e_training_performance_gate": True,
        },
        "post_p5_milestones_completed": bool(post_p5),
        "remaining_blockers": list(remaining_blockers or []),
        "summary": {
            "ready_gate_count": 5,
            "required_gate_count": 5,
            "post_p5_ready_gate_count": 14,
            "post_p5_milestone_gate_count": 14,
        },
    }


def _scope_review() -> dict[str, Any]:
    return {
        "reviewer": "v5_p36_fixture",
        "reviewed_at": "2026-06-02",
        "requested_scope": "post_archive_rollout_scope_classification",
        "approve_scope_classification_record": True,
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_ui_exposure_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_p35_archive_replay_ready": True,
        "acknowledge_default_off_boundary": True,
        "acknowledge_no_ui_exposure": True,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_no_auto_launch": True,
        "acknowledge_broader_route_coverage_required": True,
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["training_launch_allowed"] is False, report
    assert report["auto_launch_allowed"] is False, report
    assert report["runs_dispatched"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["post_scope_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == "post_archive_rollout_scope_blocked_default_off", report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "scope_classification_ready": bool(report.get("scope_classification_ready", False)),
        "controlled_rollout_policy_recorded": bool(report.get("controlled_rollout_policy_recorded", False)),
        "broader_route_evidence_claim_ready": bool(report.get("broader_route_evidence_claim_ready", False)),
        "broader_rollout_claim_ready": bool(report.get("broader_rollout_claim_ready", False)),
        "allowed_next_actions": [str(item) for item in report.get("allowed_next_actions", []) or []],
        "classification_blockers": _blocked_reasons(report),
        "rollout_blockers": [str(item) for item in report.get("rollout_blockers", []) or []],
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("classification_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
