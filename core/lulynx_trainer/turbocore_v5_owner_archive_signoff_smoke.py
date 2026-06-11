"""Smoke checks for V5-P33 owner archive signoff."""

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

from core.turbocore_v5_owner_archive_signoff import build_v5_owner_archive_signoff  # noqa: E402
from core.turbocore_v5_p31_collector_replay_owner_package import (  # noqa: E402
    build_v5_p31_collector_replay_owner_package,
)
from lulynx_trainer.turbocore_v5_p31_collector_replay_owner_package_smoke import (  # noqa: E402
    _empty_history,
    _next_stage_review,
    _p25_decision,
    _p31_audit,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p32_ready = _p32_package()
    approved = build_v5_owner_archive_signoff(
        p32_owner_package=p32_ready,
        owner_archive_review=_archive_review(approve=True),
    )
    assert approved["ok"] is True, approved
    assert approved["decision_record_ready"] is True, approved
    assert approved["owner_archive_package_ready"] is True, approved
    assert approved["decision"] == "p32_final_owner_archive_signoff_ready_default_off", approved
    assert approved["training_launch_allowed"] is False, approved
    assert approved["runs_dispatched"] is False, approved
    _assert_default_off(approved)

    pending = build_v5_owner_archive_signoff(p32_owner_package=p32_ready)
    _assert_blocked(pending, "signed", "archive", "review")
    assert pending["ready_for_signed_owner_archive_review"] is True, pending
    assert pending["decision"] == "p32_final_owner_archive_hold_for_signed_signoff_default_off", pending

    rejected = build_v5_owner_archive_signoff(
        p32_owner_package=p32_ready,
        owner_archive_review=_archive_review(approve=False),
    )
    assert rejected["ok"] is True, rejected
    assert rejected["owner_archive_package_ready"] is False, rejected
    assert rejected["rejected_for_default_off_hold"] is True, rejected
    assert rejected["rollback_required"] is True, rejected
    assert rejected["decision"] == "p32_final_owner_archive_signoff_rejected_default_off", rejected
    _assert_default_off(rejected)

    p32_blocked = build_v5_owner_archive_signoff(
        p32_owner_package=_p32_package(p31_rollback=True),
        owner_archive_review=_archive_review(approve=True),
    )
    _assert_blocked(p32_blocked, "p32")

    missing_p32 = build_v5_owner_archive_signoff(owner_archive_review=_archive_review(approve=True))
    _assert_blocked(missing_p32, "p32", "missing")

    review_launch_violation = build_v5_owner_archive_signoff(
        p32_owner_package=p32_ready,
        owner_archive_review={**_archive_review(approve=True), "approve_training_launch_allowed": True},
    )
    _assert_blocked(review_launch_violation, "training", "launch")

    review_request_adapter_violation = build_v5_owner_archive_signoff(
        p32_owner_package=p32_ready,
        owner_archive_review={**_archive_review(approve=True), "approve_request_adapter_mapping_allowed": True},
    )
    _assert_blocked(review_request_adapter_violation, "request", "adapter")

    p32_default_violation = build_v5_owner_archive_signoff(
        p32_owner_package={**p32_ready, "default_rollout_allowed": True},
        owner_archive_review=_archive_review(approve=True),
    )
    _assert_blocked(p32_default_violation, "default")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p33_owner_archive_signoff_smoke",
        "ok": True,
        "approved": _summary(approved),
        "pending": _summary(pending),
        "rejected": _summary(rejected),
        "p32_blocked": _summary(p32_blocked),
        "missing_p32": _summary(missing_p32),
        "review_launch_violation": _summary(review_launch_violation),
        "review_request_adapter_violation": _summary(review_request_adapter_violation),
        "p32_default_violation": _summary(p32_default_violation),
    }


def _p32_package(*, p31_rollback: bool = False) -> dict[str, Any]:
    return build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=_p31_audit(rollback=p31_rollback),
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )


def _archive_review(*, approve: bool) -> dict[str, Any]:
    return {
        "reviewer": "v5_p33_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "p32_final_owner_archive_signoff",
        "approve_final_owner_archive": bool(approve),
        "approve_owner_archive_package": bool(approve),
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_p32_owner_package_ready": True,
        "acknowledge_p31_collector_replay_ready": True,
        "acknowledge_p29_owner_package_ready": True,
        "acknowledge_signed_p27_review": True,
        "acknowledge_p26_gate_ready": True,
        "acknowledge_p28_collector_bundle_ready": True,
        "acknowledge_default_rollout_disabled": True,
        "acknowledge_no_ui_or_request_adapter": True,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_no_training_launch": True,
        "acknowledge_default_and_auto_rollout_off": True,
        "acknowledge_manual_review_only": True,
        "acknowledge_p31_p28_p26_p27_p29_chain_archived": True,
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "decision_record_ready": bool(report.get("decision_record_ready", False)),
        "owner_archive_package_ready": bool(report.get("owner_archive_package_ready", False)),
        "ready_for_signed_owner_archive_review": bool(
            report.get("ready_for_signed_owner_archive_review", False)
        ),
        "rejected_for_default_off_hold": bool(report.get("rejected_for_default_off_hold", False)),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
