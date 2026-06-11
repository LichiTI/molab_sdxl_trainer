"""Smoke checks for V5-P84 optimizer debug/failure-archive contract."""

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

from core.turbocore_v5_optimizer_debug_failure_archive_contract_p84 import (  # noqa: E402
    P84_SCOPE,
    REQUIRED_PUBLIC_LESSONS,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    build_v5_optimizer_debug_failure_archive_contract_p84,
)
from lulynx_trainer.turbocore_v5_optimizer_batch_validation_runner_contract_p83_smoke import (  # noqa: E402
    _gate as _p83_gate,
    _p82_ready,
)


READY_DECISION = "optimizer_debug_failure_archive_contract_p84_recorded_default_off"
BLOCKED_DECISION = "optimizer_debug_failure_archive_contract_p84_blocked_default_off"
HOLD_DECISION = "optimizer_debug_failure_archive_contract_p84_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_debug_failure_archive_contract_p84_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p83_ready = _p83_ready()
    ready = _gate(p83_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["debug_failure_archive_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p83_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_debug_failure_archive_review_missing")
    rejected = _gate(p83_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p83_missing": _gate(None),
        "p83_not_ready": _gate({**p83_ready, "optimizer_batch_validation_runner_contract_ready": False}),
        "p83_unsafe": _gate({**p83_ready, "batch_validation_runner_executed": True}),
        "missing_source": _gate(p83_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p83_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_lesson": _gate(
            p83_ready, evidence={**_evidence(), "public_debug_lessons": list(REQUIRED_PUBLIC_LESSONS[:-1])}
        ),
        "missing_section": _gate(p83_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p83_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "debug_tool_executed": _gate(p83_ready, evidence={**_evidence(), "debug_tool_executed": True}),
        "debug_command": _gate(p83_ready, evidence={**_evidence(), "debug_tool_command": "cuda-gdb"}),
        "not_min_repro": _gate(p83_ready, evidence={**_evidence(), "minimal_repro_plan_ready": False}),
        "review_scope": _gate(p83_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p83_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p83_ready, review={**_review(), "approve_debug_tool_executed": True}),
        "failure_history": _gate(p83_ready, failure_history=[{"reason": "debug_gap", "status": "open"}]),
        "rollback_history": _gate(p83_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p83_missing": "p83_optimizer_batch_validation_runner_contract_not_ready",
        "p83_not_ready": "p83_optimizer_batch_validation_runner_contract_not_ready",
        "p83_unsafe": "batch_validation_runner_executed",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_lesson": "public_debug_lesson_missing",
        "missing_section": "required_section_missing",
        "default_on": "default_off",
        "debug_tool_executed": "debug_tool_executed",
        "debug_command": "debug_tool_command",
        "not_min_repro": "minimal_repro_plan",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "debug_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p84_optimizer_debug_failure_archive_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p83_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_debug_failure_archive_contract_p84(
        p83_runner_contract=p83_ready,
        debug_failure_archive_evidence=_evidence() if evidence is _MISSING else evidence,
        debug_failure_archive_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p83_ready() -> dict[str, Any]:
    return _p83_gate(_p82_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_debug_failure_archive_contract_p84_v0",
        "ok": True,
        "debug_failure_archive_contract_ready": True,
        "public_debug_lessons_captured": True,
        "minimal_repro_plan_ready": True,
        "environment_capture_plan_ready": True,
        "cuda_debug_hooks_plan_ready": True,
        "failure_archive_schema_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "result_ingestion_contract_required": True,
        "default_off": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_off": True,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p84",
        "sha256": "p84_debug_failure_archive_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "public_debug_lessons": list(REQUIRED_PUBLIC_LESSONS),
    }


def _review(*, approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-03T00:00:00Z",
        "requested_scope": P84_SCOPE,
        "approve_debug_failure_archive_contract": approve,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
    }


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    out = dict(value)
    out.pop(key, None)
    return out


def _without_many(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    out = dict(value)
    for key in keys:
        out.pop(key, None)
    return out


def _assert_hold(report: dict[str, Any], fragment: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == HOLD_DECISION, report
    _assert_fragment(report, fragment)
    _assert_default_off(report)


def _assert_blocked(report: dict[str, Any], fragment: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_fragment(report, fragment)
    _assert_default_off(report)


def _assert_fragment(report: dict[str, Any], fragment: str) -> None:
    reasons = " ".join(str(item) for item in report.get("blocked_reasons", []))
    assert fragment in reasons, report


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "debug_tool_executed",
        "cuda_launch_blocking_run_executed",
        "nsight_profile_executed",
        "cuda_gdb_executed",
        "minimal_repro_executed",
        "failure_archive_written",
        "debug_result_ingested",
        "batch_validation_runner_executed",
        "optimizer_kernel_executed",
        "training_launch_executed",
        "request_fields_emitted",
    ):
        assert report.get(field) is False, (field, report)
    assert report.get("post_p84_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("debug_failure_archive_contract_ready")),
        "review_signed": bool(report.get("debug_failure_archive_review_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
