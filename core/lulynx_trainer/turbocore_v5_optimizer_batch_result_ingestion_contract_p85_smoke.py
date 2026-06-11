"""Smoke checks for V5-P85 optimizer batch result-ingestion contract."""

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

from core.turbocore_v5_optimizer_batch_result_ingestion_contract_p85 import (  # noqa: E402
    P85_SCOPE,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    build_v5_optimizer_batch_result_ingestion_contract_p85,
)
from core.turbocore_v5_optimizer_batch_validation_contract_p82 import (  # noqa: E402
    DEFAULT_OPTIMIZER_KINDS,
    P82_CANARY_REPEAT_COUNT,
    P82_CANARY_STEP_COUNT,
)
from lulynx_trainer.turbocore_v5_optimizer_debug_failure_archive_contract_p84_smoke import (  # noqa: E402
    _gate as _p84_gate,
    _p83_ready,
)


READY_DECISION = "optimizer_batch_result_ingestion_contract_p85_recorded_default_off"
BLOCKED_DECISION = "optimizer_batch_result_ingestion_contract_p85_blocked_default_off"
HOLD_DECISION = "optimizer_batch_result_ingestion_contract_p85_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_batch_result_ingestion_contract_p85_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p84_ready = _p84_ready()
    ready = _gate(p84_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_batch_result_ingestion_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p84_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_batch_result_review_missing")
    rejected = _gate(p84_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p84_missing": _gate(None),
        "p84_not_ready": _gate({**p84_ready, "debug_failure_archive_contract_ready": False}),
        "p84_unsafe": _gate({**p84_ready, "debug_tool_executed": True}),
        "missing_source": _gate(p84_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p84_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_section": _gate(p84_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p84_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_fields": _gate(p84_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "step_mismatch": _gate(p84_ready, evidence={**_evidence(), "canary_step_count": 80}),
        "missing_lion": _gate(
            p84_ready,
            evidence={**_evidence(), "optimizer_result_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
        ),
        "row_not_ready": _gate(p84_ready, evidence={**_evidence(), "optimizer_result_rows": _rows_with("lion", ready=False)}),
        "row_no_digest": _gate(p84_ready, evidence={**_evidence(), "optimizer_result_rows": _rows_with("lion", sha256="")}),
        "row_accepted": _gate(
            p84_ready, evidence={**_evidence(), "optimizer_result_rows": _rows_with("lion", **{"optimizer_result_accepted": True})}
        ),
        "payload": _gate(p84_ready, evidence={**_evidence(), "optimizer_result_application_payload": {"bad": True}}),
        "review_scope": _gate(p84_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p84_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p84_ready, review={**_review(), "approve_result_ingestion_executed": True}),
        "failure_history": _gate(p84_ready, failure_history=[{"reason": "result_gap", "status": "open"}]),
        "rollback_history": _gate(p84_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p84_missing": "p84_optimizer_debug_failure_archive_contract_not_ready",
        "p84_not_ready": "p84_optimizer_debug_failure_archive_contract_not_ready",
        "p84_unsafe": "debug_tool_executed",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "default_on": "default_off",
        "request_fields": "request_adapter",
        "step_mismatch": "step_count_must_be_20",
        "missing_lion": "optimizer_result_row_missing:lion",
        "row_not_ready": "optimizer_result_row_ready_missing:lion",
        "row_no_digest": "optimizer_result_row_digest_missing:lion",
        "row_accepted": "optimizer_result_row_unsafe_claim:lion:optimizer_result_accepted",
        "payload": "optimizer_result_application_payload",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "result_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p85_optimizer_batch_result_ingestion_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p84_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_batch_result_ingestion_contract_p85(
        p84_debug_failure_archive_contract=p84_ready,
        optimizer_batch_result_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_batch_result_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p84_ready() -> dict[str, Any]:
    return _p84_gate(_p83_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_batch_result_ingestion_contract_p85_v0",
        "ok": True,
        "result_ingestion_contract_ready": True,
        "result_bundle_shape_ready": True,
        "artifact_digest_ledger_ready": True,
        "keep_or_rollback_policy_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "later_keep_rollback_review_required": True,
        "default_off": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_off": True,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p85",
        "sha256": "p85_result_ingestion_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "optimizer_result_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p85_{kind}_result_row",
        "sha256": f"p85_{kind}_digest",
        "ready": True,
        "parity_summary_ready": True,
        "benchmark_summary_ready": True,
        "state_roundtrip_summary_ready": True,
        "debug_archive_reference_ready": True,
        "result_ingestion_executed": False,
        "optimizer_result_accepted": False,
        "request_fields_emitted": False,
        "blocked_reasons": [],
    }
    row.update(overrides)
    return row


def _rows_with(target: str, **overrides: Any) -> list[dict[str, Any]]:
    return [_row(kind, **(overrides if kind == target else {})) for kind in DEFAULT_OPTIMIZER_KINDS]


def _review(*, approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-03T00:00:00Z",
        "requested_scope": P85_SCOPE,
        "approve_optimizer_batch_result_ingestion_contract": approve,
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
        "result_ingestion_executed",
        "result_bundle_applied",
        "optimizer_result_accepted",
        "optimizer_result_promoted",
        "artifact_digest_ledger_applied",
        "keep_decision_applied",
        "rollback_decision_applied",
        "training_launch_executed",
        "request_fields_emitted",
    ):
        assert report.get(field) is False, (field, report)
    assert report.get("post_p85_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_batch_result_ingestion_contract_ready")),
        "review_signed": bool(report.get("optimizer_batch_result_review_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
