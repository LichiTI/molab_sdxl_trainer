"""Smoke checks for V5-P82 optimizer batch-validation contract."""

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

from core.turbocore_v5_optimizer_batch_validation_contract_p82 import (  # noqa: E402
    DEFAULT_OPTIMIZER_KINDS,
    DEFAULT_REQUIRED_SECTIONS,
    P82_CANARY_REPEAT_COUNT,
    P82_CANARY_STEP_COUNT,
    P82_SCOPE,
    REQUIRED_REVIEW_ACKS,
    build_v5_optimizer_batch_validation_contract_p82,
)
from lulynx_trainer.turbocore_v5_training_launch_execution_contract_p81_smoke import (  # noqa: E402
    _gate as _p81_gate,
    _p80_ready,
)


READY_DECISION = "optimizer_batch_validation_contract_p82_recorded_default_off"
BLOCKED_DECISION = "optimizer_batch_validation_contract_p82_blocked_default_off"
HOLD_DECISION = "optimizer_batch_validation_contract_p82_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_batch_validation_contract_p82_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p81_ready = _p81_ready()
    ready = _gate(p81_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_batch_validation_contract_ready"] is True, ready
    assert ready["canary_step_count"] == P82_CANARY_STEP_COUNT, ready
    assert ready["canary_repeat_count"] == P82_CANARY_REPEAT_COUNT, ready
    _assert_default_off(ready)

    missing_review = _gate(p81_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_batch_validation_review_missing")
    rejected = _gate(p81_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected
    _assert_default_off(rejected)

    p81_missing = _gate(None)
    _assert_blocked(p81_missing, "p81_training_launch_execution_contract_not_ready")
    p81_not_ready = _gate({**p81_ready, "training_launch_execution_contract_ready": False})
    _assert_blocked(p81_not_ready, "p81_training_launch_execution_contract_not_ready")
    p81_post_fields = _gate({**p81_ready, "post_p81_request_fields": {"bad": True}})
    _assert_blocked(p81_post_fields, "p81_training_launch_execution_contract_not_ready")
    p81_unsafe = _gate({**p81_ready, "training_launch_executed": True})
    _assert_blocked(p81_unsafe, "training_launch_executed")

    cases = {
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        "p81_missing": _summary(p81_missing),
        "p81_not_ready": _summary(p81_not_ready),
        "p81_post_fields": _summary(p81_post_fields),
        "p81_unsafe": _summary(p81_unsafe),
        **_evidence_cases(p81_ready),
        **_optimizer_row_cases(p81_ready),
        **_review_cases(p81_ready),
        **_history_cases(p81_ready),
    }
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p82_optimizer_batch_validation_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        **cases,
    }


def _evidence_cases(p81_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p81_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p81_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p81_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p81_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p81_ready, evidence={**_evidence(), "contract_only": False}),
        "not_batch_contract": _gate(
            p81_ready, evidence={**_evidence(), "optimizer_batch_validation_contract_only": False}
        ),
        "not_records_only": _gate(p81_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_manual_only": _gate(p81_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p81_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_contract": _gate(
            p81_ready, evidence={**_evidence(), "requires_later_optimizer_training_integration_contract": False}
        ),
        "step_count_mismatch": _gate(p81_ready, evidence={**_evidence(), "canary_step_count": 80}),
        "repeat_count_mismatch": _gate(p81_ready, evidence={**_evidence(), "canary_repeat_count": 1}),
        "default_on": _gate(p81_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p81_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "payload_present": _gate(p81_ready, evidence={**_evidence(), "optimizer_training_request_payload": {"bad": True}}),
        "missing_section": _gate(p81_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "evidence_blocker": _gate(p81_ready, evidence={**_evidence(), "blocked_reasons": ["optimizer_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_batch_contract": ("contract_only_scope",),
        "not_records_only": ("records_evidence_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_contract": ("later_integration_contract",),
        "step_count_mismatch": ("step_count_must_be_20",),
        "repeat_count_mismatch": ("repeat_count_must_be_5",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "payload_present": ("optimizer_training_request_payload",),
        "missing_section": ("required_section_missing",),
        "evidence_blocker": ("optimizer_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {f"evidence_{name}": _summary(report) for name, report in cases.items()}


def _optimizer_row_cases(p81_ready: dict[str, Any]) -> dict[str, Any]:
    missing_lion = _gate(
        p81_ready,
        evidence={**_evidence(), "optimizer_kernel_matrix": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
    )
    row_not_ready = _gate(p81_ready, evidence={**_evidence(), "optimizer_kernel_matrix": _rows_with("lion", ready=False)})
    row_missing_source = _gate(p81_ready, evidence={**_evidence(), "optimizer_kernel_matrix": _rows_with("lion", source="")})
    row_bad_steps = _gate(
        p81_ready, evidence={**_evidence(), "optimizer_kernel_matrix": _rows_with("lion", canary_step_count=12)}
    )
    row_native_enabled = _gate(
        p81_ready, evidence={**_evidence(), "optimizer_kernel_matrix": _rows_with("lion", native_dispatch_enabled=True)}
    )
    row_request_fields = _gate(
        p81_ready, evidence={**_evidence(), "optimizer_kernel_matrix": _rows_with("lion", request_fields_emitted=True)}
    )
    _assert_blocked(missing_lion, "optimizer_row_missing:lion")
    _assert_blocked(row_not_ready, "optimizer_row_ready_missing:lion")
    _assert_blocked(row_missing_source, "optimizer_row_source_missing:lion")
    _assert_blocked(row_bad_steps, "optimizer_row_step_count_mismatch:lion")
    _assert_blocked(row_native_enabled, "optimizer_row_unsafe_claim:lion:native_dispatch_enabled")
    _assert_blocked(row_request_fields, "optimizer_row_unsafe_claim:lion:request_fields_emitted")
    return {
        "row_missing_lion": _summary(missing_lion),
        "row_not_ready": _summary(row_not_ready),
        "row_missing_source": _summary(row_missing_source),
        "row_bad_steps": _summary(row_bad_steps),
        "row_native_enabled": _summary(row_native_enabled),
        "row_request_fields": _summary(row_request_fields),
    }


def _review_cases(p81_ready: dict[str, Any]) -> dict[str, Any]:
    scope = _gate(p81_ready, review={**_review(), "requested_scope": "wrong"})
    ack = _gate(p81_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False})
    unsafe = _gate(p81_ready, review={**_review(), "approve_training_launch_executed": True})
    _assert_blocked(scope, "review_scope_mismatch")
    _assert_blocked(ack, "review_ack_missing")
    _assert_blocked(unsafe, "unsafe_review_approval")
    return {"review_scope": _summary(scope), "review_ack": _summary(ack), "review_unsafe": _summary(unsafe)}


def _history_cases(p81_ready: dict[str, Any]) -> dict[str, Any]:
    failure = _gate(p81_ready, failure_history=[{"reason": "batch_gap", "status": "open"}])
    rollback = _gate(p81_ready, rollback_history=[{"reason": "rollback_gap", "active": True}])
    _assert_blocked(failure, "batch_gap")
    _assert_blocked(rollback, "rollback_gap")
    return {"failure_history": _summary(failure), "rollback_history": _summary(rollback)}


def _gate(
    p81_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_batch_validation_contract_p82(
        p81_training_launch_execution_contract=p81_ready,
        optimizer_batch_validation_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_batch_validation_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p81_ready() -> dict[str, Any]:
    return _p81_gate(_p80_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_batch_validation_contract_p82_v0",
        "ok": True,
        "optimizer_batch_validation_contract_ready": True,
        "batch_validation_framework_ready": True,
        "shared_safety_harness_ready": True,
        "per_optimizer_canary_matrix": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "optimizer_batch_validation_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_optimizer_training_integration_contract": True,
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
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p82",
        "sha256": "p82_batch_validation_contract_digest",
        "available_sections": list(DEFAULT_REQUIRED_SECTIONS),
        "optimizer_kernel_matrix": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p82_{kind}_row",
        "ready": True,
        "state_schema_boundary_ready": True,
        "parity_gate_ready": True,
        "benchmark_gate_ready": True,
        "safety_gate_ready": True,
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        "training_path_enabled": False,
        "native_dispatch_enabled": False,
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
        "requested_scope": P82_SCOPE,
        "approve_optimizer_batch_validation_contract": approve,
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


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == HOLD_DECISION, report
    _assert_fragments(report, *fragments)
    _assert_default_off(report)


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_fragments(report, *fragments)
    _assert_default_off(report)


def _assert_fragments(report: dict[str, Any], *fragments: str) -> None:
    reasons = " ".join(str(item) for item in report.get("blocked_reasons", []))
    assert all(fragment in reasons for fragment in fragments), report


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "training_launch_executed",
        "runtime_execution_executed",
        "native_dispatch_executed",
        "kernel_launch_executed",
        "training_step_executed",
        "request_fields_emitted",
        "optimizer_kernel_executed",
        "optimizer_training_integration_enabled",
        "optimizer_native_dispatch_enabled",
        "optimizer_request_fields_emitted",
        "optimizer_ui_exposed",
    ):
        assert report.get(field) is False, (field, report)
    assert report.get("post_p82_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_batch_validation_contract_ready")),
        "review_signed": bool(report.get("optimizer_batch_validation_review_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
