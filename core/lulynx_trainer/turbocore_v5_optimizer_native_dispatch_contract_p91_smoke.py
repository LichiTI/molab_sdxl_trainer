"""Smoke checks for V5-P91 optimizer native dispatch contract."""

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

from core.turbocore_v5_optimizer_batch_validation_contract_p82 import DEFAULT_OPTIMIZER_KINDS  # noqa: E402
from core.turbocore_v5_optimizer_native_dispatch_contract_p91 import (  # noqa: E402
    P91_SCOPE,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    UNSAFE_TRUE_FIELDS,
    build_v5_optimizer_native_dispatch_contract_p91,
)
from lulynx_trainer.turbocore_v5_optimizer_dispatch_contract_p90_smoke import (  # noqa: E402
    _gate as _p90_gate,
    _p89_ready,
)


READY_DECISION = "optimizer_native_dispatch_contract_p91_recorded_default_off"
BLOCKED_DECISION = "optimizer_native_dispatch_contract_p91_blocked_default_off"
HOLD_DECISION = "optimizer_native_dispatch_contract_p91_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_native_dispatch_contract_p91_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p90_ready = _p90_ready()
    ready = _gate(p90_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_native_dispatch_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p90_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_native_dispatch_review_missing")
    rejected = _gate(p90_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p90_missing": _gate(None),
        "p90_not_ready": _gate({**p90_ready, "optimizer_dispatch_contract_ready": False}),
        "p90_unsafe": _gate({**p90_ready, **{"native_dispatch_executed": True}}),
        "missing_source": _gate(p90_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p90_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_section": _gate(p90_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "invalid_intent": _gate(p90_ready, evidence={**_evidence(), "review_intent": "kernel_it"}),
        "default_on": _gate(p90_ready, evidence={**_evidence(), **{"default_training_path_enabled": True}}),
        "payload": _gate(p90_ready, evidence={**_evidence(), "kernel_launch_payload": {"bad": True}}),
        "missing_lion": _gate(
            p90_ready,
            evidence={**_evidence(), "optimizer_native_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
        ),
        "row_not_ready": _gate(p90_ready, evidence={**_evidence(), "optimizer_native_rows": _rows_with("lion", ready=False)}),
        "row_parity_executed": _gate(
            p90_ready, evidence={**_evidence(), "optimizer_native_rows": _rows_with("lion", **{"parity_execution_executed": True})}
        ),
        "review_scope": _gate(p90_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p90_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p90_ready, review={**_review(), "approve_kernel_launch_executed": True}),
        "failure_history": _gate(p90_ready, failure_history=[{"reason": "native_gap", "status": "open"}]),
        "rollback_history": _gate(p90_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p90_missing": "p90_optimizer_dispatch_contract_not_ready",
        "p90_not_ready": "p90_optimizer_dispatch_contract_not_ready",
        "p90_unsafe": "native_dispatch_executed",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "invalid_intent": "review_intent_invalid",
        "default_on": "default_off",
        "payload": "kernel_launch_payload",
        "missing_lion": "optimizer_native_dispatch_row_missing:lion",
        "row_not_ready": "optimizer_native_dispatch_row_ready_missing:lion",
        "row_parity_executed": "optimizer_native_dispatch_row_unsafe_claim:lion:parity_execution_executed",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "native_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p91_optimizer_native_dispatch_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p90_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_native_dispatch_contract_p91(
        p90_optimizer_dispatch_contract=p90_ready,
        optimizer_native_dispatch_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_native_dispatch_signed_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p90_ready() -> dict[str, Any]:
    return _p90_gate(_p89_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_native_dispatch_contract_p91_v0",
        "ok": True,
        "optimizer_native_dispatch_package_ready": True,
        "native_dispatch_policy_ready": True,
        "later_optimizer_kernel_launch_contract_required": True,
        "review_intent": "native_dispatch_candidate",
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_off": True,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p91",
        "sha256": "p91_optimizer_native_dispatch_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "optimizer_native_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p91_{kind}_native_row",
        "ready": True,
        "native_dispatch_review_ready": True,
        "native_dispatch_boundary_ready": True,
        "kernel_launch_boundary_ready": True,
        "tensor_transfer_boundary_ready": True,
        "parity_boundary_ready": True,
        "rollback_policy_ready": True,
        "later_kernel_launch_contract_required": True,
        "native_dispatch_executed": False,
        "kernel_launch_executed": False,
        "tensor_transfer_executed": False,
        "parity_execution_executed": False,
        "training_step_executed": False,
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
        "requested_scope": P91_SCOPE,
        "approve_optimizer_native_dispatch_contract": approve,
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
    for field in UNSAFE_TRUE_FIELDS:
        assert report.get(field) is False, (field, report)
    assert report.get("post_p91_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_native_dispatch_contract_ready")),
        "review_signed": bool(report.get("optimizer_native_dispatch_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
