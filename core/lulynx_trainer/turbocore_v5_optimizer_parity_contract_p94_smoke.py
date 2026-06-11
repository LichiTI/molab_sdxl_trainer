"""Smoke checks for V5-P94 optimizer parity contract."""

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
from core.turbocore_v5_optimizer_parity_contract_p94 import (  # noqa: E402
    P94_SCOPE,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    UNSAFE_TRUE_FIELDS,
    build_v5_optimizer_parity_contract_p94,
)
from lulynx_trainer.turbocore_v5_optimizer_tensor_transfer_contract_p93_smoke import (  # noqa: E402
    _gate as _p93_gate,
    _p92_ready,
)


READY_DECISION = "optimizer_parity_contract_p94_recorded_default_off"
BLOCKED_DECISION = "optimizer_parity_contract_p94_blocked_default_off"
HOLD_DECISION = "optimizer_parity_contract_p94_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_parity_contract_p94_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p93_ready = _p93_ready()
    ready = _gate(p93_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_parity_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p93_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_parity_review_missing")
    rejected = _gate(p93_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p93_missing": _gate(None),
        "p93_not_ready": _gate({**p93_ready, "optimizer_tensor_transfer_contract_ready": False}),
        "p93_unsafe": _gate({**p93_ready, **{"parity_execution_executed": True}}),
        "missing_source": _gate(p93_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p93_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_section": _gate(p93_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "invalid_intent": _gate(p93_ready, evidence={**_evidence(), "review_intent": "launch_it"}),
        "default_on": _gate(p93_ready, evidence={**_evidence(), **{"default_training_path_enabled": True}}),
        "payload": _gate(p93_ready, evidence={**_evidence(), "training_step_payload": {"bad": True}}),
        "missing_lion": _gate(
            p93_ready,
            evidence={**_evidence(), "optimizer_parity_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
        ),
        "row_not_ready": _gate(p93_ready, evidence={**_evidence(), "optimizer_parity_rows": _rows_with("lion", ready=False)}),
        "row_transfer_executed": _gate(
            p93_ready, evidence={**_evidence(), "optimizer_parity_rows": _rows_with("lion", **{"tensor_transfer_executed": True})}
        ),
        "row_kernel_executed": _gate(
            p93_ready, evidence={**_evidence(), "optimizer_parity_rows": _rows_with("lion", **{"kernel_launch_executed": True})}
        ),
        "row_parity_executed": _gate(
            p93_ready, evidence={**_evidence(), "optimizer_parity_rows": _rows_with("lion", **{"parity_execution_executed": True})}
        ),
        "review_scope": _gate(p93_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p93_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p93_ready, review={**_review(), "approve_parity_execution_executed": True}),
        "failure_history": _gate(p93_ready, failure_history=[{"reason": "parity_gap", "status": "open"}]),
        "rollback_history": _gate(p93_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p93_missing": "p93_optimizer_tensor_transfer_contract_not_ready",
        "p93_not_ready": "p93_optimizer_tensor_transfer_contract_not_ready",
        "p93_unsafe": "parity_execution_executed",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "invalid_intent": "review_intent_invalid",
        "default_on": "default_off",
        "payload": "training_step_payload",
        "missing_lion": "optimizer_parity_row_missing:lion",
        "row_not_ready": "optimizer_parity_row_ready_missing:lion",
        "row_transfer_executed": "optimizer_parity_row_unsafe_claim:lion:tensor_transfer_executed",
        "row_kernel_executed": "optimizer_parity_row_unsafe_claim:lion:kernel_launch_executed",
        "row_parity_executed": "optimizer_parity_row_unsafe_claim:lion:parity_execution_executed",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "parity_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p94_optimizer_parity_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p93_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_parity_contract_p94(
        p93_optimizer_tensor_transfer_contract=p93_ready,
        optimizer_parity_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_parity_signed_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p93_ready() -> dict[str, Any]:
    return _p93_gate(_p92_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_parity_contract_p94_v0",
        "ok": True,
        "optimizer_parity_package_ready": True,
        "parity_policy_ready": True,
        "later_optimizer_training_step_contract_required": True,
        "review_intent": "parity_candidate",
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
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p94",
        "sha256": "p94_optimizer_parity_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "optimizer_parity_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p94_{kind}_parity_row",
        "ready": True,
        "parity_review_ready": True,
        "kernel_launch_boundary_ready": True,
        "tensor_transfer_boundary_ready": True,
        "parity_boundary_ready": True,
        "training_step_boundary_ready": True,
        "rollback_policy_ready": True,
        "later_training_step_contract_required": True,
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
        "requested_scope": P94_SCOPE,
        "approve_optimizer_parity_contract": approve,
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
    assert report.get("post_p94_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_parity_contract_ready")),
        "review_signed": bool(report.get("optimizer_parity_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
