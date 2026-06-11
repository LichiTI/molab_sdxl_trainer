"""Smoke checks for V5-P87 optimizer integration review contract."""

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
from core.turbocore_v5_optimizer_integration_review_contract_p87 import (  # noqa: E402
    P87_SCOPE,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    UNSAFE_TRUE_FIELDS,
    build_v5_optimizer_integration_review_contract_p87,
)
from lulynx_trainer.turbocore_v5_optimizer_batch_keep_rollback_review_contract_p86_smoke import (  # noqa: E402
    _gate as _p86_gate,
    _p85_ready,
)


READY_DECISION = "optimizer_integration_review_contract_p87_recorded_default_off"
BLOCKED_DECISION = "optimizer_integration_review_contract_p87_blocked_default_off"
HOLD_DECISION = "optimizer_integration_review_contract_p87_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_integration_review_contract_p87_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p86_ready = _p86_ready()
    ready = _gate(p86_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_integration_review_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p86_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_integration_review_missing")
    rejected = _gate(p86_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p86_missing": _gate(None),
        "p86_not_ready": _gate({**p86_ready, "keep_rollback_review_contract_ready": False}),
        "p86_unsafe": _gate({**p86_ready, **{"optimizer_integration_enabled": True}}),
        "missing_source": _gate(p86_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p86_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_section": _gate(p86_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "invalid_intent": _gate(p86_ready, evidence={**_evidence(), "review_intent": "enable_it"}),
        "default_on": _gate(p86_ready, evidence={**_evidence(), **{"default_training_path_enabled": True}}),
        "payload": _gate(p86_ready, evidence={**_evidence(), "optimizer_kernel_launch_payload": {"bad": True}}),
        "missing_lion": _gate(
            p86_ready,
            evidence={**_evidence(), "optimizer_integration_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
        ),
        "row_not_ready": _gate(p86_ready, evidence={**_evidence(), "optimizer_integration_rows": _rows_with("lion", ready=False)}),
        "row_native_enabled": _gate(
            p86_ready, evidence={**_evidence(), "optimizer_integration_rows": _rows_with("lion", **{"native_dispatch_enabled": True})}
        ),
        "review_scope": _gate(p86_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p86_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p86_ready, review={**_review(), "approve_optimizer_integration_enabled": True}),
        "failure_history": _gate(p86_ready, failure_history=[{"reason": "integration_gap", "status": "open"}]),
        "rollback_history": _gate(p86_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p86_missing": "p86_optimizer_batch_keep_rollback_review_contract_not_ready",
        "p86_not_ready": "p86_optimizer_batch_keep_rollback_review_contract_not_ready",
        "p86_unsafe": "optimizer_integration_enabled",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "invalid_intent": "review_intent_invalid",
        "default_on": "default_off",
        "payload": "optimizer_kernel_launch_payload",
        "missing_lion": "optimizer_integration_row_missing:lion",
        "row_not_ready": "optimizer_integration_row_ready_missing:lion",
        "row_native_enabled": "optimizer_integration_row_unsafe_claim:lion:native_dispatch_enabled",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "integration_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p87_optimizer_integration_review_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p86_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_integration_review_contract_p87(
        p86_keep_rollback_review_contract=p86_ready,
        optimizer_integration_review_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_integration_signed_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p86_ready() -> dict[str, Any]:
    return _p86_gate(_p85_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_integration_review_contract_p87_v0",
        "ok": True,
        "optimizer_integration_review_package_ready": True,
        "integration_policy_ready": True,
        "later_optimizer_runtime_enablement_contract_required": True,
        "review_intent": "integration_candidate",
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
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p87",
        "sha256": "p87_optimizer_integration_review_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "optimizer_integration_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p87_{kind}_integration_row",
        "ready": True,
        "integration_review_ready": True,
        "request_adapter_boundary_ready": True,
        "runtime_dispatch_boundary_ready": True,
        "native_dispatch_boundary_ready": True,
        "training_path_boundary_ready": True,
        "rollback_policy_ready": True,
        "later_runtime_enablement_contract_required": True,
        "integration_enabled": False,
        "runtime_dispatch_enabled": False,
        "native_dispatch_enabled": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "ui_exposed": False,
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
        "requested_scope": P87_SCOPE,
        "approve_optimizer_integration_review_contract": approve,
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
    assert report.get("post_p87_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_integration_review_contract_ready")),
        "review_signed": bool(report.get("optimizer_integration_review_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
