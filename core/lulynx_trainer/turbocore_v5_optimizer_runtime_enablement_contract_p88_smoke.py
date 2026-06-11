"""Smoke checks for V5-P88 optimizer runtime enablement contract."""

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
from core.turbocore_v5_optimizer_runtime_enablement_contract_p88 import (  # noqa: E402
    P88_SCOPE,
    REQUIRED_REVIEW_ACKS,
    REQUIRED_SECTIONS,
    UNSAFE_TRUE_FIELDS,
    build_v5_optimizer_runtime_enablement_contract_p88,
)
from lulynx_trainer.turbocore_v5_optimizer_integration_review_contract_p87_smoke import (  # noqa: E402
    _gate as _p87_gate,
    _p86_ready,
)


READY_DECISION = "optimizer_runtime_enablement_contract_p88_recorded_default_off"
BLOCKED_DECISION = "optimizer_runtime_enablement_contract_p88_blocked_default_off"
HOLD_DECISION = "optimizer_runtime_enablement_contract_p88_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_runtime_enablement_contract_p88_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p87_ready = _p87_ready()
    ready = _gate(p87_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_runtime_enablement_contract_ready"] is True, ready
    _assert_default_off(ready)

    missing_review = _gate(p87_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_runtime_enablement_review_missing")
    rejected = _gate(p87_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected

    cases = {
        "p87_missing": _gate(None),
        "p87_not_ready": _gate({**p87_ready, "optimizer_integration_review_contract_ready": False}),
        "p87_unsafe": _gate({**p87_ready, **{"runtime_dispatch_enabled": True}}),
        "missing_source": _gate(p87_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p87_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "missing_section": _gate(p87_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "invalid_intent": _gate(p87_ready, evidence={**_evidence(), "review_intent": "activate_it"}),
        "default_on": _gate(p87_ready, evidence={**_evidence(), **{"default_training_path_enabled": True}}),
        "payload": _gate(p87_ready, evidence={**_evidence(), "kernel_launch_payload": {"bad": True}}),
        "missing_lion": _gate(
            p87_ready,
            evidence={**_evidence(), "optimizer_runtime_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
        ),
        "row_not_ready": _gate(p87_ready, evidence={**_evidence(), "optimizer_runtime_rows": _rows_with("lion", ready=False)}),
        "row_kernel_enabled": _gate(
            p87_ready, evidence={**_evidence(), "optimizer_runtime_rows": _rows_with("lion", **{"kernel_launch_enabled": True})}
        ),
        "review_scope": _gate(p87_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_ack": _gate(p87_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False}),
        "review_unsafe": _gate(p87_ready, review={**_review(), "approve_runtime_adapter_enabled": True}),
        "failure_history": _gate(p87_ready, failure_history=[{"reason": "runtime_gap", "status": "open"}]),
        "rollback_history": _gate(p87_ready, rollback_history=[{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "p87_missing": "p87_optimizer_integration_review_contract_not_ready",
        "p87_not_ready": "p87_optimizer_integration_review_contract_not_ready",
        "p87_unsafe": "runtime_dispatch_enabled",
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "invalid_intent": "review_intent_invalid",
        "default_on": "default_off",
        "payload": "kernel_launch_payload",
        "missing_lion": "optimizer_runtime_row_missing:lion",
        "row_not_ready": "optimizer_runtime_row_ready_missing:lion",
        "row_kernel_enabled": "optimizer_runtime_row_unsafe_claim:lion:kernel_launch_enabled",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": "runtime_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(report, fragments[name])
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p88_optimizer_runtime_enablement_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        **{name: _summary(report) for name, report in cases.items()},
    }


def _gate(
    p87_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_runtime_enablement_contract_p88(
        p87_optimizer_integration_review_contract=p87_ready,
        optimizer_runtime_enablement_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_runtime_enablement_signed_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p87_ready() -> dict[str, Any]:
    return _p87_gate(_p86_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_runtime_enablement_contract_p88_v0",
        "ok": True,
        "optimizer_runtime_enablement_package_ready": True,
        "runtime_enablement_policy_ready": True,
        "later_optimizer_runtime_activation_contract_required": True,
        "review_intent": "runtime_enablement_candidate",
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
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p88",
        "sha256": "p88_optimizer_runtime_enablement_digest",
        "available_sections": list(REQUIRED_SECTIONS),
        "optimizer_runtime_rows": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p88_{kind}_runtime_row",
        "ready": True,
        "runtime_enablement_review_ready": True,
        "runtime_adapter_boundary_ready": True,
        "runtime_dispatch_boundary_ready": True,
        "native_dispatch_boundary_ready": True,
        "kernel_launch_boundary_ready": True,
        "rollback_policy_ready": True,
        "later_runtime_activation_contract_required": True,
        "runtime_adapter_enabled": False,
        "runtime_dispatch_enabled": False,
        "native_dispatch_enabled": False,
        "kernel_launch_enabled": False,
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
        "requested_scope": P88_SCOPE,
        "approve_optimizer_runtime_enablement_contract": approve,
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
    assert report.get("post_p88_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_runtime_enablement_contract_ready")),
        "review_signed": bool(report.get("optimizer_runtime_enablement_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
