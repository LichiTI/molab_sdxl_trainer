"""Smoke checks for V5-P83 optimizer batch-validation runner contract."""

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
    P82_CANARY_REPEAT_COUNT,
    P82_CANARY_STEP_COUNT,
)
from core.turbocore_v5_optimizer_batch_validation_runner_contract_p83 import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P83_SCOPE,
    REQUIRED_REVIEW_ACKS,
    build_v5_optimizer_batch_validation_runner_contract_p83,
)
from lulynx_trainer.turbocore_v5_optimizer_batch_validation_contract_p82_smoke import (  # noqa: E402
    _gate as _p82_gate,
    _p81_ready,
)


READY_DECISION = "optimizer_batch_validation_runner_contract_p83_recorded_default_off"
BLOCKED_DECISION = "optimizer_batch_validation_runner_contract_p83_blocked_default_off"
HOLD_DECISION = "optimizer_batch_validation_runner_contract_p83_hold_for_signed_review_default_off"
REJECTED_DECISION = "optimizer_batch_validation_runner_contract_p83_rejected_default_off"
_MISSING = object()


def run_smoke() -> dict[str, Any]:
    p82_ready = _p82_ready()
    ready = _gate(p82_ready)
    assert ready["ok"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["optimizer_batch_validation_runner_contract_ready"] is True, ready
    assert ready["canary_step_count"] == P82_CANARY_STEP_COUNT, ready
    assert ready["canary_repeat_count"] == P82_CANARY_REPEAT_COUNT, ready
    _assert_default_off(ready)

    missing_review = _gate(p82_ready, review=None)
    _assert_hold(missing_review, "signed_optimizer_batch_validation_runner_review_missing")
    rejected = _gate(p82_ready, review=_review(approve=False))
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == REJECTED_DECISION, rejected
    assert rejected["rollback_required"] is True, rejected
    _assert_default_off(rejected)

    p82_missing = _gate(None)
    _assert_blocked(p82_missing, "p82_optimizer_batch_validation_contract_not_ready")
    p82_not_ready = _gate({**p82_ready, "optimizer_batch_validation_contract_ready": False})
    _assert_blocked(p82_not_ready, "p82_optimizer_batch_validation_contract_not_ready")
    p82_unsafe = _gate({**p82_ready, "optimizer_kernel_executed": True})
    _assert_blocked(p82_unsafe, "optimizer_kernel_executed")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p83_optimizer_batch_validation_runner_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected": _summary(rejected),
        "p82_missing": _summary(p82_missing),
        "p82_not_ready": _summary(p82_not_ready),
        "p82_unsafe": _summary(p82_unsafe),
        **_evidence_cases(p82_ready),
        **_row_cases(p82_ready),
        **_review_cases(p82_ready),
        **_history_cases(p82_ready),
    }


def _evidence_cases(p82_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p82_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p82_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p82_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p82_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p82_ready, evidence={**_evidence(), "contract_only": False}),
        "not_runner_contract": _gate(p82_ready, evidence={**_evidence(), "runner_contract_only": False}),
        "not_manifest_only": _gate(p82_ready, evidence={**_evidence(), "manual_explicit_runner_manifest_only": False}),
        "missing_result_ingestion": _gate(p82_ready, evidence={**_evidence(), "result_ingestion_contract_required": False}),
        "step_count_mismatch": _gate(p82_ready, evidence={**_evidence(), "canary_step_count": 80}),
        "repeat_count_mismatch": _gate(p82_ready, evidence={**_evidence(), "canary_repeat_count": 1}),
        "default_on": _gate(p82_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_fields": _gate(p82_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "runner_command": _gate(p82_ready, evidence={**_evidence(), "batch_validation_runner_command": "run"}),
        "missing_section": _gate(p82_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "blocker": _gate(p82_ready, evidence={**_evidence(), "blocked_reasons": ["runner_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_runner_contract": ("runner_contract_only",),
        "not_manifest_only": ("manual_explicit_runner_manifest_only",),
        "missing_result_ingestion": ("result_ingestion_contract",),
        "step_count_mismatch": ("step_count_must_be_20",),
        "repeat_count_mismatch": ("repeat_count_must_be_5",),
        "default_on": ("default_off",),
        "request_fields": ("request_adapter",),
        "runner_command": ("batch_validation_runner_command",),
        "missing_section": ("required_section_missing",),
        "blocker": ("runner_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {f"evidence_{name}": _summary(report) for name, report in cases.items()}


def _row_cases(p82_ready: dict[str, Any]) -> dict[str, Any]:
    missing_lion = _gate(
        p82_ready,
        evidence={**_evidence(), "optimizer_run_matrix": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
    )
    not_ready = _gate(p82_ready, evidence={**_evidence(), "optimizer_run_matrix": _rows_with("lion", ready=False)})
    missing_source = _gate(p82_ready, evidence={**_evidence(), "optimizer_run_matrix": _rows_with("lion", source="")})
    bad_steps = _gate(p82_ready, evidence={**_evidence(), "optimizer_run_matrix": _rows_with("lion", canary_step_count=12)})
    runner_executed = _gate(p82_ready, evidence={**_evidence(), "optimizer_run_matrix": _rows_with("lion", runner_executed=True)})
    kernel_executed = _gate(
        p82_ready, evidence={**_evidence(), "optimizer_run_matrix": _rows_with("lion", **{"optimizer_kernel_executed": True})}
    )
    _assert_blocked(missing_lion, "optimizer_run_row_missing:lion")
    _assert_blocked(not_ready, "optimizer_run_row_ready_missing:lion")
    _assert_blocked(missing_source, "optimizer_run_row_source_missing:lion")
    _assert_blocked(bad_steps, "optimizer_run_row_step_count_mismatch:lion")
    _assert_blocked(runner_executed, "optimizer_run_row_unsafe_claim:lion:runner_executed")
    _assert_blocked(kernel_executed, "optimizer_run_row_unsafe_claim:lion:optimizer_kernel_executed")
    return {
        "row_missing_lion": _summary(missing_lion),
        "row_not_ready": _summary(not_ready),
        "row_missing_source": _summary(missing_source),
        "row_bad_steps": _summary(bad_steps),
        "row_runner_executed": _summary(runner_executed),
        "row_kernel_executed": _summary(kernel_executed),
    }


def _review_cases(p82_ready: dict[str, Any]) -> dict[str, Any]:
    scope = _gate(p82_ready, review={**_review(), "requested_scope": "wrong"})
    ack = _gate(p82_ready, review={**_review(), REQUIRED_REVIEW_ACKS[0]: False})
    unsafe = _gate(p82_ready, review={**_review(), "approve_batch_validation_runner_executed": True})
    _assert_blocked(scope, "review_scope_mismatch")
    _assert_blocked(ack, "review_ack_missing")
    _assert_blocked(unsafe, "unsafe_review_approval")
    return {"review_scope": _summary(scope), "review_ack": _summary(ack), "review_unsafe": _summary(unsafe)}


def _history_cases(p82_ready: dict[str, Any]) -> dict[str, Any]:
    failure = _gate(p82_ready, failure_history=[{"reason": "runner_gap", "status": "open"}])
    rollback = _gate(p82_ready, rollback_history=[{"reason": "rollback_gap", "active": True}])
    _assert_blocked(failure, "runner_gap")
    _assert_blocked(rollback, "rollback_gap")
    return {"failure_history": _summary(failure), "rollback_history": _summary(rollback)}


def _gate(
    p82_ready: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | object = _MISSING,
    review: dict[str, Any] | None | object = _MISSING,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_optimizer_batch_validation_runner_contract_p83(
        p82_optimizer_batch_validation_contract=p82_ready,
        optimizer_batch_validation_runner_evidence=_evidence() if evidence is _MISSING else evidence,
        optimizer_batch_validation_runner_review=_review() if review is _MISSING else review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p82_ready() -> dict[str, Any]:
    return _p82_gate(_p81_ready())


def _evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "optimizer_batch_validation_runner_contract_p83_v0",
        "ok": True,
        "runner_contract_ready": True,
        "runner_manifest_ready": True,
        "manual_explicit_runner_manifest_only": True,
        "result_ingestion_contract_required": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "runner_contract_only": True,
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
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        "source": "devtools/docs/native_training_performance_roadmap_v5.md#v5-p83",
        "sha256": "p83_runner_contract_digest",
        "available_sections": list(DEFAULT_REQUIRED_SECTIONS),
        "optimizer_run_matrix": [_row(kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p83_{kind}_run_row",
        "ready": True,
        "runner_plan_ready": True,
        "artifact_plan_ready": True,
        "result_ingestion_plan_ready": True,
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        "runner_executed": False,
        "optimizer_kernel_executed": False,
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
        "requested_scope": P83_SCOPE,
        "approve_optimizer_batch_validation_runner_contract": approve,
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
        "batch_validation_runner_executed",
        "batch_validation_artifacts_written",
        "batch_validation_results_ingested",
        "optimizer_kernel_probe_executed",
        "optimizer_kernel_executed",
        "training_launch_executed",
        "runtime_execution_executed",
        "native_dispatch_executed",
        "kernel_launch_executed",
        "training_step_executed",
        "request_fields_emitted",
    ):
        assert report.get(field) is False, (field, report)
    assert report.get("post_p83_request_fields") == {}, report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("optimizer_batch_validation_runner_contract_ready")),
        "review_signed": bool(report.get("optimizer_batch_validation_runner_review_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
