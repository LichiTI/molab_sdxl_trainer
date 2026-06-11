"""Smoke checks for native-update product exposure decision gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_product_exposure_decision import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_product_exposure_decision,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "product_exposure_enabled",
    "product_exposure_approved",
    "release_gate_open",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "training_dispatch",
    "request_submission_allowed",
    "request_submitted",
    "request_payload_materialized",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
    "rollout_authorization_allowed",
)


def run_smoke() -> dict[str, Any]:
    pending = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence=_product_exposure_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_product_exposure_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_product_exposure_request_fields"] == {}, pending
    assert "native_update_product_exposure_owner_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence=_product_exposure_evidence(),
        product_exposure_review=_product_exposure_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["product_exposure_decision_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_launch = build_native_update_product_exposure_decision(
        training_launch_contract={},
        product_exposure_evidence=_product_exposure_evidence(),
    )
    _assert_evidence_blocked(missing_launch, "training_launch_contract_missing")

    missing_evidence = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "product_exposure_evidence_missing")

    unsafe_launch = build_native_update_product_exposure_decision(
        training_launch_contract={**_training_launch_contract(), "request_fields_emitted": True},
        product_exposure_evidence=_product_exposure_evidence(),
    )
    _assert_evidence_blocked(unsafe_launch, "request_fields_emitted")

    unsafe_evidence = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence={**_product_exposure_evidence(), "backend_router_registered": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "backend_router_registered")

    bad_boundary = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence=_product_exposure_evidence(boundary_patch={"ready_for_ui": True}),
    )
    _assert_evidence_blocked(bad_boundary, "not_default_off")

    unsafe_review = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence=_product_exposure_evidence(),
        product_exposure_review={**_product_exposure_review(approve=True), "approve_request_submitted": True},
    )
    _assert_review_blocked(unsafe_review, "approve_request_submitted")

    missing_ack = build_native_update_product_exposure_decision(
        training_launch_contract=_training_launch_contract(),
        product_exposure_evidence=_product_exposure_evidence(),
        product_exposure_review={
            **_product_exposure_review(approve=True),
            "acknowledge_no_request_adapter_or_schema_change": False,
        },
    )
    _assert_review_blocked(missing_ack, "ack_missing", "request_adapter")

    real_artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_product_exposure_decision_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case() -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    launch_path = temp_dir / "native_update_training_launch_contract.json"
    evidence_path = temp_dir / "native_update_product_exposure_evidence.json"
    contract_path = temp_dir / "native_update_product_exposure_decision.json"
    evidence = _product_exposure_evidence(source=str(evidence_path))
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launch = load_json(launch_path) if launch_path.exists() else _training_launch_contract()
    package = build_native_update_product_exposure_decision(
        training_launch_contract=launch,
        product_exposure_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_product_exposure_request_fields"] == {}, package
    _assert_default_off(package)
    contract_path.write_text(
        json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package


def _assert_default_off(package: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert package[field] is False, (field, package)


def _assert_evidence_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is False, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _assert_review_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _training_launch_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_training_launch_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_training_launch_review": True,
        "training_launch_contract_recorded": False,
        "decision": "native_update_training_launch_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "training_launch_enabled": False,
        "training_launch_executed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "request_submission_allowed": False,
        "request_submitted": False,
        "job_created": False,
        "queue_enqueued": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_training_launch_request_fields": {},
    }


def _product_exposure_evidence(
    *,
    source: str = "smoke://native_update_product_exposure_evidence",
    boundary_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "training_launch_contract_reference",
        "owner_exposure_decision_boundary",
        "request_adapter_boundary",
        "request_schema_boundary",
        "backend_router_boundary",
        "launcher_ui_boundary",
        "webui_boundary",
        "release_gate_boundary",
        "no_training_launch_boundary",
        "no_request_submission_boundary",
        "no_request_ui_schema_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    owner = _default_off_row("native_update_owner_product_exposure_decision", source)
    owner.update(boundary_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_product_exposure_evidence_v0",
        "ok": True,
        "product_exposure_decision_ready": True,
        "report_only": True,
        "contract_only": True,
        "product_exposure_decision_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": source,
        "artifact_digest": "smoke-product-exposure-digest",
        "sections": sections,
        "owner_exposure_decision_boundary": [owner],
        "request_adapter_boundary": [_default_off_row("native_update_request_adapter_boundary", source)],
        "request_schema_boundary": [_default_off_row("native_update_request_schema_boundary", source)],
        "backend_router_boundary": [_default_off_row("native_update_backend_router_boundary", source)],
        "launcher_ui_boundary": [_default_off_row("native_update_launcher_ui_boundary", source)],
        "webui_boundary": [_default_off_row("native_update_webui_boundary", source)],
        "release_gate_boundary": [_default_off_row("native_update_release_gate_boundary", source)],
        "rollback_policy": [{"id": "native_update_product_exposure_rollback", "ready": True, "source": source}],
        "observability_policy": [{"id": "native_update_product_exposure_observability", "ready": True, "source": source}],
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
    }


def _default_off_row(row_id: str, source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "ready": True,
        "source": source,
        "product_exposure_allowed": False,
        "product_exposure_enabled": False,
        "product_exposure_approved": False,
        "release_gate_open": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "request_submitted": False,
        "job_created": False,
        "queue_enqueued": False,
        "run_record_written": False,
        "ready_for_ui": False,
        "ui_exposure_allowed": False,
        "launcher_exposure_allowed": False,
        "webui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "backend_router_registered": False,
    }


def _product_exposure_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_product_exposure_decision",
        "approve_native_update_product_exposure_decision": bool(approve),
    }
    for field in (
        "acknowledge_training_launch_contract_ready",
        "acknowledge_product_exposure_default_off",
        "acknowledge_no_request_adapter_or_schema_change",
        "acknowledge_no_backend_router_registration",
        "acknowledge_no_launcher_or_webui_exposure",
        "acknowledge_no_training_launch_or_request_submission",
        "acknowledge_release_requires_separate_owner_decision",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
