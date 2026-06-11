"""Smoke checks for native-update rollout review package."""

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

from core.turbocore_native_update_rollout_review_package import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_rollout_review_package,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)


def run_smoke() -> dict[str, Any]:
    pending = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix=_performance_matrix(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_package_ready"] is True, pending
    assert pending["ready_for_owner_review"] is True, pending
    assert pending["owner_review_action_required"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_native_update_request_fields"] == {}, pending
    assert "native_update_rollout_owner_review_not_signed" in pending["blocked_reasons"], pending
    _assert_default_off(pending)
    progress = pending["progress_gates"]
    assert progress["representative_performance_ready"] is True, pending
    assert progress["training_loop_dispatch_smoke_ready"] is True, pending
    assert progress["native_kernel_launched"] is True, pending
    assert progress["request_ui_schema_exposure_blocked"] is True, pending

    signed = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix=_performance_matrix(),
        owner_review=_owner_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["owner_review_recorded"] is True, signed
    assert signed["native_update_rollout_review_package_ready"] is True, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_readiness = build_native_update_rollout_review_package(
        readiness_report={},
        performance_matrix=_performance_matrix(),
    )
    _assert_blocked(missing_readiness, "readiness_report_missing")

    missing_performance = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix={},
    )
    _assert_blocked(missing_performance, "performance_matrix_missing")

    performance_blocked = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix=_performance_matrix(ok=False),
    )
    _assert_blocked(performance_blocked, "performance_matrix_not_ready")

    unsafe_readiness = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(ready_for_ui=True),
        performance_matrix=_performance_matrix(),
    )
    _assert_blocked(unsafe_readiness, "ready_for_ui")

    unsafe_performance = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix=_performance_matrix(training_dispatch=True),
    )
    _assert_blocked(unsafe_performance, "training_dispatch")

    unsafe_review = build_native_update_rollout_review_package(
        readiness_report=_readiness_report(),
        performance_matrix=_performance_matrix(),
        owner_review={**_owner_review(approve=True), "approve_request_fields_emitted": True},
    )
    _assert_review_blocked(unsafe_review, "approve_request_fields_emitted")

    real_artifact = _optional_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_rollout_review_package_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _optional_real_artifact_case() -> dict[str, Any]:
    readiness_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "readiness_with_native_update_perf.json"
    matrix_path = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "native_update_dispatch_ctx_sync_free_matrix_20step"
        / "matrix_summary.json"
    )
    if not readiness_path.exists() or not matrix_path.exists():
        return {}
    package = build_native_update_rollout_review_package(
        readiness_report=load_json(readiness_path),
        performance_matrix=load_json(matrix_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_package_ready"] is True, package
    assert package["ready_for_owner_review"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_native_update_request_fields"] == {}, package
    assert package["performance_matrix_summary"]["representative_end_to_end_speedup"] >= 1.03, package
    _assert_default_off(package)
    return package


def _assert_default_off(package: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert package[field] is False, (field, package)


def _assert_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_package_ready"] is False, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _assert_review_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_package_ready"] is True, package
    assert package["ready_for_owner_review"] is True, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _readiness_report(*, ready_for_ui: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_readiness",
        "summary": {
            "ok": True,
            "ready_for_ui": bool(ready_for_ui),
            "native_update_representative_performance_ready": True,
            "native_update_performance_blockers": [],
            "native_update_training_executor_available": True,
            "native_update_training_loop_dispatch_smoke_ok": True,
            "native_update_native_kernel_launched": True,
            "native_update_promotion_ready": False,
            "native_update_promotion_blockers": [
                "native_runtime_recovery_training_dispatch_disabled",
                "owner_gradient_sync_default_off",
                "native_training_flat_owner_default_off",
            ],
            "native_training_path_locked": True,
            "recommended_next_step": "prepare explicit native-update rollout review while keeping product dispatch default-off",
        },
        "sections": {
            "native_update_training_loop_dispatch_smoke": {
                "ok": True,
                "skipped": False,
                "native_training_executor_available": True,
                "native_step_executed": True,
                "native_kernel_launched": True,
                "training_path_enabled": False,
                "default_behavior_changed": False,
            },
            "native_update_promotion_scorecard": {
                "performance_gate": {
                    "representative_performance_gate_ready": True,
                    "representative_end_to_end_speedup": 1.046,
                    "blocked_reasons": [],
                }
            },
        },
    }


def _performance_matrix(*, ok: bool = True, training_dispatch: bool = False) -> dict[str, Any]:
    blocked = [] if ok else ["end_to_end_speedup_below_threshold"]
    speedup = 1.046 if ok else 0.91
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [{"case": {"name": "baseline_phase"}}, {"case": {"name": "native_update_dispatch_promotion_perf"}}],
        "summary": {
            "all_success": True,
            "executed_count": 2,
            "native_update_performance_gate": {
                "ready": ok,
                "blocked_reasons": blocked,
                "optimizer_evidence_present": ok,
                "optimizer_evidence_quality": "promotion_benchmark" if ok else "",
            },
            "native_dispatch_ctx_sync_free_comparison": {
                "ctx_sync_free_case": "native_update_dispatch_ctx_sync_free_canary",
                "ctx_sync_free_speedup_vs_baseline": 1.0488 if ok else None,
                "ctx_sync_free_speedup_vs_context_sync_native": 1.0026 if ok else None,
                "representative_candidate_ready": False,
            },
        },
        "native_update_performance_report": {
            "training_dispatch": bool(training_dispatch),
            "training_path_enabled": False,
            "runtime_dispatch_allowed": False,
            "performance_gate": {
                "gate": "turbocore_native_update_performance_gate_v0",
                "representative_performance_gate_ready": ok,
                "promotion_gate_ok": ok,
                "required_end_to_end_speedup": 1.03,
                "training_dispatch": False,
                "training_path_enabled": False,
                "runtime_dispatch_allowed": False,
                "blocked_reasons": blocked,
                "evidence": {
                    "optimizer_microbenchmark": {
                        "ok": ok,
                        "present": ok,
                        "evidence_quality": "promotion_benchmark" if ok else "",
                        "best_speedup_vs_baseline": 6.5091 if ok else None,
                    },
                    "training_matrix": {
                        "ok": ok,
                        "present": ok,
                        "native_case": "native_update_dispatch_promotion_perf",
                        "end_to_end_speedup": speedup,
                        "representative_steps": 20,
                    },
                },
            },
        },
    }


def _owner_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-04",
        "requested_scope": "native_update_rollout_review_package",
        "approve_native_update_rollout_review_package": bool(approve),
    }
    for field in (
        "acknowledge_representative_performance_ready",
        "acknowledge_training_loop_dispatch_smoke_ready",
        "acknowledge_native_kernel_launched_in_explicit_smoke",
        "acknowledge_default_off_boundary",
        "acknowledge_no_product_training_dispatch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_request_adapter_or_schema_exposure",
        "acknowledge_later_integration_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
