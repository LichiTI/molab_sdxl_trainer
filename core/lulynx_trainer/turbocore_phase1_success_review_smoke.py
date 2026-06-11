"""Smoke checks for TurboCore Phase 1 success review aggregation."""

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

from core.turbocore_phase1_success_review import (  # noqa: E402
    BLOCKED_DECISION,
    SUCCESS_DECISION,
    build_turbocore_phase1_success_review,
)


def _parity(*, ok: bool = True) -> dict[str, Any]:
    return {
        "summary": {"ok": ok, "native_kernel_present": False},
        "results": [
            {"name": "native_optimizer_adamw", "ok": ok, "max_abs_error": 0.0},
            {
                "name": "native_optimizer_adamw_stateful",
                "ok": ok,
                "max_abs_error": 0.0,
                "details": {
                    "restore_ok": ok,
                    "nonfinite_skip_ok": ok,
                    "nonfinite_params_unchanged": ok,
                    "max_grad_norm": 1.0,
                },
            },
        ],
    }


def _performance(*, ready: bool = True, persistent_route: bool = True) -> dict[str, Any]:
    return {
        "gate": "turbocore_native_update_performance_gate_v0",
        "representative_performance_gate_ready": ready,
        "promotion_gate_ok": ready,
        "required_end_to_end_speedup": 1.03,
        "runtime_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "evidence": {
            "optimizer_microbenchmark": {
                "ok": ready,
                "best_speedup_vs_baseline": 1.35,
            },
            "owner_native_kernel": {
                "ok": ready and persistent_route,
                "kernel_executed": ready and persistent_route,
                "parity_ok": ready and persistent_route,
            },
            "training_matrix": {
                "ok": ready,
                "representative_steps": 24,
                "native_dispatch_executed": persistent_route,
                "end_to_end_speedup": 1.05 if persistent_route else 0.99,
            },
        },
        "blocked_reasons": [] if ready else ["representative_training_matrix_missing"],
    }


def _layout(*, ready: bool = True) -> dict[str, Any]:
    return {
        "probe": "turbocore_adamw_layout_probe",
        "ok": True,
        "summary": {
            "flat_kernel_gate_ok": True,
            "layout_including_gather_scatter_gate_ok": ready,
            "flat_kernel_speedup": 1.4,
            "layout_including_gather_scatter_speedup": 1.12 if ready else 0.9,
            "layout_tax_ms": 0.2,
            "recommendation": "layout_cost_still_passes_gate_try_route_level_probe",
        },
    }


def test_missing_evidence_blocks_default_off() -> None:
    report = build_turbocore_phase1_success_review()
    assert report["ok"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert "phase1_parity_report_missing" in report["blocked_reasons"], report


def test_complete_evidence_passes_but_stays_default_off() -> None:
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(),
        performance_report=_performance(),
        layout_probe=_layout(),
    )
    assert report["ok"] is True, report
    assert report["decision"] == SUCCESS_DECISION, report
    assert report["phase1_success_ready"] is True, report
    assert report["training_activation_allowed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["post_phase1_request_fields"] == {}, report


def test_bad_parity_blocks_phase1_success() -> None:
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(ok=False),
        performance_report=_performance(),
        layout_probe=_layout(),
    )
    assert report["ok"] is False, report
    assert "phase1_parity_summary_not_ok" in report["blocked_reasons"], report
    assert "phase1_stateful_clipping_finite_lifecycle_not_proven" in report["blocked_reasons"], report


def test_flat_only_layout_win_does_not_pass_success() -> None:
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(),
        performance_report=_performance(persistent_route=False),
        layout_probe=_layout(ready=False),
    )
    assert report["ok"] is False, report
    assert "phase1_layout_including_gather_scatter_gate_not_ok" in report["blocked_reasons"], report
    assert "phase1_layout_cost_not_proven_after_transfer_or_sync" in report["blocked_reasons"], report


def test_persistent_buffer_route_can_prove_layout_cost() -> None:
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(),
        performance_report=_performance(persistent_route=True),
        layout_probe=_layout(ready=False),
    )
    assert report["ok"] is True, report
    assert report["layout_summary"]["layout_cost_gate_ok"] is True, report
    assert report["layout_summary"]["integration_strategy"] == "persistent_buffer_route", report
    assert report["layout_summary"]["persistent_buffer_route_cost_ok"] is True, report


def test_unsafe_source_claim_blocks_even_with_good_evidence() -> None:
    perf = _performance()
    perf["runtime_dispatch_allowed"] = True
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(),
        performance_report=perf,
        layout_probe=_layout(),
    )
    assert report["ok"] is False, report
    assert "runtime_dispatch_allowed" in "\n".join(report["blocked_reasons"]), report
    assert report["runtime_dispatch_allowed"] is False, report


def run_smoke() -> dict[str, Any]:
    test_missing_evidence_blocks_default_off()
    test_complete_evidence_passes_but_stays_default_off()
    test_bad_parity_blocks_phase1_success()
    test_flat_only_layout_win_does_not_pass_success()
    test_persistent_buffer_route_can_prove_layout_cost()
    test_unsafe_source_claim_blocks_even_with_good_evidence()
    artifact = _write_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_phase1_success_review_smoke",
        "ok": True,
        "real_artifact_checked": bool(artifact),
    }


def _write_real_artifact_case() -> dict[str, Any]:
    report = build_turbocore_phase1_success_review(
        parity_report=_parity(),
        performance_report=_performance(persistent_route=True),
        layout_probe=_layout(ready=False),
    )
    assert report["ok"] is True, report
    assert report["decision"] == SUCCESS_DECISION, report
    assert report["layout_summary"]["integration_strategy"] == "persistent_buffer_route", report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["post_phase1_request_fields"] == {}, report
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "turbocore_phase1_success_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
