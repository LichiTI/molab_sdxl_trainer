"""Smoke checks for exact AdamW stream/event-chain ABI scorecard."""

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

from core.turbocore_exact_adamw_stream_event_chain_abi_scorecard import (  # noqa: E402
    GATE,
    build_exact_adamw_stream_event_chain_abi_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_exact_adamw_stream_event_chain_abi_scorecard(write_artifact=True)
    summary = report["summary"]
    dispatch = report["dispatch_contract"]
    stream = dispatch["stream_lifetime_ownership"]
    sync = report["sync_policy"]

    assert report["scorecard"] == "turbocore_exact_adamw_stream_event_chain_abi_scorecard_v0", report
    assert report["gate"] == GATE, report
    assert report["ok"] is True, report
    assert report["evidence_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["ready_for_optimizer_family_coverage_review"] is True, report
    assert report["optimizer_type"] == "AdamW", report
    assert report["native_route"] == "rust_cuda_adamw_v0", report
    assert report["stream_event_chain_ownership_abi_ready"] is True, report
    assert report["stream_lifetime_ownership_boundary_ready"] is True, report
    assert report["stream_lifetime_ownership_bound_evidence"] is True, report
    assert report["stream_ordering_verified"] is True, report
    assert report["event_chain_verified"] is True, report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_exposure_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["sync_fast_path_allowed"] is False, report
    assert report["blocked_reasons"] == [], report
    assert "exact_adamw_owner_release_approval_missing" in report["promotion_blockers"], report

    assert summary["optimizer_count"] == 1, summary
    assert summary["stream_event_chain_ownership_abi_ready_count"] == 1, summary
    assert summary["stream_lifetime_ownership_boundary_ready_count"] == 1, summary
    assert summary["stream_lifetime_ownership_bound_evidence_count"] == 1, summary
    assert summary["stream_ordering_verified_count"] == 1, summary
    assert summary["event_chain_verified_count"] == 1, summary
    assert summary["sync_fast_path_allowed_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary

    assert dispatch["training_dispatch"] is False, dispatch
    assert dispatch["training_path_enabled"] is False, dispatch
    assert dispatch["would_allow_native_dispatch"] is False, dispatch
    assert dispatch["native_mutation_allowed"] is False, dispatch
    assert dispatch["stream_lifetime_bound"] is True, dispatch
    assert dispatch["stream_lifetime_ownership_bound"] is True, dispatch
    assert dispatch["stream_ordering_verified"] is True, dispatch
    assert stream["ownership_boundary_ready"] is True, stream
    assert stream["ownership_bound_evidence"] is True, stream
    assert stream["ordering_verified"] is True, stream
    assert stream["default_off"] is True, stream
    assert "stream_lifetime_ownership_default_off" in stream["blocked_reasons"], stream
    assert sync["requested_mode"] == "off", sync
    assert sync["sync_fast_path_allowed"] is False, sync
    assert "v5_p8_stream_sync_policy_default_off" in sync["blocked_reasons"], sync

    return {
        "schema_version": 1,
        "probe": "turbocore_exact_adamw_stream_event_chain_abi_scorecard_smoke",
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
