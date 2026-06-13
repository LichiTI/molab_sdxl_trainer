"""Smoke checks for the v2 signature bundle packet."""

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

from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)


ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_v2_signature_bundle_packet.json"
)


def run_smoke() -> dict[str, Any]:
    packet = build_optimizer_v2_signature_bundle_packet(write_artifact=True)
    summary = packet["summary"]
    entries = {entry["signature_id"]: entry for entry in packet["signature_entries"]}

    assert packet["package"] == "turbocore_optimizer_v2_signature_bundle_packet_v0", packet
    assert packet["ok"] is True, packet
    assert packet["signature_bundle_ready"] is True, packet
    assert packet["roadmap_complete"] is False, packet
    assert packet["approval_recorded"] is False, packet
    assert packet["promotion_ready"] is False, packet
    assert packet["runtime_dispatch_ready"] is False, packet
    assert packet["native_dispatch_allowed"] is False, packet
    assert packet["training_path_enabled"] is False, packet
    assert packet["product_native_ready"] is False, packet
    assert set(entries) == {
        "owner_release_review",
        "product_exposure_review",
        "owner_release_direction",
    }, packet
    assert entries["owner_release_review"]["ready_for_signature"] is True, packet
    assert entries["product_exposure_review"]["ready_for_signature"] is True, packet
    assert entries["owner_release_direction"]["ready_for_signature"] is False, packet
    for entry in entries.values():
        assert entry["template"]["source_v2_signature_template_digest"] == entry["source_template_digest"], entry
    assert "product_exposure_decision_not_recorded" in entries["owner_release_direction"]["blocked_reasons"], packet
    assert summary["v2_signature_bundle_entry_count"] == 3, packet
    assert summary["v2_signature_bundle_ready_for_signature_count"] == 2, packet
    assert summary["v2_signature_bundle_blocked_entry_count"] == 1, packet
    assert summary["v2_signature_bundle_owner_review_ready_count"] == 1, packet
    assert summary["v2_signature_bundle_product_exposure_ready_count"] == 1, packet
    assert summary["v2_signature_bundle_owner_direction_ready_count"] == 0, packet
    assert summary["v2_signature_bundle_approval_recorded_count"] == 0, packet
    assert summary["v2_signature_bundle_runtime_dispatch_ready_count"] == 0, packet
    assert summary["v2_signature_bundle_native_dispatch_allowed_count"] == 0, packet
    assert summary["v2_signature_bundle_training_path_enabled_count"] == 0, packet
    assert summary["v2_signature_bundle_product_native_ready_count"] == 0, packet
    assert summary["v2_signature_bundle_default_behavior_changed_count"] == 0, packet
    assert summary["v2_signature_bundle_unsafe_claim_count"] == 0, packet
    assert ARTIFACT.exists(), ARTIFACT

    unsafe = build_optimizer_v2_signature_bundle_packet(
        source_reports={
            "remaining_gate_handoff": {
                "handoff_ready": True,
                "summary": {
                    "v2_remaining_gate_total_count": 6,
                    "v2_remaining_gate_default_off_guard_count": 1,
                },
            },
            "owner_release_review_packet": {
                "ready_for_owner_signature": True,
                "runtime_dispatch_allowed": True,
                "signable_review_record_template": {"requested_scope": "native_update_release_review_package"},
            },
            "product_exposure_decision": {
                "ready_for_product_exposure_review": True,
                "product_exposure_review_template": {"requested_scope": "native_update_product_exposure_decision"},
            },
            "owner_release_direction_packet": {
                "ready_for_owner_direction_signature": False,
                "blocked_reasons": ["product_exposure_decision_not_recorded"],
                "signable_owner_release_direction_template": {
                    "requested_scope": "native_update_owner_release_direction"
                },
            },
        },
        write_artifact=False,
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["v2_signature_bundle_unsafe_claim_count"] == 1, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_signature_bundle_packet_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": packet["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
