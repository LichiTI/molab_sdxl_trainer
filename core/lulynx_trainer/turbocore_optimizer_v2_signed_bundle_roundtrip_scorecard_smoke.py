"""Smoke checks for v2 signed-bundle dry-run roundtrip."""

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

from core.turbocore_optimizer_v2_signed_bundle_roundtrip_scorecard import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_signed_bundle_roundtrip_scorecard,
)


def run_smoke() -> dict[str, Any]:
    payload = build_optimizer_v2_signed_bundle_roundtrip_scorecard(write_artifact=True)
    summary = payload["summary"]
    assert payload["ok"] is True, payload
    assert payload["roundtrip_dry_run_ready"] is True, payload
    assert payload["approval_recorded"] is False, payload
    assert payload["approval_artifact_written"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["product_native_ready"] is False, payload
    assert payload["input_artifacts_are_temporary"] is True, payload
    assert summary["v2_signed_bundle_roundtrip_ready_template_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_signed_bundle_present_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_intake_valid_record_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_extracted_entry_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_extraction_signed_entry_digest_present_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_extraction_extractable_signed_entry_digest_present_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_extraction_digest_shape_ready_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_extraction_artifact_written_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_preflight_signed_payload_digest_present_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_preflight_signed_payload_digest_missing_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_preflight_signed_bundle_entry_digest_present_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_preflight_signed_payload_bundle_digest_match_count"] == 2, payload
    assert summary["v2_signed_bundle_roundtrip_preflight_digest_shape_ready_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_phase1_ready_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_phase2_ready_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_full_ready_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_owner_direction_blocked_count"] == 1, payload
    assert summary["v2_signed_bundle_roundtrip_approval_recorded_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_approval_artifact_written_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_runtime_dispatch_ready_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_native_dispatch_allowed_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_training_path_enabled_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_product_native_ready_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_default_behavior_changed_count"] == 0, payload
    assert summary["v2_signed_bundle_roundtrip_unsafe_claim_count"] == 0, payload
    assert ARTIFACT.exists(), ARTIFACT
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_signed_bundle_roundtrip_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "synthetic_roundtrip_checked_in_temp": True,
        "approval_artifact_written": False,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
