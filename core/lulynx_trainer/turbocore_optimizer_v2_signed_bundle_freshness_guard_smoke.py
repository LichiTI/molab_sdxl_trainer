"""Smoke checks for v2 signed-bundle freshness guard."""

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

from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signed_bundle_freshness_guard import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_signed_bundle_freshness_guard,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)


def run_smoke() -> dict[str, Any]:
    bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=False)
    handoff = build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=bundle,
        write_artifact=False,
        write_signed_bundle_template=False,
    )
    default_payload = build_optimizer_v2_signed_bundle_freshness_guard(
        signature_bundle=bundle,
        reviewer_handoff=handoff,
        write_artifact=True,
    )
    default_summary = default_payload["summary"]
    assert default_payload["ok"] is True, default_payload
    assert default_payload["freshness_guard_ready"] is True, default_payload
    assert default_payload["approval_recorded"] is False, default_payload
    assert default_payload["approval_artifact_written"] is False, default_payload
    assert default_payload["runtime_dispatch_ready"] is False, default_payload
    assert default_payload["native_dispatch_allowed"] is False, default_payload
    assert default_payload["training_path_enabled"] is False, default_payload
    assert default_payload["product_native_ready"] is False, default_payload
    assert default_summary["v2_signed_bundle_freshness_guard_ready_count"] == 1, default_payload
    assert default_summary["v2_signed_bundle_freshness_current_digest_present_count"] == 1, default_payload
    assert default_summary["v2_signed_bundle_freshness_template_digest_match_count"] == 1, default_payload
    assert default_summary["v2_signed_bundle_freshness_signed_bundle_present_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_stale_signed_bundle_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_approval_recorded_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_runtime_dispatch_ready_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_native_dispatch_allowed_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_training_path_enabled_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_product_native_ready_count"] == 0, default_payload
    assert default_summary["v2_signed_bundle_freshness_unsafe_claim_count"] == 0, default_payload
    assert ARTIFACT.exists(), ARTIFACT

    fresh_signed = _signed_bundle_from_handoff(handoff, default_payload["current_signature_bundle_digest"])
    fresh_payload = build_optimizer_v2_signed_bundle_freshness_guard(
        signature_bundle=bundle,
        reviewer_handoff=handoff,
        signed_bundle=fresh_signed,
        write_artifact=False,
    )
    stale_signed = _signed_bundle_from_handoff(handoff, "stale-digest-for-smoke")
    stale_payload = build_optimizer_v2_signed_bundle_freshness_guard(
        signature_bundle=bundle,
        reviewer_handoff=handoff,
        signed_bundle=stale_signed,
        write_artifact=False,
    )
    assert fresh_payload["ok"] is True, fresh_payload
    assert fresh_payload["summary"]["v2_signed_bundle_freshness_signed_bundle_digest_match_count"] == 1, fresh_payload
    assert fresh_payload["summary"]["v2_signed_bundle_freshness_stale_signed_bundle_count"] == 0, fresh_payload
    assert stale_payload["ok"] is False, stale_payload
    assert stale_payload["summary"]["v2_signed_bundle_freshness_stale_signed_bundle_count"] == 1, stale_payload
    assert stale_payload["summary"]["v2_signed_bundle_freshness_approval_recorded_count"] == 0, stale_payload

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_signed_bundle_freshness_guard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "fresh_signed_bundle_digest_checked": True,
        "stale_signed_bundle_rejected": True,
        "approval_artifact_written": False,
        "summary": default_summary,
        "recommended_next_step": default_payload["recommended_next_step"],
    }


def _signed_bundle_from_handoff(handoff: dict[str, Any], source_digest: str) -> dict[str, Any]:
    template = handoff["signed_bundle_template"]
    return {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_freshness_smoke_v0",
        "source_signature_bundle_digest": source_digest,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "signed_entries": {
            signature_id: _sign_template(signature_id, entry)
            for signature_id, entry in template["signed_entries"].items()
        },
    }


def _sign_template(signature_id: str, template: dict[str, Any]) -> dict[str, Any]:
    signed = dict(template)
    signed["reviewer"] = f"synthetic_{signature_id}_freshness_smoke"
    signed["reviewed_at"] = "2026-06-09"
    for key in list(signed):
        if key.startswith("approve_") or key.startswith("acknowledge_"):
            signed[key] = True
    return signed


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
