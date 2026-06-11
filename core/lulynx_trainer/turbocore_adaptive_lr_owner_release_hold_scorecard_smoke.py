"""Smoke checks for adaptive-LR owner/release hold evidence."""

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

from core.turbocore_adaptive_lr_dispatch_integration_review_scorecard import (  # noqa: E402
    EXPECTED_OPTIMIZERS,
    build_adaptive_lr_dispatch_integration_review_scorecard,
)
from core.turbocore_adaptive_lr_owner_release_hold_scorecard import (  # noqa: E402
    HOLD_KIND,
    build_adaptive_lr_owner_release_hold_scorecard,
)


def run_smoke() -> dict[str, Any]:
    review = build_adaptive_lr_dispatch_integration_review_scorecard()
    payload = build_adaptive_lr_owner_release_hold_scorecard(dispatch_review_report=review)
    summary = payload["summary"]
    hold = payload["hold_manifest"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_owner_release_hold_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["owner_release_hold_ready"] is True, payload
    assert payload["dispatch_integration_review"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["owner_approval_recorded"] is False, payload
    assert payload["release_approval_recorded"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["hold_kind"] == HOLD_KIND, payload

    assert set(hold["optimizer_types"]) == EXPECTED_OPTIMIZERS, hold
    assert hold["approval_state"] == "pending_owner_and_release_approval", hold
    assert hold["owner_approval_recorded"] is False, hold
    assert hold["release_approval_recorded"] is False, hold
    assert hold["allowed_initial_modes"] == ["off", "observe"], hold
    assert hold["blocked_modes_until_approval"] == ["canary", "auto"], hold
    frozen = hold["frozen_product_boundaries"]
    assert frozen["request_fields_emitted"] is False, frozen
    assert frozen["schema_exposure_allowed"] is False, frozen
    assert frozen["ui_exposure_allowed"] is False, frozen
    assert frozen["runtime_dispatch_ready"] is False, frozen
    assert frozen["native_dispatch_allowed"] is False, frozen
    assert frozen["training_path_enabled"] is False, frozen

    assert summary["owner_release_hold_ready"] is True, summary
    assert summary["dispatch_integration_review"] is True, summary
    assert summary["manual_review_required"] is True, summary
    assert summary["owner_approval_recorded"] is False, summary
    assert summary["release_approval_recorded"] is False, summary
    assert summary["optimizer_count"] == len(EXPECTED_OPTIMIZERS), summary
    assert summary["runtime_dispatch_ready"] is False, summary
    assert summary["native_dispatch_allowed"] is False, summary
    assert summary["training_path_enabled"] is False, summary
    assert summary["product_native_ready_count"] == 0, summary

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_owner_release_hold_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_owner_release_hold_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
