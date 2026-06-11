"""Smoke checks for quantized simple owner-approval hold scorecard."""

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

from core.turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_owner_approval_hold_scorecard,
)


def run_smoke() -> dict[str, Any]:
    review = build_simple_optimizer_quantized_dispatch_integration_review_scorecard()
    assert review["dispatch_integration_review"] is True, review
    payload = build_simple_optimizer_quantized_owner_approval_hold_scorecard(
        dispatch_review_report=review
    )
    hold = payload["hold_manifest"]
    frozen = hold["frozen_product_boundaries"]
    assert payload["scorecard"] == "turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["owner_approval_hold_ready"] is True, payload
    assert payload["dispatch_integration_review"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["owner_approval_recorded"] is False, payload
    assert payload["release_approval_recorded"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["summary"]["optimizer_count"] == 3, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert hold["approval_state"] == "pending_owner_approval", hold
    assert hold["optimizer_types"] == ["Lion8bit", "PagedLion8bit", "SGDNesterov8bit"], hold
    assert hold["allowed_initial_modes"] == ["off", "observe"], hold
    assert hold["blocked_modes_until_owner_approval"] == ["canary", "auto"], hold
    assert all(value is False for value in frozen.values()), frozen
    assert "simple_quantized_owner_approval_missing" in payload["promotion_blockers"], payload
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
