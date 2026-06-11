"""Smoke checks for quantized simple optimizer e2e no-regression scorecard."""

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

from core.turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_e2e_no_regression_scorecard,
)
from core.turbocore_simple_optimizer_quantized_training_loop_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    training = build_simple_optimizer_quantized_training_loop_canary_scorecard()
    assert training["training_loop_canary_ready"] is True, training
    payload = build_simple_optimizer_quantized_e2e_no_regression_scorecard(
        training_loop_canary_report=training
    )
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    assert payload["scorecard"] == "turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["e2e_no_regression_ready"] is True, payload
    assert payload["product_state_sync_review_ready"] is True, payload
    assert payload["product_optimizer_state_sync_ready"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["summary"]["target_optimizer_count"] == 3, payload
    assert payload["summary"]["training_loop_canary_ready_count"] == 3, payload
    assert payload["summary"]["e2e_no_regression_ready_count"] == 3, payload
    assert payload["summary"]["fallback_authority_unchanged_count"] == 3, payload
    assert payload["summary"]["product_state_sync_review_ready_count"] == 3, payload
    assert payload["summary"]["product_optimizer_state_sync_ready_count"] == 0, payload
    assert payload["summary"]["request_schema_ui_unchanged_count"] == 3, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert rows["Lion8bit"]["variant_status"] == "quantized_e2e_no_regression_ready", rows["Lion8bit"]
    assert rows["Lion8bit"]["optimizer_state_sync_synced"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["optimizer_state_sync_state_tensors"] == 2, rows["Lion8bit"]
    assert rows["PagedLion8bit"]["fallback_authority_unchanged"] is True, rows["PagedLion8bit"]
    assert rows["SGDNesterov8bit"]["product_optimizer_state_sync_ready"] is False, rows["SGDNesterov8bit"]
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
