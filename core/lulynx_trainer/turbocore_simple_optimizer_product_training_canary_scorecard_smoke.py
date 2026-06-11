"""Smoke checks for fp32 simple optimizer product-training canary package."""

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

from core.turbocore_simple_optimizer_family_batch_scorecard import (  # noqa: E402
    build_simple_optimizer_family_batch_scorecard,
)
from core.turbocore_simple_optimizer_product_training_canary_scorecard import (  # noqa: E402
    REQUIRED_STAGES,
    build_simple_optimizer_product_training_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    family = build_simple_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    payload = build_simple_optimizer_product_training_canary_scorecard(
        family_batch_report=family,
        workspace_root=REPO_ROOT,
    )
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    assert payload["scorecard"] == "turbocore_simple_optimizer_product_training_canary_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["representative_product_training_canary_ready"] is True, payload
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
    assert payload["summary"]["representative_product_training_canary_ready_count"] == 2, payload
    assert payload["summary"]["ready_required_stage_count"] == len(REQUIRED_STAGES), payload
    assert rows["Lion"]["canary_status"] == "representative_product_training_canary_ready", rows["Lion"]
    assert rows["SGDNesterov"]["canary_status"] == "representative_product_training_canary_ready", rows["SGDNesterov"]
    assert rows["Lion"]["native_kernel_ready"] is True, rows["Lion"]
    assert rows["Lion"]["runtime_canary_ready"] is True, rows["Lion"]
    assert rows["Lion"]["training_loop_canary_ready"] is True, rows["Lion"]
    assert all(rows["Lion"]["required_stage_ready"].values()), rows["Lion"]
    assert all(rows["SGDNesterov"]["required_stage_ready"].values()), rows["SGDNesterov"]
    assert "simple_formula_owner_approval_missing" in payload["promotion_blockers"], payload
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_product_training_canary_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_product_training_canary_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
