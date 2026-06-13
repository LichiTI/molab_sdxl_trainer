"""Smoke checks for the v2 O3 adaptive-LR chain aggregate."""

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

from core.turbocore_adaptive_lr_chain_scorecard import (  # noqa: E402
    build_adaptive_lr_chain_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adaptive_lr_chain_scorecard(
        run_live_cuda_implementation=False,
        run_live_tensor_binding_canary=True,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    summary = report["summary"]
    rows = {row["roadmap_item"]: row for row in report["rows"]}
    artifact_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_chain_scorecard.json"

    assert report["scorecard"] == "turbocore_adaptive_lr_chain_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["adaptive_lr_chain_ready"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert summary["adaptive_lr_chain_stage_count"] == 9, report
    assert summary["adaptive_lr_chain_target_optimizer_count"] == 11, report
    assert summary["adaptive_lr_chain_ready_stage_count"] == 8, report
    assert summary["adaptive_lr_chain_open_stage_count"] == 1, report
    assert summary["adaptive_lr_chain_product_exposure_gate_ready_count"] == 0, report
    assert summary["adaptive_lr_chain_runtime_dispatch_ready_count"] == 0, report
    assert summary["adaptive_lr_chain_native_dispatch_allowed_count"] == 0, report
    assert summary["adaptive_lr_chain_training_path_enabled_count"] == 0, report
    assert summary["adaptive_lr_chain_default_behavior_changed_count"] == 0, report
    assert summary["adaptive_lr_chain_product_native_ready_count"] == 0, report
    assert rows["O3-1"]["stage_ready"] is True, rows["O3-1"]
    assert rows["O3-2"]["stage_ready"] is True, rows["O3-2"]
    assert rows["O3-3"]["stage_ready"] is True, rows["O3-3"]
    assert rows["O3-4"]["stage_ready"] is True, rows["O3-4"]
    assert rows["O3-5"]["stage_ready"] is True, rows["O3-5"]
    assert rows["O3-6"]["stage_ready"] is True, rows["O3-6"]
    assert rows["O3-7"]["stage_ready"] is True, rows["O3-7"]
    assert rows["O3-8"]["stage_ready"] is True, rows["O3-8"]
    assert rows["O3-9"]["stage_ready"] is False, rows["O3-9"]
    assert "adaptive_lr_live_tensor_binding_probe_failed" not in report["blocked_reasons"], report
    assert "adaptive_lr_runtime_dispatch_shadow_row_not_ready" not in report["blocked_reasons"], report
    assert "adaptive_lr_product_exposure_gate_open" in report["promotion_blockers"], report
    assert artifact_path.exists(), artifact_path

    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_chain_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
