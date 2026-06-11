"""Smoke checks for selected plugin adaptive-LR runtime dispatch rehearsal."""

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

from core.turbocore_plugin_adaptivelr_family_batch_scorecard import TARGET_PLUGIN_OPTIMIZERS  # noqa: E402
from core.turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {str(row["selected_optimizer_name"]): row for row in report["cases"]}
    assert report["scorecard"] == "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard_v0", report
    assert report["roadmap"] == ROADMAP, report
    assert report["ok"] is True, report
    assert report["runtime_dispatch_rehearsal_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["runtime_dispatch_rehearsal_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["training_executor_called_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_step_count"] == 2, summary
    assert summary["native_kernel_launch_count"] == 2, summary
    assert summary["mapped_selected_native_step_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["mapped_selected_native_kernel_launch_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["representative_family_case_count"] == 2, summary
    assert summary["skip_pytorch_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    for expected in TARGET_PLUGIN_OPTIMIZERS:
        assert rows[expected]["runtime_dispatch_rehearsal_ready"] is True, rows[expected]
        assert rows[expected]["training_path_enabled"] is False, rows[expected]
        assert rows[expected]["native_dispatch_allowed"] is False, rows[expected]
    assert rows["prodigy"]["representative_family"] == "adaptive_lr_prodigy", rows["prodigy"]
    assert rows["dadaptadam"]["representative_family"] == "adaptive_lr_dadapt", rows["dadaptadam"]
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
