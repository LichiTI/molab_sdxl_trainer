"""Smoke for selected plugin LOMO-family fused-backward native hook canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_lomo_fused_backward_hook_canary_scorecard import (  # noqa: E402
    build_lomo_fused_backward_hook_canary_scorecard,
)
from core.turbocore_plugin_fused_backward_family_batch_scorecard import TARGET_PLUGIN_OPTIMIZERS  # noqa: E402


def run_smoke() -> dict[str, Any]:
    report = build_lomo_fused_backward_hook_canary_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["ok"] is True, report
    assert summary["case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["native_step_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["native_kernel_launch_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["optimizer_step_called_count"] == 0, report
    cases = {str(case["selected_optimizer_name"]): case for case in report["cases"]}
    assert set(cases) == set(TARGET_PLUGIN_OPTIMIZERS), cases
    for name, case in cases.items():
        assert case["ok"] is True, case
        assert case["native_step_executed"] is True, case
        assert case["native_kernel_launched"] is True, case
        assert case["fused_backward_route_executed"] is True, case
        assert case["optimizer_step_called"] is False, case
        assert case["public_optimizer_step_forbidden"] is True, case
        assert case["selected_optimizer_family"] == "fused_backward", case
    return {
        "schema_version": 1,
        "probe": "turbocore_lomo_fused_backward_hook_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
