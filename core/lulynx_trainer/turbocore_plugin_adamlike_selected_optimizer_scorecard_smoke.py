"""Smoke checks for selected adam-like plugin optimizer scorecard."""

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

from core.turbocore_plugin_adamlike_selected_optimizer_scorecard import (  # noqa: E402
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_adamlike_selected_optimizer_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adamlike_selected_optimizer_scorecard()
    assert report["scorecard"] == "turbocore_plugin_adamlike_selected_optimizer_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_optimizer_abi_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    summary = report["summary"]
    assert summary["case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["passed_case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert "adamw" in summary["compatible_optimizer_names"], summary
    assert summary["adamw_native_route_compatible_count"] >= 1, summary
    assert summary["dedicated_kernel_required_count"] >= 1, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adamlike_selected_optimizer_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
