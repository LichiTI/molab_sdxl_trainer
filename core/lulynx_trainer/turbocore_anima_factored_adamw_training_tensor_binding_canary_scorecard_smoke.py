"""Smoke checks for AnimaFactoredAdamW training tensor binding canary."""

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

from core.turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_anima_factored_adamw_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_anima_factored_adamw_training_tensor_binding_canary_scorecard(workspace_root=REPO_ROOT)
    assert report["scorecard"] == "turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_tensor_binding_canary_ready"] is True, report
    assert report["training_tensor_binding_parity_ready"] is True, report
    probe = report["live_probe"]
    assert probe["passed_case_count"] >= 2, probe
    assert probe["kernel_executed_case_count"] >= 2, probe
    assert probe["training_tensor_binding_parity_passed"] is True, probe
    assert probe["training_dispatch"] is False, probe
    assert probe["training_path_enabled"] is False, probe
    names = {item["case"] for item in probe["cases"]}
    assert {"factored_256x256", "unfactored_4x4"}.issubset(names), probe
    return {
        "schema_version": 1,
        "probe": "turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
