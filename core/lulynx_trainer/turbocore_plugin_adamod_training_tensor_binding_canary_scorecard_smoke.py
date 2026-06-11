"""Smoke checks for selected plugin adamod training tensor binding canary."""

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

from core.turbocore_plugin_adamod_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_plugin_adamod_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adamod_training_tensor_binding_canary_scorecard()
    assert report["scorecard"] == "turbocore_plugin_adamod_training_tensor_binding_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_tensor_binding_canary_ready"] is True, report
    assert report["training_tensor_binding_parity_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    live = report["live_probe"]
    assert live["kernel_executed"] is True, live
    assert live["training_tensor_binding_parity_passed"] is True, live
    assert live["launch"]["action"] == "tensor_binding_session_cuda_adamod_tensor_probe", live
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adamod_training_tensor_binding_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
