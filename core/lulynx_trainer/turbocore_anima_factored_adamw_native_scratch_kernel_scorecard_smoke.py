"""Smoke checks for AnimaFactoredAdamW CUDA scratch-kernel parity scorecard."""

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

from core.turbocore_anima_factored_adamw_native_scratch_kernel_scorecard import (  # noqa: E402
    build_anima_factored_adamw_native_scratch_kernel_scorecard,
)


def run_smoke() -> dict[str, Any]:
    payload = build_anima_factored_adamw_native_scratch_kernel_scorecard(workspace_root=REPO_ROOT)
    assert payload["scorecard"] == "turbocore_anima_factored_adamw_native_scratch_kernel_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["anima_factored_adamw_native_kernel_parity"] is True, payload
    assert payload["kernel_parity_stage_ready"] is True, payload
    case = payload["case"]
    assert case["kernel_executed"] is True, case
    assert case["native_kernel_parity_ready"] is True, case
    assert case["case_count"] >= 2, case
    assert case["passed_case_count"] == case["case_count"], case
    assert case["training_path_enabled"] is False, case
    assert case["max_abs_diff"] <= case["tolerance"], case
    names = {item["case"] for item in case["cases"]}
    assert {"factored_256x256", "unfactored_4x4"}.issubset(names), case
    return {
        "schema_version": 1,
        "probe": "turbocore_anima_factored_adamw_native_scratch_kernel_scorecard_smoke",
        "ok": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
