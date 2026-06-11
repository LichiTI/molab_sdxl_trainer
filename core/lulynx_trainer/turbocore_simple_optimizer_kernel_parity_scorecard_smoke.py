"""Smoke checks for V2-P7 simple optimizer CUDA kernel parity scorecard."""

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

from core.turbocore_simple_optimizer_kernel_parity_scorecard import (  # noqa: E402
    build_simple_optimizer_kernel_parity_scorecard,
)


def run_smoke() -> dict[str, Any]:
    payload = build_simple_optimizer_kernel_parity_scorecard(workspace_root=REPO_ROOT)
    assert payload["scorecard"] == "turbocore_simple_optimizer_kernel_parity_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["lion_native_kernel_parity"] is True, payload
    assert payload["sgd_nesterov_native_kernel_parity"] is True, payload
    assert payload["kernel_parity_stage_ready"] is True, payload
    lion = next(case for case in payload["cases"] if case["optimizer_kind"] == "lion")
    assert lion["kernel_executed"] is True, lion
    assert lion["native_kernel_parity_ready"] is True, lion
    assert lion["training_path_enabled"] is False, lion
    assert lion["max_abs_diff"] <= lion["tolerance"], lion
    sgd = next(case for case in payload["cases"] if case["optimizer_kind"] == "sgd_nesterov")
    assert sgd["kernel_executed"] is True, sgd
    assert sgd["native_kernel_parity_ready"] is True, sgd
    assert sgd["training_path_enabled"] is False, sgd
    assert sgd["max_abs_diff"] <= sgd["tolerance"], sgd
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_kernel_parity_scorecard_smoke",
        "ok": True,
        "summary": payload["summary"],
        "promotion_blockers": payload["promotion_blockers"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
