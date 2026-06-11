"""Smoke checks for quantized simple TrainingLoop canary manifests."""

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

from core.turbocore_simple_optimizer_quantized_runtime_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_runtime_canary_scorecard,
)
from core.turbocore_simple_optimizer_quantized_training_loop_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    runtime = build_simple_optimizer_quantized_runtime_canary_scorecard()
    assert runtime["runtime_canary_manifest_ready"] is True, runtime
    report = build_simple_optimizer_quantized_training_loop_canary_scorecard(runtime_canary_report=runtime)
    assert report["scorecard"] == "turbocore_simple_optimizer_quantized_training_loop_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_loop_canary_manifest_ready"] is True, report
    assert report["training_loop_canary_ready"] is True, report
    assert report["runtime_canary_manifest_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["summary"]["training_loop_canary_manifest_ready_count"] == 3, report
    assert report["summary"]["training_loop_canary_ready_count"] == 3, report
    assert report["summary"]["executor_implementation_ready_count"] == 3, report
    assert report["summary"]["native_kernel_launch_count"] == 3, report
    for row in report["rows"]:
        assert row["training_loop_canary_manifest_ready"] is True, row
        assert row["training_loop_canary_ready"] is True, row
        assert row["training_loop_executor_ready"] is True, row
        assert row["training_path_enabled"] is False, row
    for case in report["cases"]:
        assert case["pytorch_optimizer_state_synced"] is True, case
        assert case["optimizer_state_sync_synced"] is True, case
        assert case["optimizer_state_sync_state_tensors"] == 2, case
        assert case["optimizer_state_sync_parameter_tensors"] == 1, case
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_training_loop_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
