"""Smoke checks for quantized simple optimizer native scratch kernels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_quantized_native_scratch_scorecard import (  # noqa: E402
    KERNEL_BY_OPTIMIZER,
    build_simple_optimizer_quantized_native_scratch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_quantized_native_scratch_scorecard()
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["scorecard"] == "turbocore_simple_optimizer_quantized_native_scratch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["native_scratch_kernel_parity_ready"] is True, report
    assert report["native_kernel_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["summary"]["target_optimizer_count"] == 3, report
    assert report["summary"]["native_scratch_kernel_ready_count"] == 3, report
    assert report["summary"]["kernel_executed_count"] == 3, report
    assert report["summary"]["parity_ready_count"] == 3, report
    for optimizer, kernel_name in KERNEL_BY_OPTIMIZER.items():
        row = rows[optimizer.value]
        assert row["native_scratch_kernel_parity_ready"] is True, row
        assert row["native_kernel_ready"] is True, row
        assert row["runtime_canary_ready"] is False, row
        assert row["kernel_name"] == kernel_name, row
        assert row["probe"]["kernel_executed"] is True, row
        assert row["probe"]["parity_ok"] is True, row
        assert row["probe"]["state_uint8_mismatch_count"] == 0, row
        assert row["probe"]["training_path_enabled"] is False, row
        assert row["probe"]["training_dispatch"] is False, row
        assert row["probe"]["training_tensor_binding"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_native_scratch_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
