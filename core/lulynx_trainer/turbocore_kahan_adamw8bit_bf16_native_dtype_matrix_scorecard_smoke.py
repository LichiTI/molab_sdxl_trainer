"""Smoke checks for KahanAdamW8bit bf16 native dtype matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard import (  # noqa: E402
    build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["bf16_native_dtype_matrix_ready"] is True, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    cases = {case["dtype"]: case for case in report["matrix_cases"]}
    assert "float32" in cases, cases
    assert "bfloat16" in cases, cases
    if torch.cuda.is_available():
        assert cases["float32"]["status"] == "passed", cases["float32"]
        assert cases["bfloat16"]["status"] == "passed", cases["bfloat16"]
        assert cases["bfloat16"]["kernel_executed"] is True, cases["bfloat16"]
        assert cases["bfloat16"]["quantized_state_mismatch_count"] == 0, cases["bfloat16"]
    return {
        "schema_version": 1,
        "probe": "turbocore_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
