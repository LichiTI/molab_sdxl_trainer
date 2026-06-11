"""Smoke checks for KahanAdamW8bit native scratch-kernel parity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_kahan_adamw8bit_native_scratch_kernel_scorecard import (  # noqa: E402
    KERNEL_NAME,
    build_kahan_adamw8bit_native_scratch_kernel_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_kahan_adamw8bit_native_scratch_kernel_scorecard()
    assert report["ok"] is True, report
    assert report["native_scratch_kernel_parity_ready"] is True, report
    assert report["native_kernel_ready"] is True, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["kernel_name"] == KERNEL_NAME, report
    assert report["summary"]["kernel_executed"] is True, report
    assert report["summary"]["parity_ok"] is True, report
    assert report["summary"]["quantized_state_mismatch_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_kahan_adamw8bit_native_scratch_kernel_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
