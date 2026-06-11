"""Smoke checks for PagedAdamW8bit quantized update parity."""

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

from core.turbocore_paged_adamw8bit_quantized_update_scorecard import (  # noqa: E402
    SCRATCH_UPDATE_KIND,
    build_paged_adamw8bit_quantized_update_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_quantized_update_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["quantized_update_contract_ready"] is True, report
    assert report["scratch_update_kind"] == SCRATCH_UPDATE_KIND, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    contract = report["scratch_update_contract"]
    assert contract["functional_oracle"] == "bitsandbytes.functional.optimizer_update_8bit_blockwise", contract
    assert contract["runtime_policy"]["native_kernel_implemented"] is False, contract
    live = report["live_probe"]
    if torch.cuda.is_available():
        assert live["status"] == "passed", live
        assert live["bnb_oracle_parity_passed"] is True, live
        assert live["max_param_diff"] == 0.0, live
        assert live["max_state_float_diff"] == 0.0, live
        assert live["state_uint8_mismatch_count"] == 0, live
    else:
        assert live["status"] == "skipped", live
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_quantized_update_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
