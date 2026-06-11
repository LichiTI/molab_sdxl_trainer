"""Smoke checks for PagedAdamW8bit residency scorecard."""

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

from core.turbocore_paged_adamw8bit_residency_scorecard import (  # noqa: E402
    REQUIRED_LIVE_KEYS,
    build_paged_adamw8bit_residency_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_residency_scorecard(run_live_probe=True, numel=4096)
    assert report["ok"] is True, report
    assert report["residency_contract_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    static = report["static_contract"]
    assert static["residency_policy"]["paged_state"] is True, static
    assert static["residency_policy"]["quantized_state"] is True, static
    live = report["live_probe"]
    if torch.cuda.is_available():
        assert live["status"] == "passed", live
        assert live["observed_required_key_count"] == len(REQUIRED_LIVE_KEYS), live
        assert live["checkpoint_packs_quant_state"] is True, live
        assert live["resume_probe_passed"] is True, live
        tensor_rows = {row["role"]: row for row in live["tensor_metadata"]}
        assert tensor_rows["state1"]["dtype"] == "uint8", tensor_rows
        assert tensor_rows["state2"]["dtype"] == "uint8", tensor_rows
        assert tensor_rows["absmax1"]["dtype"] == "float32", tensor_rows
    else:
        assert live["status"] == "skipped", live
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_residency_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
