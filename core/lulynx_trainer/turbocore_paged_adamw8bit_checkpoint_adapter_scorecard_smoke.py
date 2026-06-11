"""Smoke checks for PagedAdamW8bit checkpoint adapter proof."""

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

from core.turbocore_paged_adamw8bit_checkpoint_adapter_scorecard import (  # noqa: E402
    CHECKPOINT_QUANT_STATE_KEY,
    build_paged_adamw8bit_checkpoint_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_checkpoint_adapter_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["checkpoint_adapter_proof_ready"] is True, report
    assert report["adapter_implemented"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    contract = report["checkpoint_layout_contract"]
    assert CHECKPOINT_QUANT_STATE_KEY in contract["checkpoint_entry_required_keys"], contract
    assert "state1" in contract["packed_quant_state_required_keys"], contract
    assert "state2" in contract["packed_quant_state_required_keys"], contract
    live = report["live_probe"]
    if torch.cuda.is_available():
        assert live["status"] == "passed", live
        assert live["pack_shadow_roundtrip_passed"] is True, live
        assert live["unpack_restores_live_buffers"] is True, live
        assert live["resume_probe_passed"] is True, live
        assert CHECKPOINT_QUANT_STATE_KEY in live["checkpoint_entry_keys"], live
    else:
        assert live["status"] == "skipped", live
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_checkpoint_adapter_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
