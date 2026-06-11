"""Smoke checks for PagedAdamW8bit native ABI sketch."""

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

from core.turbocore_paged_adamw8bit_native_abi_scorecard import (  # noqa: E402
    build_paged_adamw8bit_native_abi_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_native_abi_scorecard(native_training_mode="observe")
    roles = {str(item["role"]): item for item in report["buffer_contract"]}
    assert report["ok"] is True, report
    assert report["abi_sketch_ready"] is True, report
    assert report["checkpoint_adapter_contract_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["route_decision"]["decision"] == "would_native_shadow_but_blocked", report
    assert roles["state1_uint8"]["dtype"] == "uint8", roles["state1_uint8"]
    assert roles["state2_uint8"]["dtype"] == "uint8", roles["state2_uint8"]
    assert roles["qmap1_fp32"]["mutable"] is False, roles["qmap1_fp32"]
    assert roles["absmax1_fp32"]["mutable"] is True, roles["absmax1_fp32"]
    adapter = report["checkpoint_adapter_contract"]
    assert adapter["adapter_required"] is True, adapter
    assert adapter["adapter_implemented"] is False, adapter
    assert "paged_adamw8bit_checkpoint_adapter_missing" in report["promotion_blockers"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_native_abi_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
