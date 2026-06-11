"""Smoke checks for AdamWScheduleFree native ABI report-only gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
REPORT_PATH = REPO_ROOT / "temp" / "turbocore_optimizer" / (
    "turbocore_adamw_schedule_free_native_abi_scorecard.json"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_adamw_schedule_free_native_abi_scorecard import (  # noqa: E402
    build_adamw_schedule_free_native_abi_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_native_abi_scorecard()
    roles = {str(item["role"]): item for item in report["buffer_contract"]}
    mode_roles = {str(item["role"]): item for item in report["mode_contract"]}

    assert report["native_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert mode_roles["train_mode"]["source"] == "param_group.train_mode", mode_roles
    assert roles["z"]["source"] == "optimizer_state.z", roles["z"]
    assert roles["exp_avg_sq"]["source"] == "optimizer_state.exp_avg_sq", roles["exp_avg_sq"]
    assert roles["step"]["source"] == "param_group.k", roles["step"]
    for role in ("lr", "warmup_steps", "r", "weight_sum", "weight_lr_power"):
        assert role in roles, roles

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_native_abi_scorecard_smoke",
        "ok": bool(report["ok"]),
        "report_path": str(REPORT_PATH),
        "summary": report["summary"],
        "blocked_reasons": report["blocked_reasons"],
        "promotion_blockers": report["promotion_blockers"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
