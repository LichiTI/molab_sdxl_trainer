"""Smoke checks for schedule-free plugin native ABI sketch scorecard."""

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

from core.turbocore_plugin_schedulefree_native_abi_sketch_scorecard import (  # noqa: E402
    build_plugin_schedulefree_native_abi_sketch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_native_abi_sketch_scorecard()
    launch = report["launch_plan"]
    adapter = report["checkpoint_adapter_contract"]
    fallback = report["fallback_authority"]
    policy = report["dispatch_policy"]
    assert report["ok"] is True, report
    assert report["native_abi_sketch_ready"] is True, report
    assert report["launch_plan_contract_ready"] is True, report
    assert report["checkpoint_adapter_contract_ready"] is True, report
    assert report["fallback_authority_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert launch["plan_kind"] == "plugin_schedulefree_selected_optimizer_launch_plan_v0", launch
    assert launch["launch_allowed"] is False, launch
    assert "train_mode" in launch["required_group_state_fields"], launch
    assert "z" in launch["required_param_state_fields"], launch
    assert adapter["checkpoint_adapter_kind"] == "plugin_schedulefree_state_dict_adapter_v0", adapter
    assert adapter["runtime_adapter_enabled"] is False, adapter
    assert fallback["training_update_authority"] == "selected_pytorch_optimizer_plugin", fallback
    assert fallback["native_update_authority"] == "none_until_review", fallback
    assert policy["allowed_initial_modes"] == ["off", "observe"], policy
    assert policy["blocked_modes_until_review"] == ["canary", "auto"], policy
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_native_abi_sketch_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
