"""Smoke checks for plugin factored-memory state-layout scorecard."""

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

from core.turbocore_plugin_factored_memory_state_layout_scorecard import (  # noqa: E402
    FACTORED_MEMORY_PLUGIN_OPTIMIZERS,
    build_plugin_factored_memory_state_layout_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_factored_memory_state_layout_scorecard()
    assert report["scorecard"] == "turbocore_plugin_factored_memory_state_layout_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["selected_native_layout_abi_ready"] is True, report
    assert report["layout_quality_matrix_ready"] is True, report
    assert report["native_kernel_entry_conditions_ready"] is True, report
    assert report["selected_optimizer_abi_ready"] is False, report
    assert report["default_off_contract"]["native_ready_count"] == 0, report
    assert report["default_off_contract"]["plugin_selected_native_ready_count"] == 0, report
    summary = report["summary"]
    assert summary["case_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["native_layout_abi_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["layout_quality_matrix_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["native_kernel_entry_condition_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["plugin_selected_native_ready_count"] == 0, summary
    statuses = {row["optimizer_name"]: row["state_layout_status"] for row in report["rows"]}
    assert set(statuses) == set(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), statuses
    allowed = {"observed_resume_layout", "manual_contract_pending"}
    assert set(statuses.values()).issubset(allowed), statuses
    for row in report["rows"]:
        assert row["native_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["default_behavior_changed"] is False, row
        if row["state_layout_status"] == "observed_resume_layout":
            assert row["covers_resume"] is True, row
            assert row["after_step"]["state_present"] is True, row
            assert row["native_layout_abi_ready"] is True, row
            assert row["layout_quality_matrix_ready"] is True, row
            assert row["native_kernel_entry_condition_ready"] is True, row
            assert row["native_layout_abi"]["state_keys"], row
            assert row["layout_quality_matrix"]["ready"] is True, row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_factored_memory_state_layout_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "statuses": statuses,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
