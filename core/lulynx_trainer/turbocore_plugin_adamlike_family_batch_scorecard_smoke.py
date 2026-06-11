"""Smoke checks for selected plugin Adam-like family batch scorecard."""

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

from core.turbocore_plugin_adamlike_family_batch_scorecard import (  # noqa: E402
    build_plugin_adamlike_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adamlike_family_batch_scorecard(include_live_canaries=True, write_artifact=True)
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["selected_adamlike_family_batch_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert summary["target_count"] == 25, report
    assert summary["selected_native_canary_ready_count"] == 25, report
    assert summary["exact_adamw_route_canary_ready_count"] == 1, report
    assert summary["dedicated_route_canary_ready_count"] == 24, report
    assert summary["plugin_selected_native_ready_count"] == 0, report
    assert rows["adamw"]["native_route"] == "rust_cuda_adamw_v0", rows["adamw"]
    assert rows["padam"]["native_route"] == "rust_cuda_plugin_padam_v0", rows["padam"]
    assert rows["radam"]["native_route"] == "rust_cuda_plugin_radam_v0", rows["radam"]
    assert rows["yogi"]["native_route"] == "rust_cuda_plugin_yogi_v0", rows["yogi"]
    assert rows["dualadam"]["native_route"] == "rust_cuda_plugin_dualadam_v0", rows["dualadam"]
    assert rows["exadam"]["native_route"] == "rust_cuda_plugin_exadam_v0", rows["exadam"]
    assert rows["qhadam"]["native_route"] == "rust_cuda_plugin_qhadam_v0", rows["qhadam"]
    assert rows["nadam"]["native_route"] == "rust_cuda_plugin_nadam_v0", rows["nadam"]
    assert rows["grokfastadamw"]["native_route"] == "rust_cuda_plugin_grokfastadamw_v0", rows["grokfastadamw"]
    assert rows["ranger"]["native_route"] == "rust_cuda_plugin_ranger_v0", rows["ranger"]
    assert rows["novograd"]["native_route"] == "rust_cuda_plugin_novograd_v0", rows["novograd"]
    assert rows["ranger21"]["native_route"] == "rust_cuda_plugin_ranger21_v0", rows["ranger21"]
    assert rows["ranger25"]["native_route"] == "rust_cuda_plugin_ranger25_v0", rows["ranger25"]
    assert rows["stableadamw"]["native_route"] == "rust_cuda_plugin_stableadamw_v0", rows["stableadamw"]
    assert rows["adamwsn"]["native_route"] == "rust_cuda_plugin_adamwsn_v0", rows["adamwsn"]
    assert rows["adams"]["native_route"] == "rust_cuda_plugin_adams_v0", rows["adams"]
    assert rows["lamb"]["native_route"] == "rust_cuda_plugin_lamb_v0", rows["lamb"]
    assert rows["fadam"]["native_route"] == "rust_cuda_plugin_fadam_v0", rows["fadam"]
    assert rows["flashadamw"]["native_route"] == "rust_cuda_plugin_flashadamw_v0", rows["flashadamw"]
    for name, row in rows.items():
        assert row["selected_native_canary_ready"] is True, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        if name != "adamw":
            assert all(row["stage_status"].values()), row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adamlike_family_batch_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
