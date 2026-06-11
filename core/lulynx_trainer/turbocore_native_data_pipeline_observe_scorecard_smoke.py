"""Smoke checks for P6H native data pipeline observe scorecard."""

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

from core.turbocore_native_data_pipeline_observe_scorecard import (  # noqa: E402
    build_native_data_pipeline_observe_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_data_pipeline_observe_scorecard(
        sample_count=64,
        batch_size=4,
        prefetch_depth=8,
        chunk_size=4,
    )
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["observe_manifest_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["route_decision"]["decision"] == (
        "would_select_native_data_pipeline_observe_but_dispatch_disabled"
    ), report

    probes = report["probes"]
    assert probes["workspace_lifecycle"]["native_runtime"] is True, report
    assert probes["workspace_lifecycle"]["ok"] is True, report
    assert probes["shuffled_plan"]["native_runtime"] is True, report
    assert probes["shuffled_plan"]["provider"] == "native_dataset_staging", report
    assert probes["lazy_fast_probe"]["runtime_summary_only"] is True, report
    assert probes["lazy_fast_probe"]["native_index_materialized"] is False, report
    assert probes["descriptor_probe"]["sample_descriptors_owned"] is True, report
    assert probes["descriptor_probe"]["descriptor_parity_ok"] is True, report

    return {
        "schema_version": 1,
        "probe": "turbocore_native_data_pipeline_observe_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
