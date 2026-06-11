"""Smoke for the artifact-first TurboCore optimizer native readiness gap report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_native_readiness_gap_scorecard import (  # noqa: E402
    build_optimizer_native_readiness_gap_scorecard,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_native_readiness_gap_scorecard(write_artifact=True)
    summary = _as_dict(report.get("summary"))
    rows = report.get("rows")

    assert report.get("ok") is True, report
    assert report.get("roadmap") == ROADMAP, report
    assert report.get("artifact_first") is True, report
    assert report.get("promotion_ready") is False, report
    assert report.get("cuda_executed") is False, report
    assert report.get("runtime_dispatch_ready") is False, report
    assert report.get("native_dispatch_allowed") is False, report
    assert report.get("training_path_enabled") is False, report
    assert report.get("product_native_ready") is False, report
    assert isinstance(rows, list), report
    assert summary.get("route_family_count") == 10, report
    assert summary.get("plugin_optimizer_count") == 124, report
    assert summary.get("selected_optimizer_gate_ready_family_count") == 10, report
    assert summary.get("kernel_source_ready_optimizer_count") == 124, report
    assert summary.get("rust_probe_ready_optimizer_count") == 124, report
    assert summary.get("family_contract_ready_count") == 10, report
    assert summary.get("family_evidence_ready_count") == 10, report
    assert summary.get("runtime_rehearsal_ready_family_count") == 4, report
    assert summary.get("runtime_precondition_ready_family_count") == 6, report
    assert summary.get("family_specific_runtime_launch_adapter_ready_family_count") == 6, report
    assert summary.get("family_specific_runtime_launch_adapter_ready_optimizer_count") == 72, report
    assert summary.get("runtime_launch_coverage_ready_family_count") == 10, report
    assert summary.get("owner_release_hold_ready_family_count") == 10, report
    assert summary.get("request_schema_ui_non_exposure_ready_family_count") == 10, report
    assert summary.get("runtime_dispatch_ready_family_count") == 0, report
    assert summary.get("native_dispatch_allowed_family_count") == 0, report
    assert summary.get("training_path_enabled_family_count") == 0, report
    assert summary.get("product_native_ready_family_count") == 0, report
    assert summary.get("family_specific_runtime_launch_missing_count") == 0, report
    assert summary.get("product_training_route_missing_count") == 10, report
    assert summary.get("owner_release_approval_missing_count") == 10, report
    for row in rows:
        assert isinstance(row, dict), row
        assert row.get("family_evidence_ready") is True, row
        assert row.get("runtime_rehearsal_mode") in {"dispatch", "precondition"}, row
        if row.get("runtime_rehearsal_mode") == "precondition":
            assert row.get("family_specific_runtime_launch_adapter_ready") is True, row
        assert row.get("owner_release_hold_ready") is True, row
        assert row.get("request_schema_ui_non_exposure_ready") is True, row
        assert row.get("runtime_dispatch_ready") is False, row
        assert row.get("native_dispatch_allowed") is False, row
        assert row.get("training_path_enabled") is False, row
        assert row.get("product_native_ready") is False, row

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_native_readiness_gap_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "artifact_mode": "artifact_first",
        "summary": summary,
        "recommended_next_step": report.get("recommended_next_step", ""),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
