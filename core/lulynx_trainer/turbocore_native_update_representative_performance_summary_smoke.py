"""Smoke for native-update representative performance evidence summary."""

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

from core.turbocore_native_update_representative_performance_summary import (  # noqa: E402
    build_native_update_representative_performance_summary,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_update_representative_performance_summary(write_artifact=True)
    summary = report["summary"]
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design.md", report
    assert report["ok"] is True, report
    assert report["native_launch_evidence_ready"] is True, report
    if report["performance_artifact_present"]:
        assert report["representative_performance_gate_ready"] is True, report
        assert report["release_performance_evidence_complete"] is True, report
        assert report["blocked_reasons"] == [], report
        assert report["standard_artifact_import"] is True, report
        assert report["fresh_live_run"] is False, report
        assert report["source_evidence_quality"], report
    else:
        assert report["representative_performance_gate_ready"] is False, report
        assert report["release_performance_evidence_complete"] is False, report
        assert "native_update_performance_artifact_missing" in report["blocked_reasons"], report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["product_exposure_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert summary["native_launch_evidence_ready_count"] == 1, report
    assert summary["native_kernel_launch_count"] == 2, report
    assert summary["training_parameter_mutation_count"] == 2, report
    assert summary["performance_artifact_present_count"] in (0, 1), report
    assert summary["representative_performance_gate_ready_count"] in (0, 1), report
    assert summary["top_level_native_dispatch_allowed_count"] == 0, report
    assert summary["top_level_training_path_enabled_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_representative_performance_summary_smoke",
        "ok": True,
        "roadmap": report["roadmap"],
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
