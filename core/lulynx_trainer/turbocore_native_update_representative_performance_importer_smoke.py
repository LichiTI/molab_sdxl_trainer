"""Smoke for importing existing representative native-update performance evidence."""

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

from core.turbocore_native_update_representative_performance_importer import (  # noqa: E402
    build_native_update_representative_performance_import,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_update_representative_performance_import(write_artifacts=True)
    summary = report["summary"]
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design.md", report
    assert report["ok"] is True, report
    assert report["representative_performance_gate_ready"] is True, report
    assert report["blocked_reasons"] == [], report
    assert report["source_evidence_quality"] == "existing_imported_owner_review_and_manual_replicate_artifacts", report
    assert report["fresh_live_run"] is False, report
    assert report["promotion_grade_current_run"] is False, report
    assert summary["manual_replicate_run_count"] == 5, report
    assert summary["manual_replicate_ready_run_count"] == 5, report
    assert summary["manual_replicate_steps"] >= 20, report
    assert float(summary["manual_replicate_min_speedup"]) >= 1.03, report
    assert float(summary["optimizer_best_speedup_vs_baseline"]) >= 1.20, report
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_representative_performance_importer_smoke",
        "ok": True,
        "roadmap": report["roadmap"],
        "source_evidence_quality": report["source_evidence_quality"],
        "fresh_live_run": report["fresh_live_run"],
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
