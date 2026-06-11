"""Smoke checks for simple optimizer variant resume-parity scorecard."""

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

from core.turbocore_simple_optimizer_variant_resume_parity_scorecard import (  # noqa: E402
    build_simple_optimizer_variant_resume_parity_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_variant_resume_parity_scorecard(workspace_root=REPO_ROOT)
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["scorecard"] == "turbocore_simple_optimizer_variant_resume_parity_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["summary"]["target_optimizer_count"] == 5, report
    assert report["summary"]["resume_parity_matrix_implementation_ready_count"] == 5, report
    assert report["summary"]["quantized_resume_parity_ready_count"] == 3, report
    assert report["summary"]["schedule_free_resume_parity_ready_count"] == 2, report
    assert report["summary"]["quantized_resume_case_count"] == 3, report
    assert report["summary"]["schedule_free_resume_case_count"] == 2, report
    assert report["summary"]["product_native_ready_count"] == 0, report
    for optimizer_type in ("Lion8bit", "PagedLion8bit", "SGDNesterov8bit"):
        row = rows[optimizer_type]
        assert row["resume_parity_matrix_implementation_ready"] is True, row
        assert row["state_dict_restore_ready"] is True, row
        assert row["next_step_after_restore_ready"] is True, row
        assert row["training_path_enabled"] is False, row
    for optimizer_type in ("RAdamScheduleFree", "SGDScheduleFree"):
        row = rows[optimizer_type]
        assert row["resume_parity_matrix_implementation_ready"] is True, row
        assert row["source_native_canary_ready"] is True, row
        assert row["native_dispatch_allowed"] is False, row
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_variant_resume_parity_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_variant_resume_parity_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
