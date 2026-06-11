"""Smoke checks for Muon e2e shadow matrix evidence."""

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

from core.turbocore_muon_e2e_shadow_matrix_scorecard import (  # noqa: E402
    MATRIX_SHAPES,
    build_muon_e2e_shadow_matrix_scorecard,
)
from core.turbocore_muon_training_loop_canary_scorecard import (  # noqa: E402
    build_muon_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    loop = build_muon_training_loop_canary_scorecard(write_artifact=True)
    payload = build_muon_e2e_shadow_matrix_scorecard(
        training_loop_canary_report=loop,
        write_artifact=True,
    )
    summary = payload["summary"]
    row = payload["rows"][0]

    assert payload["scorecard"] == "turbocore_muon_e2e_shadow_matrix_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["report_only"] is True, payload
    assert payload["e2e_shadow_matrix_ready"] is True, payload
    assert payload["e2e_shadow_matrix_passed"] is False, payload
    assert payload["live_shadow_matrix_executed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["product_native_ready"] is False, payload

    assert row["optimizer_type"] == "Muon", row
    assert row["e2e_shadow_matrix_ready"] is True, row
    assert row["training_loop_canary_ready"] is True, row
    assert row["fallback_backend_authoritative"] is True, row
    assert row["training_path_enabled"] is False, row
    assert row["runtime_dispatch_ready"] is False, row
    assert row["native_dispatch_allowed"] is False, row
    assert row["product_native_ready"] is False, row

    assert summary["optimizer_count"] == 1, summary
    assert summary["case_count"] == len(MATRIX_SHAPES), summary
    assert summary["report_only_case_count"] == len(MATRIX_SHAPES), summary
    assert summary["failed_case_count"] == 0, summary
    assert summary["e2e_shadow_matrix_ready_count"] == 1, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary

    for case in payload["matrix_cases"]:
        assert case["status"] == "report_only", case
        assert case["shadow_matrix_case_ready"] is True, case
        assert case["native_call_performed"] is False, case
        assert case["kernel_executed"] is False, case
        assert case["training_path_enabled"] is False, case
        assert case["native_dispatch_allowed"] is False, case

    return {
        "schema_version": 1,
        "probe": "turbocore_muon_e2e_shadow_matrix_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
