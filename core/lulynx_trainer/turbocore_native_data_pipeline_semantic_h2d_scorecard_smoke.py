"""Smoke checks for P6J native data pipeline semantic/H2D matrix."""

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

from core.turbocore_native_data_pipeline_semantic_h2d_scorecard import (  # noqa: E402
    build_native_data_pipeline_semantic_h2d_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_data_pipeline_semantic_h2d_scorecard()
    semantic = report["semantic_matrix"]
    descriptor = report["descriptor_parity"]
    h2d = report["h2d_ownership_contract"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["semantic_h2d_matrix_ready"] is True, report
    assert report["semantic_parity_matrix_ready"] is True, report
    assert report["h2d_ownership_contract_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert semantic["case_count"] == 4, semantic
    assert semantic["failed_case_count"] == 0, semantic
    assert any(row["native_runtime"] for row in semantic["cases"]), semantic
    assert descriptor["descriptor_parity_ok"] is True, descriptor
    assert descriptor["bucket_counts"] == {"512x768": 2, "768x512": 1}, descriptor
    assert h2d["copy_independent"] is True, h2d
    assert h2d["native_pipeline_owns_device_tensor"] is False, h2d
    return {
        "schema_version": 1,
        "probe": "turbocore_native_data_pipeline_semantic_h2d_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
