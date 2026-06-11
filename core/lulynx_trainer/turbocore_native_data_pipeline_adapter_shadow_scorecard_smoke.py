"""Smoke checks for P6I native data pipeline adapter shadow."""

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

from core.turbocore_native_data_pipeline_adapter_shadow_scorecard import (  # noqa: E402
    build_native_data_pipeline_adapter_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_data_pipeline_adapter_shadow_scorecard(
        sample_count=64,
        batch_size=4,
        prefetch_depth=8,
        chunk_size=4,
    )
    route = report["adapter_route"]
    envelope = report["adapter_envelope"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["adapter_shadow_ready"] is True, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_call_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert route["decision"] == "shadow_adapter_prepared_fallback_authoritative", route
    assert route["fallback_backend"] == "standardcore_python_data_path", route
    assert envelope["training_data_authority"] == "standardcore_python_data_path", envelope
    assert envelope["native_data_authority"] == "none", envelope
    assert "dataset_semantic_parity_matrix" in envelope["required_evidence"], envelope
    return {
        "schema_version": 1,
        "probe": "turbocore_native_data_pipeline_adapter_shadow_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
