"""Smoke checks for optimizer multi-tensor native update release-hold wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_native_update_optimizer_multitensor_release_hold_scorecard import (  # noqa: E402
    build_native_update_optimizer_multitensor_release_hold_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_update_optimizer_multitensor_release_hold_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_v0", report
    assert report["gate"] == "native_update_optimizer_multitensor_release_hold", report
    assert report["ok"] is True, report
    assert report["default_off"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["training_dispatch"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_dispatch_executed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["post_release_request_fields"] == {}, report
    assert summary["top_level_training_path_enabled_count"] == 0, summary
    assert summary["top_level_native_dispatch_allowed_count"] == 0, summary
    if not torch.cuda.is_available():
        assert report["evidence_ready"] is False, report
        assert "cuda_required_for_optimizer_multitensor_update" in report["blocked_reasons"], report
        return {
            "schema_version": 1,
            "probe": "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_smoke",
            "ok": True,
            "skipped": True,
            "summary": summary,
        }
    assert report["evidence_ready"] is True, report
    assert report["ready_for_optimizer_multitensor_release_review"] is True, report
    assert summary["multitensor_evidence_ready"] is True, summary
    assert summary["nested_native_step_executed"] is True, summary
    assert summary["nested_training_path_enabled"] is True, summary
    assert summary["tensor_count"] >= 2, summary
    assert summary["dtype_bucket_count"] >= 2, summary
    assert summary["native_kernel_launch_count"] >= 2, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_smoke",
        "ok": True,
        "skipped": False,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
