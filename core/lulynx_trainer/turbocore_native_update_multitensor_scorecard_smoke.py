"""Smoke checks for V2 native optimizer multi-tensor scorecard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_native_update_multitensor_scorecard import (  # noqa: E402
    build_native_update_multitensor_scorecard,
)


def _ensure_native_artifact_dir() -> dict[str, str | None]:
    old = {"LULYNX_NATIVE_ARTIFACT_DIR": os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR")}
    if not os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR"):
        artifact_dir = REPO_ROOT / "backend" / "native" / "target" / "release"
        if artifact_dir.exists():
            os.environ["LULYNX_NATIVE_ARTIFACT_DIR"] = str(artifact_dir)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_smoke() -> dict[str, Any]:
    first_round = {
        "promotion_ready": True,
        "native_step_executed": True,
        "training_path_enabled": True,
        "performance_gate": {"representative_performance_gate_ready": True},
    }
    if not torch.cuda.is_available():
        report = build_native_update_multitensor_scorecard(first_round_report=first_round, device=torch.device("cpu"))
        assert report["promotion_ready"] is False, report
        assert "cuda_required_for_optimizer_multitensor_update" in report["blocked_reasons"], report
        return {"schema_version": 1, "probe": "turbocore_native_update_multitensor_scorecard_smoke", "ok": True, "skipped": True}
    old = _ensure_native_artifact_dir()
    try:
        report = build_native_update_multitensor_scorecard(first_round_report=first_round, device=torch.device("cuda"))
    finally:
        _restore_env(old)
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["bucketed_launch_plan"] is True, report
    assert report["tensor_count"] >= 2, report
    assert report["dtype_bucket_count"] >= 2, report
    assert report["native_step_executed"] is True, report
    assert report["should_call_pytorch_optimizer_step"] is False, report
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_multitensor_scorecard_smoke",
        "ok": True,
        "scorecard": report,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
