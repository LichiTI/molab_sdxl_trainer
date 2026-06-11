"""Smoke checks for V2 end-to-end training performance gate."""

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

from core.native_training_performance_v2_e2e import build_native_training_performance_gate_v2  # noqa: E402


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
    lora_report = {"promotion_ready": True, "ok": True}
    optimizer_report = {"promotion_ready": True, "ok": True}
    data_report = {"promotion_ready": True, "ok": True}
    if not torch.cuda.is_available():
        report = build_native_training_performance_gate_v2(
            lora_report=lora_report,
            optimizer_report=optimizer_report,
            data_report=data_report,
            device=torch.device("cpu"),
        )
        assert report["promotion_ready"] is False, report
        assert "cuda_required_for_e2e_training_performance_gate" in report["blocked_reasons"], report
        return {"schema_version": 1, "probe": "native_training_performance_v2_e2e_smoke", "ok": True, "skipped": True}
    old = _ensure_native_artifact_dir()
    try:
        report = build_native_training_performance_gate_v2(
            lora_report=lora_report,
            optimizer_report=optimizer_report,
            data_report=data_report,
            device=torch.device("cuda"),
        )
    finally:
        _restore_env(old)
    assert report["ok"] is True, report
    assert "performance_report" in report, report
    performance = report["performance_report"]
    assert performance["baseline_step_ms"] > 0, performance
    assert performance["native_step_ms"] > 0, performance
    assert performance["native_route_hit_count"] >= 1, performance
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v2_e2e_smoke",
        "ok": True,
        "scorecard": report,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
