"""Smoke checks for P6 checkpoint streaming scorecard."""

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

from core.turbocore_checkpoint_streaming_scorecard import (  # noqa: E402
    build_checkpoint_streaming_scorecard,
)


def run_smoke() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = build_checkpoint_streaming_scorecard(
        device=device,
        shape=(2, 32, 64),
        warmup=1,
        iterations=2,
        pool_gb=0.01,
    )
    assert report["ok"] is True, report
    assert report["parity"]["parity_ok"] is True, report
    assert report["parity"]["finite_gradients"] is True, report
    assert report["runtime_profile"]["requested"] is True, report
    assert report["default_behavior_changed"] is False, report
    if device.type == "cuda":
        assert report["promotion_ready"] is True, report
        assert report["parity"]["offload_operational"] is True, report
        assert report["runtime_profile"]["pinned_async_active"] is True, report
        assert report["benchmark"]["ok"] is True, report
        assert float(report["benchmark"]["pinned_async_step_ms"]) > 0.0, report
    else:
        assert "cuda_required_for_checkpoint_streaming_performance" in report["blocked_reasons"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_checkpoint_streaming_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "scorecard": report,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
