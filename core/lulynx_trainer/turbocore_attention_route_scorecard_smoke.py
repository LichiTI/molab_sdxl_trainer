"""Smoke checks for P6 attention route scorecard."""

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

from core.turbocore_attention_route_scorecard import build_attention_route_scorecard  # noqa: E402


def run_smoke() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = build_attention_route_scorecard(
        device=device,
        shape=(1, 4, 128, 32),
        warmup=1,
        iterations=3,
    )
    assert report["ok"] is True, report
    assert report["parity"]["parity_ok"] is True, report
    assert report["backward"]["backward_parity_ok"] is True, report
    assert report["runtime_profile"]["profile_active"] is True, report
    if device.type == "cuda":
        assert report["benchmark"]["ok"] is True, report
        assert float(report["benchmark"]["sdpa_step_ms"]) > 0.0, report
        assert float(report["benchmark"]["torch_attention_step_ms"]) > 0.0, report
    else:
        assert "cuda_required_for_attention_route_performance" in report["blocked_reasons"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_attention_route_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "scorecard": report,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
