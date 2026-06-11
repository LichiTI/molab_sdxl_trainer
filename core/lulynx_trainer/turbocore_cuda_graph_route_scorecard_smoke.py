"""Smoke checks for P6E CUDA graph route scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_cuda_graph_route_scorecard import build_cuda_graph_route_scorecard  # noqa: E402


def run_smoke() -> dict[str, Any]:
    report = build_cuda_graph_route_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["static_contract"]["static_contract_ready"] is True, report
    assert report["static_contract"]["shape_mismatch_blocked"] is True, report
    policy = report["policy"]
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["explicit_opt_in_required"] is True, policy
    if torch.cuda.is_available():
        assert report["promotion_ready"] is True, report
        assert report["live_probe"]["capture_replay_ready"] is True, report
        assert report["live_probe"]["max_replay_diff"] <= report["live_probe"]["tolerance"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_graph_route_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
