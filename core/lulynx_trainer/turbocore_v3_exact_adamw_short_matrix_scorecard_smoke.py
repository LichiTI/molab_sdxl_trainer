"""Smoke checks for the V3 exact AdamW short real-training matrix."""

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

from core.turbocore_v3_exact_adamw_short_matrix_scorecard import (  # noqa: E402
    build_v3_exact_adamw_short_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_exact_adamw_short_matrix_scorecard(steps=4)
    gates = report["progress_gates"]
    comparison = report["comparison"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["short_matrix_ready"] is True, report
    assert gates["rollout_manifest_ready"] is True, gates
    assert gates["baseline_default_off"] is True, gates
    assert gates["canary_native_steps"] is True, gates
    assert gates["fallback_preserved"] is True, gates
    assert gates["state_sync_ready"] is True, gates
    assert gates["final_param_parity"] is True, gates
    assert gates["metrics_recorded"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert comparison["parity_ok"] is True, comparison
    assert report["blocked_reasons"] == [], report
    return {
        "schema_version": 1,
        "probe": "turbocore_v3_exact_adamw_short_matrix_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "comparison": comparison,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
