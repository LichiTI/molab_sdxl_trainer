"""Smoke checks for the V3 exact AdamW real-canary scorecard."""

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

from core.turbocore_v3_exact_adamw_real_canary_scorecard import (  # noqa: E402
    build_v3_exact_adamw_real_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_exact_adamw_real_canary_scorecard(run_live_training=True)
    summary = report["summary"]
    live = report["live_training_probe"]
    assert report["scorecard"] == "turbocore_v3_exact_adamw_real_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["default_behavior_changed"] is False, report
    assert report["default_training_path_enabled"] is False, report
    assert report["explicit_canary_training_path_enabled"] is True, report
    assert summary["default_off"] is True, report
    assert summary["explicit_opt_in_required"] is True, report
    assert summary["explicit_request_allowed"] is True, report
    assert summary["live_training_native_step"] is True, report
    assert summary["pytorch_fallback_preserved"] is True, report
    assert live["first_step_native"] is False, report
    assert live["native_step_executed"] is True, report
    assert live["native_kernel_launched"] is True, report
    assert live["should_call_pytorch_optimizer_step"] is False, report
    assert live["fallback_to_pytorch_required"] is False, report
    assert live["training_executor_called"] is True, report
    assert live["pytorch_optimizer_state_synced"] is True, report
    assert live["owner_backend"] == "rust_cuda_adamw_v0", report
    assert report["blocked_reasons"] == [], report
    return {
        "schema_version": 1,
        "probe": "turbocore_v3_exact_adamw_real_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "owner_backend": live["owner_backend"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
