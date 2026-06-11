"""Smoke checks for V4 checkpoint/resume boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v4_checkpoint_resume_scorecard import (  # noqa: E402
    build_v4_checkpoint_resume_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v4_checkpoint_resume_scorecard(
        p1_audit={"milestone_completed": True},
    )
    gates = report["progress_gates"]
    live = report["live_probe"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["checkpoint_resume_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_training_path_enabled"] is False, report
    assert report["default_rollout_allowed"] is False, report
    assert report["auto_rollout_allowed"] is False, report
    assert all(gates.values()), gates
    assert live["checkpoint_metadata_integrated"] is True, live
    assert live["trainer_state_metadata_integrated"] is True, live
    assert live["owner_state_included"] is True, live
    assert live["checkpoint_contract_roundtrip_checked"] is True, live
    assert live["checkpoint_contract_roundtrip_ok"] is True, live
    assert live["restore_loaded"] is True, live
    assert live["restore_compatible"] is True, live
    assert live["owner_state_pending"] is True, live
    assert live["mismatch_loaded"] is True, live
    assert live["mismatch_compatible"] is False, live
    assert live["disabled_checkpoint_default_off"] is True, live
    assert live["training_path_stays_default_off"] is True, live

    blocked = build_v4_checkpoint_resume_scorecard(
        p1_audit={"milestone_completed": False},
    )
    assert blocked["ok"] is False, blocked
    assert "v4_p2_p1_result_ingestion_contract_complete_missing" in blocked["blocked_reasons"], blocked
    assert blocked["training_path_enabled"] is False, blocked

    return {
        "schema_version": 1,
        "probe": "turbocore_v4_checkpoint_resume_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "mismatch_compatible": live["mismatch_compatible"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
