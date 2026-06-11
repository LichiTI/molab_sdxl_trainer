"""Smoke checks for AdamWScheduleFree state-machine scorecard."""

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

from core.turbocore_adamw_schedule_free_state_machine_scorecard import (  # noqa: E402
    build_adamw_schedule_free_state_machine_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_state_machine_scorecard()
    cases = {str(case["case"]): case for case in report["cases"]}
    assert report["ok"] is True, report
    assert report["state_machine_reference_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert cases["step_requires_train_mode"]["ok"] is True, cases["step_requires_train_mode"]
    assert cases["trainer_request_initializes_train_mode"]["param_group_train_mode"] is True
    roundtrip = cases["roundtrip_state_machine"]
    assert roundtrip["after_step"]["has_required_param_state"] is True, roundtrip
    assert roundtrip["after_eval"]["train_mode"] is False, roundtrip
    assert roundtrip["after_train"]["train_mode"] is True, roundtrip
    assert roundtrip["max_resume_diff"] <= roundtrip["tolerance"], roundtrip
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_state_machine_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
