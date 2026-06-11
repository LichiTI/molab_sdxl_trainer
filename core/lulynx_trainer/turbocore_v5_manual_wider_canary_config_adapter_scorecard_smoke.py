"""Smoke checks for V5 manual wider-canary config adapter."""

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

from core.turbocore_v5_manual_wider_canary_config_adapter_scorecard import (  # noqa: E402
    build_v5_manual_wider_canary_config_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v5_manual_wider_canary_config_adapter_scorecard()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["config_adapter_ready"] is True, report
    assert gates["default_off"] is True, gates
    assert gates["manual_wider_scope_without_review_blocked"] is True, gates
    assert gates["approved_manual_wider_canary_enables_existing_fields"] is True, gates
    assert gates["non_exact_optimizer_blocked"] is True, gates
    assert gates["unsupported_backend_blocked"] is True, gates
    assert gates["unsupported_scope_blocked"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["blocked_reasons"] == [], report
    approved = report["adapter_cases"]["approved_manual_wider_canary"]["resolved_fields"]
    assert approved["turbocore_native_update_defer_state_sync"] is True, approved
    missing = report["adapter_cases"]["scope_without_review_blocked"]
    assert "v5_p5_manual_wider_canary_review_evidence_missing" in (
        missing["adapter_report"]["blocked_reasons"]
    ), missing
    return {
        "schema_version": 1,
        "probe": "turbocore_v5_manual_wider_canary_config_adapter_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
