"""Smoke checks for V3 exact AdamW canary config adapter."""

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

from core.turbocore_v3_exact_adamw_config_adapter_scorecard import (  # noqa: E402
    build_v3_exact_adamw_config_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_exact_adamw_config_adapter_scorecard()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["config_adapter_ready"] is True, report
    assert gates["p3_runtime_recovery_complete"] is True, gates
    assert gates["default_off"] is True, gates
    assert gates["explicit_exact_adamw_enables_existing_fields"] is True, gates
    assert gates["non_exact_optimizer_blocked"] is True, gates
    assert gates["unsupported_backend_blocked"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["blocked_reasons"] == [], report
    return {
        "schema_version": 1,
        "probe": "turbocore_v3_exact_adamw_config_adapter_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
