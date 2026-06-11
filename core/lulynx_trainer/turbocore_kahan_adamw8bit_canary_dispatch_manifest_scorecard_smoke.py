"""Smoke checks for KahanAdamW8bit canary dispatch manifest."""

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

from core.turbocore_kahan_adamw8bit_canary_dispatch_manifest_scorecard import (  # noqa: E402
    build_kahan_adamw8bit_canary_dispatch_manifest_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_kahan_adamw8bit_canary_dispatch_manifest_scorecard(
        native_training_mode="canary",
        run_live_probe=True,
    )
    route = report["route_decision"]
    assert report["ok"] is True, report
    assert report["canary_dispatch_manifest_ready"] is True, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["canary_dispatch_armed"] is False, report
    assert route["decision"] == "canary_dispatch_manifest_ready_but_disabled", route
    assert "runtime_dispatch_adapter_shadow" in route["missing_before_dispatch"], route
    return {
        "schema_version": 1,
        "probe": "turbocore_kahan_adamw8bit_canary_dispatch_manifest_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
