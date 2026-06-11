"""Smoke checks for schedule-free plugin runtime dispatch shadow."""

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

from core.turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard import (  # noqa: E402
    build_plugin_schedulefree_runtime_dispatch_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_runtime_dispatch_shadow_scorecard(native_training_mode="canary")
    route = report["adapter_route"]
    envelope = report["dispatch_envelope"]
    assert report["ok"] is True, report
    assert report["runtime_dispatch_shadow_ready"] is True, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_call_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert route["decision"] == "blocked_before_native_schedulefree_kernel", route
    assert envelope["training_update_authority"] == "selected_pytorch_optimizer_plugin", envelope
    assert envelope["native_update_authority"] == "none_until_review", envelope
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
