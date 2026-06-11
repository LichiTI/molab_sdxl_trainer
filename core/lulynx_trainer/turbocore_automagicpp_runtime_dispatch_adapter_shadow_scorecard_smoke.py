"""Smoke checks for Automagic++ runtime dispatch adapter shadow."""

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

from core.turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard import (  # noqa: E402
    build_automagicpp_runtime_dispatch_adapter_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_automagicpp_runtime_dispatch_adapter_shadow_scorecard(run_live_probe=False)
    assert report["scorecard"] == "turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["runtime_dispatch_adapter_shadow_ready"] is True, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_call_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    route = report["adapter_route"]
    assert route["decision"] == "shadow_adapter_prepared_fallback_authoritative", route
    return {
        "schema_version": 1,
        "probe": "turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
