"""Smoke checks for KahanAdamW8bit runtime canary manifest."""

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

from core.turbocore_kahan_adamw8bit_runtime_canary_scorecard import (  # noqa: E402
    MANIFEST_KIND,
    build_kahan_adamw8bit_runtime_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_kahan_adamw8bit_runtime_canary_scorecard(native_training_mode="canary")
    route = report["route_decision"]
    assert report["ok"] is True, report
    assert report["runtime_canary_manifest_ready"] is True, report
    assert report["runtime_canary_ready"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["manifest_kind"] == MANIFEST_KIND, report
    assert route["decision"] == "blocked_before_canary", route
    assert "training_tensor_binding" in route["missing_before_dispatch"], route
    assert "kahan_adamw8bit_training_tensor_binding_missing" in report["promotion_blockers"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_kahan_adamw8bit_runtime_canary_scorecard_smoke",
        "ok": True,
        "manifest_summary": report["manifest_summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
