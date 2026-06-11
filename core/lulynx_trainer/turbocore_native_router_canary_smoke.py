"""Smoke checks for V2 native training router canary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_native_router_canary import build_native_training_router_canary  # noqa: E402


def run_smoke() -> dict[str, Any]:
    ready_report = {"promotion_ready": True, "ok": True}
    observe = build_native_training_router_canary(
        lora_report=ready_report,
        optimizer_report=ready_report,
        data_report=ready_report,
        mode="observe",
    )
    auto = build_native_training_router_canary(
        lora_report=ready_report,
        optimizer_report=ready_report,
        data_report=ready_report,
        mode="auto",
    )
    canary = build_native_training_router_canary(
        lora_report=ready_report,
        optimizer_report=ready_report,
        data_report=ready_report,
        mode="canary",
    )
    assert observe["promotion_ready"] is False, observe
    assert observe["native_route_hit_count"] == 0, observe
    assert observe["would_native_count"] >= 1, observe
    assert auto["promotion_ready"] is True, auto
    assert canary["promotion_ready"] is True, canary
    assert canary["native_route_hit_count"] >= 1, canary
    return {
        "schema_version": 1,
        "probe": "turbocore_native_router_canary_smoke",
        "ok": True,
        "observe": observe,
        "auto": auto,
        "canary": canary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
