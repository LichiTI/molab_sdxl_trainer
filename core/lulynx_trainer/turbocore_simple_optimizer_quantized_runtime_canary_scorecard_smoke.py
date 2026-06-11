"""Smoke checks for quantized simple optimizer runtime canary manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_quantized_native_scratch_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_native_scratch_scorecard,
)
from core.turbocore_simple_optimizer_quantized_runtime_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_runtime_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    scratch = build_simple_optimizer_quantized_native_scratch_scorecard()
    assert scratch["native_scratch_kernel_parity_ready"] is True, scratch
    observe = build_simple_optimizer_quantized_runtime_canary_scorecard(
        native_scratch_report=scratch,
        native_training_mode="observe",
    )
    canary = build_simple_optimizer_quantized_runtime_canary_scorecard(
        native_scratch_report=scratch,
        native_training_mode="canary",
    )
    assert observe["runtime_canary_manifest_ready"] is True, observe
    assert observe["runtime_canary_ready"] is False, observe
    assert observe["summary"]["would_native_shadow_count"] == 3, observe
    assert canary["ok"] is True, canary
    assert canary["runtime_canary_manifest_ready"] is True, canary
    assert canary["runtime_canary_ready"] is False, canary
    assert canary["runtime_canary_hit"] is False, canary
    assert canary["summary"]["runtime_canary_manifest_ready_count"] == 3, canary
    assert canary["summary"]["native_route_blocked_count"] == 3, canary
    assert canary["training_path_enabled"] is False, canary
    assert canary["native_dispatch_allowed"] is False, canary
    assert canary["runtime_dispatch_ready"] is False, canary
    for row in canary["route_decisions"]:
        assert row["runtime_canary_manifest_ready"] is True, row
        assert row["runtime_canary_ready"] is False, row
        assert row["decision"] == "blocked_before_canary", row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_runtime_canary_scorecard_smoke",
        "ok": True,
        "observe": observe,
        "canary": canary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
