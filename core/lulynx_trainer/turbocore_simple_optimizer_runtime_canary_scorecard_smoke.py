"""Smoke checks for V2-P7 simple optimizer runtime canary scorecard."""

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

from core.turbocore_simple_optimizer_kernel_parity_scorecard import (  # noqa: E402
    build_simple_optimizer_kernel_parity_scorecard,
)
from core.turbocore_simple_optimizer_runtime_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_runtime_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    kernel = build_simple_optimizer_kernel_parity_scorecard(workspace_root=REPO_ROOT)
    assert kernel["kernel_parity_stage_ready"] is True, kernel
    observe = build_simple_optimizer_runtime_canary_scorecard(
        kernel_parity_report=kernel,
        native_training_mode="observe",
    )
    canary = build_simple_optimizer_runtime_canary_scorecard(
        kernel_parity_report=kernel,
        native_training_mode="canary",
    )
    assert observe["runtime_canary_ready"] is False, observe
    assert observe["would_native_count"] == 2, observe
    assert canary["ok"] is True, canary
    assert canary["runtime_canary_ready"] is True, canary
    assert canary["runtime_canary_hit"] is True, canary
    assert canary["native_route_hit_count"] == 2, canary
    assert canary["training_path_enabled"] is False, canary
    assert canary["native_dispatch_allowed"] is False, canary
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_runtime_canary_scorecard_smoke",
        "ok": True,
        "observe": observe,
        "canary": canary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
