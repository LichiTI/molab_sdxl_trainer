"""Smoke checks for P6K native data pipeline end-to-end shadow."""

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

from core.turbocore_native_data_pipeline_e2e_shadow_scorecard import (  # noqa: E402
    build_native_data_pipeline_e2e_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_native_data_pipeline_e2e_shadow_scorecard()
    shadow = report["shadow_case"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["e2e_shadow_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_shadow_updates_original"] is False, report
    assert shadow["status"] == "passed", shadow
    assert shadow["batch_descriptor_parity_ok"] is True, shadow
    assert shadow["batch_tensor_parity_ok"] is True, shadow
    assert shadow["loss_parity_ok"] is True, shadow
    assert shadow["reference_unchanged_after_shadow_mutation"] is True, shadow
    assert shadow["native_shadow_mutated_clone_only"] is True, shadow
    return {
        "schema_version": 1,
        "probe": "turbocore_native_data_pipeline_e2e_shadow_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
