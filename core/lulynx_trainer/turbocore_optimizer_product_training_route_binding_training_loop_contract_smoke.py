"""Smoke for the TurboCore optimizer TrainingLoop route-binding contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (  # noqa: E402
    build_optimizer_product_training_route_binding_training_loop_contract,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_product_training_route_binding_training_loop_contract(write_artifact=True)
    summary = _as_dict(report.get("summary"))
    assert report["ok"] is True, report
    assert report["roadmap"] == ROADMAP, report
    assert report["current_real_artifact_training_path_enabled"] is False, report
    assert report["post_approval_training_loop_context_ready"] is True, report
    assert report["requires_all_three_switches"] is True, report
    assert report["product_training_route_bound"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["post_training_route_request_fields"] == {}, report
    assert summary["candidate_switch_count"] == 3, report
    assert summary["open_training_path_enabled"] == 1, report
    assert summary["closed_training_path_enabled"] == 1, report
    assert summary["missing_training_path_closes_context_count"] == 1, report
    assert summary["missing_require_native_cuda_closes_context_count"] == 1, report
    assert summary["request_fields_emitted_count"] == 0, report
    assert summary["schema_exposure_allowed_count"] == 0, report
    assert summary["ui_exposure_allowed_count"] == 0, report
    assert report["candidate_runtime_context"]["optimizer_kind"] == "adamw", report
    assert report["candidate_runtime_context"]["require_native_cuda"] is True, report
    assert report["candidate_runtime_context"]["prefer_native_cuda"] is True, report
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_training_loop_contract_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
