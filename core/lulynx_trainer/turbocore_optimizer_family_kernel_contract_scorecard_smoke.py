"""Smoke for the shared TurboCore optimizer family kernel contract entrypoint."""

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

from core.turbocore_optimizer_family_kernel_contract_scorecard import (  # noqa: E402
    REQUIRED_FAMILIES,
    build_optimizer_family_kernel_contract_scorecard,
)


EXPECTED_SELECTED_FAMILIES = {
    "adam_like_formula",
    "adaptive_lr_state_machine",
    "closure_or_second_order",
    "custom_formula",
    "factored_memory_layout",
    "fused_backward",
    "model_or_shape_aware",
    "schedule_free_state_machine",
    "simple_formula",
    "state_adapter_special",
}


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_family_kernel_contract_scorecard(write_artifact=True)
    summary = report["summary"]
    contract_families = {str(item.get("native_route_family")) for item in report["contracts"]}

    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design.md", report
    assert report["ok"] is True, report
    assert report["native_importable"] is True, report
    assert report["entrypoint_present"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["kernel_executed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert summary["entrypoint_present_count"] == 1, report
    assert set(REQUIRED_FAMILIES) == EXPECTED_SELECTED_FAMILIES, report
    assert contract_families == EXPECTED_SELECTED_FAMILIES, report
    assert summary["required_family_count"] == 10, report
    assert summary["required_family_present_count"] == 10, report
    assert summary["optimizer_family_contract_count"] == 10, report
    assert summary["native_payload_contract_count"] == 10, report
    assert summary["validation_ok_count"] == 10, report
    assert summary["kernel_source_ready_count"] == 10, report
    assert summary["native_kernel_present_count"] == 10, report
    assert summary["runtime_dispatch_ready_count"] == 0, report
    assert summary["native_dispatch_allowed_count"] == 0, report
    assert summary["training_path_enabled_count"] == 0, report
    assert summary["kernel_executed_count"] == 0, report
    assert summary["product_native_ready_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_family_kernel_contract_scorecard_smoke",
        "ok": True,
        "roadmap": report["roadmap"],
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
