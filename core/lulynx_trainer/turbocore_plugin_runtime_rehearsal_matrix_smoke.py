"""Aggregated smoke for selected plugin runtime/precondition rehearsal coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard import (  # noqa: E402
    build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard import (  # noqa: E402
    build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_fused_backward_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    build_plugin_fused_backward_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard import (  # noqa: E402
    build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard import (  # noqa: E402
    build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    build_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard,
)


Builder = Callable[..., dict[str, Any]]

RUNTIME_DISPATCH_FAMILIES: tuple[tuple[str, Builder], ...] = (
    ("simple_formula", build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard),
    ("adamlike", build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard),
    ("schedulefree", build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard),
    ("adaptivelr", build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard),
)
RUNTIME_PRECONDITION_FAMILIES: tuple[tuple[str, Builder], ...] = (
    ("factored_memory", build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard),
    ("closure_second_order", build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard),
    ("custom_formula", build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard),
    ("model_shape_aware", build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard),
    ("state_adapter_special", build_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard),
    ("fused_backward", build_plugin_fused_backward_runtime_precondition_rehearsal_scorecard),
)
EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT = 124
EXPECTED_RUNTIME_DISPATCH_READY_COUNT = 52
EXPECTED_RUNTIME_PRECONDITION_READY_COUNT = 72


def run_smoke() -> dict[str, Any]:
    runtime_reports = [_run_family(name, builder, readiness_key="runtime_dispatch_rehearsal_ready_count") for name, builder in RUNTIME_DISPATCH_FAMILIES]
    precondition_reports = [
        _run_family(name, builder, readiness_key="runtime_precondition_rehearsal_ready_count")
        for name, builder in RUNTIME_PRECONDITION_FAMILIES
    ]
    runtime_ready = sum(item["readiness_count"] for item in runtime_reports)
    precondition_ready = sum(item["readiness_count"] for item in precondition_reports)
    adapter_ready = sum(item["adapter_ready_count"] for item in precondition_reports)
    total_cases = sum(item["case_count"] for item in runtime_reports + precondition_reports)
    assert runtime_ready == EXPECTED_RUNTIME_DISPATCH_READY_COUNT, runtime_reports
    assert precondition_ready == EXPECTED_RUNTIME_PRECONDITION_READY_COUNT, precondition_reports
    assert adapter_ready == EXPECTED_RUNTIME_PRECONDITION_READY_COUNT, precondition_reports
    assert total_cases == EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT, runtime_reports + precondition_reports
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_runtime_rehearsal_matrix_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "runtime_dispatch_ready_count": runtime_ready,
        "runtime_precondition_ready_count": precondition_ready,
        "family_specific_runtime_launch_adapter_ready_count": adapter_ready,
        "selected_plugin_optimizer_case_count": total_cases,
        "families": runtime_reports + precondition_reports,
        "notes": [
            "This aggregated smoke is the preferred daily entrypoint for selected plugin runtime/precondition coverage.",
            "Per-family smokes remain available for failure localization.",
        ],
    }


def _run_family(name: str, builder: Builder, *, readiness_key: str) -> dict[str, Any]:
    if name in {
        "closure_second_order",
        "factored_memory",
        "custom_formula",
        "fused_backward",
        "model_shape_aware",
        "state_adapter_special",
    }:
        report = builder(write_artifact=True, include_representative_runtime_canary=True)
    else:
        report = builder(write_artifact=True)
    summary = report["summary"]
    case_count = int(summary.get("case_count", summary.get("selected_optimizer_count", 0)) or 0)
    readiness_count = int(summary.get(readiness_key, 0) or 0)
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert int(summary.get("native_dispatch_allowed_count", 0) or 0) == 0, summary
    assert int(summary.get("training_path_enabled_count", 0) or 0) == 0, summary
    assert int(summary.get("product_native_ready_count", 0) or 0) == 0, summary
    assert readiness_count == case_count, summary
    return {
        "family": name,
        "scorecard": str(report.get("scorecard", "")),
        "case_count": case_count,
        "readiness_count": readiness_count,
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "representative_runtime_dispatch_rehearsal_ready_count": int(
            summary.get("representative_runtime_dispatch_rehearsal_ready_count", 0) or 0
        ),
        "adapter_ready_count": int(
            summary.get("family_specific_runtime_launch_adapter_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
