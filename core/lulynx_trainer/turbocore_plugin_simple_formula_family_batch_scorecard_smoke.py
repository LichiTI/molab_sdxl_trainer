"""Smoke checks for selected plugin simple-formula family batch scorecard."""

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

from core.turbocore_plugin_simple_formula_family_batch_scorecard import (  # noqa: E402
    build_plugin_simple_formula_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_simple_formula_family_batch_scorecard(workspace_root=REPO_ROOT)
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_plugin_simple_formula_family_batch_scorecard_v0", report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert summary["selected_simple_formula_optimizer_count"] >= 1, summary
    assert summary["selected_plugin_native_canary_ready_count"] == 18, summary
    assert summary["selected_plugin_native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert "lion" in rows, rows
    assert rows["lion"]["simple_formula_reference_canary_candidate"] is True, rows["lion"]
    assert rows["lion"]["selected_plugin_native_canary_ready"] is True, rows["lion"]
    assert rows["sgdw"]["selected_plugin_native_canary_ready"] is True, rows["sgdw"]
    assert rows["sgd"]["selected_plugin_native_canary_ready"] is True, rows["sgd"]
    assert rows["signsgd"]["selected_plugin_native_canary_ready"] is True, rows["signsgd"]
    assert rows["tiger"]["selected_plugin_native_canary_ready"] is True, rows["tiger"]
    assert rows["qhm"]["selected_plugin_native_canary_ready"] is True, rows["qhm"]
    assert rows["accsgd"]["selected_plugin_native_canary_ready"] is True, rows["accsgd"]
    assert rows["fromage"]["selected_plugin_native_canary_ready"] is True, rows["fromage"]
    assert rows["rmsprop"]["selected_plugin_native_canary_ready"] is True, rows["rmsprop"]
    assert rows["lars"]["selected_plugin_native_canary_ready"] is True, rows["lars"]
    assert rows["pid"]["selected_plugin_native_canary_ready"] is True, rows["pid"]
    assert rows["sgdp"]["selected_plugin_native_canary_ready"] is True, rows["sgdp"]
    assert rows["gravity"]["selected_plugin_native_canary_ready"] is True, rows["gravity"]
    assert rows["aggmo"]["selected_plugin_native_canary_ready"] is True, rows["aggmo"]
    assert rows["asgd"]["selected_plugin_native_canary_ready"] is True, rows["asgd"]
    assert rows["madgrad"]["selected_plugin_native_canary_ready"] is True, rows["madgrad"]
    assert rows["nero"]["selected_plugin_native_canary_ready"] is True, rows["nero"]
    assert rows["vsgd"]["selected_plugin_native_canary_ready"] is True, rows["vsgd"]
    for row in rows.values():
        assert row["selected_optimizer_family"] == "simple_formula", row
        assert row["product_native_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row

    unsafe = build_plugin_simple_formula_family_batch_scorecard(
        selector_report=_selector_fixture(runtime_dispatch_ready=True),
        simple_reference_report=_reference_fixture(),
        plugin_lion_canary_report=_plugin_lion_canary_fixture(),
        plugin_sgdw_canary_report=_plugin_sgdw_canary_fixture(),
        plugin_sgd_canary_report=_plugin_sgd_canary_fixture(),
        plugin_signsgd_canary_report=_plugin_signsgd_canary_fixture(),
        plugin_tiger_canary_report=_plugin_tiger_canary_fixture(),
        plugin_qhm_canary_report=_plugin_qhm_canary_fixture(),
        plugin_accsgd_canary_report=_plugin_accsgd_canary_fixture(),
        plugin_fromage_canary_report=_plugin_fromage_canary_fixture(),
        plugin_rmsprop_canary_report=_plugin_rmsprop_canary_fixture(),
        plugin_lars_canary_report=_plugin_lars_canary_fixture(),
        plugin_pid_canary_report=_plugin_pid_canary_fixture(),
        plugin_sgdp_canary_report=_plugin_sgdp_canary_fixture(),
        plugin_gravity_canary_report=_plugin_gravity_canary_fixture(),
        plugin_aggmo_canary_report=_plugin_aggmo_canary_fixture(),
        plugin_asgd_canary_report=_plugin_asgd_canary_fixture(),
        plugin_madgrad_canary_report=_plugin_madgrad_canary_fixture(),
        plugin_nero_canary_report=_plugin_nero_canary_fixture(),
        plugin_vsgd_canary_report=_plugin_vsgd_canary_fixture(),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_simple_formula_family_batch_scorecard_smoke",
        "ok": True,
        "real_report_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _selector_fixture(*, runtime_dispatch_ready: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "selector_fixture",
        "ok": True,
        "plugin_selector_classification_ready": True,
        "selector_boundary_ready": True,
        "all_discovered_plugins_resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": runtime_dispatch_ready,
        "native_dispatch_allowed": False,
        "summary": {
            "plugin_optimizer_count": 2,
            "missing_resume_count": 0,
            "route_family_counts": {"simple_formula": 2},
        },
        "rows": [
            _selector_row("lion"),
            _selector_row("sgd"),
            _selector_row("qhm"),
            _selector_row("accsgd"),
            _selector_row("fromage"),
            _selector_row("rmsprop"),
            _selector_row("lars"),
            _selector_row("pid"),
            _selector_row("sgdp"),
            _selector_row("gravity"),
            _selector_row("aggmo"),
            _selector_row("asgd"),
            _selector_row("madgrad"),
            _selector_row("nero"),
            _selector_row("vsgd"),
        ],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": "simple_formula",
        "resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
    }


def _reference_fixture() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "simple_reference_fixture",
        "ok": True,
        "simple_formula_native_batch_canary_ready": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "summary": {
            "batch_canary_ready_count": 2,
            "product_native_ready_count": 0,
        },
    }


def _plugin_lion_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("lion")


def _plugin_sgdw_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("sgdw")


def _plugin_sgd_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("sgd")


def _plugin_signsgd_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("signsgd")


def _plugin_tiger_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("tiger")


def _plugin_qhm_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("qhm")


def _plugin_accsgd_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("accsgd")


def _plugin_fromage_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("fromage")


def _plugin_rmsprop_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("rmsprop")


def _plugin_lars_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("lars")


def _plugin_pid_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("pid")


def _plugin_sgdp_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("sgdp")


def _plugin_gravity_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("gravity")


def _plugin_aggmo_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("aggmo")


def _plugin_asgd_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("asgd")


def _plugin_madgrad_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("madgrad")


def _plugin_nero_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("nero")


def _plugin_vsgd_canary_fixture() -> dict[str, Any]:
    return _plugin_canary_fixture("vsgd")


def _plugin_canary_fixture(name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": f"plugin_{name}_fixture",
        "ok": True,
        "selected_native_canary_ready": True,
        "selected_optimizer_name": name,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "summary": {
            "native_step_count": 1,
            "native_kernel_launch_count": 1,
        },
        "blocked_reasons": [],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
