"""Matrix smoke for selected plugin-family default-off gates."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

BuilderSpec = tuple[str, str]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_plugin_selected_default_off_matrix_scorecard.json"
)


CASES: tuple[dict[str, Any], ...] = (
    {
        "family": "adamlike",
        "optimizer_count": 25,
        "owner_builder": (
            "core.turbocore_plugin_adamlike_owner_release_hold_scorecard",
            "build_plugin_adamlike_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_adamlike_request_schema_ui_non_exposure_scorecard",
            "build_plugin_adamlike_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "adaptivelr",
        "optimizer_count": 6,
        "owner_builder": (
            "core.turbocore_plugin_adaptivelr_owner_release_hold_scorecard",
            "build_plugin_adaptivelr_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard",
            "build_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "closure_second_order",
        "optimizer_count": 5,
        "owner_builder": (
            "core.turbocore_plugin_closure_second_order_owner_release_hold_scorecard",
            "build_plugin_closure_second_order_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard",
            "build_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "custom_formula",
        "optimizer_count": 47,
        "owner_builder": (
            "core.turbocore_plugin_custom_formula_owner_release_hold_scorecard",
            "build_plugin_custom_formula_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard",
            "build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "factored_memory",
        "optimizer_count": 8,
        "owner_builder": (
            "core.turbocore_plugin_factored_memory_owner_release_hold_scorecard",
            "build_plugin_factored_memory_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_factored_memory_request_schema_ui_non_exposure_scorecard",
            "build_plugin_factored_memory_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "fused_backward",
        "optimizer_count": 2,
        "owner_builder": (
            "core.turbocore_plugin_fused_backward_owner_release_hold_scorecard",
            "build_plugin_fused_backward_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_fused_backward_request_schema_ui_non_exposure_scorecard",
            "build_plugin_fused_backward_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "model_shape_aware",
        "optimizer_count": 7,
        "owner_builder": (
            "core.turbocore_plugin_model_shape_aware_owner_release_hold_scorecard",
            "build_plugin_model_shape_aware_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard",
            "build_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "state_adapter_special",
        "optimizer_count": 3,
        "owner_builder": (
            "core.turbocore_plugin_state_adapter_special_owner_release_hold_scorecard",
            "build_plugin_state_adapter_special_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard",
            "build_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "schedulefree",
        "optimizer_count": 3,
        "owner_builder": (
            "core.turbocore_plugin_schedulefree_owner_release_hold_scorecard",
            "build_plugin_schedulefree_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_schedulefree_request_schema_ui_non_exposure_scorecard",
            "build_plugin_schedulefree_request_schema_ui_non_exposure_scorecard",
        ),
    },
    {
        "family": "simple_formula",
        "optimizer_count": 18,
        "owner_builder": (
            "core.turbocore_plugin_simple_formula_owner_release_hold_scorecard",
            "build_plugin_simple_formula_owner_release_hold_scorecard",
        ),
        "exposure_builder": (
            "core.turbocore_plugin_simple_formula_request_schema_ui_non_exposure_scorecard",
            "build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard",
        ),
    },
)


def run_smoke(*, rebuild_artifact: bool = False) -> dict[str, Any]:
    if not rebuild_artifact and ARTIFACT.exists():
        try:
            report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
            if isinstance(report, dict):
                return _validate_matrix_report(report, artifact_mode="artifact_first")
        except (OSError, json.JSONDecodeError):
            pass

    rows = [_check_case(case) for case in CASES]
    payload = {
        "schema_version": 1,
        "probe": "turbocore_plugin_selected_default_off_matrix_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "artifact_mode": "rebuild",
        "case_count": len(rows),
        "optimizer_count": sum(int(row["optimizer_count"]) for row in rows),
        "rows": rows,
    }
    _write_artifact(payload)
    return _validate_matrix_report(payload, artifact_mode="rebuild")


def _validate_matrix_report(report: dict[str, Any], *, artifact_mode: str) -> dict[str, Any]:
    rows = report.get("rows")
    assert report.get("ok") is True, report
    assert report.get("roadmap") == ROADMAP, report
    assert int(report.get("case_count", 0) or 0) == len(CASES), report
    assert int(report.get("optimizer_count", 0) or 0) == sum(int(case["optimizer_count"]) for case in CASES), report
    assert isinstance(rows, list), report

    expected = {str(case["family"]): int(case["optimizer_count"]) for case in CASES}
    by_family = {str(row.get("family")): row for row in rows if isinstance(row, dict)}
    assert set(by_family) == set(expected), {"expected": sorted(expected), "actual": sorted(by_family)}
    for family, expected_count in expected.items():
        row = by_family[family]
        assert int(row.get("optimizer_count", 0) or 0) == expected_count, row
        assert row.get("owner_release_hold_ready") is True, row
        assert row.get("request_schema_ui_non_exposure_ready") is True, row
        assert int(row.get("forbidden_token_hit_count", 0) or 0) == 0, row
        assert int(row.get("product_native_ready_count", 0) or 0) == 0, row

    payload = dict(report)
    payload["artifact_mode"] = artifact_mode
    payload["schema_version"] = int(payload.get("schema_version", 1) or 1)
    payload["summary"] = {
        "case_count": len(CASES),
        "optimizer_count": sum(expected.values()),
        "selected_plugin_family_count": len(expected),
        "selected_family_counts": dict(sorted(expected.items())),
        "product_native_ready_count": 0,
        "forbidden_token_hit_count": 0,
    }
    return payload


def _check_case(case: dict[str, Any]) -> dict[str, Any]:
    family = str(case["family"])
    expected_count = int(case["optimizer_count"])
    owner_builder: BuilderSpec = case["owner_builder"]
    exposure_builder: BuilderSpec = case["exposure_builder"]

    owner = _artifact_or_build(
        f"turbocore_plugin_{family}_owner_release_hold_scorecard.json",
        owner_builder,
        write_artifact=True,
    )
    owner_summary = _as_dict(owner.get("summary"))
    assert owner["ok"] is True, owner
    assert owner["owner_release_hold_ready"] is True, owner
    assert owner["promotion_ready"] is False, owner
    assert owner["owner_approval_recorded"] is False, owner
    assert owner["release_approval_recorded"] is False, owner
    _assert_default_off(owner)
    assert int(owner_summary.get("optimizer_count", 0) or 0) == expected_count, owner
    assert int(owner_summary.get("product_native_ready_count", 0) or 0) == 0, owner

    exposure = _artifact_or_build(
        f"turbocore_plugin_{family}_request_schema_ui_non_exposure_scorecard.json",
        exposure_builder,
        owner_release_hold_report=owner,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    exposure_summary = _as_dict(exposure.get("summary"))
    assert exposure["ok"] is True, exposure
    assert exposure["request_schema_ui_non_exposure_ready"] is True, exposure
    assert exposure["owner_release_hold_ready"] is True, exposure
    assert exposure["promotion_ready"] is False, exposure
    assert exposure["request_adapter_enabled"] is False, exposure
    assert exposure["backend_router_registered"] is False, exposure
    _assert_default_off(exposure)
    assert int(exposure_summary.get("optimizer_count", 0) or 0) == expected_count, exposure
    assert int(exposure_summary.get("forbidden_token_hit_count", 0) or 0) == 0, exposure
    assert int(exposure_summary.get("product_native_ready_count", 0) or 0) == 0, exposure
    assert int(exposure_summary.get("present_boundary_path_count", 0) or 0) > 0, exposure
    assert int(exposure_summary.get("scanned_file_count", 0) or 0) > 0, exposure

    return {
        "schema_version": 1,
        "family": family,
        "optimizer_count": expected_count,
        "owner_release_hold_ready": True,
        "request_schema_ui_non_exposure_ready": True,
        "forbidden_token_hit_count": 0,
        "product_native_ready_count": 0,
    }


def _artifact_or_build(
    artifact_name: str,
    builder: BuilderSpec,
    **kwargs: Any,
) -> dict[str, Any]:
    path = REPO_ROOT / "temp" / "turbocore_optimizer" / artifact_name
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(report, dict) and report.get("ok") is True:
                return report
        except (OSError, json.JSONDecodeError):
            pass
    module_name, function_name = builder
    module = importlib.import_module(module_name)
    build = getattr(module, function_name)
    return build(**kwargs)


def _assert_default_off(report: dict[str, Any]) -> None:
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["product_native_ready_count"] == 0, report


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _write_artifact(payload: dict[str, Any]) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-artifact", action="store_true")
    args = parser.parse_args(argv)
    payload = run_smoke(rebuild_artifact=bool(args.rebuild_artifact))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
