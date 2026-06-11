"""Report-only batch scorecard for selected plugin simple-formula routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_lion_training_loop_canary_scorecard import (
    build_plugin_lion_training_loop_canary_scorecard,
)
from core.turbocore_plugin_sgdw_training_loop_canary_scorecard import (
    build_plugin_sgdw_training_loop_canary_scorecard,
)
from core.turbocore_plugin_sgd_training_loop_canary_scorecard import (
    build_plugin_sgd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_signsgd_training_loop_canary_scorecard import (
    build_plugin_signsgd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_tiger_training_loop_canary_scorecard import (
    build_plugin_tiger_training_loop_canary_scorecard,
)
from core.turbocore_plugin_qhm_training_loop_canary_scorecard import (
    build_plugin_qhm_training_loop_canary_scorecard,
)
from core.turbocore_plugin_accsgd_training_loop_canary_scorecard import (
    build_plugin_accsgd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_fromage_training_loop_canary_scorecard import (
    build_plugin_fromage_training_loop_canary_scorecard,
)
from core.turbocore_plugin_rmsprop_training_loop_canary_scorecard import (
    build_plugin_rmsprop_training_loop_canary_scorecard,
)
from core.turbocore_plugin_lars_training_loop_canary_scorecard import (
    build_plugin_lars_training_loop_canary_scorecard,
)
from core.turbocore_plugin_pid_training_loop_canary_scorecard import (
    build_plugin_pid_training_loop_canary_scorecard,
)
from core.turbocore_plugin_sgdp_training_loop_canary_scorecard import (
    build_plugin_sgdp_training_loop_canary_scorecard,
)
from core.turbocore_plugin_gravity_training_loop_canary_scorecard import (
    build_plugin_gravity_training_loop_canary_scorecard,
)
from core.turbocore_plugin_aggmo_training_loop_canary_scorecard import (
    build_plugin_aggmo_training_loop_canary_scorecard,
)
from core.turbocore_plugin_asgd_training_loop_canary_scorecard import (
    build_plugin_asgd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_madgrad_training_loop_canary_scorecard import (
    build_plugin_madgrad_training_loop_canary_scorecard,
)
from core.turbocore_plugin_nero_training_loop_canary_scorecard import (
    build_plugin_nero_training_loop_canary_scorecard,
)
from core.turbocore_plugin_vsgd_training_loop_canary_scorecard import (
    build_plugin_vsgd_training_loop_canary_scorecard,
)
from core.turbocore_simple_optimizer_family_batch_scorecard import (
    build_simple_optimizer_family_batch_scorecard,
)


REFERENCE_CANARY_PLUGIN_NAMES = frozenset({"lion"})
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "product_native_dispatch_ready",
)


def build_plugin_simple_formula_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    simple_reference_report: Mapping[str, Any] | None = None,
    plugin_lion_canary_report: Mapping[str, Any] | None = None,
    plugin_sgdw_canary_report: Mapping[str, Any] | None = None,
    plugin_sgd_canary_report: Mapping[str, Any] | None = None,
    plugin_signsgd_canary_report: Mapping[str, Any] | None = None,
    plugin_tiger_canary_report: Mapping[str, Any] | None = None,
    plugin_qhm_canary_report: Mapping[str, Any] | None = None,
    plugin_accsgd_canary_report: Mapping[str, Any] | None = None,
    plugin_fromage_canary_report: Mapping[str, Any] | None = None,
    plugin_rmsprop_canary_report: Mapping[str, Any] | None = None,
    plugin_lars_canary_report: Mapping[str, Any] | None = None,
    plugin_pid_canary_report: Mapping[str, Any] | None = None,
    plugin_sgdp_canary_report: Mapping[str, Any] | None = None,
    plugin_gravity_canary_report: Mapping[str, Any] | None = None,
    plugin_aggmo_canary_report: Mapping[str, Any] | None = None,
    plugin_asgd_canary_report: Mapping[str, Any] | None = None,
    plugin_madgrad_canary_report: Mapping[str, Any] | None = None,
    plugin_nero_canary_report: Mapping[str, Any] | None = None,
    plugin_vsgd_canary_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected plugin simple-formula status without enabling dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    reference = (
        _as_dict(simple_reference_report)
        if simple_reference_report is not None
        else _call_simple_reference(workspace_root)
    )
    plugin_lion_canary = (
        _as_dict(plugin_lion_canary_report)
        if plugin_lion_canary_report is not None
        else _call_plugin_lion_canary()
    )
    plugin_sgdw_canary = (
        _as_dict(plugin_sgdw_canary_report)
        if plugin_sgdw_canary_report is not None
        else _call_plugin_sgdw_canary()
    )
    plugin_sgd_canary = (
        _as_dict(plugin_sgd_canary_report)
        if plugin_sgd_canary_report is not None
        else _call_plugin_sgd_canary()
    )
    plugin_signsgd_canary = (
        _as_dict(plugin_signsgd_canary_report)
        if plugin_signsgd_canary_report is not None
        else _call_plugin_signsgd_canary()
    )
    plugin_tiger_canary = (
        _as_dict(plugin_tiger_canary_report)
        if plugin_tiger_canary_report is not None
        else _call_plugin_tiger_canary()
    )
    plugin_qhm_canary = (
        _as_dict(plugin_qhm_canary_report)
        if plugin_qhm_canary_report is not None
        else _call_plugin_qhm_canary()
    )
    plugin_accsgd_canary = (
        _as_dict(plugin_accsgd_canary_report)
        if plugin_accsgd_canary_report is not None
        else _call_plugin_accsgd_canary()
    )
    plugin_fromage_canary = (
        _as_dict(plugin_fromage_canary_report)
        if plugin_fromage_canary_report is not None
        else _call_plugin_fromage_canary()
    )
    plugin_rmsprop_canary = (
        _as_dict(plugin_rmsprop_canary_report)
        if plugin_rmsprop_canary_report is not None
        else _call_plugin_rmsprop_canary()
    )
    plugin_lars_canary = (
        _as_dict(plugin_lars_canary_report)
        if plugin_lars_canary_report is not None
        else _call_plugin_lars_canary()
    )
    plugin_pid_canary = (
        _as_dict(plugin_pid_canary_report)
        if plugin_pid_canary_report is not None
        else _call_plugin_pid_canary()
    )
    plugin_sgdp_canary = (
        _as_dict(plugin_sgdp_canary_report)
        if plugin_sgdp_canary_report is not None
        else _call_plugin_sgdp_canary()
    )
    plugin_gravity_canary = (
        _as_dict(plugin_gravity_canary_report)
        if plugin_gravity_canary_report is not None
        else _call_plugin_gravity_canary()
    )
    plugin_aggmo_canary = (
        _as_dict(plugin_aggmo_canary_report)
        if plugin_aggmo_canary_report is not None
        else _call_plugin_aggmo_canary()
    )
    plugin_asgd_canary = (
        _as_dict(plugin_asgd_canary_report)
        if plugin_asgd_canary_report is not None
        else _call_plugin_asgd_canary()
    )
    plugin_madgrad_canary = (
        _as_dict(plugin_madgrad_canary_report)
        if plugin_madgrad_canary_report is not None
        else _call_plugin_madgrad_canary()
    )
    plugin_nero_canary = (
        _as_dict(plugin_nero_canary_report)
        if plugin_nero_canary_report is not None
        else _call_plugin_nero_canary()
    )
    plugin_vsgd_canary = (
        _as_dict(plugin_vsgd_canary_report)
        if plugin_vsgd_canary_report is not None
        else _call_plugin_vsgd_canary()
    )
    selected_rows = _selected_simple_rows(selector)
    plugin_canaries = {
        "lion": plugin_lion_canary,
        "sgdw": plugin_sgdw_canary,
        "sgd": plugin_sgd_canary,
        "signsgd": plugin_signsgd_canary,
        "tiger": plugin_tiger_canary,
        "qhm": plugin_qhm_canary,
        "accsgd": plugin_accsgd_canary,
        "fromage": plugin_fromage_canary,
        "rmsprop": plugin_rmsprop_canary,
        "lars": plugin_lars_canary,
        "pid": plugin_pid_canary,
        "sgdp": plugin_sgdp_canary,
        "gravity": plugin_gravity_canary,
        "aggmo": plugin_aggmo_canary,
        "asgd": plugin_asgd_canary,
        "madgrad": plugin_madgrad_canary,
        "nero": plugin_nero_canary,
        "vsgd": plugin_vsgd_canary,
    }
    rows = [_selected_row(row, reference, plugin_canaries) for row in selected_rows]
    unsafe = _unsafe_claims(
        {
            "selector": selector,
            "simple_reference": reference,
            "plugin_lion_canary": plugin_lion_canary,
            "plugin_sgdw_canary": plugin_sgdw_canary,
            "plugin_sgd_canary": plugin_sgd_canary,
            "plugin_signsgd_canary": plugin_signsgd_canary,
            "plugin_tiger_canary": plugin_tiger_canary,
            "plugin_qhm_canary": plugin_qhm_canary,
            "plugin_accsgd_canary": plugin_accsgd_canary,
            "plugin_fromage_canary": plugin_fromage_canary,
            "plugin_rmsprop_canary": plugin_rmsprop_canary,
            "plugin_lars_canary": plugin_lars_canary,
            "plugin_pid_canary": plugin_pid_canary,
            "plugin_sgdp_canary": plugin_sgdp_canary,
            "plugin_gravity_canary": plugin_gravity_canary,
            "plugin_aggmo_canary": plugin_aggmo_canary,
            "plugin_asgd_canary": plugin_asgd_canary,
            "plugin_madgrad_canary": plugin_madgrad_canary,
            "plugin_nero_canary": plugin_nero_canary,
            "plugin_vsgd_canary": plugin_vsgd_canary,
        },
        rows,
    )
    failed_sources = _failed_sources(selector, reference)
    ready = bool(rows) and not unsafe and selector.get("plugin_selector_classification_ready") is True
    reference_ready_count = sum(1 for row in rows if row["simple_formula_reference_canary_ready"] is True)
    native_canary_ready_count = sum(1 for row in rows if row["selected_plugin_native_canary_ready"] is True)

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_simple_formula_family_batch_scorecard_v0",
        "gate": "plugin_simple_formula_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_simple_formula_family_batch_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "simple_formula",
        "selector_scorecard": _compact_selector(selector),
        "simple_formula_reference_scorecard": _compact_reference(reference),
        "selected_plugin_canary_scorecards": {
            "lion": _compact_plugin_canary(plugin_lion_canary),
            "sgdw": _compact_plugin_canary(plugin_sgdw_canary),
            "sgd": _compact_plugin_canary(plugin_sgd_canary),
            "signsgd": _compact_plugin_canary(plugin_signsgd_canary),
            "tiger": _compact_plugin_canary(plugin_tiger_canary),
            "qhm": _compact_plugin_canary(plugin_qhm_canary),
            "accsgd": _compact_plugin_canary(plugin_accsgd_canary),
            "fromage": _compact_plugin_canary(plugin_fromage_canary),
            "rmsprop": _compact_plugin_canary(plugin_rmsprop_canary),
            "lars": _compact_plugin_canary(plugin_lars_canary),
            "pid": _compact_plugin_canary(plugin_pid_canary),
            "sgdp": _compact_plugin_canary(plugin_sgdp_canary),
            "gravity": _compact_plugin_canary(plugin_gravity_canary),
            "aggmo": _compact_plugin_canary(plugin_aggmo_canary),
            "asgd": _compact_plugin_canary(plugin_asgd_canary),
            "madgrad": _compact_plugin_canary(plugin_madgrad_canary),
            "nero": _compact_plugin_canary(plugin_nero_canary),
            "vsgd": _compact_plugin_canary(plugin_vsgd_canary),
        },
        "rows": rows,
        "summary": {
            "selected_simple_formula_optimizer_count": len(rows),
            "selector_simple_formula_count": len(selected_rows),
            "reference_canary_candidate_count": sum(
                1 for row in rows if row["simple_formula_reference_canary_candidate"] is True
            ),
            "reference_canary_ready_count": reference_ready_count,
            "selected_plugin_native_canary_ready_count": native_canary_ready_count,
            "selected_plugin_native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "unsafe_claim_count": len(unsafe),
            "failed_source_count": len(failed_sources),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + failed_sources
            + [
                "selected_plugin_simple_formula_native_abi_missing",
                "selected_plugin_simple_formula_training_tensor_binding_missing",
                "selected_plugin_simple_formula_dispatch_review_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(unsafe + failed_sources),
        "recommended_next_step": (
            "selected-family owner/release hold for ready simple-formula canaries with dispatch default-off"
            if ready
            else "fix selector/reference blockers before selected plugin simple-formula canaries"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Built-in simple optimizer canary evidence is reference evidence only.",
            "Plugin Lion TrainingLoop canary evidence is selected-route canary evidence, not product native readiness.",
            "Selected plugin simple-formula routes require their own ABI and dispatch review.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _call_selector() -> dict[str, Any]:
    try:
        return dict(build_plugin_optimizer_selector_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_optimizer_selector_scorecard", exc)


def _call_simple_reference(workspace_root: str | Path | None) -> dict[str, Any]:
    try:
        if workspace_root is None:
            return dict(build_simple_optimizer_family_batch_scorecard())
        return dict(build_simple_optimizer_family_batch_scorecard(workspace_root=workspace_root))
    except Exception as exc:
        return _failed_report("build_simple_optimizer_family_batch_scorecard", exc)


def _call_plugin_lion_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_lion_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_lion_training_loop_canary_scorecard", exc)


def _call_plugin_sgdw_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_sgdw_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_sgdw_training_loop_canary_scorecard", exc)


def _call_plugin_sgd_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_sgd_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_sgd_training_loop_canary_scorecard", exc)


def _call_plugin_signsgd_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_signsgd_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_signsgd_training_loop_canary_scorecard", exc)


def _call_plugin_tiger_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_tiger_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_tiger_training_loop_canary_scorecard", exc)


def _call_plugin_qhm_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_qhm_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_qhm_training_loop_canary_scorecard", exc)


def _call_plugin_accsgd_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_accsgd_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_accsgd_training_loop_canary_scorecard", exc)


def _call_plugin_fromage_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_fromage_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_fromage_training_loop_canary_scorecard", exc)


def _call_plugin_rmsprop_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_rmsprop_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_rmsprop_training_loop_canary_scorecard", exc)


def _call_plugin_lars_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_lars_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_lars_training_loop_canary_scorecard", exc)


def _call_plugin_pid_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_pid_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_pid_training_loop_canary_scorecard", exc)


def _call_plugin_sgdp_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_sgdp_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_sgdp_training_loop_canary_scorecard", exc)


def _call_plugin_gravity_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_gravity_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_gravity_training_loop_canary_scorecard", exc)


def _call_plugin_aggmo_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_aggmo_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_aggmo_training_loop_canary_scorecard", exc)


def _call_plugin_asgd_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_asgd_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_asgd_training_loop_canary_scorecard", exc)


def _call_plugin_madgrad_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_madgrad_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_madgrad_training_loop_canary_scorecard", exc)


def _call_plugin_nero_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_nero_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_nero_training_loop_canary_scorecard", exc)


def _call_plugin_vsgd_canary() -> dict[str, Any]:
    try:
        return dict(build_plugin_vsgd_training_loop_canary_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_vsgd_training_loop_canary_scorecard", exc)


def _selected_simple_rows(selector: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = selector.get("rows")
    if not isinstance(rows, list):
        return []
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("native_route_family", "")) == "simple_formula"
    ]


def _selected_row(
    selector_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    plugin_canaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    name = str(selector_row.get("optimizer_name", "")).strip().lower()
    reference_candidate = name in REFERENCE_CANARY_PLUGIN_NAMES
    reference_ready = reference_candidate and reference.get("simple_formula_native_batch_canary_ready") is True
    plugin_canary = _as_dict(plugin_canaries.get(name))
    selected_canary_ready = plugin_canary.get("selected_native_canary_ready") is True
    if selected_canary_ready:
        status = "selected_plugin_simple_formula_native_canary_ready"
    elif reference_ready:
        status = "simple_formula_reference_canary_available"
    else:
        status = "selected_plugin_simple_formula_reference_only_pending"
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": str(selector_row.get("selector", "")),
        "selected_optimizer_family": "simple_formula",
        "native_route_family": "simple_formula",
        "batch_status": status,
        "resume_proven": selector_row.get("resume_proven") is True,
        "selector_classified": True,
        "simple_formula_reference_canary_candidate": reference_candidate,
        "simple_formula_reference_canary_ready": reference_ready,
        "selected_plugin_native_canary_ready": selected_canary_ready,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": (
            "selected_plugin_simple_formula_e2e_shadow_matrix"
            if selected_canary_ready
            else "selected_plugin_simple_formula_abi_and_formula_parity"
        ),
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "simple_formula_count": int(counts.get("simple_formula", 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
    }


def _compact_reference(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "simple_formula_native_batch_canary_ready": report.get("simple_formula_native_batch_canary_ready") is True,
        "batch_canary_ready_count": int(summary.get("batch_canary_ready_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "product_native_dispatch_ready": report.get("product_native_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "selected_optimizer_name": str(report.get("selected_optimizer_name", "")),
        "selected_native_canary_ready": report.get("selected_native_canary_ready") is True,
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "training_path_enabled": report.get("training_path_enabled") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _failed_sources(selector: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    failed = []
    if selector.get("ok") is not True:
        failed.append("plugin_optimizer_selector_scorecard_not_ok")
    if not reference:
        failed.append("simple_formula_reference_scorecard_missing")
    return failed


def _unsafe_claims(
    reports: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[str]:
    out: list[str] = []
    for name, report in reports.items():
        scorecard = str(report.get("scorecard", name))
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                out.append(f"unsafe_plugin_simple_formula_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_simple_formula_row:{selected}:{field}")
    return _dedupe(out)


def _failed_report(builder_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "scorecard": builder_name,
        "error": f"{type(exc).__name__}: {exc}",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [f"builder_failed:{builder_name}"],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_simple_formula_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "REFERENCE_CANARY_PLUGIN_NAMES",
    "build_plugin_simple_formula_family_batch_scorecard",
]
