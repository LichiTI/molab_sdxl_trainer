"""Batch scorecard for selected plugin Adam-like native canary routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.turbocore_plugin_adamax_native_scratch_kernel_scorecard import (
    build_plugin_adamax_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adamax_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adamax_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adamax_training_loop_canary_scorecard import (
    build_plugin_adamax_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamax_training_tensor_binding_canary_scorecard import (
    build_plugin_adamax_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adamc_native_scratch_kernel_scorecard import (
    build_plugin_adamc_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adamc_training_loop_canary_scorecard import (
    build_plugin_adamc_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamc_training_tensor_binding_canary_scorecard import (
    build_plugin_adamc_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adamg_native_scratch_kernel_scorecard import (
    build_plugin_adamg_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adamg_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adamg_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adamg_training_loop_canary_scorecard import (
    build_plugin_adamg_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamg_training_tensor_binding_canary_scorecard import (
    build_plugin_adamg_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adamlike_selected_optimizer_scorecard import (
    build_plugin_adamlike_selected_optimizer_scorecard,
)
from core.turbocore_plugin_adamod_native_scratch_kernel_scorecard import (
    build_plugin_adamod_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adamod_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adamod_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adamod_training_loop_canary_scorecard import (
    build_plugin_adamod_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamod_training_tensor_binding_canary_scorecard import (
    build_plugin_adamod_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adamp_native_scratch_kernel_scorecard import (
    build_plugin_adamp_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adamp_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adamp_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adamp_training_loop_canary_scorecard import (
    build_plugin_adamp_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamp_training_tensor_binding_canary_scorecard import (
    build_plugin_adamp_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adam_native_scratch_kernel_scorecard import (
    build_plugin_adam_native_scratch_kernel_scorecard,
)
from core.turbocore_plugin_adam_runtime_dispatch_adapter_shadow_scorecard import (
    build_plugin_adam_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_plugin_adam_training_loop_canary_scorecard import (
    build_plugin_adam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adam_training_tensor_binding_canary_scorecard import (
    build_plugin_adam_training_tensor_binding_canary_scorecard,
)
from core.turbocore_plugin_adamw_training_loop_canary_scorecard import (
    build_plugin_adamw_training_loop_canary_scorecard,
)
from core.turbocore_plugin_padam_training_loop_canary_scorecard import (
    build_plugin_padam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_radam_training_loop_canary_scorecard import (
    build_plugin_radam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_yogi_training_loop_canary_scorecard import (
    build_plugin_yogi_training_loop_canary_scorecard,
)
from core.turbocore_plugin_dualadam_training_loop_canary_scorecard import (
    build_plugin_dualadam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_exadam_training_loop_canary_scorecard import (
    build_plugin_exadam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_qhadam_training_loop_canary_scorecard import (
    build_plugin_qhadam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_nadam_training_loop_canary_scorecard import (
    build_plugin_nadam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_grokfastadamw_training_loop_canary_scorecard import (
    build_plugin_grokfastadamw_training_loop_canary_scorecard,
)
from core.turbocore_plugin_ranger_training_loop_canary_scorecard import (
    build_plugin_ranger_training_loop_canary_scorecard,
)
from core.turbocore_plugin_ranger21_training_loop_canary_scorecard import (
    build_plugin_ranger21_training_loop_canary_scorecard,
)
from core.turbocore_plugin_ranger25_training_loop_canary_scorecard import (
    build_plugin_ranger25_training_loop_canary_scorecard,
)
from core.turbocore_plugin_novograd_training_loop_canary_scorecard import (
    build_plugin_novograd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_stableadamw_training_loop_canary_scorecard import (
    build_plugin_stableadamw_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adamwsn_training_loop_canary_scorecard import (
    build_plugin_adamwsn_training_loop_canary_scorecard,
)
from core.turbocore_plugin_adams_training_loop_canary_scorecard import (
    build_plugin_adams_training_loop_canary_scorecard,
)
from core.turbocore_plugin_lamb_training_loop_canary_scorecard import (
    build_plugin_lamb_training_loop_canary_scorecard,
)
from core.turbocore_plugin_fadam_training_loop_canary_scorecard import (
    build_plugin_fadam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_flashadamw_training_loop_canary_scorecard import (
    build_plugin_flashadamw_training_loop_canary_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ReportBuilder = Callable[[], dict[str, Any]]


_DEDICATED_ROUTES: dict[str, tuple[ReportBuilder, ReportBuilder, ReportBuilder, ReportBuilder]] = {
    "adam": (
        build_plugin_adam_native_scratch_kernel_scorecard,
        build_plugin_adam_training_tensor_binding_canary_scorecard,
        build_plugin_adam_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adam_training_loop_canary_scorecard,
    ),
    "adamax": (
        build_plugin_adamax_native_scratch_kernel_scorecard,
        build_plugin_adamax_training_tensor_binding_canary_scorecard,
        build_plugin_adamax_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adamax_training_loop_canary_scorecard,
    ),
    "adamc": (
        build_plugin_adamc_native_scratch_kernel_scorecard,
        build_plugin_adamc_training_tensor_binding_canary_scorecard,
        build_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adamc_training_loop_canary_scorecard,
    ),
    "adamg": (
        build_plugin_adamg_native_scratch_kernel_scorecard,
        build_plugin_adamg_training_tensor_binding_canary_scorecard,
        build_plugin_adamg_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adamg_training_loop_canary_scorecard,
    ),
    "adamod": (
        build_plugin_adamod_native_scratch_kernel_scorecard,
        build_plugin_adamod_training_tensor_binding_canary_scorecard,
        build_plugin_adamod_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adamod_training_loop_canary_scorecard,
    ),
    "adamp": (
        build_plugin_adamp_native_scratch_kernel_scorecard,
        build_plugin_adamp_training_tensor_binding_canary_scorecard,
        build_plugin_adamp_runtime_dispatch_adapter_shadow_scorecard,
        build_plugin_adamp_training_loop_canary_scorecard,
    ),
}


def build_plugin_adamlike_family_batch_scorecard(
    *,
    include_live_canaries: bool = True,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected Adam-like plugin route evidence without dispatch."""

    selected = build_plugin_adamlike_selected_optimizer_scorecard()
    rows = [
        _adamw_row(include_live_canaries=include_live_canaries),
        _padam_row(include_live_canaries=include_live_canaries),
        _radam_row(include_live_canaries=include_live_canaries),
        _yogi_row(include_live_canaries=include_live_canaries),
        _dualadam_row(include_live_canaries=include_live_canaries),
        _exadam_row(include_live_canaries=include_live_canaries),
        _qhadam_row(include_live_canaries=include_live_canaries),
        _nadam_row(include_live_canaries=include_live_canaries),
        _grokfastadamw_row(include_live_canaries=include_live_canaries),
        _ranger_row(include_live_canaries=include_live_canaries),
        _ranger21_row(include_live_canaries=include_live_canaries),
        _ranger25_row(include_live_canaries=include_live_canaries),
        _novograd_row(include_live_canaries=include_live_canaries),
        _stableadamw_row(include_live_canaries=include_live_canaries),
        _adamwsn_row(include_live_canaries=include_live_canaries),
        _adams_row(include_live_canaries=include_live_canaries),
        _lamb_row(include_live_canaries=include_live_canaries),
        _fadam_row(include_live_canaries=include_live_canaries),
        _flashadamw_row(include_live_canaries=include_live_canaries),
    ]
    rows.extend(_dedicated_row(name, builders, include_live_canaries) for name, builders in sorted(_DEDICATED_ROUTES.items()))
    e2e = _artifact_report("turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard.json")
    rollout = _artifact_report("turbocore_plugin_adamlike_canary_rollout_policy_scorecard.json")
    ready_rows = [row for row in rows if row["selected_native_canary_ready"] is True]
    unsafe = _unsafe_claims(selected, *[row["compact_reports"] for row in rows])
    failed = [row for row in rows if row["selected_native_canary_ready"] is not True]
    selected_names = set(str(name) for name in _summary(selected).get("dedicated_kernel_optimizer_names", []) or [])
    selected_names.update(str(name) for name in _summary(selected).get("compatible_optimizer_names", []) or [])
    covered_names = {str(row["selected_optimizer_name"]) for row in rows}
    pending_selected = sorted(name for name in selected_names if name not in covered_names)
    ready = selected.get("selected_optimizer_abi_ready") is True and not failed and not unsafe
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamlike_family_batch_scorecard_v0",
        "gate": "plugin_adamlike_selected_native_canary_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_adamlike_family_batch_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "plugin_selected_native_ready_count": 0,
        "selected_scorecard": _compact_selected(selected),
        "e2e_shadow_matrix": _compact_e2e(e2e),
        "canary_rollout_policy": _compact_rollout(rollout),
        "rows": rows,
        "summary": {
            "selected_adamlike_optimizer_count": int(_summary(selected).get("case_count", 0) or 0),
            "target_count": len(rows),
            "selected_native_canary_ready_count": len(ready_rows),
            "exact_adamw_route_canary_ready_count": sum(
                1 for row in ready_rows if row["native_route"] == "rust_cuda_adamw_v0"
            ),
            "dedicated_route_canary_ready_count": sum(
                1 for row in ready_rows if row["native_route"] != "rust_cuda_adamw_v0"
            ),
            "pending_selected_optimizer_count": len(pending_selected),
            "pending_selected_optimizer_names": pending_selected,
            "plugin_selected_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "e2e_shadow_matrix_ready": e2e.get("e2e_shadow_matrix_ready") is True,
            "canary_rollout_policy_ready": rollout.get("canary_rollout_policy_ready") is True,
        },
        "promotion_blockers": _review_blockers(unsafe, failed, e2e, rollout),
        "blocked_reasons": unsafe
        + [f"selected_adamlike_native_canary_pending:{row['selected_optimizer_name']}" for row in failed],
        "recommended_next_step": (
            _recommended_next_step(e2e, rollout)
            if ready
            else "fix selected Adam-like native canary rows before e2e shadow matrix"
        ),
        "notes": [
            "This batch aggregates existing selected-plugin canaries only.",
            "Ready rows remain default-off and are not product native-ready.",
            "Uncovered Adam-like plugin names stay in the dedicated-kernel backlog.",
        ],
    }
    if write_artifact:
        e2e, rollout = _refresh_review_artifacts(report)
        report["e2e_shadow_matrix"] = _compact_e2e(e2e)
        report["canary_rollout_policy"] = _compact_rollout(rollout)
        report["summary"]["e2e_shadow_matrix_ready"] = e2e.get("e2e_shadow_matrix_ready") is True
        report["summary"]["canary_rollout_policy_ready"] = rollout.get("canary_rollout_policy_ready") is True
        report["promotion_blockers"] = _review_blockers(unsafe, failed, e2e, rollout)
        report["recommended_next_step"] = (
            _recommended_next_step(e2e, rollout)
            if ready
            else "fix selected Adam-like native canary rows before e2e shadow matrix"
        )
        _write_artifact(report)
    return report


def _adamw_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_adamw_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "adamw",
        "native_route": "rust_cuda_adamw_v0",
        "route_kind": "exact_adamw_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_adamw_training_loop_canary",
    }


def _radam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_radam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "radam",
        "native_route": "rust_cuda_plugin_radam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_radam_training_loop_canary",
    }


def _padam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_padam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "padam",
        "native_route": "rust_cuda_plugin_padam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_padam_training_loop_canary",
    }


def _yogi_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_yogi_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "yogi",
        "native_route": "rust_cuda_plugin_yogi_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_yogi_training_loop_canary",
    }


def _dualadam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_dualadam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "dualadam",
        "native_route": "rust_cuda_plugin_dualadam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_dualadam_training_loop_canary",
    }


def _exadam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_exadam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "exadam",
        "native_route": "rust_cuda_plugin_exadam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_exadam_training_loop_canary",
    }


def _qhadam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_qhadam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "qhadam",
        "native_route": "rust_cuda_plugin_qhadam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_qhadam_training_loop_canary",
    }


def _nadam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_nadam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "nadam",
        "native_route": "rust_cuda_plugin_nadam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_nadam_training_loop_canary",
    }


def _grokfastadamw_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = (
        _call(build_plugin_grokfastadamw_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    )
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "grokfastadamw",
        "native_route": "rust_cuda_plugin_grokfastadamw_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_grokfastadamw_training_loop_canary",
    }


def _ranger_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_ranger_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "ranger",
        "native_route": "rust_cuda_plugin_ranger_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_ranger_training_loop_canary",
    }


def _ranger21_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_ranger21_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "ranger21",
        "native_route": "rust_cuda_plugin_ranger21_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_ranger21_training_loop_canary",
    }


def _ranger25_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_ranger25_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "ranger25",
        "native_route": "rust_cuda_plugin_ranger25_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_ranger25_training_loop_canary",
    }


def _novograd_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_novograd_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "novograd",
        "native_route": "rust_cuda_plugin_novograd_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_novograd_training_loop_canary",
    }


def _stableadamw_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_stableadamw_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "stableadamw",
        "native_route": "rust_cuda_plugin_stableadamw_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_stableadamw_training_loop_canary",
    }


def _adamwsn_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_adamwsn_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "adamwsn",
        "native_route": "rust_cuda_plugin_adamwsn_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_adamwsn_training_loop_canary",
    }


def _adams_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_adams_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "adams",
        "native_route": "rust_cuda_plugin_adams_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_adams_training_loop_canary",
    }


def _lamb_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_lamb_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "lamb",
        "native_route": "rust_cuda_plugin_lamb_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_lamb_training_loop_canary",
    }


def _fadam_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_fadam_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "fadam",
        "native_route": "rust_cuda_plugin_fadam_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_fadam_training_loop_canary",
    }


def _flashadamw_row(*, include_live_canaries: bool) -> dict[str, Any]:
    training_loop = _call(build_plugin_flashadamw_training_loop_canary_scorecard) if include_live_canaries else _skipped()
    ready = training_loop.get("ok") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": "flashadamw",
        "native_route": "rust_cuda_plugin_flashadamw_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": {"training_loop_canary": ready},
        "compact_reports": {"training_loop": _compact_report(training_loop)},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else "selected_flashadamw_training_loop_canary",
    }


def _dedicated_row(
    name: str,
    builders: tuple[ReportBuilder, ReportBuilder, ReportBuilder, ReportBuilder],
    include_live_canaries: bool,
) -> dict[str, Any]:
    scratch_builder, binding_builder, runtime_builder, loop_builder = builders
    scratch = _call(scratch_builder) if include_live_canaries else _skipped()
    binding = _call(binding_builder) if include_live_canaries else _skipped()
    runtime = _call(runtime_builder)
    training_loop = _call(loop_builder) if include_live_canaries else _skipped()
    stage_status = {
        "native_scratch_kernel": scratch.get("ok") is True,
        "training_tensor_binding": binding.get("ok") is True,
        "runtime_dispatch_adapter_shadow": runtime.get("ok") is True,
        "training_loop_canary": training_loop.get("ok") is True,
    }
    ready = all(stage_status.values())
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "native_route": f"rust_cuda_plugin_{name}_v0",
        "route_kind": "dedicated_selected_plugin_route",
        "selected_native_canary_ready": ready,
        "stage_status": stage_status,
        "compact_reports": {
            "native_scratch_kernel": _compact_report(scratch),
            "training_tensor_binding": _compact_report(binding),
            "runtime_dispatch_adapter_shadow": _compact_report(runtime),
            "training_loop": _compact_report(training_loop),
        },
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "next_gate": "selected_adamlike_e2e_shadow_matrix" if ready else f"selected_{name}_native_canary_repair",
    }


def _call(builder: ReportBuilder) -> dict[str, Any]:
    try:
        return dict(builder())
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "scorecard": getattr(builder, "__name__", "unknown_builder"),
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": [f"builder_failed:{getattr(builder, '__name__', 'unknown_builder')}"],
        }


def _skipped() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "scorecard": "live_canary_skipped",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": ["live_canary_skipped"],
    }


def _compact_selected(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "selected_optimizer_abi_ready": report.get("selected_optimizer_abi_ready") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "adamw_native_route_compatible_count": int(summary.get("adamw_native_route_compatible_count", 0) or 0),
        "dedicated_kernel_required_count": int(summary.get("dedicated_kernel_required_count", 0) or 0),
    }


def _compact_e2e(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "e2e_shadow_matrix_ready": report.get("e2e_shadow_matrix_ready") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "live_shadow_matrix_executed": report.get("live_shadow_matrix_executed") is True,
    }


def _compact_rollout(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "canary_auto_enabled": report.get("canary_auto_enabled") is True,
        "explicit_opt_in_required": bool(summary.get("explicit_opt_in_required", False)),
    }


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "summary": dict(_as_dict(report.get("summary"))),
    }


def _unsafe_claims(selected: Mapping[str, Any], *row_reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reports: list[Mapping[str, Any]] = [selected]
    for bundle in row_reports:
        reports.extend(bundle.values())
    out: list[str] = []
    for report in reports:
        scorecard = str(report.get("scorecard", "unknown_scorecard"))
        if report.get("training_path_enabled") is True:
            out.append(f"{scorecard}:training_path_enabled")
        if report.get("default_behavior_changed") is True:
            out.append(f"{scorecard}:default_behavior_changed")
        if report.get("runtime_dispatch_ready") is True:
            out.append(f"{scorecard}:runtime_dispatch_ready")
        if report.get("native_dispatch_allowed") is True:
            out.append(f"{scorecard}:native_dispatch_allowed")
    return _dedupe(out)


def _artifact_report(filename: str) -> dict[str, Any]:
    path = REPO_ROOT / "temp" / "turbocore_optimizer" / filename
    if not path.exists():
        return {}
    try:
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _recommended_next_step(e2e: Mapping[str, Any], rollout: Mapping[str, Any]) -> str:
    if e2e.get("e2e_shadow_matrix_ready") is not True:
        return "add e2e shadow matrix for ready selected Adam-like plugin routes"
    if rollout.get("canary_rollout_policy_ready") is not True:
        return "add default-off canary rollout policy for ready selected Adam-like plugin routes"
    return "continue selected Adam-like backlog kernels or prepare owner/release hold package"


def _review_blockers(
    unsafe: list[str],
    failed: list[Mapping[str, Any]],
    e2e: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> list[str]:
    return _dedupe(
        unsafe
        + [f"selected_adamlike_native_canary_pending:{row['selected_optimizer_name']}" for row in failed]
        + ([] if e2e.get("e2e_shadow_matrix_ready") is True else ["selected_adamlike_e2e_shadow_matrix_missing"])
        + ([] if rollout.get("canary_rollout_policy_ready") is True else ["selected_adamlike_rollout_policy_missing"])
        + ["owner_release_hold_missing"]
    )


def _refresh_review_artifacts(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from core.turbocore_plugin_adamlike_canary_rollout_policy_scorecard import (
        build_plugin_adamlike_canary_rollout_policy_scorecard,
    )
    from core.turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard import (
        build_plugin_adamlike_e2e_shadow_matrix_scorecard,
    )

    e2e = build_plugin_adamlike_e2e_shadow_matrix_scorecard(
        adamlike_batch_report=report,
        write_artifact=True,
    )
    rollout = build_plugin_adamlike_canary_rollout_policy_scorecard(
        shadow_matrix_report=e2e,
        write_artifact=True,
    )
    return e2e, rollout


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_adamlike_family_batch_scorecard.json"
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
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_plugin_adamlike_family_batch_scorecard"]
