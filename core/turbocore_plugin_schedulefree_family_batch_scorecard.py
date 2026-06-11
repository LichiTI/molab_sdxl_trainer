"""Batch scorecard for selected schedule-free plugin optimizer evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.turbocore_plugin_schedulefree_canary_rollout_policy_scorecard import (
    build_plugin_schedulefree_canary_rollout_policy_scorecard,
)
from core.turbocore_plugin_schedulefree_checkpoint_adapter_scorecard import (
    build_plugin_schedulefree_checkpoint_adapter_scorecard,
)
from core.turbocore_plugin_schedulefree_dispatch_integration_review_scorecard import (
    build_plugin_schedulefree_dispatch_integration_review_scorecard,
)
from core.turbocore_plugin_schedulefree_e2e_shadow_training_matrix_scorecard import (
    build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard,
)
from core.turbocore_plugin_schedulefree_adamw_training_loop_canary_scorecard import (
    build_plugin_schedulefree_adamw_training_loop_canary_scorecard,
)
from core.turbocore_plugin_schedulefree_sgd_training_loop_canary_scorecard import (
    build_plugin_schedulefree_sgd_training_loop_canary_scorecard,
)
from core.turbocore_plugin_schedulefree_radam_training_loop_canary_scorecard import (
    build_plugin_schedulefree_radam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_schedulefree_native_abi_sketch_scorecard import (
    build_plugin_schedulefree_native_abi_sketch_scorecard,
)
from core.turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard import (
    build_plugin_schedulefree_runtime_dispatch_shadow_scorecard,
)
from core.turbocore_plugin_schedulefree_selected_optimizer_scorecard import (
    build_plugin_schedulefree_selected_optimizer_scorecard,
)
from core.turbocore_plugin_schedulefree_training_tensor_binding_canary_scorecard import (
    build_plugin_schedulefree_training_tensor_binding_canary_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ReportBuilder = Callable[[], dict[str, Any]]


_STAGE_BUILDERS: tuple[tuple[str, ReportBuilder, str], ...] = (
    ("selected_optimizer_abi", build_plugin_schedulefree_selected_optimizer_scorecard, "selected_optimizer_abi_ready"),
    ("native_abi_sketch", build_plugin_schedulefree_native_abi_sketch_scorecard, "native_abi_sketch_ready"),
    ("checkpoint_adapter", build_plugin_schedulefree_checkpoint_adapter_scorecard, "checkpoint_adapter_proof_ready"),
    ("training_tensor_binding", build_plugin_schedulefree_training_tensor_binding_canary_scorecard, "training_tensor_binding_canary_ready"),
    ("runtime_dispatch_shadow", build_plugin_schedulefree_runtime_dispatch_shadow_scorecard, "runtime_dispatch_shadow_ready"),
    ("e2e_shadow_matrix", build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard, "e2e_shadow_training_matrix_ready"),
    ("canary_rollout_policy", build_plugin_schedulefree_canary_rollout_policy_scorecard, "canary_rollout_policy_ready"),
    ("dispatch_integration_review", build_plugin_schedulefree_dispatch_integration_review_scorecard, "review_gate_ready"),
    ("schedulefreeadamw_training_loop_canary", build_plugin_schedulefree_adamw_training_loop_canary_scorecard, "selected_native_canary_ready"),
    ("schedulefreesgd_training_loop_canary", build_plugin_schedulefree_sgd_training_loop_canary_scorecard, "selected_native_canary_ready"),
    ("schedulefreeradam_training_loop_canary", build_plugin_schedulefree_radam_training_loop_canary_scorecard, "selected_native_canary_ready"),
)


def build_plugin_schedulefree_family_batch_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Aggregate selected schedule-free plugin evidence without enabling dispatch."""

    stages = {name: _call(builder) for name, builder, _ready_field in _STAGE_BUILDERS}
    rows = [_stage_row(name, report, ready_field) for name, _builder, ready_field in _STAGE_BUILDERS for report in [stages[name]]]
    unsafe = _unsafe_claims(stages)
    not_ready = [row for row in rows if row["stage_ready"] is not True]
    ready = bool(rows) and not not_ready and not unsafe
    selected_native_canary_ready_count = _selected_native_canary_ready_count(stages)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_family_batch_scorecard_v0",
        "gate": "plugin_schedulefree_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_schedulefree_family_batch_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "schedule_free_state_machine",
        "rows": rows,
        "stage_summaries": {name: _compact_report(report) for name, report in stages.items()},
        "summary": {
            "stage_count": len(rows),
            "ready_stage_count": sum(1 for row in rows if row["stage_ready"] is True),
            "selected_optimizer_count": _selected_optimizer_count(stages),
            "e2e_shadow_case_count": _e2e_case_count(stages),
            "selected_native_canary_ready_count": selected_native_canary_ready_count,
            "native_ready_count": selected_native_canary_ready_count,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "native_kernel_ready": False,
            "dispatch_review_gate_ready": stages["dispatch_integration_review"].get("review_gate_ready") is True,
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [f"schedulefree_stage_not_ready:{row['stage']}" for row in not_ready]
            + [
                "selected_schedulefree_native_kernel_missing",
                "selected_schedulefree_remaining_native_kernels_missing",
                "selected_schedulefree_real_dispatch_wiring_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(unsafe + [f"schedulefree_stage_not_ready:{row['stage']}" for row in not_ready]),
        "recommended_next_step": (
            "selected-family owner/release hold for ready schedule-free canaries with dispatch default-off"
            if ready
            else "fix selected schedule-free family batch blockers"
        ),
        "notes": [
            "This batch aggregates existing selected schedule-free gates only.",
            "The selected pytorch_optimizer plugin remains authoritative.",
            "Native kernel, real dispatch wiring, and product exposure remain absent.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


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


def _stage_row(name: str, report: Mapping[str, Any], ready_field: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": name,
        "scorecard": str(report.get("scorecard", "")),
        "ready_field": ready_field,
        "stage_ready": report.get(ready_field) is True,
        "ok": report.get("ok") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
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


def _selected_optimizer_count(stages: Mapping[str, Mapping[str, Any]]) -> int:
    return int(_as_dict(stages["selected_optimizer_abi"].get("summary")).get("case_count", 0) or 0)


def _e2e_case_count(stages: Mapping[str, Mapping[str, Any]]) -> int:
    return int(_as_dict(stages["e2e_shadow_matrix"].get("summary")).get("case_count", 0) or 0)


def _selected_native_canary_ready_count(stages: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(
        1
        for name, report in stages.items()
        if name.endswith("_training_loop_canary") and report.get("selected_native_canary_ready") is True
    )


def _unsafe_claims(stages: Mapping[str, Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for name, report in stages.items():
        for field in ("training_path_enabled", "default_behavior_changed", "runtime_dispatch_ready", "native_dispatch_allowed"):
            if report.get(field) is True:
                out.append(f"unsafe_schedulefree_stage_claim:{name}:{field}")
    return out


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_schedulefree_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


__all__ = ["build_plugin_schedulefree_family_batch_scorecard"]
