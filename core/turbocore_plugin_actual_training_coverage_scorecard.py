"""Report-only actual-training coverage matrix for selected plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard import (
    build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard import (
    build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_adaptivelr_training_loop_canary_scorecard import (
    build_plugin_adaptivelr_training_loop_canary_scorecard,
)
from core.turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard import (
    build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_closure_second_order_training_loop_canary_scorecard import (
    build_plugin_closure_second_order_training_loop_canary_scorecard,
)
from core.turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard import (
    build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard import (
    build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_fused_backward_runtime_precondition_rehearsal_scorecard import (
    build_plugin_fused_backward_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_lomo_fused_backward_hook_canary_scorecard import (
    build_lomo_fused_backward_hook_canary_scorecard,
)
from core.turbocore_plugin_bridge_training_loop_canary_scorecard import (
    build_plugin_bridge_training_loop_canary_scorecard,
)
from core.turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard import (
    build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard,
)
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard import (
    build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard import (
    build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard,
)
from core.turbocore_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard import (
    build_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
ARTIFACT_NAME = "turbocore_plugin_actual_training_coverage_scorecard.json"
EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT = 124

Builder = Callable[..., dict[str, Any]]

RUNTIME_DISPATCH_BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("simple_formula", build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard),
    ("adam_like_formula", build_plugin_adamlike_runtime_dispatch_rehearsal_scorecard),
    ("schedule_free_state_machine", build_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard),
    ("adaptive_lr_state_machine", build_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard),
)
RUNTIME_PRECONDITION_BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("factored_memory_layout", build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard),
    ("closure_or_second_order", build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard),
    ("custom_formula", build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard),
    ("model_or_shape_aware", build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard),
    ("state_adapter_special", build_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard),
    ("fused_backward", build_plugin_fused_backward_runtime_precondition_rehearsal_scorecard),
)
PER_OPTIMIZER_NATIVE_BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("adaptive_lr_state_machine", build_plugin_adaptivelr_training_loop_canary_scorecard),
    ("closure_or_second_order", build_plugin_closure_second_order_training_loop_canary_scorecard),
    ("fused_backward", build_lomo_fused_backward_hook_canary_scorecard),
    ("bridge_selected_native", build_plugin_bridge_training_loop_canary_scorecard),
)


def build_plugin_actual_training_coverage_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    selector = build_plugin_optimizer_selector_scorecard()
    selector_rows = [_as_dict(row) for row in selector.get("rows", []) if isinstance(row, Mapping)]
    evidence = _build_evidence()
    rows = [_coverage_row(row, evidence.get(_name(row), {})) for row in selector_rows]
    route_family_counts = _count_by(rows, "native_route_family")
    status_counts = _count_by(rows, "actual_training_status")
    actual_count = sum(1 for row in rows if row["per_optimizer_native_training_executed"])
    representative_count = sum(1 for row in rows if row["representative_native_training_mapped"])
    precondition_count = sum(1 for row in rows if row["runtime_precondition_rehearsal_ready"])
    resume_count = sum(1 for row in rows if row["trainer_resume_parity_proven"])
    total = len(rows)
    gap = total - actual_count
    ok = (
        selector.get("ok") is True
        and total == EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT
        and resume_count == EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT
        and all(row["training_path_enabled"] is False for row in rows)
        and all(row["native_dispatch_allowed"] is False for row in rows)
        and all(row["product_native_ready"] is False for row in rows)
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_actual_training_coverage_scorecard_v0",
        "gate": "plugin_selected_actual_training_coverage",
        "roadmap": ROADMAP,
        "ok": ok,
        "evidence_ready": ok,
        "promotion_ready": False,
        "actual_training_complete": gap == 0,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "selector_scorecard": _compact_selector(selector),
        "rows": rows,
        "summary": {
            "selected_plugin_optimizer_count": total,
            "trainer_resume_parity_proven_count": resume_count,
            "per_optimizer_native_training_count": actual_count,
            "actual_training_gap_count": gap,
            "representative_native_training_mapped_count": representative_count,
            "runtime_precondition_rehearsal_ready_count": precondition_count,
            "actual_training_complete_count": 1 if gap == 0 else 0,
            "training_path_enabled_count": 0,
            "native_dispatch_allowed_count": 0,
            "product_native_ready_count": 0,
            "route_family_counts": route_family_counts,
            "actual_training_status_counts": status_counts,
        },
        "promotion_blockers": _promotion_blockers(gap),
        "blocked_reasons": [] if ok else ["plugin_selected_actual_training_coverage_matrix_incomplete"],
        "recommended_next_step": _recommended_next_step(actual_count, total, gap),
        "notes": [
            "Trainer resume parity is the baseline actual trainer-path proof, not CUDA native dispatch proof.",
            "Per-optimizer native training requires that the selected optimizer row itself executed native step and kernel launch.",
            "Representative native launches and precondition rehearsals stay tracked separately, but the selected optimizer row must now close the gap itself.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _build_evidence() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for family, builder in RUNTIME_DISPATCH_BUILDERS:
        report = builder(write_artifact=True)
        for case in _cases(report):
            _merge_evidence(evidence, family, case, source_kind="runtime_dispatch")
    for family, builder in RUNTIME_PRECONDITION_BUILDERS:
        report = builder(write_artifact=True, include_representative_runtime_canary=True)
        for case in _cases(report):
            _merge_evidence(evidence, family, case, source_kind="runtime_precondition")
        adapter = _as_dict(report.get("family_specific_runtime_launch_adapter_coverage"))
        for case in _cases(adapter):
            _merge_evidence(evidence, family, case, source_kind="representative_adapter")
    for family, builder in PER_OPTIMIZER_NATIVE_BUILDERS:
        report = builder(write_artifact=True)
        for case in _cases(report):
            _merge_evidence(evidence, family, case, source_kind="per_optimizer_native_training")
    return evidence


def _coverage_row(selector_row: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    name = _name(selector_row)
    family = str(selector_row.get("native_route_family") or "")
    actual = _native_step(evidence) and _native_kernel(evidence)
    representative = bool(evidence.get("representative_native_training_mapped", False))
    precondition = bool(evidence.get("runtime_precondition_rehearsal_ready", False))
    resume = selector_row.get("resume_proven") is True
    return {
        "schema_version": 1,
        "optimizer_name": name,
        "native_route_family": family,
        "trainer_resume_parity_proven": resume,
        "per_optimizer_native_training_executed": actual,
        "native_step_executed": actual,
        "native_kernel_launched": actual,
        "representative_native_training_mapped": representative,
        "runtime_precondition_rehearsal_ready": precondition,
        "runtime_dispatch_rehearsal_ready": bool(evidence.get("runtime_dispatch_rehearsal_ready", False)),
        "actual_training_status": _status(actual, representative, precondition, resume),
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "source_scorecards": sorted(set(str(item) for item in evidence.get("source_scorecards", []) if item)),
        "next_gate": (
            "per_optimizer_native_training_complete"
            if actual
            else f"add_{family}_per_optimizer_native_training"
        ),
    }


def _merge_evidence(
    evidence: dict[str, dict[str, Any]],
    family: str,
    case: Mapping[str, Any],
    *,
    source_kind: str,
) -> None:
    name = _name(case)
    if not name:
        return
    target = evidence.setdefault(
        name,
        {
            "selected_optimizer_family": family,
            "source_scorecards": [],
            "native_step_executed": False,
            "native_kernel_launched": False,
            "representative_native_training_mapped": False,
            "runtime_precondition_rehearsal_ready": False,
            "runtime_dispatch_rehearsal_ready": False,
        },
    )
    target["source_scorecards"].append(str(case.get("source_scorecard") or source_kind))
    target["runtime_precondition_rehearsal_ready"] = bool(
        target["runtime_precondition_rehearsal_ready"]
        or case.get("runtime_precondition_rehearsal_ready") is True
    )
    target["runtime_dispatch_rehearsal_ready"] = bool(
        target["runtime_dispatch_rehearsal_ready"]
        or case.get("runtime_dispatch_rehearsal_ready") is True
    )
    target["native_step_executed"] = bool(target["native_step_executed"] or _case_native_step(case))
    target["native_kernel_launched"] = bool(target["native_kernel_launched"] or _case_native_kernel(case))
    representative = (
        case.get("representative_native_step_executed") is True
        or case.get("per_optimizer_native_math_launch_executed") is True
    )
    target["representative_native_training_mapped"] = bool(
        target["representative_native_training_mapped"] or representative
    )


def _case_native_step(case: Mapping[str, Any]) -> bool:
    if "native_step_executed" in case:
        return case.get("native_step_executed") is True
    return int(case.get("native_step_count", 0) or 0) > 0


def _case_native_kernel(case: Mapping[str, Any]) -> bool:
    if "native_kernel_launched" in case:
        return case.get("native_kernel_launched") is True
    return int(case.get("native_kernel_launch_count", 0) or 0) > 0


def _native_step(evidence: Mapping[str, Any]) -> bool:
    return evidence.get("native_step_executed") is True


def _native_kernel(evidence: Mapping[str, Any]) -> bool:
    return evidence.get("native_kernel_launched") is True


def _status(actual: bool, representative: bool, precondition: bool, resume: bool) -> str:
    if actual:
        return "per_optimizer_native_training"
    if representative:
        return "representative_native_training_mapped"
    if precondition:
        return "runtime_precondition_only"
    if resume:
        return "trainer_resume_parity_only"
    return "missing_actual_training_evidence"


def _cases(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = report.get("cases")
    if isinstance(cases, list):
        return [case for case in cases if isinstance(case, Mapping)]
    rows = report.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _name(row: Mapping[str, Any]) -> str:
    return str(row.get("selected_optimizer_name") or row.get("optimizer_name") or "").strip().lower()


def _count_by(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "ok": report.get("ok") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "resume_proven_count": int(summary.get("resume_proven_count", 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "missing_classification_count": int(report.get("missing_classification_count", 0) or 0),
        "route_family_counts": _as_dict(summary.get("route_family_counts")),
    }


def _promotion_blockers(gap: int) -> list[str]:
    blockers = []
    if gap:
        blockers.append("plugin_selected_actual_training_incomplete")
    blockers.extend(
        [
            "owner_release_review_missing",
            "product_training_route_not_bound",
        ]
    )
    return blockers


def _recommended_next_step(actual_count: int, total: int, gap: int) -> str:
    if gap:
        return (
            f"expand per-optimizer native training coverage from {actual_count}/{total} "
            "by promoting representative/precondition families one route family at a time"
        )
    return "keep actual-training matrix as release evidence; continue owner/release approval and product exposure gates"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ARTIFACT_NAME",
    "EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT",
    "ROADMAP",
    "build_plugin_actual_training_coverage_scorecard",
]
