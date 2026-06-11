"""Report-only end-to-end shadow matrix for built-in Muon."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.turbocore_muon_training_loop_canary_scorecard import (
    build_muon_training_loop_canary_scorecard,
)


FALLBACK_BACKEND = "python_muon_optimizer"
MATRIX_KIND = "muon_model_shape_aware_e2e_shadow_matrix_v0"
MATRIX_SHAPES = ((4, 4), (8, 4))
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_muon_e2e_shadow_matrix_scorecard(
    *,
    training_loop_canary_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Record a Muon e2e shadow matrix contract without enabling dispatch."""

    started = time.perf_counter()
    loop = _as_dict(training_loop_canary_report or build_muon_training_loop_canary_scorecard())
    rows = [_row(loop)]
    cases = [_matrix_case(rows[0], shape, loop) for shape in MATRIX_SHAPES]
    validations = _validations(loop, rows, cases)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_e2e_shadow_matrix_scorecard_v0",
        "gate": "muon_model_shape_aware_e2e_shadow_matrix",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "e2e_shadow_matrix_ready": ready,
        "e2e_shadow_matrix_passed": False,
        "report_only_matrix_scaffold_ready": ready,
        "live_shadow_matrix_executed": False,
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "product_native_ready": False,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_family": "built_in_muon_model_shape_aware",
        "training_loop_summary": _as_dict(loop.get("summary")),
        "rows": rows,
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "optimizer_count": len(rows),
            "case_count": len(cases),
            "report_only_case_count": sum(1 for case in cases if case.get("status") == "report_only"),
            "failed_case_count": sum(1 for case in cases if case.get("status") == "failed"),
            "e2e_shadow_matrix_ready_count": sum(1 for row in rows if row["e2e_shadow_matrix_ready"] is True),
            "e2e_shadow_matrix_passed": False,
            "report_only_matrix_scaffold_ready": ready,
            "live_shadow_matrix_executed": False,
            "fallback_backend_authoritative": True,
            "native_shadow_updates_original": False,
            "native_shadow_training_mutates_authority": False,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "muon_live_e2e_shadow_matrix_missing",
                "muon_canary_rollout_policy_missing",
                "muon_owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Muon explicit canary rollout policy with dispatch still default-off"
            if ready
            else "fix Muon e2e shadow matrix blockers"
        ),
        "notes": [
            "This matrix records the Muon e2e shadow contract only; no native optimizer update is called here.",
            "TrainingLoop native canary evidence remains the prerequisite for this scaffold.",
            "The Python Muon optimizer remains the authoritative update backend.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _row(loop: Mapping[str, Any]) -> dict[str, Any]:
    source = _as_dict(next((row for row in loop.get("rows", []) if isinstance(row, Mapping)), {}))
    ready = source.get("training_loop_canary_ready") is True
    return {
        "schema_version": 1,
        "optimizer_type": "Muon",
        "family": "built_in_muon_model_shape_aware",
        "e2e_shadow_matrix_ready": ready,
        "training_loop_canary_ready": ready,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "next_gate": "muon_canary_rollout_policy",
        "blocked_reasons": [] if ready else ["Muon_muon_training_loop_canary_missing"],
    }


def _matrix_case(row: Mapping[str, Any], shape: tuple[int, int], loop: Mapping[str, Any]) -> dict[str, Any]:
    case = _training_loop_case(loop)
    ready = row.get("e2e_shadow_matrix_ready") is True and case.get("ok") is True
    rows, cols = shape
    return {
        "schema_version": 1,
        "case": f"report_only_muon_shape_{rows}x{cols}",
        "optimizer_type": "Muon",
        "family": "built_in_muon_model_shape_aware",
        "shape": [int(rows), int(cols)],
        "param_dtype": "float32",
        "grad_dtype": "float32",
        "shadow_step_count": 0,
        "status": "report_only" if ready else "failed",
        "ok": ready,
        "shadow_matrix_case_ready": ready,
        "training_loop_case_ready": case.get("ok") is True,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "native_call_performed": False,
        "kernel_executed": False,
        "runtime_dispatch_not_enabled": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "blocked_reasons": [] if ready else ["Muon_muon_e2e_shadow_matrix_case_not_ready"],
    }


def _validations(
    loop: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "training_loop_canary_ready",
            loop.get("training_loop_canary_ready") is True,
            "muon_training_loop_canary_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            all(case.get("fallback_backend_authoritative") is True for case in cases),
            "muon_e2e_shadow_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_never_updates_original",
            not any(case.get("native_shadow_updates_original") is True for case in cases)
            and not any(case.get("native_shadow_training_mutates_authority") is True for case in cases),
            "muon_e2e_shadow_mutated_original",
        ),
        _validation(
            "report_only_no_native_call",
            not any(case.get("native_call_performed") is True for case in cases)
            and not any(case.get("kernel_executed") is True for case in cases),
            "muon_e2e_shadow_native_call_performed",
        ),
        _validation(
            "e2e_shadow_matrix_scaffold_ready",
            all(row.get("e2e_shadow_matrix_ready") is True for row in rows)
            and all(case.get("shadow_matrix_case_ready") is True for case in cases),
            "muon_e2e_shadow_matrix_scaffold_missing",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            loop.get("runtime_dispatch_ready") is False
            and loop.get("native_dispatch_allowed") is False
            and loop.get("training_path_enabled") is False
            and not any(case.get("training_path_enabled") is True for case in cases),
            "muon_e2e_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            loop.get("training_path_enabled") is False
            and loop.get("default_behavior_changed") is False
            and not any(case.get("default_behavior_changed") is True for case in cases),
            "muon_e2e_shadow_changed_default_behavior",
        ),
    ]


def _training_loop_case(loop: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(next((case for case in loop.get("family_cases", []) if isinstance(case, Mapping)), {}))


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_e2e_shadow_matrix_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["MATRIX_SHAPES", "build_muon_e2e_shadow_matrix_scorecard"]
