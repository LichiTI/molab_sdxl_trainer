"""Report-only e2e shadow matrix for selected plugin Adam-like routes."""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.turbocore_plugin_adamlike_family_batch_scorecard import (
    build_plugin_adamlike_family_batch_scorecard,
)


MATRIX_KIND = "plugin_adamlike_e2e_shadow_matrix_v0"
OPTIMIZER_FAMILY = "adam_like_formula"
FALLBACK_BACKEND = "python_plugin_selected_optimizer"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_adamlike_e2e_shadow_matrix_scorecard(
    *,
    adamlike_batch_report: Mapping[str, Any] | None = None,
    include_live_canaries: bool = False,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Build a default-off selected-route matrix from canary-ready rows."""

    started = time.perf_counter()
    batch = _batch(adamlike_batch_report, include_live_canaries=include_live_canaries)
    rows = [row for row in batch.get("rows", []) if isinstance(row, Mapping)]
    cases = [_matrix_case(row) for row in rows]
    validations = _validations(batch, cases)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = bool(cases) and not failed
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard_v0",
        "gate": "plugin_adamlike_e2e_shadow_matrix",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "e2e_shadow_matrix_ready": ready,
        "e2e_shadow_matrix_passed": False,
        "report_only_matrix_scaffold_ready": ready,
        "live_shadow_matrix_executed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "adamlike_batch_summary": dict(_as_dict(batch.get("summary"))),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "ready_case_count": sum(1 for case in cases if case.get("shadow_matrix_case_ready") is True),
            "failed_case_count": 0 if ready else len(failed),
            "selected_native_canary_ready_count": int(
                _as_dict(batch.get("summary")).get("selected_native_canary_ready_count", 0) or 0
            ),
            "pending_selected_optimizer_count": int(
                _as_dict(batch.get("summary")).get("pending_selected_optimizer_count", 0) or 0
            ),
            "e2e_shadow_matrix_passed": False,
            "report_only_matrix_scaffold_ready": ready,
            "live_shadow_matrix_executed": False,
            "fallback_backend_authoritative": True,
            "native_shadow_training_mutates_authority": False,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_adamlike_live_e2e_shadow_matrix_missing",
                "plugin_adamlike_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected plugin Adam-like canary rollout policy with default off"
            if ready
            else "fix selected plugin Adam-like e2e shadow matrix blockers"
        ),
        "notes": [
            "This matrix consumes existing selected-route canaries and does not dispatch native updates.",
            "The Python/plugin optimizer remains authoritative for user training.",
            "Pending Adam-like plugin names remain in the dedicated-kernel backlog.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _batch(report: Mapping[str, Any] | None, *, include_live_canaries: bool) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    artifact = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_adamlike_family_batch_scorecard.json"
    if artifact.exists() and not include_live_canaries:
        try:
            import json

            return _as_dict(json.loads(artifact.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_adamlike_family_batch_scorecard(include_live_canaries=include_live_canaries)


def _matrix_case(row: Mapping[str, Any]) -> dict[str, Any]:
    stage_status = _as_dict(row.get("stage_status"))
    ready = row.get("selected_native_canary_ready") is True and not _row_has_unsafe_claims(row)
    return {
        "schema_version": 1,
        "case": f"selected_plugin_{row.get('selected_optimizer_name')}_shadow_matrix",
        "selected_optimizer_name": str(row.get("selected_optimizer_name", "")),
        "native_route": str(row.get("native_route", "")),
        "route_kind": str(row.get("route_kind", "")),
        "status": "report_only",
        "ok": ready,
        "shadow_matrix_case_ready": ready,
        "stage_status": dict(stage_status),
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "native_shadow_training_mutates_authority": False,
        "native_call_performed": False,
        "kernel_executed": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "blocked_reasons": [] if ready else [f"plugin_adamlike_shadow_matrix_case_not_ready:{row.get('selected_optimizer_name')}"],
    }


def _validations(batch: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _validation(
            "adamlike_family_batch_ready",
            batch.get("selected_adamlike_family_batch_ready") is True,
            "plugin_adamlike_family_batch_not_ready",
        ),
        _validation(
            "all_target_cases_ready",
            bool(cases) and all(case.get("shadow_matrix_case_ready") is True for case in cases),
            "plugin_adamlike_shadow_matrix_cases_not_ready",
        ),
        _validation(
            "fallback_backend_authoritative",
            all(case.get("fallback_backend_authoritative") is True for case in cases),
            "plugin_adamlike_shadow_matrix_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_does_not_mutate_authority",
            all(case.get("native_shadow_training_mutates_authority") is False for case in cases),
            "plugin_adamlike_shadow_matrix_mutated_authority",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            batch.get("runtime_dispatch_ready") is not True
            and batch.get("native_dispatch_allowed") is not True
            and batch.get("training_path_enabled") is not True
            and all(case.get("training_path_enabled") is False for case in cases),
            "plugin_adamlike_shadow_matrix_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            batch.get("default_behavior_changed") is not True
            and all(case.get("default_behavior_changed") is False for case in cases),
            "plugin_adamlike_shadow_matrix_changed_default_behavior",
        ),
    ]


def _row_has_unsafe_claims(row: Mapping[str, Any]) -> bool:
    if row.get("training_path_enabled") is True:
        return True
    if row.get("default_behavior_changed") is True:
        return True
    if row.get("runtime_dispatch_ready") is True:
        return True
    return row.get("native_dispatch_allowed") is True


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


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["MATRIX_KIND", "build_plugin_adamlike_e2e_shadow_matrix_scorecard"]
