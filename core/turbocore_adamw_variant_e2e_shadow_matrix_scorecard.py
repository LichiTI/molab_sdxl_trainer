"""Report-only e2e shadow matrix for AdamW variant canary-ready rows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.turbocore_adamw_variant_family_batch_scorecard import build_adamw_variant_family_batch_scorecard


MATRIX_KIND = "adamw_variant_e2e_shadow_matrix_v0"
FALLBACK_BACKEND = "existing_variant_optimizer"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adamw_variant_e2e_shadow_matrix_scorecard(
    *,
    adamw_variant_batch_report: Mapping[str, Any] | None = None,
    include_live_canaries: bool = False,
) -> dict[str, Any]:
    """Build a default-off matrix for native-canary-ready AdamW variants."""

    started = time.perf_counter()
    batch = _batch(adamw_variant_batch_report, include_live_canaries=include_live_canaries)
    ready_rows = [
        row
        for row in batch.get("rows", [])
        if isinstance(row, Mapping) and row.get("native_ready") is True
    ]
    cases = [_matrix_case(row) for row in ready_rows]
    validations = _validations(batch, cases)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = bool(cases) and not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_variant_e2e_shadow_matrix_scorecard_v0",
        "gate": "adamw_variant_e2e_shadow_matrix",
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
        "adamw_variant_batch_summary": dict(_as_dict(batch.get("summary"))),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "ready_case_count": sum(1 for case in cases if case.get("shadow_matrix_case_ready") is True),
            "native_ready_variant_count": int(_as_dict(batch.get("summary")).get("native_ready_count", 0) or 0),
            "pending_variant_count": int(_as_dict(batch.get("summary")).get("pending_count", 0) or 0),
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
                "adamw_variant_live_e2e_shadow_matrix_missing",
                "adamw_variant_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add AdamW variant canary rollout policy with default off"
            if ready
            else "fix AdamW variant e2e shadow matrix blockers"
        ),
        "notes": [
            "This matrix covers native-canary-ready AdamW variants only.",
            "AdamWScheduleFree is included after its native TrainingLoop canary passes.",
            "No runtime dispatch or training-path dispatch is enabled.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _batch(report: Mapping[str, Any] | None, *, include_live_canaries: bool) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    artifact = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adamw_variant_family_batch_scorecard.json"
    if artifact.exists() and not include_live_canaries:
        try:
            return _as_dict(json.loads(artifact.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=include_live_canaries)


def _matrix_case(row: Mapping[str, Any]) -> dict[str, Any]:
    ready = row.get("native_ready") is True and not _row_has_unsafe_claims(row)
    return {
        "schema_version": 1,
        "case": f"{row.get('optimizer_type')}_shadow_matrix",
        "optimizer_type": str(row.get("optimizer_type", "")),
        "optimizer_family": str(row.get("optimizer_family", "")),
        "status": "report_only",
        "ok": ready,
        "shadow_matrix_case_ready": ready,
        "stage_ready": dict(_as_dict(row.get("stage_ready"))),
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
        "blocked_reasons": [] if ready else [f"adamw_variant_shadow_matrix_case_not_ready:{row.get('optimizer_type')}"],
    }


def _validations(batch: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = int(_as_dict(batch.get("summary")).get("native_ready_count", 0) or 0)
    return [
        _validation("adamw_variant_batch_ready", batch.get("ok") is True, "adamw_variant_family_batch_not_ready"),
        _validation("native_ready_cases_present", expected == len(cases) and expected > 0, "adamw_variant_native_ready_cases_missing"),
        _validation(
            "all_target_cases_ready",
            all(case.get("shadow_matrix_case_ready") is True for case in cases),
            "adamw_variant_shadow_matrix_cases_not_ready",
        ),
        _validation(
            "fallback_backend_authoritative",
            all(case.get("fallback_backend_authoritative") is True for case in cases),
            "adamw_variant_shadow_matrix_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_does_not_mutate_authority",
            all(case.get("native_shadow_training_mutates_authority") is False for case in cases),
            "adamw_variant_shadow_matrix_mutated_authority",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            batch.get("runtime_dispatch_ready") is not True
            and batch.get("native_dispatch_allowed") is not True
            and batch.get("training_path_enabled") is not True
            and all(case.get("training_path_enabled") is False for case in cases),
            "adamw_variant_shadow_matrix_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            batch.get("default_behavior_changed") is not True
            and all(case.get("default_behavior_changed") is False for case in cases),
            "adamw_variant_shadow_matrix_changed_default_behavior",
        ),
    ]


def _row_has_unsafe_claims(row: Mapping[str, Any]) -> bool:
    return any(row.get(field) is True for field in ("training_path_enabled", "default_behavior_changed", "runtime_dispatch_ready", "native_dispatch_allowed"))


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


__all__ = ["MATRIX_KIND", "build_adamw_variant_e2e_shadow_matrix_scorecard"]
