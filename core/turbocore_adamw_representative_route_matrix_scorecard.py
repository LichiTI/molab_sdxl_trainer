"""Representative route matrix for the exact AdamW native update route.

This report-only gate aggregates the existing exact AdamW V3 short matrix into
an optimizer-family coverage artifact.  It proves the route has baseline/off
and explicit canary evidence while keeping product/native dispatch default-off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_v3_exact_adamw_short_matrix_scorecard import (
    NATIVE_BACKEND,
    OPTIMIZER_KIND,
    build_v3_exact_adamw_short_matrix_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adamw_representative_route_matrix_scorecard.json"


def build_adamw_representative_route_matrix_scorecard(
    *,
    short_matrix_report: Mapping[str, Any] | None = None,
    steps: int = 4,
    run_live_training: bool = True,
    write_artifact: bool = True,
) -> dict[str, Any]:
    short_matrix = dict(
        short_matrix_report
        or build_v3_exact_adamw_short_matrix_scorecard(
            steps=steps,
            run_live_training=run_live_training,
        )
    )
    summary = _short_matrix_summary(short_matrix)
    route_rows = _route_rows(summary)
    progress_gates = _progress_gates(summary)
    ready = all(progress_gates.values())
    blockers = _blockers(progress_gates)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_representative_route_matrix_scorecard_v0",
        "gate": "adamw_representative_route_matrix",
        "ok": ready,
        "representative_route_matrix_ready": ready,
        "promotion_ready": False,
        "manual_review_required": True,
        "optimizer_type": "AdamW",
        "optimizer_kind": OPTIMIZER_KIND,
        "native_backend": NATIVE_BACKEND,
        "route": NATIVE_BACKEND,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "route_rows": route_rows,
        "summary": {
            **summary,
            "route_row_count": len(route_rows),
            "route_row_ready_count": sum(1 for row in route_rows if row["ready"]),
            "product_native_ready_count": 0,
        },
        "progress_gates": progress_gates,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": (
            "record explicit owner/release approval for exact AdamW native dispatch"
            if ready
            else "complete exact AdamW representative route matrix evidence"
        ),
        "notes": [
            "This matrix is coverage evidence only; it does not enable product dispatch.",
            "The explicit canary path remains separate from default training behavior.",
            "Owner/release approval is still required before exact AdamW native dispatch can be product-ready.",
        ],
    }
    if write_artifact:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _short_matrix_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    gates = _as_dict(report.get("progress_gates"))
    comparison = _as_dict(report.get("comparison"))
    matrix = _as_dict(report.get("matrix"))
    cases = [case for case in matrix.get("cases", []) if isinstance(case, Mapping)]
    baseline = next((case for case in cases if case.get("case") == "baseline_off"), {})
    canary = next((case for case in cases if case.get("case") == "explicit_canary"), {})
    return {
        "short_matrix_present": bool(report),
        "short_matrix_ok": report.get("ok") is True,
        "short_matrix_ready": report.get("short_matrix_ready") is True,
        "baseline_default_off": gates.get("baseline_default_off") is True,
        "canary_native_steps": gates.get("canary_native_steps") is True,
        "canary_route_executed": int(_as_dict(canary).get("native_steps", 0) or 0) > 0
        and NATIVE_BACKEND in set(_as_dict(canary).get("owner_backends") or []),
        "fallback_preserved": gates.get("fallback_preserved") is True,
        "state_sync_ready": gates.get("state_sync_ready") is True,
        "final_param_parity": gates.get("final_param_parity") is True,
        "metrics_recorded": gates.get("metrics_recorded") is True,
        "default_behavior_unchanged": gates.get("default_behavior_unchanged") is True,
        "parity_ok": comparison.get("parity_ok") is True,
        "max_abs_diff": float(comparison.get("max_abs_diff", 0.0) or 0.0),
        "matrix_steps": int(matrix.get("steps", 0) or 0),
        "baseline_native_steps": int(_as_dict(baseline).get("native_steps", 0) or 0),
        "canary_native_steps_count": int(_as_dict(canary).get("native_steps", 0) or 0),
        "canary_owner_backend": _first_string(_as_dict(canary).get("owner_backends")),
    }


def _route_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "route_case": "baseline_default_off",
            "ready": bool(summary.get("baseline_default_off")),
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "native_steps": int(summary.get("baseline_native_steps", 0) or 0),
        },
        {
            "route_case": "explicit_canary_native",
            "ready": bool(summary.get("canary_route_executed")),
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "owner_backend": str(summary.get("canary_owner_backend") or ""),
            "native_steps": int(summary.get("canary_native_steps_count", 0) or 0),
        },
        {
            "route_case": "fallback_state_sync_parity",
            "ready": bool(summary.get("fallback_preserved"))
            and bool(summary.get("state_sync_ready"))
            and bool(summary.get("final_param_parity"))
            and bool(summary.get("parity_ok")),
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "max_abs_diff": float(summary.get("max_abs_diff", 0.0) or 0.0),
        },
    ]


def _progress_gates(summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "short_matrix_present": bool(summary.get("short_matrix_present")) and bool(summary.get("short_matrix_ok")),
        "baseline_default_off": bool(summary.get("baseline_default_off")),
        "explicit_canary_native_backend": bool(summary.get("canary_route_executed")),
        "fallback_and_state_sync_ready": bool(summary.get("fallback_preserved"))
        and bool(summary.get("state_sync_ready")),
        "parity_ready": bool(summary.get("final_param_parity")) and bool(summary.get("parity_ok")),
        "metrics_recorded": bool(summary.get("metrics_recorded")) and int(summary.get("matrix_steps", 0) or 0) >= 2,
        "default_behavior_unchanged": bool(summary.get("default_behavior_unchanged")),
    }


def _blockers(progress_gates: Mapping[str, bool]) -> list[str]:
    return [
        f"adamw_representative_route_matrix_{gate}_missing"
        for gate, ready in progress_gates.items()
        if not ready
    ]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_string(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    return str(value or "")


__all__ = ["build_adamw_representative_route_matrix_scorecard"]
