"""Report-only real-training matrix gate for PagedAdamW8bit."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from core.turbocore_paged_adamw8bit_dispatch_selector_dry_run_scorecard import (
    build_paged_adamw8bit_dispatch_selector_dry_run_scorecard,
)
from core.turbocore_paged_adamw8bit_training_tensor_binding_canary_scorecard import (
    build_paged_adamw8bit_training_tensor_binding_canary_scorecard,
)


MATRIX_KIND = "paged_adamw8bit_real_training_matrix_v0"
MATRIX_CASES = (
    {"case": "lora_block_matrix_4096", "numel": 4096, "param_group_count": 1},
    {"case": "lora_block_matrix_8192", "numel": 8192, "param_group_count": 1},
    {"case": "lora_block_matrix_16384", "numel": 16384, "param_group_count": 1},
)


def build_paged_adamw8bit_real_training_matrix_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Run a small matrix of isolated training-tensor native canaries."""

    selector = dict(selector_report or build_paged_adamw8bit_dispatch_selector_dry_run_scorecard())
    cases = [
        _safe_case(
            case,
            lambda item=case: _run_case(item, run_live_probe=run_live_probe),
        )
        for case in MATRIX_CASES
    ]
    matrix_probe_ready = all(str(case.get("status", "unknown")) in {"passed", "skipped"} for case in cases)
    matrix_passed = all(str(case.get("status", "unknown")) == "passed" for case in cases)
    validations = _validations(selector, cases, matrix_probe_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_real_training_matrix_scorecard_v0",
        "gate": "paged_adamw8bit_real_training_matrix",
        "ok": ready,
        "promotion_ready": False,
        "real_training_matrix_gate_ready": ready,
        "real_training_matrix_probe_ready": matrix_probe_ready,
        "real_training_matrix_passed": matrix_passed,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "selector_summary": dict(selector.get("summary") or {}),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if str(case.get("status")) == "passed"),
            "skipped_case_count": sum(1 for case in cases if str(case.get("status")) == "skipped"),
            "failed_case_count": sum(1 for case in cases if str(case.get("status")) == "failed"),
            "real_training_matrix_passed": matrix_passed,
            "max_param_diff": _max_case_value(cases, "max_param_diff"),
            "max_state_float_diff": _max_case_value(cases, "max_state_float_diff"),
            "state_uint8_mismatch_count": sum(int(case.get("state_uint8_mismatch_count") or 0) for case in cases),
            "max_loss_diff": _max_case_value(cases, "loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
                "paged_adamw8bit_canary_dispatch_manifest_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add PagedAdamW8bit canary dispatch manifest with runtime dispatch still disabled by default"
            if ready
            else "fix PagedAdamW8bit real-training matrix blockers"
        ),
        "notes": [
            "This matrix runs isolated training graphs with real autograd gradients and native optimizer kernel binding.",
            "It is still not a user training run and never enables default dispatch.",
            "Passing this gate promotes the evidence level from single synthetic canary to a small training matrix.",
        ],
    }


def _run_case(case: Mapping[str, Any], *, run_live_probe: bool) -> dict[str, Any]:
    report = build_paged_adamw8bit_training_tensor_binding_canary_scorecard(
        run_live_probe=run_live_probe,
        numel=int(case["numel"]),
    )
    live = report.get("live_probe", {}) if isinstance(report.get("live_probe"), Mapping) else {}
    status = str(live.get("status", "unknown"))
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "numel": int(case["numel"]),
        "param_group_count": int(case.get("param_group_count", 1)),
        "status": status,
        "ok": bool(report.get("ok", False)),
        "kernel_executed": bool(live.get("kernel_executed", False)),
        "training_tensor_binding_parity_passed": bool(
            live.get("training_tensor_binding_parity_passed", False)
        ),
        "e2e_no_regression_passed": bool(live.get("e2e_no_regression_passed", False)),
        "max_param_diff": live.get("max_param_diff"),
        "max_state_float_diff": live.get("max_state_float_diff"),
        "state_uint8_mismatch_count": live.get("state_uint8_mismatch_count"),
        "loss_diff": live.get("loss_diff"),
        "blocked_reasons": list(live.get("blocked_reasons", []) or [])
        + list(report.get("blocked_reasons", []) or []),
    }


def _safe_case(case: Mapping[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return payload
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": str(case.get("case") or "unknown"),
            "numel": int(case.get("numel") or 0),
            "status": "failed",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"paged_adamw8bit_real_training_matrix_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _validations(
    selector: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_probe_ready: bool,
) -> list[dict[str, Any]]:
    route = selector.get("route_decision", {}) if isinstance(selector.get("route_decision"), Mapping) else {}
    return [
        _validation(
            "p8m_dispatch_selector_dry_run_ready",
            bool(selector.get("dispatch_selector_dry_run_ready", False)),
            "paged_adamw8bit_dispatch_selector_dry_run_missing",
        ),
        _validation(
            "selector_still_blocks_dispatch",
            route.get("decision") == "blocked_before_real_training_matrix"
            and not bool(route.get("native_dispatch_allowed", True)),
            "paged_adamw8bit_selector_enabled_dispatch_before_matrix",
        ),
        _validation(
            "real_training_matrix_probe_or_skip",
            matrix_probe_ready,
            "paged_adamw8bit_real_training_matrix_probe_failed",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(selector.get("training_path_enabled", True))
            and not any(bool(case.get("training_path_enabled", False)) for case in cases),
            "paged_adamw8bit_real_training_matrix_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _max_case_value(cases: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in cases if case.get(key) is not None]
    return max(values) if values else None


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["MATRIX_CASES", "build_paged_adamw8bit_real_training_matrix_scorecard"]
