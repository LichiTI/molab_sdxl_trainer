"""Report-only end-to-end shadow training matrix for PagedAdamW8bit."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from core.turbocore_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard import (
    build_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_paged_adamw8bit_training_tensor_binding_canary_scorecard import (
    build_paged_adamw8bit_training_tensor_binding_canary_scorecard,
)


MATRIX_KIND = "paged_adamw8bit_e2e_shadow_training_matrix_v0"
MATRIX_CASES = (
    {"case": "shadow_lora_block_4096", "numel": 4096, "shadow_step_count": 2},
    {"case": "shadow_lora_block_8192", "numel": 8192, "shadow_step_count": 2},
    {"case": "shadow_lora_block_16384", "numel": 16384, "shadow_step_count": 2},
)


def build_paged_adamw8bit_e2e_shadow_training_matrix_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Run fallback-authoritative shadow checks without updating user tensors."""

    adapter = dict(
        adapter_report
        or build_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard(
            run_live_probe=run_live_probe,
            require_live_matrix=require_live_matrix,
        )
    )
    cases = [
        _safe_case(
            case,
            lambda item=case: _run_case(
                item,
                run_live_probe=run_live_probe,
                require_live_matrix=require_live_matrix,
            ),
        )
        for case in MATRIX_CASES
    ]
    matrix_ready = all(bool(case.get("shadow_e2e_ready", False)) for case in cases)
    matrix_passed = all(str(case.get("status", "unknown")) == "passed" for case in cases)
    validations = _validations(adapter, cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_e2e_shadow_training_matrix_scorecard_v0",
        "gate": "paged_adamw8bit_e2e_shadow_training_matrix",
        "ok": ready,
        "promotion_ready": False,
        "e2e_shadow_training_matrix_ready": ready,
        "e2e_shadow_training_matrix_passed": matrix_passed,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "adapter_summary": dict(adapter.get("summary") or {}),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if str(case.get("status")) == "passed"),
            "skipped_case_count": sum(1 for case in cases if str(case.get("status")) == "skipped"),
            "failed_case_count": sum(1 for case in cases if str(case.get("status")) == "failed"),
            "e2e_shadow_training_matrix_passed": matrix_passed,
            "max_param_diff": _max_case_value(cases, "max_param_diff"),
            "max_state_float_diff": _max_case_value(cases, "max_state_float_diff"),
            "state_uint8_mismatch_count": sum(int(case.get("state_uint8_mismatch_count") or 0) for case in cases),
            "max_loss_diff": _max_case_value(cases, "loss_diff"),
            "fallback_backend_authoritative": True,
            "native_shadow_updates_original": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
                "paged_adamw8bit_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add PagedAdamW8bit explicit canary rollout policy with default off"
            if ready
            else "fix PagedAdamW8bit e2e shadow training matrix blockers"
        ),
        "notes": [
            "Each case keeps the bitsandbytes/Python update authoritative and compares native on cloned tensors.",
            "No original training parameter is updated by the native shadow path.",
            "This is still an isolated matrix, not user training dispatch.",
        ],
    }


def _run_case(
    case: Mapping[str, Any],
    *,
    run_live_probe: bool,
    require_live_matrix: bool,
) -> dict[str, Any]:
    report = build_paged_adamw8bit_training_tensor_binding_canary_scorecard(
        run_live_probe=run_live_probe,
        numel=int(case["numel"]),
    )
    live = report.get("live_probe", {}) if isinstance(report.get("live_probe"), Mapping) else {}
    status = str(live.get("status", "unknown"))
    ready = status == "passed" or (not require_live_matrix and status == "skipped")
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "numel": int(case["numel"]),
        "shadow_step_count": int(case.get("shadow_step_count", 2)),
        "status": status,
        "ok": bool(report.get("ok", False)),
        "shadow_e2e_ready": ready,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "training_path_enabled": False,
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
            "shadow_e2e_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"paged_adamw8bit_e2e_shadow_training_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _validations(
    adapter: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8p_runtime_dispatch_adapter_shadow_ready",
            bool(adapter.get("runtime_dispatch_adapter_shadow_ready", False)),
            "paged_adamw8bit_runtime_dispatch_adapter_shadow_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(adapter.get("fallback_backend_authoritative", False))
            and all(bool(case.get("fallback_backend_authoritative", False)) for case in cases),
            "paged_adamw8bit_e2e_shadow_training_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_never_updates_original",
            not bool(adapter.get("native_shadow_call_allowed", True))
            and not any(bool(case.get("native_shadow_updates_original", True)) for case in cases),
            "paged_adamw8bit_e2e_shadow_training_mutated_original",
        ),
        _validation(
            "e2e_shadow_training_matrix_ready",
            matrix_ready,
            "paged_adamw8bit_e2e_shadow_training_matrix_failed",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(adapter.get("runtime_dispatch_ready", True))
            and not bool(adapter.get("native_dispatch_allowed", True))
            and not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "paged_adamw8bit_e2e_shadow_training_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(adapter.get("training_path_enabled", True))
            and not bool(adapter.get("default_behavior_changed", True)),
            "paged_adamw8bit_e2e_shadow_training_changed_default_behavior",
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


__all__ = ["MATRIX_CASES", "build_paged_adamw8bit_e2e_shadow_training_matrix_scorecard"]
