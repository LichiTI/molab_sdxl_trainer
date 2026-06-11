"""Report-only dtype matrix for KahanAdamW8bit native tensor binding."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.turbocore_kahan_adamw8bit_runtime_canary_scorecard import (
    build_kahan_adamw8bit_runtime_canary_scorecard,
)
from core.turbocore_kahan_adamw8bit_training_tensor_binding_canary_scorecard import (
    FLOAT_TOLERANCE,
    build_kahan_adamw8bit_training_tensor_binding_canary_scorecard,
)


MATRIX_CASES = (
    {"case": "kahan_native_fp32_4096", "numel": 4096, "dtype": torch.float32},
    {"case": "kahan_native_bf16_4096", "numel": 4096, "dtype": torch.bfloat16},
)


def build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard(
    *,
    runtime_canary_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Validate fp32 and bf16 live tensor binding canaries without dispatch."""

    runtime_canary = dict(
        runtime_canary_report
        or build_kahan_adamw8bit_runtime_canary_scorecard(native_training_mode="canary")
    )
    cases = [
        _run_case(case, runtime_canary=runtime_canary, run_live_probe=run_live_probe)
        for case in MATRIX_CASES
    ]
    matrix_ready = all(str(case.get("status")) in {"passed", "skipped"} for case in cases)
    validations = _validations(runtime_canary, cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard_v0",
        "gate": "kahan_adamw8bit_bf16_native_dtype_matrix",
        "ok": ready,
        "promotion_ready": False,
        "bf16_native_dtype_matrix_ready": ready,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "optimizer_kind": "kahan_adamw8bit",
        "optimizer_family": "adamw_quantized_kahan",
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if str(case.get("status")) == "passed"),
            "skipped_case_count": sum(1 for case in cases if str(case.get("status")) == "skipped"),
            "failed_case_count": sum(1 for case in cases if str(case.get("status")) == "failed"),
            "bf16_case_status": _case_status(cases, "bfloat16"),
            "max_param_diff": _max_case_value(cases, "max_param_diff"),
            "max_absmax_diff": _max_case_value(cases, "max_absmax_diff"),
            "max_kahan_comp_diff": _max_case_value(cases, "max_kahan_comp_diff"),
            "quantized_state_mismatch_count": sum(
                int(case.get("quantized_state_mismatch_count") or 0) for case in cases
            ),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_checkpoint_resume_adapter_missing",
                "kahan_adamw8bit_real_training_matrix_missing",
                "kahan_adamw8bit_runtime_dispatch_disabled_pending_review",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit checkpoint/resume adapter before real-training matrix"
            if ready
            else "fix KahanAdamW8bit bf16 native dtype matrix blockers"
        ),
        "notes": [
            "This matrix covers fp32 and bf16 isolated live tensor binding canaries.",
            "The bf16 path uses a native CUDA bf16 parameter/gradient kernel and keeps Kahan compensation in fp32.",
            "Native optimizer dispatch remains disabled.",
        ],
    }


def _run_case(
    case: Mapping[str, Any],
    *,
    runtime_canary: Mapping[str, Any],
    run_live_probe: bool,
) -> dict[str, Any]:
    report = build_kahan_adamw8bit_training_tensor_binding_canary_scorecard(
        runtime_canary_report=runtime_canary,
        run_live_probe=run_live_probe,
        numel=int(case["numel"]),
        dtype=case["dtype"],
    )
    live = dict(report.get("live_probe") or {})
    status = str(live.get("status", "unknown"))
    ok = bool(report.get("ok", False)) and status in {"passed", "skipped"}
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "numel": int(case["numel"]),
        "dtype": str(case["dtype"]).replace("torch.", ""),
        "status": status if ok else "failed",
        "ok": ok,
        "kernel_executed": bool(live.get("kernel_executed", False)),
        "training_tensor_binding_parity_passed": bool(live.get("training_tensor_binding_parity_passed", False)),
        "e2e_no_regression_passed": bool(live.get("e2e_no_regression_passed", False)),
        "max_param_diff": live.get("max_param_diff"),
        "max_absmax_diff": live.get("max_absmax_diff"),
        "max_kahan_comp_diff": live.get("max_kahan_comp_diff"),
        "quantized_state_mismatch_count": live.get("quantized_state_mismatch_count"),
        "param_tolerance": _param_tolerance(live),
        "state_tolerance": FLOAT_TOLERANCE,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else _dedupe(list(report.get("blocked_reasons", []) or [])),
    }


def _validations(
    runtime_canary: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8u_runtime_canary_manifest_ready",
            bool(runtime_canary.get("runtime_canary_manifest_ready", False)),
            "kahan_adamw8bit_runtime_canary_manifest_missing",
        ),
        _validation(
            "bf16_native_dtype_matrix_ready",
            matrix_ready,
            "kahan_adamw8bit_bf16_native_dtype_matrix_failed",
        ),
        _validation(
            "quantized_state_exact",
            sum(int(case.get("quantized_state_mismatch_count") or 0) for case in cases) == 0,
            "kahan_adamw8bit_bf16_native_dtype_matrix_quantized_state_mismatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "kahan_adamw8bit_bf16_native_dtype_matrix_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _case_status(cases: Sequence[Mapping[str, Any]], dtype: str) -> str:
    for case in cases:
        if str(case.get("dtype")) == dtype:
            return str(case.get("status", "unknown"))
    return "missing"


def _max_case_value(cases: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in cases if case.get(key) is not None]
    return max(values) if values else None


def _param_tolerance(live: Mapping[str, Any]) -> float:
    compare = live.get("param_compare")
    if isinstance(compare, Mapping) and compare.get("tolerance") is not None:
        return float(compare["tolerance"])
    return FLOAT_TOLERANCE


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "MATRIX_CASES",
    "build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard",
]
