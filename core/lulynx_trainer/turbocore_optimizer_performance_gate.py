"""Performance-first gate for TurboCore native optimizer candidates.

This module is report-only. It can mark a future native optimizer candidate as
worth deeper CUDA validation, but it never enables runtime dispatch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any


BASELINE_PRIORITY = ("torch_adamw_fused", "torch_adamw")


@dataclass(frozen=True)
class OptimizerPerformanceGateConfig:
    min_speedup_ratio: float = 1.10
    promotion_speedup_ratio: float = 1.20
    parity_abs_tol: float = 1e-4
    parity_rel_tol: float = 1e-3
    min_evidence_iters: int = 5
    min_evidence_warmup: int = 1
    promotion_iters: int = 20
    promotion_warmup: int = 5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_optimizer_performance_gate(
    probe_payload: dict[str, Any],
    config: OptimizerPerformanceGateConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Evaluate native AdamW candidates against the fastest PyTorch baseline."""

    cfg = _with_overrides(config or OptimizerPerformanceGateConfig(), overrides)
    results = [dict(row) for row in probe_payload.get("results", []) if isinstance(row, dict)]
    iters = _int_value(probe_payload.get("iters"), default=0)
    warmup = _int_value(probe_payload.get("warmup"), default=0)
    quality, warnings = _evidence_quality(iters=iters, warmup=warmup, cfg=cfg)
    stateful_gate = probe_payload.get("stateful_abi_gate") or {}
    stateful_ok = bool(stateful_gate.get("ok", False)) if isinstance(stateful_gate, dict) else False
    baseline = _select_baseline(results)
    reasons: list[str] = []

    if baseline is None:
        reasons.append("baseline_missing")
        return _result(
            cfg,
            status="baseline_missing",
            ok=False,
            promotion_ok=False,
            baseline=None,
            candidates=[],
            best_candidate=None,
            best_measured=None,
            reasons=reasons,
            quality=quality,
            warnings=warnings,
            stateful_ok=stateful_ok,
            iters=iters,
            warmup=warmup,
        )

    candidates = [
        _candidate_report(row, baseline=baseline, cfg=cfg, stateful_ok=stateful_ok)
        for row in results
        if _is_native_candidate(row)
    ]
    if not candidates:
        reasons.append("no_native_candidate")
        return _result(
            cfg,
            status="no_native_candidate",
            ok=False,
            promotion_ok=False,
            baseline=baseline,
            candidates=[],
            best_candidate=None,
            best_measured=None,
            reasons=reasons,
            quality=quality,
            warnings=warnings,
            stateful_ok=stateful_ok,
            iters=iters,
            warmup=warmup,
        )

    best_measured = max(candidates, key=lambda row: float(row.get("speedup_vs_baseline") or 0.0))
    passing = [row for row in candidates if bool(row.get("performance_gate_ok", False))]
    best_candidate = max(passing, key=lambda row: float(row.get("speedup_vs_baseline") or 0.0)) if passing else None
    promotion_ok = any(bool(row.get("promotion_gate_ok", False)) for row in candidates)
    if best_candidate is None:
        reasons.append("all_native_candidates_blocked")
        status = _blocked_status(candidates)
    else:
        status = "promotion_candidate_needs_route_validation" if promotion_ok else "candidate_promising"
        if quality != "promotion_benchmark":
            reasons.append("benchmark_evidence_not_promotion_grade")

    return _result(
        cfg,
        status=status,
        ok=best_candidate is not None,
        promotion_ok=promotion_ok,
        baseline=baseline,
        candidates=candidates,
        best_candidate=best_candidate,
        best_measured=best_measured,
        reasons=reasons,
        quality=quality,
        warnings=warnings,
        stateful_ok=stateful_ok,
        iters=iters,
        warmup=warmup,
    )


def _with_overrides(cfg: OptimizerPerformanceGateConfig, overrides: dict[str, Any]) -> OptimizerPerformanceGateConfig:
    allowed = set(cfg.as_dict())
    values = {key: value for key, value in overrides.items() if key in allowed and value is not None}
    return replace(cfg, **values) if values else cfg


def _select_baseline(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for name in BASELINE_PRIORITY:
        for row in results:
            if _name(row) == name and _is_successful_timed(row):
                return {
                    "optimizer": _name(row),
                    "step_ms": _float_value(row.get("step_ms")),
                    "state_mb": _float_value(row.get("state_mb"), allow_zero=True),
                    "parameter_mb": _float_value(row.get("parameter_mb"), allow_zero=True),
                }
    return None


def _candidate_report(
    row: dict[str, Any],
    *,
    baseline: dict[str, Any],
    cfg: OptimizerPerformanceGateConfig,
    stateful_ok: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    step_ms = _float_value(row.get("step_ms"))
    baseline_ms = _float_value(baseline.get("step_ms"))
    speedup = baseline_ms / step_ms if baseline_ms and step_ms else None
    parity_abs = _float_value(row.get("parity_max_abs_diff"), allow_zero=True)
    parity_rel = _float_value(row.get("parity_max_rel_diff"), allow_zero=True)
    if not stateful_ok:
        reasons.append("stateful_abi_gate_failed")
    if parity_abs is None or parity_rel is None:
        reasons.append("parity_missing")
    else:
        abs_ok = parity_abs <= cfg.parity_abs_tol
        rel_ok = parity_rel <= cfg.parity_rel_tol
        if not abs_ok and not rel_ok:
            reasons.append("parity_abs_rel_failed")
    if speedup is None:
        reasons.append("candidate_timing_missing")
    elif speedup < cfg.min_speedup_ratio:
        reasons.append("speedup_below_threshold")

    performance_ok = not reasons
    promotion_ok = performance_ok and speedup is not None and speedup >= cfg.promotion_speedup_ratio
    state_mb = _float_value(row.get("state_mb"), allow_zero=True)
    baseline_state_mb = _float_value(baseline.get("state_mb"), allow_zero=True)
    return {
        "optimizer": _name(row),
        "status": _candidate_status(reasons=reasons, performance_ok=performance_ok, promotion_ok=promotion_ok),
        "performance_gate_ok": performance_ok,
        "promotion_gate_ok": promotion_ok,
        "native_kernel_present": True,
        "exact_adamw_candidate": True,
        "step_ms": step_ms,
        "baseline_optimizer": baseline.get("optimizer"),
        "baseline_step_ms": baseline_ms,
        "speedup_vs_baseline": round(speedup, 4) if speedup is not None else None,
        "required_speedup_vs_baseline": cfg.min_speedup_ratio,
        "promotion_speedup_vs_baseline": cfg.promotion_speedup_ratio,
        "parity_max_abs_diff": parity_abs,
        "parity_max_rel_diff": parity_rel,
        "parity_abs_tol": cfg.parity_abs_tol,
        "parity_rel_tol": cfg.parity_rel_tol,
        "parity_policy": "pass_when_abs_or_rel_tolerance_passes",
        "state_mb": state_mb,
        "state_mb_vs_baseline_ratio": round(state_mb / baseline_state_mb, 4) if state_mb is not None and baseline_state_mb else None,
        "reasons": reasons,
    }


def _result(
    cfg: OptimizerPerformanceGateConfig,
    *,
    status: str,
    ok: bool,
    promotion_ok: bool,
    baseline: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any] | None,
    best_measured: dict[str, Any] | None,
    reasons: list[str],
    quality: str,
    warnings: list[str],
    stateful_ok: bool,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": "turbocore_optimizer_performance_gate",
        "status": status,
        "ok": bool(ok),
        "promotion_gate_ok": bool(promotion_ok),
        "training_activation_allowed": False,
        "runtime_dispatch_allowed": False,
        "baseline_priority": list(BASELINE_PRIORITY),
        "baseline_optimizer": (baseline or {}).get("optimizer"),
        "baseline_step_ms": (baseline or {}).get("step_ms"),
        "required_speedup_vs_baseline": cfg.min_speedup_ratio,
        "promotion_speedup_vs_baseline": cfg.promotion_speedup_ratio,
        "parity_tolerances": {"max_abs": cfg.parity_abs_tol, "max_rel": cfg.parity_rel_tol},
        "stateful_abi_gate_ok": bool(stateful_ok),
        "evidence_quality": quality,
        "evidence_warnings": warnings,
        "iters": int(iters),
        "warmup": int(warmup),
        "best_candidate": best_candidate,
        "best_measured_candidate": best_measured,
        "candidates": candidates,
        "reasons": reasons,
        "config": cfg.as_dict(),
        "notes": [
            "PyTorch fused AdamW is the preferred CUDA baseline when available; otherwise standard PyTorch AdamW is used.",
            "Native optimizer candidates must be exact AdamW candidates with native_kernel_present=true.",
            "This gate is performance-first and report-only; it never enables training dispatch.",
        ],
    }


def _candidate_status(*, reasons: list[str], performance_ok: bool, promotion_ok: bool) -> str:
    if promotion_ok:
        return "promotion_speed_candidate"
    if performance_ok:
        return "candidate_promising"
    if any(reason.startswith("parity_") for reason in reasons):
        return "blocked_parity"
    if "stateful_abi_gate_failed" in reasons:
        return "blocked_stateful_abi"
    if "speedup_below_threshold" in reasons:
        return "blocked_performance"
    return "blocked_incomplete_evidence"


def _blocked_status(candidates: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("status") or "") for row in candidates}
    if "blocked_parity" in statuses:
        return "blocked_parity"
    if "blocked_stateful_abi" in statuses:
        return "blocked_stateful_abi"
    if "blocked_performance" in statuses:
        return "blocked_performance"
    return "blocked_incomplete_evidence"


def _evidence_quality(*, iters: int, warmup: int, cfg: OptimizerPerformanceGateConfig) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if iters < cfg.min_evidence_iters or warmup < cfg.min_evidence_warmup:
        warnings.append("smoke_evidence_only")
        return "smoke", warnings
    if iters < cfg.promotion_iters or warmup < cfg.promotion_warmup:
        warnings.append("not_enough_iterations_for_promotion")
        return "short_benchmark", warnings
    return "promotion_benchmark", warnings


def _is_native_candidate(row: dict[str, Any]) -> bool:
    return bool(row.get("success", False)) and bool(row.get("native_kernel_present", False)) and bool(row.get("exact_adamw_candidate", False))


def _is_successful_timed(row: dict[str, Any]) -> bool:
    return bool(row.get("success", False)) and _float_value(row.get("step_ms")) is not None


def _name(row: dict[str, Any]) -> str:
    return str(row.get("optimizer") or row.get("name") or "").strip()


def _float_value(value: Any, *, allow_zero: bool = False) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if allow_zero:
        return out if out >= 0.0 else None
    return out if out > 0.0 else None


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
