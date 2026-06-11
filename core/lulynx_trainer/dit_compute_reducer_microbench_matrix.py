"""Unified microbench matrix for DiT compute reducer research."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .diffcr_token_compression_microbench import run_diffcr_token_compression_microbench
from .dit_blockskip_microbench import run_dit_blockskip_microbench
from .dit_local_window_attention_microbench import run_dit_local_window_attention_microbench
from .tread_token_route_microbench import run_tread_token_route_microbench


Runner = Callable[[Mapping[str, Any] | None], dict[str, Any]]


RUNNERS: dict[str, Runner] = {
    "tread": run_tread_token_route_microbench,
    "diffcr": run_diffcr_token_compression_microbench,
    "blockskip": run_dit_blockskip_microbench,
    "local_window_attention": run_dit_local_window_attention_microbench,
}


def build_dit_compute_reducer_microbench_matrix(
    configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    requested = dict(configs or {})
    rows = [_row(reducer_id, runner(requested.get(reducer_id))) for reducer_id, runner in RUNNERS.items()]
    candidate_rows = [
        row for row in rows
        if row["micro_ab_ready"] and row["kernel_acceleration_ready"] and not row["blocked_reasons"]
    ]
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            float(row["estimated_compute_reduction"]),
            float(row["observed_speedup"]),
        ),
        reverse=True,
    )
    blockers = {
        str(row["reducer_id"]): list(row["blocked_reasons"])
        for row in rows
        if row["blocked_reasons"]
    }
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_microbench_matrix_v0",
        "ok": bool(ranked),
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "row_count": len(rows),
        "rows": rows,
        "ranked_candidate_ids": [str(row["reducer_id"]) for row in ranked],
        "blocked_by_reducer": blockers,
        "recommended_next_step": (
            f"run cached Anima/Newbie A/B for {ranked[0]['reducer_id']}"
            if ranked
            else "resolve reducer blockers before trainer A/B"
        ),
    }


def _row(reducer_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
    if reducer_id == "blockskip":
        compute_fraction = float(report.get("estimated_block_compute_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_block_compute_reduction") or 0.0)
        variant_step_ms = float(report.get("skipped_step_ms") or 0.0)
        grad_norm = float(report.get("skipped_grad_norm") or 0.0)
    elif reducer_id == "local_window_attention":
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
        variant_step_ms = float(report.get("local_window_step_ms") or 0.0)
        grad_norm = float(report.get("local_window_grad_norm") or 0.0)
    elif reducer_id == "diffcr":
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
        variant_step_ms = float(report.get("compressed_step_ms") or 0.0)
        grad_norm = float(report.get("compressed_grad_norm") or 0.0)
    else:
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
        variant_step_ms = float(report.get("routed_step_ms") or 0.0)
        grad_norm = float(report.get("routed_grad_norm") or 0.0)
    blockers = list(report.get("blocked_reasons") or [])
    return {
        "reducer_id": reducer_id,
        "scorecard": str(report.get("scorecard") or ""),
        "ok": bool(report.get("ok", False)),
        "micro_ab_ready": bool(report.get("micro_ab_ready", report.get("measurement_ready", False))),
        "kernel_acceleration_ready": bool(report.get("kernel_acceleration_ready", True)),
        "full_step_ms": float(report.get("full_step_ms") or 0.0),
        "variant_step_ms": variant_step_ms,
        "observed_speedup": float(report.get("observed_speedup") or 0.0),
        "observed_loss_delta": float(report.get("observed_loss_delta") or 0.0),
        "estimated_compute_fraction": compute_fraction,
        "estimated_compute_reduction": compute_reduction,
        "variant_grad_norm": grad_norm,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
    }


__all__ = ["RUNNERS", "build_dit_compute_reducer_microbench_matrix"]
