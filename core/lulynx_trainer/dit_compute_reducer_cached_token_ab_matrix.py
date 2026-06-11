"""Cached-token A/B matrix for DiT compute reducers.

This matrix compares representative cache-first latent replays for reducers
that already passed tiny microbench coverage. It remains trainer-outside and
keeps all runtime activation flags disabled.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from .diffcr_cached_token_ab import run_diffcr_cached_token_ab
from .dit_blockskip_cached_token_ab import run_dit_blockskip_cached_token_ab
from .dit_local_window_cached_token_ab import run_dit_local_window_cached_token_ab
from .tread_cached_token_ab import run_tread_cached_token_ab


CachedRunner = Callable[[torch.Tensor, Mapping[str, Any] | None], dict[str, Any]]


def build_dit_compute_reducer_cached_token_ab_matrix(
    latents: torch.Tensor | None = None,
    *,
    family: str = "anima",
    configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    requested_family = _family(family)
    replay_latents = latents if latents is not None else _default_latents(requested_family)
    requested = dict(configs or {})
    rows = [
        _row("tread", run_tread_cached_token_ab(replay_latents, _with_family(requested.get("tread"), requested_family))),
        _row("diffcr", run_diffcr_cached_token_ab(replay_latents, _with_family(requested.get("diffcr"), requested_family))),
        _row(
            "blockskip",
            run_dit_blockskip_cached_token_ab(
                replay_latents,
                _with_family(requested.get("blockskip"), requested_family),
            ),
        ),
        _row(
            "local_window_attention",
            run_dit_local_window_cached_token_ab(
                replay_latents,
                _with_family(requested.get("local_window_attention"), requested_family),
            ),
        ),
    ]
    candidate_rows = [row for row in rows if row["cached_token_ab_ready"] and not row["blocked_reasons"]]
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            float(row["estimated_compute_reduction"]),
            float(row["observed_speedup"]),
        ),
        reverse=True,
    )
    blockers = {str(row["reducer_id"]): list(row["blocked_reasons"]) for row in rows if row["blocked_reasons"]}
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_cached_token_ab_matrix_v0",
        "ok": bool(ranked),
        "family": requested_family,
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
            f"run real cached {requested_family} trainer A/B for {ranked[0]['reducer_id']}"
            if ranked
            else f"resolve cached-token reducer blockers for {requested_family}"
        ),
    }


def _row(reducer_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
    if reducer_id == "blockskip":
        family_reports = list(report.get("family_reports") or [])
        full_step_ms = _mean(row.get("full_step_ms", 0.0) for row in family_reports)
        variant_step_ms = _mean(row.get("skipped_step_ms", 0.0) for row in family_reports)
        grad_norm = _mean(row.get("skipped_grad_norm", 0.0) for row in family_reports)
        loss_delta = _mean(row.get("observed_loss_delta", 0.0) for row in family_reports)
        compute_fraction = float(report.get("estimated_block_compute_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_block_compute_reduction") or 0.0)
    elif reducer_id == "diffcr":
        full_step_ms = float(report.get("full_step_ms") or 0.0)
        variant_step_ms = float(report.get("compressed_step_ms") or 0.0)
        grad_norm = float(report.get("compressed_grad_norm") or 0.0)
        loss_delta = float(report.get("observed_loss_delta") or 0.0)
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
    elif reducer_id == "local_window_attention":
        full_step_ms = float(report.get("full_step_ms") or 0.0)
        variant_step_ms = float(report.get("local_window_step_ms") or 0.0)
        grad_norm = float(report.get("local_window_grad_norm") or 0.0)
        loss_delta = float(report.get("observed_loss_delta") or 0.0)
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
    else:
        full_step_ms = float(report.get("full_step_ms") or 0.0)
        variant_step_ms = float(report.get("routed_step_ms") or 0.0)
        grad_norm = float(report.get("routed_grad_norm") or 0.0)
        loss_delta = float(report.get("observed_loss_delta") or 0.0)
        compute_fraction = float(report.get("estimated_attention_fraction") or 1.0)
        compute_reduction = float(report.get("estimated_attention_reduction") or 0.0)
    blockers = list(report.get("blocked_reasons") or [])
    return {
        "reducer_id": reducer_id,
        "scorecard": str(report.get("scorecard") or ""),
        "ok": bool(report.get("ok", False)),
        "cached_token_ab_ready": bool(report.get("cached_token_ab_ready", False)),
        "full_step_ms": full_step_ms,
        "variant_step_ms": variant_step_ms,
        "observed_speedup": float(report.get("observed_speedup") or (full_step_ms / variant_step_ms if variant_step_ms > 0 else 0.0)),
        "observed_loss_delta": loss_delta,
        "estimated_compute_fraction": compute_fraction,
        "estimated_compute_reduction": compute_reduction,
        "variant_grad_norm": grad_norm,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
    }


def _with_family(config: Mapping[str, Any] | None, family: str) -> dict[str, Any]:
    payload = dict(config or {})
    payload["family"] = family
    return payload


def _family(value: str) -> str:
    normalized = str(value or "anima").strip().lower()
    return normalized if normalized in {"anima", "newbie"} else "anima"


def _default_latents(family: str) -> torch.Tensor:
    channels = 4
    height = 4 if family == "anima" else 3
    width = 4
    values = torch.linspace(-1.0, 1.0, steps=channels * height * width, dtype=torch.float32)
    return values.reshape(1, channels, height, width)


def _mean(values: Any) -> float:
    payload = [float(value) for value in values]
    return float(sum(payload) / max(len(payload), 1))


__all__ = ["build_dit_compute_reducer_cached_token_ab_matrix"]
