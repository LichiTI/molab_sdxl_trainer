"""Shape-aware policy for TurboCore LoRA research candidates.

This is a research/benchmark policy, not a training dispatcher.  It prevents
the matrix tools from treating a candidate as universally useful when early
evidence says the win is shape-limited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoraCandidatePolicyDecision:
    candidate: str
    preset: str
    batch: int
    tokens: int
    width: int
    rank: int
    should_run: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "preset": self.preset,
            "batch": int(self.batch),
            "tokens": int(self.tokens),
            "width": int(self.width),
            "rank": int(self.rank),
            "should_run": bool(self.should_run),
            "reason": self.reason,
        }


def decide_lora_candidate_for_shape(
    *,
    candidate: str,
    preset: str,
    batch: int,
    tokens: int,
    width: int,
    rank: int,
    shape_policy: str = "auto",
) -> LoraCandidatePolicyDecision:
    candidate_name = str(candidate or "").strip().lower()
    preset_name = str(preset or "").strip().lower()
    policy = str(shape_policy or "auto").strip().lower().replace("-", "_")
    if policy in {"off", "none", "disabled"}:
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, "policy_disabled")
    if candidate_name == "triton_lora_delta_v3_dispatch":
        if int(rank) > 32:
            return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "rank_gt_32_unsupported")
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, "v3_dispatcher_routes_or_fallbacks")
    if candidate_name in {"triton_lora_delta_v2", "triton_lora_delta_v2_tc"}:
        if int(rank) > 32:
            return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "rank_gt_32_unsupported")
        if int(width) >= 1024:
            reason = "v2_tc_large_width_target" if candidate_name.endswith("_tc") else "v2_large_width_target"
            return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, reason)
        if int(width) <= 768:
            reason = "v2_tc_not_target_small_width" if candidate_name.endswith("_tc") else "v2_not_target_small_width"
            return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, reason)
        reason = "v2_tc_mid_width_probe_allowed" if candidate_name.endswith("_tc") else "v2_mid_width_probe_allowed"
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, reason)
    if candidate_name != "triton_lora_delta_v1":
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, "no_shape_policy_for_candidate")
    if int(rank) > 32:
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "rank_gt_32_unsupported")
    if preset_name.startswith("dit"):
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "dit_large_width_matrix_loss")
    if int(width) >= 1024:
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "width_ge_1024_matrix_loss")
    if int(width) <= 768:
        return _decision(candidate_name, preset_name, batch, tokens, width, rank, True, "matrix_positive_width_le_768")
    return _decision(candidate_name, preset_name, batch, tokens, width, rank, False, "unknown_shape_needs_benchmark")


def _decision(
    candidate: str,
    preset: str,
    batch: int,
    tokens: int,
    width: int,
    rank: int,
    should_run: bool,
    reason: str,
) -> LoraCandidatePolicyDecision:
    return LoraCandidatePolicyDecision(
        candidate=candidate,
        preset=preset,
        batch=int(batch),
        tokens=int(tokens),
        width=int(width),
        rank=int(rank),
        should_run=bool(should_run),
        reason=reason,
    )


__all__ = ["LoraCandidatePolicyDecision", "decide_lora_candidate_for_shape"]
