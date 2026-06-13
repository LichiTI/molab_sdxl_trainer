# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""P3 layered-tech-reserve readiness aggregator (3a + 3b + 3c).

Consolidates the three P3 reserve buckets into one report WITHOUT enabling any of
them. P3 is deliberately the "not landable on single-GPU Anima LoRA" tier, so each
bucket carries *known reserve blockers* (a missing production CUDA window kernel,
multi-rank validation pending / not wired into the trainer, a model-family route
required). Those are recorded as expected reserve boundaries, never as release
failures. Consolidation only requires every bucket to be present, default-off, and
free of *unexpected* blockers.

Buckets and the evidence id each consumes:
  * window_attention_reserve  <- dit_local_window_cached_token_ab_replay_v0  (3a)
  * arch_level_reserve        <- dit_arch_level_reserve_contract_v0          (3b)
  * multi_gpu_reserve         <- multi_gpu_tensor_parallel_v0                (3c)

Mirrors the shape of ``dit_training_frontier_readiness.py``: every safety flag
(``training_path_enabled`` / ``default_behavior_changed`` / ``promotion_ready``)
stays ``False`` and any unexpected enable is a blocker. Note the 3a window
scorecard reports ``ok=False`` by design (its only blocker is the known missing
kernel), so this gate judges reserve-readiness by presence + safety + absence of
*unexpected* blockers, never by a bucket's own ``ok`` field. Clean-room Lulynx
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# bucket -> evidence id it expects
DEFAULT_RESERVE_BUCKETS: dict[str, str] = {
    "window_attention_reserve": "dit_local_window_cached_token_ab_replay_v0",
    "arch_level_reserve": "dit_arch_level_reserve_contract_v0",
    "multi_gpu_reserve": "multi_gpu_tensor_parallel_v0",
}

# Blocker *prefixes* (the part before ":") that are expected for a reserve tier and
# must not fail consolidation. Anything else counts as an unexpected blocker.
KNOWN_RESERVE_BLOCKER_PREFIXES: frozenset[str] = frozenset(
    {
        "optimized_cuda_window_kernel_missing",
        "sparse_masked_kernel_missing",
        "multi_rank_validation_pending",
        "multi_gpu_not_wired_into_trainer",
        "new_model_family_required",
        "native_present_no_lora_patch",
        "model_family_owns_positional_encoding",
    }
)

# Boolean evidence fields whose ``False`` value signals a known reserve boundary,
# even when a scorecard reports it as a field rather than inside ``blocked_reasons``.
RESERVE_SIGNAL_WHEN_FALSE: dict[str, str] = {
    "kernel_acceleration_ready": "optimized_cuda_window_kernel_missing",
    "wired_into_trainer": "multi_gpu_not_wired_into_trainer",
}

_SAFETY_FLAGS = ("training_path_enabled", "default_behavior_changed", "promotion_ready")


@dataclass(frozen=True)
class ReserveBucketRow:
    bucket: str
    evidence_id: str
    present: bool
    safe: bool
    training_path_enabled: bool
    default_behavior_changed: bool
    promotion_ready: bool
    known_reserve_blockers: tuple[str, ...]
    unexpected_blockers: tuple[str, ...]
    recommended_next_step: str

    @property
    def ready_for_reserve(self) -> bool:
        return self.present and self.safe and not self.unexpected_blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "evidence_id": self.evidence_id,
            "present": bool(self.present),
            "safe": bool(self.safe),
            "ready_for_reserve": bool(self.ready_for_reserve),
            "training_path_enabled": bool(self.training_path_enabled),
            "default_behavior_changed": bool(self.default_behavior_changed),
            "promotion_ready": bool(self.promotion_ready),
            "known_reserve_blockers": list(self.known_reserve_blockers),
            "unexpected_blockers": list(self.unexpected_blockers),
            "recommended_next_step": self.recommended_next_step,
        }


def build_dit_layered_tech_reserve_readiness(
    evidence: Sequence[Mapping[str, Any]],
    *,
    required_buckets: Mapping[str, str] = DEFAULT_RESERVE_BUCKETS,
) -> dict[str, Any]:
    by_id = {_evidence_id(item): dict(item) for item in evidence if _evidence_id(item)}
    rows = tuple(_row(bucket, ev_id, by_id.get(ev_id)) for bucket, ev_id in required_buckets.items())

    missing = [row.bucket for row in rows if not row.present]
    unsafe_training = [row.evidence_id for row in rows if row.training_path_enabled]
    unsafe_default = [row.evidence_id for row in rows if row.default_behavior_changed]
    unsafe_promotion = [row.evidence_id for row in rows if row.promotion_ready]
    unexpected = sorted({b for row in rows for b in row.unexpected_blockers})
    known = sorted({b for row in rows for b in row.known_reserve_blockers})

    blockers: list[str] = []
    blockers.extend(f"missing_bucket:{item}" for item in missing)
    blockers.extend(f"unsafe_training_path_enabled:{item}" for item in unsafe_training)
    blockers.extend(f"default_behavior_changed:{item}" for item in unsafe_default)
    blockers.extend(f"unexpected_promotion_ready:{item}" for item in unsafe_promotion)
    blockers.extend(f"unexpected_blocker:{item}" for item in unexpected)

    consolidated = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_layered_tech_reserve_readiness_v0",
        "ok": consolidated,
        "reserve_only": True,
        "reserve_consolidated": consolidated,
        # P3 reserve never enables training, changes defaults, or promotes.
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "bucket_count": len(required_buckets),
        "bucket_ids": list(required_buckets),
        "present_count": sum(1 for row in rows if row.present),
        "rows": [row.as_dict() for row in rows],
        "known_reserve_blockers": known,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "keep all P3 buckets as reserve; promotion requires hardware-gated follow-ups "
            "(production window kernel, multi-rank torchrun, model-family route) plus real A/B"
        ),
    }


def _row(bucket: str, evidence_id: str, item: Mapping[str, Any] | None) -> ReserveBucketRow:
    if item is None:
        return ReserveBucketRow(bucket, evidence_id, False, True, False, False, False, (), (), "evidence missing")

    raw = list(item.get("blocked_reasons") or [])
    for field, signal in RESERVE_SIGNAL_WHEN_FALSE.items():
        if field in item and item.get(field) is False:
            raw.append(signal)

    seen: set[str] = set()
    known: list[str] = []
    unexpected: list[str] = []
    for blocker in raw:
        if blocker in seen:
            continue
        seen.add(blocker)
        (known if _is_known_reserve(blocker) else unexpected).append(blocker)

    flags = {flag: bool(item.get(flag, False)) for flag in _SAFETY_FLAGS}
    return ReserveBucketRow(
        bucket=bucket,
        evidence_id=evidence_id,
        present=True,
        safe=not any(flags.values()),
        training_path_enabled=flags["training_path_enabled"],
        default_behavior_changed=flags["default_behavior_changed"],
        promotion_ready=flags["promotion_ready"],
        known_reserve_blockers=tuple(known),
        unexpected_blockers=tuple(unexpected),
        recommended_next_step=str(item.get("recommended_next_step") or ""),
    )


def _is_known_reserve(blocker: str) -> bool:
    return str(blocker).split(":", 1)[0] in KNOWN_RESERVE_BLOCKER_PREFIXES


def _evidence_id(item: Mapping[str, Any]) -> str:
    for key in ("scorecard", "contract", "note", "evidence_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""
