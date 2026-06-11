# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Promotion scorecard for the Lulynx tensor-parallel subsystem (v1).

This is a **subsystem-level** deliverable: a cleanroom Megatron-style TP toolkit
verified for correctness at ``tp_size == 1`` (single GPU), default-off, and
explicitly **not** wired into the live trainer model build yet.  Promotion to a
real sharded run is a follow-up that requires multi-rank (torchrun) validation.

Gates on: column/row parallel forward+backward bit-identical to ``nn.Linear`` at
tp=1, the column→row MLP composition matches a dense MLP, the apply pass swaps
matched Linears, and shard merging is lossless.

Clean-room Lulynx module; references no external parallelism source.
"""

from __future__ import annotations

from typing import Any


def build_multi_gpu_parallel_scorecard(
    *,
    column_parity: bool = False,
    row_parity: bool = False,
    mlp_composition_parity: bool = False,
    apply_pass_verified: bool = False,
    merge_lossless: bool = False,
    world_size_tested: int = 1,
    backend: str = "nccl",
) -> dict[str, Any]:
    blockers: list[str] = []
    if not column_parity:
        blockers.append("column_parallel_not_bit_identical_at_tp1")
    if not row_parity:
        blockers.append("row_parallel_not_bit_identical_at_tp1")
    if not mlp_composition_parity:
        blockers.append("column_row_mlp_composition_mismatch")
    if not apply_pass_verified:
        blockers.append("apply_pass_did_not_swap_linears")
    if not merge_lossless:
        blockers.append("shard_merge_not_lossless")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "multi_gpu_tensor_parallel_v0",
        "gate": "tensor_parallel_subsystem",
        "ok": ready,
        "subsystem_ready": ready,
        # Honest scope: validated subsystem, NOT yet driving a real sharded run.
        "promotion_ready": False,
        "wired_into_trainer": False,
        "default_behavior_changed": False,
        "world_size_tested": int(world_size_tested),
        "backend": backend,
        "column_parity": bool(column_parity),
        "row_parity": bool(row_parity),
        "mlp_composition_parity": bool(mlp_composition_parity),
        "apply_pass_verified": bool(apply_pass_verified),
        "merge_lossless": bool(merge_lossless),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "subsystem verified at tp=1; next: multi-rank torchrun validation before wiring into the model build"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "v1 is subsystem-level: default off, not wired into the live Anima DiT build.",
            "At tp_size==1 every collective is identity, so layers are bit-identical to nn.Linear.",
            "parallel_backend='cuda_direct' sets NCCL P2P/IPC hints; falls back to nccl if unsupported.",
        ],
    }


__all__ = ["build_multi_gpu_parallel_scorecard"]
