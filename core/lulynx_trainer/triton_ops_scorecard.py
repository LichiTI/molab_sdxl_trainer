"""Promotion scorecard for the Lulynx Triton fused-LoRA acceleration path.

Unlike a report-only coverage scorecard, the fused path is wired into the
trainer (default-off) and is a real, opt-in training route. base_lora uses a
"split" design (cuDNN owns the frozen base GEMM; Triton fuses only the
memory-bound LoRA path), so its value is two-sided and measured separately:

* **forward / inference** — the fused kernel removes the small LoRA GEMM
  launches and an HBM round-trip of the low-rank hidden, a clear win.
* **training (fwd+bwd)** — the backward is dominated by the unavoidable
  ``grad_x = grad_y @ base_weight`` GEMM that eager also pays, so fusion cannot
  speed it up; the bar here is therefore "no regression", not a speedup.

Promotion gates on: at least one layer patched, forward+gradient correctness
proven against an fp32/eager oracle, a forward speedup at or above
``target_forward_speedup``, and no training regression.

Clean-room Lulynx module; shares no source with any reference implementation.
"""

from __future__ import annotations

from typing import Any

# base_lora fuses only the memory-bound LoRA path, so the honest forward bar is
# modest; the mega-fusion phases raise it later.
TARGET_FORWARD_SPEEDUP = 1.10
# Training is backward-GEMM bound; "regression-free" (within micro-bench noise
# of break-even) is the promotion bar, not a net speedup.
MIN_TRAINING_SPEEDUP = 0.98


def build_triton_ops_scorecard(
    *,
    layers_patched: int = 0,
    numerical_verified: bool = False,
    gradients_verified: bool = False,
    performance_measured: bool = False,
    forward_speedup: float | None = None,
    training_speedup: float | None = None,
    target_forward_speedup: float = TARGET_FORWARD_SPEEDUP,
    min_training_speedup: float = MIN_TRAINING_SPEEDUP,
    gpu_name: str | None = None,
    is_ada_or_newer: bool | None = None,
    qkv_fused: int = 0,
    adaln_blocks: int = 0,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the fused-LoRA promotion scorecard.

    ``ok`` / ``promotion_ready`` are True only when every gate passes. The fused
    path never changes default behavior, so ``default_behavior_changed`` is
    always False regardless of readiness.
    """
    blockers: list[str] = []
    if layers_patched <= 0:
        blockers.append("no_layers_patched")
    if not numerical_verified:
        blockers.append("forward_correctness_not_verified")
    if not gradients_verified:
        blockers.append("gradient_correctness_not_verified")
    if not performance_measured:
        blockers.append("performance_not_measured")
    else:
        if training_speedup is not None and training_speedup < min_training_speedup:
            blockers.append(f"training_regression:{training_speedup:.3f}<{min_training_speedup:.3f}")
        if forward_speedup is not None and forward_speedup < target_forward_speedup:
            blockers.append(
                f"forward_speedup_below_target:{forward_speedup:.3f}<{target_forward_speedup:.3f}"
            )

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "triton_ops_fused_lora_v0",
        "gate": "triton_fused_lora_acceleration",
        "ok": ready,
        "optimization_ready": ready,
        "promotion_ready": ready,
        "training_path_enabled": True,
        "trainer_wiring_allowed": True,
        "default_behavior_changed": False,
        "kernel": "base_lora",
        "layers_patched": int(layers_patched),
        "qkv_fused": int(qkv_fused),
        "adaln_blocks": int(adaln_blocks),
        "components": dict(components) if components else None,
        "numerical_verified": bool(numerical_verified),
        "gradients_verified": bool(gradients_verified),
        "performance_measured": bool(performance_measured),
        "forward_speedup": (float(forward_speedup) if forward_speedup is not None else None),
        "training_speedup": (float(training_speedup) if training_speedup is not None else None),
        "target_forward_speedup": float(target_forward_speedup),
        "min_training_speedup": float(min_training_speedup),
        "gpu": gpu_name,
        "is_ada_or_newer": is_ada_or_newer,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "ready: base_lora fusion may ship default-off (forward speedup, training-neutral)"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "Opt-in via config 'triton_ops_enabled'; default behavior is unchanged.",
            "Backward defaults to bf16 (eager-identical accuracy); 'triton_ops_fp32_backward' opts into the slower high-accuracy path.",
            "Every patched layer self-falls-back to the eager forward on any kernel error.",
        ],
    }
