"""Shared post-input JSON refresh command contracts for GPU bubble.

The sequences here are JSON-only bookkeeping chains. They describe refresh
ordering only and do not define any training, CUDA, or GPU-heavy entry point.
"""

from __future__ import annotations


POST_INPUT_REFRESH_SEQUENCE: tuple[str, ...] = (
    "refresh_external_input_intake_registry",
    "refresh_external_input_replay_plan",
    "refresh_source_axis_freshness_dedupe_audit",
    "refresh_source_cache_axis_identity_registry",
    "refresh_source_cache_axis_pipeline_readiness",
    "refresh_external_input_handoff_packet",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)

REPLAY_PLAN_REFRESH_SEQUENCE: tuple[str, ...] = (
    "refresh_external_input_admission",
    "refresh_source_cache_axis_manual_canary_plan",
    "refresh_external_input_intake_registry",
    "refresh_source_axis_freshness_dedupe_audit",
    "refresh_source_cache_axis_identity_registry",
    "refresh_source_cache_axis_pipeline_readiness",
    "refresh_external_input_handoff_packet",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)

TERMINAL_GUARD_REFRESH_COMMANDS: tuple[str, ...] = (
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)
