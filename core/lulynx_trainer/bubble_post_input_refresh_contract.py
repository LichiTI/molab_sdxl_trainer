"""Shared post-input JSON refresh command contracts for GPU bubble.

The sequences here are JSON-only bookkeeping chains. They describe refresh
ordering only and do not define any training, CUDA, or GPU-heavy entry point.
"""

from __future__ import annotations


POST_INPUT_REFRESH_SEQUENCE: tuple[str, ...] = (
    "refresh_external_input_intake_registry",
    "refresh_external_input_replay_plan",
    "refresh_sd15_lora512_release_gap_readiness",
    "refresh_source_axis_scout",
    "refresh_source_axis_requirement",
    "refresh_newbie_warm_cache_inventory",
    "refresh_external_input_admission",
    "refresh_source_cache_axis_admission_preflight",
    "refresh_source_cache_axis_repair_plan",
    "refresh_source_cache_axis_manual_canary_plan",
    "refresh_post_manual_evidence_rebuild_plan",
    "refresh_source_axis_freshness_dedupe_audit",
    "refresh_source_cache_axis_identity_registry",
    "refresh_source_cache_axis_pipeline_readiness",
    "refresh_external_input_handoff_packet",
    "refresh_newbie_blockskip_quality_followup_manifest",
    "refresh_newbie_blockskip_quality_stability_review",
    "refresh_newbie_blockskip_loss_curve_ab_evidence",
    "refresh_newbie_blockskip_quality_semantic_evidence",
    "refresh_newbie_internal_phase_diagnosis",
    "refresh_newbie_natural_load_gate_semantics_review",
    "refresh_newbie_compute_bound_gate_exit_policy",
    "refresh_newbie_blockskip_quality_drift_review",
    "refresh_newbie_tail8_attention_compute_review",
    "refresh_newbie_tail8_forward_anomaly_review",
    "refresh_newbie_tail8_seed2027_rerun_preflight",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)

REPLAY_PLAN_REFRESH_SEQUENCE: tuple[str, ...] = (
    "refresh_external_input_intake_registry",
    "refresh_external_input_replay_plan",
    "refresh_sd15_lora512_release_gap_readiness",
    "refresh_source_axis_scout",
    "refresh_source_axis_requirement",
    "refresh_newbie_warm_cache_inventory",
    "refresh_external_input_admission",
    "refresh_source_cache_axis_admission_preflight",
    "refresh_source_cache_axis_repair_plan",
    "refresh_source_cache_axis_manual_canary_plan",
    "refresh_post_manual_evidence_rebuild_plan",
    "refresh_source_axis_freshness_dedupe_audit",
    "refresh_source_cache_axis_identity_registry",
    "refresh_source_cache_axis_pipeline_readiness",
    "refresh_external_input_handoff_packet",
    "refresh_newbie_blockskip_quality_followup_manifest",
    "refresh_newbie_blockskip_quality_stability_review",
    "refresh_newbie_blockskip_loss_curve_ab_evidence",
    "refresh_newbie_blockskip_quality_semantic_evidence",
    "refresh_newbie_internal_phase_diagnosis",
    "refresh_newbie_natural_load_gate_semantics_review",
    "refresh_newbie_compute_bound_gate_exit_policy",
    "refresh_newbie_blockskip_quality_drift_review",
    "refresh_newbie_tail8_attention_compute_review",
    "refresh_newbie_tail8_forward_anomaly_review",
    "refresh_newbie_tail8_seed2027_rerun_preflight",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)

TERMINAL_GUARD_REFRESH_COMMANDS: tuple[str, ...] = (
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
)
