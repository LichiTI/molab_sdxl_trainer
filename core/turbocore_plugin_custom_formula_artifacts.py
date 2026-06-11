"""Report-only artifact helpers for plugin custom-formula optimizer evidence."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_plugin_custom_formula_source_inventory import custom_formula_state_inventory_ready


EVIDENCE_STAGES = (
    "formula_spec",
    "state_inventory",
    "quality_guard_matrix",
    "formula_parity_matrix",
    "resume_parity_matrix",
)

BACKLOG_TIERS: dict[str, dict[str, Any]] = {
    "adaptive_moment_variant": {
        "priority": 10,
        "state_seed": ["step", "exp_avg_or_primary_moment", "exp_avg_sq_or_secondary_moment"],
        "hparam_seed": ["lr", "betas_or_momentum", "eps", "weight_decay"],
        "quality_seed": ["finite_update_required", "eps_denominator_guard", "weight_decay_order_guard"],
        "formula_seed": "adaptive moment update with optimizer-specific correction or denominator formula",
    },
    "adaptive_lr_or_bound": {
        "priority": 20,
        "state_seed": ["step", "running_metric_or_accumulator", "effective_lr_state"],
        "hparam_seed": ["lr", "eps", "lr_bound_or_clip", "weight_decay"],
        "quality_seed": ["effective_lr_clamp_guard", "finite_update_required", "state_non_negative_guard"],
        "formula_seed": "adaptive learning-rate, bound, or accumulator driven update",
    },
    "gradient_transform_or_projection": {
        "priority": 30,
        "state_seed": ["step", "projection_or_transform_state", "optional_momentum_state"],
        "hparam_seed": ["lr", "projection_or_transform_coefficients", "weight_decay"],
        "quality_seed": ["projection_norm_guard", "finite_update_required", "zero_norm_fallback"],
        "formula_seed": "gradient transform, projection, or normalization before parameter mutation",
    },
    "quality_guard_sensitive": {
        "priority": 40,
        "state_seed": ["step", "guard_or_filter_state", "optional_momentum_state"],
        "hparam_seed": ["lr", "filter_or_guard_thresholds", "weight_decay"],
        "quality_seed": ["finite_update_required", "threshold_boundary_guard", "dtype_device_guard"],
        "formula_seed": "quality-filtered update that needs explicit edge-case guards first",
    },
    "optimizer_specific_state_machine": {
        "priority": 50,
        "state_seed": ["step", "optimizer_specific_slots", "scheduler_or_phase_state"],
        "hparam_seed": ["lr", "phase_or_switch_hparams", "weight_decay"],
        "quality_seed": ["phase_transition_guard", "resume_state_shape_guard", "finite_update_required"],
        "formula_seed": "optimizer-owned state machine or phase-dependent update",
    },
    "custom_formula_untriaged": {
        "priority": 90,
        "state_seed": ["step", "unknown_state_slots"],
        "hparam_seed": ["lr", "optimizer_specific_hparams", "weight_decay"],
        "quality_seed": ["finite_update_required", "state_shape_guard", "dtype_device_guard"],
        "formula_seed": "untriaged custom formula requiring source-level spec before grouping",
    },
}

OPTIMIZER_BACKLOG_TIERS = {
    "a2grad": "adaptive_lr_or_bound",
    "adabelief": "adaptive_moment_variant",
    "adabound": "adaptive_lr_or_bound",
    "adadelta": "adaptive_lr_or_bound",
    "adagc": "gradient_transform_or_projection",
    "adai": "adaptive_moment_variant",
    "adalite": "adaptive_lr_or_bound",
    "adan": "adaptive_moment_variant",
    "adanorm": "adaptive_moment_variant",
    "adapnm": "adaptive_moment_variant",
    "adashift": "adaptive_moment_variant",
    "adasmooth": "adaptive_moment_variant",
    "adatam": "adaptive_moment_variant",
    "ademamix": "adaptive_moment_variant",
    "adopt": "adaptive_moment_variant",
    "aida": "optimizer_specific_state_machine",
    "amos": "adaptive_lr_or_bound",
    "ano": "optimizer_specific_state_machine",
    "apollo": "adaptive_lr_or_bound",
    "apollodqn": "adaptive_lr_or_bound",
    "avagrad": "adaptive_lr_or_bound",
    "bcos": "gradient_transform_or_projection",
    "conda": "gradient_transform_or_projection",
    "diffgrad": "adaptive_moment_variant",
    "emolynx": "quality_guard_sensitive",
    "emonavi": "quality_guard_sensitive",
    "fira": "quality_guard_sensitive",
    "focus": "gradient_transform_or_projection",
    "ftrl": "optimizer_specific_state_machine",
    "grams": "gradient_transform_or_projection",
    "kate": "optimizer_specific_state_machine",
    "laprop": "adaptive_moment_variant",
    "lorarite": "gradient_transform_or_projection",
    "mars": "optimizer_specific_state_machine",
    "msvag": "quality_guard_sensitive",
    "pnm": "optimizer_specific_state_machine",
    "racs": "gradient_transform_or_projection",
    "rose": "gradient_transform_or_projection",
    "scion": "gradient_transform_or_projection",
    "scionlight": "gradient_transform_or_projection",
    "simplifiedademamix": "adaptive_moment_variant",
    "sophiah": "adaptive_moment_variant",
    "splus": "optimizer_specific_state_machine",
    "srmm": "gradient_transform_or_projection",
    "stablespam": "quality_guard_sensitive",
    "swats": "optimizer_specific_state_machine",
    "tam": "optimizer_specific_state_machine",
}

FORMULA_SPEC_SKELETONS: dict[str, tuple[str, list[str], list[str], list[str]]] = {
    "a2grad": (
        "accelerated_adaptive_delta_average_update",
        ["group.step", "state.alpha_k", "state.v_k", "state.avg_grad", "state.x_k", "optional_state.v_kk"],
        ["beta", "lips", "rho", "variant", "maximize"],
        ["variant_branch_uni_inc_exp", "scalar_v_k_non_negative", "coefficient_denominator_guard"],
    ),
    "adabelief": (
        "belief_residual_adaptive_moment_update",
        ["group.step", "state.exp_avg", "state.exp_avg_var", "optional_state.max_exp_avg_var"],
        ["lr", "betas", "weight_decay", "weight_decouple", "rectify", "ams_bound", "eps", "maximize"],
        ["grad_residual_denominator_guard", "rectify_threshold_branch", "ams_bound_monotonic_guard"],
    ),
    "adabound": (
        "adam_with_dynamic_lr_bounds",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "optional_state.max_exp_avg_sq"],
        ["lr", "final_lr", "betas", "gamma", "weight_decay", "weight_decouple", "ams_bound", "eps"],
        ["lower_upper_bound_order_guard", "base_lr_scale_guard", "ams_bound_denominator_guard"],
    ),
    "adadelta": (
        "rms_delta_ratio_accumulator_update",
        ["group.step", "state.square_avg", "state.acc_delta"],
        ["lr", "rho", "weight_decay", "weight_decouple", "fixed_decay", "eps", "maximize"],
        ["rho_range_guard", "sqrt_eps_denominator_guard", "acc_delta_non_negative_guard"],
    ),
    "adagc": (
        "gradient_clipped_adam_update_with_gamma_state",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.gamma"],
        ["lr", "betas", "beta", "lambda_abs", "lambda_rel", "warmup_steps", "weight_decay", "eps"],
        ["warmup_global_norm_guard", "relative_norm_zero_guard", "gamma_scalar_device_guard"],
    ),
    "adai": (
        "adaptive_momentum_beta_from_second_moment_mean",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.beta1_prod"],
        ["lr", "betas", "dampening", "stable_weight_decay", "weight_decay", "eps", "maximize"],
        ["global_second_moment_mean_guard", "beta1_clamp_guard", "zero_parameter_size_guard"],
    ),
    "adalite": (
        "shape_aware_trust_ratio_softmax_moment_update",
        ["group.step", "state.m_avg_or_factorized_m", "state.v_avg_or_factorized_v"],
        ["lr", "betas", "g_norm_min", "ratio_min", "tau", "eps1", "eps2", "weight_decay"],
        ["rank_branch_shape_guard", "trust_ratio_norm_floor_guard", "factorized_softmax_denominator_guard"],
    ),
    "adan": (
        "adan_gradient_difference_three_moment_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.exp_avg_diff", "state.previous_grad"],
        ["lr", "betas", "max_grad_norm", "weight_decay", "weight_decouple", "eps", "maximize"],
        ["global_grad_clip_guard", "previous_grad_sign_guard", "three_beta_bias_correction_guard"],
    ),
    "adanorm": (
        "adanorm_scaled_belief_style_moment_update",
        ["group.step", "state.exp_avg", "state.exp_avg_var", "state.exp_grad_norm", "optional_state.max_exp_avg_var"],
        ["lr", "betas", "r", "weight_decay", "weight_decouple", "fixed_decay", "ams_bound", "eps", "maximize"],
        ["adanorm_grad_norm_ema_guard", "ams_bound_denominator_guard", "complex_sparse_reject_guard"],
    ),
    "adapnm": (
        "positive_negative_momentum_adaptive_denominator_update",
        ["group.step", "state.exp_avg", "state.neg_exp_avg", "state.exp_avg_sq", "optional_state.max_exp_avg_sq"],
        ["lr", "betas", "weight_decay", "weight_decouple", "fixed_decay", "ams_bound", "eps", "maximize"],
        ["odd_even_momentum_slot_swap_guard", "noise_norm_guard", "ams_bound_denominator_guard"],
    ),
    "adashift": (
        "delayed_queue_adaptive_shift_update",
        ["group.step", "state.grad_queue", "state.exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "keep_num", "reduce_func", "eps", "maximize"],
        ["grad_queue_warmup_skip_guard", "reduce_func_shape_guard", "delayed_bias_correction_guard"],
    ),
    "adasmooth": (
        "effective_ratio_param_delta_adaptive_update",
        ["group.step", "state.prev_param", "state.s", "state.n", "state.exp_avg_sq"],
        ["lr", "betas", "weight_decay", "weight_decouple", "fixed_decay", "eps", "maximize"],
        ["param_delta_zero_sum_guard", "effective_ratio_bounds_guard", "prev_param_view_guard"],
    ),
    "adatam": (
        "adaptive_torque_correlation_moment_update",
        ["group.step", "state.s", "state.exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "decay_rate", "weight_decay", "weight_decouple", "fixed_decay", "eps", "maximize"],
        ["normalize_zero_vector_guard", "torque_correlation_range_guard", "denominator_eps_guard"],
    ),
    "ademamix": (
        "fast_slow_ema_mixed_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.exp_avg_slow"],
        ["lr", "betas", "alpha", "t_alpha_beta3", "weight_decay", "weight_decouple", "fixed_decay", "eps"],
        ["alpha_beta3_schedule_guard", "slow_ema_mix_guard", "stable_adamw_rms_guard"],
    ),
    "adopt": (
        "adopt_normalized_clipped_second_moment_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "clip_lambda", "weight_decay", "weight_decouple", "fixed_decay", "foreach", "eps"],
        ["first_step_second_moment_only_guard", "normalized_gradient_clip_guard", "foreach_parity_guard"],
    ),
    "aida": (
        "aida_projected_residual_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_var", "optional_state.max_exp_avg_var"],
        ["lr", "betas", "k", "xi", "rectify", "n_sma_threshold", "weight_decay", "ams_bound", "eps"],
        ["projection_xi_denominator_guard", "rectify_threshold_branch_guard", "ams_bound_residual_guard"],
    ),
    "amos": (
        "model_scale_adaptive_weight_decay_update",
        ["group.step", "state.exp_avg_sq", "state.decay", "optional_state.exp_avg"],
        ["lr", "beta", "momentum", "extra_l2", "c_coef", "d_coef", "foreach", "eps", "maximize"],
        ["model_scale_shape_guard", "scalar_decay_non_negative_guard", "foreach_scalar_state_guard"],
    ),
    "ano": (
        "adaptive_momentum_sign_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "logarithmic_schedule", "weight_decay", "weight_decouple", "fixed_decay", "eps"],
        ["logarithmic_beta_schedule_guard", "sign_square_variance_guard", "denominator_eps_guard"],
    ),
    "apollo": (
        "apollo_quasi_newton_scaled_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.apollo_delta"],
        ["lr", "beta", "rebound", "warmup", "weight_decay", "eps", "maximize"],
        ["quasi_newton_delta_denominator_guard", "rebound_clamp_guard", "warmup_scale_guard"],
    ),
    "apollodqn": (
        "apollo_dqn_diagonal_quasi_newton_update",
        ["group.step", "state.exp_avg_grad", "state.exp_avg_sq", "state.diagonal_hessian_approx"],
        ["lr", "beta", "rebound", "weight_decay", "eps", "maximize"],
        ["diagonal_hessian_non_negative_guard", "rebound_lower_bound_guard", "dqn_denominator_eps_guard"],
    ),
    "avagrad": (
        "variance_accumulator_gradient_average_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.grad_accumulator"],
        ["lr", "betas", "rho", "weight_decay", "weight_decouple", "eps"],
        ["variance_accumulator_non_negative_guard", "rho_range_guard", "average_grad_denominator_guard"],
    ),
    "bcos": (
        "bcos_weight_gradient_projection_update",
        ["group.step", "state.exp_avg", "state.weight_norm", "state.alignment_metric"],
        ["lr", "betas", "weight_decay", "b", "eps", "maximize"],
        ["weight_norm_zero_guard", "alignment_projection_guard", "b_exponent_finite_guard"],
    ),
    "conda": (
        "conditioned_directional_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.conditioning_buffer"],
        ["lr", "betas", "condition_beta", "weight_decay", "eps", "maximize"],
        ["conditioning_buffer_shape_guard", "direction_norm_floor_guard", "condition_beta_range_guard"],
    ),
    "diffgrad": (
        "gradient_difference_friction_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.previous_grad"],
        ["lr", "betas", "weight_decay", "weight_decouple", "eps", "maximize"],
        ["previous_grad_shape_guard", "diffgrad_friction_sigmoid_guard", "denominator_eps_guard"],
    ),
    "focus": (
        "focus_sign_centered_gradient_projection_update",
        ["group.step", "state.exp_avg", "state.focus_buffer", "state.param_center"],
        ["lr", "betas", "focus_gamma", "weight_decay", "eps", "maximize"],
        ["param_center_shape_guard", "focus_gamma_range_guard", "sign_projection_dtype_guard"],
    ),
    "ftrl": (
        "follow_the_regularized_leader_state_machine",
        ["group.step", "state.z", "state.n", "state.sigma"],
        ["lr", "lr_power", "lambda1", "lambda2", "beta", "weight_decay", "eps"],
        ["l1_threshold_branch_guard", "n_accumulator_non_negative_guard", "lr_power_boundary_guard"],
    ),
    "grams": (
        "gradient_rescaled_adaptive_moment_projection_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.rescale_metric"],
        ["lr", "betas", "rescale_eps", "weight_decay", "eps", "maximize"],
        ["rescale_metric_non_negative_guard", "zero_norm_projection_guard", "moment_projection_sign_guard"],
    ),
    "kate": (
        "kate_phase_weighted_momentum_state_machine",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.phase_buffer"],
        ["lr", "betas", "kappa", "phase_steps", "weight_decay", "eps"],
        ["phase_index_resume_guard", "kappa_range_guard", "phase_buffer_shape_guard"],
    ),
    "laprop": (
        "laprop_normalized_momentum_adaptive_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "weight_decay", "weight_decouple", "centered", "eps", "maximize"],
        ["normalized_momentum_denominator_guard", "centered_variance_guard", "weight_decay_order_guard"],
    ),
    "lorarite": (
        "lora_rank_sensitive_projection_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.rank_projection_state"],
        ["lr", "betas", "rank", "projection_gap", "weight_decay", "eps"],
        ["rank_projection_shape_guard", "lora_pair_binding_guard", "projection_gap_resume_guard"],
    ),
    "mars": (
        "mars_correction_gradient_state_machine",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.previous_grad", "state.correction"],
        ["lr", "betas", "mars_type", "gamma", "weight_decay", "eps", "maximize"],
        ["mars_type_branch_guard", "previous_grad_resume_guard", "correction_norm_guard"],
    ),
    "pnm": (
        "positive_negative_momentum_state_machine",
        ["group.step", "state.pos_exp_avg", "state.neg_exp_avg", "state.exp_avg_sq"],
        ["lr", "betas", "noise_norm", "weight_decay", "weight_decouple", "eps"],
        ["positive_negative_slot_swap_guard", "noise_norm_finite_guard", "odd_even_step_resume_guard"],
    ),
    "racs": (
        "rectified_adaptive_clipping_scale_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.clip_scale"],
        ["lr", "betas", "clip_threshold", "weight_decay", "eps", "maximize"],
        ["clip_scale_non_negative_guard", "threshold_boundary_guard", "rectified_branch_guard"],
    ),
    "rose": (
        "rank_or_sign_enhanced_projection_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.projection_basis"],
        ["lr", "betas", "rank", "update_gap", "weight_decay", "eps"],
        ["projection_basis_shape_guard", "rank_zero_fallback_guard", "update_gap_resume_guard"],
    ),
    "scion": (
        "scion_signed_projection_normalized_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.norm_buffer"],
        ["lr", "betas", "scale", "weight_decay", "eps", "maximize"],
        ["norm_buffer_zero_guard", "signed_projection_dtype_guard", "scale_boundary_guard"],
    ),
    "scionlight": (
        "scion_lightweight_signed_projection_update",
        ["group.step", "state.exp_avg", "state.norm_buffer"],
        ["lr", "momentum", "scale", "weight_decay", "eps", "maximize"],
        ["light_norm_buffer_zero_guard", "momentum_range_guard", "signed_update_dtype_guard"],
    ),
    "simplifiedademamix": (
        "simplified_fast_slow_ema_mixed_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.exp_avg_slow"],
        ["lr", "betas", "alpha", "beta3", "weight_decay", "eps", "maximize"],
        ["slow_ema_schedule_guard", "alpha_mix_range_guard", "denominator_eps_guard"],
    ),
    "sophiah": (
        "sophia_hessian_clipped_adaptive_update",
        ["group.step", "state.exp_avg", "state.hessian"],
        ["lr", "betas", "rho", "weight_decay", "weight_decouple", "eps", "maximize"],
        ["hessian_non_negative_guard", "rho_clip_guard", "hessian_refresh_resume_guard"],
    ),
    "splus": (
        "splus_phase_scaled_momentum_state_machine",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.phase_scale"],
        ["lr", "betas", "phase", "scale", "weight_decay", "eps"],
        ["phase_scale_resume_guard", "phase_transition_guard", "scale_finite_guard"],
    ),
    "srmm": (
        "srmm_rescaled_momentum_projection_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.rescale_buffer"],
        ["lr", "betas", "rescale_beta", "weight_decay", "eps", "maximize"],
        ["rescale_buffer_shape_guard", "rescale_beta_range_guard", "projection_norm_floor_guard"],
    ),
    "swats": (
        "swats_adam_to_sgd_switch_state_machine",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.exp_avg2", "state.phase"],
        ["lr", "betas", "weight_decay", "weight_decouple", "eps", "maximize"],
        ["adam_sgd_phase_transition_guard", "exp_avg2_denominator_guard", "phase_resume_guard"],
    ),
    "tam": (
        "tam_phase_tensor_adaptive_momentum_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.phase_tensor"],
        ["lr", "betas", "phase_decay", "weight_decay", "weight_decouple", "eps"],
        ["phase_tensor_shape_guard", "phase_decay_range_guard", "phase_state_resume_guard"],
    ),
    "emolynx": (
        "emo_drive_sign_momentum_with_optional_shadow",
        ["group.step", "state.exp_avg", "optional_state.shadow"],
        ["lr", "betas", "use_shadow", "shadow_weight", "weight_decay", "eps", "maximize"],
        ["loss_drive_scalar_bounds", "shadow_ratio_branch", "sign_update_dtype_device"],
    ),
    "emonavi": (
        "emo_drive_adaptive_moment_with_optional_shadow",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "optional_state.shadow"],
        ["lr", "betas", "use_shadow", "shadow_weight", "weight_decay", "eps", "maximize"],
        ["loss_drive_scalar_bounds", "denominator_eps_guard", "shadow_ratio_branch"],
    ),
    "fira": (
        "adamw_with_optional_galore_projection_limiter",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "optional_state.projector", "optional_state.scaling_grad"],
        ["lr", "betas", "rank", "update_proj_gap", "scale", "projection_type", "weight_decay", "eps"],
        ["projector_shape_guard", "scaling_limiter_guard", "denominator_eps_guard"],
    ),
    "msvag": (
        "variance_factor_clamped_momentum_update",
        ["group.step", "state.exp_avg", "state.exp_avg_sq", "state.s"],
        ["lr", "beta", "maximize"],
        ["rho_upper_bound", "nan_to_zero_factor", "factor_clamp_0_1"],
    ),
    "stablespam": (
        "stable_spam_masked_norm_scaled_adaptive_update",
        ["optimizer.total_step", "group.step", "state.exp_avg", "state.exp_avg_sq", "state.m_norm_t", "state.v_norm_t", "state.m_max_t"],
        ["lr", "betas", "gamma1", "gamma2", "theta", "t_max", "eta_min", "update_proj_gap", "eps"],
        ["gradient_mask_clip", "zero_grad_norm_skip", "projection_gap_state_reset"],
    ),
}


def custom_formula_backlog_tier(name: str) -> str:
    return OPTIMIZER_BACKLOG_TIERS.get(name, "custom_formula_untriaged")


def custom_formula_hint(name: str) -> str:
    if "ada" in name or name.startswith(("apollo", "avagrad")):
        return "adaptive_custom_formula"
    if "spam" in name or name in {"focus", "fira", "scion", "scionlight"}:
        return "quality_guard_sensitive_formula"
    if name in {"ftrl", "swats", "bcos", "grams", "mars", "pnm"}:
        return "optimizer_specific_state_machine"
    return "custom_formula"


def build_custom_formula_backlog(name: str) -> dict[str, Any]:
    tier = custom_formula_backlog_tier(name)
    profile = BACKLOG_TIERS[tier]
    return {
        "schema_version": 1,
        "backlog_tier": tier,
        "priority": int(profile["priority"]),
        "backlog_ready_for_owner": True,
        "evidence_owner_status": "unassigned",
        "native_kernel_work_allowed": False,
        "formula_spec_seed": str(profile["formula_seed"]),
        "state_inventory_seed": list(profile["state_seed"]),
        "hparam_surface_seed": list(profile["hparam_seed"]),
        "quality_guard_seed": list(profile["quality_seed"]),
        "source_review_target": f"pytorch_optimizer:{name}",
        "next_gate": f"selected_plugin_{name}_evidence_artifacts_complete",
    }


def build_custom_formula_evidence_artifacts(name: str) -> dict[str, str]:
    prefix = f"selected_plugin_custom_formula/{name}"
    return {stage: f"{prefix}/{stage}_v0" for stage in EVIDENCE_STAGES}


def build_custom_formula_evidence_status(name: str) -> dict[str, str]:
    return {
        stage: "ready" if _evidence_stage_ready(name, stage) else "pending"
        for stage in EVIDENCE_STAGES
    }


def build_custom_formula_spec_artifact(name: str, backlog: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _formula_spec_ready(name):
        return None
    formula_family, state_slots, hparams, guards = FORMULA_SPEC_SKELETONS[name]
    return {
        "schema_version": 1,
        "artifact": build_custom_formula_evidence_artifacts(name)["formula_spec"],
        "status": "ready",
        "report_only": True,
        "backlog_tier": str(backlog["backlog_tier"]),
        "source_review_target": str(backlog["source_review_target"]),
        "formula_family": formula_family,
        "step_order_skeleton": [
            "skip_none_grad",
            "reject_sparse_or_unsupported_complex_as_plugin_requires",
            "apply_optimizer_specific_guard_or_transform",
            "mutate_state_slots",
            "mutate_parameter",
        ],
        "state_inventory_skeleton": list(state_slots),
        "hparam_surface_skeleton": list(hparams),
        "quality_guard_skeleton": list(guards),
        "state_inventory_status": "skeleton_only_pending_full_inventory",
        "formula_parity_status": "pending",
        "resume_parity_status": "pending",
        "native_kernel_ready": False,
    }


def build_custom_formula_evidence_plan(
    backlog: Mapping[str, Any], artifacts: Mapping[str, str], evidence_status: Mapping[str, str]
) -> list[dict[str, Any]]:
    stage_labels = {
        "formula_spec": "extract optimizer formula, step order, weight decay placement, and skip-step policy",
        "state_inventory": "enumerate state_dict keys, tensor shapes, dtype/device rules, and hparam surface",
        "quality_guard_matrix": "define finite, epsilon, clamp, zero-norm, dtype/device, and boundary guards",
        "formula_parity_matrix": "compare plugin update against an isolated reference across edge cases",
        "resume_parity_matrix": "prove state_dict save/load and next-step parity through TrainingLoop resume",
    }
    return [
        {
            "stage": stage,
            "order": index + 1,
            "status": str(evidence_status[stage]),
            "artifact": artifacts[stage],
            "owner_hint": str(backlog["backlog_tier"]),
            "acceptance": stage_labels[stage],
            "blocks_native_kernel_work": True,
        }
        for index, stage in enumerate(EVIDENCE_STAGES)
    ]


def build_custom_formula_parallel_next_actions(
    name: str,
    backlog: Mapping[str, Any],
    evidence_status: Mapping[str, str],
) -> list[str]:
    tier = str(backlog["backlog_tier"])
    action_by_stage = {
        "formula_spec": "extract_formula_spec",
        "state_inventory": "inventory_state_and_hparams",
        "quality_guard_matrix": "write_quality_guard_cases",
        "formula_parity_matrix": "build_formula_parity_smokes",
        "resume_parity_matrix": "build_resume_parity_smokes",
    }
    return [
        f"{tier}:{name}:{action}"
        for stage, action in action_by_stage.items()
        if evidence_status.get(stage) != "ready"
    ]


def custom_formula_blocked_reasons(evidence_status: Mapping[str, str]) -> list[str]:
    stage_blockers = {
        "formula_spec": "selected_plugin_custom_formula_spec_missing",
        "state_inventory": "selected_plugin_custom_state_inventory_missing",
        "quality_guard_matrix": "selected_plugin_custom_quality_guard_matrix_missing",
        "formula_parity_matrix": "selected_plugin_custom_formula_parity_matrix_missing",
        "resume_parity_matrix": "selected_plugin_custom_resume_parity_matrix_missing",
    }
    return [
        blocker
        for stage, blocker in stage_blockers.items()
        if evidence_status.get(stage) != "ready"
    ] + ["adamw_native_simple_kernel_not_reusable", "native_dispatch_gate_not_requested"]


def _formula_spec_ready(name: str) -> bool:
    return name in FORMULA_SPEC_SKELETONS


def _evidence_stage_ready(name: str, stage: str) -> bool:
    if stage == "formula_spec":
        return _formula_spec_ready(name)
    if stage == "state_inventory":
        return custom_formula_state_inventory_ready(name)
    if stage == "quality_guard_matrix":
        return _formula_spec_ready(name) and custom_formula_state_inventory_ready(name)
    return False


__all__ = [
    "BACKLOG_TIERS",
    "EVIDENCE_STAGES",
    "build_custom_formula_backlog",
    "build_custom_formula_evidence_artifacts",
    "build_custom_formula_evidence_plan",
    "build_custom_formula_evidence_status",
    "build_custom_formula_parallel_next_actions",
    "build_custom_formula_spec_artifact",
    "custom_formula_backlog_tier",
    "custom_formula_blocked_reasons",
    "custom_formula_hint",
]
