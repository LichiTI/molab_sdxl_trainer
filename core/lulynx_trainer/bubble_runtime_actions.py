"""Advisor patch plans for bubble-aware runtime actions."""

from __future__ import annotations

from typing import Any, Mapping

from .bubble_runtime_host_actions import build_host_scheduling_plan


_DATA_SUPPLY_ACTIONS = {
    "set_dataloader_workers",
    "set_dataloader_prefetch_factor",
    "keep_dataloader_workers",
    "recommend_cache_first",
}

_TRANSFER_OFFLOAD_ACTIONS = {
    "enable_pin_memory",
    "enable_non_blocking_transfer",
    "enable_block_prefetch",
    "increase_block_prefetch_depth",
    "profile_transfer_path",
}

_OPTIMIZER_ACTIONS = {
    "enable_fused_adamw",
    "enable_foreach_optimizer",
    "recommend_native_optimizer",
    "profile_native_optimizer",
    "reduce_grad_scan_frequency",
}

_WORKLOAD_ACTIONS = {
    "increase_train_batch_size",
    "increase_gradient_accumulation",
    "adjust_microbatch",
    "recommend_larger_token_or_latent_crop",
    "recommend_compile_static_shape",
    "explain_workload_underfilled",
}

_HOST_SCHEDULING_ACTIONS = {
    "reduce_hot_path_sync",
    "increase_logging_interval",
    "increase_checkpoint_interval",
    "enable_async_checkpoint_save",
    "move_validation_after_training_window",
    "disable_sync_profiler_mode",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _normalized_mode(mode: Any) -> str:
    value = str(mode or "report_only").strip().lower().replace("-", "_")
    if value in {"advisor", "advisory", "advisor_patch"}:
        return "advisor_patch"
    if value in {"auto", "auto_apply"}:
        return "auto_apply"
    return "report_only"


def _mutation(path: str, current: Any, recommended: Any, reason: str) -> dict[str, Any]:
    return {
        "op": "set",
        "path": path,
        "current": current,
        "recommended": recommended,
        "reason": reason,
    }


def _append_mutation(
    mutations: list[dict[str, Any]],
    *,
    path: str,
    current: Any,
    recommended: Any,
    reason: str,
) -> None:
    if current == recommended:
        return
    mutations.append(_mutation(path, current, recommended, reason))


def _empty_plan(
    *,
    mode: str,
    status: str,
    action_kind: str,
    reason: str,
    domain: str = "none",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P1_report_only",
        "mode": mode,
        "domain": domain,
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "report_only",
        "can_apply_to_next_request": False,
        "can_apply_during_current_run": False,
        "mutations": [],
        "config_overlay": {},
        "effective_policy": {},
        "rollback": {},
        "notes": [reason],
    }


def _data_supply_tuning_profile(action: Mapping[str, Any]) -> dict[str, Any]:
    profile = _mapping(action.get("tuning_profile"))
    return dict(profile) if profile else {}


def _data_supply_empty_plan(
    *,
    mode: str,
    status: str,
    action_kind: str,
    reason: str,
    tuning_profile: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _empty_plan(
        mode=mode,
        status=status,
        action_kind=action_kind,
        reason=reason,
        domain="data_supply",
    )
    if tuning_profile:
        plan["effective_policy"] = {"tuning_profile": dict(tuning_profile)}
    return plan


def _config_overlay(mutations: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(item["path"]): item.get("recommended") for item in mutations if item.get("op") == "set" and item.get("path")}


def _data_supply_plan(
    *,
    snapshot: Mapping[str, Any],
    action: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    action_kind = str(action.get("kind") or "")
    tuning_profile = _data_supply_tuning_profile(action)
    controller_enabled = _safe_bool(config.get("controller_enabled"), False)
    allow_worker_tuning = _safe_bool(config.get("allow_worker_tuning"), True)
    if mode == "report_only":
        return _data_supply_empty_plan(
            mode=mode,
            status="report_only",
            action_kind=action_kind,
            reason="controller mode is report_only; config patch is not materialized",
            tuning_profile=tuning_profile,
        )
    if not controller_enabled:
        return _data_supply_empty_plan(
            mode=mode,
            status="disabled",
            action_kind=action_kind,
            reason="bubble_controller_enabled is false; emitting diagnosis only",
            tuning_profile=tuning_profile,
        )
    if action_kind in {"set_dataloader_workers", "set_dataloader_prefetch_factor"} and not allow_worker_tuning:
        return _data_supply_empty_plan(
            mode=mode,
            status="blocked_by_config",
            action_kind=action_kind,
            reason="bubble_controller_allow_worker_tuning is false",
            tuning_profile=tuning_profile,
        )

    mutations: list[dict[str, Any]] = []
    workers = _safe_int(runtime.get("workers"))
    prefetch = _safe_int(runtime.get("prefetch_factor"), 2)
    pin_memory = _safe_bool(runtime.get("pin_memory"), True)
    effective_policy: dict[str, Any] = {
        "num_workers": workers,
        "prefetch_factor": prefetch if workers > 0 else None,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    if tuning_profile:
        effective_policy["tuning_profile"] = tuning_profile

    if action_kind == "set_dataloader_workers":
        recommended = max(_safe_int(action.get("recommended"), 2), 0)
        _append_mutation(
            mutations,
            path="cached_dataloader_auto_policy",
            current=runtime.get("cached_dataloader_auto_policy", True),
            recommended=True,
            reason="keep cached route policy resolver enabled for the next run",
        )
        _append_mutation(
            mutations,
            path="cached_dataloader_workers",
            current=workers,
            recommended=recommended,
            reason=str(action.get("reason") or "reduce data_wait_share"),
        )
        if recommended > 0:
            effective_policy.update(
                {
                    "num_workers": recommended,
                    "prefetch_factor": max(prefetch, 2),
                    "persistent_workers": True,
                }
            )
    elif action_kind == "set_dataloader_prefetch_factor":
        recommended = max(_safe_int(action.get("recommended"), 4), 1)
        _append_mutation(
            mutations,
            path="cached_dataloader_prefetch_factor",
            current=prefetch,
            recommended=recommended,
            reason=str(action.get("reason") or "increase producer queue depth"),
        )
        effective_policy.update(
            {
                "num_workers": workers,
                "prefetch_factor": recommended if workers > 0 else None,
                "persistent_workers": workers > 0,
            }
        )
    elif action_kind == "keep_dataloader_workers":
        return _data_supply_empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason=str(action.get("reason") or "current DataLoader worker policy should be kept"),
            tuning_profile=tuning_profile,
        )
    elif action_kind == "recommend_cache_first":
        return _data_supply_empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "data wait persists after worker tuning"),
            tuning_profile=tuning_profile,
        )

    if not mutations:
        return _data_supply_empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason="recommended data supply settings already match the current snapshot",
            tuning_profile=tuning_profile,
        )

    status = "advisor_patch_ready" if mode == "advisor_patch" else "auto_apply_blocked_pending_p7"
    data_supply_safety = {
        "vram_safe": _safe_bool(safety.get("vram_safe"), True),
        "requires_new_dataloader": True,
        "windows_workers4_default_blocked": True,
    }
    notes = [
        "P2 only materializes an advisor patch for the next request.",
        "The current training run is not mutated.",
    ]
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P2_data_supply_advisor_patch",
        "mode": mode,
        "domain": "data_supply",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "advisor_patch",
        "can_apply_to_next_request": mode == "advisor_patch",
        "can_apply_during_current_run": False,
        "mutations": mutations,
        "config_overlay": _config_overlay(mutations),
        "effective_policy": effective_policy,
        "rollback": {
            "metric": "steady_samples_per_second",
            "max_regression_ratio": 0.02,
            "compare_window": "post_warmup_steady_window",
            "restore": {item["path"]: item.get("current") for item in mutations},
        },
        "safety": data_supply_safety,
        "notes": notes,
    }


def _transfer_offload_plan(
    *,
    snapshot: Mapping[str, Any],
    action: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    action_kind = str(action.get("kind") or "")
    controller_enabled = _safe_bool(config.get("controller_enabled"), False)
    allow_transfer_prefetch = _safe_bool(config.get("allow_transfer_prefetch"), True)
    if mode == "report_only":
        return _empty_plan(
            mode=mode,
            status="report_only",
            action_kind=action_kind,
            reason="controller mode is report_only; transfer/offload patch is not materialized",
            domain="transfer_offload",
        )
    if not controller_enabled:
        return _empty_plan(
            mode=mode,
            status="disabled",
            action_kind=action_kind,
            reason="bubble_controller_enabled is false; emitting diagnosis only",
            domain="transfer_offload",
        )
    if not allow_transfer_prefetch:
        return _empty_plan(
            mode=mode,
            status="blocked_by_config",
            action_kind=action_kind,
            reason="bubble_controller_allow_transfer_prefetch is false",
            domain="transfer_offload",
        )

    mutations: list[dict[str, Any]] = []
    effective_policy: dict[str, Any] = {
        "pin_memory": _safe_bool(runtime.get("pin_memory"), True),
        "data_transfer_non_blocking": _safe_bool(runtime.get("data_transfer_non_blocking"), True),
        "offload_active": _safe_bool(runtime.get("offload_active"), False),
        "residency_source": str(runtime.get("residency_source") or ""),
        "residency_mode": str(runtime.get("residency_mode") or ""),
        "prefetch_enabled": _safe_bool(runtime.get("prefetch_enabled"), False),
        "prefetch_depth": _safe_int(runtime.get("prefetch_depth"), 0),
    }

    if action_kind == "enable_pin_memory":
        _append_mutation(
            mutations,
            path="pin_memory",
            current=runtime.get("pin_memory"),
            recommended=True,
            reason=str(action.get("reason") or "enable pinned host memory for H2D transfers"),
        )
        _append_mutation(
            mutations,
            path="cached_dataloader_pin_memory",
            current=runtime.get("pin_memory"),
            recommended=True,
            reason="keep cached route pin_memory aligned with transfer policy",
        )
        effective_policy["pin_memory"] = True
    elif action_kind == "enable_non_blocking_transfer":
        _append_mutation(
            mutations,
            path="data_transfer_non_blocking",
            current=runtime.get("data_transfer_non_blocking"),
            recommended=True,
            reason=str(action.get("reason") or "allow tensor.to(..., non_blocking=True) when safe"),
        )
        effective_policy["data_transfer_non_blocking"] = True
    elif action_kind in {"enable_block_prefetch", "increase_block_prefetch_depth"}:
        prefetch_key = str(runtime.get("prefetch_key") or "")
        depth_key = str(runtime.get("prefetch_depth_key") or "")
        if not prefetch_key or not depth_key:
            return _empty_plan(
                mode=mode,
                status="missing_residency_route",
                action_kind=action_kind,
                reason="block prefetch needs anima/newbie residency evidence before a config patch can be built",
                domain="transfer_offload",
            )
        if not _safe_bool(runtime.get("offload_active"), False):
            return _empty_plan(
                mode=mode,
                status="not_applicable",
                action_kind=action_kind,
                reason="block prefetch is only applicable when streaming/offload residency is active",
                domain="transfer_offload",
            )
        current_depth = max(_safe_int(runtime.get("prefetch_depth"), 0), 0)
        recommended_depth = max(_safe_int(action.get("recommended", action.get("depth")), 1), 1)
        _append_mutation(
            mutations,
            path=prefetch_key,
            current=_safe_bool(runtime.get("prefetch_enabled"), False),
            recommended=True,
            reason=str(action.get("reason") or "enable async block prefetch for streaming offload"),
        )
        _append_mutation(
            mutations,
            path=depth_key,
            current=current_depth,
            recommended=recommended_depth,
            reason="set the next-run block prefetch depth from transfer/offload evidence",
        )
        effective_policy.update(
            {
                "prefetch_enabled": True,
                "prefetch_depth": recommended_depth,
                "prefetch_key": prefetch_key,
                "prefetch_depth_key": depth_key,
            }
        )
    elif action_kind == "profile_transfer_path":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "collect profiler evidence before deeper transfer changes"),
            domain="transfer_offload",
        )

    if not mutations:
        return _empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason="recommended transfer/offload settings already match the current snapshot",
            domain="transfer_offload",
        )

    status = "advisor_patch_ready" if mode == "advisor_patch" else "auto_apply_blocked_pending_p7"
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P3_transfer_offload_advisor_patch",
        "mode": mode,
        "domain": "transfer_offload",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "advisor_patch",
        "can_apply_to_next_request": mode == "advisor_patch",
        "can_apply_during_current_run": False,
        "mutations": mutations,
        "config_overlay": _config_overlay(mutations),
        "effective_policy": effective_policy,
        "rollback": {
            "metric": "steady_samples_per_second",
            "secondary_metric": "h2d_transfer_share",
            "max_regression_ratio": 0.02,
            "compare_window": "post_warmup_steady_window",
            "restore": {item["path"]: item.get("current") for item in mutations},
        },
        "safety": {
            "vram_safe": _safe_bool(safety.get("vram_safe"), True),
            "requires_new_dataloader_or_residency_setup": True,
            "current_run_hot_mutation": False,
        },
        "notes": [
            "P3 only materializes a transfer/offload advisor patch for the next request.",
            "The current training run is not mutated.",
        ],
    }


def _optimizer_plan(
    *,
    snapshot: Mapping[str, Any],
    action: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    runtime = _mapping(snapshot.get("runtime"))
    action_kind = str(action.get("kind") or "")
    controller_enabled = _safe_bool(config.get("controller_enabled"), False)
    allow_optimizer_swap = _safe_bool(config.get("allow_optimizer_swap"), False)
    if mode == "report_only":
        return _empty_plan(
            mode=mode,
            status="report_only",
            action_kind=action_kind,
            reason="controller mode is report_only; optimizer patch is not materialized",
            domain="optimizer_backward",
        )
    if not controller_enabled:
        return _empty_plan(
            mode=mode,
            status="disabled",
            action_kind=action_kind,
            reason="bubble_controller_enabled is false; emitting diagnosis only",
            domain="optimizer_backward",
        )
    if action_kind in {"enable_fused_adamw", "enable_foreach_optimizer", "recommend_native_optimizer"} and not allow_optimizer_swap:
        return _empty_plan(
            mode=mode,
            status="blocked_by_config",
            action_kind=action_kind,
            reason="bubble_controller_allow_optimizer_swap is false",
            domain="optimizer_backward",
        )

    mutations: list[dict[str, Any]] = []
    optimizer_backend = str(runtime.get("optimizer_backend", "auto") or "auto").strip().lower()
    optimizer_args = str(runtime.get("optimizer_args", "") or "")
    effective_policy: dict[str, Any] = {
        "optimizer_backend": optimizer_backend,
        "optimizer_args": optimizer_args,
    }

    if action_kind == "enable_fused_adamw":
        recommended_backend = str(action.get("recommended_backend") or "torch_fused")
        if "fused=false" in optimizer_args.lower():
            return _empty_plan(
                mode=mode,
                status="blocked_by_explicit_optimizer_args",
                action_kind=action_kind,
                reason="optimizer_args explicitly contains fused=False; remove that override before enabling torch_fused",
                domain="optimizer_backward",
            )
        _append_mutation(
            mutations,
            path="optimizer_backend",
            current=optimizer_backend,
            recommended=recommended_backend,
            reason=str(action.get("reason") or "optimizer update share is high"),
        )
        effective_policy["optimizer_backend"] = recommended_backend
    elif action_kind == "enable_foreach_optimizer":
        _append_mutation(
            mutations,
            path="optimizer_backend",
            current=optimizer_backend,
            recommended="foreach_adamw",
            reason=str(action.get("reason") or "test foreach AdamW before native optimizer promotion"),
        )
        effective_policy["optimizer_backend"] = "foreach_adamw"
    elif action_kind == "recommend_native_optimizer":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "native optimizer promotion still requires turbocore gates"),
            domain="optimizer_backward",
        )
    elif action_kind == "profile_native_optimizer":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "collect native optimizer gate evidence before promotion"),
            domain="optimizer_backward",
        )
    elif action_kind == "reduce_grad_scan_frequency":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "gradient scan cadence needs a dedicated safeguard-aware patch"),
            domain="optimizer_backward",
        )

    if not mutations:
        return _empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason="recommended optimizer settings already match the current snapshot",
            domain="optimizer_backward",
        )

    status = "advisor_patch_ready" if mode == "advisor_patch" else "auto_apply_blocked_pending_p7"
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P4_optimizer_backward_advisor_patch",
        "mode": mode,
        "domain": "optimizer_backward",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "advisor_patch",
        "can_apply_to_next_request": mode == "advisor_patch",
        "can_apply_during_current_run": False,
        "mutations": mutations,
        "config_overlay": _config_overlay(mutations),
        "effective_policy": effective_policy,
        "rollback": {
            "metric": "steady_samples_per_second",
            "secondary_metric": "optimizer_share",
            "max_regression_ratio": 0.02,
            "failure_signals": ["loss_nan", "loss_inf", "optimizer_state_incompatible"],
            "compare_window": "post_warmup_steady_window",
            "restore": {item["path"]: item.get("current") for item in mutations},
        },
        "safety": {
            "requires_optimizer_rebuild": True,
            "current_run_hot_mutation": False,
            "native_optimizer_requires_turbocore_gate": True,
        },
        "notes": [
            "P4 only materializes an optimizer/backward advisor patch for the next request.",
            "The current training run is not mutated.",
        ],
    }


def _workload_plan(
    *,
    snapshot: Mapping[str, Any],
    action: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    action_kind = str(action.get("kind") or "")
    controller_enabled = _safe_bool(config.get("controller_enabled"), False)
    allow_batch_growth = _safe_bool(config.get("allow_batch_growth"), True)
    vram_safe = _safe_bool(safety.get("vram_safe"), True)
    if mode == "report_only":
        return _empty_plan(
            mode=mode,
            status="report_only",
            action_kind=action_kind,
            reason="controller mode is report_only; workload shaping patch is not materialized",
            domain="workload_shaping",
        )
    if not controller_enabled:
        return _empty_plan(
            mode=mode,
            status="disabled",
            action_kind=action_kind,
            reason="bubble_controller_enabled is false; emitting diagnosis only",
            domain="workload_shaping",
        )
    if action_kind in {"increase_train_batch_size", "increase_gradient_accumulation", "adjust_microbatch"} and not allow_batch_growth:
        return _empty_plan(
            mode=mode,
            status="blocked_by_config",
            action_kind=action_kind,
            reason="bubble_controller_allow_batch_growth is false",
            domain="workload_shaping",
        )
    if action_kind == "explain_workload_underfilled":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "workload is too light, but automatic growth is not safe"),
            domain="workload_shaping",
        )
    if action_kind in {"recommend_larger_token_or_latent_crop", "recommend_compile_static_shape"}:
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "this workload shaping change needs an expert sweep"),
            domain="workload_shaping",
        )
    if action_kind == "increase_train_batch_size" and not vram_safe:
        return _empty_plan(
            mode=mode,
            status="blocked_by_vram",
            action_kind=action_kind,
            reason="VRAM safety gate is closed; do not increase train_batch_size",
            domain="workload_shaping",
        )

    mutations: list[dict[str, Any]] = []
    batch = max(_safe_int(runtime.get("train_batch_size"), 1), 1)
    grad_accum = max(_safe_int(runtime.get("gradient_accumulation_steps"), 1), 1)
    effective_policy: dict[str, Any] = {
        "train_batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch * grad_accum,
        "priority": "throughput_first",
    }

    if action_kind == "increase_train_batch_size":
        recommended = max(_safe_int(action.get("recommended"), batch * 2), 1)
        _append_mutation(
            mutations,
            path="train_batch_size",
            current=batch,
            recommended=recommended,
            reason=str(action.get("reason") or "GPU active util is low and workload appears underfilled"),
        )
        effective_policy.update(
            {
                "train_batch_size": recommended,
                "effective_batch_size": recommended * grad_accum,
                "saturation_candidate": True,
            }
        )
    elif action_kind == "increase_gradient_accumulation":
        recommended = max(_safe_int(action.get("recommended"), grad_accum + 1), 1)
        _append_mutation(
            mutations,
            path="gradient_accumulation_steps",
            current=grad_accum,
            recommended=recommended,
            reason=str(action.get("reason") or "increase effective work without raising microbatch"),
        )
        effective_policy.update(
            {
                "gradient_accumulation_steps": recommended,
                "effective_batch_size": batch * recommended,
                "saturation_candidate": True,
            }
        )
    elif action_kind == "adjust_microbatch":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason=str(action.get("reason") or "microbatch adjustment needs route-specific memory evidence"),
            domain="workload_shaping",
        )

    if not mutations:
        return _empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason="recommended workload settings already match the current snapshot",
            domain="workload_shaping",
        )

    status = "advisor_patch_ready" if mode == "advisor_patch" else "auto_apply_blocked_pending_p7"
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P5_workload_shaping_advisor_patch",
        "mode": mode,
        "domain": "workload_shaping",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "advisor_patch",
        "can_apply_to_next_request": mode == "advisor_patch",
        "can_apply_during_current_run": False,
        "mutations": mutations,
        "config_overlay": _config_overlay(mutations),
        "effective_policy": effective_policy,
        "rollback": {
            "metric": "steady_samples_per_second",
            "secondary_metric": "active_gpu_util_pct_mean",
            "max_regression_ratio": 0.05,
            "compare_window": "post_warmup_steady_window",
            "restore": {item["path"]: item.get("current") for item in mutations},
        },
        "safety": {
            "vram_safe": vram_safe,
            "memory_ratio": safety.get("memory_ratio"),
            "max_vram_ratio": safety.get("max_vram_ratio"),
            "current_run_hot_mutation": False,
            "do_not_chase_99pct_gpu_util_blindly": True,
        },
        "alternatives": [
            {
                "profile": "throughput_first",
                "description": "Prefer samples/s and loss stability; accept lower GPU util when workload is tiny.",
            },
            {
                "profile": "saturation_probe",
                "description": "Test larger batch/token/latent only as an expert A/B candidate.",
            },
        ],
        "notes": [
            "P5 only materializes a workload-shaping advisor patch for the next request.",
            "GPU util may remain low for tiny workloads even when throughput is healthy.",
            "The current training run is not mutated.",
        ],
    }


def build_bubble_action_plan(
    snapshot: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    *,
    mode: Any = "report_only",
) -> dict[str, Any]:
    resolved_mode = _normalized_mode(mode)
    action = _mapping(diagnosis.get("recommended_action"))
    action_kind = str(action.get("kind") or "")
    if action_kind in _DATA_SUPPLY_ACTIONS:
        return _data_supply_plan(snapshot=snapshot, action=action, mode=resolved_mode)
    if action_kind in _TRANSFER_OFFLOAD_ACTIONS:
        return _transfer_offload_plan(snapshot=snapshot, action=action, mode=resolved_mode)
    if action_kind in _OPTIMIZER_ACTIONS:
        return _optimizer_plan(snapshot=snapshot, action=action, mode=resolved_mode)
    if action_kind in _WORKLOAD_ACTIONS:
        return _workload_plan(snapshot=snapshot, action=action, mode=resolved_mode)
    if action_kind in _HOST_SCHEDULING_ACTIONS:
        return build_host_scheduling_plan(snapshot=snapshot, action=action, mode=resolved_mode)
    return _empty_plan(
        mode=resolved_mode,
        status="unsupported_phase",
        action_kind=action_kind,
        reason="this action is not handled by the implemented advisor patch phases",
        domain="unsupported",
    )


__all__ = ["build_bubble_action_plan"]
