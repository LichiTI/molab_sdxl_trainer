"""Runtime feature snapshots shared by entry_train and route trainers."""

from __future__ import annotations

from typing import Any, Mapping

from .memory_runtime_profiles import build_memory_runtime_profiles
from .model_acceleration_snapshot import build_model_acceleration_runtime_snapshot
from .bubble_runtime_controller import build_bubble_controller_report
from .multi_batch_contract import dataloader_attached_batching_contract
from .multi_batch_promotion_gate import build_lulynx_multi_batch_promotion_gate
from .training_data_pipeline_stage import (
    dataloader_attached_data_pipeline_report,
    merge_lulynx_data_pipeline_reports,
)
from .training_pipeline_execution_readiness import build_lulynx_training_pipeline_execution_readiness
from .training_step_orchestrator import build_lulynx_training_step_orchestrator_slice


_TRAINER_PROFILE_ATTRS: tuple[tuple[str, str], ...] = (
    ("compile_runtime", "_compile_runtime_profile"),
    ("compile_cache", "_compile_cache_profile"),
    ("optimizer_backend", "_optimizer_backend_profile"),
    ("advanced_optimizer_strategy", "_advanced_optimizer_strategy_profile"),
    ("data_backend", "_data_backend_profile"),
    ("fused_projection", "_fused_projection_profile"),
    ("weight_compression", "_weight_compression_profile"),
    ("lora_activation_recompute", "_lora_activation_recompute_profile"),
    ("adapter_runtime", "_adapter_runtime_profile"),
    ("attention_runtime", "_attention_runtime_profile"),
    ("checkpoint_policy", "_checkpoint_policy_profile"),
    ("newbie_cache_first_profile", "_newbie_cache_first_profile"),
    ("dataloader_rebuild_readiness", "_dataloader_rebuild_readiness_profile"),
    ("diffusers_cache_runtime", "_diffusers_cache_runtime_profile"),
    ("anima_full_finetune_experiments", "_anima_full_finetune_experiments_profile"),
)


def _dict_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value:
        return dict(value)
    return {}


def _merge_runtime_feature_overlay(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if not key:
            continue
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = dict(base.get(key) or {})
            _merge_runtime_feature_overlay(nested, value)
            base[str(key)] = nested
        elif isinstance(value, Mapping):
            base[str(key)] = dict(value)
        else:
            base[str(key)] = value


def _call_profile(source: Any, name: str) -> dict[str, Any]:
    fn = getattr(source, name, None)
    if not callable(fn):
        return {}
    try:
        profile = fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _dict_profile(profile)


def _config_bool(config: Any, name: str, default: bool = False) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_lulynx_trainer_runtime_features(trainer: Any) -> dict[str, Any]:
    """Collect already-realized LulynxTrainer runtime evidence.

    The snapshot intentionally avoids dataset scans and manifest fingerprinting;
    it is cheap enough for progress/state updates.
    """

    features: dict[str, Any] = {}
    loop = getattr(trainer, "training_loop", None)
    memory_state = _dict_profile(getattr(loop, "memory_optimization_state", None))
    if memory_state:
        features["memory_optimization"] = memory_state

    b_tier = _dict_profile(getattr(loop, "_b_tier_last_state", None))
    if b_tier:
        features["b_tier"] = b_tier

    for key, attr_name in _TRAINER_PROFILE_ATTRS:
        profile = _dict_profile(getattr(trainer, attr_name, None))
        if profile:
            features[key] = profile

    dataloader_contract = dataloader_attached_batching_contract(getattr(trainer, "_dataloader", None))
    if dataloader_contract:
        features["multi_batch_dataloader"] = dataloader_contract
    data_pipeline = dataloader_attached_data_pipeline_report(getattr(trainer, "_dataloader", None))
    if data_pipeline:
        features["training_data_pipeline"] = data_pipeline

    for key, profile in build_memory_runtime_profiles(trainer).items():
        features.setdefault(key, dict(profile))

    bubble_window = _call_profile(trainer, "_bubble_closed_loop_window_profile")
    if bubble_window:
        features["bubble_closed_loop_window"] = bubble_window

    if loop is not None:
        loop_runtime = _call_profile(loop, "get_memory_experiment_profile")
        if loop_runtime:
            features["training_loop_runtime"] = loop_runtime
            loop_data_pipeline = loop_runtime.get("training_data_pipeline") if isinstance(loop_runtime, Mapping) else None
            if isinstance(loop_data_pipeline, Mapping) and loop_data_pipeline:
                features["training_data_pipeline"] = merge_lulynx_data_pipeline_reports(
                    data_pipeline,
                    loop_data_pipeline,
                )
            loop_orchestrator = (
                loop_runtime.get("training_step_orchestrator_runtime")
                if isinstance(loop_runtime, Mapping)
                else None
            )
            if isinstance(loop_orchestrator, Mapping) and loop_orchestrator:
                features["training_step_orchestrator_runtime"] = dict(loop_orchestrator)
            trace = loop_runtime.get("training_pipeline_trace") if isinstance(loop_runtime, Mapping) else None
            if isinstance(trace, Mapping):
                features["multi_batch_promotion_gate"] = build_lulynx_multi_batch_promotion_gate(
                    training_pipeline_trace=trace,
                    multi_batch_dataloader=dataloader_contract,
                )
        native_update = _call_profile(loop, "get_turbocore_native_update_runtime_profile")
        if native_update:
            features["turbocore_native_update"] = native_update

    external_features = _dict_profile(getattr(trainer, "_bubble_controller_external_runtime_features", None))
    if external_features:
        _merge_runtime_feature_overlay(features, external_features)

    config = getattr(trainer, "config", None)
    internal_gate_requested = _config_bool(
        config,
        "training_step_orchestrator_internal_gate_enabled",
        False,
    ) if config is not None else False
    readiness = build_lulynx_training_pipeline_execution_readiness(runtime_features=features)
    if readiness:
        features["training_pipeline_execution_readiness"] = readiness
        features["training_step_orchestrator_slice"] = build_lulynx_training_step_orchestrator_slice(
            runtime_features=features,
            execution_readiness=readiness,
            internal_gate_enabled=False,
            internal_gate_requested=internal_gate_requested,
        )

    if config is not None:
        features["model_acceleration"] = build_model_acceleration_runtime_snapshot(
            config,
            runtime_features=features,
        )
        try:
            current_step = int(getattr(loop, "global_step", 0) or 0) if loop is not None else 0
            features["bubble_controller"] = build_bubble_controller_report(
                config,
                runtime_features=features,
                closed_loop_state=getattr(trainer, "_bubble_closed_loop_state", None),
                current_step=current_step,
            )
        except Exception as exc:
            features["bubble_controller_error"] = f"{type(exc).__name__}: {exc}"

    return features


__all__ = ["build_lulynx_trainer_runtime_features"]
