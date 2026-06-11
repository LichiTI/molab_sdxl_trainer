"""Benchmark-ready matrix for model-aware acceleration tracks.

The matrix is report-only: it does not start training or run probes.  It joins
policy recommendations, user/runtime requests, realized runtime evidence, and
future benchmark metric slots in one stable payload.
"""

from __future__ import annotations

from typing import Any, Mapping

from .model_acceleration_conflicts import active_compile_intent, weight_compression_requested
from .model_acceleration_matrix import (
    adapter_init_patch_for,
    advanced_optimizer_strategy_for,
    attention_backend_for,
    cache_patch_for,
    checkpoint_patch_for,
    compile_policy_for,
    lora_recompute_mode_for,
    low_bit_patch_for,
    optimizer_backend_for,
)
from .model_acceleration_policy import (
    PROFILE_OFF,
    normalize_acceleration_profile,
    resolve_model_acceleration_policy,
)


TRACKS: tuple[str, ...] = (
    "cache",
    "attention",
    "compile",
    "compile_cache",
    "optimizer",
    "advanced_optimizer",
    "adapter_init",
    "low_bit",
    "data_backend",
    "checkpoint",
    "lora_recompute",
)

TRACK_POLICY_NAMES: dict[str, set[str]] = {
    "cache": {"cache"},
    "attention": {"attention"},
    "compile": {"compile"},
    "compile_cache": {"compile"},
    "optimizer": {"optimizer"},
    "advanced_optimizer": {"advanced_optimizer"},
    "adapter_init": {"adapter_init"},
    "low_bit": {"weight_compression", "low_vram"},
    "data_backend": {"data", "data_backend"},
    "checkpoint": {"checkpoint"},
    "lora_recompute": {"lora_recompute"},
}

TRACK_EVIDENCE_KEYS: dict[str, tuple[str, ...]] = {
    "cache": ("diffusers_cache_runtime", "cache_runtime", "newbie_cache_first_profile", "data_backend"),
    "attention": ("attention_runtime",),
    "compile": ("compile_runtime",),
    "compile_cache": ("compile_cache", "compile_runtime"),
    "optimizer": ("optimizer_backend",),
    "advanced_optimizer": ("advanced_optimizer_strategy",),
    "adapter_init": ("adapter_runtime",),
    "low_bit": ("weight_compression",),
    "data_backend": ("data_backend",),
    "checkpoint": ("checkpoint_policy", "training_loop_runtime", "offloaded_checkpoint"),
    "lora_recompute": ("lora_activation_recompute", "flux_runtime", "adapter_runtime"),
}

TRACK_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "cache": (
        "cache_latents",
        "cache_latents_to_disk",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "native_cache_mode",
        "anima_cached_training",
        "use_cache",
    ),
    "attention": ("attention_backend",),
    "compile": ("compile_runtime", "torch_compile", "torch_compile_scope", "anima_compile_scope"),
    "compile_cache": ("compile_cache_enabled", "compile_cache_root", "compile_cache_reuse"),
    "optimizer": ("optimizer_backend",),
    "advanced_optimizer": ("advanced_optimizer_strategy", "lora_plus_enabled", "rs_lora_enabled", "svd_grad_proj_enabled"),
    "adapter_init": ("adapter_init_strategy", "pissa_enabled", "use_pissa"),
    "low_bit": ("weight_compression_enabled", "weight_compression_preset", "weight_compression_target", "fp8_base"),
    "data_backend": ("data_backend", "cached_collate_mode"),
    "checkpoint": ("checkpoint_policy", "gradient_checkpointing", "cpu_offload_checkpointing"),
    "lora_recompute": ("lora_activation_recompute_mode", "lora_activation_recompute"),
}

BENCHMARK_METRIC_KEYS: tuple[str, ...] = (
    "step_time_ms",
    "mean_step_time_ms",
    "steady_step_time_ms",
    "peak_vram_mb",
    "throughput_it_s",
    "loss_delta",
)


def _field(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:
            pass
    data = getattr(config, "__dict__", None)
    if isinstance(data, Mapping):
        return dict(data)
    keys = {key for values in TRACK_CONFIG_KEYS.values() for key in values}
    keys.update(("schema_id", "training_type", "model_type", "model_arch", "route", "acceleration_profile", "speed_profile"))
    return {key: _field(config, key) for key in keys}


def _compile_runtime_requests_cache(profile: Mapping[str, Any]) -> bool:
    values = (
        profile.get("requested_runtime"),
        profile.get("requested"),
        profile.get("compile_runtime"),
        profile.get("compile_kind"),
    )
    if any(_norm(value) in {"compile_cache", "cache"} for value in values):
        return True
    contract = profile.get("contract")
    if isinstance(contract, Mapping):
        return any(_norm(contract.get(key)) in {"compile_cache", "cache"} for key in ("requested", "resolved"))
    return False


def _nested_profile(track: str, key: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if track == "compile_cache" and key == "compile_runtime":
        if not _compile_runtime_requests_cache(value):
            return {}
        profile = dict(value)
        profile.setdefault("source", "compile_runtime")
        return profile
    if track == "checkpoint" and key == "training_loop_runtime":
        nested = value.get("offloaded_checkpointing")
        if isinstance(nested, Mapping) and nested:
            profile = dict(nested)
            profile.setdefault("source", "training_loop_runtime.offloaded_checkpointing")
            return profile
        return {}
    if track == "lora_recompute" and key == "flux_runtime":
        nested = value.get("lora_activation_recompute")
        if isinstance(nested, Mapping) and nested:
            profile = dict(nested)
            profile.setdefault("source", "flux_runtime.lora_activation_recompute")
            return profile
        return {}
    if track == "lora_recompute" and key == "adapter_runtime":
        if _boolish(value.get("activation_recompute_realized")):
            return {
                "enabled": True,
                "source": "adapter_runtime",
                "adapter_method": value.get("adapter_method"),
                "activation_recompute_realized": True,
            }
        return {}
    if track == "cache" and key == "data_backend":
        training = _norm(value.get("effective_training_backend"))
        native_reader = value.get("native_cache_reader")
        if training in {"anima_cached", "newbie_cached"} or isinstance(native_reader, Mapping):
            profile = dict(value)
            profile.setdefault("source", "data_backend_cache_route")
            return profile
        return {}
    return dict(value)


def _profile_from_features(features: Mapping[str, Any] | None, track: str) -> dict[str, Any]:
    if not isinstance(features, Mapping):
        return {}
    for key in TRACK_EVIDENCE_KEYS.get(track, ()):
        value = features.get(key)
        if isinstance(value, Mapping) and value:
            profile = _nested_profile(track, key, value)
            if profile:
                return profile
    memory = features.get("memory_optimization")
    if isinstance(memory, Mapping):
        for key in TRACK_EVIDENCE_KEYS.get(track, ()):
            value = memory.get(key)
            if isinstance(value, Mapping) and value:
                profile = _nested_profile(track, key, value)
                if profile:
                    return profile
    return {}


def _truthy_profile_value(profile: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_boolish(profile.get(key)) for key in keys)


def _runtime_applied(track: str, evidence: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not evidence:
        if track == "adapter_init":
            return _requested(track, config)
        return False
    if evidence.get("error"):
        return False
    if track == "cache":
        training = _norm(evidence.get("effective_training_backend"))
        return _truthy_profile_value(
            evidence,
            (
                "cache_first",
                "cache_latents",
                "text_cache_active",
                "enabled",
                "use_cache",
                "anima_cached_training",
                "latents_enabled",
                "text_enabled",
                "cache_present_before",
                "used_prebuilt_cache",
                "rebuilt_cache",
            ),
        ) or training in {"anima_cached", "newbie_cached"}
    if track == "attention":
        backend = _norm(evidence.get("resolved_backend") or evidence.get("backend") or evidence.get("resolved"))
        patched = _safe_int(evidence.get("patched_module_count") or evidence.get("patched_processors"), 0)
        if _boolish(evidence.get("profile_only")) and patched <= 0:
            return False
        return bool(evidence.get("applied")) or patched > 0 or backend not in {"", "auto", "default", "off", "none"}
    if track == "compile":
        contract = evidence.get("contract") if isinstance(evidence.get("contract"), Mapping) else {}
        resolved = _norm(evidence.get("resolved") or evidence.get("compile_kind") or contract.get("resolved"))
        return (
            bool(evidence.get("applied"))
            or _boolish(evidence.get("torch_compile"))
            or _safe_int(evidence.get("compiled_targets"), 0) > 0
            or resolved not in {"", "auto", "default", "off", "none"}
        )
    if track == "compile_cache":
        state = _norm(evidence.get("state"))
        if _compile_runtime_requests_cache(evidence):
            return (
                bool(evidence.get("applied"))
                or _boolish(evidence.get("torch_compile"))
                or _safe_int(evidence.get("compiled_targets"), 0) > 0
            )
        return bool(evidence.get("enabled")) and state not in {"disabled", "inactive", "blocked"}
    if track == "optimizer":
        resolved = _norm(evidence.get("resolved") or evidence.get("resolved_backend"))
        return resolved not in {"", "auto", "default", "off", "none"}
    if track == "advanced_optimizer":
        return bool(evidence.get("active")) or _norm(evidence.get("resolved")) in {"lora_plus", "rs_lora", "galore"}
    if track == "adapter_init":
        strategy = _norm(evidence.get("adapter_init_strategy") or config.get("adapter_init_strategy"))
        return strategy in {"pissa", "olora", "loftq"} or _boolish(config.get("pissa_enabled")) or _boolish(config.get("use_pissa"))
    if track == "low_bit":
        return bool(evidence.get("applied")) or (
            _boolish(evidence.get("enabled")) and _safe_int(evidence.get("compressed_count"), 0) > 0
        )
    if track == "data_backend":
        training = _norm(evidence.get("effective_training_backend"))
        resolved = _norm(evidence.get("resolved_backend"))
        profile_only = _boolish(evidence.get("profile_only"))
        return training not in {"", "caption", "raw"} or (resolved not in {"", "caption", "raw", "auto"} and not profile_only)
    if track == "checkpoint":
        policy = _norm(evidence.get("effective_policy") or evidence.get("mode"))
        return policy not in {"", "off", "none"} or _truthy_profile_value(
            evidence,
            ("gradient_checkpointing", "cpu_offload_checkpointing", "pinned_async_active", "enabled"),
        )
    if track == "lora_recompute":
        return _boolish(evidence.get("enabled"))
    return bool(evidence.get("applied") or evidence.get("active") or evidence.get("enabled"))


def _requested(track: str, config: Mapping[str, Any]) -> bool:
    if track == "cache":
        return any(_boolish(config.get(key)) for key in TRACK_CONFIG_KEYS[track] if key != "native_cache_mode") or _norm(
            config.get("native_cache_mode")
        ) in {"cache_first", "force_cache_only"}
    if track == "attention":
        return _norm(config.get("attention_backend")) not in {"", "auto", "default", "off", "none"}
    if track == "compile":
        return active_compile_intent(config)
    if track == "compile_cache":
        return _boolish(config.get("compile_cache_enabled")) and (
            _norm(config.get("compile_runtime")) in {"compile_cache", "cache"} or active_compile_intent(config)
        )
    if track == "optimizer":
        return _norm(config.get("optimizer_backend")) not in {"", "auto", "default", "off", "none"}
    if track == "advanced_optimizer":
        return _norm(config.get("advanced_optimizer_strategy")) not in {"", "auto", "default", "off", "none"} or any(
            _boolish(config.get(key)) for key in ("lora_plus_enabled", "rs_lora_enabled", "svd_grad_proj_enabled")
        )
    if track == "adapter_init":
        return _norm(config.get("adapter_init_strategy")) not in {"", "auto", "default", "off", "none", "standard"} or any(
            _boolish(config.get(key)) for key in ("pissa_enabled", "use_pissa")
        )
    if track == "low_bit":
        return weight_compression_requested(config)
    if track == "data_backend":
        return _norm(config.get("data_backend")) not in {"", "auto", "default", "caption", "raw"} or _norm(
            config.get("cached_collate_mode")
        ) not in {"", "auto", "default", "off", "none"}
    if track == "checkpoint":
        return _norm(config.get("checkpoint_policy")) not in {"", "auto", "default", "off", "none"} or _boolish(
            config.get("gradient_checkpointing")
        ) or _boolish(config.get("cpu_offload_checkpointing"))
    if track == "lora_recompute":
        return _norm(config.get("lora_activation_recompute_mode")) in {"on", "auto"} or _boolish(
            config.get("lora_activation_recompute")
        )
    return False


def _policy_track_patch(decision_payload: Mapping[str, Any], track: str) -> dict[str, Any]:
    names = TRACK_POLICY_NAMES.get(track, {track})
    patch: dict[str, Any] = {}
    for item in decision_payload.get("tracks", []) if isinstance(decision_payload.get("tracks"), list) else []:
        if not isinstance(item, Mapping) or str(item.get("name") or "") not in names:
            continue
        item_patch = item.get("patch")
        if isinstance(item_patch, Mapping):
            patch.update(item_patch)
    return patch


def _policy_recommended(decision_payload: Mapping[str, Any], track: str) -> bool:
    names = TRACK_POLICY_NAMES.get(track, {track})
    for item in decision_payload.get("tracks", []) if isinstance(decision_payload.get("tracks"), list) else []:
        if isinstance(item, Mapping) and str(item.get("name") or "") in names and item.get("status") == "recommended":
            return True
    return False


def _matrix_recommended(track: str, family: str, profile: str, config: Mapping[str, Any]) -> bool:
    if profile == PROFILE_OFF:
        return False
    if track == "cache":
        cache_spec = cache_patch_for(family, profile)
        return bool(cache_spec.patch)
    if track == "attention":
        return bool(attention_backend_for(family, profile))
    if track == "compile":
        return compile_policy_for(family, profile) is not None
    if track == "compile_cache":
        compile_spec = compile_policy_for(family, profile)
        return compile_spec is not None and compile_spec.runtime == "compile_cache"
    if track == "optimizer":
        return bool(optimizer_backend_for(family, profile))
    if track == "advanced_optimizer":
        allowed = any(_boolish(config.get(key)) for key in ("acceleration_allow_convergence", "acceleration_allow_lora_plus", "acceleration_allow_advanced_optimizer"))
        return allowed and bool(advanced_optimizer_strategy_for(family, profile))
    if track == "adapter_init":
        allowed = any(_boolish(config.get(key)) for key in ("acceleration_allow_convergence", "acceleration_allow_pissa", "acceleration_allow_adapter_init"))
        return allowed and bool(adapter_init_patch_for(family, profile).patch)
    if track == "low_bit":
        allowed = any(_boolish(config.get(key)) for key in ("acceleration_allow_low_bit", "acceleration_allow_weight_compression"))
        return allowed and bool(low_bit_patch_for(family, profile).patch)
    if track == "checkpoint":
        return bool(checkpoint_patch_for(family, profile).patch)
    if track == "lora_recompute":
        return bool(lora_recompute_mode_for(family, profile))
    return False


def _collect_blockers(track: str, evidence: Mapping[str, Any], cache_safety: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    cache_blockers = [str(item) for item in cache_safety.get("blockers", []) if item]
    if track == "cache":
        blockers.extend(cache_blockers)
    elif track in {"compile", "compile_cache"}:
        blockers.extend(
            item
            for item in cache_blockers
            if item in {"compile_requires_cache_first", "compile_weight_compression_conflict"}
        )
    elif track == "low_bit":
        blockers.extend(item for item in cache_blockers if item == "compile_weight_compression_conflict")
    for key in ("error", "skip_reason", "fallback_reason", "disabled_reason", "profile_only_reason", "blocker"):
        value = evidence.get(key)
        if value:
            blockers.append(str(value))
    for key in ("blockers", "warnings", "native_dispatch_blockers"):
        value = evidence.get(key)
        if isinstance(value, list):
            blockers.extend(str(item) for item in value if item)
    return sorted(dict.fromkeys(blockers))


def _metrics_for(track: str, benchmark_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    source: Mapping[str, Any] = {}
    if isinstance(benchmark_metrics, Mapping):
        scoped = benchmark_metrics.get(track)
        source = scoped if isinstance(scoped, Mapping) else benchmark_metrics
    return {key: source.get(key) if isinstance(source, Mapping) else None for key in BENCHMARK_METRIC_KEYS}


def _decision_payload(
    config: Mapping[str, Any],
    *,
    schema_id: str,
    training_type: str,
    decision: Any = None,
    resolve_policy: bool = True,
) -> dict[str, Any]:
    if decision is None:
        if not resolve_policy:
            return {}
        decision = resolve_model_acceleration_policy(config, schema_id=schema_id, training_type=training_type)
    as_dict = getattr(decision, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    return dict(decision) if isinstance(decision, Mapping) else {}


def build_model_acceleration_benchmark_matrix(
    config: Any,
    *,
    family: str = "",
    profile: str = "",
    runtime_features: Mapping[str, Any] | None = None,
    cache_safety: Mapping[str, Any] | None = None,
    decision: Any = None,
    benchmark_metrics: Mapping[str, Any] | None = None,
    resolve_policy: bool = True,
) -> dict[str, Any]:
    cfg = _config_dict(config)
    schema_id = str(cfg.get("schema_id") or "")
    training_type = str(cfg.get("training_type") or "")
    resolved_profile = normalize_acceleration_profile(profile or cfg.get("acceleration_profile") or cfg.get("speed_profile"))
    if not family:
        from .model_acceleration_snapshot import normalize_model_family_from_config

        family = normalize_model_family_from_config(cfg, schema_id=schema_id, training_type=training_type)
    family = _norm(family) or "unknown"
    decision_payload = _decision_payload(
        cfg,
        schema_id=schema_id,
        training_type=training_type,
        decision=decision,
        resolve_policy=resolve_policy,
    )
    if cache_safety is None:
        from .model_acceleration_snapshot import build_model_acceleration_cache_safety

        cache_safety = build_model_acceleration_cache_safety(cfg, family=family, profile=resolved_profile)
    cache_safety = dict(cache_safety or {})

    tracks = []
    for track in TRACKS:
        evidence = _profile_from_features(runtime_features, track)
        config_patch = _policy_track_patch(decision_payload, track)
        recommended = _policy_recommended(decision_payload, track) or _matrix_recommended(track, family, resolved_profile, cfg)
        requested = _requested(track, cfg)
        runtime_applied = _runtime_applied(track, evidence, cfg)
        tracks.append(
            {
                "name": track,
                "recommended": bool(recommended),
                "requested": bool(requested),
                "runtime_applied": bool(runtime_applied),
                "runtime_evidence_present": bool(evidence),
                "evidence": evidence,
                "blockers": _collect_blockers(track, evidence, cache_safety),
                "config_patch": config_patch,
                "benchmark_metrics": _metrics_for(track, benchmark_metrics),
            }
        )

    all_blockers = sorted({blocker for track in tracks for blocker in track["blockers"]})
    ready_tracks = [track["name"] for track in tracks if (track["recommended"] or track["requested"]) and not track["blockers"]]
    return {
        "matrix": "model_acceleration_benchmark_matrix_v0",
        "family": family,
        "profile": resolved_profile,
        "tracks": tracks,
        "summary": {
            "track_count": len(tracks),
            "recommended_count": sum(1 for track in tracks if track["recommended"]),
            "requested_count": sum(1 for track in tracks if track["requested"]),
            "runtime_applied_count": sum(1 for track in tracks if track["runtime_applied"]),
            "runtime_evidence_count": sum(1 for track in tracks if track["runtime_evidence_present"]),
            "blocked_count": sum(1 for track in tracks if track["blockers"]),
            "ready_tracks": ready_tracks,
            "missing_runtime_evidence": [
                track["name"]
                for track in tracks
                if (track["recommended"] or track["requested"]) and not track["runtime_evidence_present"]
            ],
            "blockers": all_blockers,
        },
    }


__all__ = [
    "BENCHMARK_METRIC_KEYS",
    "TRACKS",
    "build_model_acceleration_benchmark_matrix",
]
