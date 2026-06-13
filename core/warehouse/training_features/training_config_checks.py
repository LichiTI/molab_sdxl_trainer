"""Training Configuration Checks - validates training configs before launch.

Provides composable check functions that analyze a training configuration
mapping and produce structured errors, warnings, and notes.  Each check
is independent and can be called in isolation or orchestrated together.

This module is a Warehouse reimplementation of the behavioral surface
of training-specific preflight validation.  No original code bodies were
copied.  All naming uses LULYNX brand prefix only.  Pure-stdlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .anima_full_finetune_preflight import build_anima_full_finetune_preflight_profile
from .flux_preflight import build_flux_preflight_messages
from ...lulynx_trainer.model_acceleration_policy import (
    PROFILE_OFF,
    normalize_acceleration_profile,
    resolve_model_acceleration_policy,
)
from ...lulynx_trainer.module_offload_contract import (
    get_module_offload_conflict,
    is_swap_requested,
    resolve_module_offload_config,
)


def _flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _str(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _str(value).lower()


def _norm(value: Any) -> str:
    return _lower(value).replace("-", "_")


@dataclass
class ConfigCheckReport:
    """Aggregated result of all config checks."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    recommended_config_patch: dict[str, Any] = field(default_factory=dict)

    @property
    def can_start(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ConfigCheckReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.notes.extend(other.notes)
        self.recommended_config_patch.update(other.recommended_config_patch)


def check_sage_attention(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Emit warnings when SageAttention is enabled for experimental routes."""
    report = ConfigCheckReport()
    attn_mode = _lower(config.get("attn_mode"))
    uses_sage = (
        attn_mode == "sageattn"
        or _flag(config.get("sageattn"))
        or _flag(config.get("use_sage_attn"))
    )
    if not uses_sage:
        return report
    sdxl_types = {"sdxl-lora", "sdxl-finetune", "sdxl-controlnet", "sdxl-controlnet-lllite", "sdxl-textual-inversion"}
    if training_type in sdxl_types:
        report.warnings.append("SDXL SageAttention is experimental. Requires the SageAttention runtime.")
    elif training_type.startswith("anima"):
        report.warnings.append("Anima SageAttention is experimental. Drift self-check runs at startup.")
    else:
        report.warnings.append("No stable SageAttention path. Will auto-fallback to SDPA/torch.")
    return report


def check_network_targets(config: Mapping[str, Any]) -> ConfigCheckReport:
    """Detect conflicting network training target selections."""
    report = ConfigCheckReport()
    cache_te_outputs = _flag(config.get("cache_text_encoder_outputs"))
    if (
        "network_train_unet_only" not in config
        and "network_train_text_encoder_only" not in config
        and not cache_te_outputs
    ):
        return report
    if _flag(config.get("network_train_unet_only")) and _flag(config.get("network_train_text_encoder_only")):
        report.warnings.append("Both train DiT/U-Net only and train text encoder only; treating as both.")
    train_unet = not _flag(config.get("network_train_text_encoder_only"))
    train_text = not _flag(config.get("network_train_unet_only"))
    if cache_te_outputs and train_text:
        if train_unet:
            report.warnings.append(
                "cache_text_encoder_outputs is enabled; native SDXL alignment will switch text handling to UNet-only."
            )
        else:
            report.errors.append("Text encoder-only training cannot combine with cache_text_encoder_outputs.")
    return report


def check_learning_rates(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Validate that at least one active learning rate is non-zero."""
    report = ConfigCheckReport()
    base_lr = _float_or_none(config.get("learning_rate"))
    if training_type == "anima-finetune":
        group_keys = [
            ("learning_rate", "learning_rate"),
            ("self_attn_lr", "anima_self_attn_lr"),
            ("cross_attn_lr", "anima_cross_attn_lr"),
            ("mlp_lr", "anima_mlp_lr"),
            ("mod_lr", "anima_mod_lr"),
            ("llm_adapter_lr", "anima_llm_adapter_lr"),
        ]
        active = []
        for public_key, normalized_key in group_keys:
            raw = _float_or_none(config.get(public_key))
            if raw is None and normalized_key != public_key:
                raw = _float_or_none(config.get(normalized_key))
            effective = raw if raw is not None else base_lr
            if public_key != "learning_rate" and effective is None:
                effective = base_lr
            if effective is not None and effective != 0:
                active.append(public_key.replace("_lr", "").replace("learning_rate", "base"))
        if not active:
            report.errors.append("All Anima component learning rates resolve to 0.")
        else:
            report.notes.append("Anima finetune active LR groups: " + ", ".join(active) + ".")
        return report
    unet_lr = _float_or_none(config.get("unet_lr"))
    te_lr = _float_or_none(config.get("text_encoder_lr"))
    train_unet = not _flag(config.get("network_train_text_encoder_only"))
    train_text = not _flag(config.get("network_train_unet_only"))
    eu = unet_lr if unet_lr is not None else base_lr
    et = te_lr if te_lr is not None else base_lr
    if train_unet and not train_text and eu == 0:
        report.errors.append("Active DiT/U-Net learning rate resolves to 0.")
    elif train_text and not train_unet and et == 0:
        report.errors.append("Active text encoder learning rate resolves to 0.")
    elif train_unet and train_text and eu == 0 and et == 0:
        report.errors.append("Both active learning rates resolve to 0.")
    return report


def check_torch_compile(config: Mapping[str, Any]) -> ConfigCheckReport:
    """Emit notes/warnings when torch.compile is enabled."""
    report = ConfigCheckReport()
    if not _flag(config.get("torch_compile")):
        return report
    backend = _str(config.get("dynamo_backend", "inductor")) or "inductor"
    report.notes.append("torch.compile enabled with backend " + backend + ".")
    reasons = []
    if _flag(config.get("deepspeed")): reasons.append("deepspeed")
    if _flag(config.get("sdxl_fixed_block_swap")): reasons.append("sdxl_fixed_block_swap")
    if reasons:
        report.warnings.append("torch.compile alongside " + ", ".join(reasons) + ". May reduce coverage.")
    if os.name == "nt" and backend.lower() in {"inductor", "cudagraphs"}:
        report.warnings.append("Windows+CUDA: torch.compile has high startup OOM risk.")
    return report


def check_channels_last(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Note channels_last for non-conv-heavy routes."""
    report = ConfigCheckReport()
    if not _flag(config.get("opt_channels_last")):
        return report
    report.notes.append("channels_last optimization is enabled.")
    if training_type.startswith(("flux", "sd3", "anima", "lumina", "hunyuan")):
        report.warnings.append("channels_last mainly benefits conv-heavy U-Net routes. Limited gain for transformers.")
    return report


def check_masked_loss(config: Mapping[str, Any], alpha_count: int = 0) -> ConfigCheckReport:
    """Warn when masked_loss is enabled but no alpha images found."""
    report = ConfigCheckReport()
    if not _flag(config.get("masked_loss")):
        return report
    report.notes.append("Masked loss enabled. Alpha candidates: " + str(alpha_count) + ".")
    if not _flag(config.get("alpha_mask")):
        report.warnings.append(
            "masked_loss on but alpha_mask is disabled; masked loss may be a no-op unless alpha-capable images are present."
        )
    if alpha_count == 0:
        report.warnings.append("masked_loss on but no alpha-capable images found.")
    return report


def check_validation_split(config: Mapping[str, Any]) -> ConfigCheckReport:
    """Validate the validation_split range."""
    report = ConfigCheckReport()
    try:
        split = float(config.get("validation_split", 0) or 0)
    except (TypeError, ValueError):
        report.errors.append("validation_split must be a float between 0 and 1.")
        return report
    if split < 0 or split > 1:
        report.errors.append("validation_split must be between 0 and 1.")
    elif split > 0:
        report.notes.append("Validation split: {:.2%}.".format(split))
        if split < 0.05:
            report.warnings.append("Validation split very small; noisy feedback likely.")
        if split > 0.4:
            report.warnings.append("Validation split large; may reduce training data too much.")
    return report


def check_eval_dataset(config: Mapping[str, Any]) -> ConfigCheckReport:
    """Validate independent eval dataset settings."""
    report = ConfigCheckReport()
    eval_dir = str(config.get("eval_data_dir", "") or "").strip()
    if not eval_dir:
        return report

    eval_path = Path(eval_dir)
    if not eval_path.exists():
        report.errors.append("eval_data_dir does not exist: " + eval_dir)
    elif not eval_path.is_dir():
        report.errors.append("eval_data_dir must be a directory: " + eval_dir)
    else:
        report.notes.append("Independent eval dataset enabled: " + eval_dir)

    try:
        eval_every = int(config.get("eval_every_n_epochs", 0) or 0)
    except (TypeError, ValueError):
        report.errors.append("eval_every_n_epochs must be a non-negative integer.")
        eval_every = 0
    if eval_every < 0:
        report.errors.append("eval_every_n_epochs must be >= 0.")

    try:
        eval_batch = int(config.get("eval_batch_size", 0) or 0)
    except (TypeError, ValueError):
        report.errors.append("eval_batch_size must be a non-negative integer.")
        eval_batch = 0
    if eval_batch < 0:
        report.errors.append("eval_batch_size must be >= 0.")

    try:
        split = float(config.get("validation_split", 0) or 0)
    except (TypeError, ValueError):
        split = 0.0
    if split > 0:
        report.warnings.append(
            "eval_data_dir is set; validation_split will not split train_data_dir in the native trainer."
        )
    return report


def check_sdxl_clip_skip(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Warn when SDXL clip_skip > 1."""
    report = ConfigCheckReport()
    if not training_type.startswith("sdxl"):
        return report
    try:
        clip_skip = int(config.get("clip_skip", 0))
    except (TypeError, ValueError):
        return report
    if clip_skip > 1:
        report.warnings.append("SDXL clip_skip=" + str(clip_skip) + " is experimental.")
    return report


def check_anima_requirements(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Validate Anima-specific required paths and adapter mode."""
    report = ConfigCheckReport()
    if not training_type.startswith("anima"):
        return report

    if training_type == "anima-finetune":
        report.notes.append("Anima full finetune trains DiT parameters; text encoders are frozen conditioning providers.")
        if _flag(config.get("anima_full_finetune_train_text_encoder_requested")) or _flag(config.get("train_text_encoder")):
            report.warnings.append(
                "Anima text-encoder parameter training was requested, but full finetune keeps TE frozen and trains DiT-only. "
                "Use online_cache when you want frozen TE to participate in conditioning/cache generation."
            )
        cache_mode = _norm(config.get("native_cache_mode")) or _norm(config.get("anima_cache_mode")) or "cache_first"
        if cache_mode not in {"cache_first", "force_cache_only", "online_cache"}:
            report.errors.append("Anima full finetune requires cache_first or online_cache; raw online training is not validated.")
        elif cache_mode in {"cache_first", "force_cache_only"} and not _flag(config.get("anima_cached_training"), default=True):
            report.errors.append("Anima full finetune cache_first mode requires cached latent/text training to be enabled.")

    cache_mode = _norm(config.get("native_cache_mode")) or _norm(config.get("anima_cache_mode")) or "cache_first"
    cache_first_full_finetune = training_type == "anima-finetune" and cache_mode in {"cache_first", "force_cache_only"}
    online_cache_full_finetune = training_type == "anima-finetune" and cache_mode == "online_cache"
    needs_online_components = not cache_first_full_finetune

    q3 = _str(config.get("anima_qwen3_path")) or _str(config.get("qwen3"))
    if needs_online_components and not q3:
        report.errors.append("qwen3 is required for Anima training.")
    elif q3 and not os.path.exists(q3):
        report.errors.append("Qwen3 path does not exist: " + q3)

    vae = _str(config.get("vae_path")) or _str(config.get("vae"))
    if needs_online_components and not vae:
        report.errors.append("vae is required for " + training_type + ".")
    elif vae and not os.path.exists(vae):
        report.errors.append("VAE path does not exist: " + vae)
    elif vae and not (os.path.isfile(vae) or os.path.isdir(vae)):
        report.errors.append("VAE path must be a file or directory: " + vae)

    if online_cache_full_finetune:
        report.notes.append("Anima online_cache will use frozen VAE/Qwen3 to generate missing cache; DiT remains the only trainable component.")

    llm = _str(config.get("anima_llm_adapter_path")) or _str(config.get("llm_adapter_path"))
    if llm and not os.path.exists(llm):
        report.errors.append("LLM Adapter does not exist: " + llm)

    t5 = _str(config.get("anima_t5_tokenizer_path")) or _str(config.get("t5_tokenizer_path"))
    if t5 and not os.path.exists(t5):
        report.errors.append("T5 tokenizer does not exist: " + t5)
    if training_type != "anima-finetune":
        at = _str(config.get("anima_adapter_type")).lower() or "lora"
        if at == "lokr":
            report.notes.append("Anima adapter: LoKr.")
            report.warnings.append("LoKr uses built-in injection, not external LyCORIS.")
        elif at == "vera":
            report.notes.append("Anima adapter: VeRA.")
            report.warnings.append("VeRA exports as LoRA-compatible. Use save_state for exact resume.")
        elif at == "lora_fa":
            report.notes.append("Anima adapter: LoRA-FA.")
        elif _flag(config.get("dora_wd")):
            report.notes.append("Anima adapter: LoRA + DoRA.")
            report.warnings.append("Anima DoRA forces bypass_mode off.")
        else:
            report.notes.append("Anima adapter: LoRA.")
    return report


def check_anima_full_finetune_preflight(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Expose Anima full-finetune 16GB guidance through the generic report."""
    report = ConfigCheckReport()
    profile = build_anima_full_finetune_preflight_profile(config, training_type)
    if not profile.applicable:
        return report
    report.warnings.extend(profile.warnings)
    report.notes.extend(profile.notes)
    report.recommended_config_patch.update(profile.recommended_config_patch)
    return report


def check_sampler_compatibility(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Warn about sampler/scheduler compatibility for Anima preview."""
    report = ConfigCheckReport()
    if not training_type.startswith("anima"):
        return report
    ss = _lower(config.get("sample_scheduler"))
    if ss and ss != "simple":
        report.warnings.append("Anima preview scheduler falls back to simple.")
    sm = _lower(config.get("sample_sampler"))
    aliases = {"euler_a": "euler", "k_euler_a": "k_euler"}
    norm = aliases.get(sm, sm)
    if sm and norm != sm:
        report.warnings.append("Anima sampler " + sm + " maps to " + norm + ".")
    elif sm and norm not in {"euler", "k_euler"}:
        report.warnings.append("Anima sampler only supports euler/k_euler.")
    return report


def check_newbie_requirements(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Validate Newbie-specific required paths and settings."""
    report = ConfigCheckReport()
    if not training_type.startswith("newbie"):
        return report

    diffusers_path = _str(config.get("newbie_diffusers_path"))
    transformer_path = _str(config.get("newbie_transformer_path")) or _str(config.get("transformer_path"))
    base_model = _str(config.get("pretrained_model_name_or_path")) or _str(config.get("base_model_path"))

    if not diffusers_path and not transformer_path and not base_model:
        report.errors.append("Newbie requires a model path: newbie_diffusers_path, transformer_path, or base_model_path.")

    if diffusers_path and not os.path.exists(diffusers_path):
        report.errors.append("Newbie diffusers path does not exist: " + diffusers_path)

    if transformer_path and not os.path.exists(transformer_path):
        report.errors.append("Newbie transformer path does not exist: " + transformer_path)

    gemma = _str(config.get("newbie_gemma_model_path")) or _str(config.get("gemma_model_path"))
    if gemma and not os.path.exists(gemma):
        report.errors.append("Newbie Gemma model path does not exist: " + gemma)

    clip = _str(config.get("newbie_clip_model_path")) or _str(config.get("clip_model_path"))
    if clip and not os.path.exists(clip):
        report.errors.append("Newbie CLIP model path does not exist: " + clip)

    if _flag(config.get("newbie_force_cache_only")) and _flag(config.get("newbie_rebuild_cache")):
        report.warnings.append("Force cache only + rebuild cache: cache will be rebuilt then training skipped.")

    return report


def check_flux_requirements(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Validate FLUX route readiness and surface DiT optimization guidance."""
    report = ConfigCheckReport()
    messages = build_flux_preflight_messages(config, training_type)
    report.errors.extend(messages.errors)
    report.warnings.extend(messages.warnings)
    report.notes.extend(messages.notes)
    return report


def check_dataset_config_reference(config: Mapping[str, Any], *, root_dir: Path | None = None) -> ConfigCheckReport:
    """Validate dataset_config path exists."""
    report = ConfigCheckReport()
    raw = _str(config.get("dataset_config"))
    if not raw:
        return report
    p = Path(raw).expanduser()
    if not p.is_absolute() and root_dir is not None:
        p = root_dir / p
    if not p.exists():
        report.errors.append("dataset_config does not exist: " + str(p))
    elif not p.is_file():
        report.errors.append("dataset_config must be a file: " + str(p))
    return report


def check_save_state(config: Mapping[str, Any]) -> ConfigCheckReport:
    report = ConfigCheckReport()
    if _flag(config.get("save_state")):
        report.notes.append("save_state enabled; resume points will be produced.")
    elif _str(config.get("resume")) or _str(config.get("resume_path")):
        report.notes.append("Resume configured but save_state off; no new snapshots.")
    return report


def check_run_manifest_resume(config: Mapping[str, Any]) -> ConfigCheckReport:
    """Warn when a resume target has no sibling run_manifest.json."""
    report = ConfigCheckReport()
    raw_resume = _str(config.get("resume_path")) or _str(config.get("resume"))
    if not raw_resume:
        return report
    resume_path = Path(raw_resume)
    if resume_path.is_dir():
        manifest = resume_path / "run_manifest.json"
        if not manifest.is_file():
            manifest = resume_path.parent / "run_manifest.json"
    else:
        manifest = resume_path.parent / "run_manifest.json"
    if manifest.is_file():
        report.notes.append("run_manifest.json found for resume: " + str(manifest))
    else:
        report.warnings.append(
            "run_manifest.json not found for resume target; legacy resume compatibility mode will be used."
        )
    return report


def check_clear_cache(config: Mapping[str, Any]) -> ConfigCheckReport:
    report = ConfigCheckReport()
    if _flag(config.get("clear_dataset_npz_before_train")):
        report.notes.append("clear_dataset_npz_before_train enabled; caches cleared before launch.")
    return report



def check_weight_compression(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    """Validate the Warehouse frozen weight compression contract."""
    report = ConfigCheckReport()
    legacy_fp8 = _flag(config.get("fp8_base"))
    preset = _lower(config.get("weight_compression_preset")).replace("-", "_")
    preset_aliases = {
        "safe": "stable_backbone_int8",
        "stable": "stable_backbone_int8",
        "backbone_int8": "stable_backbone_int8",
        "int8": "stable_backbone_int8",
        "aggressive": "aggressive_backbone_uint4",
        "backbone_uint4": "aggressive_backbone_uint4",
        "uint4": "aggressive_backbone_uint4",
        "te_int8": "text_encoder_int8",
        "text_int8": "text_encoder_int8",
        "text_encoder": "text_encoder_int8",
        "all_int8": "both_int8",
        "both": "both_int8",
        "float8": "experimental_float8",
        "fp8_experimental": "experimental_float8",
    }
    preset = preset_aliases.get(preset, preset)
    preset_map = {
        "stable_backbone_int8": ("backbone", "torchao_int8"),
        "aggressive_backbone_uint4": ("backbone", "torchao_uint4"),
        "text_encoder_int8": ("text_encoder", "torchao_int8"),
        "both_int8": ("both", "torchao_int8"),
        "experimental_float8": ("backbone", "torchao_float8"),
    }
    preset_target, preset_format = preset_map.get(preset, ("none", ""))
    requested = _flag(config.get("weight_compression_enabled")) or legacy_fp8 or preset in preset_map
    if not requested:
        return report

    target = _lower(config.get("weight_compression_target")) or "none"
    if target in {"", "off", "disabled", "none"} and preset_target != "none":
        target = preset_target
    if target in {"", "off", "disabled", "none"} and legacy_fp8:
        target = "backbone"
    aliases = {
        "unet": "backbone",
        "dit": "backbone",
        "transformer": "backbone",
        "text": "text_encoder",
        "text_encoders": "text_encoder",
        "te": "text_encoder",
        "all": "both",
    }
    target = aliases.get(target, target)
    fmt = (_lower(config.get("weight_compression_format")) or preset_format or "fp8_e4m3").replace("-", "_")
    format_aliases = {
        "fp8": "fp8_e4m3",
        "native_fp8": "fp8_e4m3",
        "float8": "fp8_e4m3",
        "float8_e4m3": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "int8": "torchao_int8",
        "uint4": "torchao_uint4",
        "int4": "torchao_uint4",
        "torchao_int4": "torchao_uint4",
        "torchao_fp8": "torchao_float8",
        "quanto_qint8": "quanto_int8",
        "quanto_qfloat8": "quanto_float8",
    }
    fmt = format_aliases.get(fmt, fmt)
    supported_formats = {
        "fp8_e4m3",
        "torchao_int8",
        "torchao_uint4",
        "torchao_float8",
        "quanto_int8",
        "quanto_float8",
    }

    if target not in {"backbone", "text_encoder", "both"}:
        report.errors.append("weight_compression_target must be one of none, backbone, text_encoder, both.")
    if fmt not in supported_formats:
        report.errors.append("weight_compression_format must be one of fp8_e4m3, torchao_int8, torchao_uint4, torchao_float8, quanto_int8, quanto_float8.")
    elif fmt.startswith("torchao_"):
        try:
            import importlib.util
            torchao_available = importlib.util.find_spec("torchao") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            torchao_available = False
        if not torchao_available:
            report.errors.append("weight_compression_format requires torchao, but torchao is not installed.")
        elif _flag(config.get("weight_compression_verify"), default=True):
            try:
                from ...lulynx_trainer.weight_compression import probe_weight_compression_format
                ok, reason = probe_weight_compression_format(fmt)
            except Exception as exc:
                ok, reason = False, f"{type(exc).__name__}: {exc}"
            if not ok:
                report.errors.append("weight_compression_format probe failed for " + fmt + ": " + reason)
    elif fmt.startswith("quanto_"):
        try:
            import importlib.util
            quanto_available = importlib.util.find_spec("optimum.quanto") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            quanto_available = False
        if not quanto_available:
            report.errors.append("weight_compression_format requires optimum.quanto, but optimum.quanto is not installed.")
        elif _flag(config.get("weight_compression_verify"), default=True):
            try:
                from ...lulynx_trainer.weight_compression import probe_weight_compression_format
                ok, reason = probe_weight_compression_format(fmt)
            except Exception as exc:
                ok, reason = False, f"{type(exc).__name__}: {exc}"
            if not ok:
                report.errors.append("weight_compression_format probe failed for " + fmt + ": " + reason)
    if target in {"text_encoder", "both"} and _flag(config.get("train_text_encoder")):
        report.errors.append("Text encoder weight compression cannot be used while train_text_encoder is enabled.")
    if _flag(config.get("torch_compile")):
        report.errors.append("Weight compression and torch.compile are not supported together yet.")
    if _flag(config.get("module_offload_enabled")) and not _flag(config.get("weight_compression_allow_offload_combo")):
        report.errors.append("Weight compression and module_offload are experimental together; enable weight_compression_allow_offload_combo to opt in.")
    if _flag(config.get("compression_companion_enabled")):
        companion_path = _str(config.get("compression_companion_path"))
        companion_type = _lower(config.get("compression_companion_type")) or "lora"
        companion_mode = (_lower(config.get("compression_companion_mode")) or "merge_into_base").replace("-", "_")
        if not companion_path:
            report.errors.append("compression_companion_path is required when compression_companion_enabled is true.")
        elif not Path(companion_path).is_file():
            report.errors.append("compression_companion_path does not exist: " + companion_path)
        if companion_type not in {"lora"}:
            report.errors.append("compression_companion_type currently supports lora only.")
        if companion_mode in {"merge", "bake", "bake_into_base"}:
            companion_mode = "merge_into_base"
        if companion_mode != "merge_into_base":
            report.errors.append("compression_companion_mode currently supports merge_into_base only.")
        if target in {"text_encoder", "both"}:
            report.warnings.append("compression companion merge is applied through currently injected adapter layers; text-encoder compensation requires text encoder adapter injection to be active.")
    if legacy_fp8 and not _flag(config.get("weight_compression_enabled")):
        report.notes.append("fp8_base maps to weight compression target=backbone for backward compatibility.")
    return report


def check_module_offload(config: Mapping[str, Any], training_type: str) -> ConfigCheckReport:
    report = ConfigCheckReport()
    view = resolve_module_offload_config(config)
    if not view.requested:
        return report

    route = _lower(training_type or config.get("schema_id") or config.get("training_type"))
    distributed_enabled = (
        _flag(config.get("multi_gpu"))
        or _flag(config.get("enable_distributed"))
        or _flag(config.get("enable_distributed_training"))
        or int(_float_or_none(config.get("num_processes")) or 1) > 1
        or int(_float_or_none(config.get("num_machines")) or 1) > 1
    )
    pipeline_enabled = (
        any(token in route for token in ("controlnet", "ip-adapter", "lllite"))
        or _flag(config.get("ip_adapter_enabled"))
        or bool(_str(config.get("controlnet_model")))
    )
    conflict_codes: list[str] = []
    if is_swap_requested(config):
        conflict_codes.append("swap")
    if _flag(config.get("vram_swap_to_ram")):
        conflict_codes.append("vram_swap_to_ram")
    if _flag(config.get("safe_fallback")) or _flag(config.get("newbie_safe_fallback")):
        conflict_codes.append("safe_fallback")
    if _flag(config.get("torch_compile")):
        conflict_codes.append("torch_compile")
    if distributed_enabled:
        conflict_codes.append("distributed")
    if _flag(config.get("deepspeed")):
        conflict_codes.append("deepspeed")
    if pipeline_enabled:
        conflict_codes.append("pipeline")
    if _flag(config.get("gradient_checkpointing")):
        conflict_codes.append("gradient_checkpointing")
    if _flag(config.get("cpu_offload_checkpointing")):
        conflict_codes.append("cpu_offload_checkpointing")

    for code in conflict_codes:
        _, message = get_module_offload_conflict(code)
        report.errors.append(message)
    return report


def check_model_acceleration_policy(
    config: Mapping[str, Any],
    training_type: str,
    *,
    schema_id: str = "",
) -> ConfigCheckReport:
    """Expose model-aware acceleration recommendations through preflight."""
    report = ConfigCheckReport()
    requested = normalize_acceleration_profile(
        config.get("acceleration_profile", config.get("speed_profile", PROFILE_OFF))
    )
    if requested == PROFILE_OFF:
        return report

    decision = resolve_model_acceleration_policy(
        config,
        schema_id=schema_id,
        training_type=training_type,
    )
    report.warnings.extend(decision.warnings)
    report.notes.extend(decision.notes)
    if decision.recommended_config_patch:
        report.recommended_config_patch.update(decision.recommended_config_patch)
        report.notes.append(
            "Acceleration policy recommended patch: "
            + ", ".join(sorted(decision.recommended_config_patch.keys()))
        )
    if decision.skipped:
        skipped_keys = sorted({str(item.get("key", "")) for item in decision.skipped if item.get("key")})
        if skipped_keys:
            report.notes.append("Acceleration policy preserved explicit fields: " + ", ".join(skipped_keys))
    return report


@dataclass
class TrainingPreflightConfig:
    """Declarative input for the training config preflight runner."""
    config: dict[str, Any]
    training_type: str = ""
    schema_id: str = ""  # schema identity (e.g. "sdxl-lora"), distinct from trainer dispatch value
    alpha_candidate_count: int = 0
    root_dir: Path | None = None
    allow_dataset_config_without_data_dir: bool = False
    skip_model_validation: bool = False


def run_training_config_checks(preflight_config: TrainingPreflightConfig) -> ConfigCheckReport:
    """Run all applicable training config checks and aggregate results."""
    cfg = preflight_config.config
    tt = preflight_config.training_type
    # schema_id may come from preflight_config or from the config dict itself
    sid = preflight_config.schema_id or str(cfg.get("schema_id", "") or "")
    report = ConfigCheckReport()
    report.merge(check_network_targets(cfg))
    report.merge(check_learning_rates(cfg, sid or tt))
    report.merge(check_sage_attention(cfg, sid or tt))
    report.merge(check_torch_compile(cfg))
    report.merge(check_weight_compression(cfg, sid or tt))
    report.merge(check_module_offload(cfg, sid or tt))
    report.merge(check_channels_last(cfg, sid or tt))
    report.merge(check_sdxl_clip_skip(cfg, sid or tt))
    report.merge(check_validation_split(cfg))
    report.merge(check_eval_dataset(cfg))
    report.merge(check_masked_loss(cfg, preflight_config.alpha_candidate_count))
    report.merge(check_anima_requirements(cfg, sid or tt))
    report.merge(check_anima_full_finetune_preflight(cfg, sid or tt))
    report.merge(check_newbie_requirements(cfg, sid or tt))
    report.merge(check_flux_requirements(cfg, sid or tt))
    report.merge(check_model_acceleration_policy(cfg, sid or tt, schema_id=sid))
    report.merge(check_sampler_compatibility(cfg, sid or tt))
    report.merge(check_save_state(cfg))
    report.merge(check_run_manifest_resume(cfg))
    report.merge(check_clear_cache(cfg))
    report.merge(check_dataset_config_reference(cfg, root_dir=preflight_config.root_dir))
    return report


__all__ = [
    "ConfigCheckReport",
    "TrainingPreflightConfig",
    "check_anima_requirements",
    "check_anima_full_finetune_preflight",
    "check_channels_last",
    "check_clear_cache",
    "check_dataset_config_reference",
    "check_eval_dataset",
    "check_flux_requirements",
    "check_learning_rates",
    "check_masked_loss",
    "check_module_offload",
    "check_model_acceleration_policy",
    "check_network_targets",
    "check_newbie_requirements",
    "check_sampler_compatibility",
    "check_save_state",
    "check_run_manifest_resume",
    "check_sage_attention",
    "check_sdxl_clip_skip",
    "check_torch_compile",
    "check_weight_compression",
    "check_validation_split",
    "run_training_config_checks",
]

