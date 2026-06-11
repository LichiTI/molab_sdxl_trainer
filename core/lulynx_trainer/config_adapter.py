"""
Config adapter: frontend dict -> UnifiedTrainingConfig (LulynxConfig).

Handles camelCase-to-snake_case normalisation, per-field alias resolution,
and fallback logic so that the neutral native-plumbing fields
(native_secondary_model_path, native_tokenizer_path, …) are used when the
corresponding family-specific field is unset.
"""

from typing import Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

from core.services.runtime_optimization import resolve_runtime_optimization_payload
from core.warehouse.training_features.flux_preflight import (
    is_flux_network_module_supported,
    normalize_flux_network_module,
)

from .config import LulynxConfig
from .module_offload_contract import (
    clamp_module_offload_ratio,
    normalize_module_offload_patterns,
    normalize_module_offload_profile,
    parse_module_offload_float,
    parse_optional_module_offload_ratio,
)
from .model_acceleration_application import apply_model_acceleration_policy_to_config
from .model_acceleration_policy import normalize_acceleration_profile
from .sdxl_lora_low_vram_profile import normalize_low_vram_profile
from .turbocore_v3_exact_adamw_config_adapter import apply_v3_exact_adamw_canary_config_adapter
from .turbocore_v5_manual_wider_canary_config_adapter import (
    apply_v5_manual_wider_canary_config_adapter,
)


class ConfigAdapter:
    """配置适配器：Frontend -> Lulynx (UnifiedTrainingConfig)"""

    @classmethod
    def from_frontend_dict(cls, data: Dict[str, Any]) -> LulynxConfig:
        """从前端 training-store 格式转换，支持 camelCase 到 snake_case 的自动转换"""
        normalized_data = {}

        # 1. 通用键名转换 (camelCase -> snake_case)
        def to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        def boolish(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
                return True
            if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
                return False
            return default

        def normalize_anima_full_finetune_te_policy(value: Any, *, requested: bool = False) -> str:
            normalized = str(value or "").strip().lower().replace("-", "_")
            if requested or normalized == "blocked_phase1":
                return "blocked_phase1"
            return "dit_only"

        def stringify_arg_list(value: Any) -> str:
            """Normalize frontend arg-list values into a CLI-style string."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple, set, frozenset)):
                return " ".join(str(item).strip() for item in value if str(item).strip())
            return str(value).strip()

        def stringify_csv_list(value: Any) -> str:
            """Normalize textarea/list values into comma-separated trainer fields."""
            if value is None:
                return ""
            if isinstance(value, str):
                parts = [
                    part.strip()
                    for part in re.split(r"[\r\n,]+", value)
                    if part.strip()
                ]
                return ",".join(parts) if len(parts) > 1 else value.strip()
            if isinstance(value, (list, tuple, set, frozenset)):
                return ",".join(str(item).strip() for item in value if str(item).strip())
            return str(value).strip()

        def stringify_scalar(value: Any) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def has_optimizer_name_arg(value: Any) -> bool:
            text = stringify_arg_list(value)
            for part in re.split(r"[\r\n, ]+", text):
                if re.match(r"^\s*(name|optimizer_name|optimizer)\s*=", part or "", re.IGNORECASE):
                    return True
            return False

        for k, v in data.items():
            normalized_data[to_snake(k)] = v

        # 2. 核心字段兼容性修复
        if "optimizer_type" in normalized_data:
            optimizer_aliases = {
                "automagic": "Automagic++",
                "automagic++": "Automagic++",
                "autoprodigy": "AutoProdigy",
                "auto_prodigy": "AutoProdigy",
                "adafactor": "adafactor",
                "ada_factor": "adafactor",
                "prodigy": "prodigy",
                "prodigyplus.prodigyplusschedulefree": "prodigyplus.ProdigyPlusScheduleFree",
                "prodigyplusschedulefree": "prodigyplus.ProdigyPlusScheduleFree",
                "prodigyschedulefree": "prodigyplus.ProdigyPlusScheduleFree",
                "prodigy_schedule_free": "prodigyplus.ProdigyPlusScheduleFree",
                "dadaptadampreprint": "DAdaptAdamPreprint",
                "dadaptation": "DAdaptation",
                "pytorchoptimizer": "PytorchOptimizer",
                "pytorch_optimizer": "PytorchOptimizer",
                "genericoptimizer": "GenericOptimizer",
                "generic": "GenericOptimizer",
                "generic_optimizer": "GenericOptimizer",
            }
            optimizer_key = str(normalized_data.get("optimizer_type") or "").strip()
            plugin_optimizer_name = ""
            plugin_match = re.match(r"^pytorch_optimizer\.(.+)$", optimizer_key, flags=re.IGNORECASE)
            if not plugin_match:
                plugin_match = re.match(r"^pytorchoptimizer[:/](.+)$", optimizer_key, flags=re.IGNORECASE)
            if plugin_match:
                plugin_optimizer_name = plugin_match.group(1).strip()
                normalized_data["optimizer_type"] = "PytorchOptimizer"
                existing_name_sources = " ".join(
                    part
                    for part in (
                        stringify_arg_list(normalized_data.get("optimizer_args", "")),
                        stringify_arg_list(normalized_data.get("optimizer_args_custom", "")),
                    )
                    if part
                )
                if plugin_optimizer_name and not has_optimizer_name_arg(existing_name_sources):
                    existing_optimizer_args = " ".join(
                        part
                        for part in (
                            stringify_arg_list(normalized_data.get("optimizer_args", "")),
                            stringify_arg_list(normalized_data.get("optimizer_args_custom", "")),
                        )
                        if part
                    )
                    normalized_data["optimizer_args"] = " ".join(
                        part for part in (f"name={plugin_optimizer_name}", existing_optimizer_args) if part
                    )
            else:
                generic_match = re.match(r"^genericoptimizer[:/](.+)$", optimizer_key, flags=re.IGNORECASE)
                if not generic_match:
                    generic_match = re.match(r"^(bitsandbytes\.optim\..+)$", optimizer_key, flags=re.IGNORECASE)
                if generic_match:
                    generic_optimizer_name = generic_match.group(1).strip()
                    normalized_data["optimizer_type"] = "GenericOptimizer"
                    existing_name_sources = " ".join(
                        part
                        for part in (
                            stringify_arg_list(normalized_data.get("optimizer_args", "")),
                            stringify_arg_list(normalized_data.get("optimizer_args_custom", "")),
                        )
                        if part
                    )
                    if generic_optimizer_name and not has_optimizer_name_arg(existing_name_sources):
                        existing_optimizer_args = " ".join(
                            part
                            for part in (
                                stringify_arg_list(normalized_data.get("optimizer_args", "")),
                                stringify_arg_list(normalized_data.get("optimizer_args_custom", "")),
                            )
                            if part
                        )
                        normalized_data["optimizer_args"] = " ".join(
                            part for part in (f"name={generic_optimizer_name}", existing_optimizer_args) if part
                        )
                else:
                    normalized_data["optimizer_type"] = optimizer_aliases.get(
                        optimizer_key.lower().replace(" ", ""),
                        optimizer_key,
                    )
        optimizer_backend = str(normalized_data.get("optimizer_backend", "auto") or "auto").strip().lower().replace("-", "_")
        optimizer_backend_aliases = {
            "default": "auto",
            "torch": "torch_adamw",
            "adamw": "torch_adamw",
            "foreach": "foreach_adamw",
            "multi_tensor": "foreach_adamw",
            "fused": "torch_fused",
            "torchfused": "torch_fused",
            "bnb": "bnb_8bit",
            "bitsandbytes": "bnb_8bit",
            "bitsandbytes_8bit": "bnb_8bit",
            "lulynx": "lulynx_fused",
            "lulynx_fused_adamw": "lulynx_fused",
        }
        optimizer_backend = optimizer_backend_aliases.get(optimizer_backend.replace(" ", ""), optimizer_backend)
        if optimizer_backend not in {"auto", "torch_adamw", "foreach_adamw", "torch_fused", "bnb_8bit", "apex", "lulynx_fused"}:
            logger.warning("Unknown optimizer_backend=%r; using auto", optimizer_backend)
            optimizer_backend = "auto"
        normalized_data["optimizer_backend"] = optimizer_backend
        advanced_optimizer_strategy = str(normalized_data.get("advanced_optimizer_strategy", "auto") or "auto").strip().lower().replace("-", "_").replace("+", "_plus")
        advanced_optimizer_strategy_aliases = {
            "": "auto",
            "default": "auto",
            "none": "off",
            "disabled": "off",
            "dry_run": "profile_only",
            "profile": "profile_only",
            "manifest": "profile_only",
            "lora": "lora_plus",
            "loraplus": "lora_plus",
            "lora_plus_plus": "lora_plus",
            "rslora": "rs_lora",
            "rank_stabilized_lora": "rs_lora",
            "rank_stabilized": "rs_lora",
            "gradient_low_rank": "galore",
        }
        advanced_optimizer_strategy = advanced_optimizer_strategy_aliases.get(
            advanced_optimizer_strategy.replace(" ", ""),
            advanced_optimizer_strategy,
        )
        if advanced_optimizer_strategy not in {"auto", "off", "profile_only", "lora_plus", "rs_lora", "galore"}:
            logger.warning("Unknown advanced_optimizer_strategy=%r; using auto", advanced_optimizer_strategy)
            advanced_optimizer_strategy = "auto"
        normalized_data["advanced_optimizer_strategy"] = advanced_optimizer_strategy
        apply_v3_exact_adamw_canary_config_adapter(normalized_data)
        apply_v5_manual_wider_canary_config_adapter(normalized_data)
        data_backend = str(normalized_data.get("data_backend", "auto") or "auto").strip().lower().replace("-", "_")
        data_backend_aliases = {
            "": "auto",
            "default": "auto",
            "pil": "caption",
            "imagefolder": "caption",
            "image_folder": "caption",
            "caption_dataset": "caption",
            "captiondataset": "caption",
            "folder": "caption",
            "raw_caption": "caption",
            "tar": "webdataset",
            "tars": "webdataset",
            "wds": "webdataset",
            "web_dataset": "webdataset",
            "nvidia_dali": "dali",
        }
        data_backend = data_backend_aliases.get(data_backend.replace(" ", ""), data_backend)
        if data_backend not in {"auto", "caption", "raw", "webdataset", "dali"}:
            logger.warning("Unknown data_backend=%r; using auto", data_backend)
            data_backend = "auto"
        normalized_data["data_backend"] = data_backend
        image_decode_backend = str(normalized_data.get("image_decode_backend", "pil") or "pil").strip().lower().replace("-", "_")
        image_decode_aliases = {
            "": "pil",
            "default": "pil",
            "none": "pil",
            "off": "pil",
            "lru": "pil_lru",
            "pil_cache": "pil_lru",
            "cached_pil": "pil_lru",
            "torchvision": "torchvision_cpu",
            "torchvision_io": "torchvision_cpu",
            "torchvision_cpu_decode": "torchvision_cpu",
        }
        image_decode_backend = image_decode_aliases.get(image_decode_backend.replace(" ", ""), image_decode_backend)
        if image_decode_backend not in {"auto", "pil", "pil_lru", "torchvision_cpu"}:
            logger.warning("Unknown image_decode_backend=%r; using pil", image_decode_backend)
            image_decode_backend = "pil"
        normalized_data["image_decode_backend"] = image_decode_backend
        try:
            normalized_data["image_decode_cache_size"] = max(int(normalized_data.get("image_decode_cache_size", 0) or 0), 0)
        except (TypeError, ValueError):
            logger.warning("Invalid image_decode_cache_size=%r; using 0", normalized_data.get("image_decode_cache_size"))
            normalized_data["image_decode_cache_size"] = 0
        cached_collate_mode = str(normalized_data.get("cached_collate_mode", "auto") or "auto").strip().lower().replace("-", "_")
        cached_collate_aliases = {
            "": "auto",
            "default": "auto",
            "fast": "pad_sequence",
            "pad": "pad_sequence",
            "torch": "pad_sequence",
            "manual": "legacy",
            "prealloc": "legacy",
        }
        cached_collate_mode = cached_collate_aliases.get(cached_collate_mode.replace(" ", ""), cached_collate_mode)
        if cached_collate_mode not in {"auto", "legacy", "pad_sequence"}:
            logger.warning("Unknown cached_collate_mode=%r; using auto", cached_collate_mode)
            cached_collate_mode = "auto"
        normalized_data["cached_collate_mode"] = cached_collate_mode
        checkpoint_policy = str(normalized_data.get("checkpoint_policy", "auto") or "auto").strip().lower().replace("-", "_")
        checkpoint_policy_aliases = {
            "": "auto",
            "default": "auto",
            "none": "off",
            "disabled": "off",
            "gradient": "full",
            "gradient_checkpointing": "full",
            "checkpoint": "full",
            "cpu": "offloaded",
            "cpu_offload": "offloaded",
            "save_on_cpu": "offloaded",
            "sac": "selective",
            "selective_recompute": "selective",
            "selective_recomputation": "selective",
        }
        checkpoint_policy = checkpoint_policy_aliases.get(checkpoint_policy.replace(" ", ""), checkpoint_policy)
        if checkpoint_policy not in {"auto", "off", "full", "offloaded", "selective"}:
            logger.warning("Unknown checkpoint_policy=%r; using auto", checkpoint_policy)
            checkpoint_policy = "auto"
        normalized_data["checkpoint_policy"] = checkpoint_policy

        schema_id = str(
            normalized_data.get("schema_id")
            or normalized_data.get("model_train_type")
            or ""
        ).strip().lower().replace("_", "-")
        if schema_id:
            normalized_data.setdefault("schema_id", schema_id)
            if schema_id.endswith("-lora"):
                normalized_data.setdefault("training_type", "lora")
            elif schema_id.endswith("-finetune"):
                normalized_data.setdefault("training_type", "full_finetune")
            elif schema_id.endswith("-controlnet"):
                normalized_data.setdefault("training_type", "controlnet")
            elif schema_id.endswith("-textual-inversion"):
                normalized_data.setdefault("training_type", "textual_inversion")
            elif schema_id.endswith("-ip-adapter"):
                normalized_data.setdefault("training_type", "ip-adapter")
            elif schema_id.endswith("-dreambooth"):
                normalized_data.setdefault("training_type", "dreambooth")
            if schema_id.startswith("newbie-"):
                normalized_data.setdefault("model_type", "newbie")
            elif schema_id.startswith("anima-"):
                normalized_data.setdefault("model_type", "anima")
            elif schema_id.startswith("sdxl-"):
                normalized_data.setdefault("model_type", "sdxl")
            elif schema_id.startswith("sd-"):
                normalized_data.setdefault("model_type", "sd15")

        if "train_data_dir" not in normalized_data and "trainDataDir" in data:
            normalized_data["train_data_dir"] = data["trainDataDir"]
        if "pretrained_model" in normalized_data and "pretrained_model_name_or_path" not in normalized_data:
            normalized_data["pretrained_model_name_or_path"] = normalized_data.pop("pretrained_model")
        if not str(normalized_data.get("pretrained_model_name_or_path") or "").strip():
            for pretrained_alias in (
                "base_model_path",
                "pretrained_model_path",
                "sd_model_path",
                "checkpoint_path",
            ):
                if pretrained_alias not in normalized_data:
                    continue
                candidate = normalized_data.pop(pretrained_alias)
                if str(candidate or "").strip():
                    normalized_data["pretrained_model_name_or_path"] = candidate
                    break
        if "optimizer_args" in normalized_data:
            normalized_data["optimizer_args"] = stringify_arg_list(normalized_data.get("optimizer_args"))
        if "network_args_custom" in normalized_data:
            custom_network_args = stringify_arg_list(normalized_data.get("network_args_custom"))
            if custom_network_args:
                if "network_args" in normalized_data:
                    existing_network_args = stringify_arg_list(normalized_data.get("network_args"))
                    normalized_data["network_args"] = " ".join(
                        part for part in (existing_network_args, custom_network_args) if part
                    )
                else:
                    normalized_data["network_args"] = custom_network_args
            normalized_data.pop("network_args_custom", None)
        if "network_args" in normalized_data:
            normalized_data["network_args"] = stringify_arg_list(normalized_data.get("network_args"))
        if "flow_model" in normalized_data:
            normalized_data["flow_model"] = stringify_scalar(normalized_data.get("flow_model"))
        if "flow_timestep_distribution" in normalized_data and "timestep_sampling" not in normalized_data:
            normalized_data["timestep_sampling"] = normalized_data["flow_timestep_distribution"]
        if "bw_mid_weight" in normalized_data:
            normalized_data["bw_mid_weight"] = stringify_scalar(normalized_data.get("bw_mid_weight"))
        if "controlnet_model_name_or_path" in normalized_data and "controlnet_model" not in normalized_data:
            normalized_data["controlnet_model"] = normalized_data["controlnet_model_name_or_path"]
        # LLLite field defaults
        if normalized_data.get("training_type") == "lllite":
            normalized_data.setdefault("lllite_cond_emb_dim", 32)
            normalized_data.setdefault("lllite_mlp_dim", 64)
            normalized_data.setdefault("lllite_dropout", 0.0)
            normalized_data.setdefault("lllite_skip_input_blocks", False)
            normalized_data.setdefault("lllite_skip_output_blocks", True)
        if "newbie_model_path" in normalized_data and "newbie_diffusers_path" not in normalized_data:
            normalized_data["newbie_diffusers_path"] = normalized_data.pop("newbie_model_path")
        if "resume" in normalized_data and "resume_path" not in normalized_data:
            normalized_data["resume_path"] = normalized_data["resume"]
        if "execution_profile_id" in normalized_data and "runtime_id" not in normalized_data:
            normalized_data["runtime_id"] = normalized_data["execution_profile_id"]
        if "runtime_id" in normalized_data and "execution_profile_id" not in normalized_data:
            normalized_data["execution_profile_id"] = normalized_data["runtime_id"]
        if "anima_model" in normalized_data and "anima_model_path" not in normalized_data:
            normalized_data["anima_model_path"] = normalized_data.pop("anima_model")
        for anima_model_alias in ("qwen_image_model_path", "qwen_image_dit_path", "dit_path"):
            if anima_model_alias in normalized_data and "anima_model_path" not in normalized_data:
                normalized_data["anima_model_path"] = normalized_data[anima_model_alias]
        if (
            normalized_data.get("model_type") == "anima"
            and "anima_model_path" not in normalized_data
            and normalized_data.get("pretrained_model_name_or_path")
        ):
            normalized_data["anima_model_path"] = normalized_data["pretrained_model_name_or_path"]
        if "anima_transformer_path" in normalized_data and "anima_model_path" not in normalized_data:
            normalized_data["anima_model_path"] = normalized_data["anima_transformer_path"]
        if "qwen3" in normalized_data and "anima_qwen3_path" not in normalized_data:
            normalized_data["anima_qwen3_path"] = normalized_data["qwen3"]
        if "t5_tokenizer_path" in normalized_data and "anima_t5_tokenizer_path" not in normalized_data:
            normalized_data["anima_t5_tokenizer_path"] = normalized_data["t5_tokenizer_path"]
        if "llm_adapter_path" in normalized_data and "anima_llm_adapter_path" not in normalized_data:
            normalized_data["anima_llm_adapter_path"] = normalized_data["llm_adapter_path"]
        if "dit_adapter_path" in normalized_data and "anima_dit_adapter_path" not in normalized_data:
            normalized_data["anima_dit_adapter_path"] = normalized_data["dit_adapter_path"]
        if "network_weights" in normalized_data and "network_weights_path" not in normalized_data:
            normalized_data["network_weights_path"] = normalized_data["network_weights"]
        if "anima_cache_mode" in normalized_data and "native_cache_mode" not in normalized_data:
            normalized_data["native_cache_mode"] = normalized_data["anima_cache_mode"]
        if "cache_mode" in normalized_data and "native_cache_mode" not in normalized_data:
            normalized_data["native_cache_mode"] = normalized_data["cache_mode"]
        if "cooldown_until_temp_c" in normalized_data and "cooldown_until_temp" not in normalized_data:
            normalized_data["cooldown_until_temp"] = normalized_data["cooldown_until_temp_c"]

        valid_swap_granularities = {"off", "auto", "block", "merged_block", "layer"}
        swap_granularity = str(normalized_data.get("swap_granularity", "off") or "off").strip().lower().replace("-", "_")
        if swap_granularity not in valid_swap_granularities:
            logger.warning("Unknown swap_granularity=%r; disabling swap", swap_granularity)
            swap_granularity = "off"
        normalized_data["swap_granularity"] = swap_granularity

        try:
            normalized_data["swap_ratio"] = max(0.0, min(1.0, float(normalized_data.get("swap_ratio", 0.0) or 0.0)))
        except (TypeError, ValueError):
            normalized_data["swap_ratio"] = 0.0
        try:
            normalized_data["swap_count"] = max(0, int(normalized_data.get("swap_count", 0) or 0))
        except (TypeError, ValueError):
            normalized_data["swap_count"] = 0
        try:
            normalized_data["block_merge_size"] = max(2, int(normalized_data.get("block_merge_size", 2) or 2))
        except (TypeError, ValueError):
            normalized_data["block_merge_size"] = 2
        block_swap_strategy = (
            str(normalized_data.get("block_swap_strategy", "auto") or "auto")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if block_swap_strategy not in {"auto", "sync", "async", "pipeline"}:
            logger.warning("Unknown block_swap_strategy=%r; using auto", block_swap_strategy)
            block_swap_strategy = "auto"
        normalized_data["block_swap_strategy"] = block_swap_strategy

        try:
            legacy_blocks_to_swap = max(0, int(normalized_data.get("blocks_to_swap", 0) or 0))
        except (TypeError, ValueError):
            legacy_blocks_to_swap = 0
        normalized_data["blocks_to_swap"] = legacy_blocks_to_swap
        if legacy_blocks_to_swap > 0 and swap_granularity == "off" and normalized_data["swap_count"] == 0 and normalized_data["swap_ratio"] == 0.0:
            normalized_data["swap_granularity"] = "block"
            normalized_data["swap_count"] = legacy_blocks_to_swap

        normalized_data["module_offload_enabled"] = boolish(normalized_data.get("module_offload_enabled", False))
        normalized_data["module_offload_ratio"] = clamp_module_offload_ratio(
            normalized_data.get("module_offload_ratio", 0),
            default=0,
        )
        normalized_data["module_offload_backbone_ratio"] = parse_optional_module_offload_ratio(
            normalized_data.get("module_offload_backbone_ratio", None)
        )
        normalized_data["module_offload_text_encoder_ratio"] = parse_optional_module_offload_ratio(
            normalized_data.get("module_offload_text_encoder_ratio", None)
        )
        normalized_data["module_offload_profile_enabled"] = boolish(
            normalized_data.get("module_offload_profile_enabled", False)
        )
        normalized_data["module_offload_profile"] = normalize_module_offload_profile(
            normalized_data.get("module_offload_profile", "custom")
        )
        normalized_data["module_offload_min_param_mb"] = parse_module_offload_float(
            normalized_data.get("module_offload_min_param_mb", 0.0)
        )
        normalized_data["module_offload_include_patterns"] = normalize_module_offload_patterns(
            normalized_data.get("module_offload_include_patterns", "")
        )
        normalized_data["module_offload_exclude_patterns"] = normalize_module_offload_patterns(
            normalized_data.get("module_offload_exclude_patterns", "")
        )
        normalized_data["module_offload_verify_state"] = boolish(
            normalized_data.get("module_offload_verify_state", True),
            default=True,
        )
        normalized_data["module_offload_prefetch_enabled"] = boolish(
            normalized_data.get("module_offload_prefetch_enabled", False)
        )
        normalized_data["module_offload_prefetch_mode"] = str(
            normalized_data.get("module_offload_prefetch_mode", "experimental") or "experimental"
        ).strip().lower()
        if normalized_data["module_offload_prefetch_mode"] != "experimental":
            normalized_data["module_offload_prefetch_mode"] = "experimental"

        if (
            normalized_data.get("model_type") == "anima"
            and "adapter_type" in normalized_data
            and "lora_type" not in normalized_data
        ):
            normalized_data["lora_type"] = normalized_data["adapter_type"]

        # Newbie native component paths
        if "gemma3_prompt" in normalized_data and "newbie_gemma3_prompt" not in normalized_data:
            normalized_data["newbie_gemma3_prompt"] = normalized_data["gemma3_prompt"]
        if "gemma_model_path" in normalized_data and "newbie_gemma_model_path" not in normalized_data:
            normalized_data["newbie_gemma_model_path"] = normalized_data["gemma_model_path"]
        if "gemma2" in normalized_data and "newbie_gemma_model_path" not in normalized_data:
            normalized_data["newbie_gemma_model_path"] = normalized_data["gemma2"]
        if "gemma2_max_token_length" in normalized_data and "newbie_gemma_max_token_length" not in normalized_data:
            normalized_data["newbie_gemma_max_token_length"] = normalized_data["gemma2_max_token_length"]
        if normalized_data.get("model_type") == "flux" and "transformer_path" in normalized_data and "flux_transformer_path" not in normalized_data:
            normalized_data["flux_transformer_path"] = normalized_data["transformer_path"]
        if normalized_data.get("model_type") != "flux" and "transformer_path" in normalized_data and "newbie_transformer_path" not in normalized_data:
            normalized_data["newbie_transformer_path"] = normalized_data["transformer_path"]
        if "clip_model_path" in normalized_data and "newbie_clip_model_path" not in normalized_data:
            normalized_data["newbie_clip_model_path"] = normalized_data["clip_model_path"]
        if "vae_path" in normalized_data and "newbie_vae_path" not in normalized_data:
            normalized_data["newbie_vae_path"] = normalized_data["vae_path"]
        if "adapter_type" in normalized_data and "newbie_adapter_type" not in normalized_data:
            normalized_data["newbie_adapter_type"] = normalized_data["adapter_type"]
        if "lokr_rank" in normalized_data and "network_dim" not in normalized_data:
            normalized_data["network_dim"] = normalized_data["lokr_rank"]
        if "lokr_alpha" in normalized_data and "network_alpha" not in normalized_data:
            normalized_data["network_alpha"] = normalized_data["lokr_alpha"]
        if "lokr_dropout" in normalized_data and "network_dropout" not in normalized_data:
            normalized_data["network_dropout"] = normalized_data["lokr_dropout"]
        if "lokr_factor" in normalized_data and "lycoris_lokr_factor" not in normalized_data:
            normalized_data["lycoris_lokr_factor"] = normalized_data["lokr_factor"]
        if "lycoris_factor" in normalized_data and "lycoris_lokr_factor" not in normalized_data:
            normalized_data["lycoris_lokr_factor"] = normalized_data["lycoris_factor"]
        if "lycoris_preset" in normalized_data and "lycoris_presets" not in normalized_data:
            normalized_data["lycoris_presets"] = stringify_csv_list(normalized_data["lycoris_preset"])
        if "dropout" in normalized_data and "network_dropout" not in normalized_data:
            normalized_data["network_dropout"] = normalized_data["dropout"]
        if "rank_dropout" in normalized_data and "lokr_rank_dropout" not in normalized_data:
            normalized_data["lokr_rank_dropout"] = normalized_data["rank_dropout"]
        if "module_dropout" in normalized_data and "lokr_module_dropout" not in normalized_data:
            normalized_data["lokr_module_dropout"] = normalized_data["module_dropout"]
        if "full_matrix" in normalized_data and "lokr_full_matrix" not in normalized_data:
            normalized_data["lokr_full_matrix"] = normalized_data["full_matrix"]
        if "lokr_full_matrix" in normalized_data:
            normalized_data["lokr_full_matrix"] = boolish(normalized_data["lokr_full_matrix"])
        if "decompose_both" in normalized_data and "lokr_decompose_both" not in normalized_data:
            normalized_data["lokr_decompose_both"] = normalized_data["decompose_both"]
        if "lokr_decompose_both" in normalized_data:
            normalized_data["lokr_decompose_both"] = boolish(normalized_data["lokr_decompose_both"])
        if "unbalanced_factorization" in normalized_data and "lokr_unbalanced_factorization" not in normalized_data:
            normalized_data["lokr_unbalanced_factorization"] = normalized_data["unbalanced_factorization"]
        if "lokr_unbalanced_factorization" in normalized_data:
            normalized_data["lokr_unbalanced_factorization"] = boolish(normalized_data["lokr_unbalanced_factorization"])
        if "no_materialize_forward" in normalized_data and "lokr_no_materialize_forward" not in normalized_data:
            normalized_data["lokr_no_materialize_forward"] = normalized_data["no_materialize_forward"]
        if "lokr_no_materialize_forward" in normalized_data:
            normalized_data["lokr_no_materialize_forward"] = boolish(normalized_data["lokr_no_materialize_forward"])
        if "no_materialize_strategy" in normalized_data and "lokr_no_materialize_strategy" not in normalized_data:
            normalized_data["lokr_no_materialize_strategy"] = normalized_data["no_materialize_strategy"]
        if "lokr_no_materialize_strategy" in normalized_data:
            strategy = str(normalized_data["lokr_no_materialize_strategy"] or "legacy").strip().lower()
            normalized_data["lokr_no_materialize_strategy"] = strategy if strategy in {"auto", "legacy", "matmul"} else "legacy"
        if "lokr_export_mode" in normalized_data:
            mode = str(normalized_data["lokr_export_mode"]).strip().lower()
            normalized_data["lokr_export_mode"] = mode if mode in {"native", "lora_compatible"} else "native"
        if "newbie_target_modules" in normalized_data:
            raw_targets = normalized_data["newbie_target_modules"]
            if isinstance(raw_targets, (list, tuple)):
                normalized_data["newbie_target_modules"] = ",".join(
                    str(item).strip() for item in raw_targets if str(item).strip()
                )
            elif raw_targets is not None:
                normalized_data["newbie_target_modules"] = str(raw_targets)
        if "use_cache" in normalized_data:
            normalized_data["use_cache"] = boolish(normalized_data["use_cache"])
        if "newbie_force_cache_only" in normalized_data:
            normalized_data["newbie_force_cache_only"] = boolish(normalized_data["newbie_force_cache_only"])
        if "newbie_rebuild_cache" in normalized_data:
            normalized_data["newbie_rebuild_cache"] = boolish(normalized_data["newbie_rebuild_cache"])
        if "trust_cache" in normalized_data:
            normalized_data["trust_cache"] = boolish(normalized_data["trust_cache"])
        alias_pairs = (
            ("concept_geometry_enabled", "h_lora_enabled"),
            ("concept_geometry_path", "h_lora_geometry_path"),
            ("concept_geometry_sampler_mode", "h_lora_sampler_mode"),
            ("concept_geometry_loss_weighting", "h_lora_loss_weighting"),
            ("concept_geometry_density_power", "h_lora_density_power"),
        )
        for modern, legacy in alias_pairs:
            if modern in normalized_data and legacy not in normalized_data:
                normalized_data[legacy] = normalized_data[modern]
            elif legacy in normalized_data and modern not in normalized_data:
                normalized_data[modern] = normalized_data[legacy]
        for key in ("concept_geometry_enabled", "h_lora_enabled"):
            if key in normalized_data:
                normalized_data[key] = boolish(normalized_data[key])
        for key in ("concept_geometry_loss_weighting", "h_lora_loss_weighting"):
            if key in normalized_data:
                normalized_data[key] = boolish(normalized_data[key])
        for key in ("concept_geometry_sampler_mode", "h_lora_sampler_mode"):
            if key in normalized_data:
                mode = str(normalized_data[key] or "density_curriculum").strip().lower().replace("-", "_")
                normalized_data[key] = (
                    mode if mode in {"curriculum", "density", "density_curriculum", "concept_batch"} else "density_curriculum"
                )
        for key in ("concept_geometry_density_power", "h_lora_density_power"):
            if key not in normalized_data:
                continue
            try:
                normalized_data[key] = max(float(normalized_data[key] or 1.0), 0.0)
            except (TypeError, ValueError):
                normalized_data[key] = 1.0
        for int_key in ("eval_every_n_epochs", "eval_every_n_steps", "eval_batch_size", "max_validation_steps"):
            if int_key in normalized_data:
                try:
                    normalized_data[int_key] = max(0, int(normalized_data[int_key] or 0))
                except (TypeError, ValueError):
                    normalized_data[int_key] = 0
        if "trust_remote_code" in normalized_data:
            normalized_data["trust_remote_code"] = boolish(normalized_data["trust_remote_code"])
        if "pytorch_cuda_expandable_segments" in normalized_data:
            normalized_data["pytorch_cuda_expandable_segments"] = boolish(
                normalized_data["pytorch_cuda_expandable_segments"]
            )
        if "cpu_offload_checkpointing" in normalized_data:
            normalized_data["cpu_offload_checkpointing"] = boolish(
                normalized_data["cpu_offload_checkpointing"]
            )
        if "vram_swap_to_ram" in normalized_data:
            normalized_data["vram_swap_to_ram"] = boolish(
                normalized_data["vram_swap_to_ram"]
            )
        if "sdxl_low_vram_optimization" in normalized_data:
            normalized_data["sdxl_low_vram_optimization"] = boolish(
                normalized_data["sdxl_low_vram_optimization"]
            )
        if "low_vram_profile" in normalized_data:
            normalized_data["low_vram_profile"] = normalize_low_vram_profile(
                normalized_data["low_vram_profile"]
            )
        acceleration_profile = normalize_acceleration_profile(
            normalized_data.get("acceleration_profile", normalized_data.get("speed_profile", "off"))
        )
        normalized_data["acceleration_profile"] = acceleration_profile
        normalized_data["speed_profile"] = acceleration_profile
        if acceleration_profile != "off":
            acceleration = apply_model_acceleration_policy_to_config(
                normalized_data,
                schema_id=str(normalized_data.get("schema_id", "") or ""),
                training_type=str(normalized_data.get("training_type", "") or ""),
            )
            normalized_data = acceleration.config
        for str_key, allowed, default in (
            ("te_vae_offload_strategy", {"resident", "phase", "aggressive"}, "phase"),
            ("flux_transformer_offload", {"auto", "off", "aggressive"}, "auto"),
            ("preview_device", {"cpu", "gpu", "off"}, "cpu"),
            ("lulynx_precision_swap_strategy", {"off", "balanced", "aggressive"}, "balanced"),
            ("sdxl_unet_backend", {"diffusers", "native_shadow", "native_proxy", "native_skeleton", "lulynx_native"}, "diffusers"),
            ("lulynx_weight_residency", {"resident", "linear_cpu_pinned", "linear_conv_cpu_pinned"}, "resident"),
            ("pcie_transfer_format", {"off", "raw_fp16", "raw_bf16", "fp8_e4m3", "int8_rowwise", "uint4_rowwise"}, "off"),
            ("pcie_delta_cache_mode", {"observe", "cache_v0"}, "observe"),
            ("anima_block_residency", {"resident", "streaming_offload", "block_cpu_pinned"}, "resident"),
            ("newbie_block_residency", {"resident", "streaming_offload", "block_cpu_pinned"}, "resident"),
            ("anima_block_checkpointing_mode", {"block"}, "block"),
            ("newbie_block_checkpointing_mode", {"block"}, "block"),
            (
                "cuda_cache_release_strategy",
                {"off", "oom_only", "phase_boundary", "after_optimizer", "aggressive", "every_step", "after_step"},
                "oom_only",
            ),
            ("lora_activation_recompute_mode", {"auto", "on", "off"}, "auto"),
        ):
            if str_key in normalized_data:
                value = str(normalized_data.get(str_key) or default).strip().lower()
                normalized_data[str_key] = value if value in allowed else default
        for bool_key in (
            "ephemeral_preview_pipeline",
            "model_to_condition_enabled",
            "lulynx_precision_swap_enabled",
            "lora_activation_recompute",
            "anima_train_llm_adapter",
            "anima_block_checkpointing",
            "newbie_block_checkpointing",
            "anima_block_prefetch",
            "newbie_block_prefetch",
            "vram_smart_sensing_enabled",
            "vram_auto_enhance_enabled",
            "vram_smart_sensing_streaming_enabled",
            "vram_smart_sensing_sparse_swap_enabled",
            "vram_smart_sensing_delta_cache_enabled",
            "enhanced_protection_mode",
            "pcie_delta_cache_enabled",
            "sparse_swap_enabled",
            "enable_mixed_resolution_training",
        ):
            if bool_key in normalized_data:
                normalized_data[bool_key] = boolish(normalized_data[bool_key])
        for int_key in (
            "lulynx_weight_residency_min_params",
            "anima_block_residency_min_params",
            "newbie_block_residency_min_params",
            "anima_block_prefetch_depth",
            "newbie_block_prefetch_depth",
            "vram_smart_sensing_baseline_steps",
            "vram_smart_sensing_window_steps",
        ):
            if int_key not in normalized_data:
                continue
            try:
                normalized_data[int_key] = max(int(float(normalized_data.get(int_key) or 0)), 0)
            except Exception:
                normalized_data[int_key] = 0
        if "vram_smart_sensing_slowdown_ratio" in normalized_data:
            try:
                normalized_data["vram_smart_sensing_slowdown_ratio"] = max(
                    float(normalized_data.get("vram_smart_sensing_slowdown_ratio") or 1.5),
                    1.05,
                )
            except Exception:
                normalized_data["vram_smart_sensing_slowdown_ratio"] = 1.5
        if "sparse_swap_budget_mb" in normalized_data:
            try:
                normalized_data["sparse_swap_budget_mb"] = max(float(normalized_data.get("sparse_swap_budget_mb") or 0.0), 0.0)
            except Exception:
                normalized_data["sparse_swap_budget_mb"] = 0.0
        if "pcie_delta_cache_budget_mb" in normalized_data:
            try:
                normalized_data["pcie_delta_cache_budget_mb"] = max(float(normalized_data.get("pcie_delta_cache_budget_mb") or 256.0), 0.0)
            except Exception:
                normalized_data["pcie_delta_cache_budget_mb"] = 256.0
        if "sparse_swap_warm_fraction" in normalized_data:
            try:
                normalized_data["sparse_swap_warm_fraction"] = min(max(float(normalized_data.get("sparse_swap_warm_fraction") or 0.35), 0.0), 1.0)
            except Exception:
                normalized_data["sparse_swap_warm_fraction"] = 0.35
        if "cuda_cache_release_interval" in normalized_data:
            try:
                normalized_data["cuda_cache_release_interval"] = max(
                    int(float(normalized_data.get("cuda_cache_release_interval") or 1)),
                    1,
                )
            except Exception:
                normalized_data["cuda_cache_release_interval"] = 1
        if "enable_sequential_cpu_offload" in normalized_data:
            normalized_data["enable_sequential_cpu_offload"] = boolish(
                normalized_data["enable_sequential_cpu_offload"]
            )
        if "fp8_base" in normalized_data:
            normalized_data["fp8_base"] = boolish(normalized_data["fp8_base"])
        for bool_key in (
            "weight_compression_enabled",
            "weight_compression_verify",
            "weight_compression_allow_offload_combo",
            "compression_companion_enabled",
        ):
            if bool_key in normalized_data:
                normalized_data[bool_key] = boolish(normalized_data[bool_key])
        for str_key in (
            "weight_compression_preset",
            "weight_compression_target",
            "weight_compression_format",
            "weight_compression_include_patterns",
            "weight_compression_exclude_patterns",
            "compression_companion_path",
            "compression_companion_type",
            "compression_companion_mode",
        ):
            if str_key in normalized_data and normalized_data[str_key] is None:
                normalized_data[str_key] = ""
        if "compression_companion_scale" in normalized_data:
            try:
                normalized_data["compression_companion_scale"] = float(normalized_data["compression_companion_scale"] or 1.0)
            except (TypeError, ValueError):
                normalized_data["compression_companion_scale"] = 1.0
        if "merge_export" in normalized_data:
            normalized_data["merge_export"] = boolish(normalized_data["merge_export"])
        if "anima_merge_export" in normalized_data:
            normalized_data["anima_merge_export"] = boolish(normalized_data["anima_merge_export"])
        if "reft_enabled" in normalized_data:
            normalized_data["reft_enabled"] = boolish(normalized_data["reft_enabled"])
        if "reft_targets" in normalized_data and "reft_target_modules" not in normalized_data:
            normalized_data["reft_target_modules"] = normalized_data["reft_targets"]
        if "hydralora_enabled" in normalized_data:
            normalized_data["hydralora_enabled"] = boolish(normalized_data["hydralora_enabled"])
        if "hydra_lora_enabled" in normalized_data and "hydralora_enabled" not in normalized_data:
            normalized_data["hydralora_enabled"] = boolish(normalized_data["hydra_lora_enabled"])
        if "hydra_num_experts" in normalized_data and "hydralora_num_experts" not in normalized_data:
            normalized_data["hydralora_num_experts"] = normalized_data["hydra_num_experts"]
        if "hydra_routing" in normalized_data and "hydralora_routing" not in normalized_data:
            normalized_data["hydralora_routing"] = normalized_data["hydra_routing"]
        if "hydra_top_k" in normalized_data and "hydralora_top_k" not in normalized_data:
            normalized_data["hydralora_top_k"] = normalized_data["hydra_top_k"]
        if "hydra_sparse_top_k" in normalized_data and "hydralora_sparse_top_k" not in normalized_data:
            normalized_data["hydralora_sparse_top_k"] = normalized_data["hydra_sparse_top_k"]
        if "sparse_top_k" in normalized_data and "hydralora_sparse_top_k" not in normalized_data:
            normalized_data["hydralora_sparse_top_k"] = normalized_data["sparse_top_k"]
        if "hydralora_sparse_top_k" in normalized_data:
            normalized_data["hydralora_sparse_top_k"] = boolish(normalized_data["hydralora_sparse_top_k"])
        if "fera_enabled" in normalized_data:
            normalized_data["fera_enabled"] = boolish(normalized_data["fera_enabled"])
        if "repa_enabled" in normalized_data:
            normalized_data["repa_enabled"] = boolish(normalized_data["repa_enabled"])
        if "repa_layers" in normalized_data and "repa_target_modules" not in normalized_data:
            normalized_data["repa_target_modules"] = normalized_data["repa_layers"]
        if "repa_targets" in normalized_data and "repa_target_modules" not in normalized_data:
            normalized_data["repa_target_modules"] = normalized_data["repa_targets"]
        if "repa_stop_grad_target" in normalized_data:
            normalized_data["repa_stop_grad_target"] = boolish(normalized_data["repa_stop_grad_target"], default=True)
        if "softrepa_enabled" in normalized_data:
            normalized_data["softrepa_enabled"] = boolish(normalized_data["softrepa_enabled"])
        if "soft_repa_enabled" in normalized_data and "softrepa_enabled" not in normalized_data:
            normalized_data["softrepa_enabled"] = boolish(normalized_data["soft_repa_enabled"])
        if "softrepa_layers" in normalized_data and "repa_target_modules" not in normalized_data:
            normalized_data["repa_target_modules"] = normalized_data["softrepa_layers"]
        if "softrepa_weight" in normalized_data and "softrepa_max_weight" not in normalized_data:
            normalized_data["softrepa_max_weight"] = normalized_data["softrepa_weight"]
        if "softrepa_enabled" in normalized_data:
            normalized_data.setdefault("repa_enabled", normalized_data["softrepa_enabled"])
            normalized_data.setdefault("repa_loss_weight", 1.0)
        if "save_precision" not in normalized_data and "output_dtype" in normalized_data:
            normalized_data["save_precision"] = normalized_data["output_dtype"]
        if "experimental_attention_profile_enabled" in normalized_data:
            normalized_data["experimental_attention_profile_enabled"] = boolish(
                normalized_data["experimental_attention_profile_enabled"]
            )
        if "experimental_attention_profile_backend" in normalized_data:
            backend = str(normalized_data["experimental_attention_profile_backend"] or "auto").strip().lower()
            normalized_data["experimental_attention_profile_backend"] = (
                backend if backend in {"auto", "flex", "flexattn", "sdpa", "sdpa_masked", "torch", "torch_fallback"} else "auto"
            )
        if "data_transfer_profile_mode" in normalized_data:
            mode = str(normalized_data["data_transfer_profile_mode"] or "event").strip().lower()
            normalized_data["data_transfer_profile_mode"] = mode if mode in {"event", "sync", "off"} else "event"
        if "loss_precision" in normalized_data:
            mode = str(normalized_data["loss_precision"] or "fp32_loss").strip().lower()
            aliases = {
                "fp32": "fp32_loss",
                "float32": "fp32_loss",
                "full": "fp32_loss",
                "safe": "fp32_loss",
                "mixed": "mixed_loss",
                "native": "mixed_loss",
                "amp": "mixed_loss",
            }
            mode = aliases.get(mode, mode)
            normalized_data["loss_precision"] = mode if mode in {"fp32_loss", "mixed_loss"} else "fp32_loss"
        if "cross_attn_fused_kv" in normalized_data:
            normalized_data["cross_attn_fused_kv"] = boolish(
                normalized_data["cross_attn_fused_kv"]
            )
        if "anima_fused_qkv" in normalized_data:
            normalized_data["anima_fused_qkv"] = boolish(normalized_data["anima_fused_qkv"])
        fused_projection_memory_mode = str(
            normalized_data.get("fused_projection_memory_mode", "keep_original") or "keep_original"
        ).strip().lower().replace("-", "_")
        fused_projection_aliases = {
            "": "keep_original",
            "auto": "keep_original",
            "keep": "keep_original",
            "compat": "keep_original",
            "drop": "drop_original",
            "delete": "drop_original",
            "save": "materialize_on_save",
            "materialize": "materialize_on_save",
            "materialize_save": "materialize_on_save",
        }
        fused_projection_memory_mode = fused_projection_aliases.get(
            fused_projection_memory_mode,
            fused_projection_memory_mode,
        )
        if fused_projection_memory_mode not in {"keep_original", "drop_original", "materialize_on_save"}:
            logger.warning("Unknown fused_projection_memory_mode=%r; using keep_original", fused_projection_memory_mode)
            fused_projection_memory_mode = "keep_original"
        normalized_data["fused_projection_memory_mode"] = fused_projection_memory_mode
        if "blockwise_fused_optimizers" in normalized_data:
            normalized_data["blockwise_fused_optimizers"] = boolish(
                normalized_data["blockwise_fused_optimizers"]
            )
        if "lulynx_lisa_enabled" in normalized_data and "lisa_enabled" not in normalized_data:
            normalized_data["lisa_enabled"] = normalized_data["lulynx_lisa_enabled"]
        if "lulynx_lisa_active_ratio" in normalized_data and "lisa_active_ratio" not in normalized_data:
            normalized_data["lisa_active_ratio"] = normalized_data["lulynx_lisa_active_ratio"]
        if "lulynx_lisa_interval" in normalized_data and "lisa_interval" not in normalized_data:
            normalized_data["lisa_interval"] = normalized_data["lulynx_lisa_interval"]

        # 2.1 前端兼容别名 -> 训练器实际开关
        if "pissa_enabled" not in normalized_data and "use_pissa" in normalized_data:
            normalized_data["pissa_enabled"] = normalized_data["use_pissa"]
        if "pissa_init" in normalized_data:
            pissa_init_enabled = boolish(normalized_data["pissa_init"])
            normalized_data.setdefault("use_pissa", pissa_init_enabled)
            normalized_data.setdefault("pissa_enabled", pissa_init_enabled)
        if "pissa_niter" in normalized_data and "pissa_init_iters" not in normalized_data:
            normalized_data["pissa_init_iters"] = normalized_data["pissa_niter"]
        cls._normalize_adapter_init_values(normalized_data, boolish)
        if "smart_rank_enabled" not in normalized_data and "smart_rank" in normalized_data:
            normalized_data["smart_rank_enabled"] = normalized_data["smart_rank"]
        if "smart_rank_enabled" not in normalized_data and "lulynx_smart_rank_enabled" in normalized_data:
            normalized_data["smart_rank_enabled"] = boolish(normalized_data["lulynx_smart_rank_enabled"])
        if "multi_gpu" not in normalized_data and "enable_distributed_training" in normalized_data:
            normalized_data["multi_gpu"] = boolish(normalized_data["enable_distributed_training"])
        if "base_weight_path" not in normalized_data and "base_weights" in normalized_data:
            normalized_data["base_weight_path"] = stringify_csv_list(normalized_data["base_weights"])
        if "base_weight_path" in normalized_data:
            normalized_data["base_weight_path"] = stringify_csv_list(normalized_data["base_weight_path"])
        if "base_weights_multiplier" in normalized_data:
            normalized_data["base_weights_multiplier"] = stringify_csv_list(normalized_data["base_weights_multiplier"])
        if "auto_controller_enabled" not in normalized_data and "lulynx_auto_controller_enabled" in normalized_data:
            normalized_data["auto_controller_enabled"] = boolish(normalized_data["lulynx_auto_controller_enabled"])
        if "ac_early_stopping_patience" not in normalized_data and "lulynx_auto_early_stop_patience" in normalized_data:
            normalized_data["ac_early_stopping_patience"] = normalized_data["lulynx_auto_early_stop_patience"]
        if "rm_enable_adaptive_batch" not in normalized_data and "lulynx_resource_manager_enabled" in normalized_data:
            normalized_data["rm_enable_adaptive_batch"] = boolish(normalized_data["lulynx_resource_manager_enabled"])
        for bool_key in (
            "smart_rank_enabled",
            "multi_gpu",
            "auto_controller_enabled",
            "rm_enable_adaptive_batch",
        ):
            if bool_key in normalized_data:
                normalized_data[bool_key] = boolish(normalized_data[bool_key])
        if "warmup_ratio" not in normalized_data and "lr_warmup_ratio" in normalized_data:
            normalized_data["warmup_ratio"] = normalized_data["lr_warmup_ratio"]
        if "lr_scheduler_type" in normalized_data and "lr_scheduler" not in normalized_data:
            normalized_data["lr_scheduler"] = normalized_data["lr_scheduler_type"]
        if "optimizer_args_custom" in normalized_data and "optimizer_args" not in normalized_data:
            normalized_data["optimizer_args"] = stringify_arg_list(normalized_data["optimizer_args_custom"])
        if "text_encoder_batch_size" not in normalized_data and "te_batch_size" in normalized_data:
            normalized_data["text_encoder_batch_size"] = normalized_data["te_batch_size"]
        if (
            "text_encoder_outputs_cache_disk_dtype" not in normalized_data
            and "text_encoder_outputs_cache_dtype" in normalized_data
        ):
            normalized_data["text_encoder_outputs_cache_disk_dtype"] = normalized_data["text_encoder_outputs_cache_dtype"]
        if "train_batch_size" not in normalized_data and "batch_size" in normalized_data:
            normalized_data["train_batch_size"] = normalized_data["batch_size"]
        if "max_train_epochs" not in normalized_data and "epochs" in normalized_data:
            normalized_data["max_train_epochs"] = normalized_data["epochs"]
        if "dataloader_num_workers" not in normalized_data and "num_workers" in normalized_data:
            normalized_data["dataloader_num_workers"] = normalized_data["num_workers"]
        if "dataloader_num_workers" not in normalized_data and "workers" in normalized_data:
            normalized_data["dataloader_num_workers"] = normalized_data["workers"]
        if "mem_efficient_save" not in normalized_data and "mem_eff_save" in normalized_data:
            normalized_data["mem_efficient_save"] = normalized_data["mem_eff_save"]
        if "ema_use_ema" not in normalized_data and "use_ema" in normalized_data:
            normalized_data["ema_use_ema"] = normalized_data["use_ema"]
        if "ema_use_ema" not in normalized_data and "ema_enabled" in normalized_data:
            normalized_data["ema_use_ema"] = normalized_data["ema_enabled"]
        if "ema_use_ema" not in normalized_data and "lulynx_ema_enabled" in normalized_data:
            normalized_data["ema_use_ema"] = normalized_data["lulynx_ema_enabled"]
        if "ema_decay" not in normalized_data and "lulynx_ema_decay" in normalized_data:
            normalized_data["ema_decay"] = normalized_data["lulynx_ema_decay"]
        if "pcgrad_enabled" not in normalized_data and "lulynx_pcgrad_enabled" in normalized_data:
            normalized_data["pcgrad_enabled"] = normalized_data["lulynx_pcgrad_enabled"]
        if "pcgrad_conflict_threshold" not in normalized_data and "lulynx_pcgrad_conflict_threshold" in normalized_data:
            normalized_data["pcgrad_conflict_threshold"] = normalized_data["lulynx_pcgrad_conflict_threshold"]
        if "pcgrad_reduction" not in normalized_data and "lulynx_pcgrad_reduction" in normalized_data:
            normalized_data["pcgrad_reduction"] = normalized_data["lulynx_pcgrad_reduction"]
        for key in ("hutchinson_auto_freeze", "lulynx_geometric_lock", "lulynx_ghost_replay"):
            if key in normalized_data:
                normalized_data[key] = boolish(normalized_data[key])
        if "lulynx_manifold_weight" not in normalized_data and "lulynx_ln_lambda" in normalized_data:
            normalized_data["lulynx_manifold_weight"] = normalized_data["lulynx_ln_lambda"]
        for key, default in (
            ("hutchinson_freeze_ratio", 0.5),
            ("lulynx_hutchinson_probes", 30),
            ("lulynx_manifold_weight", 0.01),
            ("lulynx_proj_dim", 128),
            ("lulynx_manifold_sparse_freq", 1),
            ("lulynx_ghost_interval", 100),
            ("lulynx_ghost_weight", 0.05),
        ):
            if key in normalized_data:
                try:
                    normalized_data[key] = float(normalized_data.get(key, default) or default)
                except (TypeError, ValueError):
                    normalized_data[key] = default
        for key in ("lulynx_hutchinson_probes", "lulynx_proj_dim", "lulynx_manifold_sparse_freq", "lulynx_ghost_interval"):
            if key in normalized_data:
                normalized_data[key] = max(int(normalized_data[key]), 1)
        if "pcgrad_enabled" in normalized_data:
            normalized_data["pcgrad_enabled"] = boolish(normalized_data["pcgrad_enabled"])
        try:
            normalized_data["pcgrad_conflict_threshold"] = float(
                normalized_data.get("pcgrad_conflict_threshold", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            normalized_data["pcgrad_conflict_threshold"] = 0.0
        normalized_data["pcgrad_reduction"] = str(
            normalized_data.get("pcgrad_reduction", "mean") or "mean"
        ).strip().lower()
        if normalized_data["pcgrad_reduction"] not in {"mean", "sum"}:
            normalized_data["pcgrad_reduction"] = "mean"
        if "ema_power" not in normalized_data and "ema_warmup_power" in normalized_data:
            normalized_data["ema_power"] = normalized_data["ema_warmup_power"]
        if "sageattn" not in normalized_data and "use_sage_attn" in normalized_data:
            normalized_data["sageattn"] = normalized_data["use_sage_attn"]
        if "flashattn" not in normalized_data and "use_flash_attn" in normalized_data:
            normalized_data["flashattn"] = normalized_data["use_flash_attn"]
        if "flashattn" in normalized_data:
            flash_enabled = str(normalized_data.get("flashattn")).strip().lower() in {
                "1", "true", "yes", "on", "enable", "enabled"
            } or normalized_data.get("flashattn") is True
            if "newbie_use_flash_attn2" not in normalized_data:
                normalized_data["newbie_use_flash_attn2"] = flash_enabled
            if flash_enabled and str(normalized_data.get("attention_backend", "") or "").strip().lower() in {"", "auto"}:
                normalized_data["attention_backend"] = "flash2"
        if "attention_backend" not in normalized_data:
            attn_mode = normalized_data.get("attn_mode")
            if attn_mode is not None:
                normalized_data["attention_backend"] = attn_mode
                logger.warning(
                    "attention_backend resolved from legacy 'attn_mode' field. "
                    "Use 'attention_backend' directly."
                )
            elif "xformers" in normalized_data and normalized_data.get("xformers", False):
                normalized_data["attention_backend"] = "xformers"
                logger.warning(
                    "attention_backend resolved from legacy 'xformers=true'. "
                    "Use attention_backend='xformers' instead."
                )
            elif normalized_data.get("sdpa", False) or normalized_data.get("use_sdpa", False):
                normalized_data["attention_backend"] = "sdpa"
                logger.warning(
                    "attention_backend resolved from legacy 'sdpa'/'use_sdpa' field. "
                    "Use attention_backend='sdpa' instead."
                )
        if "sdpa" not in normalized_data and "use_sdpa" in normalized_data:
            normalized_data["sdpa"] = normalized_data["use_sdpa"]
        if "sdpa_backend" in normalized_data and "sdpa_backend_policy" not in normalized_data:
            normalized_data["sdpa_backend_policy"] = normalized_data["sdpa_backend"]
        if "sdpa_backend_policy" in normalized_data:
            normalized_data["sdpa_backend_policy"] = str(
                normalized_data.get("sdpa_backend_policy", "") or "cutlass"
            ).strip().lower()
        if "torch_compile_backend" not in normalized_data and "compile_backend" in normalized_data:
            normalized_data["torch_compile_backend"] = normalized_data["compile_backend"]
        if "torch_compile_backend" not in normalized_data and "dynamo_backend" in normalized_data:
            normalized_data["torch_compile_backend"] = normalized_data["dynamo_backend"]
        if "torch_compile_mode" not in normalized_data and "compile_mode" in normalized_data:
            normalized_data["torch_compile_mode"] = normalized_data["compile_mode"]
        if "torch_compile_dynamic" not in normalized_data and "compile_dynamic" in normalized_data:
            normalized_data["torch_compile_dynamic"] = normalized_data["compile_dynamic"]
        if "torch_compile_fullgraph" not in normalized_data and "compile_fullgraph" in normalized_data:
            normalized_data["torch_compile_fullgraph"] = normalized_data["compile_fullgraph"]
        if "compile_runtime" in normalized_data or "compile_shape_strategy" in normalized_data or "compile_target_strategy" in normalized_data:
            runtime_resolution = resolve_runtime_optimization_payload(normalized_data)
            normalized_data.update(runtime_resolution.fields)
        if "token_string" in normalized_data and "ti_placeholder_token" not in normalized_data:
            normalized_data["ti_placeholder_token"] = normalized_data["token_string"]
        if "init_word" in normalized_data and "ti_init_token" not in normalized_data:
            normalized_data["ti_init_token"] = normalized_data["init_word"]
        if "num_vectors_per_token" in normalized_data and "ti_num_vectors" not in normalized_data:
            normalized_data["ti_num_vectors"] = normalized_data["num_vectors_per_token"]
        if isinstance(normalized_data.get("preview_groups"), str):
            raw_preview_groups = str(normalized_data.get("preview_groups") or "").strip()
            if raw_preview_groups:
                try:
                    import json
                    parsed_preview_groups = json.loads(raw_preview_groups)
                    if isinstance(parsed_preview_groups, list):
                        normalized_data["preview_groups"] = parsed_preview_groups
                except (TypeError, ValueError):
                    logger.warning("Invalid preview_groups JSON string ignored.")
                    normalized_data["preview_groups"] = []
            else:
                normalized_data["preview_groups"] = []
        if "positive_prompts" in normalized_data and "sample_prompts" not in normalized_data:
            normalized_data["sample_prompts"] = normalized_data["positive_prompts"]
        if "negative_prompts" in normalized_data and "sample_negative" not in normalized_data:
            normalized_data["sample_negative"] = normalized_data["negative_prompts"]
        if "sample_cfg_scale" in normalized_data and "sample_cfg" not in normalized_data:
            normalized_data["sample_cfg"] = normalized_data["sample_cfg_scale"]
        if "sample_every_n_steps" in normalized_data and "sample_every" not in normalized_data:
            normalized_data["sample_every"] = normalized_data["sample_every_n_steps"]
        if "anima_mode_scale" not in normalized_data and "mode_scale" in normalized_data:
            normalized_data["anima_mode_scale"] = normalized_data["mode_scale"]
        if "anima_sigmoid_scale" not in normalized_data and "sdxl_sigmoid_scale" in normalized_data:
            normalized_data["anima_sigmoid_scale"] = normalized_data["sdxl_sigmoid_scale"]
        if "eval_data_dir" not in normalized_data and "validation_data_dir" in normalized_data:
            normalized_data["eval_data_dir"] = normalized_data["validation_data_dir"]
        if "eval_data_dir" not in normalized_data and "val_data_dir" in normalized_data:
            normalized_data["eval_data_dir"] = normalized_data["val_data_dir"]
        if "eval_every_n_epochs" not in normalized_data and "validate_every_n_epochs" in normalized_data:
            normalized_data["eval_every_n_epochs"] = normalized_data["validate_every_n_epochs"]
        if "eval_every_n_steps" not in normalized_data and "validate_every_n_steps" in normalized_data:
            normalized_data["eval_every_n_steps"] = normalized_data["validate_every_n_steps"]
        for int_key in ("eval_every_n_epochs", "eval_every_n_steps", "eval_batch_size", "max_validation_steps"):
            if int_key in normalized_data:
                try:
                    normalized_data[int_key] = max(0, int(normalized_data[int_key] or 0))
                except (TypeError, ValueError):
                    normalized_data[int_key] = 0
        if "wavelet_loss_high_freq_weight" not in normalized_data and "wavelet_loss_weight" in normalized_data:
            normalized_data["wavelet_loss_high_freq_weight"] = normalized_data["wavelet_loss_weight"]
        if "prodigy_d0" in normalized_data and "opt_prodigy_d0" not in normalized_data:
            normalized_data["opt_prodigy_d0"] = normalized_data["prodigy_d0"]
        if "prodigy_d_coef" in normalized_data and "opt_prodigy_d_coef" not in normalized_data:
            normalized_data["opt_prodigy_d_coef"] = normalized_data["prodigy_d_coef"]
        if "save_last_n_epochs" in normalized_data and "checkpoint_keep_last" not in normalized_data:
            normalized_data["checkpoint_keep_last"] = normalized_data["save_last_n_epochs"]
        if "save_last_n_steps" in normalized_data and "checkpoint_keep_last" not in normalized_data:
            normalized_data["checkpoint_keep_last"] = normalized_data["save_last_n_steps"]
        if "save_n_epoch_ratio" in normalized_data and normalized_data["save_n_epoch_ratio"] in (None, ""):
            normalized_data["save_n_epoch_ratio"] = 0
        if "save_state_on_train_end" in normalized_data:
            normalized_data["save_state_on_train_end"] = boolish(normalized_data["save_state_on_train_end"])
        if "caption_tag_dropout_rate" in normalized_data and "tag_dropout_rate" not in normalized_data:
            normalized_data["tag_dropout_rate"] = normalized_data["caption_tag_dropout_rate"]
        if "caption_tag_dropout_targets" in normalized_data and normalized_data["caption_tag_dropout_targets"] is not None:
            raw_targets = normalized_data["caption_tag_dropout_targets"]
            if isinstance(raw_targets, (list, tuple)):
                normalized_data["caption_tag_dropout_targets"] = ",".join(
                    str(item).strip() for item in raw_targets if str(item).strip()
                )
            else:
                normalized_data["caption_tag_dropout_targets"] = str(raw_targets)
        if "caption_source_trigger_tokens" in normalized_data and normalized_data["caption_source_trigger_tokens"] is not None:
            raw_trigger_tokens = normalized_data["caption_source_trigger_tokens"]
            if isinstance(raw_trigger_tokens, (list, tuple)):
                normalized_data["caption_source_trigger_tokens"] = ",".join(
                    str(item).strip() for item in raw_trigger_tokens if str(item).strip()
                )
            else:
                normalized_data["caption_source_trigger_tokens"] = str(raw_trigger_tokens)
        if "enable_block_weights" in normalized_data and "bw_enable" not in normalized_data:
            normalized_data["bw_enable"] = normalized_data["enable_block_weights"]
        if "lulynx_block_weight_enabled" in normalized_data and "bw_enable" not in normalized_data:
            normalized_data["bw_enable"] = normalized_data["lulynx_block_weight_enabled"]
        if "lulynx_block_lr_zero_threshold" in normalized_data and "block_lr_zero_threshold" not in normalized_data:
            normalized_data["block_lr_zero_threshold"] = normalized_data["lulynx_block_lr_zero_threshold"]
        if "down_lr_weight" in normalized_data and "bw_in_weights" not in normalized_data:
            normalized_data["bw_in_weights"] = str(normalized_data["down_lr_weight"])
        if "lulynx_down_lr_weight" in normalized_data and "bw_in_weights" not in normalized_data:
            normalized_data["bw_in_weights"] = str(normalized_data["lulynx_down_lr_weight"])
        if "mid_lr_weight" in normalized_data and "bw_mid_weight" not in normalized_data:
            normalized_data["bw_mid_weight"] = stringify_scalar(normalized_data["mid_lr_weight"])
        if "lulynx_mid_lr_weight" in normalized_data and "bw_mid_weight" not in normalized_data:
            normalized_data["bw_mid_weight"] = stringify_scalar(normalized_data["lulynx_mid_lr_weight"])
        if "up_lr_weight" in normalized_data and "bw_out_weights" not in normalized_data:
            normalized_data["bw_out_weights"] = str(normalized_data["up_lr_weight"])
        if "lulynx_up_lr_weight" in normalized_data and "bw_out_weights" not in normalized_data:
            normalized_data["bw_out_weights"] = str(normalized_data["lulynx_up_lr_weight"])

        if normalized_data.get("schema_id") == "anima-finetune":
            requested_te = boolish(normalized_data.get("train_text_encoder"), default=False) or boolish(
                normalized_data.get("anima_full_finetune_train_text_encoder_requested"),
                default=False,
            )
            te_policy = normalize_anima_full_finetune_te_policy(
                normalized_data.get("anima_full_finetune_text_encoder_policy"),
                requested=requested_te,
            )
            normalized_data.setdefault("native_cache_mode", "cache_first")
            normalized_data.setdefault("anima_cached_training", True)
            normalized_data["anima_full_finetune_train_text_encoder_requested"] = requested_te
            normalized_data["anima_full_finetune_text_encoder_policy"] = te_policy
            normalized_data["network_train_unet_only"] = True
            normalized_data["network_train_text_encoder_only"] = False
            normalized_data.pop("train_text_encoder", None)
            normalized_data["anima_full_finetune_phase"] = "dit_only_cache_first"

        # train_text_encoder UI alias → network_train_unet_only (inverted)
        # UI semantics: train_text_encoder=True means "train both UNet and TE",
        # train_text_encoder=False means "train UNet only, NOT TE".
        # We must set network_train_unet_only (not network_train_text_encoder_only).
        # Setting network_train_text_encoder_only=True would mean "train ONLY TE" which
        # is the opposite of the user's intent when train_text_encoder=False.
        if "train_text_encoder" in normalized_data:
            train_te = boolish(normalized_data.pop("train_text_encoder"), default=True)
            normalized_data["network_train_unet_only"] = not train_te
            # Ensure the other flag is not accidentally set from stale input
            normalized_data.pop("network_train_text_encoder_only", None)

        safeguard_sources = ("safeguard_enabled", "lulynx_safeguard_enabled")
        if any(key in normalized_data for key in safeguard_sources):
            enabled = boolish(next(normalized_data[key] for key in safeguard_sources if key in normalized_data))
            normalized_data.setdefault("so_enable_nan_detection", enabled)
            normalized_data.setdefault("so_enable_loss_spike_detection", enabled)
            normalized_data.setdefault("so_enable_lr_deadlock_detection", enabled)
            normalized_data.setdefault("so_enable_auto_recovery", enabled)
        safeguard_aliases = {
            "safeguard_nan_check_interval": "so_nan_check_interval",
            "lulynx_safeguard_nan_check_interval": "so_nan_check_interval",
            "safeguard_gradient_check_interval": "so_gradient_check_interval",
            "lulynx_safeguard_gradient_check_interval": "so_gradient_check_interval",
            "safeguard_gradient_scan_mode": "so_gradient_scan_mode",
            "lulynx_safeguard_gradient_scan_mode": "so_gradient_scan_mode",
            "safeguard_max_nan_count": "so_max_nan_count",
            "lulynx_safeguard_max_nan_count": "so_max_nan_count",
            "safeguard_loss_spike_threshold": "so_loss_spike_threshold",
            "lulynx_safeguard_loss_spike_threshold": "so_loss_spike_threshold",
            "safeguard_loss_window_size": "so_loss_window_size",
            "lulynx_safeguard_loss_window_size": "so_loss_window_size",
            "safeguard_lr_reduction_factor": "so_lr_reduction_factor",
            "lulynx_safeguard_lr_reduction_factor": "so_lr_reduction_factor",
        }
        for source_key, target_key in safeguard_aliases.items():
            if source_key in normalized_data and target_key not in normalized_data:
                normalized_data[target_key] = normalized_data[source_key]
        if "safeguard_auto_reduce_lr" in normalized_data and "so_enable_auto_recovery" not in normalized_data:
            normalized_data["so_enable_auto_recovery"] = boolish(normalized_data["safeguard_auto_reduce_lr"])
        if "lulynx_safeguard_auto_reduce_lr" in normalized_data and "so_enable_auto_recovery" not in normalized_data:
            normalized_data["so_enable_auto_recovery"] = boolish(normalized_data["lulynx_safeguard_auto_reduce_lr"])

        controller_enabled = any([
            bool(normalized_data.get("auto_controller_enabled", False)),
            bool(normalized_data.get("ac_enabled", False)),
            bool(normalized_data.get("ac_enable_smart_early_stopping", False)),
            bool(normalized_data.get("ac_enable_smart_lr_decay", False)),
            bool(normalized_data.get("ac_enable_auto_lr_adjustment", False)),
            bool(normalized_data.get("ac_enable_dynamic_loss_scaling", False)),
        ])
        if "auto_controller_enabled" not in normalized_data:
            normalized_data["auto_controller_enabled"] = controller_enabled
        if "smart_early_stop" not in normalized_data and "ac_enable_smart_early_stopping" in normalized_data:
            normalized_data["smart_early_stop"] = normalized_data["ac_enable_smart_early_stopping"]
        if "smart_lr_decay" not in normalized_data:
            normalized_data["smart_lr_decay"] = bool(
                normalized_data.get("ac_enable_smart_lr_decay", False)
                or normalized_data.get("ac_enable_auto_lr_adjustment", False)
            )
        if "clip_drift_threshold" not in normalized_data and "ac_clip_drift_danger" in normalized_data:
            normalized_data["clip_drift_threshold"] = normalized_data["ac_clip_drift_danger"]
        if "stable_rank_threshold" not in normalized_data and "ac_stable_rank_collapse_threshold" in normalized_data:
            normalized_data["stable_rank_threshold"] = normalized_data["ac_stable_rank_collapse_threshold"]
        if "auto_freeze_te" not in normalized_data:
            normalized_data["auto_freeze_te"] = bool(
                normalized_data.get("ac_enable_auto_te_freeze", False)
                or (
                    controller_enabled
                    and (
                        float(normalized_data.get("ac_clip_drift_warning", 0) or 0) > 0
                        or float(normalized_data.get("ac_clip_drift_danger", 0) or 0) > 0
                    )
                )
            )

        # 3. 网络模块转换
        cls._normalize_lora_alias_values(normalized_data, boolish)

        nm = str(normalized_data.get("network_module") or "").strip().lower()
        if nm == "lora":
            normalized_data["network_module"] = "networks.lora"
        elif normalized_data.get("model_type") == "flux":
            normalized_data["flux_requested_network_module"] = nm
            normalized_data["network_module"] = normalize_flux_network_module(nm) if is_flux_network_module_supported(nm) else "networks.lora"
        elif nm in {"lycoris", "lycoris.kohya"}:
            normalized_data["network_module"] = "lycoris.locon"
        elif nm in {"networks.oft", "oft", "diag-oft", "diag_oft"}:
            normalized_data["network_module"] = "lycoris.locon"
            normalized_data["lycoris_algo"] = "diag-oft"

        # 4. 空值安全处理
        if normalized_data.get("resume_path") is None:
            normalized_data["resume_path"] = ""

        # 5. 枚举兜底
        if normalized_data.get("model_type") == "custom":
            logger.warning(
                "model_type='custom' is deprecated and will be rejected in a future release. "
                "Please set an explicit architecture (sdxl, sd15, anima, newbie, flux)."
            )
            normalized_data["model_type"] = "sdxl"
        training_type = str(normalized_data.get("training_type", "") or "").strip().lower()
        model_type = str(normalized_data.get("model_type", "") or "").strip().lower()
        if training_type.startswith("anima") and not model_type:
            normalized_data["model_type"] = "anima"
        elif training_type.startswith("newbie") and not model_type:
            normalized_data["model_type"] = "newbie"
        elif training_type.startswith("flux") and not model_type:
            normalized_data["model_type"] = "flux"

        nm = str(normalized_data.get("network_module") or "").strip().lower()
        if normalized_data.get("model_type") == "flux":
            requested_nm = str(normalized_data.get("flux_requested_network_module") or nm).strip().lower()
            normalized_data["flux_requested_network_module"] = requested_nm
            normalized_data["network_module"] = normalize_flux_network_module(requested_nm) if is_flux_network_module_supported(requested_nm) else "networks.lora"
        elif nm in {"networks.oft", "oft", "diag-oft", "diag_oft"}:
            normalized_data["network_module"] = "lycoris.locon"
            normalized_data["lycoris_algo"] = "diag-oft"
        cls._sync_native_network_module_flags(normalized_data)

        if (
            normalized_data.get("model_type") == "flux"
            and normalized_data.get("newbie_transformer_path")
            and not normalized_data.get("flux_transformer_path")
        ):
            normalized_data["flux_transformer_path"] = normalized_data.pop("newbie_transformer_path")

        # Newbie fallback: if newbie_diffusers_path is empty but
        # pretrained_model_name_or_path was set (from route alias), copy it.
        if (
            normalized_data.get("model_type") == "newbie"
            and not normalized_data.get("newbie_diffusers_path")
            and normalized_data.get("pretrained_model_name_or_path")
        ):
            normalized_data["newbie_diffusers_path"] = normalized_data["pretrained_model_name_or_path"]

        if normalized_data.get("model_type") == "newbie":
            adapter_type = str(normalized_data.get("newbie_adapter_type", "") or "").strip().lower().replace("-", "_")
            adapter_type = cls._normalize_lora_type_alias(adapter_type)
            if adapter_type:
                normalized_data["newbie_adapter_type"] = adapter_type
            if adapter_type in {"loha", "locon", "lokr", "ia3", "full", "diag-oft"}:
                normalized_data["network_module"] = "lycoris.locon"
                normalized_data["lycoris_algo"] = adapter_type
            elif adapter_type == "lora_fa":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["lora_fa_enabled"] = True
            elif adapter_type == "vera":
                normalized_data["network_module"] = "networks.vera"
                normalized_data["vera_enabled"] = True
            elif adapter_type == "tlora":
                normalized_data["network_module"] = "networks.tlora"
            elif adapter_type in {"flexrank", "flexrank_lora"}:
                normalized_data["network_module"] = "networks.flexrank_lora"
                normalized_data["flexrank_lora_enabled"] = True
            elif adapter_type == "dora":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["use_dora"] = True
                normalized_data["dora_enabled"] = True
            elif adapter_type == "lora_plus":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["lora_plus_enabled"] = True
            elif adapter_type in {"rs_lora", "rslora", "rank_stabilized_lora"}:
                normalized_data["network_module"] = "networks.lora"
                normalized_data["rs_lora_enabled"] = True
            elif adapter_type in {"hydralora", "hydra_lora"}:
                normalized_data["network_module"] = "networks.lora"
                normalized_data["hydralora_enabled"] = True
            elif adapter_type == "fera":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["fera_enabled"] = True
            normalized_data.setdefault("newbie_gemma_max_token_length", 512)
            normalized_data.setdefault("newbie_clip_max_token_length", 2048)

        # 6. Semantic Base-Tuner 自动激活
        if normalized_data.get("training_type") == "semantic-tuner":
            normalized_data["semantic_tuner_enabled"] = True

        # 7. Neutral native plumbing fallbacks
        _neutral_fallbacks = [
            ("native_secondary_model_path", "anima_qwen3_path"),
            ("native_tokenizer_path", "anima_t5_tokenizer_path"),
            ("native_attn_mode", "anima_attn_mode"),
        ]
        for neutral_key, specific_key in _neutral_fallbacks:
            neutral_val = normalized_data.get(neutral_key, "")
            specific_val = normalized_data.get(specific_key, "")
            if neutral_val and not specific_val:
                normalized_data[specific_key] = neutral_val

        if normalized_data.get("native_cache_te_to_disk") and not normalized_data.get(
            "anima_cache_text_encoder_outputs_to_disk"
        ):
            normalized_data["anima_cache_text_encoder_outputs_to_disk"] = True

        execution_core = str(normalized_data.get("execution_core", "standard") or "standard").strip().lower()
        if execution_core not in {"standard", "turbo", "auto"}:
            logger.warning("Unknown execution_core=%r; falling back to standard", execution_core)
            execution_core = "standard"
        normalized_data["execution_core"] = execution_core

        for list_key in ("turbocore_features", "turbocore_disable"):
            raw_value = normalized_data.get(list_key, [])
            if isinstance(raw_value, str):
                normalized_data[list_key] = [
                    item.strip().lower()
                    for item in raw_value.split(",")
                    if item.strip()
                ]
            elif isinstance(raw_value, (list, tuple, set, frozenset)):
                normalized_data[list_key] = [
                    str(item).strip().lower()
                    for item in raw_value
                    if str(item).strip()
                ]
            elif raw_value is None:
                normalized_data[list_key] = []
            else:
                normalized_data[list_key] = [str(raw_value).strip().lower()] if str(raw_value).strip() else []

        if "turbocore_strict" in normalized_data:
            normalized_data["turbocore_strict"] = boolish(normalized_data["turbocore_strict"])
        if "turbocore_allow_fallback" in normalized_data:
            normalized_data["turbocore_allow_fallback"] = boolish(normalized_data["turbocore_allow_fallback"], default=True)
        if "turbocore_experimental_fp8" in normalized_data:
            normalized_data["turbocore_experimental_fp8"] = boolish(normalized_data["turbocore_experimental_fp8"])
        if "turbocore_workspace_mb" in normalized_data:
            try:
                normalized_data["turbocore_workspace_mb"] = max(0, int(normalized_data["turbocore_workspace_mb"] or 0))
            except (TypeError, ValueError):
                normalized_data["turbocore_workspace_mb"] = 0
        if "turbocore_prefetch_depth" in normalized_data:
            try:
                normalized_data["turbocore_prefetch_depth"] = max(0, int(normalized_data["turbocore_prefetch_depth"] or 0))
            except (TypeError, ValueError):
                normalized_data["turbocore_prefetch_depth"] = 0
        if "turbocore_profile" in normalized_data:
            normalized_data["turbocore_profile"] = str(normalized_data["turbocore_profile"] or "basic").strip().lower() or "basic"

        if "lycoris_algo" in normalized_data:
            algo = str(normalized_data["lycoris_algo"] or "loha").strip().lower().replace("_", "-")
            if algo == "diag-oft":
                normalized_data["lycoris_algo"] = "diag-oft"
            elif algo in {"loha", "lokr", "locon", "ia3", "full", "dora"}:
                normalized_data["lycoris_algo"] = algo
            if algo == "dora":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["use_dora"] = True
                normalized_data["dora_enabled"] = True

        validator = getattr(LulynxConfig, "model_validate", None)
        if callable(validator):
            return validator(normalized_data)
        return LulynxConfig.parse_obj(normalized_data)

    @staticmethod
    def _normalize_lora_type_alias(lora_type: str) -> str:
        alias_map = {
            "lycoris_lokr": "lokr",
            "lycoris_loha": "loha",
            "lycoris_locon": "locon",
            "lycoris_ia3": "ia3",
            "lycoris_full": "full",
            "lycoris_diag_oft": "diag-oft",
            "diag_oft": "diag-oft",
            "diag-oft": "diag-oft",
            "oft": "diag-oft",
            "lora_plus": "lora_plus",
            "lora+": "lora_plus",
            "flexrank_lora": "flexrank",
            "networks.flexrank_lora": "flexrank",
            "rslora": "rs_lora",
            "rank_stabilized_lora": "rs_lora",
            "rank_stabilized": "rs_lora",
        }
        return alias_map.get(lora_type, lora_type)

    @staticmethod
    def _sync_native_network_module_flags(normalized_data: Dict[str, Any]) -> None:
        network_module = str(normalized_data.get("network_module", "") or "").strip().lower()
        if network_module == "networks.lora_fa":
            normalized_data["lora_fa_enabled"] = True
        elif network_module == "networks.vera":
            normalized_data["vera_enabled"] = True
        elif network_module == "networks.flexrank_lora":
            normalized_data["flexrank_lora_enabled"] = True

    @staticmethod
    def _normalize_adapter_init_values(normalized_data: Dict[str, Any], boolish) -> None:
        raw_strategy = normalized_data.get("adapter_init_strategy", normalized_data.get("init_lora_weights", "default"))
        strategy = str(raw_strategy or "default").strip().lower().replace("-", "_")
        aliases = {
            "": "default",
            "none": "default",
            "off": "default",
            "disabled": "default",
            "standard": "default",
            "kaiming": "default",
            "pissa_init": "pissa",
            "pissa": "pissa",
            "o_lora": "olora",
            "olora": "olora",
            "orthogonal_lora": "olora",
            "loftq": "loftq",
            "loft_q": "loftq",
            "loftq_init": "loftq",
        }
        strategy = aliases.get(strategy.replace(" ", ""), strategy)
        if strategy not in {"default", "pissa", "olora", "loftq"}:
            strategy = "default"

        pissa_requested = (
            boolish(normalized_data.get("pissa_enabled"))
            or boolish(normalized_data.get("use_pissa"))
            or boolish(normalized_data.get("pissa_init"))
        )
        if strategy == "default" and pissa_requested:
            strategy = "pissa"
        normalized_data["adapter_init_strategy"] = strategy
        normalized_data["pissa_enabled"] = strategy == "pissa"
        normalized_data["use_pissa"] = strategy == "pissa"

        try:
            normalized_data["loftq_bits"] = min(max(int(normalized_data.get("loftq_bits", 4) or 4), 2), 8)
        except (TypeError, ValueError):
            normalized_data["loftq_bits"] = 4
        loftq_quant_type = str(normalized_data.get("loftq_quant_type", "rowwise") or "rowwise").strip().lower().replace("-", "_").replace(" ", "_")
        loftq_quant_type = {
            "": "rowwise",
            "default": "rowwise",
            "uniform": "rowwise",
            "symmetric": "rowwise",
            "per_channel": "rowwise",
            "per_output": "rowwise",
            "global": "tensorwise",
            "per_tensor": "tensorwise",
        }.get(loftq_quant_type, loftq_quant_type)
        normalized_data["loftq_quant_type"] = loftq_quant_type if loftq_quant_type in {"rowwise", "tensorwise"} else "rowwise"

        svd_algo = str(
            normalized_data.get("pissa_svd_algo", normalized_data.get("pissa_method", "rsvd")) or "rsvd"
        ).strip().lower().replace("-", "_")
        svd_algo = {"svd": "full", "full_svd": "full", "lowrank": "rsvd", "randomized": "rsvd"}.get(svd_algo, svd_algo)
        normalized_data["pissa_svd_algo"] = svd_algo if svd_algo in {"rsvd", "full"} else "rsvd"

        try:
            normalized_data["pissa_oversample"] = max(0, int(normalized_data.get("pissa_oversample", 8) or 0))
        except (TypeError, ValueError):
            normalized_data["pissa_oversample"] = 8
        normalized_data["pissa_apply_conv2d"] = boolish(normalized_data.get("pissa_apply_conv2d"))

        export_mode = str(normalized_data.get("pissa_export_mode", "lora_compatible") or "lora_compatible").strip().lower().replace(" ", "_")
        export_aliases = {
            "lora无损兼容导出": "lora_compatible",
            "lora_compatible_export": "lora_compatible",
            "compatible": "lora_compatible",
            "native": "lora_compatible",
            "lora快速近似导出": "approximate",
            "fast": "approximate",
            "quick": "approximate",
        }
        export_mode = export_aliases.get(export_mode, export_mode)
        normalized_data["pissa_export_mode"] = export_mode if export_mode in {"lora_compatible", "approximate"} else "lora_compatible"

        init_export_mode = str(
            normalized_data.get("adapter_init_export_mode", normalized_data.get("init_lora_weights_export_mode", "auto")) or "auto"
        ).strip().lower().replace("-", "_").replace(" ", "_")
        init_export_aliases = {
            "": "auto",
            "default": "auto",
            "none": "raw",
            "off": "raw",
            "native": "raw",
            "training": "raw",
            "lora无损兼容导出": "lora_compatible",
            "compatible": "lora_compatible",
            "standard": "lora_compatible",
            "standard_lora": "lora_compatible",
            "lora_compatible_export": "lora_compatible",
            "lora快速近似导出": "approximate",
            "fast": "approximate",
            "quick": "approximate",
        }
        init_export_mode = init_export_aliases.get(init_export_mode, init_export_mode)
        normalized_data["adapter_init_export_mode"] = init_export_mode if init_export_mode in {"auto", "raw", "lora_compatible", "approximate"} else "auto"

    @staticmethod
    def _normalize_lora_alias_values(normalized_data: Dict[str, Any], boolish) -> None:
        lora_type = str(normalized_data.pop("lora_type", "") or "").strip().lower().replace("-", "_")
        lora_type = ConfigAdapter._normalize_lora_type_alias(lora_type)
        if lora_type:
            if lora_type in {"lora", "standard"}:
                normalized_data.setdefault("network_module", "networks.lora")
            elif lora_type == "lora_plus":
                normalized_data.setdefault("network_module", "networks.lora")
                normalized_data.setdefault("lora_plus_enabled", True)
            elif lora_type in {"rs_lora", "rslora", "rank_stabilized_lora"}:
                normalized_data.setdefault("network_module", "networks.lora")
                normalized_data.setdefault("rs_lora_enabled", True)
            elif lora_type == "dora":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["use_dora"] = True
                normalized_data["dora_enabled"] = True
            elif lora_type == "lora_fa":
                normalized_data["network_module"] = "networks.lora_fa"
                normalized_data["lora_fa_enabled"] = True
            elif lora_type == "vera":
                normalized_data["network_module"] = "networks.vera"
                normalized_data["vera_enabled"] = True
            elif lora_type == "tlora":
                normalized_data["network_module"] = "networks.tlora"
            elif lora_type in {"flexrank", "flexrank_lora"}:
                normalized_data["network_module"] = "networks.flexrank_lora"
                normalized_data["flexrank_lora_enabled"] = True
            elif lora_type in {"hydralora", "hydra_lora"}:
                normalized_data["network_module"] = "networks.lora"
                normalized_data["hydralora_enabled"] = True
            elif lora_type == "fera":
                normalized_data["network_module"] = "networks.lora"
                normalized_data["fera_enabled"] = True
            elif lora_type in {"loha", "locon", "lokr", "ia3", "full", "diag-oft"}:
                normalized_data["network_module"] = "lycoris.locon"
                normalized_data["lycoris_algo"] = lora_type

        if "lokr_rank" in normalized_data and "network_dim" not in normalized_data:
            normalized_data["network_dim"] = normalized_data["lokr_rank"]
        if "lokr_alpha" in normalized_data and "network_alpha" not in normalized_data:
            normalized_data["network_alpha"] = normalized_data["lokr_alpha"]
        if "lokr_dropout" in normalized_data and "network_dropout" not in normalized_data:
            normalized_data["network_dropout"] = normalized_data["lokr_dropout"]
        if "lokr_factor" in normalized_data and "lycoris_lokr_factor" not in normalized_data:
            normalized_data["lycoris_lokr_factor"] = normalized_data["lokr_factor"]
        if "lycoris_factor" in normalized_data and "lycoris_lokr_factor" not in normalized_data:
            normalized_data["lycoris_lokr_factor"] = normalized_data["lycoris_factor"]
        if "lycoris_preset" in normalized_data and "lycoris_presets" not in normalized_data:
            normalized_data["lycoris_presets"] = stringify_csv_list(normalized_data["lycoris_preset"])
        if "dropout" in normalized_data and "network_dropout" not in normalized_data:
            normalized_data["network_dropout"] = normalized_data["dropout"]
        if "rank_dropout" in normalized_data and "lokr_rank_dropout" not in normalized_data:
            normalized_data["lokr_rank_dropout"] = normalized_data["rank_dropout"]
        if "module_dropout" in normalized_data and "lokr_module_dropout" not in normalized_data:
            normalized_data["lokr_module_dropout"] = normalized_data["module_dropout"]
        if "full_matrix" in normalized_data and "lokr_full_matrix" not in normalized_data:
            normalized_data["lokr_full_matrix"] = normalized_data["full_matrix"]
        if "lokr_full_matrix" in normalized_data:
            normalized_data["lokr_full_matrix"] = boolish(normalized_data["lokr_full_matrix"])
        if "decompose_both" in normalized_data and "lokr_decompose_both" not in normalized_data:
            normalized_data["lokr_decompose_both"] = normalized_data["decompose_both"]
        if "lokr_decompose_both" in normalized_data:
            normalized_data["lokr_decompose_both"] = boolish(normalized_data["lokr_decompose_both"])
        if "unbalanced_factorization" in normalized_data and "lokr_unbalanced_factorization" not in normalized_data:
            normalized_data["lokr_unbalanced_factorization"] = normalized_data["unbalanced_factorization"]
        if "lokr_unbalanced_factorization" in normalized_data:
            normalized_data["lokr_unbalanced_factorization"] = boolish(normalized_data["lokr_unbalanced_factorization"])
        if "no_materialize_forward" in normalized_data and "lokr_no_materialize_forward" not in normalized_data:
            normalized_data["lokr_no_materialize_forward"] = normalized_data["no_materialize_forward"]
        if "lokr_no_materialize_forward" in normalized_data:
            normalized_data["lokr_no_materialize_forward"] = boolish(normalized_data["lokr_no_materialize_forward"])
        if "no_materialize_strategy" in normalized_data and "lokr_no_materialize_strategy" not in normalized_data:
            normalized_data["lokr_no_materialize_strategy"] = normalized_data["no_materialize_strategy"]
        if "lokr_no_materialize_strategy" in normalized_data:
            strategy = str(normalized_data["lokr_no_materialize_strategy"] or "legacy").strip().lower()
            normalized_data["lokr_no_materialize_strategy"] = strategy if strategy in {"auto", "legacy", "matmul"} else "legacy"
        if "lokr_export_mode" in normalized_data:
            mode = str(normalized_data["lokr_export_mode"]).strip().lower()
            normalized_data["lokr_export_mode"] = mode if mode in {"native", "lora_compatible"} else "native"
        if "pissa_init" in normalized_data and "pissa_enabled" not in normalized_data:
            normalized_data["pissa_enabled"] = normalized_data["pissa_init"]
        if "pissa_method" in normalized_data and "pissa_svd_algo" not in normalized_data:
            normalized_data["pissa_svd_algo"] = normalized_data["pissa_method"]
        ConfigAdapter._normalize_adapter_init_values(normalized_data, boolish)
        if "dora_wd" in normalized_data:
            normalized_data["dora_wd"] = boolish(normalized_data["dora_wd"])
            if normalized_data["dora_wd"]:
                normalized_data["use_dora"] = True
                normalized_data["dora_enabled"] = True
                normalized_data.setdefault("dora_mode", "wd")
                normalized_data.setdefault("bypass_mode", False)

        # TE dropout aliases
        for alias in ("text_encoder_dropout", "conditioning_dropout", "cfg_dropout", "drop_cond"):
            if alias in normalized_data and "te_dropout" not in normalized_data:
                normalized_data["te_dropout"] = float(normalized_data[alias])

        if boolish(normalized_data.get("sageattn")):
            normalized_data["attention_backend"] = "sageattn"
            normalized_data["anima_attn_mode"] = "sageattn"
        elif boolish(normalized_data.get("flashattn")):
            normalized_data["attention_backend"] = "flash2"
            normalized_data["anima_attn_mode"] = "flash2"
        elif boolish(normalized_data.get("mem_eff_attn")) and str(
            normalized_data.get("attention_backend", "") or ""
        ).strip().lower() in {"", "auto"}:
            normalized_data["attention_backend"] = "xformers"
            normalized_data["anima_attn_mode"] = "xformers"
