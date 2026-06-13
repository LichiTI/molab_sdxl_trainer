"""
Lulynx Trainer 主入口

完整的 LoRA 训练器
"""

import torch
import logging
import os
import re
import ast
import subprocess
import json
import random
import gc
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import time
import numpy as np
from dataclasses import dataclass

from .config import LulynxConfig, ModelArch, OptimizerType, SchedulerType
from .config_adapter import ConfigAdapter
from .safe_guard import TrainingSafeGuard, SafeGuardConfig
from .model_loader import ModelLoader, LoadedModel
from .lora_injector import LoRAInjector
from .lora_activation_recompute_policy import resolve_lora_activation_recompute
from .adapter_runtime_profile import build_adapter_runtime_profile
from .diffusers_cache_runtime_profile import build_diffusers_cache_runtime_profile
from .attention_runtime_profile import build_attention_runtime_profile
from .model_family import get_model_family
from .dataset_loader import (
    CaptionDataset,
    SDXLCacheFirstDataset,
    create_dataloader,
    create_sdxl_cache_first_dataloader,
    split_dataloader,
)
from .training_loop import TrainingLoop
from ..constants import FILENAME_MODEL_TEMPLATE, FILENAME_STATE_TEMPLATE, EXT_SAFETENSORS, EXT_PT
from .te_manager import SemanticTunerAwareTEManager
from .orchestra_controller import get_orchestra
from .compile_cache import (
    build_compile_cache_layout,
    compile_cache_cold_bucket_blocker,
    compile_cache_profile,
    compile_cache_status,
    prepare_compile_cache_environment,
)
from .compile_contract import resolve_compile_contract
from .compile_runtime_profile import build_compile_runtime_profile
from .compile_probe import evaluate_compile_probe
from .dataloader_policy import resolve_cached_dataloader_policy
from .dataloader_rebuild_runtime import (
    build_dataloader_rebuild_readiness_profile,
    rebuild_dataloader_from_plan,
)
from .multi_batch_contract import dataloader_attached_batching_contract
from .multi_batch_promotion_gate import build_lulynx_multi_batch_promotion_gate
from .training_data_pipeline_stage import (
    dataloader_attached_data_pipeline_report,
    merge_lulynx_data_pipeline_reports,
)
from .training_pipeline_execution_readiness import build_lulynx_training_pipeline_execution_readiness
from .training_step_orchestrator import build_lulynx_training_step_orchestrator_slice
from .cache_reader_decode_profile import (
    compact_cache_reader_decode_sidecar_profile,
    compact_cache_reader_training_gate_profile,
)
from .memory_runtime_profiles import (
    attach_memory_runtime_profiles_to_state,
    build_memory_runtime_profiles,
)
from .runtime_feature_snapshot import build_lulynx_trainer_runtime_features
from .bubble_runtime_controller import build_bubble_controller_report
from .bubble_runtime_closed_loop_executor import (
    mark_closed_loop_action_applied,
    mark_closed_loop_action_closed,
)
from .checkpoint_policy import resolve_checkpoint_policy
from .sdxl_lora_low_vram_profile import apply_sdxl_lora_low_vram_profile
from .staged_resolution import (
    StagedResolutionStage,
    build_staged_resolution_plan,
    stages_to_summary,
)
from ..training_event_writer import TrainingEventWriter
from .runtime_optimizations import (
    apply_full_core_compile,
    apply_per_block_compile,
    build_compile_target_profile,
    build_runtime_optimization_plan,
    build_sdpa_backend_context,
)
from .lokr_export_rules import export_lokr_state_dict
from .anima_train_norm_compat import export_anima_train_norm_state_dict
from .anima_full_finetune import (
    build_anima_grouped_param_groups,
    build_anima_full_finetune_state_dict,
    collect_trainable_param_name_map,
    is_anima_full_finetune,
    load_anima_full_finetune_state,
    prepare_anima_dit_only_full_finetune,
)
from .anima_dit_runtime_guardrails import apply_anima_dit_runtime_guardrails
from .newbie_dit_runtime_guardrails import apply_newbie_dit_runtime_guardrails
from .trainer_thermal import TrainerThermalMixin
from .trainer_optimizer_factory import TrainerOptimizerFactoryMixin
from .trainer_artifact_io import TrainerArtifactIoMixin
from .trainer_anima_cache_runtime import TrainerAnimaCacheRuntimeMixin
from .trainer_logging_runtime import TrainerLoggingRuntimeMixin
from .trainer_bubble_runtime import TrainerBubbleRuntimeMixin
from .amd_runtime import (
    apply_amd_runtime_guard,
    build_amd_banner_lines,
    build_amd_runtime_guard,
)
from .mps_runtime import (
    apply_mps_runtime_guard,
    build_mps_banner_lines,
    build_mps_runtime_guard,
)
from .newbie_readiness import NewbieReadinessReport
from .newbie_cache_contract import newbie_cache_contract_for_root


def _config_bool(config: Any, name: str, default: bool = False) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
from .anima_cache_runtime import build_anima_cache_encode_bundle
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class _CaptionTrainingInput:
    dataset: Any
    dataloader: Any
    sdxl_cache_first: bool = False
    cache_root: str = ""


class _FullFinetuneParamWrapper:
    """Minimal adapter that exposes the same interface as LoRAInjector
    for get_trainable_params() and get_param_groups(), but wraps a plain
    list of nn.Parameter objects (used in full fine-tuning and TI modes)."""

    def __init__(self, params: List[torch.nn.Parameter],
                 unet: Optional[torch.nn.Module] = None,
                 text_encoder_1: Optional[torch.nn.Module] = None,
                 text_encoder_2: Optional[torch.nn.Module] = None,
                 state_dict_getter: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
                 state_dict_loader: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None):
        self._params = list(params)
        self.injected_layers: Dict[str, Any] = {}
        self._unet = unet
        self._text_encoder_1 = text_encoder_1
        self._text_encoder_2 = text_encoder_2
        self._state_dict_getter = state_dict_getter
        self._state_dict_loader = state_dict_loader

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return self._params

    def get_residency_params(self) -> List[torch.nn.Parameter]:
        return self._params

    def get_param_groups(self, base_lr: float = 1e-4, weight_decay: float = 0.0):
        return [{"params": self._params, "lr": base_lr, "weight_decay": weight_decay}]

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the full model state dict for full-finetune / TI saves."""
        if self._state_dict_getter is not None:
            return self._state_dict_getter()

        state_dict: Dict[str, torch.Tensor] = {}
        if self._unet is not None:
            for k, v in self._unet.state_dict().items():
                state_dict[f"unet.{k}"] = v
        if self._text_encoder_1 is not None:
            for k, v in self._text_encoder_1.state_dict().items():
                state_dict[f"text_encoder_1.{k}"] = v
        if self._text_encoder_2 is not None:
            for k, v in self._text_encoder_2.state_dict().items():
                state_dict[f"text_encoder_2.{k}"] = v
        return state_dict

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Restore wrapped module state from an in-memory checkpoint."""
        if self._state_dict_loader is not None:
            self._state_dict_loader(state_dict)
            return

        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        if self._unet is not None:
            unet_sd = {k[len("unet."):]: v for k, v in state_dict.items() if k.startswith("unet.")}
            if unet_sd:
                self._unet.load_state_dict(unet_sd, strict=False)

        if self._text_encoder_1 is not None:
            te1_sd = {
                k[len("text_encoder_1."):]: v
                for k, v in state_dict.items()
                if k.startswith("text_encoder_1.")
            }
            if te1_sd:
                self._text_encoder_1.load_state_dict(te1_sd, strict=False)

        if self._text_encoder_2 is not None:
            te2_sd = {
                k[len("text_encoder_2."):]: v
                for k, v in state_dict.items()
                if k.startswith("text_encoder_2.")
            }
            if te2_sd:
                self._text_encoder_2.load_state_dict(te2_sd, strict=False)


class LulynxTrainer(TrainerThermalMixin, TrainerOptimizerFactoryMixin, TrainerArtifactIoMixin, TrainerAnimaCacheRuntimeMixin, TrainerLoggingRuntimeMixin, TrainerBubbleRuntimeMixin):
    """Lulynx 原生 LoRA 训练器"""

    def __init__(self, config: Optional[LulynxConfig] = None):
        self.config = config
        self.model: Optional[LoadedModel] = None
        self.lora_injector: Optional[LoRAInjector] = None
        self.te_manager: Optional[SemanticTunerAwareTEManager] = None
        self.training_loop: Optional[TrainingLoop] = None
        self.runtime_optimization_plan = None
        self.compile_contract_decision = None
        self.compile_cache_layout = None
        self._runtime_phase_start_time = time.perf_counter()
        self._runtime_phase_last_time = self._runtime_phase_start_time
        self._runtime_phase_timings: List[Dict[str, Any]] = []
        self._anima_full_core_original_run_blocks = None
        self._anima_full_core_compiled_run_blocks = None
        self._anima_full_core_probe_result = None
        self._per_block_compile_applied = False
        self._compile_runtime_profile: Dict[str, Any] = {}
        self._compile_cache_profile: Dict[str, Any] = {}
        self._anima_block_residency_profile: Dict[str, Any] = {}
        self._anima_block_checkpoint_profile: Dict[str, Any] = {}
        # #147: set True when the faithful native forward is active so the
        # checkpoint-policy resolution disables the incompatible checkpointing.
        self._anima_faithful_active: bool = False
        self._newbie_block_residency_profile: Dict[str, Any] = {}
        self._newbie_block_checkpoint_profile: Dict[str, Any] = {}
        self._auto_vram_enhancement_profile: Dict[str, Any] = {}
        self._low_vram_guardrail_profile: Dict[str, Any] = {}
        self._sdxl_lora_low_vram_profile: Dict[str, Any] = {}
        self._native_unet_status: Dict[str, Any] = {}
        self._native_weight_residency_profile: Dict[str, Any] = {}
        self._diffusers_cache_runtime_profile: Dict[str, Any] = {}
        self._optimizer_backend_profile: Dict[str, Any] = {}
        self._advanced_optimizer_strategy_profile: Dict[str, Any] = {}
        self._anima_full_finetune_experiments_profile: Dict[str, Any] = {}
        self._data_backend_profile: Dict[str, Any] = {}
        self._fused_projection_profile: Dict[str, Any] = {}
        self._weight_compression_profile: Dict[str, Any] = {}
        self._lora_activation_recompute_profile: Dict[str, Any] = {}
        self._adapter_runtime_profile: Dict[str, Any] = {}
        self._attention_runtime_profile: Dict[str, Any] = {}
        self._checkpoint_policy_profile: Dict[str, Any] = {}
        self._newbie_cache_first_profile: Dict[str, Any] = {}
        self._anima_cache_builder_profile: Dict[str, Any] = {}
        self._newbie_cache_rebuild_handled_in_prepare = False
        self._anima_staged_resolution_plan: List[StagedResolutionStage] = []
        self._anima_staged_resolution_active_index = -1
        self._lora_staged_resolution_plan: List[StagedResolutionStage] = []
        self._lora_staged_resolution_active_index = -1
        self._lora_staged_resolution_enabled_runtime = False
        self._lora_staged_resolution_compile_drop_last = False
        self._lora_staged_resolution_sdxl_cache_first = False
        self._anima_cache_pending = False
        self._reft_interventions = []
        self._easy_control = None
        self._ip_adapter = None
        self._easycontrol_v2_adapter = None
        self._easycontrol_v2_patch_handle = None
        self._repa_projector = None
        self._dop_instance = None
        self._adapter_cpu_residency = None
        self._attention_profile = None
        self.amd_runtime_guard = None
        self.mps_runtime_guard = None

        # 状态
        self.is_running = False
        self._should_stop = False
        self._last_loss = 0.0
        self._bubble_closed_loop_step_window: List[Dict[str, Any]] = []
        self._bubble_closed_loop_state: Dict[str, Any] = {}
        self._bubble_closed_loop_last_report: Dict[str, Any] = {}
        self._bubble_dataloader_epoch_pending: Dict[str, Any] = {}
        self._dataloader_rebuild_readiness_profile: Dict[str, Any] = {}
        self._dataloader = None

        # 回调
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[int, int, float], None]] = None
        self.on_complete: Optional[Callable[[bool], None]] = None
        self.on_step: Optional[Callable[[int, int, float, float], None]] = None  # step, epoch, loss, lr
        self.on_cull_samples: Optional[Callable[[List[str]], None]] = None
        self.on_runtime_event: Optional[Callable[[Dict[str, Any]], None]] = None

        # 高级功能
        self._dynamic_pruner = None
        self._sampler = None
        self._resource_manager = None
        self._ema_tracker = None
        self._last_vram_status = "ok"
        self._block_weight_manager = None
        self._ddp_wrapper = None  # DDPModelWrapper when multi_gpu is enabled

        # V3.0: 高级功能组件
        self._auto_controller = None  # AutoController
        self._coreset_manager = None  # CoresetManager
        self._dora_layers = None      # DoRA 层
        self._tb_writer = None
        self._tb_log_dir: Optional[Path] = None
        self._tb_flush_interval_steps = max(
            1,
            int(getattr(config, "tensorboard_flush_interval_steps", 10) or 10),
        )
        self._adaptive_step_logging_enabled = bool(getattr(config, "adaptive_step_logging_enabled", True))
        self._step_logging_interval = 1
        self._step_logging_max_interval = max(
            1,
            int(getattr(config, "adaptive_step_logging_max_interval", 64) or 64),
        )
        self._step_logging_threshold = max(
            0.0,
            float(getattr(config, "adaptive_step_logging_threshold", 0.01) or 0.01),
        )
        self._step_logging_window = max(
            1,
            int(getattr(config, "adaptive_step_logging_window", 50) or 50),
        )
        self._step_logging_profile_steps = 0
        self._step_logging_profile_total = 0.0
        self._step_logging_profile_overhead = 0.0
        self._wandb_enabled = False

        # 设备配置
        requested_device = str(getattr(config, "device", "") or "").strip().lower()
        if requested_device == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self._supports_bf16() else torch.float16

    def set_callbacks(
        self,
        on_step: Optional[Callable[[int, int, float, float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
        on_cull_samples: Optional[Callable[[List[str]], None]] = None,
        on_runtime_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        设置回调函数

        Args:
            on_step: 每步回调 (step, epoch, loss, lr)
            on_log: 日志回调 (message)
            on_complete: 完成回调 (success)
        """
        if on_step:
            self.on_step = on_step
        if on_log:
            self.on_log = on_log
        if on_complete:
            self.on_complete = on_complete
        if on_cull_samples:
            self.on_cull_samples = on_cull_samples
        if on_runtime_event:
            self.on_runtime_event = on_runtime_event

    def _emit_runtime_event(self, payload: Dict[str, Any]) -> None:
        callback = self.on_runtime_event
        if callback is None:
            return
        try:
            callback(dict(payload))
        except Exception as exc:
            logger.debug("trainer runtime event callback failed: %s", exc)

    def train(self):
        """训练入口 (start 的别名)"""
        return self.start()

    def _supports_bf16(self) -> bool:
        """检查是否支持 BF16"""
        if str(getattr(self, "device", "")).startswith("mps"):
            return False
        if self.amd_runtime_guard is not None and getattr(self.amd_runtime_guard, "is_amd", False):
            return bool(self.amd_runtime_guard.bf16_supported)
        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()

    def _log(self, message: str):
        """日志输出"""
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def _tool_cuda_cache_strategy(self) -> str:
        raw = getattr(getattr(self, "config", None), "cuda_cache_release_strategy", "oom_only")
        try:
            return TrainingLoop._normalize_cuda_cache_release_strategy(raw)
        except Exception:
            return "oom_only"

    def _maybe_release_tool_cuda_cache(
        self,
        reason: str,
        *,
        force: bool = False,
        collect_gc: bool = False,
        synchronize: bool = False,
        phase_boundary: bool = True,
    ) -> bool:
        if collect_gc:
            gc.collect()
        if not torch.cuda.is_available():
            return False
        strategy = self._tool_cuda_cache_strategy()
        if not force:
            if strategy == "aggressive":
                allowed = True
            elif strategy == "phase_boundary":
                allowed = bool(phase_boundary)
            else:
                allowed = False
            if not allowed:
                return False
        try:
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()
            return True
        except Exception as exc:
            logger.debug("[cuda-cache-tool] %s skipped: %s", reason, exc)
            return False

    def _reset_runtime_phase_timings(self) -> None:
        now = time.perf_counter()
        self._runtime_phase_start_time = now
        self._runtime_phase_last_time = now
        self._runtime_phase_timings = []

    def _mark_runtime_phase(self, label: str, *, log: bool = True) -> None:
        now = time.perf_counter()
        previous = float(getattr(self, "_runtime_phase_last_time", now) or now)
        started = float(getattr(self, "_runtime_phase_start_time", previous) or previous)
        item = {
            "label": str(label),
            "dt_seconds": round(max(now - previous, 0.0), 4),
            "total_seconds": round(max(now - started, 0.0), 4),
        }
        if torch.cuda.is_available() and str(getattr(self, "device", "")).startswith("cuda"):
            try:
                item["cuda_allocated_mb"] = round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 3)
                item["cuda_reserved_mb"] = round(float(torch.cuda.memory_reserved()) / (1024 * 1024), 3)
                item["cuda_peak_allocated_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 3)
            except Exception:
                pass
        self._runtime_phase_last_time = now
        self._runtime_phase_timings.append(item)
        if log:
            self._log(
                f"[runtime-phase] {item['label']}: "
                f"+{item['dt_seconds']:.2f}s total={item['total_seconds']:.2f}s"
            )

    def _runtime_phase_report(self) -> Dict[str, Any]:
        phases = list(getattr(self, "_runtime_phase_timings", []) or [])
        return {
            "total_seconds": phases[-1]["total_seconds"] if phases else 0.0,
            "phases": phases,
        }

    def _log_runtime_phase_summary(self, *, limit: int = 6) -> None:
        phases = list(getattr(self, "_runtime_phase_timings", []) or [])
        if not phases:
            return
        slowest = sorted(phases, key=lambda item: float(item.get("dt_seconds", 0.0) or 0.0), reverse=True)
        summary = ", ".join(
            f"{item.get('label', '?')}={float(item.get('dt_seconds', 0.0) or 0.0):.2f}s"
            for item in slowest[: max(int(limit), 1)]
        )
        self._log(
            f"[runtime-phase] setup total={float(phases[-1].get('total_seconds', 0.0) or 0.0):.2f}s; "
            f"slowest: {summary}"
        )

    def _get_resolution_pair(self) -> tuple[int, int]:
        """Normalize config resolution to a `(width, height)` pair."""
        resolution = self.config.resolution

        if isinstance(resolution, int):
            return resolution, resolution

        if isinstance(resolution, str):
            parts = [part.strip() for part in resolution.split(",") if part.strip()]
            if len(parts) == 1:
                value = int(parts[0])
                return value, value
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])

        raise ValueError(f"Unsupported resolution format: {resolution!r}")

    def _get_dataset_resolution(self) -> int:
        """Get a single base resolution value for dataset bucketing."""
        width, height = self._get_resolution_pair()
        return max(width, height)

    def _dit_block_checkpoint_recommended(self, model_arch: str, residency_mode: str) -> bool:
        """Return True when DiT residency needs block recompute to avoid activation peaks."""

        normalized = str(residency_mode or "resident").strip().lower().replace("-", "_")
        if normalized in {"", "resident", "off", "gpu", "none"}:
            return False
        if normalized not in {
            "streaming_offload",
            "streaming_cpu_offload",
            "streaming_pinned",
            "steaming",
            "steaming_offload",
            "block_cpu_pinned",
            "streaming",
            "balanced",
            "hot",
            "hotaware",
            "hot_aware",
            "hot_aware_cpu_pinned",
        }:
            return False

        try:
            width, height = self._get_resolution_pair()
        except Exception:
            width, height = 0, 0
        high_resolution = max(width, height) >= 1024 or (width * height) >= (1024 * 1024)

        token_attr = "anima_fixed_visual_tokens" if model_arch == "anima" else "newbie_fixed_visual_tokens"
        try:
            fixed_visual_tokens = int(getattr(self.config, token_attr, 0) or 0)
        except Exception:
            fixed_visual_tokens = 0
        high_visual_tokens = fixed_visual_tokens >= 4096 or (fixed_visual_tokens <= 0 and high_resolution)

        return high_resolution or high_visual_tokens

    def _anima_faithful_block_checkpoint_recommended(self) -> bool:
        """Recommend native block checkpointing whenever faithful is requested.

        The faithful native forward clears the generic ``gradient_checkpointing``
        flags (inert on the native subset), so without block recompute every DiT
        block's activations stay resident — the silent 5.x->7GB regression vs the
        legacy checkpointed trainers. An explicitly-provided
        ``anima_block_checkpointing`` value (True or False) wins over the auto
        recommendation: only requests that never mention the field get auto-on.
        """

        if not bool(getattr(self.config, "anima_faithful_forward", False)):
            return False
        fields_set = getattr(self.config, "model_fields_set", None) or set()
        return "anima_block_checkpointing" not in fields_set

    def _prepare_anima_dit_runtime_guardrails(self) -> None:
        """Apply Anima DiT checkpoint/residency knobs shared by LoRA and full FT."""

        residency_recommended = self._dit_block_checkpoint_recommended(
            "anima",
            str(getattr(self.config, "anima_block_residency", "resident") or "resident"),
        )
        faithful_recommended = self._anima_faithful_block_checkpoint_recommended()
        profile = apply_anima_dit_runtime_guardrails(
            config=self.config,
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            checkpoint_auto_recommended=residency_recommended or faithful_recommended,
            checkpoint_auto_reason=(
                "non-resident 1024/4096-token DiT paths need activation recompute"
                if residency_recommended
                else "faithful native forward keeps all block activations without recompute"
            ),
            log=self._log,
        )
        if profile.get("checkpoint_profile"):
            self._anima_block_checkpoint_profile = dict(profile["checkpoint_profile"])
        if profile.get("residency_profile"):
            self._anima_block_residency_profile = dict(profile["residency_profile"])
        self._attach_memory_runtime_profiles_to_training_loop()
        self._mark_runtime_phase("anima_block_residency_prepare", log=False)

    def _prepare_newbie_dit_runtime_guardrails(self) -> None:
        """Apply Newbie DiT checkpoint/residency knobs shared by LoRA routes."""

        profile = apply_newbie_dit_runtime_guardrails(
            config=self.config,
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            checkpoint_auto_recommended=self._dit_block_checkpoint_recommended(
                "newbie",
                str(getattr(self.config, "newbie_block_residency", "resident") or "resident"),
            ),
            log=self._log,
        )
        if profile.get("checkpoint_profile"):
            self._newbie_block_checkpoint_profile = dict(profile["checkpoint_profile"])
        if profile.get("residency_profile"):
            self._newbie_block_residency_profile = dict(profile["residency_profile"])
        self._attach_memory_runtime_profiles_to_training_loop()
        self._mark_runtime_phase("newbie_block_residency_prepare", log=False)

    # ------------------------------------------------------------------
    # Logging-runtime methods moved to trainer_logging_runtime.TrainerLoggingRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _get_sample_prompts_list(self) -> List[str]:
        prompts = getattr(self.config, "sample_prompts", [])

        if isinstance(prompts, str):
            return [line.strip() for line in prompts.splitlines() if line.strip()]

        if isinstance(prompts, list):
            return [str(prompt).strip() for prompt in prompts if str(prompt).strip()]

        return []

    def _get_preview_groups(self) -> List[Dict[str, Any]]:
        raw_groups = getattr(self.config, "preview_groups", [])
        if isinstance(raw_groups, str):
            try:
                parsed = json.loads(raw_groups) if raw_groups.strip() else []
                raw_groups = parsed if isinstance(parsed, list) else []
            except (TypeError, ValueError):
                raw_groups = []

        groups: List[Dict[str, Any]] = []
        if isinstance(raw_groups, list):
            for index, raw_group in enumerate(raw_groups):
                if not isinstance(raw_group, dict):
                    continue
                prompt = str(raw_group.get("prompt") or raw_group.get("positive_prompt") or "").strip()
                if not prompt:
                    continue
                name = str(raw_group.get("name") or f"sample{index}").strip() or f"sample{index}"
                mode = str(raw_group.get("mode") or "lora").strip().lower()
                if mode not in {"lora", "base", "fit", "overfit"}:
                    mode = "lora"
                group = {
                    "name": name,
                    "mode": mode,
                    "prompt": prompt,
                    "negative_prompt": str(raw_group.get("negative_prompt") or raw_group.get("negative") or getattr(self.config, "sample_negative", "") or ""),
                    "width": raw_group.get("width", getattr(self.config, "sample_width", 0)),
                    "height": raw_group.get("height", getattr(self.config, "sample_height", 0)),
                    "seed": raw_group.get("seed", getattr(self.config, "sample_seed", 0)),
                    "steps": raw_group.get("steps", getattr(self.config, "sample_steps", 20)),
                    "cfg": raw_group.get("cfg", raw_group.get("guidance_scale", getattr(self.config, "sample_cfg", 7.5))),
                    "lora_weight": raw_group.get("lora_weight", raw_group.get("scale", 1.0)),
                    "start_epoch": raw_group.get("start_epoch", raw_group.get("min_epoch", raw_group.get("enabled_from_epoch", 0))),
                    "start_after_epochs": raw_group.get("start_after_epochs", raw_group.get("after_epochs", 0)),
                }
                groups.append(group)

        if groups:
            return groups[:8]

        fallback_groups = []
        for index, prompt in enumerate(self._get_sample_prompts_list()):
            fallback_groups.append({
                "name": f"sample{index}",
                "mode": "lora",
                "prompt": prompt,
                "negative_prompt": getattr(self.config, "sample_negative", "") or "",
                "width": getattr(self.config, "sample_width", 0),
                "height": getattr(self.config, "sample_height", 0),
                "seed": getattr(self.config, "sample_seed", 0),
                "steps": getattr(self.config, "sample_steps", 20),
                "cfg": getattr(self.config, "sample_cfg", 7.5),
                "lora_weight": 1.0,
                "start_epoch": 0,
                "start_after_epochs": 0,
            })
        return fallback_groups[:4]

    def _preview_group_start_epoch(self, group: Dict[str, Any]) -> int:
        try:
            start_epoch = max(int(group.get("start_epoch", 0) or 0), 0)
        except (TypeError, ValueError):
            start_epoch = 0
        try:
            start_after_epochs = max(int(group.get("start_after_epochs", 0) or 0), 0)
        except (TypeError, ValueError):
            start_after_epochs = 0
        if start_after_epochs > 0:
            return max(start_epoch, start_after_epochs + 1)
        return start_epoch

    def _filter_preview_groups_for_epoch(
        self,
        groups: List[Dict[str, Any]],
        current_epoch: Optional[int],
    ) -> List[Dict[str, Any]]:
        if current_epoch is None:
            return groups
        active_groups = []
        for group in groups:
            start_epoch = self._preview_group_start_epoch(group)
            if start_epoch <= 0 or int(current_epoch) >= start_epoch:
                active_groups.append(group)
        return active_groups

    def _safe_preview_slug(self, value: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "sample")).strip("._-")
        return (slug or "sample")[:48]

    @contextmanager
    def _preview_lora_scale(self, scale: float):
        modules = []
        model = getattr(self, "model", None)
        for attr in ("unet", "text_encoder_1", "text_encoder_2", "text_encoder"):
            root_module = getattr(model, attr, None) if model is not None else None
            if root_module is None or not hasattr(root_module, "modules"):
                continue
            modules.extend([m for m in root_module.modules() if hasattr(m, "lora") and hasattr(m, "original")])
        previous = [(module, getattr(module, "_preview_lora_scale", None)) for module in modules]
        for module in modules:
            setattr(module, "_preview_lora_scale", float(scale))
        try:
            yield
        finally:
            for module, old_value in previous:
                if old_value is None:
                    try:
                        delattr(module, "_preview_lora_scale")
                    except AttributeError:
                        pass
                else:
                    setattr(module, "_preview_lora_scale", old_value)

    def _get_custom_target_modules(self) -> Optional[List[str]]:
        """Normalize custom target module text from old/new UIs."""
        raw_targets = str(getattr(self.config, "newbie_target_modules", "") or "")
        if not raw_targets:
            return None
        parts = [
            part.strip()
            for part in re.split(r"[\r\n,]+", raw_targets)
            if part and part.strip()
        ]
        return parts or None

    def _network_arg_bool(self, key: str) -> Optional[bool]:
        """Read a boolean network_args flag without depending on sd-scripts parsing."""
        raw_args = getattr(self.config, "network_args", [])
        if isinstance(raw_args, str):
            parts = [part for part in re.split(r"[\s\r\n,]+", raw_args) if part]
        elif isinstance(raw_args, (list, tuple, set, frozenset)):
            parts = [str(part) for part in raw_args]
        else:
            parts = [str(raw_args)] if raw_args is not None else []

        normalized_key = key.strip().lower().replace("-", "_")
        for raw_part in parts:
            part = str(raw_part).strip()
            if not part:
                continue
            part = part.lstrip("-").strip().lower().replace("-", "_")
            if part == normalized_key:
                return True
            if part in {f"no_{normalized_key}", f"disable_{normalized_key}"}:
                return False
            if "=" not in part:
                continue
            name, value = (piece.strip() for piece in part.split("=", 1))
            if name != normalized_key:
                continue
            if value in {"1", "true", "yes", "on", "enabled"}:
                return True
            if value in {"0", "false", "no", "off", "disabled"}:
                return False
        return None

    def _anima_train_llm_adapter_enabled(self) -> bool:
        """Return whether ordinary Anima LoRA should also target llm_adapter."""
        if self._model_arch_value() != "anima":
            return False
        network_arg = self._network_arg_bool("train_llm_adapter")
        if network_arg is not None:
            return bool(network_arg)
        if bool(getattr(self.config, "anima_train_llm_adapter", False)):
            return True
        try:
            return float(getattr(self.config, "anima_llm_adapter_lr", 0) or 0) > 0
        except Exception:
            return False

    def _get_anima_target_modules(self) -> Optional[List[str]]:
        """Return Anima DiT LoRA targets, excluding llm_adapter by default.

        The reference Anima ordinary-LoRA path keeps the LLM adapter out of the
        hot training target unless explicitly requested.  Mirroring that contract
        keeps cached conditioning meaningful and avoids an avoidable adapter
        branch in every training step.
        """
        if self._model_arch_value() != "anima":
            return None
        train_llm_adapter = self._anima_train_llm_adapter_enabled()
        try:
            from .anima_targets import get_anima_dit_targets

            targets = get_anima_dit_targets(include_llm_adapter=train_llm_adapter)
        except Exception:
            targets = [
                target for target in get_model_family("anima").unet_target_modules
                if train_llm_adapter or "llm_adapter" not in str(target)
            ]
        self._log(
            "Anima LoRA target contract: "
            f"include_llm_adapter={train_llm_adapter}, target_suffixes={len(targets)}"
        )
        return targets

    def _resolve_adapter_target_policy_selection(
        self, model_arch: str
    ) -> tuple[Optional[set], Optional[Dict[str, int]]]:
        """Resolve optional FG-LoRA target selection for the native LoRA injector.

        Returns ``(selected_module_types, per_type_rank_map)``. The default
        policy ``"all"`` — or a missing/unreadable profile, or any failure —
        returns ``(None, None)`` so the injector keeps every model-family target
        at ``network_dim``, i.e. bitwise-identical to legacy injection.
        """
        policy = str(getattr(self.config, "adapter_target_policy", "all") or "all").strip().lower()
        if policy in ("", "all"):
            return None, None
        try:
            from .adapter_target_policy_consumer import load_policy_consumer_from_config
            from .model_family import get_model_family

            consumer = load_policy_consumer_from_config(self.config)
            if consumer is None:
                self._log(
                    f"adapter_target_policy='{policy}' requested but no usable profile "
                    "(adapter_target_policy_profile_path missing/not found); using all targets."
                )
                return None, None
            available = list(get_model_family(model_arch).unet_target_modules)
            base_rank = int(getattr(self.config, "network_dim", 16) or 16)
            selected, rank_map = consumer.select_targets(available, base_rank=base_rank)
            selected_set = {str(name) for name in selected}
            rank_map = {str(key): int(value) for key, value in dict(rank_map).items()}
            self._log(
                f"adapter_target_policy='{policy}' selected {len(selected_set)}/{len(available)} "
                f"target module types (base_rank={base_rank})."
            )
            return selected_set, rank_map
        except Exception as exc:  # never block training on an experimental selector
            self._log(f"adapter_target_policy resolution failed ({exc}); using all targets.")
            return None, None

    def _resolve_fg_lora_rank_plan(
        self, model_arch: str
    ) -> tuple[str, Optional[set], Optional[Dict[str, int]]]:
        """Resolve the FG-LoRA per-layer rank plan for the native LoRA injector.

        Returns ``(injector_policy_label, selected_module_types, rank_map)`` to
        feed straight into ``LoRAInjector``. The label drives the injector's
        ``_adapter_target_policy_active`` gate (must be != "all" to consume the
        map); ``selected`` (or None = keep all) optionally prunes target types;
        ``rank_map`` carries per-key rank, keyed by full module path OR module
        type. ``fg_lora_rank_policy`` (default "uniform") picks the direction:

          * ``uniform``  -> delegate to the legacy ``adapter_target_policy`` knob;
            "all" yields ("all", None, None) == bitwise-parity no-op.
          * ``coupled_prune`` -> reuse the adapter_target_policy engine (selects
            important layers, DROPS the rest, couples rank to score; saves VRAM).
          * ``orthogonal_redistribute`` -> keep ALL target layers, only reallocate
            each layer's rank by a depth profile over the live model full-paths.

        Any failure / unusable input degrades to a parity path so training never
        blocks on this experimental selector.
        """
        policy = str(getattr(self.config, "fg_lora_rank_policy", "uniform") or "uniform").strip().lower()
        legacy_label = str(getattr(self.config, "adapter_target_policy", "all") or "all")

        if policy == "coupled_prune":
            selected, rank_map = self._resolve_adapter_target_policy_selection(model_arch)
            if selected is None and rank_map is None:
                self._log(
                    "fg_lora_rank_policy='coupled_prune' needs an adapter_target_policy "
                    "profile (adapter_target_policy_profile_path); using all targets (parity)."
                )
                return "all", None, None
            return (legacy_label if legacy_label != "all" else "fg_lora_coupled"), selected, rank_map

        if policy == "orthogonal_redistribute":
            try:
                rank_map = self._build_fg_lora_orthogonal_rank_map(model_arch)
            except Exception as exc:  # never block training on an experimental selector
                self._log(f"fg_lora_rank_policy orthogonal resolution failed ({exc}); using all targets (parity).")
                return "all", None, None
            if not rank_map:
                self._log("fg_lora_rank_policy='orthogonal_redistribute' matched no targets; using all at network_dim (parity).")
                return "all", None, None
            self._log(
                f"fg_lora_rank_policy='orthogonal_redistribute' reallocated rank over "
                f"{len(rank_map)} target layers (profile="
                f"{str(getattr(self.config, 'fg_lora_rank_profile', 'center_peak'))}, "
                f"conserve={bool(getattr(self.config, 'fg_lora_rank_conserve_budget', True))})."
            )
            return "fg_lora_orthogonal", None, rank_map

        # uniform / unknown -> legacy adapter_target_policy path (parity-preserving).
        selected, rank_map = self._resolve_adapter_target_policy_selection(model_arch)
        return legacy_label, selected, rank_map

    def _build_fg_lora_orthogonal_rank_map(self, model_arch: str) -> Dict[str, int]:
        """Build a full-path rank map that keeps all target layers (orthogonal).

        Enumerates the live unet's injectable ``nn.Linear`` paths and mirrors the
        injector's target-match + exclude rules (see the inject() call site) so the
        produced keys line up with the names the injector iterates; the injector's
        full-path-first rank lookup then turns this into true per-layer rank.
        Returns ``{}`` (caller degrades to parity) when the model/targets are absent.
        """
        import torch.nn as nn
        from .fg_lora_rank_policy import (
            FgLoraRankPolicyConfig,
            build_orthogonal_rank_map,
            select_target_full_paths,
        )
        from .model_family import get_model_family

        model = getattr(self, "model", None)
        unet = getattr(model, "unet", None) if model is not None else None
        if unet is None:
            return {}

        if model_arch == "anima":
            target_names = list(self._get_anima_target_modules() or [])
            excludes = [] if self._anima_train_llm_adapter_enabled() else ["llm_adapter"]
        elif model_arch == "newbie":
            target_names = list(self._get_custom_target_modules() or [])
            excludes = []
        else:
            target_names = list(get_model_family(model_arch).unet_target_modules)
            excludes = []
        if not target_names:
            return {}

        linear_names = [n for n, m in unet.named_modules() if isinstance(m, nn.Linear)]
        matched = select_target_full_paths(linear_names, target_names, excludes)
        if not matched:
            return {}
        base_rank = int(getattr(self.config, "network_dim", 16) or 16)
        cfg = FgLoraRankPolicyConfig(
            policy="orthogonal_redistribute",
            min_rank=int(getattr(self.config, "fg_lora_rank_min", 1) or 1),
            max_rank=int(getattr(self.config, "fg_lora_rank_max", 64) or 64),
            profile=str(getattr(self.config, "fg_lora_rank_profile", "center_peak") or "center_peak"),
            conserve_budget=bool(getattr(self.config, "fg_lora_rank_conserve_budget", True)),
        )
        return build_orthogonal_rank_map(matched, base_rank, cfg)

    def _resolve_lora_activation_recompute(self, model_arch: str) -> bool:
        """Resolve Warehouse LoRA branch activation recompute."""
        normalized_arch = str(model_arch or "").strip().lower()
        auto_default = normalized_arch in {"anima", "newbie"}
        mode = str(getattr(self.config, "lora_activation_recompute_mode", "auto") or "auto").strip().lower()

        def _truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            if isinstance(value, (int, float)):
                return value != 0
            return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

        legacy_flag = _truthy(getattr(self.config, "lora_activation_recompute", False))
        enabled = resolve_lora_activation_recompute(
            self.config,
            auto_default=auto_default,
        )
        if mode in {"on", "true", "1", "yes", "enabled"}:
            source = "explicit_mode_on"
        elif mode in {"off", "false", "0", "no", "disabled"}:
            source = "explicit_mode_off"
        elif legacy_flag:
            source = "legacy_flag"
        elif auto_default and enabled:
            source = "auto_default_native_dit"
        else:
            source = "default_off"
        self._lora_activation_recompute_profile = {
            "enabled": bool(enabled),
            "mode": mode or "auto",
            "model_arch": normalized_arch,
            "auto_default": bool(auto_default),
            "legacy_flag": bool(legacy_flag),
            "source": source,
        }
        if enabled:
            try:
                setattr(self.config, "lora_activation_recompute", True)
            except Exception:
                pass
        return enabled

    def _attach_lora_activation_recompute_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_lora_activation_recompute_profile", None)
        if profile and self.training_loop is not None and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["lora_activation_recompute"] = dict(profile)

    def _refresh_adapter_runtime_profile(self, model_arch: str = "") -> Dict[str, Any]:
        if self.config is None or self.lora_injector is None:
            return {}
        try:
            self._adapter_runtime_profile = build_adapter_runtime_profile(
                self.config,
                self.lora_injector,
                model_arch=model_arch or self._model_arch_value(),
            )
        except Exception as exc:
            self._adapter_runtime_profile = {
                "enabled": False,
                "source": "runtime_injector",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return dict(self._adapter_runtime_profile)

    def _attach_adapter_runtime_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_adapter_runtime_profile", None)
        if profile and self.training_loop is not None and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["adapter_runtime"] = dict(profile)

    def _refresh_attention_runtime_profile(
        self,
        *,
        model_arch: str = "",
        route: str = "",
        patched: int = 0,
        patch_target: str = "",
        applied: bool | None = None,
        skip_reason: str = "",
        error: str = "",
        source: str = "trainer_runtime",
    ) -> Dict[str, Any]:
        try:
            self._attention_runtime_profile = build_attention_runtime_profile(
                config=self.config,
                runtime_plan=self.runtime_optimization_plan,
                model_arch=model_arch or self._model_arch_value(),
                route=route or model_arch or self._model_arch_value(),
                profile=getattr(self, "_attention_profile", None),
                patched=patched,
                patch_target=patch_target,
                applied=applied,
                skip_reason=skip_reason,
                error=error,
                source=source,
            )
        except Exception as exc:
            self._attention_runtime_profile = {
                "source": source,
                "model_arch": str(model_arch or self._model_arch_value() or ""),
                "route": str(route or model_arch or self._model_arch_value() or ""),
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return dict(self._attention_runtime_profile)

    def _attach_attention_runtime_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_attention_runtime_profile", None)
        if profile and self.training_loop is not None and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["attention_runtime"] = dict(profile)

    def _refresh_compile_runtime_profile(
        self,
        *,
        model_arch: str = "",
        target_profile: Dict[str, Any] | None = None,
        applied: bool | None = None,
        compiled_targets: int | None = None,
        compile_kind: str = "",
        source: str = "trainer_runtime",
        skip_reason: str = "",
        error: str = "",
    ) -> Dict[str, Any]:
        try:
            self._compile_runtime_profile = build_compile_runtime_profile(
                config=self.config,
                runtime_plan=self.runtime_optimization_plan,
                compile_contract=self.compile_contract_decision,
                model_arch=model_arch or self._model_arch_value(),
                target_profile=target_profile,
                applied=applied,
                compiled_targets=compiled_targets,
                compile_kind=compile_kind,
                source=source,
                skip_reason=skip_reason,
                error=error,
            )
        except Exception as exc:
            self._compile_runtime_profile = {
                "source": source,
                "route": str(model_arch or self._model_arch_value() or ""),
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return dict(self._compile_runtime_profile)

    def _attach_compile_runtime_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_compile_runtime_profile", None)
        training_loop = getattr(self, "training_loop", None)
        if profile and training_loop is not None and hasattr(training_loop, "memory_optimization_state"):
            training_loop.memory_optimization_state["compile_runtime"] = dict(profile)

    def _attach_optimizer_profiles_to_training_loop(self) -> None:
        training_loop = getattr(self, "training_loop", None)
        if training_loop is None or not hasattr(training_loop, "memory_optimization_state"):
            return
        optimizer_profile = getattr(self, "_optimizer_backend_profile", None)
        if optimizer_profile:
            training_loop.memory_optimization_state["optimizer_backend"] = dict(optimizer_profile)
        strategy_profile = getattr(self, "_advanced_optimizer_strategy_profile", None)
        if strategy_profile:
            training_loop.memory_optimization_state["advanced_optimizer_strategy"] = dict(strategy_profile)

    def _attach_data_backend_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_data_backend_profile", None)
        training_loop = getattr(self, "training_loop", None)
        if profile and training_loop is not None and hasattr(training_loop, "memory_optimization_state"):
            training_loop.memory_optimization_state["data_backend"] = dict(profile)

    def _attach_memory_runtime_profiles_to_training_loop(self) -> Dict[str, Dict[str, Any]]:
        training_loop = getattr(self, "training_loop", None)
        if training_loop is None or not hasattr(training_loop, "memory_optimization_state"):
            return build_memory_runtime_profiles(self)
        return attach_memory_runtime_profiles_to_state(
            training_loop.memory_optimization_state,
            self,
        )

    def _attach_anima_full_finetune_experiments_to_training_loop(self) -> None:
        profile = getattr(self, "_anima_full_finetune_experiments_profile", None)
        training_loop = getattr(self, "training_loop", None)
        if profile and training_loop is not None and hasattr(training_loop, "memory_optimization_state"):
            training_loop.memory_optimization_state["anima_full_finetune_experiments"] = dict(profile)

    def get_runtime_features(self) -> Dict[str, Any]:
        return build_lulynx_trainer_runtime_features(self)

    def _has_anima_cached_training_data(self) -> bool:
        """Return True when train_data_dir has paired Anima latent/text caches."""
        if not self.config or self._model_arch_value() != "anima":
            return False
        if not bool(getattr(self.config, "anima_cached_training", True)):
            return False
        cache_dirs = self._anima_cached_training_dirs()
        if not cache_dirs:
            return False
        if self._anima_staged_resolution_enabled():
            return all(self._has_anima_cache_pair_in_dir(cache_dir) for cache_dir in cache_dirs)
        return any(self._has_anima_cache_pair_in_dir(cache_dir) for cache_dir in cache_dirs)

    def _anima_staged_resolution_enabled(self) -> bool:
        return (
            self.config is not None
            and self._model_arch_value() == "anima"
            and bool(getattr(self.config, "enable_mixed_resolution_training", False))
        )

    def _build_anima_staged_resolution_plan(self) -> List[StagedResolutionStage]:
        if not self._anima_staged_resolution_enabled():
            self._anima_staged_resolution_plan = []
            return []
        if self._anima_staged_resolution_plan:
            return self._anima_staged_resolution_plan
        ratios = {
            512: getattr(self.config, "staged_resolution_ratio_512", 0),
            768: getattr(self.config, "staged_resolution_ratio_768", 0),
            1024: getattr(self.config, "staged_resolution_ratio_1024", 0),
            1536: getattr(self.config, "staged_resolution_ratio_1536", 0),
            2048: getattr(self.config, "staged_resolution_ratio_2048", 0),
        }
        self._anima_staged_resolution_plan = build_staged_resolution_plan(
            enabled=True,
            final_resolution=getattr(self.config, "resolution", 1024),
            max_epochs=int(getattr(self.config, "max_train_epochs", 1) or 1),
            ratios=ratios,
            stage_batch_sizes=getattr(self.config, "staged_resolution_stage_batch_sizes", ""),
            data_dir=str(getattr(self.config, "train_data_dir", "") or ""),
        )
        return self._anima_staged_resolution_plan

    def _lora_staged_resolution_enabled(self, *, model_arch: str, anima_cached_training: bool, newbie_cached_training: bool) -> bool:
        if self.config is None or not bool(getattr(self.config, "enable_mixed_resolution_training", False)):
            return False
        if anima_cached_training or newbie_cached_training:
            return False
        training_type = str(getattr(self.config, "training_type", "") or "").strip().lower()
        if training_type in {"full_finetune", "textual_inversion", "dreambooth"}:
            return False
        if float(getattr(self.config, "prior_loss_weight", 0.0) or 0.0) > 0.0:
            return False
        validation_split = float(getattr(self.config, "validation_split", 0.0) or 0.0)
        eval_data_dir = str(getattr(self.config, "eval_data_dir", "") or "").strip()
        if validation_split > 0.0 and not eval_data_dir:
            return False
        return str(model_arch or "").strip().lower() not in {"anima", "newbie"}

    def _build_lora_staged_resolution_plan(
        self,
        *,
        model_arch: str,
        anima_cached_training: bool,
        newbie_cached_training: bool,
    ) -> List[StagedResolutionStage]:
        if not self._lora_staged_resolution_enabled(
            model_arch=model_arch,
            anima_cached_training=anima_cached_training,
            newbie_cached_training=newbie_cached_training,
        ):
            self._lora_staged_resolution_plan = []
            return []
        if self._lora_staged_resolution_plan:
            return self._lora_staged_resolution_plan
        ratios = {
            512: getattr(self.config, "staged_resolution_ratio_512", 0),
            768: getattr(self.config, "staged_resolution_ratio_768", 0),
            1024: getattr(self.config, "staged_resolution_ratio_1024", 0),
            1536: getattr(self.config, "staged_resolution_ratio_1536", 0),
            2048: getattr(self.config, "staged_resolution_ratio_2048", 0),
        }
        self._lora_staged_resolution_plan = build_staged_resolution_plan(
            enabled=True,
            final_resolution=getattr(self.config, "resolution", 1024),
            max_epochs=int(getattr(self.config, "max_train_epochs", 1) or 1),
            ratios=ratios,
            stage_batch_sizes=getattr(self.config, "staged_resolution_stage_batch_sizes", ""),
            data_dir="",
        )
        return self._lora_staged_resolution_plan

    def _select_lora_staged_resolution_stage(self, epoch: int) -> tuple[int, StagedResolutionStage] | tuple[int, None]:
        stages = self._lora_staged_resolution_plan
        if not stages:
            return -1, None
        selected = 0
        for index, stage in enumerate(stages):
            if epoch >= int(stage.start_epoch):
                selected = index
        return selected, stages[selected]

    def _estimate_staged_epoch_limited_steps(
        self,
        *,
        stages: List[StagedResolutionStage],
        sample_count: int,
        grad_accum: int,
        drop_last: bool,
        total_epochs: int,
    ) -> int:
        if not stages or sample_count <= 0:
            return 0
        staged_epoch_limited_steps = 0
        total_epochs = max(int(total_epochs or 1), 1)
        grad_accum = max(int(grad_accum or 1), 1)
        for index, stage in enumerate(stages):
            next_start = (
                stages[index + 1].start_epoch
                if index + 1 < len(stages)
                else total_epochs
            )
            epoch_count = max(int(next_start) - int(stage.start_epoch), 0)
            stage_batch = max(int(stage.batch_size or getattr(self.config, "batch_size", 1) or 1), 1)
            stage_loader_batches = sample_count // stage_batch if drop_last else (sample_count + stage_batch - 1) // stage_batch
            stage_steps = (max(stage_loader_batches, 1) + grad_accum - 1) // grad_accum
            staged_epoch_limited_steps += max(epoch_count, 0) * max(stage_steps, 1)
        return max(staged_epoch_limited_steps, 0)

    def _anima_cached_training_dirs(self) -> List[Path]:
        data_dir = Path(str(getattr(self.config, "train_data_dir", "") or ""))
        if not data_dir.is_dir():
            return []
        if self._anima_staged_resolution_enabled():
            stages = self._build_anima_staged_resolution_plan()
            return [Path(stage.cache_dir) for stage in stages if stage.cache_dir]
        return [data_dir]

    def _has_anima_cache_pair_in_dir(self, data_dir: Path) -> bool:
        if not data_dir.is_dir():
            return False
        for text_suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
            for text_path in data_dir.rglob(f"*{text_suffix}"):
                stem = text_path.name[: -len(text_suffix)]
                if any(
                    next(data_dir.rglob(f"{stem}_*_anima{latent_suffix}"), None) is not None
                    for latent_suffix in (".npz", ".safetensors", ".pt")
                ):
                    return True
        return False

    def _infer_anima_cached_text_tokens(self, *, sample_limit: int = 256) -> int:
        """Infer a static text-token pad from cached Anima prompt embeddings."""
        cache_dirs = self._anima_cached_training_dirs()
        if not cache_dirs:
            return 0

        max_tokens = 0
        inspected = 0
        for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
            for data_dir in cache_dirs:
                for path in sorted(data_dir.rglob(f"*{suffix}")):
                    if inspected >= sample_limit:
                        return max_tokens
                    inspected += 1
                    try:
                        if path.suffix.lower() == ".npz":
                            with np.load(str(path), mmap_mode="r") as data:
                                if "prompt_embeds" in data.files:
                                    max_tokens = max(max_tokens, int(data["prompt_embeds"].shape[0]))
                        elif path.suffix.lower() == ".safetensors":
                            from safetensors import safe_open
                            with safe_open(str(path), framework="pt", device="cpu") as data:
                                if "prompt_embeds" in data.keys():
                                    max_tokens = max(max_tokens, int(data.get_slice("prompt_embeds").get_shape()[0]))
                        elif path.suffix.lower() == ".pt":
                            payload = safe_torch_load(str(path), map_location="cpu")
                            if isinstance(payload, dict) and "prompt_embeds" in payload:
                                max_tokens = max(max_tokens, int(payload["prompt_embeds"].shape[0]))
                    except Exception as exc:
                        logger.debug("Failed to inspect Anima text cache token count from %s: %s", path, exc)
        return max_tokens

    def _has_newbie_cached_training_data(self) -> bool:
        """Return True when train_data_dir has valid Newbie cache-first artifacts."""
        if not self.config or self._model_arch_value() != "newbie":
            return False
        data_dir = Path(str(getattr(self.config, "train_data_dir", "") or ""))
        if not data_dir.is_dir():
            return False
        contract = newbie_cache_contract_for_root(data_dir, recursive=True)
        return bool(contract.get("ok"))

    def _resolve_cache_policy_report(self):
        """Resolve manifest/cache policy for the active native family."""
        if not self.config:
            return None
        model_arch = self._model_arch_value()
        if model_arch not in {"anima", "newbie"}:
            return None
        data_dir = str(getattr(self.config, "train_data_dir", "") or "")
        if not data_dir:
            return None
        try:
            from .cache_policy import resolve_cache_policy
        except ImportError:
            from cache_policy import resolve_cache_policy

        mode = str(getattr(self.config, "native_cache_mode", "") or "")
        return resolve_cache_policy(
            data_dir,
            family=model_arch,
            mode=mode,
            trust_cache=bool(getattr(self.config, "trust_cache", False)),
            rebuild_requested=bool(getattr(self.config, "newbie_rebuild_cache", False))
            if model_arch == "newbie"
            else mode.strip().lower().replace("-", "_") == "rebuild_cache",
            force_cache_only=bool(getattr(self.config, "newbie_force_cache_only", False))
            if model_arch == "newbie"
            else mode.strip().lower().replace("-", "_") == "force_cache_only",
        )

    def _log_cache_policy_report(self, context: str = "cache policy") -> None:
        """Log cache manifest policy results without hard-failing legacy caches."""
        try:
            report = self._resolve_cache_policy_report()
        except Exception as exc:
            self._log(f"{context}: cache policy probe failed: {type(exc).__name__}: {exc}")
            return
        if report is None:
            return
        self._log(
            f"{context}: family={report.family}, samples={report.cache_sample_count}, "
            f"cache_files={report.cache_file_count}, manifest={'ok' if report.manifest_ok else 'missing/warn'}, "
            f"ready={getattr(report, 'cache_ready', report.can_use_cache)}, "
            f"ready_samples={getattr(report, 'ready_cache_sample_count', report.cache_sample_count)}, "
            f"ready_files={getattr(report, 'ready_cache_file_count', report.cache_file_count)}, "
            f"can_use_cache={report.can_use_cache}, should_rebuild={report.should_rebuild}"
        )
        contract_reasons = tuple(getattr(report, "cache_contract_reasons", ()) or ())
        if contract_reasons:
            self._log(f"{context} cache contract reasons: {', '.join(contract_reasons)}")
        for warning in report.warnings:
            self._log(f"{context} warning: {warning}")
        for error in report.errors:
            self._log(f"{context} error: {error}")
        for note in report.notes:
            self._log(f"{context}: {note}")

    def _log_newbie_cache_parity_report(self, context: str = "newbie cache parity") -> None:
        """Log Newbie-specific cached tensor contract validation."""
        if self._model_arch_value() != "newbie" or not self._has_newbie_cached_training_data():
            return
        data_dir = str(getattr(self.config, "train_data_dir", "") or "").strip()
        if not data_dir:
            return
        try:
            from .newbie_cache_parity import validate_newbie_cache_parity

            report = validate_newbie_cache_parity(
                data_dir,
                expected_gemma3_prompt=str(getattr(self.config, "newbie_gemma3_prompt", "") or ""),
                require_manifest=False,
            )
        except Exception as exc:
            self._log(f"{context}: probe failed: {type(exc).__name__}: {exc}")
            return

        self._log(
            f"{context}: ok={report.ok}, samples={report.sample_count}, "
            f"cache_files={report.cache_file_count}, manifest={report.manifest_path.name}"
        )
        for error in report.errors:
            self._log(f"{context} error: {error}")
        for warning in report.warnings:
            self._log(f"{context} warning: {warning}")
        for note in report.notes:
            self._log(f"{context}: {note}")

    def _resolve_data_backend_profile(
        self,
        *,
        model_arch: str,
        anima_cached_training: bool,
        newbie_cached_training: bool,
    ) -> None:
        self._data_backend_profile = {}
        try:
            from .data_backend_resolver import resolve_data_backend
        except Exception as exc:
            self._log(f"Data backend profile unavailable: {type(exc).__name__}: {exc}")
            return

        try:
            decision = resolve_data_backend(
                getattr(self.config, "data_backend", "auto"),
                data_dir=getattr(self.config, "train_data_dir", ""),
            )
        except Exception as exc:
            self._log(f"Data backend probe failed: {type(exc).__name__}: {exc}")
            return

        profile = decision.as_dict()
        requested_backend = str(profile.get("requested_backend") or "auto")
        discovered_backend = str(profile.get("resolved_backend") or "caption")
        if anima_cached_training:
            effective_training_backend = "anima_cached"
        elif newbie_cached_training:
            effective_training_backend = "newbie_cached"
        else:
            effective_training_backend = "caption"

        profile_only_reason = ""
        if effective_training_backend != "caption" and requested_backend not in {"", "auto", "caption", "raw"}:
            profile_only_reason = (
                f"{effective_training_backend} route does not consume data_backend yet; "
                "keeping the existing native dataset path."
            )
        elif effective_training_backend == "caption" and discovered_backend != "caption":
            profile_only_reason = (
                "Current trainer will try an explicit materialized WebDataset integration "
                "only when the non-cache CaptionDataset route is selected."
            )

        profile["route"] = str(model_arch or "")
        profile["effective_training_backend"] = effective_training_backend
        profile["profile_only"] = bool(profile_only_reason)
        if profile_only_reason:
            profile["profile_only_reason"] = profile_only_reason

        self._data_backend_profile = profile
        self._attach_data_backend_profile_to_training_loop()
        self._log(
            "Data backend: "
            f"requested={requested_backend}, discovered={discovered_backend}, training={effective_training_backend}"
            + (f", fallback={profile.get('fallback_reason')}" if profile.get("fallback_reason") else "")
            + (f", note={profile_only_reason}" if profile_only_reason else "")
        )
        for warning in profile.get("warnings") or []:
            self._log(f"[data-backend][warn] {warning}")

    def _capture_cache_reader_decode_sidecar_profile(
        self,
        dataloader: Any,
        *,
        route: str,
        source: str = "train",
    ) -> Dict[str, Any]:
        report = getattr(dataloader, "native_cache_reader_decode_shadow_adapter", None)
        profile = compact_cache_reader_decode_sidecar_profile(
            report,
            route=route,
            source=source,
        )
        if not profile:
            return {}

        data_profile = dict(getattr(self, "_data_backend_profile", {}) or {})
        native_profiles = dict(data_profile.get("native_cache_reader") or {})
        native_profiles["decode_sidecar"] = profile
        data_profile["native_cache_reader"] = native_profiles
        self._data_backend_profile = data_profile
        self._attach_data_backend_profile_to_training_loop()
        self._log(
            "Native cache reader decode sidecar: "
            f"route={profile.get('route', route)}, "
            f"ok={bool(profile.get('ok', False))}, "
            f"tensors={int(profile.get('tensor_decode_count', 0) or 0)}, "
            f"bytes={int(profile.get('data_payload_bytes_read', 0) or 0)}, "
            "training_path=false"
        )
        return profile

    def _capture_cache_reader_training_gate_profile(
        self,
        dataloader: Any,
        *,
        route: str,
        source: str = "train",
    ) -> Dict[str, Any]:
        report = getattr(dataloader, "native_cache_reader_training_gate", None)
        profile = compact_cache_reader_training_gate_profile(
            report,
            route=route,
            source=source,
        )
        if not profile:
            return {}

        data_profile = dict(getattr(self, "_data_backend_profile", {}) or {})
        native_profiles = dict(data_profile.get("native_cache_reader") or {})
        native_profiles["training_gate"] = profile
        data_profile["native_cache_reader"] = native_profiles
        self._data_backend_profile = data_profile
        self._attach_data_backend_profile_to_training_loop()
        self._log(
            "Native cache reader training gate: "
            f"route={profile.get('route', route)}, "
            f"raw={bool(profile.get('parity_guard_passed', False))}, "
            f"batch={bool(profile.get('batch_parity_guard_passed', False))}, "
            f"allowed={bool(profile.get('training_experimental_allowed', False))}, "
            "training_path=false"
        )
        return profile

    def _attach_easycontrol_v2_dataset_profile(self, dataset: Any) -> None:
        if not bool(getattr(self.config, "easycontrol_v2_enabled", False)):
            return
        samples = list(getattr(dataset, "samples", []) or [])
        try:
            from .easycontrol_v2_contract import (
                audit_easycontrol_v2_sidecars,
                build_easycontrol_v2_task_spec_from_config,
            )

            spec = build_easycontrol_v2_task_spec_from_config(self.config)
            target_paths = [getattr(sample, "image_path", "") for sample in samples]
            target_paths = [path for path in target_paths if path]
            audit = audit_easycontrol_v2_sidecars(target_paths, spec, check_exists=True)
            missing = list(audit.missing_required)
            summary = {
                "enabled": True,
                "task_id": spec.task_id,
                "control_kind": spec.control_kind,
                "target_family": spec.target_family,
                "sample_count": len(target_paths),
                "ready": audit.ready,
                "missing_required_count": len(missing),
                "missing_required_preview": missing[:20],
                "dataset_loader_wired": True,
                "training_step_consumption": False,
            }
            profile = dict(getattr(self, "_data_backend_profile", {}) or {})
            profile["easycontrol_v2"] = summary
            self._data_backend_profile = profile
            self._attach_data_backend_profile_to_training_loop()
            if audit.ready:
                self._log(
                    "EasyControl v2 sidecar audit passed: "
                    f"task={spec.task_id}, samples={len(target_paths)}, training_step_consumption=false"
                )
            else:
                self._log(
                    "EasyControl v2 sidecar audit found missing inputs: "
                    f"task={spec.task_id}, missing={len(missing)}, training_step_consumption=false"
                )
        except Exception as exc:
            profile = dict(getattr(self, "_data_backend_profile", {}) or {})
            profile["easycontrol_v2"] = {
                "enabled": True,
                "ready": False,
                "dataset_loader_wired": True,
                "training_step_consumption": False,
                "audit_error": f"{type(exc).__name__}: {exc}",
            }
            self._data_backend_profile = profile
            self._attach_data_backend_profile_to_training_loop()
            self._log(f"EasyControl v2 sidecar audit failed: {type(exc).__name__}: {exc}")

    def _build_caption_dataset(self, data_dir: str):
        from .dataset_loader import CaptionDataset

        kwargs = dict(
            resolution=self._get_dataset_resolution(),
            caption_extension=self.config.caption_extension,
            enable_bucket=self.config.enable_bucket,
            min_bucket_reso=self.config.min_bucket_reso,
            max_bucket_reso=self.config.max_bucket_reso,
            bucket_reso_steps=self.config.bucket_reso_steps,
            bucket_selection_mode=getattr(self.config, "bucket_selection_mode", "aspect"),
            bucket_custom_resos=getattr(self.config, "bucket_custom_resos", ""),
            shuffle_caption=self.config.shuffle_caption,
            shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
            keep_tokens=self.config.keep_tokens,
            keep_tokens_separator=getattr(self.config, "keep_tokens_separator", ""),
            caption_dropout_rate=getattr(self.config, "caption_dropout_rate", 0.0),
            caption_dropout_every_n_epochs=getattr(self.config, "caption_dropout_every_n_epochs", 0),
            tag_dropout_rate=getattr(self.config, "tag_dropout_rate", 0.0),
            caption_tag_dropout_targets=getattr(self.config, "caption_tag_dropout_targets", ""),
            caption_tag_dropout_target_mode=getattr(self.config, "caption_tag_dropout_target_mode", "drop_all"),
            caption_tag_dropout_target_count=getattr(self.config, "caption_tag_dropout_target_count", 1),
            token_warmup_min=getattr(self.config, "token_warmup_min", 0),
            token_warmup_max=getattr(self.config, "keep_tokens", 0),
            token_warmup_steps=getattr(self.config, "token_warmup_steps", 0),
            weighted_captions=getattr(self.config, "weighted_captions", False),
            masked_loss=getattr(self.config, "masked_loss", False),
            alpha_mask=getattr(self.config, "alpha_mask", False),
            caption_length_bucket_size=getattr(self, "_newbie_caption_bucket_size", 0) if self._model_arch_value() == "newbie" else 0,
            albumentations_enabled=bool(getattr(self.config, "albumentations_enabled", False)),
            albumentations_pipeline=str(getattr(self.config, "albumentations_pipeline", "") or ""),
            albumentations_mask_replay=bool(getattr(self.config, "albumentations_mask_replay", True)),
            dual_caption_enabled=bool(getattr(self.config, "dual_caption_enabled", False)),
            dual_caption_short_key=str(getattr(self.config, "dual_caption_short_key", "short") or "short"),
            dual_caption_long_key=str(getattr(self.config, "dual_caption_long_key", "long") or "long"),
            caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
            caption_source_nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
            caption_source_tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
            caption_source_trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
            caption_source_empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
            caption_source_trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
            image_decode_backend=getattr(self.config, "image_decode_backend", "pil"),
            image_decode_cache_size=getattr(self.config, "image_decode_cache_size", 0),
            easycontrol_v2_enabled=bool(getattr(self.config, "easycontrol_v2_enabled", False)),
            easycontrol_v2_task_id=str(getattr(self.config, "easycontrol_v2_task_id", "generic") or "generic"),
            easycontrol_v2_control_kind=str(
                getattr(self.config, "easycontrol_v2_control_kind", "reference_latent") or "reference_latent"
            ),
            easycontrol_v2_target_family=str(getattr(self.config, "easycontrol_v2_target_family", "") or self._model_arch_value()),
            easycontrol_v2_cond_cache_dir=str(getattr(self.config, "easycontrol_v2_cond_cache_dir", "") or ""),
            easycontrol_v2_text_cache_dir=str(getattr(self.config, "easycontrol_v2_text_cache_dir", "") or ""),
            easycontrol_v2_control_image_dir=str(getattr(self.config, "easycontrol_v2_control_image_dir", "") or ""),
            easycontrol_v2_control_suffix=str(getattr(self.config, "easycontrol_v2_control_suffix", "") or ""),
            easycontrol_v2_drop_p=float(getattr(self.config, "easycontrol_v2_drop_p", 0.1) or 0.0),
            easycontrol_v2_cond_noise_max=float(getattr(self.config, "easycontrol_v2_cond_noise_max", 0.0) or 0.0),
            easycontrol_v2_scale=float(getattr(self.config, "easycontrol_v2_scale", 1.0) or 0.0),
            easycontrol_v2_match_target_bucket=bool(getattr(self.config, "easycontrol_v2_match_target_bucket", False)),
        )
        profile = getattr(self, "_data_backend_profile", {}) or {}
        if str(profile.get("requested_backend") or "") == "webdataset" and str(profile.get("resolved_backend") or "") == "webdataset":
            try:
                from .webdataset_materialized_dataset import MaterializedWebDataset

                dataset = MaterializedWebDataset(source_data_dir=data_dir, **kwargs)
                summary = dict(getattr(dataset, "webdataset_materialization_summary", {}) or {})
                profile["training_integration"] = "materialized_captiondataset"
                profile["profile_only"] = False
                profile.pop("profile_only_reason", None)
                profile["materialization"] = summary
                self._data_backend_profile = profile
                self._attach_data_backend_profile_to_training_loop()
                self._log(
                    "WebDataset materialized training dataset: "
                    f"images={summary.get('image_count', len(dataset))}, shards={summary.get('shard_count', 0)}"
                )
                return dataset
            except Exception as exc:
                warnings = list(profile.get("warnings") or [])
                reason = f"materialized WebDataset integration failed: {type(exc).__name__}: {exc}"
                warnings.append(reason)
                profile["warnings"] = warnings
                profile["fallback_reason"] = profile.get("fallback_reason") or reason
                profile["resolved_backend"] = "caption"
                profile["training_integration"] = "caption_fallback"
                self._data_backend_profile = profile
                self._attach_data_backend_profile_to_training_loop()
                self._log(f"[data-backend][warn] {reason}; falling back to CaptionDataset")
        dataset = CaptionDataset(data_dir=data_dir, **kwargs)
        self._attach_easycontrol_v2_dataset_profile(dataset)
        return dataset

    def _sdxl_cache_first_blockers(self) -> list[str]:
        blockers: list[str] = []
        if bool(getattr(self.config, "color_aug", False)):
            blockers.append("color_aug")
        if bool(getattr(self.config, "flip_aug", False)):
            blockers.append("flip_aug")
        if bool(getattr(self.config, "random_crop", False)):
            blockers.append("random_crop")
        if bool(getattr(self.config, "albumentations_enabled", False)):
            blockers.append("albumentations")
        return blockers

    def _should_use_sdxl_cache_first_dataset(self, *, model_arch: str, dataset: Any) -> bool:
        return (
            str(model_arch or "").strip().lower() in {"sdxl", "sd15"}
            and isinstance(dataset, CaptionDataset)
            and bool(getattr(self.config, "cache_latents", False))
            and bool(getattr(self.config, "cache_text_encoder_outputs", False))
            and not bool(getattr(self.config, "train_text_encoder", False))
        )

    def _refresh_diffusers_cache_runtime_profile(
        self,
        *,
        model_arch: str = "",
        cache_first: bool | None = None,
        cache_root: str | None = None,
        component_cpu_residency: bool | None = None,
        text_cache_forced_unet_only: bool | None = None,
        text_cache_disabled_reason: str | None = None,
        blockers: list[str] | tuple[str, ...] | None = None,
    ) -> Dict[str, Any]:
        if self.config is None:
            return {}
        arch = str(model_arch or self._model_arch_value()).strip().lower()
        if arch not in {"sdxl", "sd15"}:
            return {}
        previous = dict(getattr(self, "_diffusers_cache_runtime_profile", {}) or {})
        try:
            profile = build_diffusers_cache_runtime_profile(
                self.config,
                model_arch=arch,
                cache_first=bool(previous.get("cache_first", False) if cache_first is None else cache_first),
                cache_root=str(previous.get("cache_root", "") if cache_root is None else cache_root),
                component_cpu_residency=bool(
                    previous.get("component_cpu_residency", False)
                    if component_cpu_residency is None
                    else component_cpu_residency
                ),
                text_cache_forced_unet_only=bool(
                    previous.get("text_cache_forced_unet_only", False)
                    if text_cache_forced_unet_only is None
                    else text_cache_forced_unet_only
                ),
                text_cache_disabled_reason=str(
                    previous.get("text_cache_disabled_reason", "")
                    if text_cache_disabled_reason is None
                    else text_cache_disabled_reason
                ),
                blockers=blockers if blockers is not None else previous.get("blockers", []),
                low_vram_profile=getattr(self, "_sdxl_lora_low_vram_profile", None),
            )
        except Exception as exc:
            profile = {
                "enabled": False,
                "source": "diffusers_unet_runtime",
                "model_arch": arch,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._diffusers_cache_runtime_profile = profile
        return dict(profile)

    def _attach_diffusers_cache_runtime_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_diffusers_cache_runtime_profile", None)
        if profile and self.training_loop is not None and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["diffusers_cache_runtime"] = dict(profile)

    def _wrap_sdxl_cache_first_dataset(self, dataset: CaptionDataset) -> tuple[Any, bool, str]:
        model_arch = str(self._model_arch_value() or "sdxl").strip().lower()
        blockers = self._sdxl_cache_first_blockers()
        if blockers:
            self._log(
                f"{model_arch.upper()} cache-first disabled because image-space augmentation is active: "
                + ", ".join(blockers)
            )
            self._refresh_diffusers_cache_runtime_profile(
                model_arch=model_arch,
                cache_first=False,
                cache_root="",
                blockers=blockers,
            )
            return dataset, False, ""

        model_id_source = "|".join(
            [
                str(getattr(self.config, "base_model_path", "") or ""),
                str(getattr(self.config, "vae_path", "") or ""),
                str(getattr(self.config, "mixed_precision", "") or ""),
                str(getattr(self.config, "max_token_length", "") or ""),
            ]
        )
        model_id = hashlib.sha256(model_id_source.encode("utf-8")).hexdigest()[:16]
        cache_root = Path(self.config.train_data_dir) / ".lulynx_cache" / model_arch / model_id
        wrapped = SDXLCacheFirstDataset(
            dataset,
            vae=self.model.vae,
            text_encoder_1=self.model.text_encoder_1,
            text_encoder_2=self.model.text_encoder_2,
            tokenizer_1=self.model.tokenizer_1,
            tokenizer_2=self.model.tokenizer_2,
            device=self.device,
            dtype=self.dtype,
            cache_dir=str(cache_root),
            model_arch=model_arch,
            model_id=model_id,
            cache_latents=True,
            cache_text_encoder_outputs=True,
            latent_disk_format=str(getattr(self.config, "latent_cache_disk_format", "safetensors") or "safetensors"),
            latent_disk_dtype=self._resolve_cache_disk_dtype(getattr(self.config, "latent_cache_disk_dtype", "float16")),
            text_disk_format=str(getattr(self.config, "text_encoder_outputs_cache_disk_format", "safetensors") or "safetensors"),
            text_disk_dtype=self._resolve_cache_disk_dtype(getattr(self.config, "text_encoder_outputs_cache_disk_dtype", "float16")),
            keep_vae_on_cpu=self._should_use_sdxl_component_cpu_residency(),
            keep_text_encoders_on_cpu=self._should_use_sdxl_component_cpu_residency(),
            use_model_to_condition=bool(getattr(self.config, "model_to_condition_enabled", True)),
        )
        self._log(
            f"{model_arch.upper()} cache-first dataset enabled: "
            f"{len(wrapped)} samples, cache={cache_root}, "
            f"model_to_condition={bool(getattr(self.config, 'model_to_condition_enabled', True))}"
        )
        self._refresh_diffusers_cache_runtime_profile(
            model_arch=model_arch,
            cache_first=True,
            cache_root=str(cache_root),
            component_cpu_residency=self._should_use_sdxl_component_cpu_residency(),
            blockers=[],
        )
        return wrapped, True, str(cache_root)

    def _create_caption_training_input(
        self,
        *,
        data_dir: str,
        model_arch: str,
        batch_size: int,
        drop_last: bool,
    ) -> _CaptionTrainingInput:
        dataset = self._build_caption_dataset(data_dir)
        sdxl_cache_first = False
        cache_root = ""
        if self._should_use_sdxl_cache_first_dataset(model_arch=model_arch, dataset=dataset):
            dataset, sdxl_cache_first, cache_root = self._wrap_sdxl_cache_first_dataset(dataset)

        if sdxl_cache_first:
            self._log(
                f"{str(model_arch or '').upper()} cache-first DataLoader: "
                "num_workers=0 because model-backed cache generation runs in-process."
            )
            dataloader = create_sdxl_cache_first_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=getattr(self.config, "pin_memory", True),
                drop_last=drop_last,
            )
        else:
            dataloader = create_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=getattr(self.config, "dataloader_num_workers", 0),
                pin_memory=getattr(self.config, "pin_memory", True),
                prefetch_factor=getattr(self.config, "prefetch_factor", 2),
                persistent_workers=bool(getattr(self.config, "persistent_data_loader_workers", False)),
                drop_last=drop_last,
            )
        return _CaptionTrainingInput(
            dataset=dataset,
            dataloader=dataloader,
            sdxl_cache_first=sdxl_cache_first,
            cache_root=cache_root,
        )

    def _set_lora_stage_resolution(self, resolution: int) -> None:
        resolution = int(resolution or 0)
        if resolution <= 0:
            return
        old_resolution = getattr(self.config, "resolution", 0)
        try:
            old_value = self._get_dataset_resolution()
        except Exception:
            old_value = 0
        self.config.resolution = resolution
        if old_value != resolution:
            self._log(f"Staged resolution changed from {old_resolution} to {resolution}")

    def _maybe_wrap_lora_stage_ddp(self, dataloader: Any, dataset: Any) -> Any:
        if self._ddp_wrapper is None:
            return dataloader
        try:
            from .distributed import wrap_dataloader_for_ddp

            dataloader = wrap_dataloader_for_ddp(
                dataloader,
                dataset,
                shuffle=True,
                seed=int(getattr(self.config, "seed", 42) or 42),
            )
            self._ddp_wrapper._dataloader = dataloader
            self._ddp_wrapper._ddp_sampler = getattr(dataloader, "sampler", None)
        except Exception as exc:
            self._log(f"LoRA staged resolution DDP dataloader refresh skipped: {exc}")
        return dataloader

    def _maybe_switch_lora_staged_resolution_dataset(self, *, dataloader: Any, epoch: int) -> Any:
        if not self._lora_staged_resolution_enabled_runtime:
            return dataloader
        index, stage = self._select_lora_staged_resolution_stage(epoch)
        if stage is None or index == self._lora_staged_resolution_active_index:
            return dataloader

        self._set_lora_stage_resolution(stage.resolution)
        batch_size = int(stage.batch_size or getattr(self.config, "batch_size", 1) or 1)
        training_input = self._create_caption_training_input(
            data_dir=self.config.train_data_dir,
            model_arch=self._model_arch_value(),
            batch_size=max(batch_size, 1),
            drop_last=self._lora_staged_resolution_compile_drop_last,
        )
        self._dataset = training_input.dataset
        self._lora_staged_resolution_active_index = index
        self._lora_staged_resolution_sdxl_cache_first = training_input.sdxl_cache_first
        new_dataloader = self._maybe_wrap_lora_stage_ddp(training_input.dataloader, training_input.dataset)
        self._dataloader = new_dataloader
        self._log(
            "LoRA staged resolution: "
            f"epoch={epoch + 1}, resolution={stage.resolution}, "
            f"batch={batch_size}, samples={len(training_input.dataset)}, "
            f"sdxl_cache_first={training_input.sdxl_cache_first}"
        )
        return new_dataloader

    def _maybe_create_eval_dataloader(
        self,
        *,
        model_arch: str,
        anima_cached_training: bool,
        newbie_cached_training: bool,
        caption_bucket: int = 0,
    ):
        """Create an independent eval dataloader from eval_data_dir if configured."""

        eval_dir = str(getattr(self.config, "eval_data_dir", "") or "").strip()
        if not eval_dir:
            return None
        eval_path = Path(eval_dir)
        if not eval_path.is_dir():
            raise FileNotFoundError(f"eval_data_dir does not exist: {eval_path}")

        batch_size = int(getattr(self.config, "eval_batch_size", 0) or 0)
        if batch_size <= 0:
            batch_size = int(getattr(self.config, "batch_size", 1) or 1)
        common_loader_kwargs = {
            "batch_size": max(batch_size, 1),
            "shuffle": False,
            "num_workers": int(getattr(self.config, "dataloader_num_workers", 0) or 0),
            "persistent_workers": bool(getattr(self.config, "persistent_data_loader_workers", False)),
            "pin_memory": bool(getattr(self.config, "pin_memory", True)),
            "prefetch_factor": getattr(self.config, "prefetch_factor", 2),
            "collate_mode": getattr(self.config, "cached_collate_mode", "auto"),
        }

        if anima_cached_training:
            from .anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader

            eval_dataset = AnimaCachedDataset(
                data_dir=eval_path,
                latent_crop_size=int(getattr(self.config, "anima_cached_latent_crop_size", 0) or 0),
                text_token_limit=int(getattr(self.config, "anima_cached_text_token_limit", 0) or 0),
                fixed_text_tokens=int(getattr(self.config, "anima_fixed_text_tokens", 0) or 0),
                fixed_visual_tokens=int(getattr(self.config, "anima_fixed_visual_tokens", 0) or 0),
                caption_extension=getattr(self.config, "caption_extension", ".txt"),
                shuffle_caption=bool(getattr(self.config, "shuffle_caption", False)),
                shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
                keep_tokens=int(getattr(self.config, "keep_tokens", 0) or 0),
                keep_tokens_separator=str(getattr(self.config, "keep_tokens_separator", "") or ""),
                weighted_captions=bool(getattr(self.config, "weighted_captions", False)),
                caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
            )
            eval_loader = create_anima_cached_dataloader(eval_dataset, **common_loader_kwargs)
        elif newbie_cached_training:
            from .newbie_cached_dataset import NewbieCachedDataset, create_newbie_cached_dataloader

            eval_dataset = NewbieCachedDataset(
                data_dir=eval_path,
                latent_crop_size=int(getattr(self.config, "newbie_cached_latent_crop_size", 0) or 0),
                text_token_limit=int(getattr(self.config, "newbie_cached_text_token_limit", 0) or 0),
                caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
            )
            eval_loader = create_newbie_cached_dataloader(eval_dataset, **common_loader_kwargs)
        else:
            from .eval_dataset import EvalDatasetConfig, create_eval_caption_dataset, create_eval_dataloader

            eval_dataset = create_eval_caption_dataset(
                EvalDatasetConfig(
                    data_dir=str(eval_path),
                    resolution=self._get_dataset_resolution(),
                    caption_extension=self.config.caption_extension,
                    enable_bucket=self.config.enable_bucket,
                    min_bucket_reso=self.config.min_bucket_reso,
                    max_bucket_reso=self.config.max_bucket_reso,
                    bucket_reso_steps=self.config.bucket_reso_steps,
                    bucket_selection_mode=getattr(self.config, "bucket_selection_mode", "aspect"),
                    bucket_custom_resos=getattr(self.config, "bucket_custom_resos", ""),
                    keep_tokens=self.config.keep_tokens,
                    keep_tokens_separator=getattr(self.config, "keep_tokens_separator", ""),
                    weighted_captions=bool(getattr(self.config, "weighted_captions", False)),
                    masked_loss=bool(getattr(self.config, "masked_loss", False)),
                    alpha_mask=bool(getattr(self.config, "alpha_mask", False)),
                    caption_length_bucket_size=caption_bucket if model_arch == "newbie" else 0,
                )
            )
            eval_loader = create_eval_dataloader(
                eval_dataset,
                batch_size=common_loader_kwargs["batch_size"],
                num_workers=common_loader_kwargs["num_workers"],
                pin_memory=common_loader_kwargs["pin_memory"],
                prefetch_factor=common_loader_kwargs["prefetch_factor"],
                persistent_workers=common_loader_kwargs["persistent_workers"],
            )

        self._log(
            f"Independent eval dataset enabled: {len(eval_dataset)} samples from {eval_path} "
            f"(batch_size={common_loader_kwargs['batch_size']}, shuffle=False)"
        )
        return eval_loader

    def _delete_newbie_cache_artifacts(self) -> int:
        """Delete generated Newbie cache files for an explicit rebuild request."""
        if not self.config:
            return 0
        data_dir = Path(str(getattr(self.config, "train_data_dir", "") or ""))
        if not data_dir.is_dir():
            return 0

        deleted = 0
        seen: set[Path] = set()
        patterns = (
            "*_newbie.npz",
            "*_newbie.safetensors",
            "*_newbie.pt",
            "lulynx_cache_manifest_newbie.json",
            "lulynx_cache_metadata_newbie.json",
        )
        for pattern in patterns:
            for cache_file in data_dir.rglob(pattern):
                if cache_file in seen:
                    continue
                seen.add(cache_file)
                try:
                    cache_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.debug(f"Failed to delete cache file {cache_file}: {e}")

        te_cache_dir = data_dir / "te_cache"
        if te_cache_dir.is_dir():
            for te_file in te_cache_dir.glob("te_*.npz"):
                if te_file in seen:
                    continue
                seen.add(te_file)
                try:
                    te_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.debug(f"Failed to delete TE cache file {te_file}: {e}")
        return deleted

    def _maybe_build_newbie_cache(self, *, force: bool = False) -> None:
        """Build Newbie cache artifacts when a cache mode explicitly asks for it."""
        if self._model_arch_value() != "newbie":
            return

        wants_cache = (
            bool(getattr(self.config, "use_cache", False))
            or bool(getattr(self.config, "newbie_force_cache_only", False))
            or bool(getattr(self.config, "newbie_rebuild_cache", False))
        )
        if not wants_cache:
            return
        if self._has_newbie_cached_training_data() and not force:
            return

        from .newbie_cache_builder import build_newbie_cache
        from .caption_source_mix import normalize_caption_source_mix_config

        self._log("Newbie cache builder: encoding image/caption pairs into *_newbie.npz")
        result = build_newbie_cache(
            loaded_model=self.model,
            data_dir=self.config.train_data_dir,
            device=self.device,
            dtype=self.dtype,
            resolution=self._get_resolution_pair(),
            caption_extension=getattr(self.config, "caption_extension", ".txt"),
            gemma3_prompt=getattr(self.config, "newbie_gemma3_prompt", ""),
            gemma_max_token_length=int(getattr(self.config, "newbie_gemma_max_token_length", 512) or 512),
            clip_max_token_length=int(getattr(self.config, "newbie_clip_max_token_length", 2048) or 2048),
            alpha_mask=bool(getattr(self.config, "masked_loss", False) or getattr(self.config, "alpha_mask", False)),
            force=force,
            caption_source_mix=normalize_caption_source_mix_config(
                enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
                nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
                tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
                trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
                empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
                trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
            ),
            log=self._log,
        )
        self._log(
            "Newbie cache builder complete: "
            f"written={result.written}, skipped={result.skipped}, errors={len(result.errors)}"
        )
        if isinstance(getattr(self, "_newbie_cache_first_profile", None), dict):
            samples_seen = int(result.written) + int(result.skipped)
            self._newbie_cache_first_profile["cache_builder"] = {
                "profile": "newbie_cache_builder_timing_v0",
                "written": int(result.written),
                "skipped": int(result.skipped),
                "samples_seen": int(samples_seen),
                "manifest_path": str(getattr(result, "manifest_path", "") or ""),
                "metadata_path": str(getattr(result, "metadata_path", "") or ""),
                "metadata_fast_path": bool(getattr(result, "metadata_fast_path", False)),
                "cache_trust": dict(getattr(result, "cache_trust", {}) or {}),
            }
        if result.errors:
            preview = "; ".join(result.errors[:3])
            raise RuntimeError(f"Newbie cache builder failed for {len(result.errors)} files: {preview}")
        # Cache-first Newbie training consumes only cached tensors plus the
        # native transformer. Release heavyweight encoder/VAE components after
        # cache generation so the subsequent training step does not pay for
        # keeping Gemma/Jina/VAE resident.
        if bool(getattr(self.config, "use_cache", False)) and self._has_newbie_cached_training_data():
            self._release_newbie_cache_builder_components()

    def _release_newbie_cache_builder_components(self) -> None:
        """Release cache-builder-only Newbie components after cache generation."""
        if self._model_arch_value() != "newbie" or self.model is None:
            return
        released = []
        for attr in ("text_encoder_1", "text_encoder_2", "vae", "tokenizer_1", "tokenizer_2"):
            value = getattr(self.model, attr, None)
            if value is None:
                continue
            if isinstance(value, torch.nn.Module):
                try:
                    value.to("cpu")
                except Exception:
                    pass
            setattr(self.model, attr, None)
            released.append(attr)
        if released:
            self._log(
                "Newbie cache-first training: released cache-builder components after cache generation: "
                + ", ".join(released)
            )
            self._maybe_release_tool_cuda_cache(
                "newbie_cache_builder_components_released",
                collect_gc=True,
            )

    def _refresh_newbie_cache_builder_timing_profile(self) -> Dict[str, Any]:
        profile = getattr(self, "_newbie_cache_first_profile", None)
        if not isinstance(profile, dict):
            return {}
        builder = profile.get("cache_builder")
        if not isinstance(builder, dict):
            return {}
        steps = profile.get("steps")
        build_step = None
        if isinstance(steps, list):
            for item in reversed(steps):
                if isinstance(item, dict) and item.get("label") == "build_cache":
                    build_step = item
                    break
        elapsed = float(builder.get("elapsed_seconds", 0.0) or 0.0)
        if isinstance(build_step, dict):
            elapsed = float(build_step.get("dt_seconds", 0.0) or 0.0)
            builder["profile_step_label"] = "build_cache"
            builder["profile_total_seconds"] = float(build_step.get("total_seconds", 0.0) or 0.0)
        samples_seen = int(
            builder.get("samples_seen", int(builder.get("written", 0) or 0) + int(builder.get("skipped", 0) or 0))
            or 0
        )
        builder["profile"] = "newbie_cache_builder_timing_v0"
        builder["elapsed_seconds"] = round(max(elapsed, 0.0), 4)
        builder["samples_seen"] = samples_seen
        builder["samples_per_second"] = round(samples_seen / elapsed, 4) if elapsed > 0.0 else 0.0
        profile["cache_builder"] = builder
        return dict(builder)

    def _prepare_newbie_cache_first_runtime(self) -> None:
        """Build Newbie cache with encoder-only components, then reload transformer only.

        Cache generation may need VAE/Gemma/CLIP, but the actual training step
        should not keep those components resident once explicit cache artifacts
        exist.
        """
        if self._model_arch_value() != "newbie":
            return
        wants_cache = bool(getattr(self.config, "use_cache", False))
        if not wants_cache:
            return

        profile_start = time.perf_counter()
        profile_last = profile_start
        profile_steps: List[Dict[str, Any]] = []

        def mark(label: str, **data: Any) -> None:
            nonlocal profile_last
            now = time.perf_counter()
            item: Dict[str, Any] = {
                "label": str(label),
                "dt_seconds": round(max(now - profile_last, 0.0), 4),
                "total_seconds": round(max(now - profile_start, 0.0), 4),
            }
            for key, value in data.items():
                if value is not None:
                    item[str(key)] = value
            profile_steps.append(item)
            profile_last = now

        def loader_progress(stage: str, data: Dict[str, Any]) -> None:
            now = time.perf_counter()
            payload: Dict[str, Any] = {
                "stage": str(stage),
                "elapsed_seconds": round(max(now - profile_start, 0.0), 4),
                **dict(data or {}),
            }
            if isinstance(getattr(self, "_newbie_cache_first_profile", None), dict):
                self._newbie_cache_first_profile["active_stage"] = str(stage)
                self._newbie_cache_first_profile["last_loader_progress"] = dict(payload)
            self._emit_runtime_event(
                {
                    "event_type": "newbie_cache_first_progress",
                    "severity": "info",
                    "summary": str(stage),
                    "data": payload,
                }
            )

        rebuild_requested = bool(getattr(self.config, "newbie_rebuild_cache", False))
        cache_present_before = self._has_newbie_cached_training_data()
        self._newbie_cache_first_profile = {
            "route": "newbie",
            "cache_present_before": bool(cache_present_before),
            "rebuild_cache_requested": bool(rebuild_requested),
            "used_prebuilt_cache": bool(cache_present_before and not rebuild_requested),
            "rebuilt_cache": False,
            "steps": profile_steps,
            "loader_profiles": {},
        }
        if rebuild_requested:
            deleted = self._delete_newbie_cache_artifacts()
            cache_present_before = self._has_newbie_cached_training_data()
            self._newbie_cache_first_profile["deleted_cache_artifacts"] = int(deleted)
            self._newbie_cache_first_profile["cache_present_after_rebuild_delete"] = bool(cache_present_before)
            self._log(
                f"Newbie rebuild_cache: deleted {deleted} cache artifacts from "
                f"{Path(str(getattr(self.config, 'train_data_dir', '') or ''))}"
            )
            mark("delete_rebuild_cache", deleted_cache_artifacts=int(deleted))

        def capture_loader_profile(label: str) -> None:
            profile = getattr(getattr(self, "model", None), "newbie_loader_profile", None)
            if isinstance(profile, dict):
                self._newbie_cache_first_profile.setdefault("loader_profiles", {})[label] = dict(profile)

        if cache_present_before and not rebuild_requested:
            from .newbie_loader import load_newbie_transformer_only_from_config

            self._log("Newbie cache-first prepare: cached tensors found; loading transformer-only training bundle")
            self.model = load_newbie_transformer_only_from_config(
                self.config,
                device=self.device,
                dtype=self.dtype,
            )
            capture_loader_profile("transformer_only")
            mark("load_transformer_only")
            self._newbie_cache_first_profile["total_seconds"] = profile_steps[-1]["total_seconds"] if profile_steps else 0.0
            return

        from .newbie_loader import (
            load_newbie_encoders_only_from_config,
            load_newbie_transformer_only_from_config,
            release_loaded_model_components,
        )

        self._log(
            "Newbie cache-first prepare: loading encoder-only cache-builder bundle "
            "(VAE + Gemma + CLIP, transformer deferred)"
        )
        self.model = load_newbie_encoders_only_from_config(
            self.config,
            device="cpu",
            dtype=torch.float32,
            progress_callback=loader_progress,
        )
        capture_loader_profile("encoder_bundle")
        mark("load_encoder_bundle")
        self._maybe_build_newbie_cache(force=rebuild_requested)
        cache_present_after_build = bool(self._has_newbie_cached_training_data())
        mark(
            "build_cache",
            cache_present_after_build=cache_present_after_build,
        )
        self._refresh_newbie_cache_builder_timing_profile()
        if not cache_present_after_build:
            raise RuntimeError(
                "Newbie cache-first prepare did not produce cached training data. "
                "Check train_data_dir contains supported images and cache-builder inputs are valid."
            )
        if rebuild_requested:
            self._newbie_cache_rebuild_handled_in_prepare = True
        self._newbie_cache_first_profile["rebuilt_cache"] = True
        released = release_loaded_model_components(
            self.model,
            "text_encoder_1",
            "text_encoder_2",
            "vae",
            "tokenizer_1",
            "tokenizer_2",
            "noise_scheduler",
            "unet",
        )
        mark("release_encoder_bundle", released_components=list(released or []))
        if released:
            self._log(
                "Newbie cache-first prepare: released cache-builder bundle components: "
                + ", ".join(released)
            )
        self.model = None
        self._maybe_release_tool_cuda_cache(
            "newbie_cache_first_bundle_released",
            collect_gc=True,
        )
        mark("release_cuda_cache")

        self._log("Newbie cache-first prepare: loading transformer-only training bundle")
        self.model = load_newbie_transformer_only_from_config(
            self.config,
            device=self.device,
            dtype=self.dtype,
        )
        capture_loader_profile("transformer_only")
        mark("load_transformer_only")
        self._newbie_cache_first_profile["total_seconds"] = profile_steps[-1]["total_seconds"] if profile_steps else 0.0

    # ------------------------------------------------------------------
    # R1 cache/faithful methods moved to trainer_anima_cache_runtime.TrainerAnimaCacheRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _apply_native_runtime_profile(self) -> None:
        """Apply optional native-family runtime profiles without replacing defaults."""
        if not self.config:
            return
        profile = str(getattr(self.config, "native_runtime_profile", "standard") or "standard").strip().lower()
        model_arch = self._model_arch_value()
        if profile in {"", "standard"}:
            return

        def _set_if_auto(attr: str, value: Any) -> bool:
            if not hasattr(self.config, attr):
                return False
            current = str(getattr(self.config, attr, "") or "").strip().lower()
            if current in {"", "auto", "default"}:
                setattr(self.config, attr, value)
                return True
            return False

        def _set_if_empty(attr: str, value: Any) -> bool:
            if not hasattr(self.config, attr):
                return False
            current = getattr(self.config, attr, None)
            if current in (None, "", 0, False):
                setattr(self.config, attr, value)
                return True
            return False

        def _enable_native_dit_fused_adamw() -> bool:
            """Prefer PyTorch fused AdamW for native DiT LoRA when it is safe.

            The reference Anima stack gets a large optimizer-phase win from
            torch.optim.AdamW(fused=True). Keep this automatic path narrow:
            native DiT profiles, CUDA, AdamW, and no user-supplied optimizer
            args/backend. If the user already provided args, they remain authoritative.
            """
            if not torch.cuda.is_available():
                return False
            if getattr(self.config, "optimizer", None) != OptimizerType.ADAMW:
                return False
            raw_args = getattr(self.config, "optimizer_args", "")
            if self._parse_custom_args(raw_args):
                return False
            if not hasattr(self.config, "optimizer_backend"):
                return False
            current_backend = str(getattr(self.config, "optimizer_backend", "auto") or "auto").strip().lower()
            if current_backend not in {"", "auto", "default"}:
                return False
            self.config.optimizer_backend = "torch_fused"
            return True

        if profile in {"aggressive", "anima_fast"} and model_arch in {"anima", "newbie"}:
            _set_if_auto("attention_backend", "flash2")
            _set_if_auto(f"{model_arch}_attn_mode", "flash2")
            if model_arch == "anima":
                inferred_text_tokens = self._infer_anima_cached_text_tokens()
                if _set_if_empty("anima_fixed_text_tokens", inferred_text_tokens or 256):
                    self._log(
                        f"[runtime-opt] anima_fixed_text_tokens auto-sized to "
                        f"{int(getattr(self.config, 'anima_fixed_text_tokens', 0) or 0)} from cached text metadata"
                    )
                if int(getattr(self.config, "anima_fixed_visual_tokens", 0) or 0) <= 0:
                    self._log("[runtime-opt] Anima native profile uses no-pad visual token buckets; fixed visual padding remains off.")
                if str(getattr(self.config, "native_cache_mode", "") or "").strip().lower() in {"", "auto"}:
                    self.config.native_cache_mode = "cache_first"
                if str(getattr(self.config, "anima_cache_mode", "") or "").strip().lower() in {"", "auto"}:
                    self.config.anima_cache_mode = "cache_first"
            else:
                _set_if_empty("newbie_fixed_text_tokens", getattr(self.config, "newbie_gemma_max_token_length", 512) or 512)
                if str(getattr(self.config, "native_cache_mode", "") or "").strip().lower() in {"", "auto"}:
                    self.config.native_cache_mode = "cache_first"
                if not bool(getattr(self.config, "use_cache", False)):
                    self.config.use_cache = True
            if str(getattr(self.config, "anima_compile_scope", "") or "").strip().lower() in {"", "auto"}:
                self.config.anima_compile_scope = "per_block"
            if str(getattr(self.config, "torch_compile_scope", "") or "").strip().lower() in {"", "auto"}:
                self.config.torch_compile_scope = "per_block"
            self.config.torch_compile = True
            if hasattr(self.config, "cached_dataloader_auto_policy"):
                self.config.cached_dataloader_auto_policy = True
            fused_adamw_enabled = _enable_native_dit_fused_adamw()
            self._log(
                f"[runtime-opt] native_runtime_profile={profile} resolved for {model_arch}: "
                "DiT attention auto prefers flash2->sageattn->sdpa, cache-first DataLoader auto policy, "
                "per-block compile requested behind compile contract/probe"
                + (", PyTorch fused AdamW enabled." if fused_adamw_enabled else ".")
            )
        elif profile == "aggressive" and model_arch == "sdxl":
            if str(getattr(self.config, "torch_compile_scope", "") or "").strip().lower() in {"full", "full_core"}:
                self._log("[runtime-opt][warn] aggressive SDXL keeps full compile behind compile contract fallback")
            self._log(
                "[runtime-opt] native_runtime_profile=aggressive resolved for sdxl: "
                "keeps route auto attention policy (xFormers/SDPA) and does not force DiT-only backends."
            )
        elif profile == "anima_low_vram" and model_arch == "anima":
            _set_if_auto("attention_backend", "sdpa")
            _set_if_auto("anima_attn_mode", "sdpa")
            self.config.gradient_checkpointing = True
            self.config.anima_compile_scope = ""
            self._log("Anima low-VRAM profile enabled: gradient checkpointing + SDPA preference.")
        elif profile == "anima_experimental" and model_arch == "anima":
            self._log(
                "Anima experimental profile selected. Full compile/CUDAGraph and advanced methods "
                "remain guarded until smoke-tested."
            )
        else:
            self._log(f"[runtime-opt][warn] native_runtime_profile={profile} is not supported for {model_arch}; keeping standard behavior.")

    def _apply_sdxl_low_vram_profile(self) -> None:
        """Apply VRAM-saving defaults for Diffusers UNet LoRA routes.

        Only overrides config values that are not already user-set.  Should be
        called after the model is loaded (so VAE/UNet are available) but before
        LoRA injection and training-loop creation.
        """
        if self.config is None or not self._is_diffusers_unet_route():
            return
        if not getattr(self.config, "sdxl_low_vram_optimization", False):
            return
        model_label = self._model_arch_value().upper() or "Diffusers UNet"
        self._log(f"Applying {model_label} low VRAM profile...")

        # VAE slicing
        if self.model is not None and getattr(self.model, "vae", None) is not None:
            if hasattr(self.model.vae, "enable_slicing"):
                self.model.vae.enable_slicing()
                self._log("  VAE slicing enabled")
            # Also ensure config flag is set so PipelineSlicer applies it
            if not getattr(self.config, "vae_slicing", False):
                self.config.vae_slicing = True

        # Attention slicing on UNet (if supported)
        if self.model is not None and getattr(self.model, "unet", None) is not None:
            if hasattr(self.model.unet, "enable_attention_slicing"):
                self.model.unet.enable_attention_slicing()
                self._log("  UNet attention slicing enabled")
            if not getattr(self.config, "attention_slicing", False):
                self.config.attention_slicing = True

        new_profile = str(getattr(self.config, "low_vram_profile", "off") or "off").strip().lower()
        new_profile_active = new_profile not in {"", "off", "none", "disabled"}

        # Gradient checkpointing (only override if not already enabled)
        if not getattr(self.config, "gradient_checkpointing", False):
            self.config.gradient_checkpointing = True
            self._log("  Gradient checkpointing enabled")

        # Cache latents (only override if not already enabled)
        if not getattr(self.config, "cache_latents", False):
            self.config.cache_latents = True
            self._log("  Cache latents enabled")

        strategy = str(getattr(self.config, "te_vae_offload_strategy", "phase") or "phase").strip().lower()
        if not new_profile_active and strategy in {"", "phase"}:
            self.config.te_vae_offload_strategy = "aggressive"
            self._log("  TE/VAE offload strategy set to aggressive")

        # Block swap (only set if not already set by user)
        if not new_profile_active and not getattr(self.config, "blocks_to_swap", 0):
            self.config.blocks_to_swap = 2
            self._log("  blocks_to_swap set to 2")

    def _apply_sdxl_lora_low_vram_profile(self, model_arch: str) -> None:
        if self.config is None:
            return
        decision = apply_sdxl_lora_low_vram_profile(self.config, model_arch=model_arch)
        self._sdxl_lora_low_vram_profile = decision.as_dict()
        self._attach_memory_runtime_profiles_to_training_loop()
        if not decision.enabled:
            return
        changed_keys = ", ".join(sorted(decision.changes)) or "none"
        self._log(
            "SDXL/LoRA low-VRAM profile resolved: "
            f"requested={decision.requested}, effective={decision.effective}, changes={changed_keys}"
        )
        for warning in decision.warnings:
            self._log(f"[low-vram-profile][warn] {warning}")
        for skipped in decision.skipped:
            self._log(
                "[low-vram-profile] skipped "
                f"{skipped.get('key', '?')}: {skipped.get('reason', 'unknown')}"
            )

    def _align_sdxl_text_encoder_cache_contract(self) -> None:
        """Keep Diffusers UNet text-cache semantics coherent."""
        if not self._is_diffusers_unet_route() or not bool(getattr(self.config, "cache_text_encoder_outputs", False)):
            return

        train_unet = bool(getattr(self.config, "train_unet", True))
        train_text = bool(getattr(self.config, "train_text_encoder", False))
        if not train_text:
            self._refresh_diffusers_cache_runtime_profile(
                model_arch=self._model_arch_value(),
                component_cpu_residency=self._should_use_sdxl_component_cpu_residency(),
            )
            return

        if not train_unet:
            self.config.cache_text_encoder_outputs = False
            self._refresh_diffusers_cache_runtime_profile(
                model_arch=self._model_arch_value(),
                text_cache_disabled_reason="text_encoder_only_training",
                component_cpu_residency=self._should_use_sdxl_component_cpu_residency(),
            )
            self._log("Diffusers text encoder cache disabled: text-encoder-only training needs live text updates.")
            return

        self.config.network_train_unet_only = True
        self.config.network_train_text_encoder_only = False
        self._refresh_diffusers_cache_runtime_profile(
            model_arch=self._model_arch_value(),
            text_cache_forced_unet_only=True,
            component_cpu_residency=self._should_use_sdxl_component_cpu_residency(),
        )
        self._log("Diffusers text encoder cache aligned: cache_text_encoder_outputs now forces UNet-only training.")

    def _should_use_sdxl_component_cpu_residency(self) -> bool:
        """Hold frozen Diffusers VAE / TE components off device until needed."""
        if not self._is_diffusers_unet_route():
            return False
        strategy = str(getattr(self.config, "te_vae_offload_strategy", "phase") or "phase").strip().lower()
        if strategy == "resident":
            return False
        if strategy == "aggressive":
            return True
        if bool(getattr(self.config, "enable_sequential_cpu_offload", False)):
            return True
        if bool(getattr(self.config, "sdxl_low_vram_optimization", False)):
            return True
        if (
            bool(getattr(self.config, "cache_text_encoder_outputs", False))
            and not bool(getattr(self.config, "train_text_encoder", False))
        ):
            return True
        return False

    def _resolve_cache_disk_dtype(self, value: Any = None) -> str:
        """Resolve user-facing cache dtype aliases before constructing disk caches."""
        raw = str(value if value is not None else "").strip().lower()
        if raw in {"", "auto", "default"}:
            dtype = getattr(self, "dtype", None)
            if dtype is torch.bfloat16:
                return "bfloat16"
            if dtype is torch.float32:
                return "float32"
            return "float16"
        aliases = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
            "half": "float16",
            "fp32": "float32",
            "float32": "float32",
        }
        return aliases.get(raw, raw)

    def _attach_weight_compression_profile_to_training_loop(self) -> None:
        profile = getattr(self, "_weight_compression_profile", None)
        if profile and self.training_loop is not None and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["weight_compression"] = dict(profile)

    def _apply_weight_compression(self) -> None:
        """Compress frozen base/text-encoder weights behind the Warehouse contract."""
        from .weight_compression import apply_weight_compression, resolve_weight_compression_runtime_config

        resolved = resolve_weight_compression_runtime_config(self.config)
        self._weight_compression_profile = resolved.as_dict()
        if not resolved.requested:
            self._attach_weight_compression_profile_to_training_loop()
            return
        fp8_compute = bool(getattr(self.config, "fp8_base_compute", False))
        if (
            resolved.format == "fp8_e4m3"
            and bool(getattr(self.config, "weight_compression_verify", True))
            and not fp8_compute
        ):
            self._weight_compression_profile.update({"applied": False, "skip_reason": "native_fp8_storage_only"})
            self._log(
                "[weight-compression][warn] native fp8_e4m3 direct-cast is storage-only in this PyTorch build; "
                "skipping training-time compression to avoid unsupported Linear forward. "
                "Enable fp8_base_compute for the FP8 tensor-core forward, or use a torchao/quanto format."
            )
            self._attach_weight_compression_profile_to_training_loop()
            return
        result = apply_weight_compression(
            self.model,
            enabled=resolved.enabled,
            target=resolved.target,
            format=resolved.format,
            lora_injector=self.lora_injector,
            train_text_encoder=bool(getattr(self.config, "train_text_encoder", False)),
            include_patterns=getattr(self.config, "weight_compression_include_patterns", ""),
            exclude_patterns=getattr(self.config, "weight_compression_exclude_patterns", ""),
            legacy_fp8_base=resolved.legacy_fp8_base,
        )
        self._weight_compression_profile.update(result.as_dict())
        self._weight_compression_profile["applied"] = bool(result.enabled and result.compressed_count > 0)
        if fp8_compute and resolved.format == "fp8_e4m3" and result.compressed_count > 0:
            flagged = self._flag_fp8_base_compute_linears()
            self._weight_compression_profile["fp8_base_compute_linears"] = flagged
            self._log(f"FP8 base compute enabled on {flagged} Linear layers (Ada tensor-core GEMM).")
        for warning in result.warnings:
            self._log(f"[weight-compression][warn] {warning}")
        if not result.enabled:
            self._weight_compression_profile.setdefault("skip_reason", "disabled_or_unavailable")
            self._log("Weight compression skipped")
            self._attach_weight_compression_profile_to_training_loop()
            return
        self._attach_weight_compression_profile_to_training_loop()
        detail = ", ".join(f"{c.name}:{c.compressed_count}" for c in result.components) or "none"
        self._log(
            "Weight compression applied "
            f"(target={result.target}, format={result.format}, params={result.compressed_count}, "
            f"estimated VRAM savings: {result.estimated_saved_mb:.1f} MB, components={detail})"
        )

    def _apply_fp8_quantization(self) -> None:
        """Backward-compatible wrapper for the legacy fp8_base path."""
        self._apply_weight_compression()

    def _flag_fp8_base_compute_linears(self) -> int:
        """Flag fp8 base Linears so ``LoRALinear._base_forward`` runs the FP8
        tensor-core GEMM (``torch._scaled_mm``) instead of the bf16 dequant path.

        Returns the number of Linear layers flagged.
        """
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None or self.model is None:
            return 0
        # ``self.model`` may be a LoadedModel container (anima/newbie native DiT)
        # whose backbone lives under ``.unet``, or a bare nn.Module (SDXL). Resolve
        # the module(s) to walk the same way apply_weight_compression does, so we do
        # not call ``.modules()`` on the non-Module container.
        roots = []
        backbone = getattr(self.model, "unet", None)
        if isinstance(backbone, torch.nn.Module):
            roots.append(backbone)
        elif isinstance(self.model, torch.nn.Module):
            roots.append(self.model)
        count = 0
        for root in roots:
            for module in root.modules():
                if isinstance(module, torch.nn.Linear):
                    weight = getattr(module, "weight", None)
                    if getattr(weight, "dtype", None) == fp8_dtype:
                        module._fp8_base_compute = True
                        self._install_fp8_linear_dequant_shim(module, fp8_dtype)
                        count += 1
        return count

    @staticmethod
    def _install_fp8_linear_dequant_shim(module: "torch.nn.Linear", fp8_dtype) -> None:
        """Make a *standalone* fp8 ``nn.Linear`` forward-safe via a bf16 dequant.

        After native fp8 cast, ``module.weight`` is ``float8_e4m3fn``; a plain
        ``F.linear`` then raises (fp8 matmul unsupported). LoRA-wrapped Linears
        are handled by ``LoRALinear._base_forward`` (which bypasses this shim),
        but unwrapped base Linears (x/t-embedder, adaLN modulation, final layer)
        reach their own ``forward`` and would crash. The shim transiently
        upcasts the single weight to the activation dtype and runs a normal
        differentiable ``F.linear`` — fp8 *storage* savings are kept, only one
        weight is materialised in bf16 at a time, and autograd flows to any
        downstream LoRA params. Idempotent.
        """
        if getattr(module, "_fp8_dequant_shimmed", False):
            return
        import torch.nn.functional as F

        def _fp8_dequant_forward(x):
            w = module.weight
            b = module.bias
            wb = w.to(device=x.device, dtype=x.dtype) if getattr(w, "dtype", None) == fp8_dtype else w
            bb = b.to(device=x.device, dtype=x.dtype) if b is not None else None
            return F.linear(x, wb, bb)

        module.forward = _fp8_dequant_forward  # type: ignore[assignment]
        module._fp8_dequant_shimmed = True


    def _apply_compression_companion(self) -> None:
        """Load a frozen recovery adapter and merge it into base Linear weights."""
        if not bool(getattr(self.config, "compression_companion_enabled", False)):
            return
        from .compression_companion import apply_compression_companion

        result = apply_compression_companion(
            self.lora_injector,
            path=getattr(self.config, "compression_companion_path", ""),
            companion_type=getattr(self.config, "compression_companion_type", "lora"),
            mode=getattr(self.config, "compression_companion_mode", "merge_into_base"),
            scale=float(getattr(self.config, "compression_companion_scale", 1.0) or 1.0),
            disable_mmap=bool(getattr(self.config, "disable_mmap_load_safetensors", False)),
        )
        for warning in result.warnings:
            self._log(f"[compression-companion][warn] {warning}")
        self._log(
            "Compression companion applied "
            f"(mode={result.mode}, type={result.type}, merged_layers={result.merged_layers}, "
            f"reset_layers={result.reset_layers}, scale={result.scale})"
        )

    def _apply_sequential_cpu_offload(self) -> None:
        """Apply sequential CPU offload hooks so each UNet/DiT sub-module is
        moved to GPU only for the duration of its forward pass, then
        immediately moved back to CPU.  This mimics the diffusers
        ``enable_sequential_cpu_offload()`` pattern and slashes VRAM for
        SDXL / Anima models on low-memory GPUs.
        """
        unet = getattr(self.model, "unet", None)
        if unet is None:
            self._log("Sequential CPU offload skipped: no unet/dit model found")
            return

        device = torch.device(self.device)
        moved = 0

        for name, module in unet.named_modules():
            # Only offload leaf modules with parameters (Linear, Conv2d, LayerNorm, …)
            if not list(module.children()) and list(module.parameters()):
                _orig_forward = module.forward

                def _make_offload_forward(orig_fwd, mod):
                    def _offload_forward(*args, **kwargs):
                        mod.to(device)
                        try:
                            result = orig_fwd(*args, **kwargs)
                        finally:
                            mod.to("cpu")
                        return result
                    return _offload_forward

                module.forward = _make_offload_forward(_orig_forward, module)
                moved += 1

        # Move the top-level model to CPU; each sub-module will be temporarily
        # brought to GPU inside its hooked forward.
        unet.to("cpu")
        self._maybe_release_tool_cuda_cache(
            "sequential_cpu_offload_applied",
            collect_gc=True,
        )

        self._log(f"Sequential CPU offload applied: {moved} sub-modules hooked")

    def _apply_native_cache_mode(self) -> None:
        """Normalize native cache-mode switches into existing per-family fields."""
        if not self.config:
            return
        mode = str(getattr(self.config, "native_cache_mode", "") or "").strip().lower()
        if not mode:
            return
        model_arch = self._model_arch_value()
        if model_arch == "anima":
            if mode in {"cache_first", "cache-first", "force_cache_only"}:
                self.config.anima_cached_training = True
                if mode == "force_cache_only":
                    self._log("Anima force_cache_only requested; expecting paired cache files at train_data_dir.")
            elif mode in {"online_cache", "online-cache"}:
                self.config.anima_cached_training = True
                self.config.anima_online_cache = True
                self._log("Anima native_cache_mode=online_cache: missing per-sample cache files will be generated and persisted.")
            elif mode in {"raw_online", "raw-online", "online"}:
                self.config.anima_cached_training = False
                self.config.anima_online_cache = False
                self._log(
                    "Anima native_cache_mode=raw_online: raw-image training enabled. "
                    "VAE and text encoders will run live each step without writing cache files."
                )
            elif mode == "rebuild_cache":
                self._build_anima_cache_now(force=True)
                self.config.anima_cached_training = True
        elif model_arch == "newbie":
            if mode == "rebuild_cache":
                self.config.use_cache = True
                self.config.newbie_rebuild_cache = True
            elif mode == "force_cache_only":
                self.config.use_cache = True
                self.config.newbie_force_cache_only = True
            elif mode in {"cache_first", "cache-first"}:
                self.config.use_cache = True
        self._log_cache_policy_report("native cache policy")
        if model_arch == "newbie":
            self._log_newbie_cache_parity_report("newbie cache parity")

    # ------------------------------------------------------------------
    # R2 cache-build methods moved to trainer_anima_cache_runtime.TrainerAnimaCacheRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _apply_runtime_env_hints(self) -> None:
        """Apply generic process env hints before model loading."""
        if not self.config:
            return
        if not bool(getattr(self.config, "pytorch_cuda_expandable_segments", False)):
            return
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        parts = [part.strip() for part in alloc_conf.split(",") if part.strip()]
        if not any(part == "expandable_segments:True" for part in parts):
            parts.insert(0, "expandable_segments:True")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
        self._log("Runtime env: PYTORCH_CUDA_ALLOC_CONF includes expandable_segments:True")

    def _apply_seed(self) -> None:
        """Apply process-level RNG seeding for deterministic trainer startup."""
        if not self.config:
            return
        raw_seed = int(getattr(self.config, "seed", 1337) or 0)
        if raw_seed < 0:
            self._log("Seed: negative sentinel requested, leaving RNG state unchanged")
            return

        # In DDP, offset seed by rank so each process gets different data
        from .distributed import get_rank
        rank = get_rank()
        seed = raw_seed + rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if rank > 0:
            self._log(f"Seed applied: {raw_seed} + rank={rank} → {seed}")
        else:
            self._log(f"Seed applied: {seed}")

    def _setup_ddp(self) -> None:
        """Initialize DDP if multi_gpu is enabled in config."""
        if not self.config:
            return

        multi_gpu = bool(getattr(self.config, "multi_gpu", False))
        if not multi_gpu:
            self._ddp_wrapper = None
            return

        num_processes = max(int(getattr(self.config, "num_processes", 1) or 1), 1)
        num_machines = max(int(getattr(self.config, "num_machines", 1) or 1), 1)
        main_process_ip = str(getattr(self.config, "main_process_ip", "localhost") or "localhost")
        main_process_port = int(getattr(self.config, "main_process_port", 29500) or 29500)

        from .distributed import setup_ddp, DDPModelWrapper, get_rank, get_world_size

        ddp_ok = setup_ddp(
            num_processes=num_processes,
            num_machines=num_machines,
            main_process_ip=main_process_ip,
            main_process_port=main_process_port,
        )

        if not ddp_ok:
            self._log("DDP requested but setup failed; continuing in single-process mode")
            self._ddp_wrapper = None
            return

        self._log(f"DDP active: rank={get_rank()}, world_size={get_world_size()}")

    def _cleanup_ddp(self) -> None:
        """Clean up DDP resources after training."""
        if self._ddp_wrapper is not None:
            self._ddp_wrapper = None
        from .distributed import cleanup_ddp
        cleanup_ddp()

    def _wrap_ddp(self, unet, optimizer, dataloader, dataset) -> tuple:
        """Wrap model/optimizer/dataloader for DDP if multi_gpu is enabled.

        Returns (model_to_train, optimizer, dataloader) — possibly DDP-wrapped.
        """
        from .distributed import is_ddp_active, DDPModelWrapper

        if not is_ddp_active():
            return unet, optimizer, dataloader

        if (
            dataset is not None
            and hasattr(dataset, "is_concept_geometry_enabled")
            and callable(getattr(dataset, "is_concept_geometry_enabled"))
            and dataset.is_concept_geometry_enabled()
        ):
            self._log(
                "[concept-geometry] DDP currently replaces the custom curriculum sampler with DistributedSampler; "
                "geometry loss weighting remains active."
            )

        find_unused = bool(getattr(self.config, "ddp_find_unused_parameters", False))
        bucket_view = bool(getattr(self.config, "ddp_gradient_as_bucket_view", True))
        static_graph = bool(getattr(self.config, "ddp_static_graph", False))

        wrapper = DDPModelWrapper(
            model=unet,
            optimizer=optimizer,
            dataloader=dataloader,
            dataset=dataset,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=bucket_view,
            static_graph=static_graph,
            device=torch.device(self.device),
        )
        self._ddp_wrapper = wrapper
        self._log(f"DDP wrapped model (find_unused={find_unused}, bucket_view={bucket_view}, static_graph={static_graph})")
        return wrapper.model, optimizer, wrapper.dataloader

    def _model_arch_value(self) -> str:
        raw = getattr(self.config, "model_arch", "") if self.config else ""
        return str(getattr(raw, "value", raw) or "")

    def _is_diffusers_unet_route(self) -> bool:
        return self._model_arch_value().strip().lower() in {"sdxl", "sd15"}

    def _is_lora_training_route(self) -> bool:
        if self.config is None:
            return True
        raw = getattr(self.config, "training_type", "lora")
        return "lora" in str(getattr(raw, "value", raw) or "lora").strip().lower().replace("_", "-")

    def _apply_diffusers_unet_attention_profile(self, model_arch: str) -> None:
        profile = getattr(self, "_attention_profile", None)
        if profile is None or not profile.is_active or not self._is_diffusers_unet_route():
            return
        try:
            from .runtime_optimizations import apply_sdxl_attention_profile

            patched = apply_sdxl_attention_profile(
                self.config,
                self.model,
                self.runtime_optimization_plan,
                profile=profile,
            )
            route_label = str(model_arch or "").strip().upper() or "Diffusers"
            skip_reason = ""
            if patched <= 0:
                recent_warnings = " ".join(
                    str(item).lower() for item in list(getattr(self.runtime_optimization_plan, "warnings", []) or [])[-3:]
                )
                if "native" in recent_warnings and "diffusers attention processors" in recent_warnings:
                    skip_reason = "native_unet_processors_unavailable"
                elif "unavailable" in recent_warnings:
                    skip_reason = "unet_unavailable"
                else:
                    skip_reason = "no_diffusers_attention_modules_patched"
            self._refresh_attention_runtime_profile(
                model_arch=model_arch,
                route=str(model_arch or "diffusers").strip().lower(),
                patched=patched,
                patch_target="diffusers_unet",
                applied=patched > 0,
                skip_reason=skip_reason,
                source="diffusers_unet_attention_profile",
            )
            self._attach_attention_runtime_profile_to_training_loop()
            if patched > 0:
                self._log(
                    f"Attention profile wired to {route_label} Diffusers U-Net self-attention: "
                    f"patched={patched}, window_size={profile.window_size}, "
                    f"backend={profile.backend}"
                )
        except Exception as exc:
            self._refresh_attention_runtime_profile(
                model_arch=model_arch,
                route=str(model_arch or "diffusers").strip().lower(),
                patch_target="diffusers_unet",
                applied=False,
                error=f"{type(exc).__name__}: {exc}",
                source="diffusers_unet_attention_profile",
            )
            self._attach_attention_runtime_profile_to_training_loop()
            self._log(f"Diffusers U-Net attention profile live wiring skipped: {exc}")

    def _ensure_native_family_training_ready(self) -> None:
        """Fail fast for partially wired native families instead of training on the wrong loop."""
        model_arch = self._model_arch_value()
        if model_arch == "newbie":
            report = NewbieReadinessReport.from_loaded_model(self.model)
            self._log(report.summary())
            if not report.can_train:
                raise RuntimeError(
                    "Newbie native training loop is not closed yet. "
                    "The current backend would otherwise fall back to an incompatible SDXL/DDPM path. "
                    f"Blocking issues: {', '.join(report.blocking_issues) if report.blocking_issues else 'native conditioning/transport not ready'}."
                )
        elif model_arch == "anima":
            report = getattr(self.model, "anima_load_report", None)
            if report is not None:
                self._log(f"Anima readiness: {report.summary()}")

            native_train_ready = bool(getattr(self.model, "anima_native_train_ready", False))

            # Allow training when either:
            #  1. Native DiT executable was loaded (cache-first path), OR
            #  2. Full diffusers components were loaded (live VAE/text encoding path)
            has_full_components = (
                getattr(self.model, "unet", None) is not None
                and getattr(self.model, "vae", None) is not None
                and getattr(self.model, "text_encoder_1", None) is not None
            )

            if has_full_components and not native_train_ready:
                self._log(
                    "Anima model loaded with full diffusers components — "
                    "enabling live (non-cached) training path."
                )
                self.model.anima_native_train_ready = True
                native_train_ready = True

            if not native_train_ready:
                raise RuntimeError(
                    "Anima native training loop is not ready. "
                    "Either load a diffusers-format model directory (provides UNet+VAE+TE), "
                    "or provide cached training data for the native DiT executable path. "
                    "When using a single safetensors checkpoint, the trainer can auto-build cache-first data only if VAE "
                    "and a primary text source (text_encoder_1/tokenizer_1 or Qwen3) are available."
                )

    def _anima_cache_mode_value(self) -> str:
        if not self.config:
            return ""
        return str(
            getattr(self.config, "native_cache_mode", "")
            or getattr(self.config, "anima_cache_mode", "")
            or ""
        ).strip().lower()

    def _should_auto_build_anima_cache_before_training(self) -> bool:
        return (
            self.config is not None
            and self._model_arch_value() == "anima"
            and bool(getattr(self.config, "anima_cached_training", True))
            and self._anima_cache_mode_value() != "force_cache_only"
        )

    def _ema_requested(self) -> bool:
        if not self.config:
            return False
        return bool(
            getattr(self.config, "ema_use_ema", False)
            or getattr(self.config, "use_ema", False)
        )

    def _get_model_save_extension(self) -> str:
        save_format = str(getattr(self.config, "save_model_as", "safetensors") or "safetensors").strip().lower()
        if save_format.startswith("."):
            save_format = save_format[1:]
        if save_format not in {"safetensors", "pt", "ckpt"}:
            self._log(f"Unknown save_model_as={save_format}, falling back to safetensors")
            save_format = "safetensors"
        return f".{save_format}"

    def _resolve_epoch_save_interval(self, total_epochs: Optional[int] = None) -> int:
        configured = max(int(getattr(self.config, "save_every_n_epochs", 1) or 1), 1)
        ratio = max(int(getattr(self.config, "save_n_epoch_ratio", 0) or 0), 0)
        resolved_total = max(int(total_epochs or getattr(self.config, "max_train_epochs", 1) or 1), 1)
        if ratio > 0:
            resolved = max(resolved_total // ratio, 1)
            self._log(f"save_n_epoch_ratio={ratio} resolved to save_every_n_epochs={resolved}")
            return resolved
        return configured

    def _parse_staged_resolution_steps(self) -> list[tuple[int, int]]:
        """Parse staged_resolution_steps + staged_resolution_values into (step, resolution) pairs."""
        steps_str = str(getattr(self.config, "staged_resolution_steps", "") or "").strip()
        values_str = str(getattr(self.config, "staged_resolution_values", "") or "").strip()
        if not steps_str or not values_str:
            return []
        try:
            steps = [int(s.strip()) for s in steps_str.split(",") if s.strip()]
            values = [int(v.strip()) for v in values_str.split(",") if v.strip()]
        except ValueError:
            self._log("Warning: staged_resolution_steps/values contain non-integer values, ignoring")
            return []
        # values has one more entry than steps: initial value + one per step boundary
        if len(values) < len(steps) + 1:
            self._log("Warning: staged_resolution_values must have one more entry than staged_resolution_steps, ignoring")
            return []
        result = []
        for i, step in enumerate(steps):
            result.append((step, values[i + 1]))
        return result

    def _apply_staged_resolution(self, new_resolution: int) -> None:
        """Apply a staged resolution change to the training config and dataset."""
        if not new_resolution or new_resolution <= 0:
            return
        old_resolution = getattr(self.config, "resolution", 0)
        if int(old_resolution) == new_resolution:
            return
        self.config.resolution = new_resolution
        if hasattr(self, "_dataset") and self._dataset:
            if hasattr(self._dataset, "resolution"):
                self._dataset.resolution = new_resolution
            if hasattr(self._dataset, "bucket_manager") and self._dataset.bucket_manager:
                self._dataset.bucket_manager.base_resolution = new_resolution
        self._log(f"Staged resolution changed from {old_resolution} to {new_resolution}")

    # ------------------------------------------------------------------
    # R3 cached-dataset methods moved to trainer_anima_cache_runtime.TrainerAnimaCacheRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # R1 retention methods moved to trainer_artifact_io.TrainerArtifactIoMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _maybe_save_final_training_state(self, epoch: int) -> None:
        if bool(getattr(self.config, "save_state", False)) or bool(getattr(self.config, "save_state_on_train_end", False)):
            self._save_state(epoch, final=True)

    def _sync_turbocore_native_update_state(self, reason: str) -> Dict[str, Any]:
        loop = getattr(self, "training_loop", None)
        sync = getattr(loop, "_sync_turbocore_native_update_training_executor_to_pytorch", None)
        if callable(sync):
            return dict(sync(reason) or {})
        return {}

    def _turbocore_native_update_state_sync_failed(self, report: Dict[str, Any]) -> bool:
        if not isinstance(report, dict) or not report:
            return False
        if bool(report.get("synced", True)):
            return False
        return bool(report.get("error")) or str(report.get("disable_reason", "") or "") == (
            "native_update_optimizer_state_sync_error"
        )

    def _close_turbocore_native_update_executor(self) -> None:
        loop = getattr(self, "training_loop", None)
        close = getattr(loop, "_close_turbocore_native_update_training_executor", None)
        if callable(close):
            close()

    def _write_training_advisor_report(self) -> None:
        """Write a lightweight S-tier pre-training advisor report."""
        if not bool(getattr(self.config, "training_advisor_enabled", True)):
            return
        try:
            from .distributed import is_main_process
            if not is_main_process():
                return
            from .training_advisor import write_training_advisor_report

            report = write_training_advisor_report(
                self.config,
                self.config.output_dir,
                filename=str(getattr(self.config, "training_advisor_report_name", "training_advisor_report.json") or "training_advisor_report.json"),
            )
            if bool(getattr(self.config, "training_advisor_log_summary", True)):
                summary = report.get("summary", {})
                vram = report.get("vram", {})
                self._log(
                    "Training advisor: "
                    f"status={summary.get('status', 'ok')}, "
                    f"findings={summary.get('finding_count', 0)}, "
                    f"vram={vram.get('estimated_gb', '?')}GB/{vram.get('available_gb', '?')}GB "
                    f"({vram.get('safety', 'unknown')})"
                )
        except Exception as exc:
            logger.debug("Training advisor report failed: %s", exc)

    def _maybe_apply_auto_vram_enhancement(self) -> None:
        """Apply conservative preflight DiT residency protection before model load."""

        if self.config is None:
            return
        smart_enabled = bool(getattr(self.config, "vram_smart_sensing_enabled", True))
        auto_enabled = bool(getattr(self.config, "vram_auto_enhance_enabled", True))
        if not smart_enabled or not auto_enabled:
            return
        try:
            from .training_advisor import build_training_advisor_report
            from .vram_guardrails import apply_low_vram_guardrails

            report = build_training_advisor_report(self.config).to_dict()
            vram = report.get("vram", {}) if isinstance(report, dict) else {}
            profile = apply_low_vram_guardrails(
                self.config,
                vram_report=vram,
                smart_enabled=smart_enabled,
                auto_enabled=auto_enabled,
            )
            self._low_vram_guardrail_profile = dict(profile)
            changes = dict(profile.get("changes") or {})
            if not changes:
                self._auto_vram_enhancement_profile = {
                    "enabled": bool(profile.get("enabled")),
                    "triggered": bool(profile.get("triggered")),
                    "changes": {},
                    "notes": list(profile.get("notes") or []),
                }
                self._attach_memory_runtime_profiles_to_training_loop()
                return
            self._auto_vram_enhancement_profile = {
                "enabled": True,
                "family": str(profile.get("family") or ""),
                "trigger": "advisor_preflight_low_vram_guardrail",
                "safety": profile.get("safety"),
                "estimated_gb": profile.get("estimated_gb"),
                "available_gb": profile.get("available_gb"),
                "usage_ratio": profile.get("usage_ratio"),
                "changes": dict(changes),
                "hardware": dict(profile.get("hardware") or {}),
                "skipped": list(profile.get("skipped") or []),
                "notes": list(profile.get("notes") or []),
                "recommendations": list(profile.get("recommendations") or []),
                "vram_smart_sensing_enabled": smart_enabled,
                "vram_smart_sensing_streaming_enabled": bool(getattr(self.config, "vram_smart_sensing_streaming_enabled", True)),
                "vram_smart_sensing_sparse_swap_enabled": bool(getattr(self.config, "vram_smart_sensing_sparse_swap_enabled", True)),
                "vram_smart_sensing_delta_cache_enabled": bool(getattr(self.config, "vram_smart_sensing_delta_cache_enabled", False)),
                "enhanced_protection_mode": bool(getattr(self.config, "enhanced_protection_mode", False)),
            }
            self._attach_memory_runtime_profiles_to_training_loop()
            self._log(
                "VRAM auto enhancement applied: "
                f"family={profile.get('family', '')}, safety={profile.get('safety', 'unknown')}, changes={changes}"
            )
        except Exception as exc:
            self._low_vram_guardrail_profile = {
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._auto_vram_enhancement_profile = {
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._attach_memory_runtime_profiles_to_training_loop()
            logger.debug("VRAM auto enhancement skipped: %s", exc)

    def _write_run_manifest(
        self,
        status: str,
        *,
        epoch: int = 0,
        checkpoint_path: str = "",
        state_path: str = "",
    ) -> None:
        """Best-effort run manifest update; never fail training for metadata."""
        try:
            from .distributed import is_main_process
            if not is_main_process():
                return
            from .run_manifest import write_run_manifest

            global_step = int(getattr(self.training_loop, "global_step", 0) or 0) if self.training_loop else 0
            write_run_manifest(
                self.config.output_dir,
                config=self.config,
                status=status,
                epoch=int(epoch or 0),
                global_step=global_step,
                total_steps=int(getattr(self, "_total_steps", 0) or getattr(self.training_loop, "total_steps", 0) or 0),
                steps_per_epoch=int(getattr(self, "_steps_per_epoch", 0) or getattr(self.training_loop, "steps_per_epoch", 0) or 0),
                checkpoint_path=str(checkpoint_path or ""),
                state_path=str(state_path or ""),
                extra=self._run_manifest_extra(),
            )
        except Exception as exc:
            logger.debug("Failed to update run manifest: %s", exc)

    def _run_manifest_extra(self) -> Dict[str, Any]:
        self._refresh_pcie_delta_cache_reports(log=False)
        extra: Dict[str, Any] = {}
        phase_report = self._runtime_phase_report()
        if phase_report.get("phases"):
            extra["runtime_phase_timings"] = phase_report
        optimizer = getattr(getattr(self, "training_loop", None), "optimizer", None)
        dataset = getattr(self, "_dataset", None)
        if dataset is not None and hasattr(dataset, "get_token_bucket_summary"):
            try:
                extra["native_token_buckets"] = dataset.get_token_bucket_summary()
            except Exception as exc:
                extra["native_token_buckets_error"] = f"{type(exc).__name__}: {exc}"
        if dataset is not None and hasattr(dataset, "get_cache_metadata_summary"):
            try:
                extra["cache_metadata"] = dataset.get_cache_metadata_summary()
            except Exception as exc:
                extra["cache_metadata_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_compile_runtime_profile", None):
            extra["compile_runtime"] = dict(self._compile_runtime_profile)
        if getattr(self, "_compile_cache_profile", None):
            extra["compile_cache"] = dict(self._compile_cache_profile)
        extra.update(build_memory_runtime_profiles(self))
        if getattr(self, "_optimizer_backend_profile", None):
            extra["optimizer_backend"] = dict(self._optimizer_backend_profile)
        if getattr(self, "_advanced_optimizer_strategy_profile", None):
            extra["advanced_optimizer_strategy"] = dict(self._advanced_optimizer_strategy_profile)
        if is_anima_full_finetune(self.config) or getattr(self, "_anima_full_finetune_experiments_profile", None):
            anima_experiment_profile = self._refresh_anima_full_finetune_experiments_profile()
            if anima_experiment_profile:
                extra["anima_full_finetune_experiments"] = dict(anima_experiment_profile)
        try:
            from .fp8_te_profile import build_fp8_te_profile

            fp8_profile = build_fp8_te_profile(self.config).as_dict()
            if fp8_profile.get("requested") or str(fp8_profile.get("resolved") or "") == "fp8_te":
                extra["fp8_te_profile"] = fp8_profile
        except Exception as exc:
            extra["fp8_te_profile_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_data_backend_profile", None):
            extra["data_backend"] = dict(self._data_backend_profile)
        data_pipeline = dataloader_attached_data_pipeline_report(getattr(self, "_dataloader", None))
        loop = getattr(self, "training_loop", None)
        if loop is not None and hasattr(loop, "get_memory_experiment_profile"):
            try:
                training_loop_runtime = loop.get_memory_experiment_profile()
                if isinstance(training_loop_runtime, dict) and training_loop_runtime:
                    extra["training_loop_runtime"] = dict(training_loop_runtime)
                    loop_data_pipeline = training_loop_runtime.get("training_data_pipeline")
                    if isinstance(loop_data_pipeline, dict) and loop_data_pipeline:
                        extra["training_data_pipeline"] = merge_lulynx_data_pipeline_reports(
                            data_pipeline,
                            loop_data_pipeline,
                        )
                    loop_orchestrator = training_loop_runtime.get("training_step_orchestrator_runtime")
                    if isinstance(loop_orchestrator, dict) and loop_orchestrator:
                        extra["training_step_orchestrator_runtime"] = dict(loop_orchestrator)
            except Exception as exc:
                extra["training_loop_runtime_error"] = f"{type(exc).__name__}: {exc}"
        multi_batch_dataloader = dataloader_attached_batching_contract(getattr(self, "_dataloader", None))
        if multi_batch_dataloader:
            extra["multi_batch_dataloader"] = multi_batch_dataloader
        if data_pipeline and "training_data_pipeline" not in extra:
            extra["training_data_pipeline"] = data_pipeline
        training_trace = None
        if isinstance(extra.get("training_loop_runtime"), dict):
            training_trace = extra["training_loop_runtime"].get("training_pipeline_trace")
        if isinstance(training_trace, dict):
            extra["multi_batch_promotion_gate"] = build_lulynx_multi_batch_promotion_gate(
                training_pipeline_trace=training_trace,
                multi_batch_dataloader=multi_batch_dataloader,
            )
        readiness = build_lulynx_training_pipeline_execution_readiness(runtime_features=extra)
        if readiness:
            extra["training_pipeline_execution_readiness"] = readiness
            extra["training_step_orchestrator_slice"] = build_lulynx_training_step_orchestrator_slice(
                runtime_features=extra,
                execution_readiness=readiness,
                internal_gate_enabled=False,
                internal_gate_requested=_config_bool(
                    self.config,
                    "training_step_orchestrator_internal_gate_enabled",
                    False,
                ),
            )
        if getattr(self, "_fused_projection_profile", None):
            extra["fused_projection"] = dict(self._fused_projection_profile)
        if getattr(self, "_weight_compression_profile", None):
            extra["weight_compression"] = dict(self._weight_compression_profile)
        if getattr(self, "_lora_activation_recompute_profile", None):
            extra["lora_activation_recompute"] = dict(self._lora_activation_recompute_profile)
        adapter_profile = getattr(self, "_adapter_runtime_profile", None)
        if adapter_profile or self.lora_injector is not None:
            extra["adapter_runtime"] = dict(adapter_profile or self._refresh_adapter_runtime_profile())
        if getattr(self, "_triton_ops_profile", None):
            extra["triton_ops_runtime"] = dict(self._triton_ops_profile)
        if getattr(self, "_attention_runtime_profile", None):
            extra["attention_runtime"] = dict(self._attention_runtime_profile)
        try:
            from .transfer_format import transfer_format_experiment_plan

            reuse_factor = max(
                int(getattr(self, "_total_steps", 0) or 0),
                int(getattr(getattr(self, "training_loop", None), "total_steps", 0) or 0),
                int(getattr(getattr(self, "training_loop", None), "global_step", 0) or 0),
                1,
            )
            extra["pcie_transfer_format_experiment"] = transfer_format_experiment_plan(reuse_factor=float(reuse_factor))
        except Exception as exc:
            extra["pcie_transfer_format_experiment_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from .tensorcore_transfer_kernel import tensorcore_kernel_roadmap

            extra["tensorcore_transfer_kernel"] = tensorcore_kernel_roadmap()
        except Exception as exc:
            extra["tensorcore_transfer_kernel_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_checkpoint_policy_profile", None):
            extra["checkpoint_policy"] = dict(self._checkpoint_policy_profile)
        if getattr(self, "_newbie_cache_first_profile", None):
            extra["newbie_cache_first_profile"] = dict(self._newbie_cache_first_profile)
        if getattr(self, "_anima_cache_builder_profile", None):
            extra["anima_cache_builder_profile"] = dict(self._anima_cache_builder_profile)
        diffusers_cache_profile = getattr(self, "_diffusers_cache_runtime_profile", None)
        if diffusers_cache_profile or self._is_diffusers_unet_route():
            extra["diffusers_cache_runtime"] = dict(
                diffusers_cache_profile or self._refresh_diffusers_cache_runtime_profile()
            )
        training_loop = getattr(self, "training_loop", None)
        native_update_profile_fn = getattr(training_loop, "get_turbocore_native_update_runtime_profile", None)
        if callable(native_update_profile_fn):
            try:
                native_update_profile = native_update_profile_fn()
                if native_update_profile:
                    extra["turbocore_native_update"] = dict(native_update_profile)
            except Exception as exc:
                extra["turbocore_native_update_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_lora_staged_resolution_plan", None):
            extra["lora_staged_resolution"] = {
                "enabled": bool(getattr(self, "_lora_staged_resolution_enabled_runtime", False)),
                "active_index": int(getattr(self, "_lora_staged_resolution_active_index", -1) or -1),
                "sdxl_cache_first": bool(getattr(self, "_lora_staged_resolution_sdxl_cache_first", False)),
                "compile_drop_last": bool(getattr(self, "_lora_staged_resolution_compile_drop_last", False)),
                "stages": stages_to_summary(list(self._lora_staged_resolution_plan)),
            }
        if getattr(self, "_anima_full_finetune_setup", None):
            extra["anima_full_finetune"] = dict(self._anima_full_finetune_setup)
        if getattr(self, "_anima_block_residency_profile", None):
            extra["anima_block_residency"] = dict(self._anima_block_residency_profile)
            controller = getattr(getattr(self.model, "unet", None), "_lulynx_dit_prefetch_controller", None)
            if controller is not None and hasattr(controller, "as_dict"):
                try:
                    extra["anima_block_residency"]["prefetch"] = controller.as_dict()
                except Exception as exc:
                    extra["anima_block_residency"]["prefetch_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_anima_block_checkpoint_profile", None):
            extra["anima_block_checkpointing"] = dict(self._anima_block_checkpoint_profile)
        if getattr(self, "_newbie_block_residency_profile", None):
            extra["newbie_block_residency"] = dict(self._newbie_block_residency_profile)
            controller = getattr(getattr(self.model, "unet", None), "_lulynx_dit_prefetch_controller", None)
            if controller is not None and hasattr(controller, "as_dict"):
                try:
                    extra["newbie_block_residency"]["prefetch"] = controller.as_dict()
                except Exception as exc:
                    extra["newbie_block_residency"]["prefetch_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_newbie_block_checkpoint_profile", None):
            extra["newbie_block_checkpointing"] = dict(self._newbie_block_checkpoint_profile)
        if optimizer is not None and hasattr(optimizer, "get_telemetry_snapshot"):
            try:
                extra["mn_lora"] = optimizer.get_telemetry_snapshot()
            except Exception as exc:
                extra["mn_lora_error"] = f"{type(exc).__name__}: {exc}"
        bubble_action_history = getattr(self.config, "bubble_advisor_action_history", None)
        if isinstance(bubble_action_history, list) and bubble_action_history:
            extra["bubble_advisor_action_history"] = [
                dict(item) for item in bubble_action_history if isinstance(item, dict)
            ]
        closed_loop_action_history = getattr(self.config, "bubble_closed_loop_action_history", None)
        if isinstance(closed_loop_action_history, list) and closed_loop_action_history:
            extra["bubble_closed_loop_action_history"] = [
                dict(item) for item in closed_loop_action_history if isinstance(item, dict)
            ]
        bubble_action_ledger = getattr(self.config, "bubble_advisor_action_ledger", None)
        if isinstance(bubble_action_ledger, dict) and bubble_action_ledger:
            extra["bubble_advisor_action_ledger"] = dict(bubble_action_ledger)
        bubble_closed_loop_state = getattr(self, "_bubble_closed_loop_state", None)
        if isinstance(bubble_closed_loop_state, dict) and bubble_closed_loop_state:
            extra["bubble_closed_loop_state"] = dict(bubble_closed_loop_state)
        bubble_window = self._bubble_closed_loop_window_profile()
        if bubble_window:
            extra["bubble_closed_loop_window"] = bubble_window
        try:
            current_step = int(getattr(getattr(self, "training_loop", None), "global_step", 0) or 0)
            extra["bubble_controller"] = build_bubble_controller_report(
                self.config,
                runtime_features=extra,
                closed_loop_state=bubble_closed_loop_state if isinstance(bubble_closed_loop_state, dict) else None,
                current_step=current_step,
            )
        except Exception as exc:
            extra["bubble_controller_error"] = f"{type(exc).__name__}: {exc}"
        if (
            isinstance(bubble_action_ledger, dict)
            and isinstance(bubble_action_ledger.get("source_report"), dict)
            and isinstance(extra.get("training_loop_runtime"), dict)
        ):
            try:
                from .bubble_runtime_ab_evidence import build_bubble_advisor_ab_evidence_from_run_manifest

                loop = getattr(self, "training_loop", None)
                extra["bubble_advisor_ab_evidence"] = build_bubble_advisor_ab_evidence_from_run_manifest(
                    {
                        "manifest_version": 1,
                        "status": "runtime_snapshot",
                        "global_step": int(getattr(loop, "global_step", 0) or 0),
                        "config": self.config,
                        "extra": extra,
                    }
                )
            except Exception as exc:
                extra["bubble_advisor_ab_evidence_error"] = f"{type(exc).__name__}: {exc}"
        return extra

    def _refresh_pcie_delta_cache_reports(self, *, log: bool = False) -> None:
        if not bool(getattr(self.config, "pcie_delta_cache_enabled", False)):
            return
        unet = getattr(getattr(self, "model", None), "unet", None)
        if unet is None:
            return
        try:
            from .pcie_cache_profiler import build_active_module_pcie_cache_profile
            from .pcie_cache_runtime import collect_pcie_cache_v0_report

            refreshed: list[tuple[str, Dict[str, Any]]] = []
            active_units: list[tuple[str, Any, int]] = []
            named_modules = getattr(unet, "named_modules", None)
            if callable(named_modules):
                for name, module in named_modules():
                    if not bool(getattr(module, "lulynx_weight_residency_active", False)):
                        continue
                    if not callable(getattr(module, "get_transfer_format_stats", None)):
                        continue
                    try:
                        parameter_count = sum(int(param.numel()) for param in module.parameters(recurse=False))
                    except Exception:
                        parameter_count = 0
                    active_units.append((str(name), module, int(parameter_count)))
            cache_mode = str(getattr(self.config, "pcie_delta_cache_mode", "observe") or "observe")
            cache_budget = max(float(getattr(self.config, "pcie_delta_cache_budget_mb", 256.0) or 0.0), 0.0)
            if isinstance(getattr(self, "_anima_block_residency_profile", None), dict) and self._anima_block_residency_profile.get("mode"):
                mode = str(self._anima_block_residency_profile.get("mode", ""))
                profile = build_active_module_pcie_cache_profile(unet, enabled=True, family="anima", mode=mode).as_dict()
                profile["reason"] = "observe_only_post_training"
                self._anima_block_residency_profile["pcie_delta_cache"] = profile
                self._anima_block_residency_profile["pcie_cache_v0_recommendation"] = self._build_pcie_cache_v0_recommendation(
                    profile,
                    prefetch=self._anima_block_residency_profile.get("prefetch"),
                    cache_budget_mb=cache_budget,
                )
                self._anima_block_residency_profile["pcie_cache_v0"] = collect_pcie_cache_v0_report(
                    active_units,
                    mode=cache_mode,
                    budget_mb=cache_budget,
                    reason="post_training",
                ).as_dict()
                refreshed.append(("Anima", profile))
            if isinstance(getattr(self, "_newbie_block_residency_profile", None), dict) and self._newbie_block_residency_profile.get("mode"):
                mode = str(self._newbie_block_residency_profile.get("mode", ""))
                profile = build_active_module_pcie_cache_profile(unet, enabled=True, family="newbie", mode=mode).as_dict()
                profile["reason"] = "observe_only_post_training"
                self._newbie_block_residency_profile["pcie_delta_cache"] = profile
                self._newbie_block_residency_profile["pcie_cache_v0_recommendation"] = self._build_pcie_cache_v0_recommendation(
                    profile,
                    prefetch=self._newbie_block_residency_profile.get("prefetch"),
                    cache_budget_mb=cache_budget,
                )
                self._newbie_block_residency_profile["pcie_cache_v0"] = collect_pcie_cache_v0_report(
                    active_units,
                    mode=cache_mode,
                    budget_mb=cache_budget,
                    reason="post_training",
                ).as_dict()
                refreshed.append(("Newbie", profile))
            if isinstance(getattr(self, "_native_weight_residency_profile", None), dict) and self._native_weight_residency_profile.get("mode"):
                mode = str(self._native_weight_residency_profile.get("mode", ""))
                profile = build_active_module_pcie_cache_profile(unet, enabled=True, family="sdxl", mode=mode).as_dict()
                profile["reason"] = "observe_only_post_training"
                self._native_weight_residency_profile["pcie_delta_cache"] = profile
                self._native_weight_residency_profile["pcie_cache_v0_recommendation"] = self._build_pcie_cache_v0_recommendation(
                    profile,
                    prefetch=self._native_weight_residency_profile.get("prefetch"),
                    cache_budget_mb=cache_budget,
                )
                self._native_weight_residency_profile["pcie_cache_v0"] = collect_pcie_cache_v0_report(
                    active_units,
                    mode=cache_mode,
                    budget_mb=cache_budget,
                    reason="post_training",
                ).as_dict()
                native_status = getattr(self, "_native_unet_status", None)
                if isinstance(native_status, dict):
                    weight_residency = native_status.get("weight_residency")
                    if isinstance(weight_residency, dict):
                        weight_residency["pcie_delta_cache"] = profile
                refreshed.append(("Native SDXL", profile))
            if log and refreshed and not bool(getattr(self, "_pcie_delta_cache_summary_logged", False)):
                for label, profile in refreshed:
                    summary = str(profile.get("summary_text") or "")
                    if summary:
                        self._log(f"{label} {summary}")
                    residency_profile = (
                        self._anima_block_residency_profile if label == "Anima"
                        else self._newbie_block_residency_profile if label == "Newbie"
                        else self._native_weight_residency_profile
                    )
                    cache_v0 = residency_profile.get("pcie_cache_v0", {}) if isinstance(residency_profile, dict) else {}
                    if cache_v0:
                        self._log(
                            f"{label} PCIe Cache v0: "
                            f"enabled={bool(cache_v0.get('enabled'))} "
                            f"selected={cache_v0.get('selected_count', 0)} "
                            f"cache={float(cache_v0.get('cache_mb', 0.0) or 0.0):.1f}MB "
                            f"hits={cache_v0.get('hit_count', 0)} "
                            f"misses={cache_v0.get('miss_count', 0)} "
                            f"errors={cache_v0.get('error_count', 0)} "
                            f"budget={float(cache_v0.get('budget_mb', 0.0) or 0.0):.1f}MB "
                            f"reason={cache_v0.get('reason', '')}"
                        )
                    recommendation = residency_profile.get("pcie_cache_v0_recommendation", {}) if isinstance(residency_profile, dict) else {}
                    if recommendation:
                        self._log(f"{label} {recommendation.get('summary_text', 'PCIe Cache v0 recommendation: observe')}")
                self._pcie_delta_cache_summary_logged = True
            if refreshed:
                self._attach_memory_runtime_profiles_to_training_loop()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for attr_name in ("_anima_block_residency_profile", "_newbie_block_residency_profile", "_native_weight_residency_profile"):
                profile = getattr(self, attr_name, None)
                if isinstance(profile, dict):
                    profile["pcie_delta_cache_refresh_error"] = message
            self._attach_memory_runtime_profiles_to_training_loop()
            if log:
                self._log(f"PCIe Delta/Cache observe refresh skipped: {message}")

    def _build_pcie_cache_v0_recommendation(
        self,
        profile: Dict[str, Any],
        *,
        prefetch: Optional[Dict[str, Any]] = None,
        cache_budget_mb: float = 256.0,
    ) -> Dict[str, Any]:
        """Recommend cache_v0 only. This never flips pcie_delta_cache_mode."""

        if not isinstance(profile, dict) or not profile:
            return {"action": "keep_observing", "reason": "no_profile"}
        high = int(profile.get("high_value_count", 0) or 0)
        candidates = int(profile.get("candidate_count", 0) or 0)
        transfer_mb = float(profile.get("total_transfer_mb", 0.0) or 0.0)
        miss_total = int(profile.get("prefetch_missed_total", 0) or 0)
        errors = int(profile.get("error_count", 0) or 0)
        prefetch_enabled = bool((prefetch or {}).get("enabled", False))
        prefetch_missed = int((prefetch or {}).get("missed", (prefetch or {}).get("missed_total", 0)) or 0)
        prefetch_submitted = int((prefetch or {}).get("submitted", (prefetch or {}).get("submitted_total", 0)) or 0)
        prefetch_consumed = int((prefetch or {}).get("consumed", (prefetch or {}).get("consumed_total", 0)) or 0)
        perfect_prefetch = prefetch_enabled and prefetch_submitted > 0 and prefetch_consumed > 0 and prefetch_missed <= 0
        reuse_factor = max(
            int(getattr(self, "_total_steps", 0) or 0),
            int(getattr(getattr(self, "training_loop", None), "total_steps", 0) or 0),
            int(getattr(getattr(self, "training_loop", None), "global_step", 0) or 0),
            1,
        )
        amortized_transfer_mb = transfer_mb / max(float(reuse_factor), 1.0)
        current_mode = str(getattr(self.config, "pcie_delta_cache_mode", "observe") or "observe")
        recommendation = {
            "enabled": True,
            "action": "recommend_only",
            "mode": "cache_v0",
            "suggested_budget_mb": round(max(float(cache_budget_mb or 0.0), 0.0), 1),
            "candidate_count": candidates,
            "high_value_count": high,
            "total_transfer_mb": round(transfer_mb, 3),
            "prefetch_enabled": prefetch_enabled,
            "prefetch_missed": prefetch_missed,
            "profile_prefetch_missed": miss_total,
            "reuse_factor": int(reuse_factor),
            "amortized_transfer_mb_per_step": round(float(amortized_transfer_mb), 4),
            "pack_amortization_note": "FP8 pack is a one-time cost for frozen CPU-pinned weights and should be judged across repeated training steps.",
            "current_mode": current_mode,
            "will_auto_enable": False,
        }
        if errors > 0:
            recommendation.update({"decision": "do_not_try_yet", "reason": "transfer_errors_present"})
        elif perfect_prefetch:
            recommendation.update({"decision": "not_recommended", "reason": "prefetch_already_covers_transfer"})
        elif high >= 8 and transfer_mb >= 128.0 and (miss_total > 0 or prefetch_missed > 0 or not prefetch_enabled):
            recommendation.update({"decision": "try_manually", "reason": "large_repeated_cpu_pinned_transfers"})
        elif candidates > 0:
            recommendation.update({"decision": "keep_observing", "reason": "candidate_signal_not_strong_enough"})
        else:
            recommendation.update({"decision": "not_recommended", "reason": "no_cache_candidate"})
        recommendation["summary_text"] = (
            "PCIe Cache v0 recommendation: "
            f"decision={recommendation['decision']} reason={recommendation['reason']} "
            f"budget={recommendation['suggested_budget_mb']:.1f}MB auto=False"
        )
        return recommendation

    def _preflight_resume_manifest(self) -> None:
        """Log run-manifest resume checks before loading optimizer state."""
        resume_path = str(getattr(self.config, "resume_path", "") or "").strip()
        if not resume_path:
            return
        try:
            from .run_manifest import validate_resume_manifest

            report = validate_resume_manifest(resume_path, config=self.config)
            if report.found:
                self._log(
                    f"Resume manifest found: {report.manifest_path} "
                    f"(step={report.previous_global_step}, status={report.previous_status or 'unknown'})"
                )
            else:
                self._log(f"Resume manifest missing: {report.manifest_path}; legacy resume compatibility mode.")
            for warning in report.warnings:
                self._log(f"Resume manifest warning: {warning}")
            for error in report.errors:
                self._log(f"Resume manifest error: {error}")
            for note in report.notes:
                self._log(f"Resume manifest: {note}")
        except Exception as exc:
            self._log(f"Resume manifest preflight failed: {type(exc).__name__}: {exc}")

    # GPU thermal / power-limit / epoch-cooldown helpers live in
    # TrainerThermalMixin (trainer_thermal.py); they remain bound methods of the
    # trainer via inheritance with identical behaviour and call sites.

    # ------------------------------------------------------------------
    # R2 state-dict methods moved to trainer_artifact_io.TrainerArtifactIoMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _initialize_ema_tracker(self):
        self._ema_tracker = None

        if not self._ema_requested():
            return

        if getattr(self.config, "semantic_tuner_enabled", False):
            self._log("EMA requested, but Semantic Base-Tuner mode currently skips native EMA shadow weights.")
            return

        adapter_state = self._get_current_adapter_state_dict(use_ema=False)
        if not adapter_state:
            self._log("EMA requested, but no adapter state is available yet.")
            return

        from .ema import EMAStateTracker

        self._ema_tracker = EMAStateTracker(
            initial_state=adapter_state,
            decay=float(getattr(self.config, "ema_decay", 0.999) or 0.999),
            update_after_step=int(getattr(self.config, "ema_update_after_step", 0) or 0),
            update_every=max(int(getattr(self.config, "ema_update_every", 1) or 1), 1),
            device="cpu",
            use_ema_warmup=bool(getattr(self.config, "ema_use_ema_warmup", False)),
            inv_gamma=float(getattr(self.config, "ema_inv_gamma", 1.0) or 1.0),
            power=float(getattr(self.config, "ema_power", 0.666) or 0.666),
        )
        self._log(
            f"EMA enabled (decay={self._ema_tracker.decay}, every={self._ema_tracker.update_every}, warmup={self._ema_tracker.use_ema_warmup})"
        )

    def _initialize_resource_manager(self):
        self._resource_manager = None
        self._last_vram_status = "ok"

        adaptive_requested = bool(
            getattr(self.config, "rm_enable_adaptive_accumulation", False)
            or getattr(self.config, "rm_enable_adaptive_batch", False)
        )
        current_thresholds = (
            float(getattr(self.config, "rm_vram_warning_threshold", 0.85) or 0.85),
            float(getattr(self.config, "rm_vram_critical_threshold", 0.92) or 0.92),
            float(getattr(self.config, "rm_vram_emergency_threshold", 0.97) or 0.97),
        )
        thresholds_overridden = current_thresholds != (0.85, 0.92, 0.97)

        if not adaptive_requested and not thresholds_overridden:
            return

        if not torch.cuda.is_available():
            self._log("ResourceManager skipped: CUDA unavailable.")
            return

        from .resource_manager import DynamicResourceManager, ResourceConfig

        if getattr(self.config, "rm_enable_adaptive_batch", False):
            self._log("ResourceManager: adaptive batch is deferred; native trainer currently applies VRAM monitoring + adaptive accumulation only.")

        resource_config = ResourceConfig(
            vram_warning_threshold=float(getattr(self.config, "rm_vram_warning_threshold", 0.85) or 0.85),
            vram_critical_threshold=float(getattr(self.config, "rm_vram_critical_threshold", 0.92) or 0.92),
            vram_emergency_threshold=float(getattr(self.config, "rm_vram_emergency_threshold", 0.97) or 0.97),
            enable_adaptive_batch=False,
            min_batch_size=max(int(getattr(self.config, "rm_min_batch_size", 1) or 1), 1),
            max_batch_size=max(int(getattr(self.config, "rm_max_batch_size", getattr(self.config, "batch_size", 1)) or getattr(self.config, "batch_size", 1)), 1),
            enable_adaptive_accumulation=bool(getattr(self.config, "rm_enable_adaptive_accumulation", False)),
            min_accumulation=max(int(getattr(self.config, "rm_min_accumulation", 1) or 1), 1),
            max_accumulation=max(int(getattr(self.config, "rm_max_accumulation", getattr(self.config, "gradient_accumulation_steps", 1)) or getattr(self.config, "gradient_accumulation_steps", 1)), 1),
            cache_clear_interval=max(int(getattr(self.config, "rm_cache_clear_interval", 100) or 100), 1),
        )
        self._resource_manager = DynamicResourceManager(resource_config)
        self._resource_manager.current_batch_size = max(int(getattr(self.config, "batch_size", 1) or 1), 1)
        self._resource_manager.current_accumulation = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
        self._log(
            f"ResourceManager enabled (warn={resource_config.vram_warning_threshold:.0%}, critical={resource_config.vram_critical_threshold:.0%}, adaptive_accum={resource_config.enable_adaptive_accumulation})"
        )

    def _build_lulynx_baseline_inputs(self) -> Dict[str, torch.Tensor]:
        """Build a minimal UNet input dict for wrapper baseline capture."""
        width, height = self._get_resolution_pair()
        latent_noise = torch.randn(1, 4, height // 8, width // 8, device=self.device, dtype=self.dtype)
        timestep = torch.full(
            (1,),
            self.model.noise_scheduler.config.num_train_timesteps // 2,
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            prompt_embeds = self.training_loop._encode_prompt([""])
            time_embeds = self.training_loop._get_timestep_embedding(
                1,
                [(width, height)],
                [(width, height)],
                [(0, 0, width, height)],
            )

        baseline_inputs = {
            "sample": latent_noise,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds["encoder_hidden_states"],
        }

        if time_embeds:
            baseline_inputs["added_cond_kwargs"] = {
                **time_embeds.get("added_cond_kwargs", {}),
                "text_embeds": prompt_embeds.get("pooled_prompt_embeds"),
            }

        return baseline_inputs

    def _apply_local_compile_cache(self, *, route: str, model_path: str) -> None:
        if not bool(getattr(self.config, "compile_cache_enabled", True)):
            self.compile_cache_layout = None
            self._compile_cache_profile = {
                "enabled": False,
                "state": "disabled",
                "reason": "compile_cache_enabled=false",
            }
            return
        if not bool(getattr(self.config, "torch_compile", False)) and str(
            getattr(self.config, "anima_compile_scope", "") or ""
        ).strip().lower() not in {"per_block", "full", "full_core", "full_cudagraph"}:
            self.compile_cache_layout = None
            self._compile_cache_profile = {
                "enabled": False,
                "state": "inactive",
                "reason": "torch_compile/anima_compile_scope not active",
            }
            return

        layout = build_compile_cache_layout(
            self.config,
            route=route,
            model_path=model_path,
            device=self.device,
        )
        status_before = compile_cache_status(layout)
        blocker = compile_cache_cold_bucket_blocker(layout)
        if blocker is not None:
            self.compile_cache_layout = layout
            self._compile_cache_profile = compile_cache_profile(
                layout,
                status_before=status_before,
                reuse=bool(getattr(self.config, "compile_cache_reuse", True)),
                blocker=blocker,
            )
            self._log(
                "[compile-cache][warn] "
                f"cold-bucket cache takeover skipped: {blocker}. "
                "Existing default runtime caches may still be used."
            )
            return
        reuse = bool(getattr(self.config, "compile_cache_reuse", True))
        env_updates = prepare_compile_cache_environment(
            layout,
            reuse=reuse,
        )
        self.compile_cache_layout = layout
        status_after = compile_cache_status(layout)
        self._compile_cache_profile = compile_cache_profile(
            layout,
            status_before=status_before,
            status_after=status_after,
            reuse=reuse,
            env_updates=env_updates,
        )
        for line in layout.log_lines():
            self._log(line)
        hit_text = "hit" if status_before["hit"] else "miss"
        self._log(
            "[compile-cache] "
            f"status={hit_text} manifest={'yes' if status_before['manifest_exists'] else 'no'} "
            f"inductor_files={status_before['inductor_files']} triton_files={status_before['triton_files']}"
        )

    def _compile_probe_cuda_devices(self) -> List[int]:
        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            return []
        try:
            device = torch.device(self.device)
            if device.index is not None:
                return [int(device.index)]
        except Exception:
            pass
        try:
            return [int(torch.cuda.current_device())]
        except Exception:
            return []

    def _set_anima_full_core_variant(self, variant: str) -> bool:
        if self.model is None or getattr(self.model, "unet", None) is None:
            return False
        if variant == "compiled":
            target = self._anima_full_core_compiled_run_blocks
        else:
            target = self._anima_full_core_original_run_blocks
        if target is None:
            return False
        setattr(self.model.unet, "_run_blocks", target)
        return True

    def _run_anima_compile_probe_microstep(self, batch: Dict[str, Any]) -> float:
        if self.training_loop is None:
            raise RuntimeError("Anima compile probe requires an initialized TrainingLoop")
        if self.training_loop.optimizer is None:
            raise RuntimeError("Anima compile probe requires an initialized optimizer")

        self.training_loop.optimizer.zero_grad(set_to_none=True)
        loss = self.training_loop._train_step_impl(
            batch,
            accumulation_steps=1,
            do_backward=True,
            return_loss_tensor=False,
        )
        trainable_params = self.training_loop._get_trainable_params()
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, self.training_loop.max_grad_norm)
        self.training_loop.optimizer.zero_grad(set_to_none=True)
        return float(loss)

    def _measure_anima_compile_probe_training_microstep(
        self,
        batch: Dict[str, Any],
        *,
        steps: int,
        warmup_steps: int = 1,
    ) -> tuple[float, float]:
        if self.training_loop is None:
            raise RuntimeError("Anima compile probe requires an initialized TrainingLoop")

        steps = max(int(steps or 1), 1)
        warmup_steps = max(int(warmup_steps or 0), 0)
        device_type = torch.device(self.device).type
        old_scope = str(getattr(self.training_loop, "_anima_compile_scope", "") or "")
        had_scope_cached = hasattr(self.training_loop, "_cudagraph_scope_cached")
        old_scope_cached = getattr(self.training_loop, "_cudagraph_scope_cached", None)
        old_cudagraph_capture = getattr(self.training_loop, "_cudagraph_capture", None)
        old_cudagraph_active = bool(getattr(self.training_loop, "_cudagraph_active", False))

        try:
            self.training_loop._anima_compile_scope = ""
            if had_scope_cached:
                delattr(self.training_loop, "_cudagraph_scope_cached")
            self.training_loop._cudagraph_capture = None
            self.training_loop._cudagraph_active = False

            if torch.cuda.is_available() and device_type == "cuda":
                self._maybe_release_tool_cuda_cache(
                    "anima_compile_probe_prepare",
                    force=True,
                    collect_gc=True,
                    synchronize=True,
                )

            for _ in range(warmup_steps):
                self._run_anima_compile_probe_microstep(batch)

            peak_vram = 0.0
            if torch.cuda.is_available() and device_type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            for _ in range(steps):
                self._run_anima_compile_probe_microstep(batch)
            if torch.cuda.is_available() and device_type == "cuda":
                torch.cuda.synchronize()
                peak_vram = float(torch.cuda.max_memory_allocated())

            return time.perf_counter() - start, peak_vram
        finally:
            if self.training_loop.optimizer is not None:
                self.training_loop.optimizer.zero_grad(set_to_none=True)
            self.training_loop._anima_compile_scope = old_scope
            if had_scope_cached:
                self.training_loop._cudagraph_scope_cached = old_scope_cached
            self.training_loop._cudagraph_capture = old_cudagraph_capture
            self.training_loop._cudagraph_active = old_cudagraph_active

    def _maybe_probe_anima_full_core_compile(self, dataloader: Any) -> None:
        if (
            self.model is None
            or getattr(self.model, "unet", None) is None
            or self.runtime_optimization_plan is None
            or self.compile_contract_decision is None
        ):
            return
        if self._model_arch_value() != "anima":
            return
        if getattr(self.compile_contract_decision, "resolved", "") != "full_core":
            return
        if not bool(getattr(self.config, "compile_probe_enabled", True)):
            return
        if self._anima_full_core_original_run_blocks is None or self._anima_full_core_compiled_run_blocks is None:
            self._log("[compile-probe][warn] Anima full-core probe skipped: compiled target was not prepared")
            return

        probe_steps = max(int(getattr(self.config, "compile_probe_steps", 3) or 3), 1)
        seed = int(getattr(self.config, "seed", 1337) or 1337)
        eager_seconds = 0.0
        eager_peak_vram = 0.0
        compiled_seconds = 0.0
        compiled_peak_vram = 0.0
        failed_reason = ""

        with torch.random.fork_rng(devices=self._compile_probe_cuda_devices(), enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                probe_batch = next(iter(dataloader))
                if not self._set_anima_full_core_variant("original"):
                    raise RuntimeError("missing eager _run_blocks target")
                eager_seconds, eager_peak_vram = self._measure_anima_compile_probe_training_microstep(
                    probe_batch,
                    steps=probe_steps,
                    warmup_steps=1,
                )
                if not self._set_anima_full_core_variant("compiled"):
                    raise RuntimeError("missing compiled _run_blocks target")
                compiled_seconds, compiled_peak_vram = self._measure_anima_compile_probe_training_microstep(
                    probe_batch,
                    steps=probe_steps,
                    warmup_steps=1,
                )
            except Exception as exc:
                failed_reason = str(exc)

        result = evaluate_compile_probe(
            route="anima",
            target="_run_blocks",
            eager_seconds=eager_seconds,
            compiled_seconds=compiled_seconds,
            eager_peak_vram=eager_peak_vram,
            compiled_peak_vram=compiled_peak_vram,
            min_speedup_ratio=float(getattr(self.config, "compile_probe_min_speedup_ratio", 0.03) or 0.03),
            max_vram_increase_ratio=float(
                getattr(self.config, "compile_probe_max_vram_increase_ratio", 0.15) or 0.15
            ),
            failed_reason=failed_reason,
        )
        self._anima_full_core_probe_result = result
        for line in result.log_lines():
            self._log(line)

        if result.keep:
            self._set_anima_full_core_variant("compiled")
            self._log("[compile-contract] Anima full-core compile kept after live probe")
            return

        self._set_anima_full_core_variant("original")
        self._log(
            "[compile-contract][warn] Anima full-core compile disabled after live probe; "
            "falling back to per-block compile"
        )

        self.runtime_optimization_plan.torch_compile = True
        self.runtime_optimization_plan.torch_compile_scope = "per_block"
        self.runtime_optimization_plan.anima_compile_scope = "per_block"
        self.runtime_optimization_plan.warnings.append(
            f"full_core_compile: disabled after probe ({result.reason}); falling back to per_block"
        )
        self.config.torch_compile = True
        self.config.torch_compile_scope = "per_block"
        self.config.anima_compile_scope = "per_block"
        self.compile_contract_decision.resolved = "per_block"
        self.compile_contract_decision.reasons.append(
            f"Anima full_core compile downgraded to per_block after live probe: {result.reason}"
        )
        self._refresh_compile_runtime_profile(
            model_arch="anima",
            applied=False,
            compile_kind="full_core_probe",
            source="full_core_probe",
            skip_reason=f"probe_rejected: {result.reason}",
        )
        self._attach_compile_runtime_profile_to_training_loop()
        if self.training_loop is not None:
            self.training_loop._anima_compile_scope = "per_block"
            if hasattr(self.training_loop, "_cudagraph_scope_cached"):
                delattr(self.training_loop, "_cudagraph_scope_cached")
            self.training_loop._cudagraph_capture = None
            self.training_loop._cudagraph_active = False
        self._maybe_apply_per_block_compile()

    def _maybe_apply_per_block_compile(self) -> None:
        if (
            self.model is None
            or self.runtime_optimization_plan is None
            or self.compile_contract_decision is None
        ):
            return
        if self._per_block_compile_applied:
            return
        if getattr(self.compile_contract_decision, "resolved", "") != "per_block":
            return
        before_reasons = len(getattr(self.runtime_optimization_plan, "reasons", []) or [])
        before_warnings = len(getattr(self.runtime_optimization_plan, "warnings", []) or [])
        apply_per_block_compile(
            self.model,
            self.runtime_optimization_plan,
            route=self._model_arch_value(),
        )
        self._per_block_compile_applied = True
        compiled_targets = 0
        emitted = False
        for reason in self.runtime_optimization_plan.reasons[before_reasons:]:
            if "per_block_compile:" in reason:
                emitted = True
                if reason.startswith("per_block_compile: compiled "):
                    compiled_targets += 1
                self._log(f"[runtime-opt] {reason}")
        for warning in self.runtime_optimization_plan.warnings[before_warnings:]:
            if "per_block_compile:" in warning or "full_core_compile:" in warning:
                emitted = True
                self._log(f"[runtime-opt][warn] {warning}")
        target_profile = build_compile_target_profile(
            self.model,
            self.runtime_optimization_plan,
            route=self._model_arch_value(),
            reasons_start=before_reasons,
            warnings_start=before_warnings,
        )
        target_profile.update({
            "resolved": getattr(self.compile_contract_decision, "resolved", ""),
            "effective_static_shape_source": self._resolve_compile_static_shape_source(),
            "compiled_target_messages": int(target_profile.get("compiled_targets", compiled_targets) or 0),
            "warning_count": max(len(getattr(self.runtime_optimization_plan, "warnings", []) or []) - before_warnings, 0),
        })
        self._refresh_compile_runtime_profile(
            model_arch=self._model_arch_value(),
            target_profile=target_profile,
            compiled_targets=int(target_profile.get("compiled_targets", compiled_targets) or 0),
            compile_kind="per_block",
            source="per_block_compile",
        )
        self._attach_compile_runtime_profile_to_training_loop()
        if emitted:
            self._mark_runtime_phase("per_block_compile")

    @classmethod
    def from_frontend_dict(cls, data: dict) -> "LulynxTrainer":
        """从前端配置创建"""
        config = ConfigAdapter.from_frontend_dict(data)
        return cls(config)

    def configure(self, config: LulynxConfig):
        """设置配置"""
        self.config = config
        self.amd_runtime_guard = build_amd_runtime_guard(config)
        apply_amd_runtime_guard(config, self.amd_runtime_guard)
        self.mps_runtime_guard = build_mps_runtime_guard(config)
        apply_mps_runtime_guard(config, self.mps_runtime_guard)
        requested_device = str(getattr(config, "device", "") or "").strip().lower()
        if requested_device == "mps":
            self.device = "mps"

        # 更新精度设置
        if config.mixed_precision == "bf16" and self._supports_bf16():
            self.dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def _should_enable_fixed_token_padding(self) -> bool:
        if bool(getattr(self.config, "enable_fixed_token_padding", False)):
            return True
        if not bool(getattr(self.config, "torch_compile", False)):
            return False
        shape_strategy = "auto"
        if getattr(self, "runtime_optimization_plan", None) is not None:
            shape_strategy = str(
                getattr(self.runtime_optimization_plan, "compile_shape_strategy", "auto") or "auto"
            ).strip().lower()
        else:
            shape_strategy = str(getattr(self.config, "compile_shape_strategy", "auto") or "auto").strip().lower()
        return shape_strategy not in {"token_flatten", "native"}

    def _resolve_compile_static_shape_source(self) -> str:
        plan = getattr(self, "runtime_optimization_plan", None)
        if plan is None:
            return "unknown"
        if not bool(getattr(plan, "torch_compile", False)):
            return "disabled"

        shape_strategy = str(getattr(plan, "compile_shape_strategy", "auto") or "auto").strip().lower()
        if shape_strategy in {"token_flatten", "native"}:
            summary = getattr(self, "_native_token_bucket_summary", None)
            if isinstance(summary, dict) and summary.get("error"):
                return "native_token_buckets_error"
            return "native_token_buckets"

        model_arch = self._model_arch_value()
        token_attr = "anima_fixed_visual_tokens" if model_arch == "anima" else "newbie_fixed_visual_tokens"
        try:
            fixed_visual_tokens = int(getattr(self.config, token_attr, 0) or 0)
        except Exception:
            fixed_visual_tokens = 0
        if fixed_visual_tokens > 0:
            return "fixed_visual_tokens"
        if self._should_enable_fixed_token_padding():
            return "fixed_token_padding"
        return "dynamic_or_unknown"

    def prepare(self):
        """准备训练"""
        if not self.config:
            raise ValueError("Config not set")
        self._reset_runtime_phase_timings()
        self.amd_runtime_guard = build_amd_runtime_guard(self.config)
        apply_amd_runtime_guard(self.config, self.amd_runtime_guard)
        for line in build_amd_banner_lines(self.amd_runtime_guard):
            self._log(line)
        self.mps_runtime_guard = build_mps_runtime_guard(self.config)
        apply_mps_runtime_guard(self.config, self.mps_runtime_guard)
        if str(getattr(self.config, "device", "") or "").strip().lower() == "mps":
            self.device = "mps"
            self.dtype = torch.float32
        for line in build_mps_banner_lines(self.mps_runtime_guard):
            self._log(line)

        model_arch = self._model_arch_value()
        self._apply_native_runtime_profile()
        self._apply_native_cache_mode()
        self._apply_sdxl_lora_low_vram_profile(model_arch)
        self._apply_runtime_env_hints()
        self._mark_runtime_phase("runtime_profile")
        model_path = self.config.base_model_path
        vae_path = getattr(self.config, "vae_path", "") or None

        from .model_family import get_model_family
        _family = get_model_family(model_arch)

        if model_arch == "anima":
            model_path = str(getattr(self.config, "anima_model_path", "") or model_path)
            self._log("Loading Anima base model...")
        elif model_arch == "newbie":
            model_path = str(
                getattr(self.config, "newbie_diffusers_path", "")
                or getattr(self.config, "newbie_transformer_path", "")
                or model_path
            )
            self._log("Loading Newbie base model...")
        else:
            self._log("Loading base model...")

        self._apply_local_compile_cache(route=model_arch, model_path=str(model_path or ""))
        self._mark_runtime_phase("compile_cache_setup")

        training_type = getattr(self.config, "training_type", "lora")
        if training_type == "textual_inversion":
            # TI owns model loading because it needs a tokenizer that supports add_tokens().
            # Loading the generic base first would keep a duplicate SDXL copy on GPU.
            self._log("Textual Inversion mode: loading TI components without generic base pre-load")
            from .textual_inversion import TextualInversionTrainer, TextualInversionConfig
            ti_config = TextualInversionConfig(
                model_path=self.config.base_model_path,
                model_type=model_arch,
                placeholder_token=getattr(self.config, "ti_placeholder_token", "<new>"),
                initializer_token=getattr(self.config, "ti_init_token", "") or "person",
                num_vectors=getattr(self.config, "ti_num_vectors", 1),
                learning_rate=self.config.learning_rate,
                max_train_steps=getattr(self.config, "max_train_steps", 1000),
                mixed_precision=getattr(self.config, "mixed_precision", "fp16"),
            )
            self._ti_trainer = TextualInversionTrainer(ti_config)
            self._ti_trainer.load_model()
            ti_modules = [self._ti_trainer.model, self._ti_trainer.text_encoder, self._ti_trainer.vae]
            if getattr(self._ti_trainer, "text_encoder_2", None) is not None:
                self._ti_trainer.text_encoder_2.requires_grad_(False)
                ti_modules.append(self._ti_trainer.text_encoder_2)
            for module in ti_modules:
                module.to(device=self.device, dtype=self.dtype)
            self._ti_trainer._init_concept_embedding()
            self._ti_trainer.prepare_optimizer()
            self.trainable_params = list(self._ti_trainer.concept_embedding.parameters())
            self.lora_injector = _FullFinetuneParamWrapper(self.trainable_params)
            self._ti_mode = True
            self._log("Textual Inversion trainer ready")
            self._mark_runtime_phase("textual_inversion_prepare")
            return self
        # 加载模型
        loader = ModelLoader(device=self.device, dtype=self.dtype)
        self.runtime_optimization_plan = build_runtime_optimization_plan(self.config)
        self.compile_contract_decision = resolve_compile_contract(
            self.config,
            self.runtime_optimization_plan,
            model_arch=model_arch,
        )
        self._refresh_compile_runtime_profile(
            model_arch=model_arch,
            applied=False,
            source="compile_contract",
        )
        for line in self.runtime_optimization_plan.log_lines():
            self._log(line)
        for line in self.compile_contract_decision.log_lines():
            self._log(line)
        if model_arch == "anima":
            from .anima_loader import load_anima_model

            self.model, anima_report = load_anima_model(
                model_path=model_path,
                qwen3_path=str(getattr(self.config, "anima_qwen3_path", "") or ""),
                t5_tokenizer_path=str(getattr(self.config, "anima_t5_tokenizer_path", "") or ""),
                attn_mode=str(getattr(self.config, "anima_attn_mode", "") or ""),
                vae_path=str(vae_path or ""),
                llm_adapter_path=str(getattr(self.config, "anima_llm_adapter_path", "") or ""),
                dit_adapter_path=str(getattr(self.config, "anima_dit_adapter_path", "") or ""),
                device=self.device,
                dtype=self.dtype,
                disable_mmap=bool(getattr(self.config, "disable_mmap_load_safetensors", False)),
            )
            self._log(f"Anima native loader: {anima_report.summary()}")
            self._mark_runtime_phase("anima_model_load")
            self._prepare_anima_cached_executable(model_path)
            self._mark_runtime_phase("anima_native_executable")
        elif model_arch == "newbie":
            self._prepare_newbie_cache_first_runtime()
            self._mark_runtime_phase("newbie_cache_first_prepare")
            if self.model is None:
                from .newbie_loader import load_newbie_from_config

                self.model = load_newbie_from_config(
                    self.config,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._mark_runtime_phase("newbie_model_load")
                if getattr(self.model, "newbie_scaffold_mode", False):
                    self._log(
                        "Newbie native loader is currently running in SDXL-compatible scaffold mode; "
                        "Gemma3-specific secondary text encoding is not wired yet."
                    )
            if self.model is not None and not bool(getattr(self.model, "newbie_forward_smoke_passed", False)):
                from .newbie_smoke import run_loaded_newbie_smoke

                smoke = run_loaded_newbie_smoke(
                    self.model,
                    target_modules=self._get_custom_target_modules(),
                    latent_size=4,
                )
                if not smoke.passed:
                    raise RuntimeError(f"Newbie transformer smoke failed: {smoke.reason}")
                self._log(
                    "Newbie transformer smoke passed: "
                    f"latent={smoke.latent_shape}, targets={list(smoke.gradient_targets)}"
                )
                self._mark_runtime_phase("newbie_transformer_smoke")
        else:
            self.model = loader.load(
                model_path,
                model_arch,
                vae_path,
            )
            self._mark_runtime_phase(f"{model_arch}_model_load")

        if model_arch in {"anima", "newbie"}:
            self._ensure_native_family_training_ready()
            self._mark_runtime_phase("native_family_ready")

        # Diffusers UNet Low VRAM Profile: apply after model load so VAE/UNet are available
        if self._is_diffusers_unet_route():
            self._apply_sdxl_low_vram_profile()
            self._align_sdxl_text_encoder_cache_contract()

        sdxl_component_residency = self._should_use_sdxl_component_cpu_residency()
        if self._is_diffusers_unet_route() and sdxl_component_residency:
            self._log("Diffusers component residency alignment enabled: VAE and frozen text encoders will move on-demand.")

        if model_arch == "sdxl":
            unet_backend = str(getattr(self.config, "sdxl_unet_backend", "diffusers") or "diffusers").strip().lower()
            if unet_backend != "diffusers":
                try:
                    from .native_unet import install_sdxl_native_unet_backend

                    status = install_sdxl_native_unet_backend(
                        self.model,
                        backend=unet_backend,
                        model_path=model_path,
                        logger=logger,
                    )
                    self._native_unet_status = status.as_dict()
                    self._attach_memory_runtime_profiles_to_training_loop()
                    self._log(
                        "SDXL native UNet backend prepared: "
                        f"backend={status.backend}, mode={status.mode}, blocks={len(status.blocks)}"
                    )
                except NotImplementedError as exc:
                    raise RuntimeError(str(exc)) from exc
                except Exception as exc:
                    self._log(f"SDXL native UNet backend skipped: {exc}")

        # 准备训练
        preserve_unet_residency = False
        if model_arch == "newbie":
            newbie_residency_mode = str(getattr(self.config, "newbie_block_residency", "resident") or "resident")
            preserve_unet_residency = newbie_residency_mode.strip().lower().replace("-", "_") not in {
                "",
                "resident",
                "off",
                "none",
                "gpu",
            }
        checkpoint_decision = resolve_checkpoint_policy(
            self.config,
            route=model_arch,
            cuda_available=str(self.device).startswith("cuda"),
        )
        self._checkpoint_policy_profile = checkpoint_decision.as_dict()
        self.config.gradient_checkpointing = bool(checkpoint_decision.gradient_checkpointing)
        self.config.cpu_offload_checkpointing = bool(checkpoint_decision.cpu_offload_checkpointing)
        # #147/#166: faithful native forward (3D RoPE) is now compatible with native
        # block checkpointing (rope_emb threads through _checkpoint_block). Faithful is
        # the explicit opt-in, so we still force the generic HF gradient/cpu-offload
        # checkpointing off (inert for the native subset), but we no longer disable the
        # native block-checkpointing lever — selective policy below applies to it too.
        anima_faithful_active = bool(getattr(self, "_anima_faithful_active", False))
        if anima_faithful_active:
            self.config.gradient_checkpointing = False
            self.config.cpu_offload_checkpointing = False
        self._log(
            "Checkpoint policy: "
            f"requested={checkpoint_decision.requested_policy}, "
            f"effective={checkpoint_decision.effective_policy}, "
            f"gradient_checkpointing={'on' if self.config.gradient_checkpointing else 'off'}, "
            f"cpu_offload={'on' if self.config.cpu_offload_checkpointing else 'off'}"
            + (f", fallback={checkpoint_decision.fallback_reason}" if checkpoint_decision.fallback_reason else "")
        )
        for warning in checkpoint_decision.warnings:
            self._log(f"[checkpoint-policy][warn] {warning}")
        if checkpoint_decision.effective_policy == "selective":
            try:
                if model_arch == "anima":
                    self.config.anima_block_checkpointing = True
                    self.config.anima_block_checkpointing_mode = "selective"
                elif model_arch == "newbie" and not anima_faithful_active:
                    self.config.newbie_block_checkpointing = True
                    self.config.newbie_block_checkpointing_mode = "selective"
            except Exception:
                pass
        loader.prepare_for_training(
            self.model,
            gradient_checkpointing=self.config.gradient_checkpointing,
            xformers=self.config.xformers,
            runtime_plan=self.runtime_optimization_plan,
            train_text_encoder=bool(getattr(self.config, "train_text_encoder", False)),
            keep_text_encoders_on_cpu=sdxl_component_residency and not bool(getattr(self.config, "train_text_encoder", False)),
            keep_vae_on_cpu=sdxl_component_residency,
            preserve_unet_residency=preserve_unet_residency,
            defer_per_block_compile=self._is_lora_training_route(),
        )
        if self.runtime_optimization_plan is not None:
            self._refresh_attention_runtime_profile(
                model_arch=model_arch,
                route=model_arch,
                patched=0,
                patch_target="runtime_attention_backend",
                applied=True,
                source="runtime_attention_backend",
            )
        if model_arch in {"anima", "newbie"}:
            self._mark_runtime_phase(f"{model_arch}_prepare_for_training", log=False)
        if self.runtime_optimization_plan is not None:
            for reason in self.runtime_optimization_plan.reasons:
                if reason.startswith(("attention backend", "sdpa ", "xformers fallback")) or "set_use_sdpa" in reason:
                    self._log(f"[runtime-opt] {reason}")
            for warning in self.runtime_optimization_plan.warnings:
                if "attention" in warning.lower() or "xformers" in warning.lower() or "sdpa" in warning.lower():
                    self._log(f"[runtime-opt][warn] {warning}")

        if model_arch == "sdxl" and bool(getattr(self.config, "lulynx_precision_swap_enabled", False)):
            try:
                from .precision_swap import build_sdxl_precision_swap_plan

                strategy = str(getattr(self.config, "lulynx_precision_swap_strategy", "balanced") or "balanced")
                plan = build_sdxl_precision_swap_plan(
                    self.model.unet,
                    strategy=strategy,
                    resolution=self._get_dataset_resolution(),
                    residency_mode=str(getattr(self.config, "lulynx_weight_residency", "resident") or "resident"),
                )
                self._precision_swap_profile = plan.as_profile_dict()
                residency_mode = str(getattr(self.config, "lulynx_weight_residency", "resident") or "resident").strip().lower()
                if residency_mode != "resident":
                    warnings = list(self._precision_swap_profile.get("warnings") or [])
                    self._precision_swap_profile["warnings"] = warnings
                    self._precision_swap_profile["joint_strategy"] = {
                        "weight_residency": residency_mode,
                        "order": "apply layer residency first, then block swap selected native units",
                    }
                manual_swap_requested = (
                    int(getattr(self.config, "blocks_to_swap", 0) or 0) > 0
                    or str(getattr(self.config, "swap_granularity", "off") or "off").strip().lower() != "off"
                    or int(getattr(self.config, "swap_count", 0) or 0) > 0
                    or float(getattr(self.config, "swap_ratio", 0.0) or 0.0) > 0.0
                )
                low_vram_default_swap = (
                    bool(getattr(self.config, "sdxl_low_vram_optimization", False))
                    and int(getattr(self.config, "blocks_to_swap", 0) or 0) == 2
                    and str(getattr(self.config, "swap_granularity", "off") or "off").strip().lower() == "off"
                    and int(getattr(self.config, "swap_count", 0) or 0) == 0
                    and float(getattr(self.config, "swap_ratio", 0.0) or 0.0) == 0.0
                )
                if low_vram_default_swap:
                    manual_swap_requested = False
                if plan.compatible_blocks_to_swap > 0 and not manual_swap_requested:
                    self.config.swap_granularity = "block"
                    self.config.swap_count = plan.compatible_blocks_to_swap
                    self.config.blocks_to_swap = plan.compatible_blocks_to_swap
                    self._log(
                        "Lulynx Precision Swap wired to BlockSwap: "
                        f"backend={plan.backend}, blocks_to_swap={plan.compatible_blocks_to_swap}"
                    )
                elif plan.compatible_blocks_to_swap == 0 and low_vram_default_swap:
                    self.config.swap_granularity = "off"
                    self.config.swap_count = 0
                    self.config.blocks_to_swap = 0
                    self._log("Lulynx Precision Swap disabled the low-VRAM default block swap for the current joint residency plan.")
                elif manual_swap_requested:
                    self._log("Lulynx Precision Swap planning kept advisory because manual swap settings are already active.")
                self._log(
                    "Lulynx Precision Swap plan: "
                    f"strategy={plan.strategy}, selected={plan.selected_names}, "
                    f"selected_params={plan.selected_parameter_mb:.1f}MB, "
                    f"activation_hint={plan.selected_activation_hint_mb:.1f}MB"
                )
                if residency_mode != "resident":
                    self._log(f"Lulynx Precision Swap joint mode: layer residency={residency_mode}")
            except Exception as exc:
                self._log(f"Lulynx Precision Swap planning skipped: {exc}")

        if (
            model_arch == "anima"
            and getattr(self.compile_contract_decision, "resolved", "") == "full_core"
            and self.runtime_optimization_plan is not None
        ):
            self._anima_full_core_original_run_blocks = getattr(self.model.unet, "_run_blocks", None)
            compiled = apply_full_core_compile(
                self.model,
                self.runtime_optimization_plan,
                route="anima",
            )
            self._refresh_compile_runtime_profile(
                model_arch=model_arch,
                applied=bool(compiled),
                compiled_targets=1 if compiled else 0,
                compile_kind="full_core",
                source="full_core_compile",
                skip_reason="no_full_core_target_compiled" if not compiled else "",
            )
            if compiled:
                self._anima_full_core_compiled_run_blocks = getattr(self.model.unet, "_run_blocks", None)
                self._log("[compile-contract] Anima full-core compile applied to stable core target")
            else:
                self._anima_full_core_compiled_run_blocks = None
                self._log("[compile-contract][warn] Anima full-core compile requested but no target was compiled")
            for reason in self.runtime_optimization_plan.reasons:
                if "full_core_compile:" in reason:
                    self._log(f"[runtime-opt] {reason}")
            for warning in self.runtime_optimization_plan.warnings:
                if "full_core_compile:" in warning:
                    self._log(f"[runtime-opt][warn] {warning}")

        # === 显存优化 ===
        from .memory_optimizations import (
            apply_channels_last,
            PipelineSlicer,
            SlicingConfig,
        )

        # channels_last 内存格式
        if getattr(self.config, "opt_channels_last", False):
            apply_channels_last(self.model.unet, self.model.vae, verbose=True)

        # Frozen weight compression. Legacy fp8_base remains Anima/Newbie DiT-only.
        if (getattr(self.config, "weight_compression_enabled", False) or getattr(self.config, "fp8_base", False)) and not getattr(self.config, "compression_companion_enabled", False) and (model_arch in ("anima", "newbie") or getattr(self.config, "weight_compression_enabled", False)):
            self._apply_weight_compression()

        # VAE slicing/tiling + attention slicing
        slicer = PipelineSlicer(SlicingConfig(
            vae_slicing=getattr(self.config, "vae_slicing", True),
            vae_tiling=getattr(self.config, "vae_tiling", False),
            attention_slicing=getattr(self.config, "attention_slicing", True),
        ))
        slicer.apply_to_unet(self.model.unet)
        slicer.apply_to_vae(self.model.vae)
        if model_arch in {"anima", "newbie"}:
            self._mark_runtime_phase(f"{model_arch}_memory_slicing", log=False)

        # Sequential CPU offload: move each sub-module to GPU only during its
        # forward pass, then immediately back to CPU.  This mirrors the
        # diffusers `enable_sequential_cpu_offload()` behaviour and is driven
        # by the config flag of the same name.
        if getattr(self.config, "enable_sequential_cpu_offload", False):
            self._apply_sequential_cpu_offload()

        # Initialize TE Manager (Semantic Base-Tuner aware)
        self.te_manager = SemanticTunerAwareTEManager(self.config, device=self.device, dtype=self.dtype)
        self.te_manager.prepare()

        if getattr(self.config, "semantic_tuner_enabled", False):
            self._log("🧠 Semantic Base-Tuner Mode (V3.1): Initializing Sidecar Architecture...")

            # 1. Get Context
            ctx = self.te_manager.get_semantic_context()
            if not ctx:
                raise RuntimeError("Failed to get Semantic Context from TE Manager")

            projector = ctx.get("projector")
            llm_dim = 1024 # TODO: Get from projector in_dim or config
            if projector:
                 # Projector config has in_dim
                 # But Sidecar expects LLM output dim, which is Projector Input Dim?
                 # NO.
                 # Sidecar Input = Projector OUTPUT.
                 # Projector maps LLM_Dim -> Target_Dim.
                 # Sidecar (W_new) maps Target_Dim -> Inner_Head_Dim.
                 # So llm_dim passed to inject_neural_sidecar should generally match Projector Output Dim (e.g. 2048) or U-Net Context Dim.
                 # Let's verify projector's output dimension.
                 # In V3.1 loader, we set target_dim = 2048.
                 llm_dim = 2048

            # 2. Inject Sidecar
            # This patches the U-Net in-place and returns the trainable Sidecar Network
            from .semantic_tuner.unet_sidecar import inject_neural_sidecar
            self.model.unet, self.sidecar_net = inject_neural_sidecar(self.model.unet, llm_dim=llm_dim)
            self._log(f"Sidecar injected into U-Net (Input Dim: {llm_dim})")

            # 3. Freeze Base U-Net
            self.model.unet.requires_grad_(False) # Base weights W_orig are FROZEN

            # 4. Collect Trainable Params (Sidecar + Projector)
            self.trainable_params = []

            # Sidecar (W_new)
            self.trainable_params.extend(list(self.sidecar_net.parameters()))

            # Projector
            if projector:
                projector.train() # Ensure train mode
                projector.requires_grad_(True)
                self.trainable_params.extend(list(projector.parameters()))

            # Semantic-Tuner still relies on the generic training loop for
            # gradient clipping / safeguard integration, so expose a minimal
            # adapter-like wrapper around the trainable sidecar + projector params.
            self.lora_injector = _FullFinetuneParamWrapper(
                self.trainable_params,
                state_dict_getter=self._build_semantic_tuner_state_dict,
                state_dict_loader=self._load_semantic_tuner_state_dict,
            )

            self._log(f"Sidecar Training Ready. Trainable Params: {sum(p.numel() for p in self.trainable_params):,}")
            return self

        # ── Full Fine-tuning Mode ──
        training_type = getattr(self.config, "training_type", "lora")
        if training_type == "full_finetune":
            if is_anima_full_finetune(self.config):
                setup = prepare_anima_dit_only_full_finetune(
                    config=self.config,
                    model=self.model,
                    log=self._log,
                )
                self.trainable_params = list(setup.trainable_params)
                self.lora_injector = _FullFinetuneParamWrapper(
                    self.trainable_params,
                    unet=self.model.unet,
                    state_dict_getter=lambda: build_anima_full_finetune_state_dict(unet=self.model.unet),
                )
                self._anima_full_finetune_setup = setup.as_dict()
                self._log(
                    "Anima full finetune ready: "
                    f"mode={setup.mode}, trainable_params={setup.total_params:,}, "
                    f"text_encoder_blocked={setup.train_text_encoder_blocked}"
                )
                self._prepare_anima_dit_runtime_guardrails()
                if self.config.resume_path:
                    resume_path = Path(self.config.resume_path)
                    if resume_path.is_file() and resume_path.suffix.lower() in {EXT_SAFETENSORS, EXT_PT, ".ckpt"}:
                        self._log(f"Resuming Anima full-finetune DiT weights from {resume_path}")
                        state = self._load_state_dict_from_path(resume_path)
                        load_anima_full_finetune_state(
                            unet=self.model.unet,
                            state_dict=state,
                            log=self._log,
                        )
                    else:
                        self._log(
                            f"Resume path {resume_path} is not a supported Anima full-finetune weight file, skipping."
                        )
                return self
            self._log("Full Fine-tuning mode: UNet will be trained directly (no LoRA)")
            self.model.unet.requires_grad_(True)
            if self.config.train_text_encoder:
                self.model.text_encoder_1.requires_grad_(True)
                if _family.has_dual_text_encoders and self.model.text_encoder_2:
                    self.model.text_encoder_2.requires_grad_(True)
            self.trainable_params = list(
                p for p in self.model.unet.parameters() if p.requires_grad
            )
            if self.config.train_text_encoder:
                self.trainable_params.extend(
                    p for p in self.model.text_encoder_1.parameters() if p.requires_grad
                )
                if _family.has_dual_text_encoders and self.model.text_encoder_2:
                    self.trainable_params.extend(
                        p for p in self.model.text_encoder_2.parameters() if p.requires_grad
                    )
            # Collect model references for full-state save
            te1 = self.model.text_encoder_1 if self.config.train_text_encoder else None
            te2 = None
            if self.config.train_text_encoder and _family.has_dual_text_encoders and self.model.text_encoder_2:
                te2 = self.model.text_encoder_2
            self.lora_injector = _FullFinetuneParamWrapper(
                self.trainable_params,
                unet=self.model.unet,
                text_encoder_1=te1,
                text_encoder_2=te2,
            )
            total_params = sum(p.numel() for p in self.trainable_params)
            self._log(f"Full FT trainable parameters: {total_params:,}")
            # Resume full-finetune weights from checkpoint
            if self.config.resume_path:
                resume_path = Path(self.config.resume_path)
                if resume_path.is_file() and resume_path.suffix.lower() in {EXT_SAFETENSORS, EXT_PT, ".ckpt"}:
                    self._log(f"Resuming full-finetune weights from {resume_path}")
                    state = self._load_state_dict_from_path(resume_path)
                    unet_keys = {k: v for k, v in state.items() if k.startswith("unet.")}
                    te1_keys = {k: v for k, v in state.items() if k.startswith("text_encoder_1.")}
                    te2_keys = {k: v for k, v in state.items() if k.startswith("text_encoder_2.")}
                    if unet_keys:
                        prefix_len = len("unet.")
                        unet_sd = {k[prefix_len:]: v for k, v in unet_keys.items()}
                        missing, unexpected = self.model.unet.load_state_dict(unet_sd, strict=False)
                        self._log(f"  unet: loaded {len(unet_sd) - len(missing)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
                    if te1_keys and self.model.text_encoder_1 is not None:
                        prefix_len = len("text_encoder_1.")
                        te1_sd = {k[prefix_len:]: v for k, v in te1_keys.items()}
                        missing, unexpected = self.model.text_encoder_1.load_state_dict(te1_sd, strict=False)
                        self._log(f"  text_encoder_1: loaded {len(te1_sd) - len(missing)} keys")
                    if te2_keys and self.model.text_encoder_2 is not None:
                        prefix_len = len("text_encoder_2.")
                        te2_sd = {k[prefix_len:]: v for k, v in te2_keys.items()}
                        missing, unexpected = self.model.text_encoder_2.load_state_dict(te2_sd, strict=False)
                        self._log(f"  text_encoder_2: loaded {len(te2_sd) - len(missing)} keys")
                else:
                    self._log(f"Resume path {resume_path} is not a supported full-finetune weight file, skipping.")
            return self

        # ── Textual Inversion Mode ──
        if training_type == "textual_inversion":
            self._log("Textual Inversion mode: only embedding will be trained")
            from .textual_inversion import TextualInversionTrainer, TextualInversionConfig
            ti_config = TextualInversionConfig(
                model_path=self.config.base_model_path,
                model_type=model_arch,
                placeholder_token=getattr(self.config, "ti_placeholder_token", "<new>"),
                initializer_token=getattr(self.config, "ti_init_token", "") or "person",
                num_vectors=getattr(self.config, "ti_num_vectors", 1),
                learning_rate=self.config.learning_rate,
                max_train_steps=getattr(self.config, "max_train_steps", 1000),
                mixed_precision=getattr(self.config, "mixed_precision", "fp16"),
            )
            self._ti_trainer = TextualInversionTrainer(ti_config)
            self._ti_trainer.load_model()
            # Move all TI components to the training device/dtype
            ti_modules = [self._ti_trainer.model, self._ti_trainer.text_encoder, self._ti_trainer.vae]
            if getattr(self._ti_trainer, 'text_encoder_2', None) is not None:
                ti_modules.append(self._ti_trainer.text_encoder_2)
            for module in ti_modules:
                module.to(device=self.device, dtype=self.dtype)
            # Re-init concept embedding on the now-correct device
            self._ti_trainer._init_concept_embedding()
            self._ti_trainer.prepare_optimizer()
            self.trainable_params = list(self._ti_trainer.concept_embedding.parameters())
            self.lora_injector = _FullFinetuneParamWrapper(self.trainable_params)
            self._ti_mode = True
            self._log("Textual Inversion trainer ready")
            return self

        # ── DreamBooth Mode ──
        if training_type == "dreambooth":
            use_lora = getattr(self.config, "use_lora", False)
            if use_lora:
                self._log("DreamBooth + LoRA mode")
                # Fall through to LoRA branch below
            else:
                self._log("DreamBooth + Full Finetune mode")
                # Redirect to full finetune logic
                self.model.unet.requires_grad_(True)
                if self.config.train_text_encoder:
                    self.model.text_encoder_1.requires_grad_(True)
                    if _family.has_dual_text_encoders and self.model.text_encoder_2:
                        self.model.text_encoder_2.requires_grad_(True)
                self.trainable_params = list(
                    p for p in self.model.unet.parameters() if p.requires_grad
                )
                if self.config.train_text_encoder:
                    self.trainable_params.extend(
                        p for p in self.model.text_encoder_1.parameters() if p.requires_grad
                    )
                    if _family.has_dual_text_encoders and self.model.text_encoder_2:
                        self.trainable_params.extend(
                            p for p in self.model.text_encoder_2.parameters() if p.requires_grad
                        )
                te1 = self.model.text_encoder_1 if self.config.train_text_encoder else None
                te2 = None
                if self.config.train_text_encoder and _family.has_dual_text_encoders and self.model.text_encoder_2:
                    te2 = self.model.text_encoder_2
                self.lora_injector = _FullFinetuneParamWrapper(
                    self.trainable_params,
                    unet=self.model.unet,
                    text_encoder_1=te1,
                    text_encoder_2=te2,
                )
                total_params = sum(p.numel() for p in self.trainable_params)
                self._log(f"DreamBooth FT trainable parameters: {total_params:,}")
                return self

        self._log("Injecting LoRA layers...")

        # === V4.0: MN-LoRA & Hyperparam Management ===
        from core.lulynx.hyperparam_manager import LulynxHyperparamManager

        if self.config.mn_lora_enabled:
            # 应用 MN-LoRA 预设 (自动识别架构)
            mn_preset = getattr(self.config, "mn_lora_preset", None) or "slim"
            mn_config = LulynxHyperparamManager.apply_preset(
                preset_name=mn_preset,
                model_arch=getattr(self.config.model_type, "value", self.config.model_type) # e.g. "sdxl", "flux"
            )
            model_type_name = getattr(self.config.model_type, "value", self.config.model_type)
            self._log(f"MN-LoRA Presets applied ({mn_preset} for {model_type_name})")

            # 覆盖 config 中的相关参数 (如果是 None 或默认值)
            if self.config.lulynx_proj_dim == 128: # Default
                self.config.lulynx_proj_dim = mn_config.manifold_proj_dim
            if self.config.lulynx_manifold_sparse_freq == 1: # Default
                self.config.lulynx_manifold_sparse_freq = mn_config.manifold_sparse_freq

        # Lulynx training wrapper removed — Warehouse cutoff.
        # Individual components (ManifoldConstraint, LNGuard, etc.) are still
        # available via core.lulynx but the orchestrating wrapper is no longer
        # wired into the main training path.
        self._lulynx_wrapper = None

        # Initialize Vortex Memory Manager (Global)
        if hasattr(self.config, 'vortex_enabled') and self.config.vortex_enabled:
            from core.memory_vortex_v2 import vortex_manager_v2 as vortex_manager

            # [V2.1] Apply Profile (which sets strategy and cache_limit)
            if hasattr(self.config, 'vortex_profile'):
                 vortex_manager.config.profile = self.config.vortex_profile
                 vortex_manager.config.apply_profile()

            # [V2] Strategy override (if explicitly set)
            if hasattr(self.config, 'vortex_strategy'):
                 vortex_manager.config.strategy = self.config.vortex_strategy

            vortex_manager.initialize()
            self._log(f"Vortex Memory Manager initialized (Profile: {getattr(vortex_manager.config, 'profile', 'N/A')}, Strategy: {vortex_manager.config.strategy}).")

        # 根据网络类型选择注入器
        from .config import NetworkType

        enable_dora_layers = self.config.use_dora or self.config.dora_enabled

        network_module = getattr(self.config.network_module, "value", self.config.network_module)
        try:
            from .method_adapter_contract import adapter_contract_summary, resolve_adapter_method

            adapter_spec = resolve_adapter_method(self.config, family=model_arch)
            self._log(adapter_contract_summary(adapter_spec))
            for warning in adapter_spec.warnings:
                self._log(f"adapter contract warning: {warning}")
        except Exception as exc:
            logger.debug("Adapter contract summary failed: %s", exc)
        from .dim_from_weights import apply_dim_from_weights

        apply_dim_from_weights(self.config, model_arch=model_arch, log_fn=self._log)

        is_lycoris = network_module == NetworkType.LYCORIS.value
        lora_activation_recompute = self._resolve_lora_activation_recompute(model_arch)
        if lora_activation_recompute:
            self._log("LoRA branch activation recompute enabled for native DiT adapter path.")
        if model_arch == "newbie":
            newbie_adapter_type = str(getattr(self.config, "newbie_adapter_type", "") or "").strip().lower().replace("-", "_")
            if newbie_adapter_type in {"lycoris_lokr", "lycoris_loha", "lycoris_locon", "lycoris_ia3", "lycoris_full", "lycoris_diag_oft"}:
                newbie_adapter_type = newbie_adapter_type.replace("lycoris_", "")
            if newbie_adapter_type in {"diag_oft", "oft"}:
                newbie_adapter_type = "diag-oft"
            if newbie_adapter_type in {"lokr", "loha", "locon", "ia3", "full", "diag-oft"}:
                is_lycoris = True
                self.config.lycoris_algo = newbie_adapter_type
                self._log(f"Newbie adapter_type={newbie_adapter_type}: using LyCORIS injector.")
            elif newbie_adapter_type == "lora_fa":
                self.config.lora_fa_enabled = True
                self._log("Newbie adapter_type=lora_fa requested; using LoRA injector with lora_fa_enabled marker.")
            elif newbie_adapter_type == "vera":
                self.config.vera_enabled = True
                self._log("Newbie adapter_type=vera requested; using LoRA injector with VeRA enabled.")
            elif newbie_adapter_type in {"hydralora", "hydra_lora"}:
                self.config.hydralora_enabled = True
                self._log("Newbie adapter_type=hydralora requested; using HydraLoRA injector.")
            elif newbie_adapter_type == "fera":
                self.config.fera_enabled = True
                self._log("Newbie adapter_type=fera requested; using FeRA injector.")
            elif newbie_adapter_type == "tlora":
                network_module = NetworkType.TLORA.value
                self.config.network_module = NetworkType.TLORA
                self._log("Newbie adapter_type=tlora requested; using T-LoRA injector.")
            elif newbie_adapter_type == "dora":
                self.config.use_dora = True
                self.config.dora_enabled = True
                self._log("Newbie adapter_type=dora requested; using DoRA injector.")
            elif newbie_adapter_type == "lora_plus":
                self.config.lora_plus_enabled = True
                self._log("Newbie adapter_type=lora_plus requested; using LoRA+ optimizer grouping.")

        # Check for advanced adapter types (Step Expert, ChimeraHydra)
        use_step_expert = bool(getattr(self.config, "step_expert_enabled", False))
        use_chimera_hydra = bool(getattr(self.config, "chimera_hydra_enabled", False))

        if use_step_expert and use_chimera_hydra:
            raise ValueError("Cannot enable both Step Expert and ChimeraHydra simultaneously. Choose one.")

        if use_step_expert:
            from .step_expert_injector import StepExpertInjector
            from .step_expert_routing import StepExpertConfig

            num_experts = int(getattr(self.config, "step_expert_num_experts", 4) or 4)
            boundaries_raw = getattr(self.config, "step_expert_boundaries", None)
            if boundaries_raw is None:
                # Auto-generate boundaries for uniform splits
                boundaries = tuple((i + 1) / num_experts for i in range(num_experts - 1))
            else:
                boundaries = tuple(float(b) for b in boundaries_raw)

            step_expert_config = StepExpertConfig(
                num_experts=num_experts,
                rank=self.config.network_dim,
                alpha=self.config.network_alpha,
                boundaries=boundaries,
            )
            self.lora_injector = StepExpertInjector(
                config=step_expert_config,
                target_modules=None,  # Will use model family defaults
                model_arch=model_arch,
            )
            self._log(f"Using Step Expert (experts={num_experts}, boundaries={boundaries})")

        elif use_chimera_hydra:
            from .chimera_hydra_injector import ChimeraHydraInjector
            from .chimera_hydra import ChimeraHydraConfig

            content_experts = int(getattr(self.config, "chimera_hydra_content_experts", 4) or 4)
            frequency_experts = int(getattr(self.config, "chimera_hydra_frequency_experts", 2) or 2)
            routing = str(getattr(self.config, "chimera_hydra_routing", "top_k") or "top_k")
            content_top_k = int(getattr(self.config, "chimera_hydra_content_top_k", 2) or 2)
            frequency_top_k = int(getattr(self.config, "chimera_hydra_frequency_top_k", 1) or 1)
            use_fft = bool(getattr(self.config, "chimera_hydra_use_fft_features", True))

            chimera_config = ChimeraHydraConfig(
                content_experts=content_experts,
                frequency_experts=frequency_experts,
                rank=self.config.network_dim,
                alpha=self.config.network_alpha,
                routing=routing,
                content_top_k=content_top_k,
                frequency_top_k=frequency_top_k,
            )
            self.lora_injector = ChimeraHydraInjector(
                config=chimera_config,
                target_modules=None,  # Will use model family defaults
                model_arch=model_arch,
                use_fft_features=use_fft,
            )
            self._log(
                f"Using ChimeraHydra (content={content_experts}, frequency={frequency_experts}, "
                f"routing={routing}, fft_features={use_fft})"
            )

        elif is_lycoris:
            from .lycoris_layers import LyCORISConfig, LyCORISInjector, LyCORISType

            lycoris_algo = getattr(self.config, "lycoris_algo", "loha")
            try:
                lycoris_type = LyCORISType(lycoris_algo)
            except ValueError:
                lycoris_type = LyCORISType.LOHA

            lycoris_train_norm = bool(getattr(self.config, "lycoris_train_norm", False)) or bool(
                getattr(self.config, "lokr_train_norm", False)
            )
            self.lora_injector = LyCORISInjector(
                LyCORISConfig(
                    lycoris_type=lycoris_type,
                    rank=self.config.network_dim,
                    alpha=self.config.network_alpha,
                    dropout=self.config.network_dropout,
                    loha_use_effective=getattr(self.config, "lycoris_loha_use_effective", True),
                    lokr_factor=getattr(self.config, "lycoris_lokr_factor", -1),
                    lokr_rank_dropout=getattr(self.config, "lokr_rank_dropout", 0.0),
                    lokr_module_dropout=getattr(self.config, "lokr_module_dropout", 0.0),
                    lokr_full_matrix=getattr(self.config, "lokr_full_matrix", False),
                    lokr_decompose_both=getattr(self.config, "lokr_decompose_both", False),
                    lokr_unbalanced_factorization=getattr(self.config, "lokr_unbalanced_factorization", False),
                    lokr_no_materialize_forward=getattr(self.config, "lokr_no_materialize_forward", False),
                    lokr_no_materialize_strategy=getattr(self.config, "lokr_no_materialize_strategy", "legacy"),
                    train_norm=lycoris_train_norm,
                    conv_dim=getattr(self.config, "lycoris_conv_dim", 0),
                    conv_alpha=getattr(self.config, "lycoris_conv_alpha", 0.0),
                    presets=getattr(self.config, "lycoris_presets", ""),
                )
            )
            norm_info = " +norm" if lycoris_train_norm else ""
            self._log(f"Using LyCORIS ({lycoris_type.value}{norm_info})")
        elif self.config.use_dora:
            adapter_target_policy_label, adapter_target_selected, adapter_target_rank_map = self._resolve_fg_lora_rank_plan(model_arch)
            self.lora_injector = LoRAInjector(
                rank=self.config.network_dim,
                alpha=self.config.network_alpha,
                dropout=self.config.network_dropout,
                adapter_init_strategy=getattr(self.config, "adapter_init_strategy", "default"),
                pissa_enabled=self.config.pissa_enabled,
                pissa_niter=self.config.pissa_init_iters,
                svd_algo=self.config.pissa_svd_algo,
                pissa_oversample=int(getattr(self.config, "pissa_oversample", 8) or 8),
                pissa_apply_conv2d=bool(getattr(self.config, "pissa_apply_conv2d", False)),
                loftq_bits=int(getattr(self.config, "loftq_bits", 4) or 4),
                loftq_quant_type=getattr(self.config, "loftq_quant_type", "rowwise"),
                adapter_init_export_mode=getattr(self.config, "adapter_init_export_mode", "auto"),
                dora_enabled=True,
                dora_mode=getattr(self.config, "dora_mode", "full"),
                vortex_enabled=self.config.vortex_enabled,
                model_arch=model_arch,
                vera_enabled=bool(getattr(self.config, "vera_enabled", False)),
                vera_d_initial=float(getattr(self.config, "vera_d_initial", 0.1)),
                vera_prng_key=int(getattr(self.config, "vera_prng_key", 0)),
                lora_fa_enabled=bool(getattr(self.config, "lora_fa_enabled", False)),
                hydralora_enabled=bool(getattr(self.config, "hydralora_enabled", False)),
                hydralora_num_experts=int(getattr(self.config, "hydralora_num_experts", 4) or 4),
                hydralora_routing=str(getattr(self.config, "hydralora_routing", "top_k") or "top_k"),
                hydralora_top_k=int(getattr(self.config, "hydralora_top_k", 2) or 2),
                hydralora_sparse_top_k=bool(getattr(self.config, "hydralora_sparse_top_k", False)),
                fera_enabled=bool(getattr(self.config, "fera_enabled", False)),
                fera_gate_init=float(getattr(self.config, "fera_gate_init", 0.0) or 0.0),
                flexrank_enabled=bool(getattr(self.config, "flexrank_lora_enabled", False)),
                flexrank_rank_range_min=int(getattr(self.config, "flexrank_lora_rank_range_min", 1) or 1),
                activation_recompute=lora_activation_recompute,
                rs_lora_enabled=bool(getattr(self.config, "rs_lora_enabled", False)),
                adapter_target_policy=adapter_target_policy_label,
                adapter_target_selected=adapter_target_selected,
                adapter_target_rank_map=adapter_target_rank_map,
            )
            self._log(f"Using DoRA via unified LoRAInjector (mode: {getattr(self.config, 'dora_mode', 'full')})")
        else:
            # 标准 LoRA / T-LoRA 注入器
            tlora_enabled = (
                network_module == NetworkType.TLORA.value
                or str(network_module).endswith("tlora")
                or bool(getattr(self.config, "t_lora_enabled", False))
            )
            lora_fa_enabled = network_module == NetworkType.LORA_FA.value or bool(getattr(self.config, "lora_fa_enabled", False))
            vera_enabled = network_module == NetworkType.VERA.value or bool(getattr(self.config, "vera_enabled", False))
            flexrank_enabled = network_module == NetworkType.FLEXRANK_LORA.value or bool(getattr(self.config, "flexrank_lora_enabled", False))
            tlora_min_rank = int(getattr(self.config, "tlora_min_rank", 1) or 1)
            tlora_rank_schedule = str(getattr(self.config, "tlora_rank_schedule", "constant") or "constant")
            tlora_orthogonal_init = bool(getattr(self.config, "tlora_orthogonal_init", False))

            # Compute total_steps for T-LoRA schedule
            tlora_total_steps = max(int(getattr(self.config, "max_train_steps", 0) or 0), 1000)

            adapter_target_policy_label, adapter_target_selected, adapter_target_rank_map = self._resolve_fg_lora_rank_plan(model_arch)
            self.lora_injector = LoRAInjector(
                rank=self.config.network_dim,
                alpha=self.config.network_alpha,
                dropout=self.config.network_dropout,
                adapter_init_strategy=getattr(self.config, "adapter_init_strategy", "default"),
                pissa_enabled=self.config.pissa_enabled,
                pissa_niter=self.config.pissa_init_iters,
                svd_algo=self.config.pissa_svd_algo,
                pissa_oversample=int(getattr(self.config, "pissa_oversample", 8) or 8),
                pissa_apply_conv2d=bool(getattr(self.config, "pissa_apply_conv2d", False)),
                loftq_bits=int(getattr(self.config, "loftq_bits", 4) or 4),
                loftq_quant_type=getattr(self.config, "loftq_quant_type", "rowwise"),
                adapter_init_export_mode=getattr(self.config, "adapter_init_export_mode", "auto"),
                dora_enabled=enable_dora_layers,
                dora_mode=getattr(self.config, "dora_mode", "full"),
                vortex_enabled=self.config.vortex_enabled,
                model_arch=model_arch,
                tlora_enabled=tlora_enabled,
                tlora_min_rank=tlora_min_rank,
                tlora_rank_schedule=tlora_rank_schedule,
                tlora_orthogonal_init=tlora_orthogonal_init,
                tlora_total_steps=tlora_total_steps,
                vera_enabled=vera_enabled,
                vera_d_initial=float(getattr(self.config, "vera_d_initial", 0.1)),
                vera_prng_key=int(getattr(self.config, "vera_prng_key", 0)),
                lora_fa_enabled=lora_fa_enabled,
                hydralora_enabled=bool(getattr(self.config, "hydralora_enabled", False)),
                hydralora_num_experts=int(getattr(self.config, "hydralora_num_experts", 4) or 4),
                hydralora_routing=str(getattr(self.config, "hydralora_routing", "top_k") or "top_k"),
                hydralora_top_k=int(getattr(self.config, "hydralora_top_k", 2) or 2),
                hydralora_sparse_top_k=bool(getattr(self.config, "hydralora_sparse_top_k", False)),
                fera_enabled=bool(getattr(self.config, "fera_enabled", False)),
                fera_gate_init=float(getattr(self.config, "fera_gate_init", 0.0) or 0.0),
                flexrank_enabled=flexrank_enabled,
                flexrank_rank_range_min=int(getattr(self.config, "flexrank_lora_rank_range_min", 1) or 1),
                activation_recompute=lora_activation_recompute,
                rs_lora_enabled=bool(getattr(self.config, "rs_lora_enabled", False)),
                adapter_target_policy=adapter_target_policy_label,
                adapter_target_selected=adapter_target_selected,
                adapter_target_rank_map=adapter_target_rank_map,
            )
            if bool(getattr(self.config, "hydralora_enabled", False)):
                self._log(
                    f"Using HydraLoRA (experts={int(getattr(self.config, 'hydralora_num_experts', 4) or 4)}, "
                    f"routing={str(getattr(self.config, 'hydralora_routing', 'top_k') or 'top_k')}, "
                    f"top_k={int(getattr(self.config, 'hydralora_top_k', 2) or 2)}, "
                    f"sparse_top_k={bool(getattr(self.config, 'hydralora_sparse_top_k', False))})"
                )
            elif bool(getattr(self.config, "fera_enabled", False)):
                self._log(f"Using FeRA (gate_init={float(getattr(self.config, 'fera_gate_init', 0.0) or 0.0)})")
            elif flexrank_enabled:
                self._log(
                    f"Using FlexRank LoRA (min_rank={int(getattr(self.config, 'flexrank_lora_rank_range_min', 1) or 1)}, "
                    f"max_rank={int(getattr(self.config, 'network_dim', 0) or 0)})"
                )
            elif tlora_enabled:
                self._log(f"Using T-LoRA (min_rank={tlora_min_rank}, schedule={tlora_rank_schedule}, "
                          f"orthogonal_init={tlora_orthogonal_init}, total_steps={tlora_total_steps})")
            else:
                self._log(f"Using LoRA (PiSSA: {self.config.pissa_enabled}, DoRA Layer: {enable_dora_layers}, Vortex: {self.config.vortex_enabled})")

        # DOP: deepcopy UNet BEFORE LoRA injection for reference model
        self._dop_instance = None
        if getattr(self.config, "dop_enabled", False) and hasattr(self.model, "unet"):
            import copy
            from .dop import DifferentialOutputPreservation
            _dop_ref_unet = copy.deepcopy(self.model.unet)
            for p in _dop_ref_unet.parameters():
                p.requires_grad_(False)
            self._dop_instance = DifferentialOutputPreservation(
                reference_model=_dop_ref_unet,
                weight=float(getattr(self.config, "dop_weight", 0.1) or 0.1),
                start_step=int(getattr(self.config, "dop_start_step", 0) or 0),
                interval=int(getattr(self.config, "dop_interval", 1) or 1),
                detach_reference=bool(getattr(self.config, "dop_detach_reference", True)),
            )
            self._log(f"DOP enabled: weight={self._dop_instance.weight}, interval={self._dop_instance.interval}")

        # Allow native-family configs to override target modules.
        custom_target_modules = None
        unet_exclude_substrings = None
        if model_arch == "anima":
            custom_target_modules = self._get_anima_target_modules()
            # The frozen native llm_adapter (faithful cross-attn context) shares
            # self_attn/cross_attn.{q,k,v}_proj suffixes with the DiT blocks, so the
            # injector's substring match leaks LoRA into it even though the target
            # contract excludes the llm_adapter dotted targets. When the adapter is
            # not trained, exclude its subtree by name so it gets no dead (zero-grad)
            # adapters. When training the adapter is requested, leave it injectable.
            if not self._anima_train_llm_adapter_enabled():
                unet_exclude_substrings = ["llm_adapter"]
        elif model_arch == "newbie":
            custom_target_modules = self._get_custom_target_modules()
            if custom_target_modules:
                self._log(f"Newbie custom target modules: {custom_target_modules}")

        if self.config.train_unet:
            if is_lycoris:
                self.lora_injector.inject(
                    self.model.unet,
                    custom_target_modules or _family.unet_target_modules,
                    prefix="unet",
                    exclude_name_substrings=unet_exclude_substrings,
                )
            elif custom_target_modules:
                # Native-family explicit unet targets opt into the adapter target
                # policy. Default policy "all" leaves _adapter_target_policy_active
                # False, so this stays a parity no-op (every target injected at
                # network_dim); only an active gradient/profiled/cka selection
                # actually restricts the injected subset and per-type rank.
                self.lora_injector.inject(
                    self.model.unet, custom_target_modules, prefix="unet",
                    apply_policy=True, exclude_name_substrings=unet_exclude_substrings,
                )
            else:
                self.lora_injector.inject_unet(self.model.unet)

        if self.config.train_text_encoder:
            if model_arch == "anima" and getattr(self.model, "anima_cached_training_ready", False):
                self._log("Anima cache-first training uses cached text conditioning; text encoder LoRA is skipped.")
            elif model_arch == "newbie" and self._has_newbie_cached_training_data():
                self._log("Newbie cache-first training uses cached conditioning; text encoder LoRA is skipped.")
            elif self.model.text_encoder_1 is None:
                self._log("Text encoder training requested but no text encoder is loaded; skipping text encoder LoRA.")
            elif is_lycoris:
                self.lora_injector.inject(self.model.text_encoder_1, _family.text_encoder_target_modules, prefix="te1")
            else:
                self.lora_injector.inject_text_encoder(
                    self.model.text_encoder_1, "te1"
                )
            if _family.has_dual_text_encoders and self.model.text_encoder_2:
                if is_lycoris:
                    self.lora_injector.inject(self.model.text_encoder_2, _family.text_encoder_target_modules, prefix="te2")
                else:
                    self.lora_injector.inject_text_encoder(
                        self.model.text_encoder_2, "te2"
                    )
        if model_arch in {"anima", "newbie"}:
            self._mark_runtime_phase(f"{model_arch}_adapter_inject", log=False)

        self._block_weight_manager = None
        if getattr(self.config, "bw_enable", False):
            try:
                from .block_weight import create_block_weight_manager_from_settings

                self._block_weight_manager = create_block_weight_manager_from_settings(
                    preset=str(getattr(self.config, "bw_preset", "") or ""),
                    in_weights=getattr(self.config, "bw_in_weights", ""),
                    mid_weight=getattr(self.config, "bw_mid_weight", ""),
                    out_weights=getattr(self.config, "bw_out_weights", ""),
                    te_weight=getattr(self.config, "bw_te_weight", 1.0),
                    te2_weight=getattr(self.config, "bw_te2_weight", 1.0),
                    zero_threshold=getattr(self.config, "block_lr_zero_threshold", 0.0),
                )
                layer_weights = self._block_weight_manager.apply_to_lora_injector(self.lora_injector)
                frozen_layers = len(self._block_weight_manager.get_frozen_layers())
                active_layers = sum(1 for weight in layer_weights.values() if weight > 0)
                summary = self._block_weight_manager.get_summary()
                self._log(
                    f"Block Weight enabled ({active_layers} active / {frozen_layers} frozen, preset={summary.get('preset', 'custom')})"
                )
            except Exception as e:
                self._block_weight_manager = None
                self._log(f"Block Weight initialization failed: {e}")

        # 统计可训练参数
        trainable_params = self.lora_injector.get_trainable_params()
        if not trainable_params:
            raise ValueError("No trainable adapter parameters remain after applying current Block Weight settings.")
        total_params = sum(p.numel() for p in trainable_params)
        self._log(f"Trainable parameters: {total_params:,}")

        # 将所有 LoRA 层移动到正确的设备和精度
        for _, injected_layer in self.lora_injector.injected_layers.items():
            target_layer = getattr(injected_layer, "lora", injected_layer)
            target_layer.to(device=self.device, dtype=self.dtype)
        self._log(f"LoRA layers moved to {self.device} ({self.dtype})")

        # Triton fused-LoRA acceleration (default-off, enabled via config like
        # FasterDiT SNR). Eligible standard LoRALinear layers get a fused bf16
        # forward; the qkv path additionally fuses a self-attention's q/k/v LoRA
        # into one kernel via the patched-attention _fused_qkv hook. Ineligible
        # layers (DoRA/dropout/Vortex/non-bf16) are skipped and any kernel error
        # self-falls-back to the eager path at call time.
        self._triton_ops_profile = {
            "report": "triton_ops_runtime_profile_v0",
            "enabled": bool(getattr(self.config, "triton_ops_enabled", False)),
            "requested": bool(getattr(self.config, "triton_ops_enabled", False)),
            "available": False,
            "dtype": str(self.dtype).replace("torch.", ""),
            "patched_lora_layers": 0,
            "patched_qkv_blocks": 0,
            "patched_adaln_blocks": 0,
            "fp32_backward": bool(getattr(self.config, "triton_ops_fp32_backward", False)),
            "status": "disabled",
        }
        if getattr(self.config, "triton_ops_enabled", False):
            try:
                from .triton_ops import triton_inject
                from .triton_ops.config import can_run_fused_bf16, describe_gpu, detect_gpu

                gpu_info = detect_gpu()
                self._triton_ops_profile["gpu"] = describe_gpu(gpu_info)
                if self.dtype == torch.bfloat16 and can_run_fused_bf16():
                    fp32_bwd = getattr(self.config, "triton_ops_fp32_backward", False)
                    unet = getattr(self.model, "unet", None)
                    n_patched = 0
                    if getattr(self.config, "triton_ops_inject_lora", True):
                        n_patched = triton_inject.apply(
                            unet,
                            getattr(self.model, "text_encoder_1", None),
                            getattr(self.model, "text_encoder_2", None),
                            fp32_backward=fp32_bwd,
                        )
                    n_qkv = 0
                    if getattr(self.config, "triton_ops_inject_qkv", True):
                        n_qkv = triton_inject.apply_qkv(unet, fp32_backward=fp32_bwd)
                    n_adaln = 0
                    if getattr(self.config, "triton_ops_inject_adaln", True):
                        n_adaln = triton_inject.apply_adaln(unet)
                    self._triton_ops_profile.update(
                        {
                            "available": True,
                            "status": "patched",
                            "patched_lora_layers": int(n_patched),
                            "patched_qkv_blocks": int(n_qkv),
                            "patched_adaln_blocks": int(n_adaln),
                            "inject_lora": bool(getattr(self.config, "triton_ops_inject_lora", True)),
                            "inject_qkv": bool(getattr(self.config, "triton_ops_inject_qkv", True)),
                            "inject_adaln": bool(getattr(self.config, "triton_ops_inject_adaln", True)),
                        }
                    )
                    self._log(
                        f"Triton fused LoRA enabled: {n_patched} layers patched, "
                        f"{n_qkv} self-attn q/k/v fused, {n_adaln} adaln blocks fused "
                        f"[{describe_gpu(gpu_info)}]"
                    )
                    if not gpu_info.is_ada_or_newer:
                        self._log(
                            "Triton ops tuned for Ada (SM 8.9+); this GPU may see smaller gains."
                        )
                else:
                    self._triton_ops_profile.update(
                        {
                            "status": "unavailable",
                            "reason": f"dtype={self.dtype}, bf16_capable={can_run_fused_bf16()}",
                        }
                    )
                    self._log(
                        f"Triton ops requested but unavailable "
                        f"(dtype={self.dtype}, bf16_capable={can_run_fused_bf16()}); using eager LoRA."
                    )
            except Exception as e:  # never let acceleration setup break training
                self._triton_ops_profile.update(
                    {"status": "init_failed", "reason": f"{type(e).__name__}: {e}"}
                )
                self._log(f"Triton ops init failed ({e}); using eager LoRA.")

        adapter_profile = self._refresh_adapter_runtime_profile(model_arch)
        if adapter_profile.get("enabled"):
            self._log(
                "Adapter runtime profile: "
                f"method={adapter_profile.get('adapter_method')}, "
                f"layers={adapter_profile.get('injected_layer_count')}, "
                f"types={adapter_profile.get('layer_types')}"
            )
        if model_arch in {"anima", "newbie"}:
            self._mark_runtime_phase(f"{model_arch}_adapter_to_device", log=False)

        self._maybe_apply_per_block_compile()

        if model_arch == "anima":
            self._prepare_anima_dit_runtime_guardrails()

        if model_arch == "newbie":
            self._prepare_newbie_dit_runtime_guardrails()

        if model_arch == "sdxl" and str(getattr(self.config, "sdxl_unet_backend", "") or "").strip().lower() == "lulynx_native":
            if self.config.train_unet and self.model is not None and hasattr(self.model, "unet"):
                trainable_ids = {id(param) for param in trainable_params}
                frozen_base_params = 0
                for param in self.model.unet.parameters():
                    if id(param) in trainable_ids:
                        param.requires_grad_(True)
                    elif param.requires_grad:
                        param.requires_grad_(False)
                        frozen_base_params += 1
                if frozen_base_params:
                    self._log(f"Native SDXL LoRA base parameters frozen: {frozen_base_params}")
            residency_mode = str(getattr(self.config, "lulynx_weight_residency", "resident") or "resident")
            if residency_mode.strip().lower() != "resident":
                try:
                    from .native_unet.weight_residency import apply_weight_residency

                    residency_min_params = max(int(getattr(self.config, "lulynx_weight_residency_min_params", 0) or 0), 0)
                    pcie_transfer_format = str(getattr(self.config, "pcie_transfer_format", "off") or "off").strip().lower()
                    if pcie_transfer_format in {"off", "none", "disabled"}:
                        pcie_transfer_format = ""
                    report = apply_weight_residency(
                        self.model.unet,
                        mode=residency_mode,
                        min_parameter_count=residency_min_params,
                        transfer_format=pcie_transfer_format or None,
                        pcie_delta_cache_enabled=bool(getattr(self.config, "pcie_delta_cache_enabled", False)),
                        pcie_delta_cache_mode=str(getattr(self.config, "pcie_delta_cache_mode", "observe") or "observe"),
                        pcie_delta_cache_budget_mb=max(float(getattr(self.config, "pcie_delta_cache_budget_mb", 256.0) or 0.0), 0.0),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    self._native_weight_residency_profile = report.as_dict()
                    native_status = getattr(self.model, "native_unet_status", None)
                    if isinstance(native_status, dict):
                        native_status["weight_residency"] = self._native_weight_residency_profile
                        setattr(self.model, "native_unet_status", native_status)
                    if isinstance(getattr(self, "_native_unet_status", None), dict):
                        self._native_unet_status["weight_residency"] = self._native_weight_residency_profile
                    self._attach_memory_runtime_profiles_to_training_loop()
                    self._log(
                        "Native weight residency: "
                        f"mode={report.mode}, active_linear={report.active_linear_count}/{report.managed_linear_count}, "
                        f"active_conv2d={report.active_conv2d_count}/{report.managed_conv2d_count}, "
                        f"min_params={report.min_parameter_count}, skipped_small={report.skipped_small_count}, "
                        f"cpu_params={report.cpu_parameter_mb:.1f}MB, "
                        f"transfer_format={report.transfer_format}, "
                        f"packed_linear={report.transfer_packed_linear_count}, "
                        f"transfer_h2d={report.transfer_h2d_mb:.1f}MB"
                    )
                except Exception as exc:
                    self._native_weight_residency_profile = {"mode": residency_mode, "error": str(exc)}
                    self._attach_memory_runtime_profiles_to_training_loop()
                    self._log(f"Native weight residency skipped: {exc}")

        if model_arch == "anima" and bool(getattr(self.config, "easy_control_enabled", False)):
            from .easy_control_dit import EasyControl, EasyControlConfig
            self._easy_control = EasyControl(
                EasyControlConfig(
                    in_channels=max(int(getattr(self.config, "easy_control_channels", 3) or 3), 1),
                    scale=float(getattr(self.config, "easy_control_scale", 1.0) or 1.0),
                )
            ).to(device=self.device, dtype=self.dtype)
            trainable_params.extend(self._easy_control.get_trainable_params())
            self._log(f"EasyControl enabled: extra params={sum(p.numel() for p in self._easy_control.get_trainable_params()):,}")

        if model_arch == "anima" and bool(getattr(self.config, "ip_adapter_enabled", False)):
            from .anima_ip_adapter import AnimaIPAdapter, IPAdapterConfig
            def _missing_ip_encoder(_image):
                raise RuntimeError("ip_adapter_enabled requires cached ip_adapter_image_features or a configured vision encoder.")
            self._ip_adapter = AnimaIPAdapter(
                _missing_ip_encoder,
                IPAdapterConfig(
                    encoder_dim=int(getattr(self.config, "ip_adapter_encoder_dim", 1024) or 1024),
                    cond_dim=int(getattr(self.config, "ip_adapter_cond_dim", 1152) or 1152),
                    num_image_tokens=int(getattr(self.config, "ip_adapter_num_image_tokens", 16) or 16),
                    scale=float(getattr(self.config, "ip_adapter_scale", 1.0) or 1.0),
                    cond_mode=str(getattr(self.config, "ip_adapter_cond_mode", "concat") or "concat"),
                ),
            ).to(device=self.device, dtype=self.dtype)
            trainable_params.extend(self._ip_adapter.get_trainable_params())
            self._log(f"Anima IP-Adapter projector enabled: extra params={sum(p.numel() for p in self._ip_adapter.get_trainable_params()):,}")

        if model_arch == "anima" and bool(getattr(self.config, "easycontrol_v2_enabled", False)):
            # EasyControl v2 two-stream consumption: install the executable-subset
            # patch so the target stream attends to the condition stream, add the
            # adapter to the optimizer, and feed condition per step (handler).
            # The two-stream patch only supports the *faithful* executable subset
            # (real q/k/v/output + adaLN + 3D RoPE); on any other unet shape we
            # log a skip rather than installing on an incompatible module.
            unet = self.model.unet
            if not bool(getattr(unet, "is_anima_executable_subset", False)):
                self._log("EasyControl v2 enabled but unet is not a faithful executable subset; two-stream patch skipped.")
            else:
                from .easycontrol_v2_adapter import EasyControlV2Adapter, EasyControlV2AdapterConfig
                from .easycontrol_v2_anima_patch import install_easycontrol_v2_anima_executable_subset_patch
                blocks = list(getattr(getattr(unet, "net", None), "blocks", []) or [])
                hidden = int(blocks[0].self_attn.q_proj.weight.shape[0])
                num_blocks = len(blocks)
                self._easycontrol_v2_adapter = EasyControlV2Adapter(
                    EasyControlV2AdapterConfig(
                        hidden_size=hidden,
                        cond_channels=int(getattr(self.config, "easycontrol_v2_cond_channels", 16) or 16),
                        cond_lora_rank=int(getattr(self.config, "easycontrol_v2_cond_lora_rank", 8) or 8),
                        num_blocks=num_blocks,
                        # Gentle-but-active start: exp(-4) condition mass. Disabled-path
                        # parity is guaranteed by *not installing*; once opted in this is
                        # a trainable adapter the optimizer opens further.
                        b_cond_init=-4.0,
                        init_zero_out=True,
                    )
                ).to(device=self.device, dtype=self.dtype)
                self._easycontrol_v2_patch_handle = install_easycontrol_v2_anima_executable_subset_patch(
                    unet, self._easycontrol_v2_adapter
                )
                trainable_params.extend(self._easycontrol_v2_adapter.get_trainable_params())
                self._log(
                    "EasyControl v2 two-stream enabled: "
                    f"patched blocks={self._easycontrol_v2_patch_handle.block_count}, hidden={hidden}, "
                    f"extra params={sum(p.numel() for p in self._easycontrol_v2_adapter.get_trainable_params()):,}"
                )

        if bool(getattr(self.config, "reft_enabled", False)):
            raw_targets = str(getattr(self.config, "reft_target_modules", "") or "").strip()
            target_modules = [part.strip() for part in re.split(r"[,\n;]+", raw_targets) if part.strip()]
            if target_modules:
                from .reft import install_reft, get_reft_params
                self._reft_interventions = install_reft(
                    self.model.unet,
                    target_modules,
                    rank=max(int(getattr(self.config, "reft_rank", 8) or 8), 1),
                    init_scale=float(getattr(self.config, "reft_init_scale", 0.0) or 0.0),
                )
                reft_params = get_reft_params(self.model.unet)
                trainable_params.extend(reft_params)
                self._log(f"ReFT enabled: targets={len(self._reft_interventions)}, extra params={sum(p.numel() for p in reft_params):,}")
            else:
                self._log("ReFT enabled but reft_target_modules is empty; skipping ReFT installation.")

        if (bool(getattr(self.config, "repa_enabled", False)) and float(getattr(self.config, "repa_loss_weight", 0.0) or 0.0) > 0.0) or bool(getattr(self.config, "softrepa_enabled", False)):
            projection_dim = int(getattr(self.config, "repa_projection_dim", 0) or 0)
            if projection_dim > 0:
                from .repa import REPAFeatureProjector
                self._repa_projector = REPAFeatureProjector(hidden_dim=0, projection_dim=projection_dim).to(device=self.device, dtype=self.dtype)
                trainable_params.extend(list(self._repa_projector.parameters()))
                self._log(f"REPA projector enabled: projection_dim={projection_dim}")
            else:
                self._log("REPA enabled without projection projector; using direct feature alignment.")

        # ── Prefix/Postfix Soft-Prompt Tuning (#113) ──
        prefix_length = int(getattr(self.config, "prefix_tuning_length", 0) or 0)
        postfix_length = int(getattr(self.config, "postfix_tuning_length", 0) or 0)
        if prefix_length > 0 or postfix_length > 0:
            from .prefix_tuning import install_prefix_tuning, get_prefix_tuning_params
            target_family = str(getattr(self.config, "model_type", "") or getattr(self.config, "model_arch", ""))
            init = str(getattr(self.config, "prefix_tuning_init", "normal") or "normal")
            install_prefix_tuning(
                self.model,
                prefix_length=prefix_length,
                postfix_length=postfix_length,
                target_family=target_family,
                init=init,
            )
            # Extend trainable params with soft-prompt parameters
            prefix_params = get_prefix_tuning_params(self.model)
            trainable_params.extend(prefix_params)
            total_prefix_params = sum(p.numel() for p in prefix_params)
            self._log(f"Prefix/Postfix tuning: prefix={prefix_length}, postfix={postfix_length}, "
                      f"init={init}, extra params={total_prefix_params:,}")

        # ── Shared runtime features: adapter CPU residency & attention profile ──
        self._adapter_cpu_residency = None
        if getattr(self.config, "vram_swap_to_ram", False):
            from .memory_optimizations import AdapterCPUResidency
            self._adapter_cpu_residency = AdapterCPUResidency(device=self.device)
            residency_params = list(trainable_params)
            get_residency_params = getattr(self.lora_injector, "get_residency_params", None)
            if callable(get_residency_params):
                residency_params.extend(get_residency_params())
            registered = self._adapter_cpu_residency.register_parameters(residency_params)
            savings_mb = self._adapter_cpu_residency.estimate_vram_savings_mb()
            self._log(f"Adapter CPU residency: {registered} params registered, ~{savings_mb:.1f} MB VRAM savings when idle")

        self._attention_profile = None
        if getattr(self.config, "experimental_attention_profile_enabled", False):
            from .runtime_optimizations import AttentionProfile
            self._attention_profile = AttentionProfile.from_config(self.config)
            self._attention_profile.launcher_attention_backend = getattr(
                self.runtime_optimization_plan,
                "attention_backend",
                self._attention_profile.launcher_attention_backend,
            )
            if self._attention_profile.is_active:
                self._log(
                    "Attention profile: sliding window enabled, "
                    f"window_size={self._attention_profile.window_size}, "
                    f"backend={self._attention_profile.backend}"
                )

        # Apply cross-attention fused KV if requested
        fused_projection_mode = str(
            getattr(self.config, "fused_projection_memory_mode", "keep_original") or "keep_original"
        ).strip().lower().replace("-", "_")
        fused_projection_profile: Dict[str, Any] = {
            "requested_memory_mode": fused_projection_mode,
            "cross_attn_fused_kv": bool(getattr(self.config, "cross_attn_fused_kv", False)),
            "anima_fused_qkv": bool(getattr(self.config, "anima_fused_qkv", False)),
        }
        if getattr(self.config, "cross_attn_fused_kv", False):
            from .runtime_optimizations import apply_cross_attn_fused_kv
            apply_cross_attn_fused_kv(self.config, self.model, self.runtime_optimization_plan)
            fused_projection_profile["sdxl_cross_kv_applied"] = True

        # Anima DiT fused KV (cross-attention) — always apply when cross_attn_fused_kv is set
        if getattr(self.config, "cross_attn_fused_kv", False):
            from .runtime_optimizations import apply_anima_fused_kv
            fused_projection_profile["anima_cross_kv_count"] = apply_anima_fused_kv(
                self.model,
                self.runtime_optimization_plan,
                memory_mode=fused_projection_mode,
            )

        # Anima DiT fused QKV (self-attention)
        if getattr(self.config, "anima_fused_qkv", False):
            from .runtime_optimizations import apply_anima_fused_qkv
            fused_projection_profile["anima_self_qkv_count"] = apply_anima_fused_qkv(
                self.model,
                self.runtime_optimization_plan,
                memory_mode=fused_projection_mode,
            )
        if fused_projection_profile["cross_attn_fused_kv"] or fused_projection_profile["anima_fused_qkv"]:
            fused_projection_profile["resolved_memory_mode"] = fused_projection_mode
            self._fused_projection_profile = fused_projection_profile

        if self._attention_profile is not None and self._attention_profile.is_active and model_arch in {"anima", "newbie"}:
            try:
                from .anima_attention import patch_anima_attention

                patched = patch_anima_attention(
                    self.model.unet if getattr(self.model, "unet", None) is not None else self.model,
                    backend=str(getattr(self.runtime_optimization_plan, "attention_backend", "sdpa") or "sdpa"),
                    split_chunks=int(getattr(self.runtime_optimization_plan, "attention_split_chunks", 0) or 0),
                    amd_sdpa_slice_trigger_gb=float(
                        getattr(self.runtime_optimization_plan, "amd_sdpa_slice_trigger_gb", 0.0) or 0.0
                    ),
                    amd_sdpa_slice_target_gb=float(
                        getattr(self.runtime_optimization_plan, "amd_sdpa_slice_target_gb", 0.0) or 0.0
                    ),
                    early_deletion=bool(getattr(self.runtime_optimization_plan, "attention_early_deletion", False)),
                    attention_profile=self._attention_profile,
                )
                self._refresh_attention_runtime_profile(
                    model_arch=model_arch,
                    route=model_arch,
                    patched=patched,
                    patch_target="dit_attention",
                    applied=patched > 0,
                    skip_reason="no_dit_attention_modules_patched" if patched <= 0 else "",
                    source="dit_attention_profile",
                )
                self._attach_attention_runtime_profile_to_training_loop()
                self._log(
                    "Attention profile wired to DiT attention: "
                    f"patched={patched}, window_size={self._attention_profile.window_size}, "
                    f"backend={self._attention_profile.backend}"
                )
            except Exception as exc:
                self._refresh_attention_runtime_profile(
                    model_arch=model_arch,
                    route=model_arch,
                    patch_target="dit_attention",
                    applied=False,
                    error=f"{type(exc).__name__}: {exc}",
                    source="dit_attention_profile",
                )
                self._attach_attention_runtime_profile_to_training_loop()
                self._log(f"Attention profile live wiring skipped: {exc}")

        self._apply_diffusers_unet_attention_profile(model_arch)

        # Base Weight — diff-training base LoRA (supports comma-separated paths + multipliers)
        from .base_lora_weights import load_base_lora_weights

        load_base_lora_weights(self.config, self.lora_injector, log_fn=self._log)

        # Adapter pre-fusion: merge an existing LoRA into base model before new training
        _prefuse_path = str(getattr(self.config, "prefuse_adapter_path", "") or "").strip()
        if _prefuse_path:
            from .adapter_prefusion import prefuse_adapter_into_model
            _prefuse_scale = float(getattr(self.config, "prefuse_adapter_scale", 1.0) or 1.0)
            _fused = prefuse_adapter_into_model(self.model, _prefuse_path, scale=_prefuse_scale)
            self._log(f"Pre-fused {_fused} layers from {_prefuse_path} (scale={_prefuse_scale})")

        network_weights_path = getattr(self.config, "network_weights_path", "")
        if network_weights_path:
            network_path = Path(network_weights_path)
            if network_path.is_file() and hasattr(self.lora_injector, "load_lora"):
                self._log(f"Loading initial network weights from {network_path}")
                self.lora_injector.load_lora(str(network_path))
            else:
                self._log(
                    f"Network weights path {network_weights_path} not found or injector has no load_lora, skipping"
                )

        # Resume Checkpoint (Weights)
        if self.config.resume_path:
            resume_path = Path(self.config.resume_path)
            if (
                resume_path.is_file()
                and resume_path.suffix.lower() in {EXT_SAFETENSORS, EXT_PT, ".ckpt"}
                and not resume_path.stem.endswith("-state")
                and hasattr(self.lora_injector, "load_lora")
            ):
                self._log(f"Resuming adapter weights from {resume_path}")
                self.lora_injector.load_lora(str(resume_path))
            else:
                self._log(f"Resume path {resume_path} is not a supported adapter file, skipping weight load (may be state only.)")

        # DiT Adapter Loading (Anima-specific, for pre-trained DiT adapter weights)
        dit_adapter = str(getattr(self.config, "anima_dit_adapter_path", "") or "").strip()
        if dit_adapter and model_arch == "anima" and hasattr(self.lora_injector, "load_lora"):
            dit_path = Path(dit_adapter)
            if dit_path.is_file() and dit_path.suffix.lower() in {EXT_SAFETENSORS, EXT_PT, ".ckpt"}:
                self._log(f"Loading DiT adapter weights from {dit_adapter}")
                self.lora_injector.load_lora(str(dit_path))
            else:
                self._log(f"DiT adapter path {dit_adapter} is not a valid weight file, skipping.")

        if bool(getattr(self.config, "compression_companion_enabled", False)):
            self._apply_compression_companion()
            if getattr(self.config, "weight_compression_enabled", False) or getattr(self.config, "fp8_base", False):
                self._apply_weight_compression()

        if model_arch in {"anima", "newbie"}:
            self._ensure_native_family_training_ready()
        self._mark_runtime_phase("adapter_prepare")
        return self

    def _build_anima_grouped_param_groups(self) -> Optional[List[Dict]]:
        """Build Anima-specific parameter groups with per-module LR.

        Returns None if Anima grouped LR is not configured, so the caller
        can fall back to the default single-LR path.
        """
        model_arch = self._model_arch_value()
        if model_arch != "anima":
            return None

        # Collect all trainable params from LoRA injector
        all_params = self.lora_injector.get_trainable_params()
        if not all_params:
            return None

        # Build name->param mapping from injected layers
        param_to_name: Dict[int, str] = {}
        injected_layers = getattr(self.lora_injector, "injected_layers", {})
        if isinstance(injected_layers, dict) and injected_layers:
            for layer_name, layer in injected_layers.items():
                for p in layer.parameters():
                    if p.requires_grad:
                        param_to_name[id(p)] = layer_name
        elif is_anima_full_finetune(self.config) and self.model is not None and getattr(self.model, "unet", None) is not None:
            param_to_name = collect_trainable_param_name_map(self.model.unet)

        return build_anima_grouped_param_groups(
            config=self.config,
            trainable_params=all_params,
            param_to_name=param_to_name,
            log=self._log,
        )

    def _parse_anima_progressive_ranges(self, value: Any) -> set[int] | None:
        text = str(value or "").strip().lower()
        if not text or text in {"all", "*"}:
            return None
        result: set[int] = set()
        for chunk in text.replace(";", ",").split(","):
            part = chunk.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                try:
                    start = int(left.strip())
                    end = int(right.strip())
                except ValueError:
                    continue
                if end < start:
                    start, end = end, start
                result.update(range(start, end + 1))
                continue
            try:
                result.add(int(part))
            except ValueError:
                continue
        return result

    def _parse_anima_progressive_schedule(self) -> list[tuple[int, set[int] | None]]:
        if not bool(getattr(self.config, "anima_progressive_full_finetune_enabled", False)):
            return []
        raw = str(getattr(self.config, "anima_progressive_full_finetune_schedule", "") or "").strip()
        if not raw:
            default_range = self._parse_anima_progressive_ranges(
                getattr(self.config, "anima_progressive_full_finetune_default", "all")
            )
            return [(0, default_range)]
        entries: list[tuple[int, set[int] | None]] = []
        chunks = re.split(r"[;|]|,(?=\s*\d+\s*[:=])", raw)
        for chunk in chunks:
            part = chunk.strip()
            if not part:
                continue
            if ":" in part:
                step_text, range_text = part.split(":", 1)
            elif "=" in part:
                step_text, range_text = part.split("=", 1)
            else:
                step_text, range_text = "0", part
            try:
                step = max(int(step_text.strip()), 0)
            except ValueError:
                continue
            entries.append((step, self._parse_anima_progressive_ranges(range_text)))
        entries.sort(key=lambda item: item[0])
        return entries

    def _apply_anima_progressive_full_finetune(self, *, global_step: int = 0, reason: str = "") -> None:
        if not is_anima_full_finetune(self.config):
            return
        schedule = self._parse_anima_progressive_schedule()
        if not schedule:
            return
        unet = getattr(getattr(self, "model", None), "unet", None)
        blocks = list(getattr(getattr(unet, "net", None), "blocks", []) or [])
        if not blocks:
            self._anima_full_finetune_experiments_profile["progressive_full_finetune"] = {
                "enabled": True,
                "status": "skipped",
                "reason": "no_native_blocks",
            }
            self._attach_anima_full_finetune_experiments_to_training_loop()
            return
        active: set[int] | None = schedule[0][1]
        active_step = schedule[0][0]
        for step, block_range in schedule:
            if int(global_step) >= step:
                active = block_range
                active_step = step
        active_indices = set(range(len(blocks))) if active is None else {idx for idx in active if 0 <= idx < len(blocks)}
        for idx, block in enumerate(blocks):
            block.requires_grad_(idx in active_indices)
        for name in ("x_embedder", "t_embedder", "t_embedding_norm", "final_layer"):
            module = getattr(getattr(unet, "net", None), name, None)
            if module is not None and hasattr(module, "requires_grad_"):
                module.requires_grad_(True)
        trainable = sum(param.numel() for param in unet.parameters() if param.requires_grad)
        total = sum(param.numel() for param in unet.parameters())
        profile = {
            "enabled": True,
            "status": "active",
            "active_step": int(active_step),
            "active_block_count": len(active_indices),
            "total_block_count": len(blocks),
            "active_blocks": sorted(active_indices)[:64],
            "trainable_params": int(trainable),
            "total_params": int(total),
            "trainable_ratio": round(float(trainable) / float(total or 1), 6),
            "schedule": [
                {"step": int(step), "blocks": "all" if ranges is None else sorted(ranges)}
                for step, ranges in schedule
            ],
            "reason": reason,
        }
        previous = self._anima_full_finetune_experiments_profile.get("progressive_full_finetune", {})
        if previous.get("active_blocks") != profile["active_blocks"]:
            self._log(
                "Anima progressive full finetune active blocks: "
                f"{profile['active_block_count']}/{profile['total_block_count']} "
                f"(step={global_step}, trainable_ratio={profile['trainable_ratio']:.3f})"
            )
        self._anima_full_finetune_experiments_profile["progressive_full_finetune"] = profile
        self._attach_anima_full_finetune_experiments_to_training_loop()

    def _refresh_anima_full_finetune_experiments_profile(self) -> Dict[str, Any]:
        """Collect telemetry for Anima full-finetune 16G research switches."""

        profile = dict(getattr(self, "_anima_full_finetune_experiments_profile", {}) or {})
        if is_anima_full_finetune(self.config):
            profile.setdefault("enabled", True)

        optimizer = getattr(getattr(self, "training_loop", None), "optimizer", None)
        if optimizer is not None:
            if hasattr(optimizer, "get_profile"):
                try:
                    opt_profile = optimizer.get_profile()
                    if isinstance(opt_profile, dict):
                        key = "optimizer_state_paging" if type(optimizer).__name__ == "OptimizerStatePagingWrapper" else "optimizer"
                        profile[key] = opt_profile
                except Exception as exc:
                    profile["optimizer_profile_error"] = f"{type(exc).__name__}: {exc}"
            base_optimizer = getattr(optimizer, "_base", None)
            if base_optimizer is not None and hasattr(base_optimizer, "get_profile"):
                try:
                    base_profile = base_optimizer.get_profile()
                    if isinstance(base_profile, dict):
                        profile["optimizer"] = base_profile
                except Exception as exc:
                    profile["base_optimizer_profile_error"] = f"{type(exc).__name__}: {exc}"

        loop = getattr(self, "training_loop", None)
        if loop is not None and hasattr(loop, "get_memory_experiment_profile"):
            try:
                memory_profile = loop.get_memory_experiment_profile()
                if isinstance(memory_profile, dict):
                    profile.update(memory_profile)
            except Exception as exc:
                profile["memory_experiment_profile_error"] = f"{type(exc).__name__}: {exc}"

        if bool(getattr(self.config, "optimizer_state_paging_enabled", False)):
            profile.setdefault(
                "optimizer_state_paging",
                {
                    "enabled": True,
                    "status": "requested",
                    "min_tensor_mb": float(getattr(self.config, "optimizer_state_paging_min_tensor_mb", 1.0) or 1.0),
                    "pin_memory": bool(getattr(self.config, "optimizer_state_paging_pin_memory", False)),
                },
            )

        if bool(getattr(self.config, "anima_rematerializable_block_enabled", False)):
            profile["rematerializable_block"] = {
                "enabled": True,
                "mode": str(getattr(self.config, "anima_rematerializable_block_mode", "profile_only") or "profile_only"),
                "status": "profile_only",
                "current_safe_path": "anima_block_checkpointing",
                "reason": "Reversible DiT block rewrite is kept as a research boundary; selective checkpointing is the safe runtime approximation.",
            }

        self._anima_full_finetune_experiments_profile = profile
        self._attach_anima_full_finetune_experiments_to_training_loop()
        return profile

    # --- Optimizer/scheduler factory ---------------------------------------
    # _parse_custom_args .. _create_scheduler now live in
    # trainer_optimizer_factory.py (TrainerOptimizerFactoryMixin); they stay
    # bound methods of LulynxTrainer via its class bases (MRO).


    def start(self):
        """开始训练"""
        if self.is_running:
            self._log("Training already running")
            return False

        if not self.model and not getattr(self, "_ti_mode", False):
            self._maybe_apply_auto_vram_enhancement()

        if (not self.model and not getattr(self, "_ti_mode", False)) or (not self.lora_injector and not self.config.semantic_tuner_enabled):
            self.prepare()

        self._apply_seed()
        self._apply_gpu_power_limit_if_requested()
        self._apply_gpu_clock_lock_if_requested()

        # Initialize DDP if multi_gpu is enabled
        self._setup_ddp()

        self.is_running = True
        self._should_stop = False
        get_orchestra().reset()
        self._initialize_logging_runtime()

        try:
            with build_sdpa_backend_context(self.runtime_optimization_plan or build_runtime_optimization_plan(self.config)):
                self._run_training()
            success = not self._should_stop or bool(getattr(self, "_completed_by_step_limit", False))
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            self._log(f"Training failed: {e}")
            success = False
        finally:
            self.is_running = False
            self._reset_gpu_clock_lock_if_applied()
            self._finalize_logging_runtime()
            self._cleanup_ddp()
            if self.on_complete:
                self.on_complete(success)

        return success

    def _run_ti_training(self):
        """Textual Inversion 专用训练路径"""
        from .dataset_loader import CaptionDataset, create_dataloader

        self._log("Starting Textual Inversion training...")

        ti = self._ti_trainer
        self._completed_by_step_limit = False
        dataset = CaptionDataset(
            data_dir=self.config.train_data_dir,
            resolution=self.config.resolution,
            caption_extension=self.config.caption_extension,
            enable_bucket=self.config.enable_bucket,
            image_decode_backend=getattr(self.config, "image_decode_backend", "pil"),
            image_decode_cache_size=getattr(self.config, "image_decode_cache_size", 0),
        )
        dataloader = create_dataloader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 1),
            shuffle=True,
            num_workers=getattr(self.config, "dataloader_num_workers", 0),
            persistent_workers=bool(getattr(self.config, "persistent_data_loader_workers", False)),
        )

        num_epochs = max(int(getattr(self.config, "max_train_epochs", 1)), 1)
        save_every = self._resolve_epoch_save_interval(num_epochs)
        max_steps = max(int(getattr(self.config, "max_train_steps", 0) or 0), 0)
        global_step = 0

        device = self.device
        dtype = self.dtype

        for epoch in range(num_epochs):
            if self._should_stop:
                break
            if hasattr(dataset, "set_current_epoch"):
                dataset.set_current_epoch(epoch)

            epoch_loss = 0.0
            num_steps = 0

            for batch in dataloader:
                if self._should_stop:
                    break
                if max_steps > 0 and global_step >= max_steps:
                    self._completed_by_step_limit = True
                    self._should_stop = True
                    break

                images = batch["images"].to(device, dtype=dtype)
                captions = batch["captions"]

                # Encode latents (no grad — only done once per batch)
                with torch.no_grad():
                    latents = ti.vae.encode(images).latent_dist.sample()
                    latents = latents * ti.vae.config.scaling_factor

                # train_step() calls update_text_encoder_embeddings() first,
                # then encodes text freshly, so gradients flow to concept_embedding.
                loss = ti.train_step({
                    "latents": latents,
                    "captions": captions,
                })
                epoch_loss += loss
                num_steps += 1
                global_step += 1

                if hasattr(self, "_on_step_end") and self._on_step_end:
                    self._on_step_end(global_step, loss, {})

                if max_steps > 0 and global_step >= max_steps:
                    self._completed_by_step_limit = True
                    self._should_stop = True
                    break

            avg_loss = epoch_loss / max(num_steps, 1)
            self._log(f"TI epoch {epoch + 1}/{num_epochs} — avg loss: {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                ti.update_text_encoder_embeddings()
                self._save_model(epoch + 1)
            self._maybe_cooldown_after_epoch(epoch, num_epochs)

        completed_by_step_limit = max_steps > 0 and global_step >= max_steps
        if not self._should_stop or completed_by_step_limit:
            ti.update_text_encoder_embeddings()
            self._save_model(num_epochs, final=True)
            self._maybe_save_final_training_state(num_epochs)
            self._write_run_manifest("completed", epoch=int(num_epochs))

        self._log("Textual Inversion training complete.")

    def _run_training(self):
        """执行训练"""
        # Textual Inversion has its own training loop
        if getattr(self, "_ti_mode", False):
            return self._run_ti_training()

        self._log("Creating dataset...")

        # Vortex Cleanup (Ensure fresh state)
        if hasattr(self.config, 'vortex_enabled') and self.config.vortex_enabled:
            # Maybe clear cache?
            pass

        # Newbie cache-only mode: rebuild cache then exit
        model_arch = self._model_arch_value()
        newbie_force_cache = (
            model_arch == "newbie"
            and getattr(self.config, "newbie_force_cache_only", False)
        )
        newbie_rebuild = getattr(self.config, "newbie_rebuild_cache", False)

        if newbie_rebuild and model_arch == "newbie" and not self._newbie_cache_rebuild_handled_in_prepare:
            deleted = self._delete_newbie_cache_artifacts()
            data_dir = Path(str(getattr(self.config, "train_data_dir", "") or ""))
            self._log(f"Newbie rebuild_cache: deleted {deleted} cache artifacts from {data_dir}")
            self._newbie_rebuild_cache = True
        else:
            self._newbie_rebuild_cache = False

        if model_arch == "newbie" and not self._newbie_cache_rebuild_handled_in_prepare:
            self._maybe_build_newbie_cache(force=bool(newbie_rebuild))

        # Resolve caption-length bucket size (Newbie feature, generic field)
        caption_bucket = int(getattr(self.config, "newbie_caption_length_bucket_size", 0) or 0)

        anima_cached_training = (
            model_arch == "anima"
            and bool(getattr(self.model, "anima_cached_training_ready", False))
            and self._has_anima_cached_training_data()
        )
        anima_online_cache = model_arch == "anima" and bool(getattr(self.config, "anima_online_cache", False))
        if anima_online_cache:
            anima_cached_training = True
        if (
            model_arch == "anima"
            and bool(getattr(self.config, "anima_cached_training", True))
            and not anima_cached_training
        ):
            raise RuntimeError(
                "Anima cache-first training was requested, but paired cache files were not ready. "
                "Expected *_anima.npz latents plus matching *_anima_te.npz text-conditioning caches "
                "and a native executable Anima model. Use native_cache_mode=online_cache to generate missing cache "
                "with frozen VAE/text encoders before each DiT step. Raw online Anima training is intentionally blocked."
            )
        newbie_cached_training = (
            model_arch == "newbie"
            and bool(getattr(self.config, "use_cache", False))
            and self._has_newbie_cached_training_data()
        )
        compile_contract = getattr(self, "compile_contract_decision", None)
        if bool(getattr(compile_contract, "cache_first_required", False)):
            if model_arch == "anima" and not anima_cached_training:
                raise RuntimeError(
                    "[cache-contract] Anima compile route requires paired latent/text cache data, "
                    "but cache-first training is not ready."
                )
            if model_arch == "newbie" and not newbie_cached_training:
                raise RuntimeError(
                    "[cache-contract] Newbie compile route requires cache-first training, "
                    "but cached training data is not ready."
                )
            self._log(
                f"[cache-contract] route={model_arch} cache_first=ok mode=compile-ready"
            )
        self._resolve_data_backend_profile(
            model_arch=model_arch,
            anima_cached_training=anima_cached_training,
            newbie_cached_training=newbie_cached_training,
        )
        self._newbie_caption_bucket_size = caption_bucket
        compile_drop_last = bool(
            getattr(getattr(self, "compile_contract_decision", None), "static_drop_last", False)
        )

        self._lora_staged_resolution_plan = []
        self._lora_staged_resolution_active_index = -1
        self._lora_staged_resolution_enabled_runtime = False
        self._lora_staged_resolution_compile_drop_last = compile_drop_last
        self._lora_staged_resolution_sdxl_cache_first = False
        lora_initial_batch_size = max(int(getattr(self.config, "batch_size", 1) or 1), 1)
        lora_staged_plan = self._build_lora_staged_resolution_plan(
            model_arch=model_arch,
            anima_cached_training=anima_cached_training,
            newbie_cached_training=newbie_cached_training,
        )
        if lora_staged_plan:
            initial_index, initial_stage = self._select_lora_staged_resolution_stage(0)
            if initial_stage is None:
                raise RuntimeError("LoRA staged resolution enabled but no stage plan was produced.")
            self._set_lora_stage_resolution(initial_stage.resolution)
            self._lora_staged_resolution_active_index = initial_index
            self._lora_staged_resolution_enabled_runtime = True
            lora_initial_batch_size = max(int(initial_stage.batch_size or lora_initial_batch_size), 1)
            self._log(
                "LoRA staged-resolution dataset plan: "
                f"{stages_to_summary(lora_staged_plan)}, initial_batch={lora_initial_batch_size}"
            )

        # 创建数据集
        caption_training_input = None
        if anima_cached_training:
            # Check if online_cache mode is enabled
            online_cache_enabled = bool(getattr(self.config, "anima_online_cache", False))

            if online_cache_enabled:
                from .anima_online_cache_dataset import AnimaOnlineCacheDataset
                from .anima_cache_builder import AnimaCacheBuilderConfig
                try:
                    encode_bundle = build_anima_cache_encode_bundle(
                        model=self.model,
                        device=self.device,
                        dtype=self.dtype,
                        config=self.config,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Anima online_cache mode requires cache-builder inputs to be ready: "
                        f"{exc}"
                    ) from exc

                cache_config = AnimaCacheBuilderConfig(
                    data_dir=self.config.train_data_dir,
                    output_dir=self.config.train_data_dir,
                    vae_chunk_size=int(getattr(self.config, "anima_vae_chunk_size", 0) or 0),
                    text_token_limit=int(getattr(self.config, "anima_text_token_limit", 0) or 0),
                    include_loss_mask=bool(getattr(self.config, "masked_loss", False)),
                    disk_format=str(getattr(self.config, "latent_cache_disk_format", "npz") or "npz"),
                    disk_dtype=self._resolve_cache_disk_dtype(getattr(self.config, "latent_cache_disk_dtype", "float16")),
                    text_disk_format=str(getattr(self.config, "text_encoder_outputs_cache_disk_format", "npz") or "npz"),
                    text_disk_dtype=self._resolve_cache_disk_dtype(
                        getattr(self.config, "text_encoder_outputs_cache_disk_dtype", "float16")
                    ),
                )

                dataset = AnimaOnlineCacheDataset(
                    data_dir=self.config.train_data_dir,
                    vae_encode_fn=encode_bundle.vae_encode_fn,
                    text_encode_fn=encode_bundle.text_encode_fn,
                    cache_config=cache_config,
                    latent_crop_size=int(getattr(self.config, "anima_cached_latent_crop_size", 0) or 0),
                    text_token_limit=int(getattr(self.config, "anima_cached_text_token_limit", 0) or 0),
                    fixed_text_tokens=int(getattr(self.config, "anima_fixed_text_tokens", 0) or 0),
                    fixed_visual_tokens=int(getattr(self.config, "anima_fixed_visual_tokens", 0) or 0),
                    fixed_qwen3_tokens=int(getattr(self.config, "anima_fixed_qwen3_tokens", 0) or 0),
                    fixed_t5_tokens=int(getattr(self.config, "anima_fixed_t5_tokens", 0) or 0),
                    caption_extension=getattr(self.config, "caption_extension", ".txt"),
                    shuffle_caption=bool(getattr(self.config, "shuffle_caption", False)),
                    shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
                    keep_tokens=int(getattr(self.config, "keep_tokens", 0) or 0),
                    keep_tokens_separator=str(getattr(self.config, "keep_tokens_separator", "") or ""),
                    weighted_captions=bool(getattr(self.config, "weighted_captions", False)),
                    concept_geometry_enabled=bool(getattr(self.config, "concept_geometry_enabled", getattr(self.config, "h_lora_enabled", False))),
                    concept_geometry_path=str(getattr(self.config, "concept_geometry_path", getattr(self.config, "h_lora_geometry_path", "")) or ""),
                    concept_geometry_sampler_mode=str(getattr(self.config, "concept_geometry_sampler_mode", getattr(self.config, "h_lora_sampler_mode", "density_curriculum")) or "density_curriculum"),
                    concept_geometry_loss_weighting=bool(getattr(self.config, "concept_geometry_loss_weighting", getattr(self.config, "h_lora_loss_weighting", False))),
                    concept_geometry_density_power=float(getattr(self.config, "concept_geometry_density_power", getattr(self.config, "h_lora_density_power", 1.0)) or 1.0),
                    concept_geometry_seed=int(getattr(self.config, "seed", 42) or 42),
                    concept_geometry_total_epochs=int(getattr(self.config, "max_train_epochs", 1) or 1),
                    concept_geometry_total_steps=int(getattr(self.config, "max_train_steps", 0) or 0),
                )
                self._log(
                    f"Anima online-cache dataset: {len(dataset)} images (cache generated on-demand, {encode_bundle.summary})"
                )
            else:
                staged_plan = self._build_anima_staged_resolution_plan()
                if staged_plan:
                    initial_index, initial_stage = self._select_anima_staged_resolution_stage(0)
                    if initial_stage is None:
                        raise RuntimeError("Anima staged resolution enabled but no stage plan was produced.")
                    dataset = self._create_anima_cached_dataset(initial_stage.cache_dir)
                    self._anima_staged_resolution_active_index = initial_index
                    self._apply_staged_resolution(initial_stage.resolution)
                    self._log(
                        "Anima staged-resolution cache-first dataset: "
                        f"{len(dataset)} samples, resolution={initial_stage.resolution}, "
                        f"batch={initial_stage.batch_size or getattr(self.config, 'batch_size', 1)}, "
                        f"cache={initial_stage.cache_dir}"
                    )
                else:
                    dataset = self._create_anima_cached_dataset(self.config.train_data_dir)
                self._log(
                    "Anima cache-first dataset: "
                    f"{len(dataset)} paired latent/text cache samples"
                )
            if (
                hasattr(dataset, "is_concept_geometry_enabled")
                and callable(getattr(dataset, "is_concept_geometry_enabled"))
                and dataset.is_concept_geometry_enabled()
                and hasattr(dataset, "get_concept_geometry_summary")
                and callable(getattr(dataset, "get_concept_geometry_summary"))
            ):
                summary = dataset.get_concept_geometry_summary()
                stage_counts = summary.get("stage_counts", {})
                self._log(
                    "[concept-geometry] enabled: "
                    f"path={summary.get('geometry_path', '')}, "
                    f"samples={summary.get('attached_count', 0)}/{summary.get('sample_count', len(dataset))}, "
                    f"core={stage_counts.get('core', 0)}, "
                    f"mid={stage_counts.get('mid', 0)}, "
                    f"edge={stage_counts.get('edge', 0)}, "
                    f"mode={summary.get('sampler_mode', 'density_curriculum')}, "
                    f"loss_weighting={summary.get('loss_weighting', False)}"
                )
                if bool(getattr(self.config, "anima_online_cache", False)):
                    self._log(
                        "[concept-geometry] online-cache may start without stable bucket metadata; "
                        "curriculum sampling falls back to non-bucket batches until cache coverage is built."
                    )
        elif newbie_cached_training:
            from .newbie_cached_dataset import NewbieCachedDataset

            dataset = NewbieCachedDataset(
                data_dir=self.config.train_data_dir,
                latent_crop_size=int(getattr(self.config, "newbie_cached_latent_crop_size", 0) or 0),
                text_token_limit=int(getattr(self.config, "newbie_cached_text_token_limit", 0) or 0),
                caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
                caption_source_nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
                caption_source_tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
                caption_source_trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
                caption_source_empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
                caption_source_trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
            )
            self._log(f"Newbie cache-first dataset: {len(dataset)} cached samples")
        else:
            caption_training_input = self._create_caption_training_input(
                data_dir=self.config.train_data_dir,
                model_arch=model_arch,
                batch_size=lora_initial_batch_size,
                drop_last=compile_drop_last,
            )
            dataset = caption_training_input.dataset
            self._lora_staged_resolution_sdxl_cache_first = caption_training_input.sdxl_cache_first

        self._mark_runtime_phase("dataset_build")
        self._dataset = dataset  # Store for step callbacks (token warmup, etc.)

        if getattr(self.config, "advanced_monitoring_enabled", False) and hasattr(dataset, "samples"):
            try:
                from .dataset_analyzer import DatasetAnalyzer
                _ds_report = DatasetAnalyzer(dataset, self.config.caption_extension).analyze()
                for line in _ds_report.summary_lines():
                    self._log(f"[DatasetAnalysis]{line}")
                if self._tb_writer:
                    self._tb_writer.add_text("dataset/analysis", "\n".join(_ds_report.summary_lines()), 0)
            except Exception as e:
                self._log(f"[DatasetAnalysis] skipped: {e}")

        if compile_drop_last:
            batch_size = lora_initial_batch_size if caption_training_input is not None else max(int(getattr(self.config, "batch_size", 1) or 1), 1)
            dropped = len(dataset) % batch_size
            self._log(
                "[compile-contract] static batch shape enabled: "
                f"DataLoader drop_last=True"
                + (f", last incomplete batch drops {dropped} sample(s)" if dropped else ", no samples dropped")
            )

        if model_arch in {"anima", "newbie"} and hasattr(dataset, "get_token_bucket_summary"):
            try:
                token_bucket_summary = dataset.get_token_bucket_summary()
                self._native_token_bucket_summary = token_bucket_summary
                bucket_count = int(token_bucket_summary.get("bucket_count", 0) or 0)
                bucket_mode = str(token_bucket_summary.get("mode", "unknown") or "unknown")
                buckets = token_bucket_summary.get("buckets", {}) or {}
                top = sorted(
                    buckets.items(),
                    key=lambda item: int((item[1] or {}).get("sample_count", 0) or 0),
                    reverse=True,
                )[:4]
                preview = ", ".join(
                    f"{key}={int((value or {}).get('sample_count', 0) or 0)}"
                    for key, value in top
                )
                self._log(
                    f"Native token buckets: family={model_arch}, mode={bucket_mode}, "
                    f"bucket_count={bucket_count}"
                    + (f", top={preview}" if preview else "")
                )
            except Exception as exc:
                self._native_token_bucket_summary = {"error": f"{type(exc).__name__}: {exc}"}
                self._log(f"Native token bucket summary skipped: {exc}")

        if caption_bucket and model_arch == "newbie":
            # Count how many distinct caption-length buckets were formed
            length_buckets = set()
            for s in dataset.samples:
                tok = getattr(s, "caption_token_length", 0) or 0
                length_buckets.add((tok // caption_bucket) * caption_bucket)
            self._log(
                f"Newbie caption_length_bucket_size={caption_bucket}: "
                f"{len(length_buckets)} distinct length-buckets across {len(dataset)} samples"
            )

        # Newbie force-cache-only: dataset is prepared, skip training
        if newbie_force_cache:
            self._log(
                "Newbie force_cache_only: dataset prepared, skipping training loop. "
                "Cache artifacts (if any) have been written."
            )
            return

        # Prior Preservation: load regularization dataset if configured
        self._reg_dataloader = None
        prior_loss_weight = getattr(self.config, "prior_loss_weight", 0.0)
        instance_prompt = getattr(self.config, "instance_prompt", "") or ""
        class_prompt = getattr(self.config, "class_prompt", "") or ""

        # DreamBooth: auto-generate class images if reg_data_dir is empty
        if prior_loss_weight > 0 and not self.config.reg_data_dir and class_prompt:
            num_class = int(getattr(self.config, "num_class_images", 100))
            auto_dir = Path(self.config.train_data_dir).parent / "class_images"
            self._log(f"DreamBooth: auto-generating {num_class} class images → {auto_dir}")
            try:
                base_model = Path(self.config.base_model_path)
                if model_arch == "sdxl":
                    from diffusers import StableDiffusionXLPipeline as ClassImagePipeline
                else:
                    from diffusers import StableDiffusionPipeline as ClassImagePipeline

                if base_model.is_file():
                    if not hasattr(ClassImagePipeline, "from_single_file"):
                        raise RuntimeError(
                            f"{ClassImagePipeline.__name__} cannot load single-file checkpoints in this diffusers version. "
                            "Set reg_data_dir with prepared class images instead."
                        )
                    pipe = ClassImagePipeline.from_single_file(
                        str(base_model),
                        torch_dtype=self.dtype,
                        safety_checker=None if model_arch != "sdxl" else None,
                    ).to(self.device)
                else:
                    pipe = ClassImagePipeline.from_pretrained(
                        str(base_model),
                        torch_dtype=self.dtype,
                        safety_checker=None if model_arch != "sdxl" else None,
                    ).to(self.device)
                from .dreambooth import ClassImageGenerator
                gen = ClassImageGenerator(
                    pipeline=pipe,
                    class_prompt=class_prompt,
                    num_images=num_class,
                    output_dir=str(auto_dir),
                )
                gen.generate()
                self.config.reg_data_dir = str(auto_dir)
                del pipe
                self._maybe_release_tool_cuda_cache(
                    "dreambooth_class_image_generation_cleanup",
                    collect_gc=True,
                )
                self._log(f"DreamBooth: generated {num_class} class images")
            except Exception as e:
                self._log(f"DreamBooth class image generation failed: {e}")
                self._log("Proceeding without prior preservation.")

        if prior_loss_weight > 0 and self.config.reg_data_dir:
            self._log(f"Prior Preservation enabled (weight={prior_loss_weight})")
            reg_dataset = CaptionDataset(
                data_dir=self.config.reg_data_dir,
                resolution=self._get_dataset_resolution(),
                caption_extension=self.config.caption_extension,
                enable_bucket=self.config.enable_bucket,
                min_bucket_reso=self.config.min_bucket_reso,
                max_bucket_reso=self.config.max_bucket_reso,
                bucket_reso_steps=self.config.bucket_reso_steps,
                bucket_selection_mode=getattr(self.config, "bucket_selection_mode", "aspect"),
                bucket_custom_resos=getattr(self.config, "bucket_custom_resos", ""),
                shuffle_caption=self.config.shuffle_caption,
                shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
                keep_tokens=self.config.keep_tokens,
                keep_tokens_separator=getattr(self.config, "keep_tokens_separator", ""),
                caption_dropout_rate=getattr(self.config, "caption_dropout_rate", 0.0),
                caption_dropout_every_n_epochs=getattr(self.config, "caption_dropout_every_n_epochs", 0),
                tag_dropout_rate=getattr(self.config, "tag_dropout_rate", 0.0),
                caption_tag_dropout_targets=getattr(self.config, "caption_tag_dropout_targets", ""),
                caption_tag_dropout_target_mode=getattr(self.config, "caption_tag_dropout_target_mode", "drop_all"),
                caption_tag_dropout_target_count=getattr(self.config, "caption_tag_dropout_target_count", 1),
                caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
                caption_source_nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
                caption_source_tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
                caption_source_trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
                caption_source_empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
                caption_source_trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
                image_decode_backend=getattr(self.config, "image_decode_backend", "pil"),
                image_decode_cache_size=getattr(self.config, "image_decode_cache_size", 0),
            )
            self._reg_dataloader = create_dataloader(
                reg_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=getattr(self.config, "dataloader_num_workers", 0),
                pin_memory=getattr(self.config, "pin_memory", True),
                prefetch_factor=getattr(self.config, "prefetch_factor", 2),
                persistent_workers=bool(getattr(self.config, "persistent_data_loader_workers", False)),
                drop_last=compile_drop_last,
            )
            # DreamBooth: override class image captions with class_prompt
            if class_prompt:
                _reg_dl = self._reg_dataloader
                class _RegCaptionWrapper:
                    def __init__(self, dl, prompt):
                        self._dl = dl
                        self._prompt = prompt
                    def __iter__(self):
                        for batch in self._dl:
                            batch["captions"] = [self._prompt] * len(batch["captions"])
                            yield batch
                    def __len__(self):
                        return len(self._dl)
                    def __getattr__(self, name):
                        return getattr(self._dl, name)
                self._reg_dataloader = _RegCaptionWrapper(_reg_dl, class_prompt)
            self._log(f"Regularization dataset: {len(reg_dataset)} samples")

        if anima_cached_training:
            policy = resolve_cached_dataloader_policy(
                self.config,
                route="anima",
                cached=True,
                cuda_available=str(self.device).startswith("cuda"),
            )
            for line in policy.log_lines():
                self._log(line)

            initial_batch_size = int(getattr(self.config, "batch_size", 1) or 1)
            if self._anima_staged_resolution_plan and self._anima_staged_resolution_active_index >= 0:
                initial_stage = self._anima_staged_resolution_plan[self._anima_staged_resolution_active_index]
                initial_batch_size = int(initial_stage.batch_size or initial_batch_size)
            dataloader = self._create_anima_cached_dataloader(
                dataset,
                batch_size=initial_batch_size,
                drop_last=compile_drop_last,
            )
            self._capture_cache_reader_decode_sidecar_profile(
                dataloader,
                route="anima_cached",
            )
            self._capture_cache_reader_training_gate_profile(
                dataloader,
                route="anima_cached",
            )
            self._mark_runtime_phase("anima_dataloader")
        elif newbie_cached_training:
            from .newbie_cached_dataset import create_newbie_cached_dataloader
            policy = resolve_cached_dataloader_policy(
                self.config,
                route="newbie",
                cached=True,
                cuda_available=str(self.device).startswith("cuda"),
            )
            for line in policy.log_lines():
                self._log(line)

            dataloader = create_newbie_cached_dataloader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=policy.num_workers,
                persistent_workers=policy.persistent_workers,
                pin_memory=policy.pin_memory,
                prefetch_factor=policy.prefetch_factor,
                drop_last=compile_drop_last,
                collate_mode=getattr(self.config, "cached_collate_mode", "auto"),
            )
            self._capture_cache_reader_decode_sidecar_profile(
                dataloader,
                route="newbie_cached",
            )
            self._capture_cache_reader_training_gate_profile(
                dataloader,
                route="newbie_cached",
            )
            self._mark_runtime_phase("newbie_dataloader")
        else:
            if caption_training_input is None:
                caption_training_input = self._create_caption_training_input(
                    data_dir=self.config.train_data_dir,
                    model_arch=model_arch,
                    batch_size=lora_initial_batch_size,
                    drop_last=compile_drop_last,
                )
                dataset = caption_training_input.dataset
                self._dataset = dataset
                self._lora_staged_resolution_sdxl_cache_first = caption_training_input.sdxl_cache_first
            dataloader = caption_training_input.dataloader
            self._mark_runtime_phase("sdxl_cache_dataloader" if caption_training_input.sdxl_cache_first else "dataloader")

        # DreamBooth: override instance image captions with instance_prompt
        training_type = str(getattr(self.config, "training_type", "") or "")
        if training_type == "dreambooth" and instance_prompt and not anima_cached_training and not newbie_cached_training:
            self._log(f"DreamBooth: instance captions → '{instance_prompt}'")
            _orig_collate = getattr(dataloader, "collate_fn", None)

            class _DreamBoothCaptionWrapper:
                """Replaces batch captions with instance_prompt for DreamBooth."""
                def __init__(self, dl, prompt):
                    self._dl = dl
                    self._prompt = prompt
                def __iter__(self):
                    for batch in self._dl:
                        batch["captions"] = [self._prompt] * len(batch["captions"])
                        yield batch
                def __len__(self):
                    return len(self._dl)
                def __getattr__(self, name):
                    return getattr(self._dl, name)

            dataloader = _DreamBoothCaptionWrapper(dataloader, instance_prompt)

        # ── Validation split ──
        validation_split = float(getattr(self.config, "validation_split", 0.0) or 0.0)
        validation_dataloader = None
        eval_data_dir = str(getattr(self.config, "eval_data_dir", "") or "").strip()
        if eval_data_dir:
            try:
                validation_dataloader = self._maybe_create_eval_dataloader(
                    model_arch=model_arch,
                    anima_cached_training=anima_cached_training,
                    newbie_cached_training=newbie_cached_training,
                    caption_bucket=caption_bucket,
                )
                if validation_split > 0.0:
                    self._log(
                        "eval_data_dir is set; validation_split will not split train_data_dir. "
                        "Using independent eval dataset instead."
                    )
            except Exception as e:
                self._log(f"Independent eval dataset setup failed, proceeding without validation: {e}")
                validation_dataloader = None
        elif validation_split > 0.0:
            try:
                if (
                    hasattr(dataset, "is_concept_geometry_enabled")
                    and callable(getattr(dataset, "is_concept_geometry_enabled"))
                    and dataset.is_concept_geometry_enabled()
                ):
                    self._log(
                        "[concept-geometry] validation_split rebuilds train/val loaders from Subset views; "
                        "the curriculum sampler falls back to regular shuffled subsets after the split."
                    )
                dataloader, validation_dataloader = split_dataloader(
                    dataloader,
                    fraction=validation_split,
                    seed=int(getattr(self.config, "seed", 42)),
                )
                val_n = len(validation_dataloader.dataset) if hasattr(validation_dataloader, "dataset") else "?"
                self._log(f"Validation split enabled: {validation_split:.0%} -> {val_n} val samples")
            except Exception as e:
                self._log(f"Validation split failed, proceeding without validation: {e}")
                validation_dataloader = None

        # 计算总步数
        if len(dataset) == 0:
            raise ValueError("No training samples found in train_data_dir.")

        grad_accum = max(int(self.config.gradient_accumulation_steps), 1)
        # TrainingLoop will step on the last (incomplete) accumulation group.
        steps_per_epoch = (len(dataloader) + grad_accum - 1) // grad_accum
        epoch_limited_steps = max(steps_per_epoch * int(self.config.max_train_epochs), 1)
        staged_step_plan = []
        if anima_cached_training and self._anima_staged_resolution_plan:
            staged_step_plan = self._anima_staged_resolution_plan
        elif self._lora_staged_resolution_enabled_runtime and self._lora_staged_resolution_plan:
            staged_step_plan = self._lora_staged_resolution_plan
        if staged_step_plan:
            staged_epoch_limited_steps = self._estimate_staged_epoch_limited_steps(
                stages=staged_step_plan,
                sample_count=len(dataset),
                grad_accum=grad_accum,
                drop_last=compile_drop_last,
                total_epochs=int(self.config.max_train_epochs),
            )
            if staged_epoch_limited_steps > 0:
                epoch_limited_steps = max(staged_epoch_limited_steps, 1)
        requested_max_steps = max(int(getattr(self.config, "max_train_steps", 0) or 0), 0)
        total_steps = min(epoch_limited_steps, requested_max_steps) if requested_max_steps > 0 else epoch_limited_steps
        total_steps = max(total_steps, 1)
        self._steps_per_epoch = steps_per_epoch
        self._total_steps = total_steps
        if hasattr(dataset, "set_concept_geometry_total_steps"):
            dataset.set_concept_geometry_total_steps(total_steps)
        self._write_run_manifest("prepared", epoch=0)

        self._log(f"Dataset: {len(dataset)} samples")
        self._log(f"Steps per epoch: {steps_per_epoch}")
        self._log(f"Total steps: {total_steps}")

        if bool(getattr(self.config, "hutchinson_auto_freeze", False)):
            try:
                from .b_tier_runtime import run_hutchinson_auto_freeze

                report = run_hutchinson_auto_freeze(
                    self.model,
                    output_dir=getattr(self.config, "output_dir", "."),
                    num_probes=int(getattr(self.config, "lulynx_hutchinson_probes", 30) or 30),
                    freeze_ratio=float(getattr(self.config, "hutchinson_freeze_ratio", 0.5) or 0.5),
                    device="cuda" if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu",
                )
                self._hutchinson_report = report
                self._log(
                    "Hutchinson auto-freeze applied "
                    f"({report.get('frozen_tensors', 0)} tensors, {report.get('frozen_params', 0):,} params)."
                )
            except Exception as e:
                self._hutchinson_report = {"enabled": True, "error": str(e)}
                self._log(f"Hutchinson auto-freeze failed; continuing without it: {e}")

        if getattr(self.config, "advanced_monitoring_enabled", False):
            try:
                from .vram_estimator import estimate_vram_breakdown
                _lora_params = self.lora_injector.parameters() if self.lora_injector else None
                _vram = estimate_vram_breakdown(self.model, self.config, lora_params=_lora_params)
                for line in _vram.summary_lines():
                    self._log(f"[VRAM]{line}")
                if self._tb_writer:
                    self._tb_writer.add_text("vram/breakdown", "\n".join(_vram.summary_lines()), 0)
            except Exception as e:
                self._log(f"[VRAM] estimation skipped: {e}")

        # 创建优化器和调度器
        optimizer = self._create_optimizer()
        self._mark_runtime_phase("optimizer_create")

        # Optionally replace AdamW with fused AdamW (single-pass step per param)
        if getattr(self.config, "fused_optimizer", False):
            from .fused_adamw import maybe_replace_optimizer
            optimizer = maybe_replace_optimizer(optimizer, self.config)
            if type(optimizer).__name__ == "FusedAdamW":
                self._set_optimizer_backend_profile(
                    "fused_optimizer_legacy",
                    "lulynx_fused",
                    optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
                    optimizer_class=type(optimizer).__name__,
                    notes=["Activated by legacy fused_optimizer=True."],
                )
                self._log("FusedAdamW optimizer activated (fused_optimizer=True)")

        # Optionally wrap optimizer with stochastic rounding
        if getattr(self.config, "stochastic_rounding", False):
            from .stochastic_rounding import apply_stochastic_rounding_to_optimizer
            optimizer = apply_stochastic_rounding_to_optimizer(optimizer, enabled=True)

        if bool(getattr(self.config, "optimizer_state_paging_enabled", False)):
            from .optimizer_state_paging import maybe_wrap_optimizer_state_paging

            optimizer = maybe_wrap_optimizer_state_paging(
                optimizer,
                enabled=True,
                min_tensor_mb=max(float(getattr(self.config, "optimizer_state_paging_min_tensor_mb", 1.0)), 0.0),
                pin_memory=bool(getattr(self.config, "optimizer_state_paging_pin_memory", False)),
            )
            self._log("Optimizer state paging enabled (experimental).")

        # Optionally wrap optimizer with SVD/GaLore-style gradient projection
        if getattr(self.config, "svd_grad_proj_enabled", False):
            from .svd_grad_projection import apply_svd_gradient_projection, SVDGradientProjectionWrapper
            optimizer = apply_svd_gradient_projection(
                optimizer,
                enabled=True,
                rank=int(getattr(self.config, "svd_grad_proj_rank", 128) or 128),
                update_interval=int(getattr(self.config, "svd_grad_proj_update_interval", 200) or 200),
                scale=float(getattr(self.config, "svd_grad_proj_scale", 1.0) or 1.0),
                warmup_steps=int(getattr(self.config, "svd_grad_proj_warmup_steps", 0) or 0),
            )
            self._mark_galore_runtime_outcome(
                applied=(
                    isinstance(optimizer, SVDGradientProjectionWrapper)
                    or type(optimizer).__name__ == "SVDGradientProjectionWrapper"
                    or hasattr(optimizer, "_projectors")
                ),
                fallback_reason="SVD/GaLore gradient projection wrapper did not activate.",
                note="Applied SVD/GaLore-style gradient projection wrapper during optimizer construction.",
            )

        # Optionally wrap optimizer with gradient guard (AGC / centralization)
        _gg_strategy = str(getattr(self.config, "gradient_guard_strategy", "none") or "none")
        if _gg_strategy != "none":
            from .gradient_guard import apply_gradient_guard
            optimizer = apply_gradient_guard(
                optimizer,
                strategy=_gg_strategy,
                agc_clip_factor=float(getattr(self.config, "gradient_guard_agc_clip_factor", 0.01) or 0.01),
                agc_eps=float(getattr(self.config, "gradient_guard_agc_eps", 1e-3) or 1e-3),
            )

        # Create adaptive loss weighter if enabled
        _adaptive_loss_weighter = None
        if getattr(self.config, "adaptive_loss_weighting_enabled", False):
            from .adaptive_loss_weighting import AdaptiveLossWeighter
            _adaptive_loss_weighter = AdaptiveLossWeighter(
                init_gamma=float(getattr(self.config, "adaptive_loss_weighting_init_gamma", 5.0) or 5.0),
            ).to(self.device)
            _alw_lr = float(getattr(self.config, "adaptive_loss_weighting_lr", 1e-3) or 1e-3)
            optimizer.add_param_group({
                "params": list(_adaptive_loss_weighter.parameters()),
                "lr": _alw_lr,
                "weight_decay": 0.0,
            })
            self._log(f"Adaptive loss weighting enabled (lr={_alw_lr})")

        # Create EDM2-style flow uncertainty weighter if enabled (anima flow route only)
        _flow_uncertainty_weighter = None
        if (
            getattr(self.config, "flow_uncertainty_weighting_enabled", False)
            and self._model_arch_value() == "anima"
        ):
            from .flow_uncertainty_weighting import FlowUncertaintyWeighter
            _flow_uncertainty_weighter = FlowUncertaintyWeighter(
                num_channels=int(getattr(self.config, "flow_uncertainty_weighting_channels", 128) or 128),
            ).to(self.device)
            _fuw_lr = float(getattr(self.config, "flow_uncertainty_weighting_lr", 1e-2) or 1e-2)
            optimizer.add_param_group({
                "params": list(_flow_uncertainty_weighter.parameters()),
                "lr": _fuw_lr,
                "weight_decay": 0.0,
            })
            self._log(f"Flow uncertainty weighting (EDM2) enabled (lr={_fuw_lr})")

        # Create FasterDiT SNR weighter if enabled
        _faster_dit_snr_weighter = None
        if getattr(self.config, "faster_dit_snr_enabled", False):
            from .faster_dit_snr import FasterDiTSNRConfig, FasterDiTSNRWeighter
            _snr_mode = str(getattr(self.config, "faster_dit_snr_mode", "sqrt") or "sqrt")
            _snr_gamma = float(getattr(self.config, "faster_dit_snr_gamma", 5.0) or 5.0)
            _snr_sampling = str(getattr(self.config, "faster_dit_snr_sampling", "uniform") or "uniform")
            _snr_low_weight = float(getattr(self.config, "faster_dit_snr_low_snr_weight", 1.5) or 1.5)
            _faster_dit_config = FasterDiTSNRConfig(
                mode=_snr_mode,
                gamma=_snr_gamma,
                timestep_sampling=_snr_sampling,
                low_snr_weight=_snr_low_weight,
            )
            _faster_dit_snr_weighter = FasterDiTSNRWeighter(_faster_dit_config).to(self.device)
            self._log(
                f"FasterDiT SNR enabled (mode={_snr_mode}, gamma={_snr_gamma}, "
                f"sampling={_snr_sampling}, low_snr_weight={_snr_low_weight})"
            )

        if getattr(self.config, "lr_finder_enabled", False):
            try:
                from .lr_finder import LRFinder
                _lr_finder_step_fn = self._make_lr_finder_step_fn(
                    unet_for_training if 'unet_for_training' in dir() else self.model.unet,
                    optimizer, dataloader,
                )
                _lrf = LRFinder(
                    model=self.model.unet,
                    optimizer=optimizer,
                    step_fn=_lr_finder_step_fn,
                    start_lr=float(getattr(self.config, "lr_finder_start_lr", 1e-7) or 1e-7),
                    end_lr=float(getattr(self.config, "lr_finder_end_lr", 1e-1) or 1e-1),
                    num_steps=int(getattr(self.config, "lr_finder_num_steps", 100) or 100),
                )
                _lrf_result = _lrf.run()
                for line in _lrf_result.summary_lines():
                    self._log(f"[LR Finder]{line}")
                if _lrf_result.suggested_lr > 0:
                    for pg in optimizer.param_groups:
                        pg["lr"] = _lrf_result.suggested_lr
                    self._log(f"[LR Finder] Applied suggested LR: {_lrf_result.suggested_lr:.2e}")
                if self._tb_writer:
                    self._tb_writer.add_text("lr_finder/result", "\n".join(_lrf_result.summary_lines()), 0)
            except Exception as e:
                self._log(f"[LR Finder] skipped: {e}")

        self._attach_optimizer_profiles_to_training_loop()

        scheduler = self._create_scheduler(optimizer, total_steps)
        self._mark_runtime_phase("scheduler_create")

        # DDP: wrap model/optimizer/dataloader if multi_gpu is enabled
        unet_for_training = self.model.unet
        unet_for_training, optimizer, dataloader = self._wrap_ddp(
            unet_for_training, optimizer, dataloader, dataset
        )
        self._dataloader = dataloader
        self._mark_runtime_phase("ddp_wrap")

        sdxl_component_residency = self._should_use_sdxl_component_cpu_residency()
        self._refresh_diffusers_cache_runtime_profile(
            model_arch=model_arch,
            cache_first=bool(getattr(caption_training_input, "sdxl_cache_first", False)),
            cache_root=str(getattr(caption_training_input, "cache_root", "") or ""),
            component_cpu_residency=sdxl_component_residency,
        )

        # 创建训练循环
        self.training_loop = TrainingLoop(
            unet=unet_for_training,
            text_encoder_1=self.model.text_encoder_1,
            text_encoder_2=self.model.text_encoder_2,
            vae=self.model.vae,
            tokenizer_1=self.model.tokenizer_1,
            tokenizer_2=self.model.tokenizer_2,
            noise_scheduler=self.model.noise_scheduler,
            lora_injector=self.lora_injector,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            device=self.device,
            dtype=self.dtype,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_accumulation_mode=getattr(self.config, "gradient_accumulation_mode", "fast"),
            thermal_throttler=self._create_thermal_throttler(),
            max_grad_norm=self.config.max_grad_norm,
            noise_offset=self.config.noise_offset,
            snr_gamma=self.config.min_snr_gamma if self.config.min_snr_gamma > 0 else None,
            loss_type=getattr(self.config, "loss_type", "l2"),
            loss_precision=getattr(self.config, "loss_precision", "fp32_loss"),
            huber_c=getattr(self.config, "huber_c", 0.1),
            huber_schedule=getattr(self.config, "huber_schedule", "constant"),
            huber_scale=getattr(self.config, "huber_scale", 1.0),
            te_manager=self.te_manager,
            model_arch=model_arch,
            train_text_encoder=bool(getattr(self.config, "train_text_encoder", False)),
            text_encoder_cpu_residency=sdxl_component_residency and not bool(getattr(self.config, "train_text_encoder", False)),
            vae_cpu_residency=sdxl_component_residency,
            multires_noise_iterations=getattr(self.config, "multires_noise_iterations", 0),
            multires_noise_discount=getattr(self.config, "multires_noise_discount", 0.3),
            adaptive_noise_scale=getattr(self.config, "adaptive_noise_scale", 0.0),
            ip_noise_gamma=getattr(self.config, "ip_noise_gamma", 0.0),
            noise_offset_random_strength=getattr(self.config, "noise_offset_random_strength", False),
            ip_noise_gamma_random_strength=getattr(self.config, "ip_noise_gamma_random_strength", False),
            debiased_estimation=getattr(self.config, "debiased_estimation_loss", False),
            zero_terminal_snr=getattr(self.config, "zero_terminal_snr", False),
            v_parameterization=getattr(self.config, "v_parameterization", False),
            scale_v_pred_loss_like_noise_pred=getattr(self.config, "scale_v_pred_loss_like_noise_pred", False),
            masked_loss=getattr(self.config, "masked_loss", False),
            alpha_mask=getattr(self.config, "alpha_mask", False),
            strict_masked_loss=getattr(self.config, "strict_masked_loss", False),
            blocks_to_swap=getattr(self.config, "blocks_to_swap", 0),
            swap_granularity=getattr(self.config, "swap_granularity", "off"),
            swap_ratio=getattr(self.config, "swap_ratio", 0.0),
            swap_count=getattr(self.config, "swap_count", 0),
            block_merge_size=getattr(self.config, "block_merge_size", 2),
            block_swap_strategy=getattr(self.config, "block_swap_strategy", "auto"),
            module_offload_enabled=getattr(self.config, "module_offload_enabled", False),
            module_offload_ratio=getattr(self.config, "module_offload_ratio", 0),
            module_offload_backbone_ratio=getattr(self.config, "module_offload_backbone_ratio", None),
            module_offload_text_encoder_ratio=getattr(self.config, "module_offload_text_encoder_ratio", None),
            module_offload_profile=getattr(self.config, "module_offload_profile", "custom"),
            module_offload_profile_enabled=getattr(self.config, "module_offload_profile_enabled", False),
            module_offload_min_param_mb=getattr(self.config, "module_offload_min_param_mb", 0.0),
            module_offload_include_patterns=getattr(self.config, "module_offload_include_patterns", ""),
            module_offload_exclude_patterns=getattr(self.config, "module_offload_exclude_patterns", ""),
            module_offload_verify_state=getattr(self.config, "module_offload_verify_state", True),
            module_offload_prefetch_enabled=getattr(self.config, "module_offload_prefetch_enabled", False),
            module_offload_prefetch_mode=getattr(self.config, "module_offload_prefetch_mode", "experimental"),
            module_offload_enhanced=getattr(self.config, "module_offload_enhanced", False),
            gradient_checkpointing=getattr(self.config, "gradient_checkpointing", False),
            vram_swap_to_ram=getattr(self.config, "vram_swap_to_ram", False),
            torch_compile=getattr(self.config, "torch_compile", False),
            cpu_offload_checkpointing=getattr(self.config, "cpu_offload_checkpointing", False),
            cpu_offload_checkpointing_mode=getattr(self.config, "cpu_offload_checkpointing_mode", "standard"),
            cpu_offload_checkpointing_pool_gb=float(getattr(self.config, "cpu_offload_checkpointing_pool_gb", 1.0) or 1.0),
            adapter_cpu_residency=self._adapter_cpu_residency,
            attention_profile=getattr(self, "_attention_profile", None),
            data_transfer_non_blocking=bool(getattr(self.config, "data_transfer_non_blocking", True)),
            data_transfer_profile_enabled=bool(getattr(self.config, "data_transfer_profile_enabled", False)),
            data_transfer_profile_mode=str(getattr(self.config, "data_transfer_profile_mode", "event") or "event"),
            data_transfer_profile_window=int(getattr(self.config, "data_transfer_profile_window", 50) or 50),
            step_phase_profile_enabled=bool(getattr(self.config, "step_phase_profile_enabled", False)),
            turbocore_update_shadow_mode=str(getattr(self.config, "turbocore_update_shadow_mode", "off") or "off"),
            turbocore_update_shadow_max_params=int(getattr(self.config, "turbocore_update_shadow_max_params", 0) or 0),
            turbocore_update_shadow_compare_interval=int(getattr(self.config, "turbocore_update_shadow_compare_interval", 1) or 1),
            turbocore_update_shadow_direct_grad=bool(getattr(self.config, "turbocore_update_shadow_direct_grad", False)),
            turbocore_update_shadow_prefer_triton=bool(getattr(self.config, "turbocore_update_shadow_prefer_triton", False)),
            turbocore_update_shadow_compare_sample_params=int(getattr(self.config, "turbocore_update_shadow_compare_sample_params", 0) or 0),
            turbocore_update_shadow_stop_after_consecutive_passes=int(getattr(self.config, "turbocore_update_shadow_stop_after_consecutive_passes", 0) or 0),
            turbocore_update_shadow_checkpoint_contract=bool(getattr(self.config, "turbocore_update_shadow_checkpoint_contract", False)),
            turbocore_update_shadow_copyback_probe=bool(getattr(self.config, "turbocore_update_shadow_copyback_probe", False)),
            turbocore_update_shadow_copyback_dispatch_experimental=bool(
                getattr(self.config, "turbocore_update_shadow_copyback_dispatch_experimental", False)
            ),
            turbocore_update_shadow_native_binding_probe=bool(getattr(self.config, "turbocore_update_shadow_native_binding_probe", False)),
            turbocore_update_shadow_owner_native_launch_probe=bool(
                getattr(self.config, "turbocore_update_shadow_owner_native_launch_probe", False)
            ),
            turbocore_update_shadow_owner_native_launch_max_numel=int(
                getattr(self.config, "turbocore_update_shadow_owner_native_launch_max_numel", 1048576) or 1048576
            ),
            turbocore_update_shadow_owner_native_event_chain_probe=bool(
                getattr(self.config, "turbocore_update_shadow_owner_native_event_chain_probe", False)
            ),
            turbocore_update_shadow_save_owner_state=bool(getattr(self.config, "turbocore_update_shadow_save_owner_state", False)),
            turbocore_native_update_mode=str(getattr(self.config, "turbocore_native_update_mode", "off") or "off"),
            turbocore_native_update_required_shadow_passes=int(getattr(self.config, "turbocore_native_update_required_shadow_passes", 3) or 3),
            turbocore_native_update_max_abs_diff=float(getattr(self.config, "turbocore_native_update_max_abs_diff", 5e-5) or 0.0),
            turbocore_native_update_max_mean_abs_diff=float(getattr(self.config, "turbocore_native_update_max_mean_abs_diff", 1e-6) or 0.0),
            turbocore_native_update_allow_missing_kernel=bool(getattr(self.config, "turbocore_native_update_allow_missing_kernel", False)),
            turbocore_native_update_strict=bool(getattr(self.config, "turbocore_native_update_strict", False)),
            turbocore_native_update_dispatch_enabled=bool(
                getattr(self.config, "turbocore_native_update_dispatch_enabled", False)
            ),
            turbocore_native_update_training_path_enabled=bool(
                getattr(self.config, "turbocore_native_update_training_path_enabled", False)
            ),
            turbocore_native_update_require_native_cuda=bool(
                getattr(self.config, "turbocore_native_update_require_native_cuda", False)
            ),
            turbocore_native_update_diagnostic_executor_replay=bool(
                getattr(self.config, "turbocore_native_update_diagnostic_executor_replay", False)
            ),
            turbocore_native_update_defer_state_sync=bool(
                getattr(self.config, "turbocore_native_update_defer_state_sync", False)
            ),
            turbocore_native_update_runtime_synchronization_policy=str(
                getattr(
                    self.config,
                    "turbocore_native_update_runtime_synchronization_policy",
                    "context_synchronize",
                )
                or "context_synchronize"
            ),
            safe_fallback=(model_arch == "newbie" and bool(getattr(self.config, "newbie_safe_fallback", False))),
            anima_timestep_sampling=str(getattr(self.config, "timestep_sampling", "sigma") or "sigma"),
            anima_sigmoid_scale=float(getattr(self.config, "anima_sigmoid_scale", getattr(self.config, "sigmoid_scale", 1.0)) or 1.0),
            anima_discrete_flow_shift=float(getattr(self.config, "discrete_flow_shift", 1.0) or 1.0),
            anima_weighting_scheme=str(getattr(self.config, "anima_weighting_scheme", getattr(self.config, "weighting_scheme", "none")) or "none"),
            anima_model_prediction_type=str(getattr(self.config, "anima_model_prediction_type", "velocity") or "velocity"),
            anima_mode_scale=float(getattr(self.config, "anima_mode_scale", 1.0) or 1.0),
            anima_faithful_forward=bool(getattr(self.config, "anima_faithful_forward", False)),
            # JLT EMA feature self-distillation alignment (default-off reserve)
            anima_ema_feat_align_enabled=bool(getattr(self.config, "anima_ema_feat_align_enabled", False)),
            anima_ema_feat_align_weight=float(getattr(self.config, "anima_ema_feat_align_weight", 0.0) or 0.0),
            anima_ema_feat_align_teacher_layers=str(getattr(self.config, "anima_ema_feat_align_teacher_layers", "") or ""),
            anima_ema_feat_align_student_layers=str(getattr(self.config, "anima_ema_feat_align_student_layers", "") or ""),
            anima_ema_feat_align_decay=float(getattr(self.config, "anima_ema_feat_align_decay", 0.9999) or 0.9999),
            # SDXL Flow Matching
            flow_model=str(getattr(self.config, "flow_model", "") or ""),
            sdxl_timestep_sampling=str(getattr(self.config, "timestep_sampling", "uniform") or "uniform"),
            sdxl_sigmoid_scale=float(getattr(self.config, "anima_sigmoid_scale", 1.0) or 1.0),
            sdxl_flow_shift=float(getattr(self.config, "sdxl_flow_shift", 1.0) or 1.0),
            sdxl_flow_weighting_scheme=str(getattr(self.config, "sdxl_flow_weighting_scheme", "none") or "none"),
            sdxl_model_prediction_type=str(getattr(self.config, "sdxl_model_prediction_type", "epsilon") or "epsilon"),
            flow_logit_mean=float(getattr(self.config, "flow_logit_mean", 0.0) or 0.0),
            flow_logit_std=float(getattr(self.config, "flow_logit_std", 1.0) or 1.0),
            flow_uniform_shift=bool(getattr(self.config, "flow_uniform_shift", False)),
            flow_uniform_base_pixels=int(getattr(self.config, "flow_uniform_base_pixels", 256) or 256),
            flow_uniform_static_ratio=float(getattr(self.config, "flow_uniform_static_ratio", 0.0) or 0.0),
            cfm_lambda=float(getattr(self.config, "cfm_lambda", 1.0) or 1.0),
            flow_use_ot=bool(getattr(self.config, "flow_use_ot", False)),
            immiscible_diffusion_enabled=bool(getattr(self.config, "immiscible_diffusion_enabled", False)),
            immiscible_metric=str(getattr(self.config, "immiscible_metric", "l2") or "l2"),
            te_dropout=float(getattr(self.config, "te_dropout", 0.0)),
            clip_l_dropout_rate=float(getattr(self.config, "clip_l_dropout_rate", 0.0) or 0.0),
            clip_g_dropout_rate=float(getattr(self.config, "clip_g_dropout_rate", 0.0) or 0.0),
            t5_dropout_rate=float(getattr(self.config, "t5_dropout_rate", 0.0) or 0.0),
            wavelet_loss_enabled=bool(getattr(self.config, "wavelet_loss_enabled", False)),
            wavelet_loss_levels=int(getattr(self.config, "wavelet_loss_levels", 2)),
            wavelet_loss_high_freq_weight=float(getattr(self.config, "wavelet_loss_high_freq_weight", 2.0)),
            wavelet_loss_approx_weight=float(getattr(self.config, "wavelet_loss_approx_weight", 0.0)),
            wavelet_loss_base_loss=getattr(self.config, "wavelet_loss_base_loss", "l2"),
            # Qwen3 secondary encoder (Anima)
            qwen3_encoder=getattr(self.model, "anima_qwen3_encoder", None),
            qwen3_tokenizer=getattr(self.model, "anima_qwen3_tokenizer", None),
            # Fixed token padding for torch.compile static shape contract
            max_token_length=self.config.max_token_length,
            enable_fixed_token_padding=self._should_enable_fixed_token_padding(),
            easy_control=self._easy_control,
            ip_adapter=self._ip_adapter,
            easycontrol_v2_adapter=self._easycontrol_v2_adapter,
            repa_enabled=bool(getattr(self.config, "repa_enabled", False)),
            repa_target_modules=str(getattr(self.config, "repa_target_modules", "") or ""),
            repa_loss_type=str(getattr(self.config, "repa_loss_type", "cosine") or "cosine"),
            repa_loss_weight=float(getattr(self.config, "repa_loss_weight", 0.0) or 0.0),
            repa_projection_dim=int(getattr(self.config, "repa_projection_dim", 0) or 0),
            repa_stop_grad_target=bool(getattr(self.config, "repa_stop_grad_target", True)),
            repa_projector=self._repa_projector,
            lulynx_geometric_lock=bool(getattr(self.config, "lulynx_geometric_lock", False)),
            lulynx_manifold_weight=float(getattr(self.config, "lulynx_manifold_weight", getattr(self.config, "lulynx_ln_lambda", 0.01)) or 0.0),
            lulynx_proj_dim=int(getattr(self.config, "lulynx_proj_dim", 128) or 128),
            lulynx_manifold_sparse_freq=int(getattr(self.config, "lulynx_manifold_sparse_freq", 1) or 1),
            lulynx_anchor_layers=str(getattr(self.config, "lulynx_anchor_layers", "") or ""),
            lulynx_ghost_replay=bool(getattr(self.config, "lulynx_ghost_replay", False)),
            lulynx_ghost_path=str(getattr(self.config, "lulynx_ghost_path", "") or ""),
            lulynx_ghost_interval=int(getattr(self.config, "lulynx_ghost_interval", 100) or 100),
            lulynx_ghost_weight=float(getattr(self.config, "lulynx_ghost_weight", 0.05) or 0.0),
            softrepa_enabled=bool(getattr(self.config, "softrepa_enabled", False)),
            softrepa_schedule=str(getattr(self.config, "softrepa_schedule", "linear") or "linear"),
            softrepa_min_weight=float(getattr(self.config, "softrepa_min_weight", 0.0) or 0.0),
            softrepa_max_weight=float(getattr(self.config, "softrepa_max_weight", 1.0) or 1.0),
            softrepa_sigma_min=float(getattr(self.config, "softrepa_sigma_min", 0.0) or 0.0),
            softrepa_sigma_max=float(getattr(self.config, "softrepa_sigma_max", 1.0) or 1.0),
            sra2_haste_enabled=bool(getattr(self.config, "sra2_haste_enabled", False)),
            sra2_haste_capture_layers=str(getattr(self.config, "sra2_haste_capture_layers", "") or ""),
            sra2_haste_policy={
                "loss_type": str(getattr(self.config, "sra2_haste_loss_type", "cosine") or "cosine"),
                "normalize_targets": bool(getattr(self.config, "sra2_haste_normalize_targets", True)),
                "stop_grad_target": bool(getattr(self.config, "sra2_haste_stop_grad_target", True)),
                "base_weight": float(getattr(self.config, "sra2_haste_base_weight", 1.0) or 1.0),
                "start_step": int(getattr(self.config, "sra2_haste_start_step", 0) or 0),
                "stop_step": int(getattr(self.config, "sra2_haste_stop_step", -1)),
                "decay_start_step": int(getattr(self.config, "sra2_haste_decay_start_step", -1)),
                "decay_end_step": int(getattr(self.config, "sra2_haste_decay_end_step", -1)),
                "min_weight": float(getattr(self.config, "sra2_haste_min_weight", 0.0) or 0.0),
                "plateau_patience": int(getattr(self.config, "sra2_haste_plateau_patience", 0) or 0),
                "min_relative_improvement": float(getattr(self.config, "sra2_haste_min_relative_improvement", 0.0) or 0.0),
            },
            dit_compute_reducer_strategy=str(getattr(self.config, "dit_compute_reducer_strategy", "none") or "none"),
            dit_compute_reducer_keep_ratio=float(getattr(self.config, "dit_compute_reducer_keep_ratio", 1.0) or 1.0),
            dit_compute_reducer_min_keep_tokens=int(getattr(self.config, "dit_compute_reducer_min_keep_tokens", 1) or 1),
            dit_compute_reducer_compression_ratio=float(getattr(self.config, "dit_compute_reducer_compression_ratio", 1.0) or 1.0),
            dit_compute_reducer_min_tokens=int(getattr(self.config, "dit_compute_reducer_min_tokens", 1) or 1),
            dit_compute_reducer_skip_ratio=float(getattr(self.config, "dit_compute_reducer_skip_ratio", 0.0) or 0.0),
            dit_compute_reducer_skip_every=int(getattr(self.config, "dit_compute_reducer_skip_every", 0) or 0),
            dit_compute_reducer_warmup_steps=int(getattr(self.config, "dit_compute_reducer_warmup_steps", 0) or 0),
            dit_compute_reducer_min_block=int(getattr(self.config, "dit_compute_reducer_min_block", 0) or 0),
            dit_compute_reducer_score_mode=str(getattr(self.config, "dit_compute_reducer_score_mode", "l2") or "l2"),
            multi_gpu=bool(getattr(self.config, "multi_gpu", False)),
            num_processes=int(getattr(self.config, "num_processes", 1) or 1),
            num_machines=int(getattr(self.config, "num_machines", 1) or 1),
            training_type=str(getattr(self.config, "training_type", "") or ""),
            deepspeed=bool(getattr(self.config, "deepspeed", False)),
            # Port 5 进阶显存优化
            gradient_release_enabled=bool(getattr(self.config, "gradient_release_enabled", False)),
            gradient_release_mode=str(getattr(self.config, "gradient_release_mode", "post_step") or "post_step"),
            activation_compression_enabled=bool(getattr(self.config, "activation_compression_enabled", False)),
            activation_compression_dtype=str(getattr(self.config, "activation_compression_dtype", "fp16") or "fp16"),
            activation_compression_min_tensor_mb=max(float(getattr(self.config, "activation_compression_min_tensor_mb", 1.0)), 0.0),
            activation_cpu_offload_enabled=bool(getattr(self.config, "activation_cpu_offload_enabled", False)),
            activation_cpu_offload_min_tensor_mb=max(float(getattr(self.config, "activation_cpu_offload_min_tensor_mb", 1.0)), 0.0),
            activation_cpu_offload_pool_gb=max(float(getattr(self.config, "activation_cpu_offload_pool_gb", 1.0)), 0.0),
            resolution_aware_batch_enabled=bool(getattr(self.config, "resolution_aware_batch_enabled", False)),
            resolution_aware_batch_base_resolution=int(getattr(self.config, "resolution_aware_batch_base_resolution", 1024) or 1024),
            resolution_aware_batch_max_factor=float(getattr(self.config, "resolution_aware_batch_max_factor", 4.0) or 4.0),
            resolution_aware_batch_min_factor=float(getattr(self.config, "resolution_aware_batch_min_factor", 0.25) or 0.25),
            pipeline_parallel_enabled=bool(getattr(self.config, "pipeline_parallel_enabled", False)),
            pipeline_parallel_chunks=int(getattr(self.config, "pipeline_parallel_chunks", 2) or 2),
            pipeline_parallel_split_points=str(getattr(self.config, "pipeline_parallel_split_points", "") or ""),
            # Feature batch: 9 Warehouse additions
            ddpm_timestep_sampling=str(getattr(self.config, "ddpm_timestep_sampling", "") or ""),
            stochastic_grad_accumulation=bool(getattr(self.config, "stochastic_grad_accumulation", False)),
            spectral_noise_blend=float(getattr(self.config, "spectral_noise_blend", 0.0) or 0.0),
            spectral_noise_sigma=float(getattr(self.config, "spectral_noise_sigma", 4.0) or 4.0),
            huber_auto_percentile=float(getattr(self.config, "huber_auto_percentile", 0.9) or 0.9),
            adaptive_loss_weighter=_adaptive_loss_weighter,
            flow_uncertainty_weighter=_flow_uncertainty_weighter,
            faster_dit_snr_weighter=_faster_dit_snr_weighter,
            sageattn_drift_check_interval=int(getattr(self.config, "sageattn_drift_check_interval", 0) or 0),
            sageattn_drift_threshold=float(getattr(self.config, "sageattn_drift_threshold", 0.01) or 0.01),
            sageattn_drift_fallback=str(getattr(self.config, "sageattn_drift_fallback", "warn") or "warn"),
            stepped_loss_enabled=bool(getattr(self.config, "stepped_loss_enabled", False)),
            stepped_loss_schedule=str(getattr(self.config, "stepped_loss_schedule", "") or ""),
            pattern_loss_enabled=bool(getattr(self.config, "pattern_loss_enabled", False)),
            pattern_loss_levels=int(getattr(self.config, "pattern_loss_levels", 1) or 1),
            pattern_loss_ll_type=str(getattr(self.config, "pattern_loss_ll_type", "l2") or "l2"),
            pattern_loss_ll_weight=float(getattr(self.config, "pattern_loss_ll_weight", 1.0) or 1.0),
            pattern_loss_high_type=str(getattr(self.config, "pattern_loss_high_type", "huber") or "huber"),
            pattern_loss_high_weight=float(getattr(self.config, "pattern_loss_high_weight", 2.0) or 2.0),
            pattern_loss_high_huber_c=float(getattr(self.config, "pattern_loss_high_huber_c", 0.1) or 0.1),
            perlin_noise_offset_enabled=bool(getattr(self.config, "perlin_noise_offset_enabled", False)),
            perlin_noise_offset_strength=float(getattr(self.config, "perlin_noise_offset_strength", 0.1) or 0.1),
            perlin_noise_offset_scale=float(getattr(self.config, "perlin_noise_offset_scale", 4.0) or 4.0),
            optimal_noise_enabled=bool(getattr(self.config, "optimal_noise_enabled", False)),
            optimal_noise_candidates=int(getattr(self.config, "optimal_noise_candidates", 4) or 4),
            dop=getattr(self, "_dop_instance", None),
            concept_direction=None,
            # 高级训练监控
            advanced_monitoring_enabled=bool(getattr(self.config, "advanced_monitoring_enabled", False)),
            peak_vram_diagnostics_interval=int(getattr(self.config, "peak_vram_diagnostics_interval", 25) or 25),
            cuda_cache_release_strategy=str(getattr(self.config, "cuda_cache_release_strategy", "off") or "off"),
            cuda_cache_release_interval=int(getattr(self.config, "cuda_cache_release_interval", 1) or 1),
            audit_mode_override=str(getattr(self.config, "audit_mode_override", "") or ""),
            attn_entropy_interval=int(getattr(self.config, "attn_entropy_interval", 100) or 100),
            act_drift_interval=int(getattr(self.config, "act_drift_interval", 100) or 100),
            act_drift_anchor_layers=str(getattr(self.config, "act_drift_anchor_layers", "") or ""),
            # 深度诊断
            deep_diagnostics_enabled=bool(getattr(self.config, "deep_diagnostics_enabled", False)),
            hessian_trace_interval=int(getattr(self.config, "hessian_trace_interval", 200) or 200),
            grad_cosine_enabled=bool(getattr(self.config, "grad_cosine_enabled", False)),
            # 遗忘探针
            forgetting_probe_interval=int(getattr(self.config, "forgetting_probe_interval", 50) or 50),
            # 流形追踪
            manifold_snapshot_interval=int(getattr(self.config, "manifold_snapshot_interval", 20) or 20),
            precision_swap_profile=getattr(self, "_precision_swap_profile", None),
            te_vae_offload_strategy=str(getattr(self.config, "te_vae_offload_strategy", "phase") or "phase"),
            layer_monitor_enabled=bool(getattr(self.config, "layer_monitor_enabled", True)),
            layer_monitor_interval=int(getattr(self.config, "layer_monitor_interval", 3) or 3),
            layer_monitor_max_layers=int(getattr(self.config, "layer_monitor_max_layers", 10) or 10),
            layer_monitor_sparsity_epsilon=float(getattr(self.config, "layer_monitor_sparsity_epsilon", 1e-8) or 1e-8),
            layer_monitor_mode=str(getattr(self.config, "layer_monitor_mode", "sampled") or "sampled"),
            layer_monitor_sample_size=int(getattr(self.config, "layer_monitor_sample_size", 4096) or 4096),
            vram_smart_sensing_enabled=bool(getattr(self.config, "vram_smart_sensing_enabled", True)),
            vram_smart_sensing_baseline_steps=int(getattr(self.config, "vram_smart_sensing_baseline_steps", 50) or 50),
            vram_smart_sensing_slowdown_ratio=float(getattr(self.config, "vram_smart_sensing_slowdown_ratio", 1.5) or 1.5),
            vram_smart_sensing_window_steps=int(getattr(self.config, "vram_smart_sensing_window_steps", 5) or 5),
        )
        self._mark_runtime_phase("training_loop_create")
        self.training_loop.newbie_backward_op_profile_enabled = bool(
            getattr(self.config, "newbie_backward_op_profile_enabled", False)
        )
        self.training_loop.newbie_backward_op_profile_top_k = max(
            int(getattr(self.config, "newbie_backward_op_profile_top_k", 12) or 12),
            1,
        )
        self.training_loop.newbie_backward_op_profile_max_samples = max(
            int(getattr(self.config, "newbie_backward_op_profile_max_samples", 1) or 1),
            1,
        )
        self.training_loop.newbie_backward_op_profile_record_shapes = bool(
            getattr(self.config, "newbie_backward_op_profile_record_shapes", False)
        )
        self.training_loop.newbie_module_timing_profile_enabled = bool(
            getattr(self.config, "newbie_module_timing_profile_enabled", False)
        )
        self.training_loop.newbie_module_timing_profile_top_k = max(
            int(getattr(self.config, "newbie_module_timing_profile_top_k", 12) or 12),
            1,
        )
        self.training_loop.newbie_module_timing_profile_max_samples = max(
            int(getattr(self.config, "newbie_module_timing_profile_max_samples", 1) or 1),
            1,
        )
        # Pass prior preservation config to training loop
        if self._reg_dataloader:
            self.training_loop.prior_loss_weight = getattr(self.config, "prior_loss_weight", 1.0)
            self.training_loop.reg_dataloader = self._reg_dataloader
        # Attach validation dataloader if split was performed
        if validation_dataloader is not None:
            self.training_loop.validation_dataloader = validation_dataloader
            self.training_loop.eval_every_n_steps = max(int(getattr(self.config, "eval_every_n_steps", 0) or 0), 0)
            self.training_loop.max_validation_steps = max(int(getattr(self.config, "max_validation_steps", 0) or 0), 0)
        precision_swap_profile = getattr(self, "_precision_swap_profile", None)
        if precision_swap_profile and hasattr(self.training_loop, "memory_optimization_state"):
            self.training_loop.memory_optimization_state["precision_swap_profile"] = precision_swap_profile
        self._attach_lora_activation_recompute_profile_to_training_loop()
        self._attach_adapter_runtime_profile_to_training_loop()
        self._attach_diffusers_cache_runtime_profile_to_training_loop()
        self._attach_weight_compression_profile_to_training_loop()
        self._attach_attention_runtime_profile_to_training_loop()
        self._attach_compile_runtime_profile_to_training_loop()
        self._attach_optimizer_profiles_to_training_loop()
        self._attach_data_backend_profile_to_training_loop()
        self._attach_memory_runtime_profiles_to_training_loop()
        # 遗忘探针初始化
        if getattr(self.config, "forgetting_probe_enabled", False) and validation_dataloader is not None:
            try:
                from .forgetting_probe import ForgettingProbe
                _fp = ForgettingProbe(
                    num_anchors=int(getattr(self.config, "forgetting_probe_num_anchors", 4) or 4),
                )
                _fp.capture_anchors(validation_dataloader)
                if _fp.has_anchors:
                    self.training_loop._forgetting_probe = _fp
                    self._log(f"[ForgettingProbe] captured {len(_fp._anchor_batches)} anchor batches")
            except Exception as e:
                self._log(f"[ForgettingProbe] init skipped: {e}")
        # 流形追踪初始化
        if getattr(self.config, "manifold_enabled", False):
            try:
                from .manifold_tracker import ManifoldTracker
                self.training_loop._manifold_tracker = ManifoldTracker()
                self._log("[ManifoldTracker] enabled")
            except Exception as e:
                self._log(f"[ManifoldTracker] init skipped: {e}")
        # Expose for callbacks (progress/scheduler rebuild/auditor)
        self.training_loop.steps_per_epoch = steps_per_epoch
        self.training_loop.total_steps = total_steps
        pcgrad_reduction = str(getattr(self.config, "pcgrad_reduction", "mean") or "mean").strip().lower()
        if pcgrad_reduction not in {"mean", "sum"}:
            pcgrad_reduction = "mean"
        try:
            pcgrad_conflict_threshold = float(getattr(self.config, "pcgrad_conflict_threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            pcgrad_conflict_threshold = 0.0
        self.training_loop.pcgrad_enabled = bool(getattr(self.config, "pcgrad_enabled", False))
        self.training_loop.pcgrad_conflict_threshold = pcgrad_conflict_threshold
        self.training_loop.pcgrad_reduction = pcgrad_reduction
        self.training_loop._pcgrad_param_names = self._optimizer_param_names()
        # Propagate torch.compile status so TrainingLoop can guard against
        # incompatible features (e.g. BlockSwap + torch.compile).
        if self.runtime_optimization_plan is not None and self.runtime_optimization_plan.torch_compile:
            self.training_loop._torch_compile_active = True
        # Pass compile scope so CUDAGraph capture can be triggered on eligible steps
        self.training_loop._anima_compile_scope = str(
            getattr(self.config, "anima_compile_scope", "") or ""
        )
        self._maybe_probe_anima_full_core_compile(dataloader)
        self._attach_compile_runtime_profile_to_training_loop()
        self._initialize_resource_manager()
        self._initialize_ema_tracker()

        # 设置 Lulynx 包装器
        if hasattr(self, '_lulynx_wrapper'):
            self.training_loop.set_lulynx_wrapper(self._lulynx_wrapper)
        if self.training_loop.pcgrad_enabled:
            self._log(
                f"PCGrad enabled (threshold={self.training_loop.pcgrad_conflict_threshold}, "
                f"reduction={self.training_loop.pcgrad_reduction})"
            )

        sample_groups = self._get_preview_groups()
        # Anima preview is now supported via flow-matching sampler
        if sample_groups and (
            getattr(self.config, "sample_every", 0) > 0
            or getattr(self.config, "sample_every_n_epochs", 0) > 0
        ):
            try:
                from .sampler import create_sampler_from_trainer, get_preview_state, PreviewState
                preview_state = get_preview_state(self)
                preview_device = str(getattr(self.config, "preview_device", "gpu") or "gpu").strip().lower()
                if preview_state == PreviewState.REAL_PREVIEW or preview_device == "cpu":
                    self._sampler = create_sampler_from_trainer(self)
                    if self._sampler:
                        if bool(getattr(self.config, "micro_vae_preview", False)):
                            if hasattr(self._sampler, "load_micro_decoder"):
                                self._sampler.load_micro_decoder(
                                    str(getattr(self.config, "micro_vae_model", "auto") or "auto")
                                )
                        mode = "CPU preview queue" if preview_device == "cpu" else "Sampling"
                        self._log(f"{mode} enabled ({len(sample_groups)} preview groups)")
                        if bool(getattr(self.config, "sample_at_first", False)):
                            try:
                                self._run_sampling(0, current_epoch=0)
                            except Exception as e:
                                logger.warning(f"Initial sampling failed: {e}")
                    else:
                        self._log("Sampling requested, but sampler creation failed.")
                elif preview_state == PreviewState.ADAPTER_INSPECT:
                    self._log(
                        "Real-time preview unavailable (TE/VAE released in cache-first mode). "
                        "Adapter metadata inspection is still available."
                    )
                else:
                    self._log("Sampling requested, but this training mode has no compatible sampler pipeline.")
            except Exception as e:
                self._log(f"Sampling initialization failed: {e}")

            # 训练前准备 (捕获基线)
            self._log("Capturing baseline for Manifold Constraint...")
            baseline_inputs = self._build_lulynx_baseline_inputs()

            # 执行 Baseline 捕获 (需要 forward 一个样本)
            if not self._lulynx_wrapper:
                self._log("Lulynx wrapper not available; skipping baseline capture.")
                baseline_stats = {}
            else:
                baseline_stats = self._lulynx_wrapper.pre_training(
                    model=self.model.unet,
                    sample_inputs=baseline_inputs,
                    network=self.lora_injector,
                    unet=self.model.unet,
                    text_encoder=self.model.text_encoder_1,
                )
            if baseline_stats.get("manifold_baseline_captured"):
                self._log("Baseline captured.")
            else:
                self._log("Baseline capture skipped or failed; continuing without manifold baseline.")

        # 设置回调
        self.training_loop.on_step_end = self._on_step_end
        self.training_loop.on_epoch_end = self._on_epoch_end
        self.training_loop.on_before_train_step = (
            lambda step: self._apply_anima_progressive_full_finetune(global_step=step, reason="before_train_step")
        )
        self.training_loop.on_before_optimizer_step = (
            lambda step: self._apply_anima_progressive_full_finetune(global_step=step, reason="before_optimizer_step")
        )

        # 集成审计
        if self.config.enable_auditor:
            try:
                from ..auditor import LoRAAuditor, AuditConfig, SVDAlgorithm

                # 配置转换
                svd_algo = SVDAlgorithm.STANDARD if self.config.monitor_svd_algo == "full" else SVDAlgorithm.RSVD

                audit_config = AuditConfig(
                    svd_algorithm=svd_algo,
                    advanced_stats_enabled=True, # 启用高级统计以支持 SmartRank
                )

                auditor = LoRAAuditor(config=audit_config)
                if hasattr(auditor, 'watchdog') and hasattr(self.config, 'audit_mode_override'):
                    override = str(getattr(self.config, 'audit_mode_override', '') or '')
                    if override:
                        auditor.watchdog._mode_override = override.upper().strip()
                self.training_loop.set_auditor(auditor, self.config.auditor_interval)
                self._log(f"Auditor enabled (Algo: {self.config.monitor_svd_algo})")
            except Exception as e:
                self._log(f"LoRAAuditor initialization failed: {e}")

        # 集成 SafeGuard (Loss Spike Detection & Bad Sample Culling)
        try:
            sg_config = SafeGuardConfig(
                enable_loss_spike_detection=getattr(self.config, "so_enable_loss_spike_detection", True),
                loss_spike_threshold=getattr(self.config, "so_loss_spike_threshold", 2.5),
                loss_window_size=getattr(self.config, "so_loss_window_size", 50),
                enable_nan_detection=getattr(self.config, "so_enable_nan_detection", True),
                nan_check_interval=getattr(self.config, "so_nan_check_interval", 10),
                gradient_check_interval=getattr(
                    self.config,
                    "so_gradient_check_interval",
                    getattr(self.config, "so_nan_check_interval", 10),
                ),
                gradient_scan_mode=getattr(self.config, "so_gradient_scan_mode", "batched"),
                max_nan_count=getattr(self.config, "so_max_nan_count", 3),
                enable_lr_deadlock_detection=getattr(self.config, "so_enable_lr_deadlock_detection", True),
                lr_deadlock_threshold=getattr(self.config, "so_lr_deadlock_threshold", 1e-8),
                lr_deadlock_steps=getattr(self.config, "so_lr_deadlock_steps", 200),
                enable_auto_recovery=getattr(self.config, "so_enable_auto_recovery", True),
                lr_reduction_factor=getattr(self.config, "so_lr_reduction_factor", 0.5),
                enable_bad_sample_culling=getattr(
                    self.config,
                    "so_enable_bad_sample_culling",
                    getattr(self.config, "enable_bad_sample_culling", False),
                ),
                bad_sample_mode=str(getattr(self.config, "so_bad_sample_mode", "report") or "report"),
                quarantine_dir=getattr(
                    self.config,
                    "so_quarantine_dir",
                    getattr(self.config, "quarantine_dir", "quarantine"),
                ),
                bad_sample_report_path=str(
                    Path(self.config.output_dir) / str(getattr(self.config, "so_bad_sample_report_name", "safeguard_events.jsonl") or "safeguard_events.jsonl")
                ),
                max_reported_samples=int(getattr(self.config, "so_bad_sample_max_reported", 32) or 32),
                on_cull_samples=self.on_cull_samples,
            )
            safeguard = TrainingSafeGuard(sg_config)
            self.training_loop.set_safeguard(safeguard)
            self._log(f"SafeGuard enabled (bad_sample_culling={sg_config.enable_bad_sample_culling}, mode={sg_config.bad_sample_mode})")
        except Exception as e:
            self._log(f"SafeGuard initialization failed: {e}")

        # 集成 LISA
        if self.config.lisa_enabled:
            try:
                from .lisa import LISAScheduler
                lisa_scheduler = LISAScheduler(
                    model=self.model.unet,
                    active_ratio=self.config.lisa_active_ratio,
                    interval=self.config.lisa_interval,
                )
                self.training_loop.set_lisa_scheduler(lisa_scheduler)
                self._log(f"LISA enabled (ratio: {self.config.lisa_active_ratio})")
            except Exception as e:
                self._log(f"LISA initialization failed: {e}")

        # === V3.0: 高级功能集成 ===

        # AutoController 自动控制器
        controller_enabled = any([
            bool(getattr(self.config, "auto_controller_enabled", False)),
            bool(getattr(self.config, "ac_enabled", False)),
            bool(getattr(self.config, "ac_enable_smart_early_stopping", False)),
            bool(getattr(self.config, "ac_enable_smart_lr_decay", False)),
            bool(getattr(self.config, "ac_enable_auto_lr_adjustment", False)),
            bool(getattr(self.config, "ac_enable_dynamic_loss_scaling", False)),
        ])
        if controller_enabled:
            try:
                from .auto_controller import AutoController, AutoControlConfig

                clip_drift_warning = float(getattr(self.config, "ac_clip_drift_warning", 0.03) or 0.03)
                clip_drift_danger = float(
                    getattr(self.config, "ac_clip_drift_danger", 0.0)
                    or getattr(self.config, "clip_drift_threshold", 0.05)
                    or 0.05
                )
                smart_lr_decay_enabled = bool(
                    getattr(self.config, "smart_lr_decay", False)
                    or getattr(self.config, "ac_enable_smart_lr_decay", False)
                    or getattr(self.config, "ac_enable_auto_lr_adjustment", False)
                )
                stable_rank_threshold = float(getattr(self.config, "ac_stable_rank_collapse_threshold", 0.3) or 0.3)
                if stable_rank_threshold <= 0 or stable_rank_threshold > 1:
                    stable_rank_threshold = 0.3

                ac_config = AutoControlConfig(
                    auto_freeze_te=bool(
                        getattr(self.config, "auto_freeze_te", False)
                        or getattr(self.config, "ac_enable_auto_te_freeze", False)
                        or (controller_enabled and (clip_drift_warning > 0 or clip_drift_danger > 0))
                    ),
                    clip_drift_warning=clip_drift_warning,
                    clip_drift_danger=clip_drift_danger,
                    clip_drift_consecutive=int(getattr(self.config, "ac_clip_drift_consecutive", 5) or 5),
                    smart_early_stop=bool(
                        getattr(self.config, "smart_early_stop", False)
                        or getattr(self.config, "ac_enable_smart_early_stopping", False)
                    ),
                    stable_rank_collapse_threshold=stable_rank_threshold,
                    stable_rank_consecutive=int(
                        getattr(self.config, "ac_stable_rank_consecutive", 0)
                        or getattr(self.config, "ac_early_stopping_patience", 10)
                        or 10
                    ),
                    loss_plateau_window=int(getattr(self.config, "ac_loss_plateau_window", 50) or 50),
                    smart_lr_decay=smart_lr_decay_enabled,
                    lr_decay_factor=float(
                        getattr(self.config, "ac_lr_decay_factor", 0.0)
                        or getattr(self.config, "ac_decay_factor", 0.5)
                        or 0.5
                    ),
                    gradient_rank_plateau_window=int(getattr(self.config, "ac_gradient_rank_plateau_window", 30) or 30),
                    max_decays=int(getattr(self.config, "ac_max_decays", 3) or 3),
                    target_gsnr=float(getattr(self.config, "ac_target_gsnr", 5.0) or 5.0),
                    batch_size_step=int(getattr(self.config, "ac_batch_size_step", 1) or 1),
                    warmup_steps=int(
                        getattr(self.config, "ac_warmup_steps", 0)
                        or getattr(self.config, "ac_warmup_shield", 100)
                        or 100
                    ),
                )
                self._auto_controller = AutoController(ac_config)
                self._log("AutoController enabled")
            except ImportError as e:
                self._log(f"AutoController not available: {e}")

        # Coreset 智能采样
        if self.config.coreset_enabled:
            try:
                from .coreset import CoresetManager
                self._coreset_manager = CoresetManager(
                    easy_weight=self.config.coreset_easy_weight,
                    hard_weight=self.config.coreset_hard_weight,
                    toxic_weight=getattr(self.config, "coreset_toxic_weight", 0.0),
                    auto_classify_after=getattr(
                        self.config,
                        "coreset_auto_classify_after",
                        getattr(self.config, "coreset_classify_after", 50),
                    ),
                    easy_threshold=getattr(self.config, "coreset_easy_threshold", 0.1),
                    hard_loss_threshold=getattr(self.config, "coreset_hard_loss_threshold", 1.5),
                    toxic_std_threshold=getattr(self.config, "coreset_toxic_std_threshold", 3.0),
                )
                self._log("Coreset diagnostics enabled")
            except ImportError as e:
                self._log(f"Coreset not available: {e}")

        # 集成 SmartRank (V2.1 Core)
        if self.config.smart_rank_enabled:
            try:
                from .smart_rank import SmartRankController
                smart_rank_controller = SmartRankController(
                    lora_injector=self.lora_injector,
                    min_rank=self.config.smart_rank_min,
                    max_rank=self.config.smart_rank_max,
                    interval=self.config.smart_rank_interval,
                )
                self.training_loop.set_smart_rank_controller(smart_rank_controller)
                self.training_loop.on_params_changed = self._on_params_changed
                self._log(f"SmartRank enabled (interval: {self.config.smart_rank_interval})")
            except Exception as e:
                self._log(f"SmartRank initialization failed: {e}")

        # 输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_training_advisor_report()

        # Resume Checkpoint (State)
        start_epoch = 0
        if self.config.resume_path:
            self._preflight_resume_manifest()
            state = self._load_state(self.config.resume_path)
            if state:
                start_epoch = state.get("epoch", 0)
                if self.training_loop:
                    self.training_loop.global_step = state.get("global_step", 0)
                    if "optimizer_state_dict" in state:
                        try:
                            optimizer.load_state_dict(state["optimizer_state_dict"])
                        except Exception as e:
                           self._log(f"Warning: Failed to load optimizer state: {e}")
                    if "scheduler_state_dict" in state:
                        try:
                            scheduler.load_state_dict(state["scheduler_state_dict"])
                        except Exception as e:
                            self._log(f"Warning: Failed to load scheduler state: {e}")
                    if self._ema_tracker and state.get("ema_state_dict"):
                        try:
                            self._ema_tracker.load_state_dict(state["ema_state_dict"])
                            self._log("EMA state restored")
                        except Exception as e:
                            self._log(f"Warning: Failed to load EMA state: {e}")
                    if self._resource_manager and state.get("resource_manager_state"):
                        try:
                            rm_state = state["resource_manager_state"]
                            self._resource_manager.current_batch_size = int(
                                rm_state.get("current_batch_size", self._resource_manager.current_batch_size)
                            )
                            self._resource_manager.current_accumulation = int(
                                rm_state.get("current_accumulation", self._resource_manager.current_accumulation)
                            )
                            self.training_loop.gradient_accumulation_steps = self._resource_manager.current_accumulation
                            self.config.gradient_accumulation_steps = self._resource_manager.current_accumulation
                            self._log(
                                f"ResourceManager state restored (accumulation={self._resource_manager.current_accumulation})"
                            )
                        except Exception as e:
                            self._log(f"Warning: Failed to load ResourceManager state: {e}")
                    if state.get("turbocore_update_state") and hasattr(self.training_loop, "load_turbocore_update_checkpoint_state"):
                        try:
                            report = self.training_loop.load_turbocore_update_checkpoint_state(state["turbocore_update_state"])
                            if report.get("loaded"):
                                self._log(
                                    "TurboCore update checkpoint metadata restored "
                                    f"(compatible={bool(report.get('compatible', False))}, "
                                    f"owner_state_pending={bool(report.get('owner_state_pending', False))})"
                                )
                        except Exception as e:
                            self._log(f"Warning: Failed to load TurboCore update metadata: {e}")
                self._log(f"Resuming training from Epoch {start_epoch}, Step {self.training_loop.global_step}")

        initial_epoch = max(int(getattr(self.config, "initial_epoch", 0) or 0), 0)
        initial_step = max(int(getattr(self.config, "initial_step", 0) or 0), 0)
        skip_until_initial_step = bool(getattr(self.config, "skip_until_initial_step", False))
        if initial_epoch > start_epoch:
            start_epoch = min(initial_epoch, int(self.config.max_train_epochs))
            self._log(f"Initial epoch override applied: starting from epoch {start_epoch}")
        if self.training_loop and initial_step > self.training_loop.global_step and not skip_until_initial_step:
            self.training_loop.global_step = initial_step
            self._log(f"Initial step override applied: global_step={initial_step}")
        if self.training_loop and skip_until_initial_step and initial_step > self.training_loop.global_step:
            self.training_loop.initial_step_target = initial_step
            self.training_loop.skip_until_initial_step = True
            self._log(f"Skipping dataloader batches until initial_step={initial_step}")

        if self.training_loop:
            self._apply_anima_progressive_full_finetune(
                global_step=int(getattr(self.training_loop, "global_step", 0) or 0),
                reason="before_training",
            )
            self._refresh_anima_full_finetune_experiments_profile()

        self._mark_runtime_phase("before_training")
        self._log_runtime_phase_summary()
        self._log("Starting training...")
        start_time = time.time()

        # 训练循环
        save_every_epochs = self._resolve_epoch_save_interval(int(self.config.max_train_epochs))

        # Staged resolution: parse step boundaries and target resolutions
        anima_staged_resolution_training = bool(anima_cached_training and self._anima_staged_resolution_plan)
        staged_steps = [] if (anima_staged_resolution_training or self._lora_staged_resolution_enabled_runtime) else self._parse_staged_resolution_steps()
        staged_res_index = 0
        if staged_steps:
            self._apply_staged_resolution(staged_steps[0][1])

        for epoch in range(start_epoch, self.config.max_train_epochs):
            if self._should_stop:
                break

            if self.training_loop:
                self._apply_anima_progressive_full_finetune(
                    global_step=int(getattr(self.training_loop, "global_step", 0) or 0),
                    reason=f"epoch_{epoch + 1}_start",
                )

            # Staged resolution: check if we need to change resolution at this epoch
            if staged_steps and self.training_loop:
                current_step = self.training_loop.global_step
                while (staged_res_index < len(staged_steps)
                       and current_step >= staged_steps[staged_res_index][0]):
                    self._apply_staged_resolution(staged_steps[staged_res_index][1])
                    self._log(f"Staged resolution: switching to {staged_steps[staged_res_index][1]} at step {current_step}")
                    staged_res_index += 1

            if anima_staged_resolution_training:
                dataloader = self._maybe_switch_anima_staged_resolution_dataset(
                    dataloader=dataloader,
                    epoch=epoch,
                    drop_last=compile_drop_last,
                )
            elif self._lora_staged_resolution_enabled_runtime:
                dataloader = self._maybe_switch_lora_staged_resolution_dataset(
                    dataloader=dataloader,
                    epoch=epoch,
                )

            if hasattr(self, "_dataset") and self._dataset and hasattr(self._dataset, "set_current_epoch"):
                self._dataset.set_current_epoch(epoch)

            # DDP: inform sampler of epoch for proper shuffling
            if self._ddp_wrapper is not None:
                self._ddp_wrapper.set_epoch(epoch)

            self._refresh_dataloader_rebuild_readiness(
                dataloader,
                epoch=epoch,
                boundary="epoch_start",
            )
            dataloader = self._maybe_apply_bubble_epoch_boundary_dataloader_rebuild(
                dataloader,
                epoch=epoch,
            )

            epoch_wall_start = time.perf_counter()
            result = self.training_loop.train_epoch(dataloader, epoch)
            self._runtime_phase_timings.append(
                {
                    "label": f"epoch_{epoch + 1}_train",
                    "dt_seconds": round(max(time.perf_counter() - epoch_wall_start, 0.0), 4),
                    "total_seconds": round(max(time.perf_counter() - self._runtime_phase_start_time, 0.0), 4),
                    **(
                        {
                            "cuda_allocated_mb": round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 3),
                            "cuda_reserved_mb": round(float(torch.cuda.memory_reserved()) / (1024 * 1024), 3),
                            "cuda_peak_allocated_mb": round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 3),
                        }
                        if torch.cuda.is_available() and str(getattr(self, "device", "")).startswith("cuda")
                        else {}
                    ),
                }
            )
            self._log(f"Epoch {epoch + 1} completed, avg loss: {result['avg_loss']:.4f}")

            # Stop the epoch loop the moment the step budget (max_train_steps) is
            # reached. train_epoch() already set completed_by_step_limit and broke
            # out with 0 further steps; without this break the loop would keep
            # spinning empty epochs up to max_train_epochs (each saving a
            # checkpoint), which can exhaust the disk. The post-loop final save
            # below already handles completed_by_step_limit, so the trained
            # adapter is still persisted.
            if getattr(self.training_loop, "completed_by_step_limit", False):
                self._log(
                    f"Reached max_train_steps; stopping epoch loop after epoch {epoch + 1}"
                )
                break

            # Validation
            validation_every = int(getattr(self.config, "eval_every_n_epochs", 0) or 0)
            step_validation_every = int(getattr(self.config, "eval_every_n_steps", 0) or 0)
            if validation_every <= 0 and step_validation_every <= 0:
                validation_every = int(getattr(self.config, "validation_every_n_epochs", 1) or 1)
            if validation_every > 0 and self.training_loop.validation_dataloader is not None:
                if (epoch + 1) % validation_every == 0:
                    try:
                        val_result = self.training_loop.validate_epoch(
                            self.training_loop.validation_dataloader, epoch
                        )
                        self._log(
                            f"Validation {epoch + 1}: avg_loss={val_result['avg_loss']:.4f} "
                            f"({val_result['steps']} steps)"
                        )
                    except Exception as e:
                        self._log(f"Validation failed at epoch {epoch + 1}: {e}")

            # 保存模型
            if (epoch + 1) % save_every_epochs == 0:
                self._sync_turbocore_native_update_state("before_epoch_save")
                self._save_model(epoch + 1)
            self._maybe_cooldown_after_epoch(epoch, int(self.config.max_train_epochs))

        # 最终保存
        completed_by_step_limit = bool(
            self.training_loop and getattr(self.training_loop, "completed_by_step_limit", False)
        )
        if not self._should_stop or completed_by_step_limit:
            self._close_turbocore_native_update_executor()
            self._save_model(self.config.max_train_epochs, final=True)
            self._maybe_save_final_training_state(self.config.max_train_epochs)
            self._refresh_pcie_delta_cache_reports(log=True)
            self._write_run_manifest("completed", epoch=int(self.config.max_train_epochs))
            self._write_coreset_report(epoch=int(self.config.max_train_epochs) - 1, final=True)

        duration = time.time() - start_time
        self._refresh_pcie_delta_cache_reports(log=True)
        self._log(f"Training completed in {duration:.1f}s")

        # Newbie auto_swap_release: release VRAM after training
        if (
            model_arch == "newbie"
            and getattr(self.config, "newbie_auto_swap_release", False)
        ):
            self._log("Newbie auto_swap_release: releasing GPU memory after training.")
            try:
                # Move model components to CPU to free VRAM
                if self.model:
                    if hasattr(self.model, "unet") and self.model.unet is not None:
                        self.model.unet.to("cpu")
                    if hasattr(self.model, "vae") and self.model.vae is not None:
                        self.model.vae.to("cpu")
                    if hasattr(self.model, "text_encoder_1") and self.model.text_encoder_1 is not None:
                        self.model.text_encoder_1.to("cpu")
                    if hasattr(self.model, "text_encoder_2") and self.model.text_encoder_2 is not None:
                        self.model.text_encoder_2.to("cpu")
                self._log("Newbie auto_swap_release: model components moved to CPU.")
                self._maybe_release_tool_cuda_cache(
                    "newbie_auto_swap_release",
                    collect_gc=True,
                )
            except Exception as e:
                self._log(f"Newbie auto_swap_release: cleanup failed — {e}")

    # ------------------------------------------------------------------
    # Bubble closed-loop methods moved to trainer_bubble_runtime.TrainerBubbleRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _check_hot_swap(self, step: int):
        """检查是否有 Pit Stop 策略更新 (Hot Swap)"""
        # 每 10 步检查一次，避免 IO 过于频繁
        if step % 10 != 0:
            return

        output_dir = Path(self.config.output_dir)
        update_file = output_dir / "pit_stop_update.json"

        if update_file.exists():
            try:
                import json
                with open(update_file, 'r', encoding='utf-8') as f:
                    updates = json.load(f)

                self._log(f"[Pit Stop] Detected update request: {updates}")

                # 1. Update Learning Rate
                if "learning_rate" in updates:
                    new_lr = float(updates["learning_rate"])
                    # Update optimizer
                    if self.training_loop and self.training_loop.optimizer:
                        for param_group in self.training_loop.optimizer.param_groups:
                            param_group['lr'] = new_lr
                    self._log(f"[Pit Stop] Learning Rate updated to {new_lr}")

                # 2. Update MN-LoRA k_ratio (Geometric Weight)
                if "k_ratio" in updates:
                    new_k = float(updates["k_ratio"])
                    self.config.mn_lora_k_ratio = new_k
                    # Update Wrapper
                    if hasattr(self, '_lulynx_wrapper') and self._lulynx_wrapper:
                        self._lulynx_wrapper.config.geometric_weight = new_k
                    self._log(f"[Pit Stop] MN-LoRA k_ratio updated to {new_k}")

                # 3. Update Weight Decay
                if "weight_decay" in updates:
                    new_wd = float(updates["weight_decay"])
                    # Update optimizer
                    if self.training_loop and self.training_loop.optimizer:
                        for param_group in self.training_loop.optimizer.param_groups:
                             # 只有原本有 weight_decay 的 group 才更新，避免把 embedding (wd=0) 也改了
                            if param_group.get('weight_decay', 0) > 0:
                                param_group['weight_decay'] = new_wd
                    self._log(f"[Pit Stop] Weight Decay updated to {new_wd}")

                # Consume file
                try:
                    update_file.unlink()
                except Exception:
                    pass

            except Exception as e:
                self._log(f"[Pit Stop] Failed to apply updates: {e}")

    def _on_step_end(self, step: int, loss: float, info: Dict):
        """步骤结束回调"""
        self._last_loss = loss
        lr = info.get("lr", 0.0)

        # Update dataset step for token warmup
        if hasattr(self, '_dataset') and self._dataset and hasattr(self._dataset, 'set_global_step'):
            self._dataset.set_global_step(step)
        epoch = info.get("epoch", 0)
        runtime_features: Dict[str, Any] = {}
        b_tier = info.get("b_tier")
        if b_tier:
            runtime_features["b_tier"] = b_tier
        pcgrad = info.get("pcgrad")
        if pcgrad:
            runtime_features["pcgrad"] = pcgrad
        native_unet_status = getattr(self, "_native_unet_status", None)
        if native_unet_status:
            runtime_features["native_unet"] = native_unet_status
        low_vram_profile_status = getattr(self, "_sdxl_lora_low_vram_profile", None)
        if low_vram_profile_status:
            runtime_features["sdxl_lora_low_vram_profile"] = dict(low_vram_profile_status)
        if self.training_loop is not None:
            memory_optimization = getattr(self.training_loop, "memory_optimization_state", None)
            if memory_optimization:
                runtime_features["memory_optimization"] = memory_optimization
        if info.get("step_wall_seconds") is not None:
            runtime_features["step_wall_seconds"] = float(info.get("step_wall_seconds") or 0.0)
        if info.get("peak_vram_stages"):
            runtime_features["peak_vram_stages"] = info["peak_vram_stages"]
        if info.get("peak_vram_diagnostics"):
            runtime_features["peak_vram_diagnostics"] = info["peak_vram_diagnostics"]
        if info.get("cuda_cache_release"):
            runtime_features["cuda_cache_release"] = info["cuda_cache_release"]
        if info.get("precision_swap_offload"):
            runtime_features["precision_swap_offload"] = info["precision_swap_offload"]
        if info.get("data_transfer_profile"):
            runtime_features["data_transfer_profile"] = info["data_transfer_profile"]
        data_profile = getattr(self, "_data_backend_profile", {}) or {}
        native_cache_reader = data_profile.get("native_cache_reader") if isinstance(data_profile, dict) else None
        decode_sidecar_profile = native_cache_reader.get("decode_sidecar") if isinstance(native_cache_reader, dict) else None
        if isinstance(decode_sidecar_profile, dict) and decode_sidecar_profile:
            runtime_features["native_cache_reader_decode_sidecar"] = dict(decode_sidecar_profile)
        training_gate_profile = native_cache_reader.get("training_gate") if isinstance(native_cache_reader, dict) else None
        if isinstance(training_gate_profile, dict) and training_gate_profile:
            runtime_features["native_cache_reader_training_gate"] = dict(training_gate_profile)
        if info.get("vram_smart_sensing_runtime"):
            runtime_features["vram_smart_sensing_runtime"] = info["vram_smart_sensing_runtime"]
        if self.training_loop is not None and is_anima_full_finetune(self.config):
            experiment_profile = self._refresh_anima_full_finetune_experiments_profile()
            if experiment_profile:
                runtime_features["anima_full_finetune_experiments"] = dict(experiment_profile)
        if info.get("attention_stats"):
            runtime_features["attention_stats"] = info["attention_stats"]
        if info.get("audit_mode"):
            runtime_features["audit_mode"] = info["audit_mode"]
        if info.get("attn_entropy") is not None:
            runtime_features["attn_entropy"] = info["attn_entropy"]
        if info.get("act_drift"):
            runtime_features["act_drift"] = info["act_drift"]
        if info.get("loss_modifiers"):
            runtime_features["loss_modifiers"] = info["loss_modifiers"]
        if info.get("grad_stats"):
            runtime_features["grad_stats"] = info["grad_stats"]
        if info.get("hessian_trace") is not None:
            runtime_features["hessian_trace"] = info["hessian_trace"]
        if info.get("hessian_layers"):
            runtime_features["hessian_layers"] = info["hessian_layers"]
        if info.get("layer_monitor"):
            runtime_features["layer_monitor"] = info["layer_monitor"]
        if info.get("forgetting"):
            runtime_features["forgetting"] = info["forgetting"]
        if self.training_loop and self.training_loop.auditor:
            last_report = self.training_loop.auditor.get_last_report()
            if last_report and last_report.get("metrics"):
                icu = last_report["metrics"].get("icu_score")
                if icu is not None:
                    info["icu_score"] = icu
                    runtime_features["icu_score"] = icu

        self._record_bubble_closed_loop_step_sample(step, loss, info)
        bubble_window = self._bubble_closed_loop_window_profile()
        if bubble_window:
            runtime_features["bubble_closed_loop_window"] = bubble_window
        self._maybe_run_bubble_closed_loop(step, loss, info)
        if self._bubble_closed_loop_last_report:
            runtime_features["bubble_closed_loop"] = self._bubble_closed_loop_last_report.get("closed_loop", {})

        self._emit_runtime_event(
            {
                "event_type": "step",
                "step": int(step),
                "epoch": int(epoch),
                "severity": "info",
                "summary": f"loss={float(loss):.6f}",
                "data": {
                    "loss": float(loss),
                    "lr": float(lr),
                    **runtime_features,
                },
            }
        )

        # 文件级日志（供 API 端点解析）
        import json as _json
        if info.get("hessian_layers"):
            try:
                print(f"HESSIAN_JSON:{_json.dumps({'step': int(step), 'layers': info['hessian_layers']})}")
            except Exception:
                pass
        if (
            self.training_loop
            and self.training_loop._manifold_tracker is not None
            and self.training_loop._manifold_tracker.num_snapshots >= 4
            and int(step) % 100 == 0
        ):
            try:
                _m_result = self.training_loop._manifold_tracker.compute_pca()
                if _m_result:
                    print(f"MANIFOLD_PCA_JSON:{_json.dumps(_m_result.as_dict())}")
            except Exception:
                pass

        # === V5.0: Pit Stop Hot Swap ===
        self._check_hot_swap(step)

        # Progress/API/TensorBoard logging is pure reporting, so it can be
        # adaptively throttled without skipping training-control hooks below.
        if self._should_emit_step_logging(step):
            logging_started = time.perf_counter()
            try:
                # 进度回调
                if self.on_progress:
                    total_steps = getattr(self, "_total_steps", None) or getattr(self.training_loop, "total_steps", None)
                    if not total_steps:
                        steps_per_epoch = getattr(self.training_loop, "steps_per_epoch", 0) or 0
                        total_steps = steps_per_epoch * self.config.epochs if steps_per_epoch else (self.config.epochs * 1000)
                    self.on_progress(step, int(total_steps), loss)

                # 新增: on_step 回调 (用于后端 API)
                if self.on_step:
                    try:
                        self.on_step(step, epoch, loss, lr)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:
                        # AUDIT FIX: Log callback error instead of silent catch
                        logger.error(f"Error in on_step callback: {e}")

                try:
                    self._write_step_log(step, loss, lr, info=info)
                except Exception as exc:
                    logger.debug("Step logging write failed: %s", exc)
            finally:
                self._record_step_logging_overhead(
                    step,
                    time.perf_counter() - logging_started,
                    float(info.get("step_wall_seconds", 0.0) or 0.0),
                )

        if self._ema_tracker:
            try:
                current_state = self._get_current_adapter_state_dict(use_ema=False)
                if current_state:
                    self._ema_tracker.step(current_state)
            except Exception as e:
                logger.warning(f"EMA update failed: {e}")

        if self._resource_manager:
            try:
                rm_result = self._resource_manager.step()
                status = rm_result.get("vram_status", "ok")
                if status != self._last_vram_status:
                    if status == "ok":
                        self._log("[ResourceManager] VRAM usage returned to normal.")
                    else:
                        self._log(
                            f"[ResourceManager] VRAM status={status} (usage={self._resource_manager._last_vram_usage:.1%})"
                        )
                    self._last_vram_status = status

                if (
                    self.training_loop
                    and getattr(self.config, "rm_enable_adaptive_accumulation", False)
                ):
                    new_accumulation = int(
                        rm_result.get("new_accumulation", self._resource_manager.current_accumulation)
                    )
                    if new_accumulation != self.training_loop.gradient_accumulation_steps:
                        self.training_loop.gradient_accumulation_steps = new_accumulation
                        self.config.gradient_accumulation_steps = new_accumulation
                        self._log(f"[ResourceManager] Gradient accumulation adjusted to {new_accumulation}")
            except Exception as e:
                logger.warning(f"ResourceManager step failed: {e}")

        # 动态剪枝
        if self._dynamic_pruner:
            try:
                total_steps = self.config.max_train_steps or (self.config.epochs * 1000)
                prune_result = self._dynamic_pruner.step(
                    step=step,
                    total_steps=total_steps,
                    network=self.lora_injector,
                    optimizer=self.training_loop.optimizer,
                )
                if prune_result:
                    self._log(f"Pruned {prune_result.get('layers_pruned', 0)} layers")
            except RuntimeError as e:
                # AUDIT FIX: Catch specific tensor/optimizer errors
                logger.error(f"Dynamic pruning runtime error: {e}")
            except Exception as e:
                logger.warning(f"Dynamic pruning unexpected failure: {e}")

        # === V3.0: AutoController 自动控制 ===
        if self._auto_controller:
            try:
                # 从 auditor 获取指标 (如果可用)
                metrics = {}
                if getattr(self.training_loop, "auditor", None):
                    auditor_report = self.training_loop.auditor.get_last_report()
                    auditor_stats = auditor_report.get("metrics", {}) if auditor_report else {}
                    if auditor_stats:
                        metrics.update({
                            "clip_drift": auditor_stats.get("clip_drift", 0),
                            "stable_rank": auditor_stats.get("stable_rank", 0),
                            "gsnr": auditor_stats.get("gsnr", 0),
                            "gradient_rank": auditor_stats.get("gradient_rank", auditor_stats.get("gsnr", 0)),
                        })
                metrics["loss"] = loss

                # 调用控制器
                result = self._auto_controller.step(
                    step=step,
                    metrics=metrics,
                    model=self.model.unet,
                    optimizer=self.training_loop.optimizer,
                )

                # 处理结果
                if result.get("te_frozen"):
                    self._log("[AutoController] Text Encoder frozen (CLIP drift detected)")
                if result.get("should_stop") is True:
                    self._log(f"[AutoController] Early stop triggered: {result.get('reason')}")
                    self._should_stop = True
                if result.get("lr_decayed"):
                    self._log(f"[AutoController] LR decayed (count: {result.get('decay_count')})")
            except (ValueError, KeyError, RuntimeError) as e:
                # AUDIT FIX: Catch logic and runtime errors separately
                logger.error(f"AutoController logic/runtime error: {e}")
            except Exception as e:
                logger.warning(f"AutoController unexpected failure: {e}")

        try:
            get_orchestra().step()
        except Exception as e:
            logger.debug(f"Orchestra step update skipped: {e}")

        try:
            if (
                self.training_loop
                and self.training_loop.optimizer
                and self.te_manager
                and not getattr(self.config, "semantic_tuner_enabled", False)
                and not getattr(self, "_ti_mode", False)
                and getattr(self.config, "train_text_encoder", False)
            ):
                removed = self.te_manager.maybe_remove_text_encoders(
                    text_encoder_1=self.model.text_encoder_1,
                    text_encoder_2=self.model.text_encoder_2,
                    optimizer=self.training_loop.optimizer,
                )
                if removed:
                    if self._sampler is not None:
                        self._sampler = None
                        self._log("[TE Manager] Sampling disabled after TE offload (sampler pipeline shares text encoders).")
                    self._log("[TE Manager] Text encoders offloaded to CPU after dual validation.")
        except Exception as e:
            logger.warning(f"TE removal check failed: {e}")

        # === V3.0: Coreset 统计更新 ===
        if self._coreset_manager and info.get("filenames"):
            try:
                filenames = info["filenames"]
                losses = info.get("per_sample_losses", [loss] * len(filenames))
                self._coreset_manager.update_batch(filenames, losses)
            except (KeyError, ValueError) as e:
                logger.error(f"Coreset data error: {e}")
            except Exception as e:
                logger.warning(f"Coreset update unexpected failure: {e}")

        # 训练中采样
        if self._sampler and self.config.sample_every > 0:
            if step > 0 and step % self.config.sample_every == 0:
                try:
                    self._run_sampling(step, current_epoch=epoch + 1)
                except Exception as e:
                    logger.warning(f"Sampling failed: {e}")

        save_every_n_steps = max(int(getattr(self.config, "save_every_n_steps", 0) or 0), 0)
        if save_every_n_steps > 0 and step > 0 and step % save_every_n_steps == 0:
            try:
                self._sync_turbocore_native_update_state("before_step_save")
                self._save_model(epoch=epoch + 1, step=step)
            except Exception as e:
                logger.warning(f"Step checkpoint save failed at step {step}: {e}")

    def _run_sampling(self, step: int, current_epoch: Optional[int] = None):
        """运行训练中采样"""
        groups = self._filter_preview_groups_for_epoch(self._get_preview_groups(), current_epoch)
        if not self._sampler or not groups:
            return

        epoch_suffix = f" (epoch {current_epoch})" if current_epoch is not None else ""
        self._log(f"Generating samples at step {step}{epoch_suffix}...")

        output_dir = Path(self.config.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, group in enumerate(groups):
            try:
                sample_w = int(group.get("width", 0) or 0)
                sample_h = int(group.get("height", 0) or 0)
                sample_seed = int(group.get("seed", 0) or 0)
                mode = str(group.get("mode", "lora") or "lora").strip().lower()
                lora_scale = 0.0 if mode == "base" else float(group.get("lora_weight", 1.0) or 1.0)
                gen_kwargs = dict(
                    prompt=str(group.get("prompt") or ""),
                    negative_prompt=str(group.get("negative_prompt") or ""),
                    num_inference_steps=int(group.get("steps", getattr(self.config, "sample_steps", 20)) or 20),
                    guidance_scale=float(group.get("cfg", getattr(self.config, "sample_cfg", 7.5)) or 7.5),
                )
                if sample_w > 0 and sample_h > 0:
                    gen_kwargs["width"] = sample_w
                    gen_kwargs["height"] = sample_h
                if sample_seed != 0:
                    gen_kwargs["seed"] = sample_seed

                safe_name = self._safe_preview_slug(str(group.get("name") or f"sample{i}"))
                safe_mode = self._safe_preview_slug(mode)
                stem = f"step{step:05d}_{i:02d}_{safe_name}_{safe_mode}"
                with self._preview_lora_scale(lora_scale):
                    image = self._sampler.generate(**gen_kwargs)
                if image is None:
                    job = None
                    if hasattr(self._sampler, "consume_last_job_metadata"):
                        job = self._sampler.consume_last_job_metadata()
                    if job and hasattr(self._sampler, "write_manifest"):
                        if isinstance(job, dict):
                            job.setdefault("preview_group", group)
                            job.setdefault("preview_lora_scale", lora_scale)
                        manifest_path = output_dir / f"{stem}.cpu_preview.json"
                        self._sampler.write_manifest(manifest_path, job)
                        logger.info("CPU preview job scheduled: %s", manifest_path)
                    else:
                        logger.warning("Preview sampling returned no image for sample %d; skipping save.", i)
                    continue
                save_path = output_dir / f"{stem}.png"
                image.save(save_path)
            except (IOError, RuntimeError) as e:
                # AUDIT FIX: Catch I/O and Generation errors
                logger.error(f"Failed to generate/save sample {i}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected sampling error: {e}")

    def _on_epoch_end(self, epoch: int, info: Dict):
        """Epoch 结束回调"""
        if hasattr(self, '_dataset') and self._dataset and hasattr(self._dataset, 'set_current_epoch'):
            self._dataset.set_current_epoch(epoch)
        self._log(f"Epoch {epoch + 1} - Loss: {info.get('avg_loss', 0):.4f}")
        self._emit_runtime_event(
            {
                "event_type": "epoch",
                "epoch": int(epoch),
                "severity": "info",
                "summary": f"epoch {int(epoch) + 1} avg_loss={float(info.get('avg_loss', 0.0) or 0.0):.6f}",
                "data": {"avg_loss": float(info.get("avg_loss", 0.0) or 0.0)},
            }
        )
        try:
            self._write_epoch_log(epoch, float(info.get("avg_loss", 0.0) or 0.0))
        except Exception as exc:
            logger.debug("Epoch logging write failed: %s", exc)

        sample_every_n_epochs = max(int(getattr(self.config, "sample_every_n_epochs", 0) or 0), 0)
        if self._sampler and sample_every_n_epochs > 0 and (epoch + 1) % sample_every_n_epochs == 0:
            try:
                current_step = self.training_loop.global_step if self.training_loop else (epoch + 1)
                self._run_sampling(current_step, current_epoch=epoch + 1)
            except Exception as e:
                logger.warning(f"Epoch-end sampling failed: {e}")

        if self._coreset_manager:
            try:
                self._coreset_manager.on_epoch_end()
                every = max(int(getattr(self.config, "coreset_report_every_n_epochs", 1) or 1), 1)
                if bool(getattr(self.config, "coreset_report_enabled", True)) and (epoch + 1) % every == 0:
                    self._write_coreset_report(epoch=epoch)
            except Exception as e:
                logger.warning(f"Coreset epoch report failed: {e}")

    def _write_coreset_report(self, epoch: Optional[int] = None, final: bool = False) -> None:
        if not self._coreset_manager or not bool(getattr(self.config, "coreset_report_enabled", True)):
            return
        from .distributed import is_main_process
        if not is_main_process():
            return

        try:
            self._coreset_manager.classify_all()
            top_k = max(int(getattr(self.config, "coreset_report_top_k", 20) or 20), 1)
            output_dir = Path(str(getattr(self.config, "save_to", "") or getattr(self.config, "output_dir", "outputs")))
            output_dir.mkdir(parents=True, exist_ok=True)
            if final:
                filename = "coreset_report_final.json"
            elif epoch is not None:
                filename = f"coreset_report_epoch{int(epoch) + 1}.json"
            else:
                filename = "coreset_report.json"
            path = output_dir / filename
            report = self._coreset_manager.save_report(str(path), top_k=top_k, include_samples=True)
            summary = report.get("summary", {})
            cats = summary.get("categories", {})
            self._log(
                f"Coreset report saved: {path} "
                f"(total={summary.get('total', 0)}, easy={cats.get('easy', 0)}, "
                f"hard={cats.get('hard', 0)}, toxic={cats.get('toxic', 0)})"
            )
        except Exception as e:
            logger.warning(f"Coreset report write failed: {e}")

    # ------------------------------------------------------------------
    # R3 save/load methods moved to trainer_artifact_io.TrainerArtifactIoMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def stop(self):
        """停止训练"""
        self._should_stop = True
        if self.training_loop:
            self.training_loop.stop()
        self._log("Training stop requested")

    @property
    def progress(self) -> Dict:
        """获取当前进度"""
        if not self.training_loop:
            return {"step": 0, "epoch": 0, "loss": 0}

        return {
            "step": self.training_loop.global_step,
            "epoch": self.training_loop.current_epoch,
            "loss": self._last_loss,  # 使用记录的最后一次 loss
        }

    def _on_params_changed(self):
        """参数发生变化（如 SmartRank 剪枝），重建优化器"""
        self._log("SmartRank: Parameters changed, rebuilding optimizer...")
        if not self.training_loop:
            return

        # 1. 获取新参数
        trainable_params = self.lora_injector.get_trainable_params()

        # 2. 重建优化器
        optimizer = self._create_optimizer()

        # 3. 更新 loop 中的引用
        self.training_loop.optimizer = optimizer
        self.training_loop._pcgrad_param_names = self._optimizer_param_names()

        # 4. 重新绑定调度器 (尽可能保持当前步数)
        # 这里使用 steps_per_epoch 来重新计算总步数
        steps_per_epoch = int(getattr(self.training_loop, "steps_per_epoch", 0) or getattr(self, "_steps_per_epoch", 0) or 1000)
        total_steps = max(steps_per_epoch * int(self.config.epochs), 1)
        scheduler = self._create_scheduler(optimizer, total_steps)

        # 尝试恢复调度器的步数进度
        # 注意: 许多调度器在初始化后 step(current_step) 可能会有跳跃，这里采用简单替换
        self.training_loop.lr_scheduler = scheduler
        self.training_loop.total_steps = total_steps
        self._total_steps = total_steps
        if self._ema_requested():
            self._initialize_ema_tracker()

        self._log(f"Optimizer rebuilt with {len(trainable_params)} parameter groups.")
