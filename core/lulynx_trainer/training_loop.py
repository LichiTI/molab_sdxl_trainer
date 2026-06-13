"""
训练循环

核心训练逻辑，集成审计回调
"""

import torch
import torch.nn.functional as F
import logging
import copy
import random
from contextlib import ExitStack, nullcontext
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import numpy as np
from .model_family import ModelFamily, get_model_family
from .fixed_token_padding import FixedTokenPadder, resolve_fixed_token_length
from .module_offload import ModuleResidencyManager, build_module_offload_plan
from .module_offload_contract import (
    get_module_offload_conflict,
    is_swap_requested,
    resolve_module_offload_config,
)
from .device_state import capture_module_state, module_runtime_state
from .activation_compression import ActivationCompressionContext
from .step_phase_profile import StepPhaseProfiler
from .newbie_backward_op_profile import NewbieModuleTimingProfiler
from core.turbocore_native_update_dispatch_arming import TurboCoreNativeUpdateDispatchArmer
from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_native_update_probe_cache import (
    can_retain_native_update_probe_evidence,
)
from core.turbocore_kahan_adamw8bit_training_executor import build_kahan_adamw8bit_training_executor
from core.turbocore_native_update_training_executor import build_native_update_training_executor
from core.turbocore_simple_optimizer_training_executor import build_simple_optimizer_training_executor
from core.turbocore_v5_stream_lifetime_lease_evidence import build_single_step_lifetime_lease_request
from core.turbocore_update_gate import TurboCoreNativeUpdateGate, build_native_update_gate_config
from core.turbocore_update_shadow import TurboCoreUpdateShadow, build_update_shadow_config
from .turbocore_native_update_readiness_adapter import (
    build_native_update_runtime_context,
    build_training_loop_native_update_readiness,
)
from .pipeline_parallel_runtime_profile import build_pipeline_parallel_runtime_profile
from .offloaded_checkpoint_runtime_profile import build_offloaded_checkpoint_runtime_profile
from .multi_batch_promotion_gate import build_lulynx_multi_batch_promotion_gate
from .training_pipeline_trace import LulynxTrainingPipelineTrace, compact_lulynx_pipeline_trace
from .training_step_orchestrator_handlers import (
    run_lulynx_backward_execution_stage_handler,
    run_lulynx_backward_plan_stage_handler,
    run_lulynx_batch_contract_stage_handler,
    run_lulynx_epoch_finalization_stage_handler,
    run_lulynx_epoch_iteration_guard_stage_handler,
    run_lulynx_after_optimizer_hook_stage_handler,
    run_lulynx_before_optimizer_hook_stage_handler,
    run_lulynx_forward_execution_stage_handler,
    run_lulynx_forward_input_stage_handler,
    run_lulynx_layer_monitor_stage_handler,
    run_lulynx_loss_accounting_stage_handler,
    run_lulynx_loss_execution_stage_handler,
    run_lulynx_loss_plan_stage_handler,
    run_lulynx_loss_plugin_hook_stage_handler,
    run_lulynx_accumulation_group_tail_stage_handler,
    run_lulynx_microbatch_group_stage_handler,
    run_lulynx_noise_timestep_stage_handler,
    run_lulynx_optimizer_execution_stage_handler,
    run_lulynx_optimizer_finalize_stage_handler,
    run_lulynx_optimizer_step_route_stage_handler,
    run_lulynx_post_optimizer_maintenance_stage_handler,
    run_lulynx_safeguard_stage_handler,
    run_lulynx_post_optimizer_housekeeping_stage_handler,
    run_lulynx_telemetry_callback_stage_handler,
    run_lulynx_telemetry_execution_stage_handler,
    run_lulynx_telemetry_no_callback_maintenance_stage_handler,
    run_lulynx_telemetry_side_effects_stage_handler,
    run_lulynx_telemetry_step_info_stage_handler,
    run_lulynx_train_step_invocation_stage_handler,
    run_lulynx_turbocore_native_update_post_optimizer_stage_handler,
    run_lulynx_turbocore_native_update_pre_optimizer_stage_handler,
    run_lulynx_turbocore_shadow_compare_stage_handler,
    run_lulynx_turbocore_shadow_prepare_stage_handler,
    run_lulynx_turbocore_native_update_runtime_profile_stage_handler,
    run_lulynx_transfer_conditioning_stage_handler,
)
from .training_loop_turbocore_runtime import TrainingLoopTurbocoreRuntimeMixin
from .training_loop_runtime_memory import TrainingLoopRuntimeMemoryMixin
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is missing
    def tqdm(iterable, *args, **kwargs):
        return iterable
import time
from .optimizer_step_contracts import (
    optimizer_requires_create_graph_backward,
    optimizer_requires_step_closure,
    optimizer_step_closure_requires_initial_backward,
    optimizer_uses_fused_backward,
    run_optimizer_fused_backward,
)
from .pcgrad import resolve_pcgrad_gradients
from .b_tier_runtime import BTierRuntime

logger = logging.getLogger(__name__)


class _LossScalarCache:
    """Cache Python float conversions for loss tensors within one train step."""

    def __init__(self) -> None:
        self._values: Dict[int, float] = {}

    def get(self, tensor: torch.Tensor) -> float:
        key = id(tensor)
        cached = self._values.get(key)
        if cached is not None:
            return cached
        value = float(tensor.detach().float().item())
        self._values[key] = value
        return value


def _tqdm_kwargs() -> Dict[str, Any]:
    """Return tqdm kwargs safer for Windows console and web log capture."""
    return {
        "ascii": True,
        "dynamic_ncols": False,
    }


def _normalize_dropout_rate(value: Any) -> float:
    """Clamp user-facing probability fields to a safe [0, 1] range."""
    try:
        rate = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(rate, 1.0))


def _none_memory_optimization_state(reason: str = "", *, source: str = "runtime") -> Dict[str, Any]:
    return {
        "enabled": False,
        "mode": "none",
        "source": source,
        "reason": reason,
        "warnings": [],
    }


def _module_offload_conflict_details(
    *,
    request: Any,
    device: Any,
    swap_requested: bool,
    vram_swap_to_ram: bool,
    safe_fallback: bool,
    torch_compile: bool,
    gradient_checkpointing: bool,
    cpu_offload_checkpointing: bool,
    multi_gpu: bool,
    num_processes: int,
    num_machines: int,
    deepspeed: bool,
    training_type: str,
    easy_control: Any,
    ip_adapter: Any,
    easycontrol_v2_adapter: Any = None,
) -> Optional[tuple[str, str]]:
    if not request.requested:
        return None
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return get_module_offload_conflict("single_cuda_gpu_required")
    if multi_gpu or int(num_processes or 1) > 1 or int(num_machines or 1) > 1:
        return get_module_offload_conflict("distributed")
    if deepspeed:
        return get_module_offload_conflict("deepspeed")
    route = str(training_type or "").strip().lower().replace("_", "-")
    if easy_control is not None or ip_adapter is not None or easycontrol_v2_adapter is not None or any(
        token in route for token in ("controlnet", "ip-adapter", "lllite")
    ):
        return get_module_offload_conflict("pipeline")
    if swap_requested:
        return get_module_offload_conflict("swap")
    if vram_swap_to_ram:
        return get_module_offload_conflict("vram_swap_to_ram")
    if safe_fallback:
        return get_module_offload_conflict("safe_fallback")
    if torch_compile:
        return get_module_offload_conflict("torch_compile")
    if gradient_checkpointing:
        return get_module_offload_conflict("gradient_checkpointing")
    if cpu_offload_checkpointing:
        return get_module_offload_conflict("cpu_offload_checkpointing")
    return None


class TrainingLoop(TrainingLoopTurbocoreRuntimeMixin, TrainingLoopRuntimeMemoryMixin):
    """训练循环"""
    
    def __init__(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: Optional[torch.nn.Module],  # None for SD1.5
        vae: torch.nn.Module,
        tokenizer_1: Any,
        tokenizer_2: Optional[Any],  # None for SD1.5
        noise_scheduler: Any,
        lora_injector: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gradient_accumulation_steps: int = 4,
        gradient_accumulation_mode: str = "fast",
        thermal_throttler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        noise_offset: float = 0.0,
        snr_gamma: Optional[float] = None,
        loss_type: str = "l2",
        loss_precision: str = "fp32_loss",
        huber_c: float = 0.1,
        huber_schedule: str = "constant",
        huber_scale: float = 1.0,
        te_manager: Optional[Any] = None,
        model_family: Optional[ModelFamily] = None,
        model_arch: Optional[str] = None,
        train_text_encoder: bool = True,
        text_encoder_cpu_residency: bool = False,
        vae_cpu_residency: bool = False,
        # ── Advanced noise/loss options ──
        multires_noise_iterations: int = 0,
        multires_noise_discount: float = 0.3,
        adaptive_noise_scale: float = 0.0,
        ip_noise_gamma: float = 0.0,
        noise_offset_random_strength: bool = False,
        ip_noise_gamma_random_strength: bool = False,
        debiased_estimation: bool = False,
        zero_terminal_snr: bool = False,
        v_parameterization: bool = False,
        scale_v_pred_loss_like_noise_pred: bool = False,
        masked_loss: bool = False,
        alpha_mask: bool = False,
        strict_masked_loss: bool = False,
        # ── 显存优化 ──
        blocks_to_swap: int = 0,
        swap_granularity: str = "off",
        swap_ratio: float = 0.0,
        swap_count: int = 0,
        block_merge_size: int = 2,
        block_swap_strategy: str = "auto",
        module_offload_enabled: bool = False,
        module_offload_ratio: int = 0,
        module_offload_backbone_ratio: Optional[int] = None,
        module_offload_text_encoder_ratio: Optional[int] = None,
        module_offload_profile: str = "custom",
        module_offload_profile_enabled: bool = False,
        module_offload_min_param_mb: float = 0.0,
        module_offload_include_patterns: str = "",
        module_offload_exclude_patterns: str = "",
        module_offload_verify_state: bool = True,
        module_offload_prefetch_enabled: bool = False,
        module_offload_prefetch_mode: str = "experimental",
        module_offload_enhanced: bool = False,
        gradient_checkpointing: bool = False,
        vram_swap_to_ram: bool = False,
        torch_compile: bool = False,
        cpu_offload_checkpointing: bool = False,
        cpu_offload_checkpointing_mode: str = "standard",
        cpu_offload_checkpointing_pool_gb: float = 1.0,
        adapter_cpu_residency: Any = None,
        attention_profile: Any = None,
        data_transfer_non_blocking: bool = True,
        data_transfer_profile_enabled: bool = False,
        data_transfer_profile_mode: str = "event",
        data_transfer_profile_window: int = 50,
        step_phase_profile_enabled: bool = False,
        turbocore_update_shadow_mode: str = "off",
        turbocore_update_shadow_max_params: int = 0,
        turbocore_update_shadow_compare_interval: int = 1,
        turbocore_update_shadow_direct_grad: bool = False,
        turbocore_update_shadow_prefer_triton: bool = False,
        turbocore_update_shadow_compare_sample_params: int = 0,
        turbocore_update_shadow_stop_after_consecutive_passes: int = 0,
        turbocore_update_shadow_checkpoint_contract: bool = False,
        turbocore_update_shadow_copyback_probe: bool = False,
        turbocore_update_shadow_copyback_dispatch_experimental: bool = False,
        turbocore_update_shadow_native_binding_probe: bool = False,
        turbocore_update_shadow_owner_native_launch_probe: bool = False,
        turbocore_update_shadow_owner_native_launch_max_numel: int = 1048576,
        turbocore_update_shadow_owner_native_event_chain_probe: bool = False,
        turbocore_update_shadow_save_owner_state: bool = False,
        turbocore_native_update_mode: str = "off",
        turbocore_native_update_required_shadow_passes: int = 3,
        turbocore_native_update_max_abs_diff: float = 5e-5,
        turbocore_native_update_max_mean_abs_diff: float = 1e-6,
        turbocore_native_update_allow_missing_kernel: bool = False,
        turbocore_native_update_strict: bool = False,
        turbocore_native_update_dispatch_enabled: bool = False,
        turbocore_native_update_training_path_enabled: bool = False,
        turbocore_native_update_require_native_cuda: bool = False,
        turbocore_native_update_diagnostic_executor_replay: bool = False,
        turbocore_native_update_defer_state_sync: bool = False,
        turbocore_native_update_runtime_synchronization_policy: str = "context_synchronize",
        turbocore_native_update_simple_optimizer_kind: str = "",
        turbocore_native_update_quantized_optimizer_kind: str = "",
        # ── Newbie safe fallback ──
        safe_fallback: bool = False,
        anima_timestep_sampling: str = "sigma",
        anima_sigmoid_scale: float = 1.0,
        anima_discrete_flow_shift: float = 1.0,
        anima_weighting_scheme: str = "none",
        anima_model_prediction_type: str = "velocity",
        anima_mode_scale: float = 1.0,
        anima_faithful_forward: bool = False,
        # ── JLT EMA feature self-distillation alignment (default-off) ──
        anima_ema_feat_align_enabled: bool = False,
        anima_ema_feat_align_weight: float = 0.0,
        anima_ema_feat_align_teacher_layers: str = "",
        anima_ema_feat_align_student_layers: str = "",
        anima_ema_feat_align_decay: float = 0.9999,
        # ── SDXL Flow Matching ──
        flow_model: str = "",
        sdxl_timestep_sampling: str = "uniform",
        sdxl_sigmoid_scale: float = 1.0,
        sdxl_flow_shift: float = 1.0,
        sdxl_flow_weighting_scheme: str = "none",
        sdxl_model_prediction_type: str = "epsilon",
        flow_logit_mean: float = 0.0,
        flow_logit_std: float = 1.0,
        flow_uniform_shift: bool = False,
        flow_uniform_base_pixels: int = 256,
        flow_uniform_static_ratio: float = 0.0,
        cfm_lambda: float = 1.0,
        flow_use_ot: bool = False,
        immiscible_diffusion_enabled: bool = False,
        immiscible_metric: str = "l2",
        # ── Text Encoder Dropout ──
        te_dropout: float = 0.0,
        clip_l_dropout_rate: float = 0.0,
        clip_g_dropout_rate: float = 0.0,
        t5_dropout_rate: float = 0.0,
        # ── Wavelet Loss ──
        wavelet_loss_enabled: bool = False,
        wavelet_loss_levels: int = 2,
        wavelet_loss_high_freq_weight: float = 2.0,
        wavelet_loss_approx_weight: float = 0.0,
        wavelet_loss_base_loss: str = "l2",
        # ── Qwen3 secondary encoder (Anima) ──
        qwen3_encoder: Optional[torch.nn.Module] = None,
        qwen3_tokenizer: Optional[Any] = None,
        # ── Fixed token padding for torch.compile ──
        max_token_length: int = 0,  # 0 = use tokenizer default (dynamic)
        enable_fixed_token_padding: bool = False,
        easy_control: Optional[torch.nn.Module] = None,
        ip_adapter: Optional[torch.nn.Module] = None,
        easycontrol_v2_adapter: Optional[torch.nn.Module] = None,
        repa_enabled: bool = False,
        repa_target_modules: str = "",
        repa_loss_type: str = "cosine",
        repa_loss_weight: float = 0.0,
        repa_projection_dim: int = 0,
        repa_stop_grad_target: bool = True,
        repa_projector: Optional[torch.nn.Module] = None,
        lulynx_geometric_lock: bool = False,
        lulynx_manifold_weight: float = 0.01,
        lulynx_proj_dim: int = 128,
        lulynx_manifold_sparse_freq: int = 1,
        lulynx_anchor_layers: str = "",
        lulynx_ghost_replay: bool = False,
        lulynx_ghost_path: str = "",
        lulynx_ghost_interval: int = 100,
        lulynx_ghost_weight: float = 0.05,
        softrepa_enabled: bool = False,
        softrepa_schedule: str = "linear",
        softrepa_min_weight: float = 0.0,
        softrepa_max_weight: float = 1.0,
        softrepa_sigma_min: float = 0.0,
        softrepa_sigma_max: float = 1.0,
        sra2_haste_enabled: bool = False,
        sra2_haste_capture_layers: str = "",
        sra2_haste_policy: Optional[dict] = None,
        dit_compute_reducer_strategy: str = "none",
        dit_compute_reducer_keep_ratio: float = 1.0,
        dit_compute_reducer_min_keep_tokens: int = 1,
        dit_compute_reducer_compression_ratio: float = 1.0,
        dit_compute_reducer_min_tokens: int = 1,
        dit_compute_reducer_skip_ratio: float = 0.0,
        dit_compute_reducer_skip_every: int = 0,
        dit_compute_reducer_warmup_steps: int = 0,
        dit_compute_reducer_min_block: int = 0,
        dit_compute_reducer_score_mode: str = "l2",
        multi_gpu: bool = False,
        num_processes: int = 1,
        num_machines: int = 1,
        training_type: str = "",
        deepspeed: bool = False,
        # ── Port 5 进阶显存优化 ──
        gradient_release_enabled: bool = False,
        gradient_release_mode: str = "post_step",
        activation_compression_enabled: bool = False,
        activation_compression_dtype: str = "fp16",
        activation_compression_min_tensor_mb: float = 1.0,
        activation_cpu_offload_enabled: bool = False,
        activation_cpu_offload_min_tensor_mb: float = 1.0,
        activation_cpu_offload_pool_gb: float = 1.0,
        resolution_aware_batch_enabled: bool = False,
        resolution_aware_batch_base_resolution: int = 1024,
        resolution_aware_batch_max_factor: float = 4.0,
        resolution_aware_batch_min_factor: float = 0.25,
        pipeline_parallel_enabled: bool = False,
        pipeline_parallel_chunks: int = 2,
        pipeline_parallel_split_points: str = "",
        # ── Feature batch: 9 Warehouse additions ──
        ddpm_timestep_sampling: str = "",
        stochastic_grad_accumulation: bool = False,
        spectral_noise_blend: float = 0.0,
        spectral_noise_sigma: float = 4.0,
        huber_auto_percentile: float = 0.9,
        adaptive_loss_weighter: Optional[torch.nn.Module] = None,
        flow_uncertainty_weighter: Optional[torch.nn.Module] = None,
        faster_dit_snr_weighter: Optional[torch.nn.Module] = None,
        sageattn_drift_check_interval: int = 0,
        sageattn_drift_threshold: float = 0.01,
        sageattn_drift_fallback: str = "warn",
        stepped_loss_enabled: bool = False,
        stepped_loss_schedule: str = "",
        pattern_loss_enabled: bool = False,
        pattern_loss_levels: int = 1,
        pattern_loss_ll_type: str = "l2",
        pattern_loss_ll_weight: float = 1.0,
        pattern_loss_high_type: str = "huber",
        pattern_loss_high_weight: float = 2.0,
        pattern_loss_high_huber_c: float = 0.1,
        perlin_noise_offset_enabled: bool = False,
        perlin_noise_offset_strength: float = 0.1,
        perlin_noise_offset_scale: float = 4.0,
        optimal_noise_enabled: bool = False,
        optimal_noise_candidates: int = 4,
        dop: Optional[object] = None,
        concept_direction: Optional[object] = None,
        # ── 高级训练监控 ──
        advanced_monitoring_enabled: bool = False,
        peak_vram_diagnostics_interval: int = 25,
        cuda_cache_release_strategy: str = "oom_only",
        cuda_cache_release_interval: int = 1,
        audit_mode_override: str = "",
        attn_entropy_interval: int = 100,
        act_drift_interval: int = 100,
        act_drift_anchor_layers: str = "",
        # ── 深度诊断 ──
        deep_diagnostics_enabled: bool = False,
        hessian_trace_interval: int = 200,
        grad_cosine_enabled: bool = False,
        # ── 遗忘探针 ──
        forgetting_probe_interval: int = 50,
        # ── 流形追踪 ──
        manifold_snapshot_interval: int = 20,
        precision_swap_profile: Optional[Dict[str, Any]] = None,
        te_vae_offload_strategy: str = "phase",
        layer_monitor_enabled: bool = True,
        layer_monitor_interval: int = 3,
        layer_monitor_max_layers: int = 10,
        layer_monitor_sparsity_epsilon: float = 1e-8,
        layer_monitor_mode: str = "sampled",
        layer_monitor_sample_size: int = 4096,
        vram_smart_sensing_enabled: bool = True,
        vram_smart_sensing_baseline_steps: int = 50,
        vram_smart_sensing_slowdown_ratio: float = 1.5,
        vram_smart_sensing_window_steps: int = 5,
    ):
        self.unet = unet
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.noise_scheduler = noise_scheduler
        self.lora_injector = lora_injector
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_type = str(training_type or "")
        self.multi_gpu = bool(multi_gpu)
        self.num_processes = int(num_processes or 1)
        self.num_machines = int(num_machines or 1)
        self.deepspeed = bool(deepspeed)
        self.device = device
        self._runtime_device = torch.device(device)
        self.dtype = dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_mode = self._normalize_gradient_accumulation_mode(gradient_accumulation_mode)
        # Step-boundary GPU duty-cycle/temperature governor (None = off).
        self._thermal_throttler = thermal_throttler
        self._last_thermal_throttle_report: Dict[str, Any] = {}
        self.max_grad_norm = max_grad_norm
        self.noise_offset = noise_offset
        self.snr_gamma = snr_gamma
        self.loss_type = loss_type
        self.loss_precision = self._normalize_loss_precision(loss_precision)
        self.huber_c = huber_c
        self.huber_schedule = huber_schedule
        self.huber_scale = huber_scale
        self.te_manager = te_manager
        self.cpu_offload_checkpointing = cpu_offload_checkpointing
        self.cpu_offload_checkpointing_mode = cpu_offload_checkpointing_mode
        self.cpu_offload_checkpointing_pool_gb = cpu_offload_checkpointing_pool_gb
        self.adapter_cpu_residency = adapter_cpu_residency
        self.attention_profile = attention_profile
        self.data_transfer_non_blocking = bool(data_transfer_non_blocking)
        self.data_transfer_profile_enabled = bool(data_transfer_profile_enabled)
        self.data_transfer_profile_mode = self._normalize_data_transfer_profile_mode(data_transfer_profile_mode)
        self.data_transfer_profile_window = max(int(data_transfer_profile_window or 50), 1)
        self._transfer_profile_steps = 0
        self._transfer_profile_step_seconds = 0.0
        self._transfer_profile_seconds = 0.0
        self._transfer_profile_bytes = 0
        self._transfer_profile_ops = 0
        self._transfer_profile_by_label: Dict[str, Dict[str, float]] = {}
        self._transfer_profile_pending_events: List[Dict[str, Any]] = []
        self._last_transfer_profile_snapshot: Optional[Dict[str, Any]] = None
        self._step_timing_history: List[Dict[str, Any]] = []
        self._step_timing_max_history = 128
        self._step_timing_steady_warmup_steps = 1
        self._last_step_timing_window: Dict[str, Any] = {}
        self._step_phase_profiler = StepPhaseProfiler(enabled=bool(step_phase_profile_enabled))
        self._pipeline_trace = LulynxTrainingPipelineTrace()
        self._last_pipeline_trace: Dict[str, Any] = {}
        self._training_data_pipeline_runtime_report: Dict[str, Any] = {}
        self._training_step_orchestrator_runtime_profile: Dict[str, Any] = {}
        self._turbocore_update_shadow = TurboCoreUpdateShadow(
            build_update_shadow_config(
                turbocore_update_shadow_mode,
                max_params=turbocore_update_shadow_max_params,
                compare_interval=turbocore_update_shadow_compare_interval,
                direct_grad=turbocore_update_shadow_direct_grad,
                prefer_triton=turbocore_update_shadow_prefer_triton,
                compare_sample_params=turbocore_update_shadow_compare_sample_params,
                stop_after_consecutive_passes=turbocore_update_shadow_stop_after_consecutive_passes,
                checkpoint_contract=turbocore_update_shadow_checkpoint_contract,
                copyback_probe=turbocore_update_shadow_copyback_probe,
                copyback_dispatch_experimental=turbocore_update_shadow_copyback_dispatch_experimental,
                native_binding_probe=turbocore_update_shadow_native_binding_probe,
                owner_native_launch_probe=turbocore_update_shadow_owner_native_launch_probe,
                owner_native_launch_max_numel=turbocore_update_shadow_owner_native_launch_max_numel,
                owner_native_event_chain_probe=turbocore_update_shadow_owner_native_event_chain_probe,
            )
        )
        self._turbocore_update_shadow_save_owner_state = bool(turbocore_update_shadow_save_owner_state)
        self._turbocore_native_update_gate = TurboCoreNativeUpdateGate(
            build_native_update_gate_config(
                turbocore_native_update_mode,
                required_shadow_passes=turbocore_native_update_required_shadow_passes,
                max_abs_diff=turbocore_native_update_max_abs_diff,
                max_mean_abs_diff=turbocore_native_update_max_mean_abs_diff,
                allow_missing_native_kernel=turbocore_native_update_allow_missing_kernel,
                strict=turbocore_native_update_strict,
                dispatch_enabled=turbocore_native_update_dispatch_enabled,
            )
        )
        self._turbocore_native_update_readiness: Dict[str, Any] = {}
        self._turbocore_native_update_dispatch_armer = TurboCoreNativeUpdateDispatchArmer()
        self._turbocore_native_update_dispatch_runtime = TurboCoreNativeUpdateDispatchRuntime()
        self._turbocore_native_update_training_executor: Optional[Any] = None
        self._turbocore_native_update_runtime_profile: Dict[str, Any] = {}
        self._turbocore_native_update_diagnostic_executor_replay = bool(
            turbocore_native_update_diagnostic_executor_replay
        )
        self._turbocore_native_update_training_path_enabled = bool(turbocore_native_update_training_path_enabled)
        self._turbocore_native_update_require_native_cuda = bool(turbocore_native_update_require_native_cuda)
        self._turbocore_native_update_defer_state_sync = bool(turbocore_native_update_defer_state_sync)
        self._turbocore_native_update_runtime_synchronization_policy = str(
            turbocore_native_update_runtime_synchronization_policy or "context_synchronize"
        )
        self._turbocore_native_update_simple_optimizer_kind = self._normalize_turbocore_simple_optimizer_kind(
            turbocore_native_update_simple_optimizer_kind
        )
        self._turbocore_native_update_quantized_optimizer_kind = self._normalize_turbocore_quantized_optimizer_kind(
            turbocore_native_update_quantized_optimizer_kind
        )
        self.vram_smart_sensing_enabled = bool(vram_smart_sensing_enabled)
        self.vram_smart_sensing_baseline_steps = max(int(vram_smart_sensing_baseline_steps or 50), 1)
        self.vram_smart_sensing_slowdown_ratio = max(float(vram_smart_sensing_slowdown_ratio or 1.5), 1.05)
        self.vram_smart_sensing_window_steps = max(int(vram_smart_sensing_window_steps or 5), 1)
        self._smart_sensing_observed_steps = 0
        self._smart_sensing_baseline_total_seconds = 0.0
        self._smart_sensing_recent_seconds: List[float] = []
        self._last_vram_smart_sensing_report: Dict[str, Any] = {}
        self.safe_fallback = safe_fallback
        self.anima_timestep_sampling = anima_timestep_sampling
        self.anima_sigmoid_scale = anima_sigmoid_scale
        self.anima_discrete_flow_shift = anima_discrete_flow_shift
        self.anima_weighting_scheme = anima_weighting_scheme
        self.anima_model_prediction_type = anima_model_prediction_type
        self.anima_mode_scale = anima_mode_scale
        # Opt-in faithful native forward (#147): t=sigma in [0,1] + frozen
        # llm_adapter cross-attn context + 3D-RoPE forward. Default False keeps
        # the legacy (#132) path bitwise-unchanged.
        self.anima_faithful_forward = bool(anima_faithful_forward)
        # JLT EMA feature self-distillation alignment (default-off reserve).
        # Shadow + captured student features are created lazily on the first
        # enabled step (see forward/loss handlers), so the default path is
        # untouched. See anima_ema_feature_align.py.
        self.anima_ema_feat_align_enabled = bool(anima_ema_feat_align_enabled)
        self.anima_ema_feat_align_weight = float(anima_ema_feat_align_weight or 0.0)
        self.anima_ema_feat_align_teacher_layers = str(anima_ema_feat_align_teacher_layers or "")
        self.anima_ema_feat_align_student_layers = str(anima_ema_feat_align_student_layers or "")
        self.anima_ema_feat_align_decay = float(anima_ema_feat_align_decay or 0.9999)
        self._ema_lora_shadow = None
        self._ema_feat_student_features = None
        # SDXL Flow Matching
        self.flow_model = flow_model
        self.sdxl_timestep_sampling = sdxl_timestep_sampling
        self.sdxl_sigmoid_scale = sdxl_sigmoid_scale
        self.sdxl_flow_shift = sdxl_flow_shift
        self.sdxl_flow_weighting_scheme = sdxl_flow_weighting_scheme
        self.sdxl_model_prediction_type = sdxl_model_prediction_type
        self.flow_logit_mean = flow_logit_mean
        self.flow_logit_std = flow_logit_std
        self.flow_uniform_shift = flow_uniform_shift
        self.flow_uniform_base_pixels = flow_uniform_base_pixels
        self.flow_uniform_static_ratio = flow_uniform_static_ratio
        self.cfm_lambda = cfm_lambda
        self.flow_use_ot = flow_use_ot
        self.immiscible_diffusion_enabled = immiscible_diffusion_enabled
        self.immiscible_metric = immiscible_metric
        self.te_dropout = _normalize_dropout_rate(te_dropout)
        self.clip_l_dropout_rate = _normalize_dropout_rate(clip_l_dropout_rate)
        self.clip_g_dropout_rate = _normalize_dropout_rate(clip_g_dropout_rate)
        self.t5_dropout_rate = _normalize_dropout_rate(t5_dropout_rate)
        self.wavelet_loss_enabled = wavelet_loss_enabled
        self.wavelet_loss_levels = wavelet_loss_levels
        self.wavelet_loss_high_freq_weight = wavelet_loss_high_freq_weight
        self.wavelet_loss_approx_weight = wavelet_loss_approx_weight
        self.wavelet_loss_base_loss = wavelet_loss_base_loss
        self._sdxl_flow_sigmas = None
        self._sdxl_flow_weighting = "none"
        # Advanced noise/loss
        self.multires_noise_iterations = multires_noise_iterations
        self.multires_noise_discount = multires_noise_discount
        self.adaptive_noise_scale = adaptive_noise_scale
        self.ip_noise_gamma = ip_noise_gamma
        self.noise_offset_random_strength = noise_offset_random_strength
        self.ip_noise_gamma_random_strength = ip_noise_gamma_random_strength
        self.debiased_estimation = debiased_estimation
        self.zero_terminal_snr = zero_terminal_snr
        self.v_parameterization = v_parameterization
        self.scale_v_pred_loss_like_noise_pred = scale_v_pred_loss_like_noise_pred
        self.masked_loss = masked_loss
        self.alpha_mask = alpha_mask
        self.strict_masked_loss = strict_masked_loss
        self._masked_loss_warned = False
        # Feature batch: 9 Warehouse additions
        self.ddpm_timestep_sampling = str(ddpm_timestep_sampling or "")
        self.stochastic_grad_accumulation = bool(stochastic_grad_accumulation)
        self.spectral_noise_blend = float(spectral_noise_blend)
        self.spectral_noise_sigma = float(spectral_noise_sigma)
        self.huber_auto_percentile = float(huber_auto_percentile)
        self.adaptive_loss_weighter = adaptive_loss_weighter
        self.flow_uncertainty_weighter = flow_uncertainty_weighter
        self.faster_dit_snr_weighter = faster_dit_snr_weighter
        self._drift_monitor = None
        if int(sageattn_drift_check_interval or 0) > 0:
            from .attention_drift_monitor import AttentionDriftMonitor
            self._drift_monitor = AttentionDriftMonitor(
                threshold=float(sageattn_drift_threshold),
                fallback=str(sageattn_drift_fallback or "warn"),
            )
            self._drift_check_interval = int(sageattn_drift_check_interval)

        # Feature batch 2: 6 Warehouse additions
        self._stepped_loss_schedule = None
        if bool(stepped_loss_enabled) and stepped_loss_schedule:
            from .stepped_loss import SteppedLossSchedule
            self._stepped_loss_schedule = SteppedLossSchedule(stepped_loss_schedule)
            if not self._stepped_loss_schedule.enabled:
                self._stepped_loss_schedule = None
        self.pattern_loss_enabled = bool(pattern_loss_enabled)
        self.pattern_loss_levels = int(pattern_loss_levels)
        self.pattern_loss_ll_type = str(pattern_loss_ll_type or "l2")
        self.pattern_loss_ll_weight = float(pattern_loss_ll_weight)
        self.pattern_loss_high_type = str(pattern_loss_high_type or "huber")
        self.pattern_loss_high_weight = float(pattern_loss_high_weight)
        self.pattern_loss_high_huber_c = float(pattern_loss_high_huber_c)
        self.perlin_noise_offset_enabled = bool(perlin_noise_offset_enabled)
        self.perlin_noise_offset_strength = float(perlin_noise_offset_strength)
        self.perlin_noise_offset_scale = float(perlin_noise_offset_scale)
        self.optimal_noise_enabled = bool(optimal_noise_enabled)
        self.optimal_noise_candidates = int(optimal_noise_candidates)
        self.dop = dop
        self.concept_direction = concept_direction

        # 高级训练监控
        self._advanced_monitoring = bool(advanced_monitoring_enabled)
        self._peak_vram_diag_interval = max(int(peak_vram_diagnostics_interval or 25), 1)
        self._cuda_cache_release_strategy = self._normalize_cuda_cache_release_strategy(cuda_cache_release_strategy)
        self._cuda_cache_release_interval = max(int(cuda_cache_release_interval or 1), 1)
        self._last_cuda_cache_release: Dict[str, Any] = {}
        self._cuda_cache_release_seen_steps: Dict[str, int] = {}
        self._audit_mode_override = str(audit_mode_override or "")
        self._attn_entropy_interval = max(int(attn_entropy_interval or 100), 1)
        self._act_drift_interval = max(int(act_drift_interval or 100), 1)
        self._act_drift_anchor_layers = str(act_drift_anchor_layers or "")
        self._loss_tracker = None
        self._act_drift_tracker = None
        self._grad_tracker = None
        self._deep_diagnostics = bool(deep_diagnostics_enabled)
        self._hessian_interval = max(int(hessian_trace_interval or 200), 1)
        self._hessian_estimator = None
        if self._advanced_monitoring:
            from .loss_tracker import LossTracker
            self._loss_tracker = LossTracker()
            self._loss_tracker.enable()
            from .grad_tracker import GradientCovarianceTracker
            self._grad_tracker = GradientCovarianceTracker()
            if bool(grad_cosine_enabled):
                self._grad_tracker.enable_cosine()
        if self._deep_diagnostics:
            from .hessian_trace import HessianTraceEstimator
            self._hessian_estimator = HessianTraceEstimator()
        self._forgetting_probe = None
        self._forgetting_probe_interval = max(int(forgetting_probe_interval or 50), 1)
        self._manifold_tracker = None
        self._manifold_snapshot_interval = max(int(manifold_snapshot_interval or 20), 1)
        self._layer_monitor_enabled = bool(layer_monitor_enabled)
        self._layer_monitor_interval = max(int(layer_monitor_interval or 3), 1)
        self._layer_monitor_max_layers = max(int(layer_monitor_max_layers or 10), 0)
        self._layer_monitor_sparsity_epsilon = float(layer_monitor_sparsity_epsilon or 1e-8)
        self._layer_monitor_mode = str(layer_monitor_mode or "sampled").lower()
        self._layer_monitor_sample_size = max(int(layer_monitor_sample_size or 4096), 128)

        # Apply zero terminal SNR modification to scheduler if enabled
        if self.zero_terminal_snr:
            from ..training_components.noise_utils import rescale_zero_terminal_snr
            rescale_zero_terminal_snr(self.noise_scheduler)

        # Qwen3 secondary encoder (Anima DiT)
        self.qwen3_encoder = qwen3_encoder
        self.qwen3_tokenizer = qwen3_tokenizer
        self.easy_control = easy_control
        self.ip_adapter = ip_adapter
        self.easycontrol_v2_adapter = easycontrol_v2_adapter
        self.repa_enabled = bool(repa_enabled) and float(repa_loss_weight or 0.0) > 0.0
        self.repa_loss_type = str(repa_loss_type or "cosine")
        self.repa_loss_weight = float(repa_loss_weight or 0.0)
        self.repa_projection_dim = int(repa_projection_dim or 0)
        self.repa_stop_grad_target = bool(repa_stop_grad_target)
        self.repa_projector = repa_projector
        self.softrepa_enabled = bool(softrepa_enabled)
        self.softrepa_schedule = str(softrepa_schedule or "linear")
        self.softrepa_min_weight = float(softrepa_min_weight or 0.0)
        self.softrepa_max_weight = float(softrepa_max_weight or 0.0)
        self.softrepa_sigma_min = float(softrepa_sigma_min or 0.0)
        self.softrepa_sigma_max = float(softrepa_sigma_max or 1.0)
        if self.softrepa_enabled:
            self.repa_enabled = True
        self.repa_capture = None
        self._repa_warned = False
        self.b_tier_runtime = None
        self._b_tier_last_state: Dict[str, Any] = {}
        if self.repa_enabled:
            targets = [part.strip() for part in str(repa_target_modules or "").replace(";", ",").replace("\n", ",").split(",") if part.strip()]
            if targets:
                from .repa import REPAFeatureCapture
                self.repa_capture = REPAFeatureCapture(self.unet, targets).install()
        # SRA2 + HASTE alignment auxiliary loss (default-off). Reuses the generic
        # REPA forward-hook capture; disabled -> no hooks, no loss, full parity.
        self.sra2_haste_enabled = bool(sra2_haste_enabled)
        self.sra2_haste_policy = dict(sra2_haste_policy or {})
        self.sra2_haste_capture = None
        self._sra2_haste_warned = False
        if self.sra2_haste_enabled:
            sra2_targets = [part.strip() for part in str(sra2_haste_capture_layers or "").replace(";", ",").replace("\n", ",").split(",") if part.strip()]
            if sra2_targets:
                from .repa import REPAFeatureCapture
                self.sra2_haste_capture = REPAFeatureCapture(self.unet, sra2_targets).install()

        # DiT block compute reducer (default-off, strategy-selectable). The seam
        # is built lazily at the first forward (blockskip needs the live block
        # count) and published only while a non-"none" strategy is active, so a
        # default run never enters the context -> bitwise-identical forward.
        self.dit_compute_reducer_strategy = str(dit_compute_reducer_strategy or "none").strip().lower()
        self.dit_compute_reducer_keep_ratio = float(dit_compute_reducer_keep_ratio)
        self.dit_compute_reducer_min_keep_tokens = int(dit_compute_reducer_min_keep_tokens)
        self.dit_compute_reducer_compression_ratio = float(dit_compute_reducer_compression_ratio)
        self.dit_compute_reducer_min_tokens = int(dit_compute_reducer_min_tokens)
        self.dit_compute_reducer_skip_ratio = float(dit_compute_reducer_skip_ratio)
        self.dit_compute_reducer_skip_every = int(dit_compute_reducer_skip_every)
        self.dit_compute_reducer_warmup_steps = int(dit_compute_reducer_warmup_steps)
        self.dit_compute_reducer_min_block = int(dit_compute_reducer_min_block)
        self.dit_compute_reducer_score_mode = str(dit_compute_reducer_score_mode or "l2")
        self._compute_reducer_seam = None
        self._compute_reducer_seam_built = False
        if bool(lulynx_geometric_lock) or bool(lulynx_ghost_replay):
            self.b_tier_runtime = BTierRuntime(
                self.unet,
                device=self._runtime_device,
                model_arch=str(model_arch or ""),
                manifold_enabled=bool(lulynx_geometric_lock),
                manifold_weight=float(lulynx_manifold_weight or 0.0),
                proj_dim=int(lulynx_proj_dim or 128),
                sparse_freq=int(lulynx_manifold_sparse_freq or 1),
                anchor_layers=str(lulynx_anchor_layers or ""),
                ghost_enabled=bool(lulynx_ghost_replay),
                ghost_path=str(lulynx_ghost_path or ""),
                ghost_interval=int(lulynx_ghost_interval or 100),
                ghost_weight=float(lulynx_ghost_weight or 0.0),
            )

        # Fixed token padding for torch.compile static shape contract
        self.max_token_length = max_token_length
        self.enable_fixed_token_padding = enable_fixed_token_padding
        self._token_padder_1: Optional[FixedTokenPadder] = None
        self._token_padder_2: Optional[FixedTokenPadder] = None

        if self.enable_fixed_token_padding:
            fixed_len = resolve_fixed_token_length(
                tokenizer_1,
                requested_length=max_token_length,
                text_encoder=text_encoder_1,
                default_length=77,
            )
            self._token_padder_1 = FixedTokenPadder(tokenizer_1, fixed_length=fixed_len)
            if tokenizer_2 is not None:
                fixed_len_2 = resolve_fixed_token_length(
                    tokenizer_2,
                    requested_length=max_token_length,
                    text_encoder=text_encoder_2,
                    default_length=77,
                )
                self._token_padder_2 = FixedTokenPadder(tokenizer_2, fixed_length=fixed_len_2)
            logger.info(f"Fixed token padding enabled: token_lengths={fixed_len}/{getattr(self._token_padder_2, 'fixed_length', 0) or '-'}")

        # Capability registry – drives dual-encoder / time-id paths
        # instead of relying on component presence heuristics.
        self._model_arch = model_arch or "sdxl"
        if model_family is not None:
            self._family = model_family
        else:
            self._family = get_model_family(model_arch)
        self._train_text_encoder_1 = bool(train_text_encoder and self.text_encoder_1 is not None)
        self._train_text_encoder_2 = bool(train_text_encoder and self.text_encoder_2 is not None)
        self._train_text_encoder_any = self._train_text_encoder_1 or self._train_text_encoder_2
        self._text_encoder_cpu_residency = bool(text_encoder_cpu_residency) and not self._train_text_encoder_any
        self._vae_cpu_residency = bool(vae_cpu_residency)
        self._te_vae_offload_strategy = str(te_vae_offload_strategy or "phase").strip().lower()
        self._aggressive_component_residency = self._te_vae_offload_strategy == "aggressive"
        self._module_offload_verify_state = bool(module_offload_verify_state) or self._aggressive_component_residency
        self._ensure_cpu_resident_components("init")
        
        # 状态
        self.global_step = 0
        self.current_epoch = 0
        self._current_micro_batch_index = 1
        self._current_micro_batch_count = 1
        self._current_sync_gradients = True
        self._current_accumulation_group_start = True
        self._turbocore_direct_grad_lifecycle_report: Dict[str, Any] = {}
        self._turbocore_native_update_direct_grad_lifecycle_report: Dict[str, Any] = {}
        self._should_stop = False
        self.pcgrad_enabled = False
        self.pcgrad_conflict_threshold = 0.0
        self.pcgrad_reduction = "mean"
        self._pcgrad_param_names: Dict[int, str] = {}
        self._pcgrad_pending_grads: List[Dict[str, torch.Tensor]] = []
        self._pcgrad_last_stats: Dict[str, Any] = {}
        # Set by Trainer (optimizer-step based, respects grad accumulation)
        self.steps_per_epoch: int = 0
        self.total_steps: int = 0
        self.completed_by_step_limit: bool = False
        self.initial_step_target: int = 0
        self.skip_until_initial_step: bool = False
        self.on_before_train_step: Optional[Callable[[int], None]] = None
        self.on_before_optimizer_step: Optional[Callable[[int], None]] = None

        # Validation dataloader (set externally by Trainer when validation_split > 0)
        self.validation_dataloader: Optional[Any] = None
        self.eval_every_n_steps: int = 0
        self.max_validation_steps: int = 0
        
        # 回调
        self.on_step_end: Optional[Callable[[int, float, Dict], None]] = None
        self.on_epoch_end: Optional[Callable[[int, Dict], None]] = None
        self.on_save_model: Optional[Callable[[int, str], None]] = None
        self.on_params_changed: Optional[Callable[[], None]] = None
        self.on_runtime_event: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # 审计回调 (集成 LoRAAuditor)
        # 审计回调 (集成 LoRAAuditor)
        self.auditor = None
        self.auditor_interval = 50
        
        # 安全卫士
        self.safeguard = None

        # Prior Preservation (set externally by trainer)
        self.prior_loss_weight: float = 0.0
        self.reg_dataloader = None
        self._reg_iter = None
        
        # LISA
        self.lisa_scheduler = None
        
        # Lulynx Wrapper (MN-LoRA)
        self.lulynx_wrapper = None

        # torch.compile active flag — set externally by the Trainer when
        # torch.compile is applied to the UNet.  Used here and in
        # _train_step_impl for compatibility guards.
        self._torch_compile_active: bool = False

        # CUDAGraph capture — used when anima_compile_scope="full_cudagraph"
        # and fixed token counts are set.  After warmup, each training step
        # replays the captured graph instead of running a full forward pass.
        self._cudagraph_capture = None
        self._cudagraph_active = False

        # 显存交换优化
        self._block_offloader = None
        self._module_offload_manager: Optional[ModuleResidencyManager] = None
        self.memory_optimization_state: Dict[str, Any] = _none_memory_optimization_state()
        self._precision_swap_profile = dict(precision_swap_profile or {})
        self._runtime_observation_steps = 0
        self._runtime_observation_total_step_seconds = 0.0
        self._module_offload_disabled_reason: str = ""
        swap_requested = is_swap_requested(
            {
                "swap_granularity": swap_granularity,
                "swap_ratio": swap_ratio,
                "swap_count": swap_count,
                "block_swap_strategy": block_swap_strategy,
                "blocks_to_swap": blocks_to_swap,
            }
        )
        module_offload_request = resolve_module_offload_config(
            {
                "module_offload_enabled": module_offload_enabled,
                "module_offload_ratio": module_offload_ratio,
                "module_offload_backbone_ratio": module_offload_backbone_ratio,
                "module_offload_text_encoder_ratio": module_offload_text_encoder_ratio,
                "module_offload_profile": module_offload_profile,
                "module_offload_profile_enabled": module_offload_profile_enabled,
                "module_offload_min_param_mb": module_offload_min_param_mb,
                "module_offload_include_patterns": module_offload_include_patterns,
                "module_offload_exclude_patterns": module_offload_exclude_patterns,
                "module_offload_verify_state": module_offload_verify_state,
                "module_offload_prefetch_enabled": module_offload_prefetch_enabled,
                "module_offload_prefetch_mode": module_offload_prefetch_mode,
                "module_offload_enhanced": module_offload_enhanced,
            }
        )
        module_offload_conflict = _module_offload_conflict_details(
            request=module_offload_request,
            device=device,
            swap_requested=swap_requested,
            vram_swap_to_ram=vram_swap_to_ram,
            safe_fallback=safe_fallback,
            torch_compile=torch_compile,
            gradient_checkpointing=gradient_checkpointing,
            cpu_offload_checkpointing=cpu_offload_checkpointing,
            multi_gpu=multi_gpu,
            num_processes=num_processes,
            num_machines=num_machines,
            deepspeed=deepspeed,
            training_type=training_type,
            easy_control=easy_control,
            ip_adapter=ip_adapter,
            easycontrol_v2_adapter=self.easycontrol_v2_adapter,
        )
        if module_offload_conflict is not None:
            _, self._module_offload_disabled_reason = module_offload_conflict
            logger.warning("Module offload disabled: %s", self._module_offload_disabled_reason)
        elif module_offload_request.requested:
            module_offload_plan = build_module_offload_plan(
                backbone=unet,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                config=module_offload_request,
            )
            if module_offload_plan.enabled:
                self._module_offload_manager = ModuleResidencyManager(module_offload_plan, device=torch.device(device))
                self._module_offload_verify_state = bool(module_offload_plan.config.verify_state)
                self.memory_optimization_state = module_offload_plan.as_dict()
                logger.info(
                    "Module offload enabled: backbone=%s%% text_encoder=%s%% selected=%s/%s (~%.1f MB)",
                    module_offload_plan.config.effective_backbone_ratio,
                    module_offload_plan.config.effective_text_encoder_ratio,
                    module_offload_plan.selected_count,
                    module_offload_plan.candidate_count,
                    self._module_offload_manager.estimate_vram_savings_mb(),
                )
                for scope_name, scope in module_offload_plan.scopes.items():
                    logger.info(
                        "Module offload scope %s: ratio=%s%% candidates=%s selected=%s",
                        scope_name,
                        scope.ratio,
                        scope.candidate_count,
                        scope.selected_count,
                    )
            else:
                self._module_offload_disabled_reason = module_offload_plan.reason
                if self._module_offload_disabled_reason:
                    logger.info("Module offload disabled: %s", self._module_offload_disabled_reason)

        if self._module_offload_manager is not None:
            return

        if vram_swap_to_ram and (blocks_to_swap > 0 or swap_granularity != "off" or swap_count > 0 or swap_ratio > 0.0):
            logger.warning("Memory swap is incompatible with vram_swap_to_ram; disabling memory swap")
            swap_granularity = "off"
            blocks_to_swap = 0
        if safe_fallback and (blocks_to_swap > 0 or swap_granularity != "off" or swap_count > 0 or swap_ratio > 0.0):
            logger.warning("Memory swap is incompatible with safe_fallback; disabling memory swap")
            swap_granularity = "off"
            blocks_to_swap = 0
        if torch_compile and (blocks_to_swap > 0 or swap_granularity != "off" or swap_count > 0 or swap_ratio > 0.0):
            logger.warning("Memory swap is incompatible with torch_compile; disabling memory swap")
            swap_granularity = "off"
            blocks_to_swap = 0
        if swap_granularity == "layer" and gradient_checkpointing:
            raise ValueError("Layer swap is incompatible with gradient_checkpointing")

        from .memory_optimizations import BlockSwapOffloader, LayerSwapOffloader, build_swap_plan, build_swap_units

        all_blocks = []
        stage_metadata = []
        if hasattr(unet, "down_blocks"):
            down_blocks = list(unet.down_blocks)
            all_blocks.extend(down_blocks)
            stage_metadata.extend(["down"] * len(down_blocks))
        if hasattr(unet, "mid_block"):
            all_blocks.append(unet.mid_block)
            stage_metadata.append("mid")
        if hasattr(unet, "up_blocks"):
            up_blocks = list(unet.up_blocks)
            all_blocks.extend(up_blocks)
            stage_metadata.extend(["up"] * len(up_blocks))
        if not all_blocks and hasattr(unet, "net") and hasattr(unet.net, "blocks"):
            dit_blocks = list(unet.net.blocks)
            all_blocks.extend(dit_blocks)
            stage_metadata.extend(["dit"] * len(dit_blocks))
        if not all_blocks and hasattr(unet, "_block_modules"):
            fallback_source = unet._block_modules
            fallback_blocks = list(fallback_source() if callable(fallback_source) else fallback_source)
            all_blocks.extend(fallback_blocks)
            stage_metadata.extend(["fallback"] * len(fallback_blocks))

        class _SwapConfig:
            pass

        swap_config = _SwapConfig()
        swap_config.swap_granularity = swap_granularity
        swap_config.swap_ratio = swap_ratio
        swap_config.swap_count = swap_count
        swap_config.block_merge_size = block_merge_size
        swap_config.blocks_to_swap = blocks_to_swap
        swap_plan = build_swap_plan(swap_config, len(all_blocks), stage_metadata)
        self.memory_optimization_state = swap_plan.as_dict()
        self.memory_optimization_state.setdefault("mode", "swap" if swap_plan.enabled else "none")
        if self._module_offload_disabled_reason:
            self.memory_optimization_state["module_offload_disabled_reason"] = self._module_offload_disabled_reason
            if not self.memory_optimization_state.get("reason"):
                self.memory_optimization_state["reason"] = self._module_offload_disabled_reason

        if all_blocks and swap_plan.enabled:
            should_swap = lambda mod, name: (
                not getattr(mod, "_lora_leaf", False)
                and not getattr(mod, "lulynx_weight_residency_active", False)
            )
            if swap_plan.effective_granularity == "layer":
                self._block_offloader = LayerSwapOffloader(
                    blocks=all_blocks,
                    layers_to_swap=swap_plan.units_swapped,
                    device=torch.device(device),
                    should_swap=should_swap,
                )
            else:
                units = None
                if swap_plan.effective_granularity == "merged_block":
                    units = build_swap_units(stage_metadata, len(all_blocks), swap_plan.block_merge_size)
                self._block_offloader = BlockSwapOffloader(
                    blocks=all_blocks,
                    blocks_to_swap=swap_plan.units_swapped,
                    device=torch.device(device),
                    enable_backward=(swap_plan.effective_granularity == "block"),
                    should_swap=should_swap,
                    units=units,
                    selected_unit_indices=(
                        self._precision_swap_profile.get("selected_indices")
                        if self._precision_swap_profile
                        and swap_plan.effective_granularity == "block"
                        else None
                    ),
                    strategy=block_swap_strategy,
                    release_cache_on_prepare=(self._cuda_cache_release_strategy == "aggressive"),
                )
            self._block_offloader.prepare_before_forward()
            self._block_offloader.install_forward_hooks(unet)
            self.memory_optimization_state.update(self._block_offloader.strategy_state())
            self._update_block_swap_profile()
            logger.info(
                "Memory swap enabled: requested=%s effective=%s units=%s/%s ratio=%.2f merge=%s strategy=%s",
                swap_plan.requested_granularity,
                swap_plan.effective_granularity,
                swap_plan.units_swapped,
                swap_plan.units_total,
                swap_plan.swap_ratio,
                swap_plan.block_merge_size,
                self.memory_optimization_state.get("block_swap_strategy"),
            )
            if self._precision_swap_profile:
                self.memory_optimization_state["precision_swap_profile"] = self._precision_swap_profile
        elif self._module_offload_disabled_reason:
            self.memory_optimization_state = _none_memory_optimization_state(
                self._module_offload_disabled_reason,
                source="runtime",
            )

        # ── Port 5: Gradient Release ──
        self._gradient_release_manager = None
        if gradient_release_enabled:
            from .gradient_release import GradientReleaseManager
            self._gradient_release_manager = GradientReleaseManager(
                mode=gradient_release_mode,
                accumulation_steps=gradient_accumulation_steps,
            )
            self._gradient_release_manager.register_parameters(
                (p for p in unet.parameters() if p.requires_grad),
                optimizer,
            )
            logger.info("Gradient release enabled: mode=%s", gradient_release_mode)
        self._activation_compression_ctx = ActivationCompressionContext(
            enabled=bool(activation_compression_enabled),
            storage_dtype=str(activation_compression_dtype or "fp16"),
            min_tensor_bytes=int(max(float(activation_compression_min_tensor_mb or 0.0), 0.0) * 1024 * 1024),
        )
        if activation_compression_enabled:
            logger.info(
                "Activation compression enabled: dtype=%s min_tensor_mb=%.2f",
                activation_compression_dtype,
                float(activation_compression_min_tensor_mb or 0.0),
            )
        from .activation_offload import ActivationCpuOffloadContext
        self._activation_offload_ctx = ActivationCpuOffloadContext(
            enabled=bool(activation_cpu_offload_enabled),
            min_tensor_bytes=int(max(float(activation_cpu_offload_min_tensor_mb or 0.0), 0.0) * 1024 * 1024),
            pool_gb=float(activation_cpu_offload_pool_gb or 1.0),
            device=str(device),
        )
        if activation_cpu_offload_enabled:
            guarded = self._activation_offload_ctx.register_parameter_guard(unet)
            logger.info(
                "Activation CPU offload enabled: min_tensor_mb=%.2f pool_gb=%.2f guarded_tensors=%d",
                float(activation_cpu_offload_min_tensor_mb or 0.0),
                float(activation_cpu_offload_pool_gb or 1.0),
                guarded,
            )
            if activation_compression_enabled:
                logger.warning(
                    "Activation CPU offload and activation compression are both enabled; "
                    "saved-tensor hooks do not nest, offload takes precedence."
                )

        if self._turbocore_native_update_gate.requested:
            self._refresh_turbocore_native_update_readiness()
        self._refresh_turbocore_native_update_runtime_profile()

        # ── Port 5: Offloaded Checkpointing (pinned_async) ──
        self._offloaded_checkpoint_ctx = None
        self._offloaded_checkpoint_runtime_profile: Dict[str, Any] = build_offloaded_checkpoint_runtime_profile(
            requested=bool(cpu_offload_checkpointing),
            mode=cpu_offload_checkpointing_mode,
            pool_gb=cpu_offload_checkpointing_pool_gb,
        )
        if cpu_offload_checkpointing and cpu_offload_checkpointing_mode == "pinned_async":
            from .offloaded_checkpointing import OffloadedCheckpointContext
            self._offloaded_checkpoint_ctx = OffloadedCheckpointContext(
                pool_gb=cpu_offload_checkpointing_pool_gb,
                device=device,
            )
            self._offloaded_checkpoint_runtime_profile = build_offloaded_checkpoint_runtime_profile(
                requested=True,
                mode=cpu_offload_checkpointing_mode,
                pool_gb=cpu_offload_checkpointing_pool_gb,
                context=self._offloaded_checkpoint_ctx,
            )
            logger.info(
                "Offloaded checkpointing (pinned_async): pool=%.2f GB",
                cpu_offload_checkpointing_pool_gb,
            )

        # ── Port 5: Resolution-Aware Dynamic Micro-Batch ──
        self._dynamic_batch_scheduler = None
        if resolution_aware_batch_enabled:
            from .dynamic_resolution_batch import DynamicMicroBatchScheduler, ResolutionBatchConfig
            self._dynamic_batch_scheduler = DynamicMicroBatchScheduler(
                config=ResolutionBatchConfig(
                    base_resolution=resolution_aware_batch_base_resolution,
                    base_accumulation_steps=gradient_accumulation_steps,
                    max_factor=resolution_aware_batch_max_factor,
                    min_factor=resolution_aware_batch_min_factor,
                ),
            )
            logger.info(
                "Resolution-aware batch enabled: base_res=%d base_steps=%d",
                resolution_aware_batch_base_resolution,
                gradient_accumulation_steps,
            )

        # ── Port 5: Pipeline Parallelism ──
        self._pipeline_manager = None
        self._pipeline_parallel_runtime_profile: Dict[str, Any] = build_pipeline_parallel_runtime_profile(
            requested=bool(pipeline_parallel_enabled),
            chunks=pipeline_parallel_chunks,
            split_points=pipeline_parallel_split_points,
            available=False,
            disabled_reason="not_requested" if not pipeline_parallel_enabled else "",
        )
        if pipeline_parallel_enabled:
            from .pipeline_parallel import PipelineParallelManager, PipelineConfig, is_pipeline_parallel_available
            pipeline_available = is_pipeline_parallel_available()
            block_accessor = ""
            num_stages = 0
            if pipeline_available:
                pp_config = PipelineConfig(
                    num_chunks=pipeline_parallel_chunks,
                    split_points=pipeline_parallel_split_points,
                )
                self._pipeline_manager = PipelineParallelManager(pp_config)
                if hasattr(unet, "net") and hasattr(unet.net, "blocks"):
                    block_accessor = "net.blocks"
                elif hasattr(unet, "down_blocks"):
                    block_accessor = "down_blocks"
                num_stages = self._pipeline_manager.partition_model(unet, block_accessor)
                if num_stages < 2:
                    logger.warning("Pipeline parallelism: not enough stages (%d), disabled", num_stages)
                    self._pipeline_parallel_runtime_profile = build_pipeline_parallel_runtime_profile(
                        requested=True,
                        chunks=pipeline_parallel_chunks,
                        split_points=pipeline_parallel_split_points,
                        available=True,
                        block_accessor=block_accessor,
                        manager=self._pipeline_manager,
                        partition_stages=num_stages,
                        disabled_reason="not_enough_pipeline_stages",
                    )
                    self._pipeline_manager = None
                else:
                    self._pipeline_parallel_runtime_profile = build_pipeline_parallel_runtime_profile(
                        requested=True,
                        chunks=pipeline_parallel_chunks,
                        split_points=pipeline_parallel_split_points,
                        available=True,
                        block_accessor=block_accessor,
                        manager=self._pipeline_manager,
                        partition_stages=num_stages,
                    )
                    logger.info("Pipeline parallelism enabled: %d stages", num_stages)
            else:
                self._pipeline_parallel_runtime_profile = build_pipeline_parallel_runtime_profile(
                    requested=True,
                    chunks=pipeline_parallel_chunks,
                    split_points=pipeline_parallel_split_points,
                    available=False,
                    disabled_reason="requires_at_least_two_cuda_gpus",
                )
                logger.warning("Pipeline parallelism requires >= 2 GPUs; disabled")

        self._refresh_training_loop_runtime_profile()

    def get_memory_experiment_profile(self) -> Dict[str, Any]:
        """Return lightweight telemetry for experimental memory optimizers."""

        profile: Dict[str, Any] = {}
        newbie_backward_op_profile = getattr(self, "_newbie_backward_op_profile", None)
        if isinstance(newbie_backward_op_profile, dict) and newbie_backward_op_profile:
            profile["newbie_backward_op_profile"] = dict(newbie_backward_op_profile)
        newbie_module_timing_profile = getattr(self, "_newbie_module_timing_profile", None)
        if isinstance(newbie_module_timing_profile, dict) and newbie_module_timing_profile:
            profile["newbie_module_timing_profile"] = dict(newbie_module_timing_profile)
        if getattr(self, "_activation_compression_ctx", None) is not None:
            try:
                profile["activation_compression"] = self._activation_compression_ctx.as_dict()
            except Exception as exc:
                profile["activation_compression_error"] = f"{type(exc).__name__}: {exc}"
        if getattr(self, "_activation_offload_ctx", None) is not None:
            try:
                profile["activation_cpu_offload"] = self._activation_offload_ctx.as_dict()
            except Exception as exc:
                profile["activation_cpu_offload_error"] = f"{type(exc).__name__}: {exc}"
        gr = getattr(self, "_gradient_release_manager", None)
        if gr is not None:
            try:
                stats = dict(getattr(gr, "stats", {}) or {})
                stats.update(
                    {
                        "enabled": True,
                        "needs_external_optimizer_step": bool(getattr(gr, "needs_external_optimizer_step", True)),
                    }
                )
                profile["gradient_release"] = stats
            except Exception as exc:
                profile["gradient_release_error"] = f"{type(exc).__name__}: {exc}"
        scheduler = getattr(self, "_dynamic_batch_scheduler", None)
        if scheduler is not None:
            try:
                profile["resolution_aware_batch"] = {
                    "enabled": True,
                    "base_resolution": int(getattr(scheduler.config, "base_resolution", 0) or 0),
                    "base_accumulation_steps": int(getattr(scheduler.config, "base_accumulation_steps", 0) or 0),
                    "max_factor": float(getattr(scheduler.config, "max_factor", 0.0) or 0.0),
                    "min_factor": float(getattr(scheduler.config, "min_factor", 0.0) or 0.0),
                    "stats": dict(getattr(scheduler, "stats", {}) or {}),
                }
            except Exception as exc:
                profile["resolution_aware_batch_error"] = f"{type(exc).__name__}: {exc}"
        pipeline_profile = getattr(self, "_pipeline_parallel_runtime_profile", None)
        if pipeline_profile:
            manager = getattr(self, "_pipeline_manager", None)
            if manager is not None:
                pipeline_profile = build_pipeline_parallel_runtime_profile(
                    requested=bool(pipeline_profile.get("requested", False)),
                    chunks=int(pipeline_profile.get("chunks", 1) or 1),
                    split_points=str(pipeline_profile.get("split_points", "") or ""),
                    available=bool(pipeline_profile.get("available", False)),
                    block_accessor=str(pipeline_profile.get("block_accessor", "") or ""),
                    manager=manager,
                    partition_stages=int(pipeline_profile.get("partition_stages", 0) or 0),
                )
                self._pipeline_parallel_runtime_profile = dict(pipeline_profile)
            profile["pipeline_parallel"] = dict(pipeline_profile)
        offloaded_checkpoint_profile = getattr(self, "_offloaded_checkpoint_runtime_profile", None)
        if offloaded_checkpoint_profile:
            offloaded_checkpoint_profile = build_offloaded_checkpoint_runtime_profile(
                requested=bool(offloaded_checkpoint_profile.get("requested", False)),
                mode=str(offloaded_checkpoint_profile.get("mode", "standard") or "standard"),
                pool_gb=float(offloaded_checkpoint_profile.get("pool_gb", 0.0) or 0.0),
                context=getattr(self, "_offloaded_checkpoint_ctx", None),
            )
            self._offloaded_checkpoint_runtime_profile = dict(offloaded_checkpoint_profile)
            profile["offloaded_checkpointing"] = dict(offloaded_checkpoint_profile)
        if bool(getattr(self, "data_transfer_profile_enabled", False)):
            profile["data_transfer_profile"] = {
                "enabled": True,
                "mode": str(getattr(self, "data_transfer_profile_mode", "event") or "event"),
                "window": int(getattr(self, "data_transfer_profile_window", 50) or 50),
                "last": dict(getattr(self, "_last_transfer_profile_snapshot", None) or {}),
            }
        step_timing_window = getattr(self, "_last_step_timing_window", None)
        if isinstance(step_timing_window, dict) and step_timing_window:
            profile["step_timing_window"] = dict(step_timing_window)
        if getattr(self, "_step_phase_profiler", None) is not None:
            profile["step_phase_profile"] = {
                "enabled": bool(getattr(self._step_phase_profiler, "enabled", False)),
                "sync_cuda": bool(getattr(self._step_phase_profiler, "sync_cuda", False)),
            }
            latest_phase = self._step_phase_profiler.latest_snapshot()
            if latest_phase:
                profile["step_phase_profile"]["last"] = latest_phase
            latest_bubble = self._step_phase_profiler.latest_bubble_profile()
            if latest_bubble:
                profile["step_phase_profile"]["gpu_bubble_profile"] = latest_bubble
        data_pipeline = getattr(self, "_training_data_pipeline_runtime_report", None)
        if isinstance(data_pipeline, dict) and data_pipeline:
            profile["training_data_pipeline"] = dict(data_pipeline)
        orchestrator_runtime = getattr(self, "_training_step_orchestrator_runtime_profile", None)
        if isinstance(orchestrator_runtime, dict) and orchestrator_runtime:
            profile["training_step_orchestrator_runtime"] = dict(orchestrator_runtime)
        pipeline_trace = compact_lulynx_pipeline_trace(getattr(self, "_last_pipeline_trace", None))
        if pipeline_trace:
            profile["training_pipeline_trace"] = pipeline_trace
            profile["multi_batch_promotion_gate"] = build_lulynx_multi_batch_promotion_gate(
                training_pipeline_trace=pipeline_trace,
            )
        profile["cuda_cache_release"] = {
            "strategy": str(getattr(self, "_cuda_cache_release_strategy", "off") or "off"),
            "interval": int(getattr(self, "_cuda_cache_release_interval", 1) or 1),
            "last": dict(getattr(self, "_last_cuda_cache_release", {}) or {}),
        }
        return profile

    def _refresh_training_loop_runtime_profile(self) -> Dict[str, Any]:
        profile = self.get_memory_experiment_profile()
        if profile:
            self.memory_optimization_state["training_loop_runtime"] = dict(profile)
        return dict(profile)

    def _should_profile_newbie_backward_ops(self) -> bool:
        if str(getattr(self, "_model_arch", "") or "").strip().lower() != "newbie":
            return False
        if not bool(getattr(self, "newbie_backward_op_profile_enabled", False)):
            return False
        max_samples = max(int(getattr(self, "newbie_backward_op_profile_max_samples", 1) or 1), 1)
        samples = getattr(self, "_newbie_backward_op_profile_samples", None)
        return len(samples) < max_samples if isinstance(samples, list) else True

    def _record_newbie_backward_op_profile(self, report: dict[str, Any]) -> None:
        if not isinstance(report, dict) or not report:
            return
        sample = dict(report)
        sample["step"] = int(getattr(self, "global_step", 0) or 0)
        samples = getattr(self, "_newbie_backward_op_profile_samples", None)
        if not isinstance(samples, list):
            samples = []
        samples.append(sample)
        max_samples = max(int(getattr(self, "newbie_backward_op_profile_max_samples", 1) or 1), 1)
        if len(samples) > max_samples:
            samples = samples[-max_samples:]
        self._newbie_backward_op_profile_samples = samples
        self._newbie_backward_op_profile = {
            "report": "newbie_backward_op_profile_window_v0",
            "enabled": True,
            "sample_count": len(samples),
            "max_samples": max_samples,
            "latest": dict(sample),
            "samples": [dict(item) for item in samples],
        }

    def _should_profile_newbie_module_timing(self) -> bool:
        if str(getattr(self, "_model_arch", "") or "").strip().lower() != "newbie":
            return False
        if not bool(getattr(self, "newbie_module_timing_profile_enabled", False)):
            return False
        max_samples = max(int(getattr(self, "newbie_module_timing_profile_max_samples", 1) or 1), 1)
        samples = getattr(self, "_newbie_module_timing_profile_samples", None)
        return len(samples) < max_samples if isinstance(samples, list) else True

    def _start_newbie_module_timing_profile(self) -> NewbieModuleTimingProfiler | None:
        if not self._should_profile_newbie_module_timing():
            return None
        try:
            profiler = NewbieModuleTimingProfiler(
                self.unet,
                top_k=max(int(getattr(self, "newbie_module_timing_profile_top_k", 12) or 12), 1),
            ).attach()
        except Exception as exc:
            self._record_newbie_module_timing_profile(
                {
                    "report": "newbie_module_timing_profile_v0",
                    "status": "profile_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_default_change": False,
                    "probe_only": True,
                }
            )
            return None
        return profiler if profiler.enabled else None

    def _record_newbie_module_timing_profile(self, report: dict[str, Any]) -> None:
        if not isinstance(report, dict) or not report:
            return
        sample = dict(report)
        sample["step"] = int(getattr(self, "global_step", 0) or sample.get("step", 0) or 0)
        samples = getattr(self, "_newbie_module_timing_profile_samples", None)
        if not isinstance(samples, list):
            samples = []
        samples.append(sample)
        max_samples = max(int(getattr(self, "newbie_module_timing_profile_max_samples", 1) or 1), 1)
        if len(samples) > max_samples:
            samples = samples[-max_samples:]
        self._newbie_module_timing_profile_samples = samples
        self._newbie_module_timing_profile = {
            "report": "newbie_module_timing_profile_window_v0",
            "enabled": True,
            "sample_count": len(samples),
            "max_samples": max_samples,
            "latest": dict(sample),
            "samples": [dict(item) for item in samples],
        }

    @staticmethod
    def _normalize_gradient_accumulation_mode(mode: Optional[str]) -> str:
        normalized = str(mode or "fast").strip().lower()
        aliases = {
            "turbo": "fast",
            "quick": "fast",
            "快速": "fast",
            "legacy": "classic",
            "strict": "classic",
            "经典": "classic",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"fast", "classic"} else "fast"

    @staticmethod
    def _normalize_data_transfer_profile_mode(mode: Optional[str]) -> str:
        normalized = str(mode or "event").strip().lower()
        aliases = {
            "cuda_event": "event",
            "events": "event",
            "async": "event",
            "synchronize": "sync",
            "synchronized": "sync",
            "legacy": "sync",
            "none": "off",
            "disabled": "off",
            "false": "off",
            "0": "off",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"event", "sync", "off"} else "event"

    @staticmethod
    def _normalize_loss_precision(mode: Optional[str]) -> str:
        normalized = str(mode or "fp32_loss").strip().lower()
        aliases = {
            "fp32": "fp32_loss",
            "float32": "fp32_loss",
            "full": "fp32_loss",
            "safe": "fp32_loss",
            "mixed": "mixed_loss",
            "native": "mixed_loss",
            "amp": "mixed_loss",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"fp32_loss", "mixed_loss"} else "fp32_loss"

    def _loss_operands(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "loss_precision", "fp32_loss") == "mixed_loss":
            return prediction, target
        return prediction.float(), target.float()

    # ------------------------------------------------------------------
    # A vram/offload/transfer methods moved to training_loop_runtime_memory.TrainingLoopRuntimeMemoryMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _encode_latents_with_vae(self, images: torch.Tensor) -> torch.Tensor:
        self._ensure_cpu_resident_components("before_vae_encode")
        runtime_context = (
            module_runtime_state(self.vae, device=self._runtime_device, dtype=torch.float32)
            if self._vae_cpu_residency
            else nullcontext()
        )
        with runtime_context:
            with torch.no_grad():
                self.vae.to(dtype=torch.float32)
                latents = self.vae.encode(images.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                result = latents.to(dtype=self.dtype)
        self._ensure_cpu_resident_components("after_vae_encode")
        self._verify_phase_module_states("vae_encode")
        return result

    # ------------------------------------------------------------------
    # B step-timing/transfer methods moved to training_loop_runtime_memory.TrainingLoopRuntimeMemoryMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _compute_diffusion_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        reduction: str = "mean",
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_type = (self.loss_type or "l2").lower()
        stepped_weight = 1.0

        stepped_schedule = getattr(self, "_stepped_loss_schedule", None)
        if stepped_schedule is not None:
            loss_type, stepped_weight = stepped_schedule.resolve(getattr(self, "global_step", 0))

        if loss_type == "huber":
            delta = self._huber_delta(timesteps, noise_pred, target=noise)
            return self._elementwise_huber_loss(noise_pred, noise, delta, reduction) * stepped_weight

        if loss_type == "smooth_l1":
            beta = self._huber_delta(timesteps, noise_pred, target=noise)
            return self._elementwise_smooth_l1_loss(noise_pred, noise, beta, reduction) * stepped_weight

        if loss_type == "l1":
            return F.l1_loss(noise_pred, noise, reduction=reduction) * stepped_weight

        return F.mse_loss(noise_pred, noise, reduction=reduction) * stepped_weight

    def _huber_delta(self, timesteps: Optional[torch.Tensor], reference: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return scalar or per-sample Huber threshold for the configured schedule."""
        base = max(float(self.huber_c), 1e-6)
        scale = max(float(self.huber_scale), 1e-6)
        schedule = str(self.huber_schedule or "constant").strip().lower()
        if timesteps is None or schedule == "constant":
            return torch.tensor(base * scale, device=reference.device, dtype=reference.dtype)

        steps = timesteps.to(device=reference.device, dtype=torch.float32).view(-1)
        # Normalize timesteps to sigma in [0, 1] so schedules are scale-correct
        # for every route: DDPM steps are 0..num_train_timesteps, flow routes
        # carry sigma*1000, and the anima faithful forward carries raw sigma.
        arch = str(getattr(self, "_model_arch", "") or "").strip().lower()
        uses_flow = arch in {"anima", "newbie"} or bool(getattr(self, "flow_model", False))
        if uses_flow:
            t_scale = 1.0 if (arch == "anima" and bool(getattr(self, "anima_faithful_forward", False))) else 1000.0
        else:
            t_scale = float(getattr(getattr(self.noise_scheduler, "config", None), "num_train_timesteps", 1000) or 1000)
        sigma_norm = (steps / max(t_scale, 1.0)).clamp(0.0, 1.0)
        if schedule == "exponential":
            # Smoothly decays from scale at sigma=0 toward base*scale at sigma=1
            # (identical curve to the old steps/max_steps form on DDPM routes).
            log_base = torch.log(torch.tensor(base, device=reference.device, dtype=torch.float32))
            delta = torch.exp(log_base * sigma_norm) * scale
        elif schedule == "snr":
            if uses_flow:
                # Rectified flow has no DDPM alphas_cumprod table; its SNR is
                # ((1 - sigma) / sigma)^2 by construction of x_t.
                sigma_f = sigma_norm.clamp(1e-6, 1.0 - 1e-6)
                snr = torch.square((1.0 - sigma_f) / sigma_f)
            else:
                snr = self._compute_snr(steps.long()).to(device=reference.device, dtype=torch.float32)
            sigma = torch.rsqrt(torch.clamp(snr, min=1e-8))
            delta = ((1.0 - base) / torch.square(1.0 + sigma) + base) * scale
        elif schedule == "auto":
            if target is not None:
                residuals = (reference - target).detach().float()
                per_sample = residuals.flatten(1).norm(dim=1)
                percentile_val = torch.quantile(per_sample, float(self.huber_auto_percentile))
                delta = percentile_val.clamp(min=base * scale).expand_as(steps) * scale
            else:
                delta = torch.full_like(steps, base * scale)
        else:
            logger.warning("[Loss] Unknown huber_schedule=%s; falling back to constant", schedule)
            delta = torch.full_like(steps, base * scale)

        view_shape = (delta.shape[0],) + (1,) * (reference.dim() - 1)
        return delta.to(dtype=reference.dtype).view(view_shape)

    @staticmethod
    def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()

    def _elementwise_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        abs_err = (pred - target).abs()
        quadratic = torch.minimum(abs_err, delta)
        linear = abs_err - quadratic
        loss = 0.5 * quadratic.square() + delta * linear
        return self._reduce_loss(loss, reduction)

    def _elementwise_smooth_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        beta: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        abs_err = (pred - target).abs()
        loss = torch.where(abs_err < beta, 0.5 * abs_err.square() / beta, abs_err - 0.5 * beta)
        return self._reduce_loss(loss, reduction)

    def _sample_strength(self, base_strength: float, randomize: bool, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
        base = float(base_strength or 0.0)
        if base <= 0:
            return torch.zeros(shape, device=reference.device, dtype=reference.dtype)
        if randomize:
            return torch.rand(shape, device=reference.device, dtype=reference.dtype) * base
        return torch.full(shape, base, device=reference.device, dtype=reference.dtype)

    def _velocity_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if hasattr(self.noise_scheduler, "get_velocity"):
            return self.noise_scheduler.get_velocity(latents, noise, timesteps)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        alpha_t = alphas_cumprod[timesteps]
        while alpha_t.dim() < latents.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        return alpha_t.sqrt() * noise - (1.0 - alpha_t).sqrt() * latents

    def _scale_v_prediction_loss(self, loss: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if not self.scale_v_pred_loss_like_noise_pred:
            return loss
        snr = torch.clamp(self._compute_snr(timesteps).to(device=loss.device, dtype=loss.dtype), max=1000.0)
        scale = snr / (snr + 1.0)
        while scale.dim() < loss.dim():
            scale = scale.unsqueeze(-1)
        return loss * scale
        
    def set_auditor(self, auditor: Any, interval: int = 50):
        """设置审计器"""
        self.auditor = auditor
        self.auditor_interval = interval
        
    def set_smart_rank_controller(self, controller: Any):
        """设置 SmartRank 控制器"""
        self.smart_rank_controller = controller

    def set_safeguard(self, safeguard: Any):
        """设置安全卫士"""
        self.safeguard = safeguard

    def emit_runtime_event(self, payload: Dict[str, Any]) -> None:
        callback = getattr(self, "on_runtime_event", None)
        if callback is None:
            return
        try:
            callback(dict(payload))
        except Exception as exc:
            logger.debug("runtime event callback failed: %s", exc)
        
    def set_lisa_scheduler(self, lisa_scheduler: Any):
        """设置 LISA 调度器"""
        self.lisa_scheduler = lisa_scheduler
        
    def set_lulynx_wrapper(self, wrapper: Any):
        """设置 Lulynx 包装器 (MN-LoRA, GSP, Ghost etc.)"""
        self.lulynx_wrapper = wrapper
        
    def _encode_prompt(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        """编码文本提示"""
        self._ensure_cpu_resident_components("before_te_encode")
        if self.te_manager:
            res = self.te_manager.encode_prompts(captions)
            if res:
                encoder_hidden_states = res.get("encoder_hidden_states")
                needs_native_pooled = (
                    isinstance(encoder_hidden_states, dict)
                    or (
                        self._family.uses_clip_pooled_features
                        and "pooled_prompt_embeds" not in res
                    )
                    or (
                        self._family.uses_time_ids
                        and "pooled_prompt_embeds" not in res
                    )
                )
                if not needs_native_pooled:
                    self.te_manager.cache_encoder_output(res)
                    self._ensure_cpu_resident_components("after_te_encode")
                    self._verify_phase_module_states("te_encode")
                    return res

                native_res = self._encode_prompt_native(captions)
                merged = dict(res)
                if "pooled_prompt_embeds" not in merged and "pooled_prompt_embeds" in native_res:
                    merged["pooled_prompt_embeds"] = native_res["pooled_prompt_embeds"]
                self.te_manager.cache_encoder_output(merged)
                self._ensure_cpu_resident_components("after_te_encode")
                self._verify_phase_module_states("te_encode")
                return merged

        result = self._encode_prompt_native(captions)
        if self.te_manager:
            self.te_manager.cache_encoder_output(result)
        self._ensure_cpu_resident_components("after_te_encode")
        self._verify_phase_module_states("te_encode")
        return result

    def _encode_prompt_native(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        """Native text encoding path without Semantic Base-Tuner overrides."""
        def _module_device(module: torch.nn.Module) -> torch.device:
            try:
                return next(module.parameters()).device
            except StopIteration:
                return self._runtime_device

        def _runtime_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device != self._runtime_device:
                tensor = tensor.to(self._runtime_device)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=self.dtype)
            return tensor

        grad_context = torch.enable_grad() if self._train_text_encoder_any else torch.no_grad()
        te1_context = (
            module_runtime_state(self.text_encoder_1, device=self._runtime_device, dtype=self.dtype)
            if self._text_encoder_cpu_residency
            else nullcontext()
        )
        te2_context = (
            module_runtime_state(self.text_encoder_2, device=self._runtime_device, dtype=self.dtype)
            if self._text_encoder_cpu_residency and self.text_encoder_2 is not None
            else nullcontext()
        )
        with te1_context, te2_context, grad_context:
                # Tokenize with fixed padding if enabled (for torch.compile static shape)
                if self._token_padder_1 is not None:
                    tokens_1 = self._token_padder_1.encode_batch(captions, return_tensors="pt")
                else:
                    tokens_1 = self.tokenizer_1(
                        captions,
                        padding="max_length",
                        max_length=self.tokenizer_1.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                te1_device = _module_device(self.text_encoder_1)
                input_ids_1 = tokens_1.input_ids.to(te1_device)

                # Text Encoder 1
                encoder_hidden_states_1 = self.text_encoder_1(
                    input_ids_1,
                    output_hidden_states=True,
                )

                if self._family.has_dual_text_encoders:
                    if self.tokenizer_2 is None or self.text_encoder_2 is None:
                        raise RuntimeError(
                            f"Model family '{self._model_arch}' requires a CLIP-compatible secondary "
                            "text encoder and tokenizer, but the loader did not provide them."
                        )
                    if self._token_padder_2 is not None:
                        tokens_2 = self._token_padder_2.encode_batch(captions, return_tensors="pt")
                    else:
                        tokens_2 = self.tokenizer_2(
                            captions,
                            padding="max_length",
                            max_length=self.tokenizer_2.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                    te2_device = _module_device(self.text_encoder_2)
                    input_ids_2 = tokens_2.input_ids.to(te2_device)

                    encoder_hidden_states_2 = self.text_encoder_2(
                        input_ids_2,
                        output_hidden_states=True,
                    )

                    if self._family.uses_clip_pooled_features:
                        # Newbie-style: primary encoder hidden states + CLIP pooled features
                        hidden_states = _runtime_tensor(encoder_hidden_states_1.last_hidden_state)
                        clip_pooled = getattr(encoder_hidden_states_2, "text_embeds", None)
                        if clip_pooled is None:
                            # Fallback: pool last hidden state
                            clip_pooled = encoder_hidden_states_2.last_hidden_state.mean(dim=1)
                        clip_pooled = _runtime_tensor(clip_pooled)
                        return {
                            "encoder_hidden_states": hidden_states,
                            "pooled_prompt_embeds": clip_pooled,
                        }
                    else:
                        # SDXL-style: concatenate hidden states from both encoders
                        text_embeds = _runtime_tensor(encoder_hidden_states_2.text_embeds)
                        clip_l_hidden = _runtime_tensor(encoder_hidden_states_1.hidden_states[-2])
                        clip_g_hidden = _runtime_tensor(encoder_hidden_states_2.hidden_states[-2])
                        hidden_states = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)
                        return {
                            "encoder_hidden_states": hidden_states,
                            "pooled_prompt_embeds": text_embeds,
                            "_clip_l_hidden_size": clip_l_hidden.shape[-1],
                            "_clip_g_hidden_size": clip_g_hidden.shape[-1],
                        }
                else:
                    hidden_states = _runtime_tensor(encoder_hidden_states_1.last_hidden_state)
                    return {
                        "encoder_hidden_states": hidden_states,
                    }

    # ------------------------------------------------------------------
    # Qwen3 secondary text encoding (Anima)
    # ------------------------------------------------------------------

    def _encode_qwen3(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        """Encode captions with the Qwen3 secondary text encoder.

        Returns a dict with ``"qwen3_hidden_states"`` (``[batch, seq, dim]``)
        and ``"qwen3_attention_mask"`` (``[batch, seq]``).  Falls back to
        empty tensors when the Qwen3 encoder is unavailable.
        """
        if self.qwen3_encoder is None or self.qwen3_tokenizer is None:
            return {}

        def _module_device(module: torch.nn.Module) -> torch.device:
            try:
                return next(module.parameters()).device
            except StopIteration:
                return torch.device(self.device)

        qwen3_device = _module_device(self.qwen3_encoder)

        try:
            tokens = self.qwen3_tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=getattr(self.qwen3_tokenizer, "model_max_length", 512),
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(qwen3_device)
            attention_mask = tokens["attention_mask"].to(qwen3_device)

            with torch.no_grad():
                outputs = self.qwen3_encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Qwen3 causal LMs expose last_hidden_state
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                return {}

            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
            attention_mask = attention_mask.to(device=self.device)

            return {
                "qwen3_hidden_states": hidden_states,
                "qwen3_attention_mask": attention_mask,
            }

        except Exception as exc:
            logger.warning("Qwen3 encoding failed: %s", exc)
            return {}

    def _drop_tensor_samples(
        self,
        tensor: Optional[torch.Tensor],
        rate: float,
        *,
        feature_slice: Optional[slice] = None,
        mask_shape_dims: int = 1,
    ) -> Optional[torch.Tensor]:
        """Randomly zero whole samples, optionally limited to a feature slice."""
        rate = _normalize_dropout_rate(rate)
        if tensor is None or rate <= 0.0 or tensor.dim() == 0:
            return tensor

        batch_size = tensor.shape[0]
        drop_mask = torch.rand(batch_size, device=tensor.device) < rate
        if not drop_mask.any():
            return tensor

        mask_shape = [batch_size] + [1] * max(int(mask_shape_dims), 0)
        mask = drop_mask.view(*mask_shape)
        if feature_slice is None:
            return torch.where(mask.expand_as(tensor), torch.zeros_like(tensor), tensor)

        result = tensor.clone()
        target = result[..., feature_slice]
        result[..., feature_slice] = torch.where(mask.expand_as(target), torch.zeros_like(target), target)
        return result

    def _apply_text_encoder_dropout(
        self,
        prompt_embeds: Optional[Dict[str, torch.Tensor]],
        *,
        do_backward: bool,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Apply generic and SDXL encoder-specific conditioning dropout."""
        if not do_backward or prompt_embeds is None:
            return prompt_embeds

        encoder_hs = prompt_embeds.get("encoder_hidden_states")
        if encoder_hs is None or not isinstance(encoder_hs, torch.Tensor) or encoder_hs.dim() < 2:
            return prompt_embeds

        batch_size = encoder_hs.shape[0]
        if self.te_dropout > 0.0:
            prompt_embeds["encoder_hidden_states"] = self._drop_tensor_samples(
                encoder_hs,
                self.te_dropout,
                mask_shape_dims=encoder_hs.dim() - 1,
            )
            encoder_hs = prompt_embeds["encoder_hidden_states"]
            pooled = prompt_embeds.get("pooled_prompt_embeds")
            if isinstance(pooled, torch.Tensor) and pooled.shape[0] == batch_size:
                prompt_embeds["pooled_prompt_embeds"] = self._drop_tensor_samples(
                    pooled,
                    self.te_dropout,
                    mask_shape_dims=pooled.dim() - 1,
                )
            attn_mask = prompt_embeds.get("attention_mask")
            if isinstance(attn_mask, torch.Tensor) and attn_mask.shape[0] == batch_size:
                prompt_embeds["attention_mask"] = self._drop_tensor_samples(
                    attn_mask,
                    self.te_dropout,
                    mask_shape_dims=attn_mask.dim() - 1,
                )
            qwen3_hs = prompt_embeds.get("qwen3_hidden_states")
            if isinstance(qwen3_hs, torch.Tensor) and qwen3_hs.shape[0] == batch_size:
                prompt_embeds["qwen3_hidden_states"] = self._drop_tensor_samples(
                    qwen3_hs,
                    self.te_dropout,
                    mask_shape_dims=qwen3_hs.dim() - 1,
                )
            qwen3_mask = prompt_embeds.get("qwen3_attention_mask")
            if isinstance(qwen3_mask, torch.Tensor) and qwen3_mask.shape[0] == batch_size:
                prompt_embeds["qwen3_attention_mask"] = self._drop_tensor_samples(
                    qwen3_mask,
                    self.te_dropout,
                    mask_shape_dims=qwen3_mask.dim() - 1,
                )

        clip_l_size = int(prompt_embeds.get("_clip_l_hidden_size") or 0)
        clip_g_size = int(prompt_embeds.get("_clip_g_hidden_size") or 0)
        if clip_l_size <= 0:
            clip_l_size = int(getattr(getattr(self.text_encoder_1, "config", None), "hidden_size", 0) or 0)
        if clip_g_size <= 0:
            clip_g_size = int(getattr(getattr(self.text_encoder_2, "config", None), "hidden_size", 0) or 0)
        total_clip_size = clip_l_size + clip_g_size
        if encoder_hs.dim() >= 3 and total_clip_size > 0 and encoder_hs.shape[-1] >= total_clip_size:
            if self.clip_l_dropout_rate > 0.0 and clip_l_size > 0:
                encoder_hs = self._drop_tensor_samples(
                    encoder_hs,
                    self.clip_l_dropout_rate,
                    feature_slice=slice(0, clip_l_size),
                    mask_shape_dims=encoder_hs.dim() - 1,
                )
            if self.clip_g_dropout_rate > 0.0 and clip_g_size > 0:
                encoder_hs = self._drop_tensor_samples(
                    encoder_hs,
                    self.clip_g_dropout_rate,
                    feature_slice=slice(clip_l_size, clip_l_size + clip_g_size),
                    mask_shape_dims=encoder_hs.dim() - 1,
                )
            prompt_embeds["encoder_hidden_states"] = encoder_hs

            pooled = prompt_embeds.get("pooled_prompt_embeds")
            if self.clip_g_dropout_rate > 0.0 and isinstance(pooled, torch.Tensor) and pooled.shape[0] == batch_size:
                prompt_embeds["pooled_prompt_embeds"] = self._drop_tensor_samples(
                    pooled,
                    self.clip_g_dropout_rate,
                    mask_shape_dims=pooled.dim() - 1,
                )

        t5_hs = prompt_embeds.get("t5_hidden_states")
        if self.t5_dropout_rate > 0.0 and isinstance(t5_hs, torch.Tensor) and t5_hs.shape[0] == batch_size:
            prompt_embeds["t5_hidden_states"] = self._drop_tensor_samples(
                t5_hs,
                self.t5_dropout_rate,
                mask_shape_dims=t5_hs.dim() - 1,
            )
            t5_mask = prompt_embeds.get("t5_attention_mask")
            if isinstance(t5_mask, torch.Tensor) and t5_mask.shape[0] == batch_size:
                prompt_embeds["t5_attention_mask"] = self._drop_tensor_samples(
                    t5_mask,
                    self.t5_dropout_rate,
                    mask_shape_dims=t5_mask.dim() - 1,
                )
        return prompt_embeds

    # ------------------------------------------------------------------
    # CUDAGraph capture
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # C cudagraph methods moved to training_loop_runtime_memory.TrainingLoopRuntimeMemoryMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _get_timestep_embedding(self, batch_size: int, original_sizes, target_sizes, crop_coords):
        """获取 SDXL 时间步嵌入"""
        if not self._family.uses_time_ids:
            return {}
        
        # SDXL 需要额外的条件
        add_time_ids = []
        for orig, tgt, crop in zip(original_sizes, target_sizes, crop_coords):
            time_id = list(orig) + list(crop[:2]) + list(tgt)
            add_time_ids.append(time_id)
        
        add_time_ids = torch.tensor(add_time_ids, dtype=self.dtype, device=self.device)
        
        return {"added_cond_kwargs": {"time_ids": add_time_ids}}

    def _get_trainable_params(self) -> List[torch.nn.Parameter]:
        if self.lora_injector and hasattr(self.lora_injector, "get_trainable_params"):
            return list(self.lora_injector.get_trainable_params())
        return []

    def _build_compute_reducer_seam(self):
        """Build the block compute-reducer seam from config (lazy, first forward)."""
        try:
            from .dit_compute_reducer_seam import build_compute_reducer_seam
        except ImportError:  # pragma: no cover - direct-file fallback
            from core.lulynx_trainer.dit_compute_reducer_seam import build_compute_reducer_seam
        total_blocks = 0
        net = getattr(getattr(self, "unet", None), "net", None)
        blocks = getattr(net, "blocks", None)
        if blocks is not None:
            try:
                total_blocks = len(blocks)
            except TypeError:
                total_blocks = 0
        return build_compute_reducer_seam(
            enabled=True,
            strategy=self.dit_compute_reducer_strategy,
            total_blocks=total_blocks,
            keep_ratio=self.dit_compute_reducer_keep_ratio,
            min_keep_tokens=self.dit_compute_reducer_min_keep_tokens,
            compression_ratio=self.dit_compute_reducer_compression_ratio,
            min_tokens=self.dit_compute_reducer_min_tokens,
            skip_ratio=self.dit_compute_reducer_skip_ratio,
            skip_every=self.dit_compute_reducer_skip_every,
            warmup_steps=self.dit_compute_reducer_warmup_steps,
            min_block=self.dit_compute_reducer_min_block,
            score_mode=self.dit_compute_reducer_score_mode,
        )

    def _compute_reducer_context(self):
        """Return the live reducer seam context, or a null context when off.

        Off by default -> ``nullcontext`` -> the wrapped forward runs verbatim,
        so the block dispatch is bitwise-identical to legacy behaviour.
        """
        from contextlib import nullcontext

        strategy = str(getattr(self, "dit_compute_reducer_strategy", "none") or "none").strip().lower()
        if strategy in ("", "none"):
            return nullcontext()
        if not self._compute_reducer_seam_built:
            self._compute_reducer_seam = self._build_compute_reducer_seam()
            self._compute_reducer_seam_built = True
        seam = self._compute_reducer_seam
        if seam is None or not getattr(seam, "enabled", False):
            return nullcontext()
        # Advance step/topology for the only step-dependent reducer (blockskip).
        if hasattr(seam, "set_total_blocks"):
            net = getattr(getattr(self, "unet", None), "net", None)
            blocks = getattr(net, "blocks", None)
            if blocks is not None:
                try:
                    seam.set_total_blocks(len(blocks))
                except TypeError:
                    pass
        if hasattr(seam, "set_step"):
            seam.set_step(int(getattr(self, "global_step", 0)), int(getattr(self, "total_steps", 0) or 0))
        try:
            from .dit_compute_reducer_seam import compute_reducer_seam_context
        except ImportError:  # pragma: no cover - direct-file fallback
            from core.lulynx_trainer.dit_compute_reducer_seam import compute_reducer_seam_context
        return compute_reducer_seam_context(seam)

    # ------------------------------------------------------------------
    # Turbocore native-update runtime methods moved to
    # training_loop_turbocore_runtime.TrainingLoopTurbocoreRuntimeMixin
    # (verbatim; resolved via MRO — same self, same call sites)
    # ------------------------------------------------------------------

    def _clone_to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: self._clone_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu(v) for v in value)
        return copy.deepcopy(value)

    @staticmethod
    def _normalize_safeguard_gradient_scan_mode(mode: Any) -> str:
        normalized = str(mode or "batched").strip().lower()
        aliases = {
            "auto": "batched",
            "batch": "batched",
            "batched_cpu": "batched",
            "foreach_norm": "foreach",
            "none": "off",
            "disabled": "off",
            "false": "off",
            "0": "off",
            "true": "batched",
            "1": "batched",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"legacy", "batched", "foreach", "off"} else "batched"

    def _collect_gradients_for_safeguard(self, mode: Optional[str] = None) -> Optional[torch.Tensor]:
        scan_mode = self._normalize_safeguard_gradient_scan_mode(mode)
        if scan_mode == "off":
            return None

        grads = [
            param.grad.detach()
            for param in self._get_trainable_params()
            if param.grad is not None
        ]
        if not grads:
            return None

        if scan_mode == "legacy":
            return torch.stack([grad.float().norm().cpu() for grad in grads])

        grouped: Dict[torch.device, List[torch.Tensor]] = {}
        for grad in grads:
            grouped.setdefault(grad.device, []).append(grad)

        chunks: List[torch.Tensor] = []
        for device_grads in grouped.values():
            device_norms = None
            if scan_mode == "foreach" and hasattr(torch, "_foreach_norm"):
                try:
                    foreach_inputs = [grad.float() for grad in device_grads]
                    device_norms = torch.stack(list(torch._foreach_norm(foreach_inputs, 2.0)))
                except Exception as exc:
                    logger.debug("SafeGuard foreach gradient scan fell back to batched mode: %s", exc)
                    device_norms = None
            if device_norms is None:
                device_norms = torch.stack([grad.float().norm() for grad in device_grads])
            chunks.append(device_norms.detach().cpu())

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks)

    def _capture_safe_state(self) -> Optional[Dict[str, Any]]:
        if not self.lora_injector or not hasattr(self.lora_injector, "get_lora_state_dict"):
            return None

        adapter_state = self.lora_injector.get_lora_state_dict()
        if not adapter_state:
            return None

        return {
            "adapter_state": self._clone_to_cpu(adapter_state),
            "optimizer_state_dict": self._clone_to_cpu(self.optimizer.state_dict()) if self.optimizer else None,
            "scheduler_state_dict": self._clone_to_cpu(self.lr_scheduler.state_dict()) if self.lr_scheduler else None,
            "global_step": self.global_step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }

    def _move_optimizer_state_to_param_device(self):
        if not self.optimizer:
            return

        param_device = None
        for group in self.optimizer.param_groups:
            for param in group.get("params", []):
                if isinstance(param, torch.Tensor):
                    param_device = param.device
                    break
            if param_device is not None:
                break

        if param_device is None:
            return

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=param_device)

    def _restore_safe_state(self, state: Optional[Dict[str, Any]]) -> bool:
        if not state or not self.lora_injector or not hasattr(self.lora_injector, "load_lora_state_dict"):
            return False

        adapter_state = state.get("adapter_state")
        if not adapter_state:
            return False

        self.lora_injector.load_lora_state_dict(adapter_state)

        optimizer_state = state.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_param_device()

        scheduler_state = state.get("scheduler_state_dict")
        if self.lr_scheduler is not None and scheduler_state:
            self.lr_scheduler.load_state_dict(scheduler_state)

        rng_state = state.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)

        cuda_rng_state = state.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)

        python_rng_state = state.get("python_rng_state")
        if python_rng_state is not None:
            random.setstate(python_rng_state)

        numpy_rng_state = state.get("numpy_rng_state")
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)

        self.global_step = int(state.get("global_step", self.global_step))
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self._clear_pcgrad_pending_grads()
        return True

    def _clear_pcgrad_pending_grads(self) -> None:
        self._pcgrad_pending_grads.clear()

    def _pcgrad_named_params(self) -> List[tuple[str, torch.nn.Parameter]]:
        named: List[tuple[str, torch.nn.Parameter]] = []
        seen = set()
        for param in self._get_trainable_params():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            name = self._pcgrad_param_names.get(id(param))
            if not name:
                name = f"param_{len(named)}"
            named.append((name, param))
        return named

    def _capture_pcgrad_microbatch(self, accumulation_steps: int) -> None:
        named_params = self._pcgrad_named_params()
        if not named_params:
            return
        scale = float(accumulation_steps or 1)
        captured: Dict[str, torch.Tensor] = {}
        for name, param in named_params:
            grad = param.grad
            if grad is None:
                continue
            captured[name] = grad.detach().clone() * scale
        if not captured:
            return
        self._pcgrad_pending_grads.append(captured)

    def _apply_pcgrad_pending_grads(self) -> None:
        if not self.pcgrad_enabled:
            return
        if not self._pcgrad_pending_grads:
            return
        named_params = self._pcgrad_named_params()
        if not named_params:
            self._clear_pcgrad_pending_grads()
            self._pcgrad_last_stats = {}
            return

        resolved_grads, stats = resolve_pcgrad_gradients(
            self._pcgrad_pending_grads,
            conflict_threshold=self.pcgrad_conflict_threshold,
            reduction=self.pcgrad_reduction,
        )
        for name, param in named_params:
            grad = resolved_grads.get(name)
            if grad is None:
                param.grad = None
                continue
            if param.grad is None:
                param.grad = grad.detach().to(device=param.device, dtype=param.dtype)
            else:
                param.grad.copy_(grad.detach().to(device=param.grad.device, dtype=param.grad.dtype))
        self._pcgrad_last_stats = stats
        self._clear_pcgrad_pending_grads()

    def _pcgrad_runtime_state(self) -> Dict[str, Any]:
        if not self.pcgrad_enabled:
            return {
                "enabled": False,
                "pending_microbatches": len(self._pcgrad_pending_grads),
            }
        state = {
            "enabled": True,
            "conflict_threshold": float(self.pcgrad_conflict_threshold),
            "reduction": str(self.pcgrad_reduction or "mean"),
            "pending_microbatches": len(self._pcgrad_pending_grads),
        }
        if self._pcgrad_last_stats:
            state["last_step"] = dict(self._pcgrad_last_stats)
        return state

    def _maybe_save_safe_state(self):
        if not self.safeguard or not getattr(self.safeguard.config, "enable_auto_recovery", False):
            return

        interval = max(int(getattr(self.safeguard.config, "nan_check_interval", 1) or 1), 1)
        if self.global_step != 0 and self.global_step % interval != 0:
            return

        safe_state = self._capture_safe_state()
        if safe_state is not None:
            self.safeguard.save_safe_state(self.global_step, safe_state)

    def _loss_to_per_sample(self, loss: torch.Tensor, batch: Dict) -> torch.Tensor:
        """Reduce elementwise loss to one scalar per sample, applying masks first."""
        if loss.dim() <= 1:
            return loss.view(-1)

        masks = batch.get("loss_masks")
        if (self.masked_loss or self.alpha_mask) and isinstance(masks, torch.Tensor):
            weights = masks.to(device=loss.device, dtype=loss.dtype)
            if weights.dim() == 3:
                weights = weights.unsqueeze(1)
            if weights.shape[-2:] != loss.shape[-2:]:
                weights = F.interpolate(weights, size=loss.shape[-2:], mode="bilinear", align_corners=False)
            if weights.shape[1] == 1 and loss.shape[1] != 1:
                weights = weights.expand(-1, loss.shape[1], -1, -1)
            weighted = loss * weights
            reduce_dims = tuple(range(1, weighted.dim()))
            return weighted.sum(dim=reduce_dims) / weights.sum(dim=reduce_dims).clamp_min(1e-6)

        if (self.masked_loss or self.alpha_mask) and not isinstance(masks, torch.Tensor):
            import warnings
            if self.strict_masked_loss:
                raise RuntimeError(
                    "strict_masked_loss is enabled but batch has no loss_masks. "
                    "When using masked_loss=True with cache-first datasets, ensure "
                    "the cache files include a 'loss_mask' array (or set strict_masked_loss=False)."
                )
            if not self._masked_loss_warned:
                warnings.warn(
                    "masked_loss/alpha_mask is enabled but batch has no loss_masks — "
                    "loss will be computed without masking. This is likely a silent no-op. "
                    "To raise an error, set strict_masked_loss=True.",
                    UserWarning,
                    stacklevel=2,
                )
                self._masked_loss_warned = True

        return loss.mean(dim=tuple(range(1, loss.dim())))

    def _weighted_mean_loss(self, per_sample_loss: torch.Tensor, batch: Dict) -> torch.Tensor:
        """Apply sample weights without changing the average scale."""
        combined_weights: Optional[torch.Tensor] = None
        for key in ("caption_weights", "geometry_weights"):
            weights = batch.get(key)
            if not isinstance(weights, torch.Tensor):
                continue
            weights = weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).view(-1)
            if weights.numel() != per_sample_loss.numel():
                continue
            combined_weights = weights if combined_weights is None else (combined_weights * weights)

        if combined_weights is None:
            return per_sample_loss.mean()
        return (per_sample_loss * combined_weights).sum() / combined_weights.sum().clamp_min(1e-6)

    def _compute_repa_loss(self, batch: Dict, prompt_embeds: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.repa_enabled:
            return None

        target = batch.get("repa_target_features")
        if not isinstance(target, torch.Tensor):
            target = prompt_embeds.get("encoder_hidden_states")
        if not isinstance(target, torch.Tensor):
            if not self._repa_warned:
                logger.warning("REPA enabled but no repa_target_features or encoder_hidden_states are available; skipping REPA loss.")
                self._repa_warned = True
            return None

        hidden = None
        if self.repa_capture is not None and self.repa_capture.features:
            hidden = next(reversed(self.repa_capture.features.values()))
        if hidden is None:
            hidden = prompt_embeds.get("encoder_hidden_states")
        if not isinstance(hidden, torch.Tensor):
            return None

        weight = self.repa_loss_weight
        if self.softrepa_enabled:
            sigma = batch.get("flow_sigmas") or batch.get("sigmas")
            if not isinstance(sigma, torch.Tensor):
                sigma = batch.get("repa_sigmas")
            from .repa import SoftREPAConfig, softrepa_weight
            weight = softrepa_weight(
                self.global_step,
                max(int(getattr(self, "total_steps", 0) or 0), 1),
                sigma if isinstance(sigma, torch.Tensor) else None,
                SoftREPAConfig(
                    enabled=True,
                    schedule=self.softrepa_schedule,
                    min_weight=self.softrepa_min_weight,
                    max_weight=self.softrepa_max_weight,
                    sigma_window=(self.softrepa_sigma_min, self.softrepa_sigma_max),
                ),
            )
        if weight <= 0.0:
            return None

        from .repa import REPALossConfig, repa_alignment_loss
        cfg = REPALossConfig(
            loss_type=self.repa_loss_type,
            weight=weight,
            projection_dim=self.repa_projection_dim,
            stop_grad_target=self.repa_stop_grad_target,
        )
        target = target.to(device=hidden.device, dtype=hidden.dtype)
        return repa_alignment_loss(hidden, target, cfg, self.repa_projector)

    def _repa_active(self, batch: Dict) -> bool:
        if not getattr(self, "repa_enabled", False):
            return False
        if isinstance(batch, dict) and batch.get("repa_target_features") is not None:
            return True
        capture = getattr(self, "repa_capture", None)
        return bool(getattr(capture, "features", None))

    def _compute_sra2_haste_loss(self, batch: Dict, prompt_embeds: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Default-off SRA2/HASTE VAE self-representation alignment auxiliary loss.

        Returns ``None`` (no contribution) unless the feature is enabled, a DiT
        hidden capture is available, and a VAE/latent feature target is present in
        the batch -- so the disabled path adds nothing and stays parity-safe. Any
        failure degrades to ``None`` rather than breaking the training step.
        """
        if not getattr(self, "sra2_haste_enabled", False):
            return None
        capture = getattr(self, "sra2_haste_capture", None)
        hidden = None
        if capture is not None and getattr(capture, "features", None):
            hidden = next(reversed(capture.features.values()))
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return None
        target = None
        if isinstance(batch, dict):
            for key in ("sra2_vae_features", "vae_features", "latents", "model_input"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    target = value
                    break
        if not isinstance(target, torch.Tensor):
            if not getattr(self, "_sra2_haste_warned", False):
                logger.warning("SRA2/HASTE alignment enabled but no VAE/latent features in batch; skipping alignment loss.")
                self._sra2_haste_warned = True
            return None
        try:
            from .sra2_haste_alignment_facade import SRA2HasteAlignmentPolicy, sra2_haste_alignment_loss

            policy = SRA2HasteAlignmentPolicy(**{**self.sra2_haste_policy, "enabled": True})
            loss, profile = sra2_haste_alignment_loss(
                hidden,
                target,
                policy,
                step=int(getattr(self, "global_step", 0) or 0),
                total_steps=max(int(getattr(self, "total_steps", 0) or 0), 1),
            )
        except Exception as exc:  # never break the training step on an aux loss
            if not getattr(self, "_sra2_haste_warned", False):
                logger.warning(f"SRA2/HASTE alignment loss failed ({exc}); skipping.")
                self._sra2_haste_warned = True
            return None
        if not bool(profile.get("active")):
            return None
        return loss

    def _micro_batch_hook_context(self, batch: Dict, accumulation_steps: int, sync_gradients: bool) -> Dict[str, Any]:
        filenames = batch.get("filenames")
        if isinstance(filenames, (list, tuple)):
            micro_batch_size = len(filenames)
        else:
            for value in batch.values():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    micro_batch_size = int(value.shape[0])
                    break
            else:
                micro_batch_size = 1
        return {
            "training_type": self.training_type,
            "global_step": int(self.global_step),
            "micro_batch_index": int(getattr(self, "_current_micro_batch_index", 1) or 1),
            "micro_batch_count": int(getattr(self, "_current_micro_batch_count", accumulation_steps) or accumulation_steps),
            "micro_batch_size": max(int(micro_batch_size or 1), 1),
            "gradient_accumulation_steps": max(int(accumulation_steps or 1), 1),
            "sync_gradients": bool(sync_gradients),
            "source": "training_loop",
        }

    def _optimizer_hook_context(self, loss: float) -> Dict[str, Any]:
        learning_rates = [float(group.get("lr", 0.0) or 0.0) for group in getattr(self.optimizer, "param_groups", [])]
        return {
            "training_type": self.training_type,
            "global_step": int(self.global_step),
            "current_loss": float(loss),
            "optimizer_type": type(self.optimizer).__name__ if self.optimizer is not None else "",
            "scheduler_type": type(self.lr_scheduler).__name__ if self.lr_scheduler is not None else "",
            "learning_rates": learning_rates,
            "max_grad_norm": float(self.max_grad_norm or 0.0),
            "gradient_accumulation_steps": max(int(self.gradient_accumulation_steps or 1), 1),
            "sync_gradients": True,
            "source": "training_loop",
        }

    def _capture_optimizer_step_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
        if torch.cuda.is_available():
            try:
                state["cuda_all"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
            except Exception:
                pass
        return state

    def _restore_optimizer_step_rng_state(self, state: Dict[str, Any]) -> None:
        cpu_state = state.get("cpu") if isinstance(state, dict) else None
        if isinstance(cpu_state, torch.Tensor):
            torch.set_rng_state(cpu_state)
        cuda_all = state.get("cuda_all") if isinstance(state, dict) else None
        if cuda_all is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_all)
            except Exception:
                pass

    def _make_optimizer_step_closure(
        self,
        microbatches: List[Dict[str, Any]],
        accumulation_steps: int,
    ) -> Callable[[], torch.Tensor]:
        if not microbatches:
            raise RuntimeError("Closure-required optimizer reached optimizer.step() without accumulated microbatches.")

        def closure() -> torch.Tensor:
            if self.optimizer is None:
                raise RuntimeError("Closure-required optimizer step called without an optimizer.")
            self.optimizer.zero_grad(set_to_none=True)
            closure_entry_rng_state = self._capture_optimizer_step_rng_state()
            previous_active = bool(getattr(self, "_optimizer_step_closure_active", False))
            previous_index = getattr(self, "_current_micro_batch_index", 1)
            previous_count = getattr(self, "_current_micro_batch_count", accumulation_steps)
            previous_sync = getattr(self, "_current_sync_gradients", True)
            previous_group_start = getattr(self, "_current_accumulation_group_start", True)
            losses: List[torch.Tensor] = []
            self._optimizer_step_closure_active = True
            try:
                for item in microbatches:
                    self._restore_optimizer_step_rng_state(item.get("rng_state", {}))
                    self._current_micro_batch_index = int(item.get("micro_batch_index", 1) or 1)
                    self._current_micro_batch_count = int(item.get("micro_batch_count", accumulation_steps) or accumulation_steps)
                    self._current_sync_gradients = bool(item.get("sync_gradients", True))
                    self._current_accumulation_group_start = bool(item.get("accumulation_group_start", False))
                    try:
                        loss_value = self._train_step_impl(
                            item["batch"],
                            accumulation_steps=accumulation_steps,
                            do_backward=True,
                            return_loss_tensor=True,
                        )
                    except Exception as exc:
                        self._finish_pipeline_trace_failed(exc)
                        raise
                    if isinstance(loss_value, torch.Tensor):
                        losses.append(loss_value.detach().float())
                    else:
                        losses.append(torch.tensor(float(loss_value), device=self._runtime_device))
                trainable_params = self._get_trainable_params()
                if trainable_params and float(self.max_grad_norm or 0.0) > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
            finally:
                self._optimizer_step_closure_active = previous_active
                self._current_micro_batch_index = previous_index
                self._current_micro_batch_count = previous_count
                self._current_sync_gradients = previous_sync
                self._current_accumulation_group_start = previous_group_start
                self._restore_optimizer_step_rng_state(closure_entry_rng_state)

            base_device = losses[0].device if losses else self._runtime_device
            if not losses:
                return torch.zeros((), device=base_device)
            return torch.stack([loss.to(device=base_device) for loss in losses]).mean()

        return closure


    @staticmethod
    def _mutation_from_report(report: Dict[str, Any] | None) -> tuple[float, float]:
        if not isinstance(report, dict):
            return 1.0, 0.0
        payload = report.get("result_payload", {})
        mutation = payload.get("mutation", {}) if isinstance(payload, dict) else {}
        if not isinstance(mutation, dict):
            return 1.0, 0.0
        try:
            scale = float(mutation.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        try:
            bias = float(mutation.get("bias", 0.0))
        except (TypeError, ValueError):
            bias = 0.0
        if not torch.isfinite(torch.tensor(scale)):
            scale = 1.0
        if not torch.isfinite(torch.tensor(bias)):
            bias = 0.0
        return scale, bias

    def train_step(
        self,
        batch: Dict,
        accumulation_steps: Optional[int] = None,
        return_loss_tensor: bool = False,
    ) -> float | torch.Tensor:
        """单步训练"""
        with ExitStack() as stack:
            if self.adapter_cpu_residency is not None:
                stack.enter_context(self.adapter_cpu_residency.step_context())
            if self._module_offload_manager is not None:
                stack.enter_context(self._module_offload_manager.step_context())
            try:
                return self._train_step_impl(
                    batch,
                    accumulation_steps,
                    do_backward=True,
                    return_loss_tensor=return_loss_tensor,
                )
            except Exception as exc:
                self._finish_pipeline_trace_failed(exc)
                raise
            finally:
                self._refresh_module_offload_stats()

    def validation_step(self, batch: Dict) -> float:
        """Compute validation loss without backward or optimizer-side training hooks."""
        with ExitStack() as stack:
            if self.adapter_cpu_residency is not None:
                stack.enter_context(self.adapter_cpu_residency.step_context())
            if self._module_offload_manager is not None:
                stack.enter_context(self._module_offload_manager.step_context())
            try:
                return self._train_step_impl(batch, accumulation_steps=1, do_backward=False)
            except Exception as exc:
                self._finish_pipeline_trace_failed(exc)
                raise
            finally:
                self._refresh_module_offload_stats()

    def _finish_pipeline_trace_failed(self, exc: Exception) -> None:
        trace = getattr(self, "_pipeline_trace", None)
        if trace is None:
            return
        try:
            self._last_pipeline_trace = trace.finish(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass

    def _train_step_impl(
        self,
        batch: Dict,
        accumulation_steps: Optional[int] = None,
        do_backward: bool = True,
        return_loss_tensor: bool = False,
    ) -> float | torch.Tensor:
        """Internal training step implementation."""
        accumulation_steps = max(int(accumulation_steps or self.gradient_accumulation_steps), 1)
        _newbie_compute_substage_enabled = str(self._model_arch or "").strip().lower() == "newbie"

        def _newbie_compute_substage_start(*, sync_cuda: bool = False) -> Any:
            if not _newbie_compute_substage_enabled:
                return None
            return self._step_phase_profiler.start() if sync_cuda else self._step_phase_profiler.start_cpu()

        def _record_newbie_compute_substage(label: str, started: Any, *, sync_cuda: bool = False) -> None:
            if started is None:
                return
            phase_label = f"train_step_compute_substage.newbie.{label}"
            if sync_cuda:
                self._step_phase_profiler.record(phase_label, started)
            else:
                self._step_phase_profiler.record_cpu(phase_label, started)

        batch_contract_execution = run_lulynx_batch_contract_stage_handler(
            batch=batch,
            model_arch=self._model_arch,
            trace=self._pipeline_trace,
            previous_training_data_pipeline_report=getattr(self, "_training_data_pipeline_runtime_report", None),
            accumulation_steps=accumulation_steps,
            do_backward=do_backward,
        )
        batch_stage_plan = batch_contract_execution.batch_stage_plan
        self._training_data_pipeline_runtime_report = batch_contract_execution.training_data_pipeline_report
        self._training_step_orchestrator_runtime_profile = batch_contract_execution.orchestrator_runtime
        sync_gradients = bool(getattr(self, "_current_sync_gradients", True))
        hook_context = self._micro_batch_hook_context(batch, accumulation_steps, sync_gradients)
        loss_scalars = _LossScalarCache()
        _vram_diag_step = False
        _vram_forward_mb = 0.0
        _vram_backward_mb = 0.0
        _vram_optimizer_mb = 0.0
        _vram_forward_diag: Dict[str, float] = {}
        _vram_backward_diag: Dict[str, float] = {}
        _train_prepare_phase_start = self._step_phase_profiler.start()
        self._last_peak_vram_stages = None
        self._last_peak_vram_diagnostics = None
        _entropy_probe_step = False
        cached_native = batch_stage_plan.cached_native
        images = None if cached_native else self._profiled_to(
            batch["images"],
            label="images",
            dtype=self.dtype,
        )
        captions = batch["captions"]

        wrapper_lisa_scheduler = getattr(self.lulynx_wrapper, "lisa_scheduler", None) if self.lulynx_wrapper else None
        step_closure_active = bool(getattr(self, "_optimizer_step_closure_active", False))

        # T-LoRA temporal rank schedule: advance the effective adapter rank for the
        # current training step before the forward pass. Gated on the injector's
        # ``tlora_enabled`` flag so plain-LoRA runs pay only one bool check and the
        # injected layers stay bitwise-identical (set_global_step is never invoked).
        if self.lora_injector is not None and getattr(self.lora_injector, "tlora_enabled", False):
            self.lora_injector.set_global_step(int(self.global_step))

        if do_backward and not step_closure_active and self.lisa_scheduler and wrapper_lisa_scheduler is None:
            self.lisa_scheduler.step()

        if do_backward and not step_closure_active and self.lulynx_wrapper:
            self.lulynx_wrapper.pre_step(model=self.unet, step=self.global_step, network=self.lora_injector)

        transfer_conditioning_execution = run_lulynx_transfer_conditioning_stage_handler(
            batch=batch,
            batch_stage_plan=batch_stage_plan,
            model_arch=self._model_arch,
            trace=self._pipeline_trace,
            target_dtype=self.dtype,
            images=images,
            captions=captions,
            qwen3_encoder_available=self.qwen3_encoder is not None,
            do_backward=do_backward,
            te_dropout=self.te_dropout,
            clip_l_dropout_rate=self.clip_l_dropout_rate,
            clip_g_dropout_rate=self.clip_g_dropout_rate,
            t5_dropout_rate=self.t5_dropout_rate,
            profiled_to=self._profiled_to,
            encode_latents_with_vae=self._encode_latents_with_vae,
            encode_prompt=self._encode_prompt,
            encode_qwen3=self._encode_qwen3,
            apply_text_encoder_dropout=self._apply_text_encoder_dropout,
        )
        latents = transfer_conditioning_execution.latents
        padding_mask = transfer_conditioning_execution.padding_mask
        prompt_embeds = transfer_conditioning_execution.prompt_embeds
        self._training_step_orchestrator_runtime_profile = transfer_conditioning_execution.orchestrator_runtime

        noise_timestep_execution = run_lulynx_noise_timestep_stage_handler(
            latents=latents,
            model_arch=self._model_arch,
            trace=self._pipeline_trace,
            device=self.device,
            flow_model=getattr(self, "flow_model", ""),
            noise_scheduler=self.noise_scheduler,
            v_parameterization=self.v_parameterization,
            optimal_noise_enabled=self.optimal_noise_enabled,
            optimal_noise_candidates=self.optimal_noise_candidates,
            multires_noise_iterations=self.multires_noise_iterations,
            multires_noise_discount=self.multires_noise_discount,
            spectral_noise_blend=self.spectral_noise_blend,
            spectral_noise_sigma=self.spectral_noise_sigma,
            noise_offset=self.noise_offset,
            adaptive_noise_scale=self.adaptive_noise_scale,
            noise_offset_random_strength=self.noise_offset_random_strength,
            perlin_noise_offset_enabled=self.perlin_noise_offset_enabled,
            perlin_noise_offset_strength=self.perlin_noise_offset_strength,
            perlin_noise_offset_scale=self.perlin_noise_offset_scale,
            flow_use_ot=self.flow_use_ot,
            immiscible_enabled=self.immiscible_diffusion_enabled,
            immiscible_metric=self.immiscible_metric,
            ddpm_timestep_sampling=self.ddpm_timestep_sampling,
            anima_timestep_sampling=self.anima_timestep_sampling,
            anima_sigmoid_scale=self.anima_sigmoid_scale,
            anima_discrete_flow_shift=self.anima_discrete_flow_shift,
            anima_weighting_scheme=self.anima_weighting_scheme,
            anima_model_prediction_type=self.anima_model_prediction_type,
            anima_faithful_forward=self.anima_faithful_forward,
            sdxl_timestep_sampling=self.sdxl_timestep_sampling,
            sdxl_sigmoid_scale=self.sdxl_sigmoid_scale,
            sdxl_flow_shift=self.sdxl_flow_shift,
            sdxl_flow_weighting_scheme=self.sdxl_flow_weighting_scheme,
            sdxl_model_prediction_type=self.sdxl_model_prediction_type,
            flow_logit_mean=self.flow_logit_mean,
            flow_logit_std=self.flow_logit_std,
            ip_noise_gamma=self.ip_noise_gamma,
            ip_noise_gamma_random_strength=self.ip_noise_gamma_random_strength,
            sample_strength=self._sample_strength,
            velocity_target=self._velocity_target,
            log_debug=logger.debug,
        )
        noise = noise_timestep_execution.noise
        noisy_latents = noise_timestep_execution.noisy_latents
        target = noise_timestep_execution.target
        timesteps = noise_timestep_execution.timesteps
        batch_size = noise_timestep_execution.batch_size
        uses_flow_matching = noise_timestep_execution.uses_flow_matching
        uses_sdxl_flow = noise_timestep_execution.uses_sdxl_flow
        self._sdxl_flow_sigmas = noise_timestep_execution.sdxl_flow_sigmas
        self._sdxl_flow_weighting = noise_timestep_execution.sdxl_flow_weighting
        self._training_step_orchestrator_runtime_profile = noise_timestep_execution.orchestrator_runtime
        
        # Check for incompatible features with safe_fallback (one-shot)
        if not getattr(self, "_safe_fallback_compat_checked", False):
            self._safe_fallback_compat_checked = True
            if self.safe_fallback and self._block_offloader is not None:
                logger.warning(
                    "[SafeFallback] blocks_to_swap is incompatible with safe_fallback. "
                    "Disabling BlockSwap for this run."
                )
                self._block_offloader = None

            if self.safe_fallback and self._torch_compile_active:
                logger.warning(
                    "[SafeFallback] torch.compile is incompatible with safe_fallback. "
                    "Disabling torch.compile for this run."
                )
                self._torch_compile_active = False

        if not hasattr(self, "_cudagraph_scope_cached"):
            self._cudagraph_scope_cached = (
                str(getattr(self, "_anima_compile_scope", "") or "") == "full_cudagraph"
            )
        _should_try_cudagraph = (
            self._cudagraph_scope_cached
            and self._cudagraph_eligible()
            and not self.cpu_offload_checkpointing
        )
        _forward_input_substage_start = _newbie_compute_substage_start()
        forward_input_execution = run_lulynx_forward_input_stage_handler(
            batch=batch,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            padding_mask=padding_mask,
            batch_size=batch_size,
            model_arch=self._model_arch,
            cached_native=cached_native,
            trace=self._pipeline_trace,
            device=self.device,
            target_dtype=self.dtype,
            easy_control=self.easy_control,
            ip_adapter=self.ip_adapter,
            easycontrol_v2_adapter=self.easycontrol_v2_adapter,
            easycontrol_v2_clean_latents=latents,
            cudagraph_active=self._cudagraph_active,
            cudagraph_requested=_should_try_cudagraph,
            cpu_offload_checkpointing=self.cpu_offload_checkpointing,
            offloaded_checkpoint_context_available=self._offloaded_checkpoint_ctx is not None,
            profiled_to=self._profiled_to,
            get_timestep_embedding=self._get_timestep_embedding,
            anima_faithful_forward=self.anima_faithful_forward,
            anima_llm_adapter=getattr(self.unet, "anima_llm_adapter", None),
        )
        _record_newbie_compute_substage("forward_input_prepare", _forward_input_substage_start)
        noisy_latents = forward_input_execution.noisy_latents
        prompt_embeds = forward_input_execution.prompt_embeds
        unet_kwargs = forward_input_execution.unet_kwargs
        self._training_step_orchestrator_runtime_profile = forward_input_execution.orchestrator_runtime

        self._step_phase_profiler.record("train_batch_prepare", _train_prepare_phase_start)
        _forward_phase_start = self._step_phase_profiler.start()
        _newbie_module_timing_profiler = self._start_newbie_module_timing_profile()
        with self._compute_reducer_context():
            _forward_model_substage_start = _newbie_compute_substage_start(sync_cuda=True)
            forward_execution = run_lulynx_forward_execution_stage_handler(
                owner=self,
                unet_kwargs=unet_kwargs,
                hook_context=hook_context,
                do_backward=do_backward,
                should_try_cudagraph=_should_try_cudagraph,
                multi_batch_execution_strategy=batch_contract_execution.execution_strategy,
                multi_batch_execution_strategy_gate=batch_contract_execution.execution_strategy_gate,
                logger=logger,
            )
            _record_newbie_compute_substage(
                "forward_model_execution",
                _forward_model_substage_start,
                sync_cuda=True,
            )
        noise_pred = forward_execution.noise_pred
        _vram_diag_step = forward_execution.vram_diag_step
        _entropy_probe_step = forward_execution.entropy_probe_step
        self._training_step_orchestrator_runtime_profile = forward_execution.orchestrator_runtime
        self._step_phase_profiler.record("forward_total", _forward_phase_start)
        _loss_phase_start = self._step_phase_profiler.start()
        _loss_plan_substage_start = _newbie_compute_substage_start()
        loss_plan_execution = run_lulynx_loss_plan_stage_handler(
            batch=batch,
            model_arch=self._model_arch,
            batch_size=batch_size,
            loss_type=self.loss_type,
            uses_flow_matching=uses_flow_matching,
            uses_sdxl_flow=uses_sdxl_flow,
            sdxl_flow_sigmas_available=self._sdxl_flow_sigmas is not None,
            v_parameterization=self.v_parameterization,
            masked_loss=self.masked_loss,
            alpha_mask=self.alpha_mask,
            strict_masked_loss=self.strict_masked_loss,
            debiased_estimation=self.debiased_estimation,
            snr_gamma=self.snr_gamma,
            adaptive_loss_weighter_available=self.adaptive_loss_weighter is not None,
            wavelet_loss_enabled=self.wavelet_loss_enabled,
            pattern_loss_enabled=self.pattern_loss_enabled,
            prior_loss_weight=self.prior_loss_weight,
            reg_dataloader_available=self.reg_dataloader is not None,
            lulynx_wrapper_available=self.lulynx_wrapper is not None,
            repa_active=self._repa_active(batch),
            dop_active=self.dop is not None and self.dop.should_compute(self.global_step),
            b_tier_runtime_available=self.b_tier_runtime is not None,
            do_backward=do_backward,
            trace=self._pipeline_trace,
        )
        _record_newbie_compute_substage("loss_plan", _loss_plan_substage_start)
        self._training_step_orchestrator_runtime_profile = loss_plan_execution.orchestrator_runtime
        
        _lt = self._loss_tracker
        _loss_execution_substage_start = _newbie_compute_substage_start(sync_cuda=True)
        loss_execution = run_lulynx_loss_execution_stage_handler(
            owner=self,
            batch=batch,
            prompt_embeds=prompt_embeds,
            noise_pred=noise_pred,
            target=target,
            timesteps=timesteps,
            padding_mask=padding_mask,
            noisy_latents=noisy_latents,
            uses_flow_matching=uses_flow_matching,
            uses_sdxl_flow=uses_sdxl_flow,
            do_backward=do_backward,
            loss_scalars=loss_scalars,
            logger=logger,
        )
        _record_newbie_compute_substage("loss_execution", _loss_execution_substage_start, sync_cuda=True)
        loss = loss_execution.loss
        _lt_val = loss_execution.loss_tracker_value
        self._training_step_orchestrator_runtime_profile = loss_execution.orchestrator_runtime

        self._step_phase_profiler.record("loss_total", _loss_phase_start)
        raw_loss = loss
        if do_backward:
            _backward_phase_start = self._step_phase_profiler.start()
            _backward_plugin_substage_start = _newbie_compute_substage_start()
            plugin_hook_execution = run_lulynx_loss_plugin_hook_stage_handler(
                hook_context=hook_context,
                loss=loss,
                loss_tracker_value=_lt_val,
                accumulation_steps=accumulation_steps,
                loss_scalars=loss_scalars,
                loss_tracker=_lt,
                mutation_from_report=self._mutation_from_report,
            )
            _record_newbie_compute_substage("backward_plugin_hook", _backward_plugin_substage_start)
            loss = plugin_hook_execution.loss
            raw_loss = plugin_hook_execution.raw_loss
            raw_loss_value = plugin_hook_execution.raw_loss_value
            loss_scale = plugin_hook_execution.loss_scale
            _lt_val = plugin_hook_execution.loss_tracker_value
            emit_after_backward_event = plugin_hook_execution.emit_after_backward_event
            self._training_step_orchestrator_runtime_profile = plugin_hook_execution.orchestrator_runtime
            # 梯度累积：按当前有效累积步数缩放（支持 epoch 末尾不足一组的情况）
            loss = loss / accumulation_steps
            # ── Hessian trace deep diagnostic (before backward) ──
            _hessian_trace_val = None
            _hessian_layers = None
            _hessian_sample_step = (
                self._deep_diagnostics
                and self._hessian_estimator is not None
                and self.global_step % self._hessian_interval == 0
            )
            if _hessian_sample_step:
                try:
                    _h_named = [(n, p) for n, p in self.unet.named_parameters() if p.requires_grad]
                    if _h_named:
                        _h_lw = self._hessian_estimator.estimate_per_layer(loss, _h_named)
                        _hessian_trace_val = _h_lw.total_trace
                        _hessian_layers = _h_lw.layer_traces
                except Exception:
                    pass
            # ── Peak VRAM: capture forward, reset for backward ──
            if _vram_diag_step:
                _vram_forward_diag = self._cuda_memory_snapshot()
                _vram_forward_mb = float(_vram_forward_diag.get("peak_reserved_mb", 0.0) or 0.0)
                torch.cuda.reset_peak_memory_stats()
            _backward_prepare_substage_start = _newbie_compute_substage_start()
            self._turbocore_direct_grad_lifecycle_report = {}
            self._turbocore_native_update_direct_grad_lifecycle_report = {}
            trainable_params_for_direct_grad = self._get_trainable_params()
            self._turbocore_native_update_direct_grad_lifecycle_report = (
                self._prepare_turbocore_native_update_direct_grad_executor_before_backward(
                    trainable_params_for_direct_grad
                )
            )
            if self._turbocore_update_shadow.enabled and bool(self._turbocore_update_shadow.config.direct_grad):
                try:
                    self._turbocore_direct_grad_lifecycle_report = self._turbocore_update_shadow.prepare_before_backward(
                        trainable_params_for_direct_grad,
                        optimizer=self.optimizer,
                        max_grad_norm=self.max_grad_norm,
                        step=self.global_step,
                        reset_owner_grad=bool(getattr(self, "_current_accumulation_group_start", True)),
                    )
                except Exception as exc:
                    self._turbocore_direct_grad_lifecycle_report = {
                        "schema_version": 1,
                        "stage": "before_backward",
                        "mode": self._turbocore_update_shadow.config.mode,
                        "error": f"{type(exc).__name__}: {exc}",
                        "training_path_enabled": False,
                    }
                    logger.debug("TurboCore direct-grad lifecycle prepare skipped: %s", exc)
            _gr_ctx = None
            if self._gradient_release_manager is not None and self._gradient_release_manager.mode == "during_backward":
                _gr_ctx = self._gradient_release_manager.step_context(
                    is_accumulation_boundary=bool(getattr(self, "_current_sync_gradients", True)),
                )
                _gr_ctx.__enter__()
            _optimizer_needs_step_closure = optimizer_requires_step_closure(self.optimizer)
            _optimizer_deferred_step_closure = (
                _optimizer_needs_step_closure
                and not step_closure_active
                and not optimizer_step_closure_requires_initial_backward(self.optimizer)
            )
            _optimizer_used_fused_backward = False
            if not _optimizer_deferred_step_closure:
                _optimizer_used_fused_backward = run_optimizer_fused_backward(
                    self.optimizer,
                    loss,
                    float(self.optimizer.param_groups[0].get("lr", 0.0)) if self.optimizer and self.optimizer.param_groups else 0.0,
                )
            _record_newbie_compute_substage(
                "backward_pre_autograd_prepare",
                _backward_prepare_substage_start,
            )
            _backward_plan_substage_start = _newbie_compute_substage_start()
            backward_plan_execution = run_lulynx_backward_plan_stage_handler(
                do_backward=do_backward,
                sync_gradients=sync_gradients,
                accumulation_steps=accumulation_steps,
                uses_step_closure=bool(_optimizer_deferred_step_closure),
                uses_fused_backward=bool(_optimizer_used_fused_backward),
                gradient_release_mode=str(getattr(self._gradient_release_manager, "mode", "") or "")
                if self._gradient_release_manager is not None
                else "",
                create_graph_backward=bool(optimizer_requires_create_graph_backward(self.optimizer)),
                trace=self._pipeline_trace,
            )
            _record_newbie_compute_substage("backward_plan", _backward_plan_substage_start)
            self._training_step_orchestrator_runtime_profile = backward_plan_execution.orchestrator_runtime
            _backward_autograd_substage_start = _newbie_compute_substage_start(sync_cuda=True)
            _newbie_backward_op_profile_enabled = self._should_profile_newbie_backward_ops()
            backward_execution = run_lulynx_backward_execution_stage_handler(
                loss=loss,
                gradient_release_context=_gr_ctx,
                optimizer_used_fused_backward=bool(_optimizer_used_fused_backward),
                optimizer_deferred_step_closure=bool(_optimizer_deferred_step_closure),
                create_graph_backward=bool(optimizer_requires_create_graph_backward(self.optimizer)),
                step_phase_profiler=self._step_phase_profiler,
                substage_label_prefix="train_step_compute_substage.newbie"
                if _newbie_compute_substage_enabled
                else "",
                backward_op_profile_enabled=_newbie_backward_op_profile_enabled,
                backward_op_profile_top_k=max(int(getattr(self, "newbie_backward_op_profile_top_k", 12) or 12), 1),
                backward_op_profile_record_shapes=bool(
                    getattr(self, "newbie_backward_op_profile_record_shapes", False)
                ),
                backward_op_profile_sink=self._record_newbie_backward_op_profile
                if _newbie_backward_op_profile_enabled
                else None,
            )
            if _newbie_module_timing_profiler is not None:
                try:
                    self._record_newbie_module_timing_profile(
                        _newbie_module_timing_profiler.snapshot(step=int(getattr(self, "global_step", 0) or 0))
                    )
                finally:
                    _newbie_module_timing_profiler.detach()
                    _newbie_module_timing_profiler = None
            _record_newbie_compute_substage(
                "backward_autograd_execution",
                _backward_autograd_substage_start,
                sync_cuda=True,
            )
            self._training_step_orchestrator_runtime_profile = backward_execution.orchestrator_runtime
            # EasyControl v2: the condition was set per-step in the input handler and
            # read by the patched blocks. Clear it only NOW (after backward), because
            # under use_reentrant=False block checkpointing the blocks are recomputed
            # during backward and must see the same condition as the forward pass.
            # No-op when disabled. The next step's handler re-sets or clears it.
            if self.easycontrol_v2_adapter is not None:
                self.easycontrol_v2_adapter.clear_cond()
            _backward_postprocess_substage_start = _newbie_compute_substage_start()
            # ── Peak VRAM: capture backward, reset for optimizer ──
            if _vram_diag_step:
                _vram_backward_diag = self._cuda_memory_snapshot()
                _vram_backward_mb = float(_vram_backward_diag.get("peak_reserved_mb", 0.0) or 0.0)
                torch.cuda.reset_peak_memory_stats()
            # Stochastic gradient accumulation: round bf16/fp16 gradients to prevent quantization bias
            if self.stochastic_grad_accumulation and self.gradient_accumulation_steps > 1:
                if not getattr(self, "_current_sync_gradients", True):
                    from .stochastic_rounding import stochastic_round_
                    for _pg in self.optimizer.param_groups:
                        for _p in _pg["params"]:
                            if _p.grad is not None and _p.grad.dtype in (torch.bfloat16, torch.float16):
                                stochastic_round_(_p.grad)
            if self.pcgrad_enabled and not _optimizer_needs_step_closure:
                self._capture_pcgrad_microbatch(accumulation_steps)
            if emit_after_backward_event is not None and not _optimizer_deferred_step_closure:
                emit_after_backward_event(
                    **hook_context,
                    loss_value=raw_loss_value,
                    loss_scale=loss_scale,
                    backward_loss=raw_loss_value * loss_scale,
                    weighted_loss=raw_loss_value,
                )
            if self.pcgrad_enabled and not _optimizer_needs_step_closure and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)

            if (
                _optimizer_needs_step_closure
                and not step_closure_active
                and not optimizer_step_closure_requires_initial_backward(self.optimizer)
            ):
                self.optimizer.zero_grad(set_to_none=True)
            _record_newbie_compute_substage("backward_postprocess", _backward_postprocess_substage_start)
            self._step_phase_profiler.record("backward_total", _backward_phase_start)

        if _newbie_module_timing_profiler is not None:
            _newbie_module_timing_profiler.detach()

        try:
            # 必须清理捕获的特征，防止显存泄漏
            if self.lulynx_wrapper:
                self.lulynx_wrapper._current_features.clear()
            if self.repa_capture is not None:
                self.repa_capture.clear()
            if getattr(self, "sra2_haste_capture", None) is not None:
                self.sra2_haste_capture.clear()
            if self.b_tier_runtime is not None:
                self.b_tier_runtime.clear()
        except Exception:
            pass

        if _vram_diag_step:
            self._last_peak_vram_stages = {
                "forward_mb": round(float(_vram_forward_mb), 1),
                "backward_mb": round(float(_vram_backward_mb), 1),
            }
            self._last_peak_vram_diagnostics = self._build_peak_vram_diagnostics(
                {
                    "forward": _vram_forward_diag,
                    "backward": _vram_backward_diag,
                }
            )

        if return_loss_tensor:
            self._last_pipeline_trace = self._pipeline_trace.finish(status="completed")
            return raw_loss.detach()
        result = loss_scalars.get(raw_loss)
        self._last_pipeline_trace = self._pipeline_trace.finish(status="completed")
        return result
    
    def _compute_snr(self, timesteps):
        """计算 Signal-to-Noise Ratio (含缓存)"""
        # 简单缓存策略: 假设 noise_scheduler 不变
        if not hasattr(self, "_snr_cache"):
            self._snr_cache = {}
            
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        
        # Batch 处理，不适合逐个缓存，直接计算其实很快，主要开销在 to(device)
        # 如果 timesteps 是 scalar，可以缓存
        # 这里仅对 alphas_cumprod 搬运做优化? 
        # 原代码的问题可能是重复搬运
        
        alpha = sqrt_alphas_cumprod[timesteps]
        sigma = sqrt_one_minus_alphas_cumprod[timesteps]
        
        return (alpha / sigma) ** 2
    
    def train_epoch(self, dataloader, epoch: int) -> Dict:
        """训练一个 epoch"""
        self.current_epoch = epoch
        total_loss = 0.0
        num_steps = 0
        self._clear_pcgrad_pending_grads()
        self._pcgrad_last_stats = {}

        total_microbatches = len(dataloader)
        current_group_target = max(min(int(self.gradient_accumulation_steps), total_microbatches), 1)
        microbatches_in_group = 0
        gradient_accumulation_mode = self._normalize_gradient_accumulation_mode(
            getattr(self, "gradient_accumulation_mode", "fast")
        )
        fast_accumulation = gradient_accumulation_mode == "fast"
        pending_loss_total: Optional[torch.Tensor] = None
        pending_loss_count = 0
        pending_filenames: List[str] = []
        closure_microbatches: List[Dict[str, Any]] = []
        accumulation_wall_start: Optional[float] = None
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=False,
            **_tqdm_kwargs(),
        )
        _data_wait_started = time.perf_counter()

        self._maybe_save_safe_state()
        
        for step, batch in enumerate(progress_bar):
            _vram_diag_step = False
            _vram_forward_mb = 0.0
            _vram_backward_mb = 0.0
            _vram_optimizer_mb = 0.0
            _entropy_probe_step = (
                self._advanced_monitoring
                and self.global_step % self._attn_entropy_interval == 0
            )
            _act_drift_step = (
                self._advanced_monitoring
                and self.global_step % self._act_drift_interval == 0
            )
            _hessian_trace_val = None
            _hessian_layers = None
            iteration_guard_execution = run_lulynx_epoch_iteration_guard_stage_handler(
                should_stop=bool(self._should_stop),
                total_steps=self.total_steps,
                global_step=self.global_step,
                skip_until_initial_step=bool(self.skip_until_initial_step),
                initial_step_target=self.initial_step_target,
                lora_injector=self.lora_injector,
                progress_bar=progress_bar,
                callback=self.on_step_end,
                epoch=epoch,
            )
            self.global_step = iteration_guard_execution.global_step
            if iteration_guard_execution.completed_by_step_limit:
                self.completed_by_step_limit = True
            if iteration_guard_execution.should_stop:
                self._should_stop = True
            self._training_step_orchestrator_runtime_profile = (
                iteration_guard_execution.orchestrator_runtime
            )
            if iteration_guard_execution.should_break_epoch:
                break
            if iteration_guard_execution.should_continue_epoch:
                _data_wait_started = time.perf_counter()
                continue

            microbatch_group_execution = run_lulynx_microbatch_group_stage_handler(
                batch=batch,
                step=step,
                total_microbatches=total_microbatches,
                current_group_target=current_group_target,
                microbatches_in_group=microbatches_in_group,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                dynamic_batch_scheduler=self._dynamic_batch_scheduler,
                step_phase_profiler=self._step_phase_profiler,
                data_wait_started=_data_wait_started,
                accumulation_wall_start=accumulation_wall_start,
                closure_microbatches=closure_microbatches,
                step_requires_closure=optimizer_requires_step_closure(self.optimizer),
                capture_optimizer_step_rng_state=self._capture_optimizer_step_rng_state,
                on_before_train_step=self.on_before_train_step,
                global_step=self.global_step,
                block_offloader=self._block_offloader,
                logger=logger,
            )
            current_group_target = microbatch_group_execution.current_group_target
            accumulation_wall_start = microbatch_group_execution.accumulation_wall_start
            closure_microbatches = microbatch_group_execution.closure_microbatches
            _data_wait_started = microbatch_group_execution.data_wait_started
            is_accumulation_boundary = microbatch_group_execution.sync_gradients
            self._current_micro_batch_index = microbatch_group_execution.micro_batch_index
            self._current_micro_batch_count = microbatch_group_execution.micro_batch_count
            self._current_sync_gradients = microbatch_group_execution.sync_gradients
            self._current_accumulation_group_start = microbatch_group_execution.accumulation_group_start
            self._training_step_orchestrator_runtime_profile = microbatch_group_execution.orchestrator_runtime
            train_step_invocation_execution = run_lulynx_train_step_invocation_stage_handler(
                train_step=self.train_step,
                batch=batch,
                accumulation_steps=current_group_target,
                return_loss_tensor=fast_accumulation,
                step_phase_profiler=self._step_phase_profiler,
            )
            loss_result = train_step_invocation_execution.loss_result
            self._training_step_orchestrator_runtime_profile = (
                train_step_invocation_execution.orchestrator_runtime
            )

            microbatches_in_group += 1
            loss_accounting_execution = run_lulynx_loss_accounting_stage_handler(
                loss_result=loss_result,
                fast_accumulation=fast_accumulation,
                is_accumulation_boundary=is_accumulation_boundary,
                batch_filenames=batch.get("filenames"),
                pending_loss_total=pending_loss_total,
                pending_loss_count=pending_loss_count,
                pending_filenames=pending_filenames,
                total_loss=total_loss,
                num_steps=num_steps,
                last_loss=getattr(self, "_last_loss", 0.0),
            )
            loss = loss_accounting_execution.loss
            total_loss = loss_accounting_execution.total_loss
            num_steps = loss_accounting_execution.num_steps
            pending_loss_total = loss_accounting_execution.pending_loss_total
            pending_loss_count = loss_accounting_execution.pending_loss_count
            pending_filenames = loss_accounting_execution.pending_filenames
            if loss_accounting_execution.last_loss_updated:
                self._last_loss = loss_accounting_execution.last_loss
            self._training_step_orchestrator_runtime_profile = (
                loss_accounting_execution.orchestrator_runtime
            )

            if is_accumulation_boundary:
                self._last_loss = loss
                if self.pcgrad_enabled and not optimizer_requires_step_closure(self.optimizer):
                    self._apply_pcgrad_pending_grads()
            
            safeguard_execution = run_lulynx_safeguard_stage_handler(
                safeguard=getattr(self, "safeguard", None),
                is_accumulation_boundary=is_accumulation_boundary,
                fast_accumulation=fast_accumulation,
                global_step=self.global_step,
                epoch=epoch,
                loss=loss,
                lr_scheduler=self.lr_scheduler,
                batch_filenames=batch.get("filenames"),
                pending_filenames=pending_filenames,
                collect_gradients_for_safeguard=self._collect_gradients_for_safeguard,
                normalize_safeguard_gradient_scan_mode=self._normalize_safeguard_gradient_scan_mode,
                emit_runtime_event=self.emit_runtime_event,
                optimizer=self.optimizer,
                clear_pcgrad_pending_grads=self._clear_pcgrad_pending_grads,
                restore_safe_state=self._restore_safe_state,
                closure_microbatches=closure_microbatches,
                microbatches_in_group=microbatches_in_group,
                pending_loss_total=pending_loss_total,
                pending_loss_count=pending_loss_count,
                logger=logger,
            )
            pending_filenames = safeguard_execution.pending_filenames
            self._training_step_orchestrator_runtime_profile = safeguard_execution.orchestrator_runtime
            if safeguard_execution.should_stop:
                self._should_stop = True
            if safeguard_execution.should_break_epoch:
                break
            if safeguard_execution.should_continue_epoch:
                closure_microbatches = safeguard_execution.closure_microbatches
                microbatches_in_group = safeguard_execution.microbatches_in_group
                pending_loss_total = safeguard_execution.pending_loss_total
                pending_loss_count = safeguard_execution.pending_loss_count
                _data_wait_started = safeguard_execution.data_wait_started
                continue
            
            # 梯度累积后更新（以及 epoch 尾部的最后一次不足累积）
            if is_accumulation_boundary:
                _update_phase_start = self._step_phase_profiler.start()
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                _vram_diag_step = (
                    self._advanced_monitoring
                    and torch.cuda.is_available()
                    and self.global_step % self._peak_vram_diag_interval == 0
                )
                _vram_optimizer_diag: Dict[str, float] = {}
                _turbocore_native_update_loop_timing: Dict[str, float] = {}
                step_requires_closure = optimizer_requires_step_closure(self.optimizer)
                if self.pcgrad_enabled and not step_requires_closure:
                    self._apply_pcgrad_pending_grads()
                # 梯度裁剪
                trainable_params = self._get_trainable_params()
                _gr = self._gradient_release_manager
                _turbocore_native_update_runtime_context = self._turbocore_native_update_runtime_context()
                _loss_value_for_optimizer_step = loss
                if isinstance(_loss_value_for_optimizer_step, torch.Tensor):
                    _loss_value_for_optimizer_step = float(_loss_value_for_optimizer_step.detach().float().item())
                _turbocore_native_update_runtime_context["optimizer_loss_value_for_step"] = float(
                    _loss_value_for_optimizer_step
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_pre_step_setup",
                    _optimizer_update_substage_start,
                )
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                native_update_pre_optimizer_execution = (
                    run_lulynx_turbocore_native_update_pre_optimizer_stage_handler(
                        gate=self._turbocore_native_update_gate,
                        dispatch_armer=self._turbocore_native_update_dispatch_armer,
                        dispatch_runtime=self._turbocore_native_update_dispatch_runtime,
                        runtime_context=_turbocore_native_update_runtime_context,
                        trainable_params=trainable_params,
                        step=self.global_step,
                        get_training_executor=self._get_turbocore_native_update_training_executor,
                        native_update_loop_timing=_turbocore_native_update_loop_timing,
                        logger=logger,
                    )
                )
                _previous_gate = native_update_pre_optimizer_execution.previous_gate
                _turbocore_native_update_dispatch_arming = native_update_pre_optimizer_execution.dispatch_arming
                _turbocore_native_update_dispatch_runtime_report = (
                    native_update_pre_optimizer_execution.dispatch_runtime_report
                )
                _turbocore_native_update_runtime_recovery_observation = (
                    native_update_pre_optimizer_execution.runtime_recovery_observation
                )
                _turbocore_native_update_runtime_context = native_update_pre_optimizer_execution.runtime_context
                self._training_step_orchestrator_runtime_profile = (
                    native_update_pre_optimizer_execution.orchestrator_runtime
                )
                _native_update_skip_pytorch_grad_clip = bool(
                    _turbocore_native_update_dispatch_arming.get("execute_native_step", False)
                    and _turbocore_native_update_dispatch_arming.get("native_mutation_allowed", False)
                    and not step_requires_closure
                )
                _native_update_skipped_pytorch_grad_clip = False
                _grad_snap = None
                if trainable_params and not step_requires_closure:
                    if _native_update_skip_pytorch_grad_clip:
                        _native_update_skipped_pytorch_grad_clip = True
                        _turbocore_native_update_loop_timing["pytorch_grad_clip_skipped_for_native"] = 1.0
                    else:
                        _total_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )
                        if self._grad_tracker:
                            _grad_snap = self._grad_tracker.update(
                                float(_total_norm), trainable_params,
                            )
                layer_monitor_execution = run_lulynx_layer_monitor_stage_handler(
                    enabled=bool(self._layer_monitor_enabled),
                    global_step=self.global_step,
                    interval=self._layer_monitor_interval,
                    lora_injector=self.lora_injector,
                    optimizer=self.optimizer,
                    max_layers=self._layer_monitor_max_layers,
                    sparsity_epsilon=self._layer_monitor_sparsity_epsilon,
                    mode=self._layer_monitor_mode,
                    sample_size=self._layer_monitor_sample_size,
                    logger=logger,
                )
                _layer_monitor_info = layer_monitor_execution.layer_monitor_info
                self._training_step_orchestrator_runtime_profile = (
                    layer_monitor_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_native_pre_and_grad_clip",
                    _optimizer_update_substage_start,
                )

                optimizer_hook_context = self._optimizer_hook_context(loss)
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                before_optimizer_hook_execution = run_lulynx_before_optimizer_hook_stage_handler(
                    hook_context=optimizer_hook_context,
                    global_step=int(self.global_step),
                    on_before_optimizer_step=self.on_before_optimizer_step,
                    logger=logger,
                )
                emit_after_optimizer_step_event = before_optimizer_hook_execution.emit_after_optimizer_step_event
                self._training_step_orchestrator_runtime_profile = (
                    before_optimizer_hook_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_before_optimizer_hook",
                    _optimizer_update_substage_start,
                )

                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                turbocore_shadow_prepare_execution = run_lulynx_turbocore_shadow_prepare_stage_handler(
                    shadow=self._turbocore_update_shadow,
                    trainable_params=trainable_params,
                    optimizer=self.optimizer,
                    max_grad_norm=self.max_grad_norm,
                    step=self.global_step,
                    native_update_loop_timing=_turbocore_native_update_loop_timing,
                    logger=logger,
                )
                _turbocore_update_shadow_report = turbocore_shadow_prepare_execution.shadow_report
                self._training_step_orchestrator_runtime_profile = (
                    turbocore_shadow_prepare_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_shadow_prepare",
                    _optimizer_update_substage_start,
                )

                _turbocore_native_update_diagnostic_replay_report: Dict[str, Any] = {}

                optimizer_step_executed = False
                scheduler_step_executed = False
                zero_grad_called = False
                optimizer_step_route_execution = run_lulynx_optimizer_step_route_stage_handler(
                    optimizer=self.optimizer,
                    loss=loss,
                    gradient_release_manager=_gr,
                    step_requires_closure=bool(step_requires_closure),
                    closure_microbatches=list(closure_microbatches),
                    accumulation_steps=current_group_target,
                    make_step_closure=self._make_optimizer_step_closure,
                    native_update_runtime=_turbocore_native_update_dispatch_runtime_report,
                    native_update_skipped_pytorch_grad_clip=_native_update_skipped_pytorch_grad_clip,
                    trainable_params=trainable_params,
                    max_grad_norm=self.max_grad_norm,
                    grad_tracker=self._grad_tracker,
                    native_update_loop_timing=_turbocore_native_update_loop_timing,
                    sync_native_update_training_executor_to_pytorch=self._sync_turbocore_native_update_training_executor_to_pytorch,
                    step_phase_profiler=self._step_phase_profiler,
                )
                optimizer_step_executed = optimizer_step_route_execution.optimizer_step_executed
                _native_update_skipped_pytorch_grad_clip = (
                    optimizer_step_route_execution.native_update_skipped_pytorch_grad_clip
                )
                if optimizer_step_route_execution.grad_snapshot is not None:
                    _grad_snap = optimizer_step_route_execution.grad_snapshot
                self._training_step_orchestrator_runtime_profile = optimizer_step_route_execution.orchestrator_runtime
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                turbocore_shadow_compare_execution = run_lulynx_turbocore_shadow_compare_stage_handler(
                    shadow=self._turbocore_update_shadow,
                    shadow_report=_turbocore_update_shadow_report,
                    step=self.global_step,
                    native_update_loop_timing=_turbocore_native_update_loop_timing,
                    logger=logger,
                )
                _turbocore_update_shadow_report = turbocore_shadow_compare_execution.shadow_report
                self._training_step_orchestrator_runtime_profile = (
                    turbocore_shadow_compare_execution.orchestrator_runtime
                )
                native_update_post_optimizer_execution = (
                    run_lulynx_turbocore_native_update_post_optimizer_stage_handler(
                        gate=self._turbocore_native_update_gate,
                        dispatch_armer=self._turbocore_native_update_dispatch_armer,
                        previous_gate=_previous_gate,
                        shadow_report=_turbocore_update_shadow_report,
                        dispatch_runtime_report=_turbocore_native_update_dispatch_runtime_report,
                        runtime_context=_turbocore_native_update_runtime_context,
                        optimizer=self.optimizer,
                        trainable_param_count=len(trainable_params),
                        step=self.global_step,
                        can_retain_gate=self._can_retain_turbocore_native_update_gate,
                        refresh_readiness=self._refresh_turbocore_native_update_readiness,
                        diagnostic_executor_replay=bool(self._turbocore_native_update_diagnostic_executor_replay),
                        native_update_loop_timing=_turbocore_native_update_loop_timing,
                        logger=logger,
                    )
                )
                _turbocore_native_update_gate_report = native_update_post_optimizer_execution.gate_report
                _turbocore_native_update_arming_observation = (
                    native_update_post_optimizer_execution.arming_observation
                )
                _turbocore_native_update_diagnostic_replay_report = (
                    native_update_post_optimizer_execution.diagnostic_replay_report
                )
                _turbocore_native_update_runtime_context = native_update_post_optimizer_execution.runtime_context
                self._training_step_orchestrator_runtime_profile = (
                    native_update_post_optimizer_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_post_step_native_reconcile",
                    _optimizer_update_substage_start,
                )
                _tc_timing_started = time.perf_counter()
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                _turbocore_native_update_runtime_profile = self._refresh_turbocore_native_update_runtime_profile(
                    shadow_report=_turbocore_update_shadow_report,
                    gate_report=_turbocore_native_update_gate_report,
                    dispatch_arming=_turbocore_native_update_dispatch_arming,
                    dispatch_runtime_report=_turbocore_native_update_dispatch_runtime_report,
                    dispatch_recovery=_turbocore_native_update_runtime_recovery_observation,
                    diagnostic_replay=_turbocore_native_update_diagnostic_replay_report,
                    runtime_context=_turbocore_native_update_runtime_context,
                    step=self.global_step,
                )
                if self._turbocore_native_update_gate.requested or self._turbocore_update_shadow.enabled:
                    _turbocore_native_update_loop_timing["runtime_profile_refresh_ms"] = round(
                        (time.perf_counter() - _tc_timing_started) * 1000.0,
                        4,
                    )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_runtime_profile_refresh",
                    _optimizer_update_substage_start,
                )
                optimizer_finalize_execution = run_lulynx_optimizer_finalize_stage_handler(
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    loss=loss,
                    gradient_release_manager=_gr,
                    step_phase_profiler=self._step_phase_profiler,
                )
                scheduler_step_executed = optimizer_finalize_execution.scheduler_step_executed
                zero_grad_called = optimizer_finalize_execution.zero_grad_called
                self._training_step_orchestrator_runtime_profile = optimizer_finalize_execution.orchestrator_runtime
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                optimizer_execution = run_lulynx_optimizer_execution_stage_handler(
                    optimizer=self.optimizer,
                    trace=self._pipeline_trace,
                    gradient_accumulation_steps=current_group_target,
                    optimizer_step_executed=optimizer_step_executed,
                    scheduler_step_executed=scheduler_step_executed,
                    zero_grad_called=zero_grad_called,
                    uses_step_closure=bool(step_requires_closure),
                    uses_fused_backward=bool(optimizer_uses_fused_backward(self.optimizer)),
                    native_update_runtime=_turbocore_native_update_dispatch_runtime_report,
                )
                self._last_pipeline_trace = optimizer_execution.completed_trace
                self._training_step_orchestrator_runtime_profile = optimizer_execution.orchestrator_runtime
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_execution_trace_record",
                    _optimizer_update_substage_start,
                )
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                post_optimizer_maintenance_execution = run_lulynx_post_optimizer_maintenance_stage_handler(
                    clear_pcgrad_pending_grads=self._clear_pcgrad_pending_grads,
                    vram_diag_step=bool(_vram_diag_step),
                    cuda_memory_snapshot=self._cuda_memory_snapshot,
                    last_peak_vram_diagnostics=getattr(self, "_last_peak_vram_diagnostics", None),
                    build_peak_vram_diagnostics=self._build_peak_vram_diagnostics,
                    block_offloader=self._block_offloader,
                    maybe_release_cuda_cache=self._maybe_release_cuda_cache,
                    global_step=self.global_step,
                )
                _vram_optimizer_mb = post_optimizer_maintenance_execution.vram_optimizer_mb
                self._last_peak_vram_diagnostics = (
                    post_optimizer_maintenance_execution.peak_vram_diagnostics
                )
                _precision_swap_offload_report = (
                    post_optimizer_maintenance_execution.precision_swap_offload_report
                )
                _cache_release_report = post_optimizer_maintenance_execution.cache_release_report
                self._training_step_orchestrator_runtime_profile = (
                    post_optimizer_maintenance_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_post_finalize_maintenance",
                    _optimizer_update_substage_start,
                )
                _optimizer_update_substage_start = self._step_phase_profiler.start_cpu()
                after_optimizer_hook_execution = run_lulynx_after_optimizer_hook_stage_handler(
                    hook_context=self._optimizer_hook_context(loss),
                    optimizer_step_executed=optimizer_step_executed,
                    scheduler_step_executed=scheduler_step_executed,
                    zero_grad_called=zero_grad_called,
                    emit_after_optimizer_step_event=emit_after_optimizer_step_event,
                )
                self._training_step_orchestrator_runtime_profile = (
                    after_optimizer_hook_execution.orchestrator_runtime
                )
                self._step_phase_profiler.record_optimizer_update_substage(
                    "optimizer_update_after_optimizer_hook",
                    _optimizer_update_substage_start,
                )

                housekeeping_execution = run_lulynx_post_optimizer_housekeeping_stage_handler(
                    te_manager=self.te_manager,
                    text_encoder_1=self.text_encoder_1,
                    text_encoder_2=self.text_encoder_2,
                    optimizer=self.optimizer,
                    train_text_encoder_any=bool(self._train_text_encoder_any),
                    step_phase_profiler=self._step_phase_profiler,
                    update_phase_start=_update_phase_start,
                    global_step=self.global_step,
                    lora_injector=self.lora_injector,
                    drift_monitor=self._drift_monitor,
                    drift_check_interval=getattr(self, "_drift_check_interval", 1),
                    unet=self.unet,
                    maybe_save_safe_state=self._maybe_save_safe_state,
                    progress_bar=progress_bar,
                    loss=loss,
                    lr_scheduler=self.lr_scheduler,
                    validation_dataloader=self.validation_dataloader,
                    eval_every_n_steps=self.eval_every_n_steps,
                    validate_epoch=self.validate_epoch,
                    epoch=epoch,
                    logger=logger,
                )
                self.global_step = housekeeping_execution.global_step
                _validation_info = housekeeping_execution.validation_info
                self._training_step_orchestrator_runtime_profile = (
                    housekeeping_execution.orchestrator_runtime
                )

                # 回调
                if self.on_step_end:
                    step_wall_seconds = (
                        time.perf_counter() - accumulation_wall_start
                        if accumulation_wall_start is not None
                        else 0.0
                    )
                    self._record_step_timing_window(
                        step_wall_seconds,
                        global_step=self.global_step,
                        accumulation_steps=current_group_target,
                    )
                    _transfer_profile = self._record_transfer_profile_step(step_wall_seconds)
                    telemetry_step_info_execution = run_lulynx_telemetry_step_info_stage_handler(
                        lr_scheduler=self.lr_scheduler,
                        epoch=epoch,
                        step_wall_seconds=step_wall_seconds,
                        pcgrad_runtime_state=self._pcgrad_runtime_state,
                        b_tier_last_state=self._b_tier_last_state,
                        step_phase_profiler=self._step_phase_profiler,
                        accumulation_steps=current_group_target,
                        transfer_profile=_transfer_profile,
                        validation_info=_validation_info,
                        vram_diag_step=bool(_vram_diag_step),
                        peak_vram_stages=getattr(self, "_last_peak_vram_stages", None),
                        vram_forward_mb=_vram_forward_mb,
                        vram_backward_mb=_vram_backward_mb,
                        vram_optimizer_mb=_vram_optimizer_mb,
                        peak_vram_diagnostics=getattr(self, "_last_peak_vram_diagnostics", None),
                        report_fields={
                            "cuda_cache_release": _cache_release_report,
                            "precision_swap_offload": _precision_swap_offload_report,
                            "turbocore_update_shadow": _turbocore_update_shadow_report,
                            "turbocore_direct_grad_lifecycle": self._turbocore_direct_grad_lifecycle_report,
                            "turbocore_native_update_direct_grad_lifecycle": (
                                self._turbocore_native_update_direct_grad_lifecycle_report
                            ),
                            "turbocore_native_update_dispatch_arming": _turbocore_native_update_dispatch_arming,
                            "turbocore_native_update_runtime_recovery_observation": (
                                _turbocore_native_update_runtime_recovery_observation
                            ),
                            "turbocore_native_update_dispatch_runtime": (
                                _turbocore_native_update_dispatch_runtime_report
                            ),
                            "turbocore_native_update_diagnostic_replay": (
                                _turbocore_native_update_diagnostic_replay_report
                            ),
                            "turbocore_native_update_gate": _turbocore_native_update_gate_report,
                            "turbocore_native_update_dispatch_arming_observation": (
                                _turbocore_native_update_arming_observation
                            ),
                            "turbocore_native_update_runtime_profile": _turbocore_native_update_runtime_profile,
                            "turbocore_native_update_loop_timing": _turbocore_native_update_loop_timing,
                        },
                    )
                    _step_info = telemetry_step_info_execution.step_info
                    _transfer_profile = telemetry_step_info_execution.transfer_profile
                    self._training_step_orchestrator_runtime_profile = (
                        telemetry_step_info_execution.orchestrator_runtime
                    )
                    telemetry_side_effects_execution = run_lulynx_telemetry_side_effects_stage_handler(
                        step_info=_step_info,
                        step_wall_seconds=step_wall_seconds,
                        global_step=self.global_step,
                        loss=loss,
                        advanced_monitoring=bool(self._advanced_monitoring),
                        peak_vram_diag_interval=self._peak_vram_diag_interval,
                        auditor=self.auditor,
                        entropy_probe_step=bool(_entropy_probe_step),
                        loss_tracker=self._loss_tracker,
                        act_drift_step=bool(_act_drift_step),
                        act_drift_tracker=self._act_drift_tracker,
                        grad_snapshot=_grad_snap,
                        hessian_trace=_hessian_trace_val,
                        hessian_layers=_hessian_layers,
                        layer_monitor_info=_layer_monitor_info,
                        forgetting_probe=self._forgetting_probe,
                        forgetting_probe_interval=self._forgetting_probe_interval,
                        validation_step=self.validation_step,
                        manifold_tracker=self._manifold_tracker,
                        manifold_snapshot_interval=self._manifold_snapshot_interval,
                        get_trainable_params=self._get_trainable_params,
                        aggressive_component_residency=bool(self._aggressive_component_residency),
                        ensure_cpu_resident_components=self._ensure_cpu_resident_components,
                        verify_phase_module_states=self._verify_phase_module_states,
                        refresh_module_offload_stats=self._refresh_module_offload_stats,
                        update_block_swap_profile=self._update_block_swap_profile,
                        update_precision_swap_observations=self._update_precision_swap_observations,
                        update_vram_smart_sensing_runtime=self._update_vram_smart_sensing_runtime,
                        refresh_training_loop_runtime_profile=self._refresh_training_loop_runtime_profile,
                        memory_optimization_state=self.memory_optimization_state,
                    )
                    _step_info = telemetry_side_effects_execution.step_info
                    _training_loop_runtime = telemetry_side_effects_execution.training_loop_runtime
                    self._training_step_orchestrator_runtime_profile = (
                        telemetry_side_effects_execution.orchestrator_runtime
                    )
                    telemetry_execution = run_lulynx_telemetry_execution_stage_handler(
                        trace=self._pipeline_trace,
                        step_info=_step_info,
                        step_wall_seconds=step_wall_seconds,
                    )
                    self._last_pipeline_trace = telemetry_execution.completed_trace
                    self._training_step_orchestrator_runtime_profile = telemetry_execution.orchestrator_runtime
                    _training_loop_runtime = self._refresh_training_loop_runtime_profile()
                    if _training_loop_runtime:
                        _step_info["training_loop_runtime"] = dict(_training_loop_runtime)
                    telemetry_callback_execution = run_lulynx_telemetry_callback_stage_handler(
                        callback=self.on_step_end,
                        global_step=self.global_step,
                        loss=loss,
                        step_info=_step_info,
                    )
                    self._training_step_orchestrator_runtime_profile = (
                        telemetry_callback_execution.orchestrator_runtime
                    )
                elif accumulation_wall_start is not None:
                    step_wall_seconds = time.perf_counter() - accumulation_wall_start
                    self._record_step_timing_window(
                        step_wall_seconds,
                        global_step=self.global_step,
                        accumulation_steps=current_group_target,
                    )
                    no_callback_telemetry_maintenance = run_lulynx_telemetry_no_callback_maintenance_stage_handler(
                        step_wall_seconds=step_wall_seconds,
                        record_transfer_profile_step=self._record_transfer_profile_step,
                        refresh_module_offload_stats=self._refresh_module_offload_stats,
                        update_block_swap_profile=self._update_block_swap_profile,
                        update_precision_swap_observations=self._update_precision_swap_observations,
                        update_vram_smart_sensing_runtime=self._update_vram_smart_sensing_runtime,
                    )
                    self._training_step_orchestrator_runtime_profile = (
                        no_callback_telemetry_maintenance.orchestrator_runtime
                    )
                    telemetry_execution = run_lulynx_telemetry_execution_stage_handler(
                        trace=self._pipeline_trace,
                        step_info={},
                        step_wall_seconds=step_wall_seconds,
                    )
                    self._last_pipeline_trace = telemetry_execution.completed_trace
                    self._training_step_orchestrator_runtime_profile = telemetry_execution.orchestrator_runtime
                    self._refresh_training_loop_runtime_profile()
                
                if self._thermal_throttler is not None:
                    # Duty-cycle/temperature governor: sleeps the configured
                    # share of the measured busy time at the step boundary.
                    # Async dataloader prefetch keeps filling during the sleep.
                    _thermal_report = self._thermal_throttler.on_step_boundary()
                    if _thermal_report:
                        self._last_thermal_throttle_report = _thermal_report

                tail_execution = run_lulynx_accumulation_group_tail_stage_handler(
                    auditor=self.auditor,
                    auditor_interval=self.auditor_interval,
                    run_audit=self._run_audit,
                    global_step=self.global_step,
                    total_steps=self.total_steps,
                    step=step,
                    total_microbatches=total_microbatches,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    current_group_target=current_group_target,
                    closure_microbatches=closure_microbatches,
                    accumulation_wall_start=accumulation_wall_start,
                )
                if tail_execution.completed_by_step_limit:
                    self.completed_by_step_limit = True
                if tail_execution.should_stop:
                    self._should_stop = True
                current_group_target = tail_execution.current_group_target
                microbatches_in_group = tail_execution.microbatches_in_group
                closure_microbatches = tail_execution.closure_microbatches
                accumulation_wall_start = tail_execution.accumulation_wall_start
                self._training_step_orchestrator_runtime_profile = tail_execution.orchestrator_runtime
                if tail_execution.should_break_epoch:
                    break
            _data_wait_started = time.perf_counter()
        
        epoch_finalization_execution = run_lulynx_epoch_finalization_stage_handler(
            total_loss=total_loss,
            num_steps=num_steps,
            epoch=epoch,
            on_epoch_end=self.on_epoch_end,
            turbocore_native_update_defer_state_sync=bool(self._turbocore_native_update_defer_state_sync),
            sync_turbocore_native_update_training_executor_to_pytorch=(
                self._sync_turbocore_native_update_training_executor_to_pytorch
            ),
            close_turbocore_native_update_training_executor=(
                self._close_turbocore_native_update_training_executor
            ),
        )
        self._training_step_orchestrator_runtime_profile = (
            epoch_finalization_execution.orchestrator_runtime
        )
        return epoch_finalization_execution.result

    def validate_epoch(self, dataloader, epoch: int) -> Dict:
        """Run one validation pass over *dataloader* and return the average loss.

        The model is evaluated in ``torch.no_grad()`` mode -- no gradients are
        computed and no optimizer steps are taken.

        Args:
            dataloader: A DataLoader yielding validation batches (should NOT shuffle).
            epoch: Current epoch number (used for logging).

        Returns:
            Dict with ``avg_loss`` (float) and ``steps`` (int) keys.
        """
        if dataloader is None:
            return {"avg_loss": 0.0, "steps": 0}

        total_loss = 0.0
        num_steps = 0

        self.unet.eval()
        if self.text_encoder_1 is not None and getattr(self, '_train_text_encoder_1', False):
            self.text_encoder_1.eval()
        if self.text_encoder_2 is not None and getattr(self, '_train_text_encoder_2', False):
            self.text_encoder_2.eval()

        try:
            with torch.no_grad():
                progress_bar = tqdm(
                    dataloader,
                    desc=f"Validation {epoch + 1}",
                    disable=False,
                    **_tqdm_kwargs(),
                )
                max_steps = max(int(getattr(self, "max_validation_steps", 0) or 0), 0)
                for batch in progress_bar:
                    if max_steps > 0 and num_steps >= max_steps:
                        break
                    loss = self.validation_step(batch)
                    total_loss += loss
                    num_steps += 1

                    if hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix({"val_loss": f"{loss:.4f}"})
        finally:
            self.unet.train()
            if self.text_encoder_1 is not None and getattr(self, '_train_text_encoder_1', False):
                self.text_encoder_1.train()
            if self.text_encoder_2 is not None and getattr(self, '_train_text_encoder_2', False):
                self.text_encoder_2.train()

        avg_loss = total_loss / max(num_steps, 1)
        logger.info(
            "Validation epoch %d: avg_loss=%.4f (%d steps)",
            epoch + 1,
            avg_loss,
            num_steps,
        )
        return {"avg_loss": avg_loss, "steps": num_steps}
    
    def _run_audit(self):
        """运行审计"""
        if not self.auditor:
            return
        
        try:
            # FIX: 使用正确的 auditor.step() API
            # 之前错误地调用 log_step()，该方法不存在
            self.auditor.step(
                step=self.global_step,
                total_steps=getattr(self, 'total_steps', 10000),
                epoch=self.current_epoch,
                loss=getattr(self, '_last_loss', 0.0),
                lr_unet=self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0,
                network=self.lora_injector,
            )
            
            # SmartRank 逻辑
            if hasattr(self, 'smart_rank_controller') and self.smart_rank_controller:
                # 获取最新的审计报告
                report = self.auditor.get_last_report() if hasattr(self.auditor, 'get_last_report') else None
                if report and "layers" in report:
                    # SmartRank 分析并可能修改权重
                    old_params_count = len(self.lora_injector.get_trainable_params())
                    self.smart_rank_controller.step(self.global_step, report["layers"])
                    
                    # 如果参数发生了物理变化，需要通知 Trainer 更新优化器
                    # 在 prototype 中，我们通过检测参数对象 ID 或数量变化来判断
                    # 这里简化为只要运行了就检查
                    if self.on_params_changed:
                        self.on_params_changed()
                        
        except Exception as e:
            logger.warning(f"Audit/SmartRank failed: {e}")
    
    def stop(self):
        """停止训练"""
        self._should_stop = True
        # Clear resources to prevent leaks
        if self.auditor:
            try:
                if hasattr(self.auditor, 'clear'):
                    self.auditor.clear()
            except Exception:
                pass
        logger.info("Training stop requested")
