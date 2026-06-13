import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .services.runtime_optimization import resolve_runtime_optimization_payload
from .lulynx_trainer.model_acceleration_application import apply_model_acceleration_policy_to_config
from .lulynx_trainer.model_acceleration_policy import normalize_acceleration_profile

# ==============================================================================
from .constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_LOG_DIR,
    DEFAULT_NETWORK_DIM,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_EMA_DECAY,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_SAVE_PRECISION,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_RESOLUTION,
    DEFAULT_CHECKPOINT_KEEP_LAST,
    DEFAULT_AUDITOR_INTERVAL,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_MIN_RANK,
    DEFAULT_SEED,
)

# ==============================================================================
# Enums
# ==============================================================================

class ModelArch(str, Enum):
    SD15 = 'sd15'
    SDXL = 'sdxl'
    ANIMA = 'anima'
    FLUX = 'flux'
    SD3 = 'sd3'
    NEWBIE = 'newbie'

class NetworkType(str, Enum):
    LORA = 'networks.lora'
    LORA_FA = 'networks.lora_fa'
    VERA = 'networks.vera'
    TLORA = 'networks.tlora'
    LYCORIS = 'lycoris.locon'
    DORA = 'networks.lora'  # DoRA usually uses the same module as LoRA but with different args
    FLEXRANK_LORA = 'networks.flexrank_lora'

class AdapterInitStrategy(str, Enum):
    DEFAULT = 'default'
    PISSA = 'pissa'
    OLORA = 'olora'
    LOFTQ = 'loftq'


def _config_boolish(value: Any, default: bool = False) -> bool:
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

class LyCORISAlgo(str, Enum):
    LOHA = 'loha'
    LOKR = 'lokr'
    DORA = 'dora'
    IA3 = 'ia3'
    FULL = 'full'
    DIAG_OFT = 'diag-oft'
    LOCON = 'locon'
    GLORA = 'glora'
    GLOKR = 'glokr'

class OptimizerType(str, Enum):
    ADAMW = 'AdamW'
    ADAMW_8BIT = 'AdamW8bit'
    PAGED_ADAMW = 'PagedAdamW'
    PAGED_ADAMW_32BIT = 'PagedAdamW32bit'
    PAGED_ADAMW_8BIT = 'PagedAdamW8bit'
    PAGED_LION_8BIT = 'PagedLion8bit'
    AUTOMAGIC_PLUS_PLUS = 'Automagic++'
    AUTO_PRODIGY = 'AutoProdigy'
    ADAFACTOR = 'adafactor'
    PRODIGY = 'prodigy'
    LION = 'Lion'
    LION_8BIT = 'Lion8bit'
    SGD_NESTEROV = 'SGDNesterov'
    SGD_NESTEROV_8BIT = 'SGDNesterov8bit'
    DADAPTATION = 'DAdaptation'
    DADAPT_ADAM_PREPRINT = 'DAdaptAdamPreprint'
    DADAPT_ADAGRAD = 'DAdaptAdaGrad'
    DADAPT_ADAM = 'DAdaptAdam'
    DADAPT_ADAN = 'DAdaptAdan'
    DADAPT_ADAN_IP = 'DAdaptAdanIP'
    DADAPT_LION = 'DAdaptLion'
    DADAPT_SGD = 'DAdaptSGD'
    ADAMW_SCHEDULE_FREE = 'AdamWScheduleFree'
    RADAM_SCHEDULE_FREE = 'RAdamScheduleFree'
    SGD_SCHEDULE_FREE = 'SGDScheduleFree'
    PRODIGY_PLUS_SCHEDULE_FREE = 'prodigyplus.ProdigyPlusScheduleFree'
    PYTORCH_OPTIMIZER = 'PytorchOptimizer'
    KAHAN_ADAMW_8BIT = 'KahanAdamW8bit'
    GENERIC = 'GenericOptimizer'
    ANIMA_FACTORED_ADAMW = 'AnimaFactoredAdamW'
    MUON = 'Muon'

class SchedulerType(str, Enum):
    COSINE = 'cosine'
    COSINE_RESTARTS = 'cosine_with_restarts'
    COSINE_WITH_MIN_LR = 'cosine_with_min_lr'
    LOSS_GATED_COSINE = 'loss_gated_cosine'
    LOSS_WEIGHTED_ANNEALED_COSINE = 'loss_weighted_annealed_cosine'
    CONSTANT = 'constant'
    CONSTANT_WARMUP = 'constant_with_warmup'
    LINEAR = 'linear'
    POLYNOMIAL = 'polynomial'
    PIECEWISE_CONSTANT = 'piecewise_constant'
    TSD = 'warmup_stable_decay'
    ONE_CYCLE = 'one_cycle'
    INVERSE_SQRT = 'inverse_sqrt'
    ADAFACTOR = 'adafactor'
    RESTART_LINEAR = 'restart_linear'

class MixedPrecision(str, Enum):
    NO = 'no'
    FP16 = 'fp16'
    BF16 = 'bf16'


class ExecutionCore(str, Enum):
    STANDARD = 'standard'
    TURBO = 'turbo'
    AUTO = 'auto'

class PissaCacheMode(str, Enum):
    NONE = "none"
    INIT_LORA = "init_lora"
    FULL_SVD = "full_svd"

class NewbieLoraTarget(str, Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    FULL = "full"

# ==============================================================================
# Unified Configuration
# ==============================================================================

class UnifiedTrainingConfig(BaseModel):
    """
    Central training configuration consumed by the native Lulynx trainer.

    Fields are grouped by concern: core paths, architecture, network,
    optimizer, training loop, dataset, regularisation, per-family overrides,
    and advanced Lulynx features (auditor, pilot, MN-LoRA, etc.).
    """

    # === Core Paths ===
    pretrained_model_name_or_path: str = Field(default="")
    train_data_dir: str = Field(default="")
    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR)
    output_name: str = Field(default=DEFAULT_OUTPUT_NAME)
    logging_dir: str = Field(default=DEFAULT_LOG_DIR)
    resume_path: str = Field(default="")
    vae_path: str = Field(default="")
    runtime_id: str = ""
    execution_profile_id: str = ""
    device: str = ""

    # === FLUX Specific Paths ===
    ae_path: str = ""  # FLUX autoencoder path
    flux_transformer_path: str = ""  # optional standalone FLUX transformer path
    flux_requested_network_module: str = ""  # original FLUX network module selected by UI/saved config
    flux_transformer_offload: str = "auto"  # auto | off | aggressive; consumed by Flux LoRA preview trainer
    t5xxl_path: str = ""  # FLUX T5-XXL text encoder path
    clip_l_path: str = ""  # FLUX CLIP-L text encoder path
    t5_max_token_length: int = 512  # T5 max token length for FLUX

    # === Architecture ===
    model_type: ModelArch = Field(default=ModelArch.SDXL)
    v2: bool = False
    v_parameterization: bool = False

    # === Network (LoRA) ===
    network_module: NetworkType = Field(default=NetworkType.LORA)
    network_dim: int = Field(default=DEFAULT_NETWORK_DIM)
    network_alpha: float = Field(default=DEFAULT_NETWORK_ALPHA)
    network_dropout: float = 0.0
    network_train_unet_only: bool = False
    network_train_text_encoder_only: bool = False
    network_weights_path: str = ""  # path to pretrained LoRA/network weights

    # LyCORIS Specific
    lycoris_algo: LyCORISAlgo = Field(default=LyCORISAlgo.LOHA)
    conv_dim: int = 4
    conv_alpha: int = 1
    dim_from_weights: bool = False  # infer network_dim from pretrained weights

    # ControlNet
    controlnet_model: str = ""  # pretrained ControlNet model path (optional)
    conditioning_data_dir: str = ""  # optional separate ControlNet conditioning image directory
    data_backend: str = "auto"  # auto | caption | raw | webdataset | dali
    image_decode_backend: str = "pil"  # auto | pil | pil_lru | torchvision_cpu
    image_decode_cache_size: int = 0  # decoded PIL image LRU entries per DataLoader worker

    # ControlNet-LLLite
    lllite_cond_emb_dim: int = 32  # conditioning image embedding dimension
    lllite_mlp_dim: int = 64  # adapter bottleneck dimension
    lllite_dropout: float = 0.0  # dropout in adapter path
    lllite_skip_input_blocks: bool = False  # skip injecting into UNet input blocks
    lllite_skip_output_blocks: bool = True  # skip injecting into UNet output blocks

    # IP-Adapter
    ip_image_encoder_path: str = "openai/clip-vit-large-patch14"
    ip_num_tokens: int = 0  # 0 = trainer default (SD1.5: 4, SDXL: 16)

    # === Optimizer & Scheduler ===
    optimizer_type: OptimizerType = Field(default=OptimizerType.ADAMW_8BIT)
    optimizer_backend: str = "auto"  # auto | torch_adamw | foreach_adamw | torch_fused | bnb_8bit | ao_8bit | compiled_step | apex | lulynx_fused
    advanced_optimizer_strategy: str = "auto"  # auto | off | profile_only | lora_plus | rs_lora | galore
    learning_rate: float = Field(default=DEFAULT_LR)
    unet_lr: float = 0.0
    text_encoder_lr: float = 0.0
    control_net_lr: float = 0.0

    lr_scheduler: SchedulerType = Field(default=SchedulerType.COSINE)
    lr_scheduler_type: str = ""
    lr_scheduler_args: str = ""
    lr_warmup_steps: int = 0
    lr_scheduler_num_cycles: int = 1
    loss_scheduler_ema_alpha: float = 0.1
    loss_scheduler_min_delta: float = 5e-4
    loss_scheduler_relative_delta: float = 1e-3
    loss_scheduler_patience: int = 8
    loss_scheduler_cooldown: int = 0
    loss_scheduler_max_hold_steps: int = 0
    loss_scheduler_late_gamma: float = 2.0
    loss_scheduler_lock_weight_threshold: float = 0.7
    loss_scheduler_min_advance_ratio: float = 0.25
    warmup_ratio: float = 0.05  # Added for proportional warmup
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer_args: str = ""  # Added for custom args
    auto_prodigy_profile: str = "balanced"  # safe | balanced | aggressive | custom
    auto_prodigy_d0: float = 1e-6
    auto_prodigy_d_coef: float = 1.0
    auto_prodigy_growth_rate: float = 1.02
    auto_prodigy_max_update_rms_ratio: float = 0.01
    auto_prodigy_damping: float = 1.0
    auto_prodigy_beta3: float = 0.99
    auto_prodigy_safeguard_warmup: bool = True

    # === Performance & Precision ===
    acceleration_profile: str = "off"  # off | safe | balanced | aggressive | low_vram
    speed_profile: str = "off"  # legacy/UI alias for acceleration_profile
    mixed_precision: MixedPrecision = Field(default=MixedPrecision.BF16)
    save_precision: str = Field(default=DEFAULT_SAVE_PRECISION)
    gradient_checkpointing: bool = True
    checkpoint_policy: str = "auto"  # auto | off | full | offloaded | selective
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs_to_disk: bool = False
    text_encoder_outputs_cache_dtype: str = ""
    attention_backend: str = "auto"           # Canonical field: "auto", "sdpa", "xformers", "flash2", "sageattn", "flexattn", "spargeattn2", "torch"
    sdpa_backend_policy: str = "cutlass"      # auto | cutlass | flash | cudnn | math ; only used when attention_backend resolves to sdpa
    attention_early_deletion: bool = False     # del Q/K/V tensors immediately after attention computation to free VRAM sooner
    sageattn_drift_check_interval: int = 0    # 0 = disabled; N = compare SageAttn vs SDPA every N steps to detect quantization drift
    sageattn_drift_threshold: float = 0.01    # relative error threshold before triggering warning or fallback
    sageattn_drift_fallback: str = "warn"     # "warn" = log warning only; "fallback_sdpa" = switch backend to SDPA
    xformers: bool = False                    # DEPRECATED: use attention_backend="xformers" instead. Kept for legacy config compat.
    sdpa: bool = False                        # DEPRECATED: use attention_backend="sdpa" instead. Kept for legacy config compat.
    full_bf16: bool = False
    full_fp16: bool = False  # full FP16 precision (alternative to bf16)
    fp8_base: bool = False  # legacy alias: compress frozen backbone weights to FP8
    # FP8 base compute: run the frozen base GEMM on Ada FP8 tensor cores
    # (torch._scaled_mm). Opt-in on top of fp8 storage; makes the otherwise
    # storage-only fp8 base forward-safe. Falls back to bf16 dequant if unsupported.
    fp8_base_compute: bool = False
    fp8_base_unet: bool = False
    weight_compression_preset: str = "off"  # off | stable_backbone_int8 | aggressive_backbone_uint4 | text_encoder_int8 | both_int8 | experimental_float8
    weight_compression_enabled: bool = False
    weight_compression_target: str = "none"  # none | text_encoder | backbone | both
    weight_compression_format: str = "fp8_e4m3"  # first backend; future: int8/uint4/float8 backends
    weight_compression_include_patterns: str = ""
    weight_compression_exclude_patterns: str = ""
    weight_compression_verify: bool = True
    weight_compression_allow_offload_combo: bool = False
    compression_companion_enabled: bool = False  # load a frozen recovery adapter and merge it into base weights before compression
    compression_companion_path: str = ""
    compression_companion_type: str = "lora"
    compression_companion_mode: str = "merge_into_base"
    compression_companion_scale: float = 1.0
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_dynamic: bool = False
    torch_compile_fullgraph: bool = False
    torch_compile_scope: str = ""  # "", "per_block", "full" — per_block compiles each block separately
    torch_compile_allow_full_with_per_block: bool = False
    dynamo_recompile_limit: int = 0  # pin dynamo recompile budget across compile contexts (0=leave default)
    activation_memory_budget: float = 0.0  # AOT partitioner recompute cap in (0,1]; 0=off; skipped under grad-ckpt
    compile_runtime: str = "off"  # off | auto | compile | compile_cache | cudagraph | compile_cudagraph
    compile_shape_strategy: str = "auto"  # auto | fixed_pad | token_flatten | native
    compile_target_strategy: str = "auto"  # auto | block | inner_forward
    amd_empty_cache_interval: int = 0
    amd_sdpa_slice_trigger_gb: float = 0.0
    amd_sdpa_slice_target_gb: float = 0.0
    compile_cache_enabled: bool = True
    compile_cache_root: str = "model.cache"
    compile_cache_reuse: bool = True
    compile_cache_prewarm: bool = False
    compile_contract_strict: bool = True  # enforce route-aware compile safety gates before training
    compile_static_shape_drop_last: bool = True  # drop incomplete batches when static compile needs fixed batch shape
    compile_require_cache_first: bool = True  # require cached latents/text for static/full compile routes where applicable
    compile_probe_enabled: bool = True  # benchmark/probe gate for experimental compile routes
    compile_probe_steps: int = 3
    compile_probe_max_vram_increase_ratio: float = 0.15
    compile_probe_min_speedup_ratio: float = 0.03
    compile_anima_full_core_enabled: bool = False  # CLI/config-only experimental full-core Anima compile gate
    native_token_bucket_compile: bool = True  # allow no-pad cache buckets as the static-shape unit for native DiT per-block compile
    native_runtime_profile: str = "standard"  # standard | aggressive | anima_fast | anima_low_vram | anima_experimental
    native_cache_mode: str = ""  # empty=trainer default | cache_first | rebuild_cache | online_cache | raw_online | force_cache_only
    anima_cache_mode: str = ""  # Anima-specific alias for native_cache_mode
    trust_cache: bool = False  # CLI/config-only: allow cache use when manifest fingerprints warn
    native_training_method: str = "lora"  # lora | ortholora | tlora | lora_ortho_tlora
    execution_core: ExecutionCore = Field(default=ExecutionCore.STANDARD)
    training_step_orchestrator_internal_gate_enabled: bool = False  # dev-only report gate; does not replace _train_step_impl
    turbocore_features: List[str] = Field(default_factory=list)
    turbocore_disable: List[str] = Field(default_factory=list)
    turbocore_strict: bool = False
    turbocore_allow_fallback: bool = True
    turbocore_profile: str = "basic"
    turbocore_workspace_mb: int = 0
    turbocore_prefetch_depth: int = 2
    turbocore_experimental_fp8: bool = False
    turbocore_update_shadow_mode: str = "off"  # off | profile | shadow; dev-only sidecar around real optimizer step
    turbocore_update_shadow_max_params: int = 0
    turbocore_update_shadow_compare_interval: int = 1
    turbocore_update_shadow_direct_grad: bool = False
    turbocore_update_shadow_prefer_triton: bool = False
    turbocore_update_shadow_compare_sample_params: int = 0
    turbocore_update_shadow_stop_after_consecutive_passes: int = 0
    turbocore_update_shadow_checkpoint_contract: bool = False
    turbocore_update_shadow_copyback_probe: bool = False
    turbocore_update_shadow_copyback_dispatch_experimental: bool = False
    turbocore_update_shadow_native_binding_probe: bool = False
    turbocore_update_shadow_owner_native_launch_probe: bool = False
    turbocore_update_shadow_owner_native_launch_max_numel: int = 1048576
    turbocore_update_shadow_owner_native_event_chain_probe: bool = False
    turbocore_update_shadow_save_owner_state: bool = False
    turbocore_native_update_mode: str = "off"  # off | profile | native_experimental; default-off native update gate
    turbocore_native_update_required_shadow_passes: int = 3
    turbocore_native_update_max_abs_diff: float = 5e-5
    turbocore_native_update_max_mean_abs_diff: float = 1e-6
    turbocore_native_update_allow_missing_kernel: bool = False
    turbocore_native_update_strict: bool = False
    turbocore_native_update_dispatch_enabled: bool = False  # default-off internal dispatch request
    turbocore_native_update_training_path_enabled: bool = False  # explicit dev-only native optimizer dispatch path
    turbocore_native_update_require_native_cuda: bool = False  # require rust/CUDA AdamW; never fall back to torch owner as native dispatch
    turbocore_native_update_diagnostic_executor_replay: bool = False  # default-off cloned replay of owner-native shadow evidence
    turbocore_native_update_defer_state_sync: bool = False  # experimental perf path; syncs PyTorch optimizer state only on fallback/close
    turbocore_native_update_runtime_synchronization_policy: str = "context_synchronize"  # context_synchronize | borrowed_stream_event_chain
    triton_ops_enabled: bool = False  # default-off fused LoRA/block kernels for protected probes
    triton_ops_inject_lora: bool = True
    triton_ops_inject_qkv: bool = True
    triton_ops_inject_adaln: bool = True
    triton_ops_fp32_backward: bool = False

    # === 显存优化 ===
    swap_granularity: str = "off"  # off | auto | block | merged_block | layer
    swap_ratio: float = 0.0  # 交换比例 (0.0-1.0)，主推荐入口
    swap_count: int = 0  # 绝对交换数量；优先级高于 swap_ratio
    block_merge_size: int = 2  # merged_block 模式下每组包含的 block 数
    block_swap_strategy: str = "auto"  # auto | sync | async | pipeline
    blocks_to_swap: int = 0  # legacy: UNet block 级别 CPU↔GPU 搬运数量 (0=禁用)
    module_offload_enabled: bool = False  # clean-room module-level CPU offload for frozen Linear / Conv modules
    module_offload_ratio: int = 0  # 0-100; percentage of manageable modules to offload per enabled scope
    module_offload_backbone_ratio: Optional[int] = None  # None = inherit module_offload_ratio
    module_offload_text_encoder_ratio: Optional[int] = None  # None = inherit module_offload_ratio per text encoder
    module_offload_profile: str = "custom"  # custom | conservative | balanced | aggressive
    module_offload_profile_enabled: bool = False
    module_offload_min_param_mb: float = 0.0
    module_offload_include_patterns: str = ""
    module_offload_exclude_patterns: str = ""
    module_offload_verify_state: bool = True
    module_offload_prefetch_enabled: bool = False
    module_offload_prefetch_mode: str = "experimental"
    module_offload_enhanced: bool = False
    opt_channels_last: bool = False  # channels_last 内存格式，提升 GPU cache 命中率
    vae_slicing: bool = True  # VAE 分片解码
    vae_tiling: bool = False  # VAE 分块解码 (超分辨率场景)
    attention_slicing: bool = True  # UNet attention 分片
    pin_memory: bool = True  # DataLoader pin_memory
    prefetch_factor: int = 2  # DataLoader prefetch_factor
    cached_dataloader_auto_policy: bool = True  # auto-tune cached route DataLoader workers/prefetch/pin_memory
    cached_dataloader_workers: Union[int, str] = "auto"  # auto or explicit worker count for cached routes
    cached_dataloader_prefetch_factor: Union[int, str] = "auto"  # auto or explicit prefetch factor for cached routes
    cached_dataloader_pin_memory: Union[bool, str] = "auto"  # auto or explicit pin_memory for cached routes
    cached_collate_mode: str = "auto"  # auto | legacy | pad_sequence
    lossless_cache_replacement_mode: str = "off"  # off | anima_lxfs_probe | anima_lynx_manifest_probe; explicit dev A/B only
    lossless_cache_replacement_codecs: str = "lz4fast"
    lossless_cache_replacement_sidecar_dir: str = ""
    lossless_cache_replacement_sidecar_suffix: str = ".lxfs"
    lossless_cache_replacement_manifest_path: str = ""
    lossless_cache_replacement_shard_size: int = 16
    lossless_cache_replacement_copy_arrays: bool = True
    lossless_cache_replacement_prefetch_depth: int = 2
    lossless_cache_replacement_prepare_sidecars: bool = False
    lossless_cache_replacement_strict: bool = False
    lossless_cache_replacement_fallback_to_raw: bool = True
    lossless_cache_replacement_min_saving: float = 0.0
    lossless_cache_replacement_focus_sample_ids: str = ""
    bubble_controller_enabled: bool = False  # enable bubble-aware runtime diagnosis/advisor patches
    bubble_controller_mode: str = "report_only"  # report_only | advisor_patch | auto_apply
    bubble_controller_target_active_gpu_util: float = 0.90
    bubble_controller_target_saturated_ratio: float = 0.50
    bubble_controller_min_throughput_gain: float = 0.03
    bubble_controller_warmup_steps: int = 8
    bubble_controller_tune_interval_steps: int = 32
    bubble_controller_max_actions_per_run: int = 3
    bubble_controller_max_vram_ratio: float = 0.92
    bubble_controller_allow_batch_growth: bool = True
    bubble_controller_allow_worker_tuning: bool = True
    bubble_controller_allow_transfer_prefetch: bool = True
    bubble_controller_allow_optimizer_swap: bool = False
    bubble_controller_allow_checkpoint_async: bool = True
    bubble_controller_allow_dataloader_rebuild_current_run: bool = False
    bubble_advisor_action_ledger: Dict[str, Any] = Field(default_factory=dict)
    bubble_advisor_action_history: List[Dict[str, Any]] = Field(default_factory=list)
    bubble_closed_loop_action_history: List[Dict[str, Any]] = Field(default_factory=list)
    bubble_closed_loop_cross_run_cooldown_runs: int = 1
    data_transfer_non_blocking: bool = True  # use non_blocking tensor.to() when tensors come from pinned memory
    data_transfer_profile_enabled: bool = False  # profile tensor transfer cost; may synchronize CUDA when enabled
    data_transfer_profile_mode: str = "event"  # event | sync | off
    data_transfer_profile_window: int = 50
    step_phase_profile_enabled: bool = False  # benchmark-only; synchronizes CUDA to time forward/backward/update phases
    newbie_backward_op_profile_enabled: bool = False  # benchmark-only; torch.profiler around Newbie loss.backward()
    newbie_backward_op_profile_top_k: int = 12
    newbie_backward_op_profile_max_samples: int = 1
    newbie_backward_op_profile_record_shapes: bool = False
    newbie_module_timing_profile_enabled: bool = False  # benchmark-only; hook-based Newbie module timing probe
    newbie_module_timing_profile_top_k: int = 12
    newbie_module_timing_profile_max_samples: int = 1
    bubble_controller_benchmark_data_wait_stall_ms: float = 0.0  # benchmark-only; injects real DataLoader item wait
    bubble_controller_benchmark_data_wait_direct_action: bool = False  # benchmark-only; bypass sync-profiler guard for stall probes
    cpu_offload_checkpointing: bool = False  # offload checkpoint activations to CPU during forward pass
    enable_sequential_cpu_offload: bool = False  # layer-by-layer CPU offload during forward (diffusers-style)
    vram_swap_to_ram: bool = False  # swap adapter weights to CPU RAM between training steps
    low_vram_profile: str = "off"  # off | standard_16g | low_12g | very_low_8g | experimental
    sdxl_low_vram_optimization: bool = False  # auto-combine gc+vae_slicing+attention_slicing+cpu_offload+block_swap for low VRAM SDXL
    sdxl_low_vram_auto_protection: bool = False
    sdxl_low_vram_auto_resolution_probe: bool = False
    sdxl_low_vram_bucket_reso_steps: int = 0
    sdxl_low_vram_component_cpu_residency: bool = False
    sdxl_low_vram_fixed_block_swap: bool = False
    sdxl_low_vram_preview_policy: str = ""
    sdxl_low_vram_resolution_mode: str = ""
    sdxl_low_vram_swap_input_blocks: bool = False
    sdxl_low_vram_swap_middle_block: bool = False
    sdxl_low_vram_swap_offload_after_backward: bool = False
    sdxl_low_vram_swap_output_blocks: bool = False
    sdxl_low_vram_swap_vram_threshold: int = 0
    sdxl_low_vram_two_phase_cache: bool = False
    te_vae_offload_strategy: str = "phase"  # resident | phase | aggressive
    cuda_cache_release_strategy: str = "oom_only"  # off | oom_only | phase_boundary | after_optimizer | aggressive
    cuda_cache_release_interval: int = 1
    model_to_condition_enabled: bool = True
    lulynx_precision_swap_enabled: bool = False
    lulynx_precision_swap_strategy: str = "balanced"
    sdxl_unet_backend: str = "diffusers"  # diffusers | native_shadow | native_proxy | native_skeleton | lulynx_native
    lulynx_weight_residency: str = "resident"  # resident | linear_cpu_pinned | linear_conv_cpu_pinned
    lulynx_weight_residency_min_params: int = 0
    vram_smart_sensing_enabled: bool = True
    vram_smart_sensing_baseline_steps: int = 50
    vram_smart_sensing_slowdown_ratio: float = 1.5
    vram_smart_sensing_window_steps: int = 5
    vram_auto_enhance_enabled: bool = True
    vram_smart_sensing_streaming_enabled: bool = True
    vram_smart_sensing_sparse_swap_enabled: bool = True
    vram_smart_sensing_delta_cache_enabled: bool = False
    enhanced_protection_mode: bool = False
    pcie_transfer_format: str = "off"  # off | raw_fp16 | raw_bf16 | fp8_e4m3 | int8_rowwise | uint4_rowwise
    pcie_delta_cache_enabled: bool = False
    pcie_delta_cache_mode: str = "observe"  # observe | cache_v0
    pcie_delta_cache_budget_mb: float = 256.0
    sparse_swap_enabled: bool = False
    sparse_swap_budget_mb: float = 0.0
    sparse_swap_warm_fraction: float = 0.35
    sdxl_block_swap_enabled: bool = False
    sdxl_block_swap_input_blocks: bool = False
    sdxl_block_swap_middle_block: bool = False
    sdxl_block_swap_offload_after_backward: bool = False
    sdxl_block_swap_output_blocks: bool = False
    sdxl_block_swap_vram_threshold: int = 0
    latent_cache_disk_format: str = "npz"  # "npz" | "safetensors" | "pt" — cache file format on disk
    latent_cache_disk_dtype: str = "float16"  # "float16" | "bfloat16" | "float32" — dtype for cached latents on disk
    text_encoder_outputs_cache_disk_format: str = "npz"  # "npz" | "safetensors" | "pt" — TE-output cache file format on disk
    text_encoder_outputs_cache_disk_dtype: str = "float16"  # "float16" | "bfloat16" | "float32" — dtype for cached TE outputs on disk
    disable_mmap_load_safetensors: bool = False  # if True, materialize safetensors fully in RAM instead of mmap (avoids slow random reads on network/HDD storage)

    # === Attention Profiles ===
    experimental_attention_profile_enabled: bool = False  # enable sliding-window or chunked attention profile
    experimental_attention_profile_window: int = 0  # attention window size (0 = full attention)
    experimental_attention_profile_backend: str = "auto"  # auto | flex | sdpa_masked | torch_fallback
    experimental_attention_profile_torch_max_tokens: int = 2048  # guard pure torch O(n^2) fallback
    cross_attn_fused_kv: bool = False  # fuse cross-attention K/V projections to save memory
    fused_projection_memory_mode: str = "keep_original"  # keep_original | drop_original | materialize_on_save

    # === Fused Optimizer ===
    blockwise_fused_optimizers: bool = False  # fuse optimizer updates per-block for memory efficiency
    fused_optimizer: bool = False  # replace AdamW with FusedAdamW (single-pass step per param, reduces kernel launch overhead)

    # === 进阶显存优化 (Port 5) ===
    gradient_release_enabled: bool = False  # release gradients per-parameter during backward to reduce peak gradient memory
    gradient_release_mode: str = "post_step"  # "post_step" = iterate+release after step; "during_backward" = hook-based release during backward
    cpu_offload_checkpointing_mode: str = "standard"  # "standard" = save_on_cpu; "pinned_async" = pinned memory + async CUDA stream transfer
    cpu_offload_checkpointing_pool_gb: float = 1.0  # pinned memory pool size in GB (pinned_async mode only)
    resolution_aware_batch_enabled: bool = False  # adjust effective batch based on input resolution to keep VRAM constant
    resolution_aware_batch_base_resolution: int = 1024  # reference resolution at which train_batch_size applies
    resolution_aware_batch_max_factor: float = 4.0  # max batch multiplier for small images
    resolution_aware_batch_min_factor: float = 0.25  # min batch multiplier for large images
    pipeline_parallel_enabled: bool = False  # split model layers across multiple GPUs
    pipeline_parallel_chunks: int = 2  # number of micro-batches for pipeline schedule
    pipeline_parallel_split_points: str = ""  # comma-separated layer names for manual split (empty=auto-balance)
    stochastic_rounding: bool = False  # use stochastic rounding in optimizer step for unbiased bf16/fp16 parameter updates
    stochastic_grad_accumulation: bool = False  # apply stochastic rounding to gradients during bf16/fp16 accumulation steps
    optimizer_state_paging_enabled: bool = False  # experimental: park large optimizer state tensors on CPU between steps
    optimizer_state_paging_min_tensor_mb: float = 1.0
    optimizer_state_paging_pin_memory: bool = False
    activation_compression_enabled: bool = False  # experimental: compress autograd-saved activations on device
    activation_compression_dtype: str = "fp16"  # fp16 | bf16 | fp8_e4m3
    activation_compression_min_tensor_mb: float = 1.0
    activation_cpu_offload_enabled: bool = False  # experimental: offload large saved activations to pinned CPU memory
    activation_cpu_offload_min_tensor_mb: float = 1.0  # only activations at/above this size leave the GPU
    activation_cpu_offload_pool_gb: float = 1.0  # pre-allocated pinned CPU pool for activation transfers
    anima_progressive_full_finetune_enabled: bool = False
    anima_progressive_full_finetune_schedule: str = ""  # e.g. "0:24-27,100:16-27,200:all"
    anima_progressive_full_finetune_default: str = "all"
    anima_rematerializable_block_enabled: bool = False  # profile-only prototype marker
    anima_rematerializable_block_mode: str = "profile_only"

    # === Training Loop ===
    max_train_epochs: int = DEFAULT_EPOCHS
    max_train_steps: int = 0
    validation_split: float = 0.0  # fraction of training data to hold out for validation (0 = no validation)
    validation_every_n_epochs: int = 1  # run validation every N epochs (only if validation_split > 0)
    validation_loss_only: bool = True  # only compute loss during validation (no sampling)
    te_dropout: float = 0.0  # probability of dropping text encoder conditioning during training (0 = disabled)
    train_batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = 1
    gradient_accumulation_mode: str = "fast"  # "fast" gates sync/checks per optimizer step; "classic" keeps legacy microbatch checks
    max_grad_norm: float = 1.0
    gradient_guard_strategy: str = "none"           # "none" | "agc" | "centralized" | "agc_centralized"
    gradient_guard_agc_clip_factor: float = 0.01    # AGC: max gradient norm as fraction of parameter norm
    gradient_guard_agc_eps: float = 1e-3            # AGC: epsilon for parameter norm clamping

    # === SVD Gradient Projection ===
    svd_grad_proj_enabled: bool = False              # enable SVD gradient projection for memory-efficient training
    svd_grad_proj_rank: int = 128                    # rank of the gradient projection subspace
    svd_grad_proj_update_interval: int = 200         # re-compute SVD projection basis every N steps
    svd_grad_proj_scale: float = 1.0                 # scaling factor for projected gradients
    svd_grad_proj_target: str = "lora"               # "lora" = LoRA params only; "all" = all trainable params
    svd_grad_proj_warmup_steps: int = 0              # use full gradients for first N steps before enabling projection

    # === Adapter Target Policy (FG-LoRA style selective injection; default-off) ===
    # "all" (default) keeps every model-family target module at network_dim -> bitwise
    # identical to legacy injection. The other policies require a profile JSON and
    # select a subset of module types (and optionally per-type rank) from it.
    adapter_target_policy: str = "all"                  # "all" | "profiled" | "gradient_selected" | "cka_selected"
    adapter_target_policy_profile_path: str = ""        # JSON profile with per-layer metrics; required for non-"all" policies
    adapter_target_policy_min_rank: int = 1
    adapter_target_policy_max_rank: int = 64
    adapter_target_policy_fraction: float = 1.0         # keep this top fraction of scored module types
    adapter_target_policy_top_k: int = 0                # keep top-k module types (0 = use fraction)
    adapter_target_policy_min_score: float = 0.0        # drop module types scoring below this threshold

    # === FG-LoRA per-layer rank policy (frontier #4; default-off, two directions) ===
    # A unified selector over the two rank-budget philosophies; the operator picks the
    # direction, lulynx ships both as default-off reserves (real-model gain is the
    # operator's A/B). "uniform" (default) keeps every target layer at network_dim ->
    # bitwise identical to legacy injection. "coupled_prune" delegates to the
    # adapter_target_policy engine (selects important layers, DROPS the rest, couples
    # rank to score -> saves VRAM). "orthogonal_redistribute" keeps ALL target layers
    # and only reallocates each layer's rank by a depth profile (efficiency; worst case
    # = parity). See lulynx_trainer/fg_lora_rank_policy.py.
    fg_lora_rank_policy: str = "uniform"                 # "uniform" | "coupled_prune" | "orthogonal_redistribute"
    fg_lora_rank_min: int = 1                            # floor for any per-layer rank
    fg_lora_rank_max: int = 64                           # ceiling for any per-layer rank
    fg_lora_rank_profile: str = "center_peak"            # "center_peak" | "ascending" | "descending" | "flat"
    fg_lora_rank_conserve_budget: bool = True            # keep sum(rank) ~= N*network_dim (pure redistribution)

    # === SRA2 + HASTE alignment auxiliary loss (default-off, additive) ===
    # When enabled, adds a VAE self-representation alignment loss on captured DiT
    # hidden states, gated by a HASTE schedule. Disabled -> the term is never
    # computed, so the training loss path stays bitwise-identical to legacy.
    sra2_haste_enabled: bool = False
    sra2_haste_capture_layers: str = ""                 # comma-separated module-name suffixes to capture
    sra2_haste_loss_type: str = "cosine"                # "cosine" | "l2" | "l1"
    sra2_haste_base_weight: float = 1.0
    sra2_haste_start_step: int = 0
    sra2_haste_stop_step: int = -1                      # -1 = no early stop
    sra2_haste_decay_start_step: int = -1
    sra2_haste_decay_end_step: int = -1
    sra2_haste_min_weight: float = 0.0
    sra2_haste_plateau_patience: int = 0
    sra2_haste_min_relative_improvement: float = 0.0
    sra2_haste_normalize_targets: bool = True
    sra2_haste_stop_grad_target: bool = True

    # === DiT block compute reducer (default-off, strategy-selectable) ===
    # Selects ONE block-level compute-reduction strategy driven inside the live
    # DiT _run_blocks loop. "none" -> the seam is never published, so the block
    # forward is bitwise-identical to legacy. Loss-parity / quality-drift A/B
    # stays the operator's real-model job; this only wires the runtime seam.
    dit_compute_reducer_strategy: str = "none"          # none|tread|diffcr|blockskip
    dit_compute_reducer_keep_ratio: float = 1.0         # tread: fraction of tokens kept (1.0 = no-op)
    dit_compute_reducer_min_keep_tokens: int = 1        # tread: floor on kept tokens
    dit_compute_reducer_compression_ratio: float = 1.0  # diffcr: kept fraction after merge (1.0 = no-op)
    dit_compute_reducer_min_tokens: int = 1             # diffcr: floor on compressed tokens
    dit_compute_reducer_skip_ratio: float = 0.0         # blockskip: fraction of blocks skipped (0.0 = no-op)
    dit_compute_reducer_skip_every: int = 0             # blockskip: skip every Nth block (0 = derive from ratio)
    dit_compute_reducer_warmup_steps: int = 0           # blockskip: never skip before this step
    dit_compute_reducer_min_block: int = 0              # blockskip: never skip blocks below this index
    dit_compute_reducer_score_mode: str = "l2"          # token scoring for tread/diffcr

    dataloader_num_workers: int = 0 if os.name == "nt" else 4
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 0
    save_n_epoch_ratio: int = 0
    checkpoint_keep_last: int = Field(default=DEFAULT_CHECKPOINT_KEEP_LAST)
    save_last_n_epochs: int = 0
    save_last_n_steps: int = 0
    save_model_as: str = "safetensors"
    save_to: str = ""  # alternative save directory (empty = use output_dir)
    mem_efficient_save: bool = False  # move tensors to CPU before saving to reduce peak VRAM
    save_state: bool = False
    save_state_on_train_end: bool = False
    save_last_n_epochs_state: int = 0
    save_last_n_steps_state: int = 0
    save_state_to_huggingface: bool = False
    seed: int = DEFAULT_SEED
    training_comment: str = ""  # user note stored in metadata
    no_metadata: bool = False
    initial_epoch: int = 0
    initial_step: int = 0
    skip_until_initial_step: bool = False
    log_with: str = ""  # logging backend (e.g. "tensorboard", "wandb")
    log_prefix: str = ""  # optional run-directory prefix inside logging_dir
    tensorboard_flush_interval_steps: int = 10
    adaptive_step_logging_enabled: bool = True
    adaptive_step_logging_threshold: float = 0.01  # double log interval when step logging exceeds this share of step wall time
    adaptive_step_logging_window: int = 50
    adaptive_step_logging_max_interval: int = 64
    layer_monitor_enabled: bool = True
    layer_monitor_interval: int = 3
    layer_monitor_max_layers: int = 10
    layer_monitor_sparsity_epsilon: float = 1e-8
    layer_monitor_mode: str = "sampled"
    layer_monitor_sample_size: int = 4096
    wandb_api_key: str = ""  # optional WandB login key
    wandb_run_name: str = ""  # optional WandB run display name

    # === Validation ===
    validation_split: float = 0.0  # fraction of training data to use for validation (0=no validation)
    validation_every_n_epochs: int = 1  # run validation every N epochs
    eval_every_n_steps: int = 0  # reserved: step-based eval trigger
    validation_loss_only: bool = True  # True=only compute loss, False=also generate samples
    eval_data_dir: str = ""  # optional independent eval dataset; does not split/mutate train_data_dir
    eval_every_n_epochs: int = 0  # 0 = use validation_every_n_epochs
    eval_batch_size: int = 0  # 0 = use training batch_size
    max_validation_steps: int = 0
    validation_seed: int = 0
    train_split: str = ""
    val_split: str = ""
    val_ratio: float = 0.0

    # === Masked Loss ===
    masked_loss: bool = False  # use alpha mask for loss computation
    alpha_mask: bool = False  # use image alpha channel or *_mask sidecar as loss mask
    strict_masked_loss: bool = False  # raise RuntimeError when masked_loss=True but batch has no loss_masks

    # === Dataset / Bucketing ===
    resolution: Union[int, str] = Field(default=DEFAULT_RESOLUTION) # Allow string "1024,1024"
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    bucket_selection_mode: str = "aspect"
    bucket_custom_resos: Union[str, List[str]] = ""
    bucket_no_upscale: bool = False
    # Staged / progressive resolution training
    enable_mixed_resolution_training: bool = False
    staged_resolution_steps: str = ""  # comma-separated step counts e.g. "500,1000"
    staged_resolution_values: str = ""  # comma-separated resolutions e.g. "512,768,1024"
    staged_resolution_stage_batch_sizes: str = ""  # optional "512:2,768:1,1024:1"; blank inherits train_batch_size
    staged_resolution_ratio_512: int = 0
    staged_resolution_ratio_768: int = 0
    staged_resolution_ratio_1024: int = 0
    staged_resolution_ratio_1536: int = 0
    staged_resolution_ratio_2048: int = 0
    caption_extension: str = ".txt"
    shuffle_caption: bool = True
    shuffle_caption_tags_only: bool = False
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    flip_aug: bool = False
    color_aug: bool = False

    # === Albumentations Pipeline ===
    albumentations_enabled: bool = False             # enable albumentations augmentation pipeline
    albumentations_pipeline: str = ""                # JSON: [{"name": "GaussianBlur", "params": {"blur_limit": 7, "p": 0.3}}, ...]
    albumentations_mask_replay: bool = True          # apply same spatial transforms to loss mask

    random_crop: bool = False
    max_token_length: int = 225
    enable_fixed_token_padding: bool = False  # enforce fixed-length tokenization for torch.compile static shape
    clip_skip: int = 1

    # === Regularization / Advanced ===
    noise_offset: float = 0.0
    noise_offset_type: str = ""  # noise offset variant (empty = default)
    noise_offset_random_strength: bool = False  # randomize offset strength in [0, noise_offset]
    adaptive_noise_scale: float = 0.0
    perlin_noise_offset_enabled: bool = False
    perlin_noise_offset_strength: float = 0.1
    perlin_noise_offset_scale: float = 4.0
    multires_noise_iterations: int = 0
    multires_noise_discount: float = 0.3
    min_snr_gamma: float = 0.0
    optimal_noise_enabled: bool = False
    optimal_noise_candidates: int = 4
    adaptive_loss_weighting_enabled: bool = False   # learnable SNR gamma — trains gamma/offset/scale params for loss-to-timestep weighting
    adaptive_loss_weighting_lr: float = 1e-3        # learning rate for the adaptive loss weighting parameters
    adaptive_loss_weighting_init_gamma: float = 5.0 # initial gamma value for adaptive loss weighter
    spectral_noise_blend: float = 0.0               # 0 = disabled; >0 = blend ratio alpha for low-frequency noise mixing
    spectral_noise_sigma: float = 4.0               # Gaussian blur sigma for spectral noise blending
    ip_noise_gamma: float = 0.0
    ip_noise_gamma_random_strength: bool = False  # randomize IP noise strength in [0, ip_noise_gamma]
    min_timestep: int = 0  # minimum timestep for training (0 = default)
    max_timestep: int = 1000  # maximum timestep for training (1000 = default)
    caption_dropout_rate: float = 0.0 # Mapped from captionDropout
    caption_dropout_every_n_epochs: int = 0
    dual_caption_enabled: bool = False
    dual_caption_short_key: str = "short"
    dual_caption_long_key: str = "long"
    caption_source_mix_enabled: bool = False
    caption_source_nl_ratio: float = 65.0
    caption_source_tag_ratio: float = 20.0
    caption_source_trigger_only_ratio: float = 10.0
    caption_source_empty_ratio: float = 5.0
    caption_source_trigger_tokens: str = ""
    tag_dropout_rate: float = 0.0     # Mapped from tagDropout
    caption_tag_dropout_targets: str = ""
    caption_tag_dropout_target_mode: str = "drop_all"
    caption_tag_dropout_target_count: int = 1
    # Caption shuffle variants (#100)
    tag_swap_rate: float = 0.0           # probability of swapping each adjacent tag pair
    tag_group_shuffle: bool = False      # shuffle tags within groups split by separator
    tag_group_separator: str = "|||"     # separator marking tag-group boundaries

    # === Caption Variants (Multi-Caption Training) ===
    caption_variants_enabled: bool = False
    caption_variants: str = ""  # JSON string: [{"suffix": ".tag", "shuffle": true, "dropout": 0.1}, ...]
    caption_variant_schedule: str = "alternate"  # alternate | ratio | curriculum | custom
    caption_variant_ratio: str = ""  # JSON array for ratio mode: [0.7, 0.3]
    caption_variant_custom_sequence: str = ""  # JSON array for custom mode: [0, 0, 1, 0, 1, 1]
    caption_variant_loss_adaptive: bool = False  # adaptive ratio adjustment based on per-variant loss

    loss_type: str = "l2"
    loss_precision: str = "fp32_loss"  # fp32_loss | mixed_loss
    huber_c: float = 0.1
    huber_schedule: str = "constant"  # constant | exponential | snr
    huber_scale: float = 1.0
    huber_auto_percentile: float = 0.9              # when huber_schedule="auto", compute delta from this percentile of batch residuals

    # === Stepped Loss Schedule ===
    stepped_loss_enabled: bool = False              # enable step-based loss type/weight schedule
    stepped_loss_schedule: str = ""                 # JSON: [{"step": 0, "loss_type": "l2", "weight": 1.0}, ...]

    debiased_estimation_loss: bool = False
    scale_v_pred_loss_like_noise_pred: bool = False
    wavelet_loss_enabled: bool = False  # enable wavelet frequency-aware loss
    wavelet_loss_levels: int = 2  # DWT decomposition levels
    wavelet_loss_high_freq_weight: float = 2.0  # weight for high-freq sub-bands
    wavelet_loss_approx_weight: float = 0.0  # reserved low-frequency LL weight
    wavelet_loss_base_loss: str = "l2"  # base loss inside wavelet: "l2" or "l1"

    # === Pattern Loss (per-band loss functions) ===
    pattern_loss_enabled: bool = False              # enable frequency-band-specific loss functions
    pattern_loss_levels: int = 1                    # DWT decomposition levels
    pattern_loss_ll_type: str = "l2"                # loss function for low-frequency (LL) band
    pattern_loss_ll_weight: float = 1.0             # weight for LL band
    pattern_loss_high_type: str = "huber"           # loss function for high-frequency (LH/HL/HH) bands
    pattern_loss_high_weight: float = 2.0           # weight for high-frequency bands
    pattern_loss_high_huber_c: float = 0.1          # Huber delta for high-freq bands (when type=huber)

    repa_enabled: bool = False
    repa_target_modules: str = ""
    repa_loss_type: str = "cosine"
    repa_loss_weight: float = 0.0
    repa_projection_dim: int = 0
    repa_stop_grad_target: bool = True
    softrepa_enabled: bool = False
    softrepa_schedule: str = "linear"
    softrepa_min_weight: float = 0.0
    softrepa_max_weight: float = 1.0
    softrepa_sigma_min: float = 0.0
    softrepa_sigma_max: float = 1.0
    scale_weight_norms: float = 0.0
    weighted_captions: bool = False
    concept_geometry_enabled: bool = False  # enable Concept Geometry Sampling
    concept_geometry_path: str = ""  # empty = <train_data_dir>/concept_geometry.json or legacy h_lora_geometry.json
    concept_geometry_sampler_mode: str = "density_curriculum"  # curriculum | density | density_curriculum | concept_batch
    concept_geometry_loss_weighting: bool = False  # apply geometry-derived per-sample loss weights
    concept_geometry_density_power: float = 1.0  # density exponent used by concept geometry sampling/weighting
    concept_geometry_semantic_enabled: bool = False  # optional prep-time text embedding enhancement
    concept_geometry_embedding_provider: str = "local_path"  # local_path | auto_download | api
    concept_geometry_embedding_backend: str = "pytorch"  # pytorch | onnx (reserved extension point)
    concept_geometry_embedding_model: str = "BAAI/bge-m3"
    concept_geometry_embedding_model_path: str = ""
    concept_geometry_embedding_cache_dir: str = ""
    concept_geometry_embedding_allow_download: bool = False
    concept_geometry_embedding_api_base: str = ""
    concept_geometry_embedding_api_key: str = ""
    concept_geometry_embedding_api_model: str = ""
    concept_geometry_embedding_batch_size: int = 8
    concept_geometry_embedding_device: str = "cpu"
    concept_geometry_translation_enabled: bool = False
    concept_geometry_translation_provider: str = "local_path"  # local_path | api
    concept_geometry_translation_model_path: str = ""
    concept_geometry_translation_api_base: str = ""
    concept_geometry_translation_api_key: str = ""
    concept_geometry_translation_api_model: str = ""
    concept_geometry_translation_batch_size: int = 8
    concept_geometry_alias_map: str = ""  # optional JSON alias map used by prep-time concept/tag parsing
    concept_geometry_alias_map_path: str = ""  # optional JSON file for prep-time concept/tag aliases
    concept_geometry_source_priority: str = "explicit,folder,nl,identity,tag,stem"
    h_lora_enabled: bool = False  # legacy alias for concept_geometry_enabled
    h_lora_geometry_path: str = ""  # legacy alias for concept_geometry_path
    h_lora_sampler_mode: str = "density_curriculum"  # legacy alias for concept_geometry_sampler_mode
    h_lora_loss_weighting: bool = False  # legacy alias for concept_geometry_loss_weighting
    h_lora_density_power: float = 1.0  # legacy alias for concept_geometry_density_power
    cache_text_encoder_outputs: bool = False
    persistent_data_loader_workers: bool = False

    reg_data_dir: str = ""
    prior_loss_weight: float = 0.0  # DreamBooth prior preservation weight (0 = disabled)

    # === DOP (Differential Output Preservation) ===
    dop_enabled: bool = False                        # enable differential output preservation
    dop_weight: float = 0.1                          # weight for DOP regularization loss
    dop_target: str = "output"                       # "output" = final noise prediction; "features" = intermediate
    dop_start_step: int = 0                          # start applying DOP after this many steps (0 = immediate)
    dop_interval: int = 1                            # apply DOP every N steps (1 = every step)
    dop_detach_reference: bool = True                # detach reference outputs (safety knob)

    # === DreamBooth Specific ===
    instance_prompt: str = ""  # DreamBooth trigger word (e.g. "sks person")
    class_prompt: str = ""  # DreamBooth class description (e.g. "a person")
    num_class_images: int = 100  # prior preservation class images to generate
    use_lora: bool = False  # DreamBooth LoRA mode (vs full finetune)
    lora_rank: int = 16  # DreamBooth LoRA rank (when use_lora=True)

    # === Concept Direction Training ===
    concept_direction_enabled: bool = False           # enable concept direction training mode
    concept_direction_pairs: str = ""                 # JSON: [{"positive": "a smiling person", "negative": "a person"}, ...]
    concept_direction_weight: float = 1.0             # weight for the direction loss
    concept_direction_guidance_scale: float = 1.0     # CFG scale for direction difference
    concept_direction_timestep_range: str = ""        # optional: "200,800" to restrict timestep range
    concept_direction_neutral_reg: float = 0.0        # regularization for neutral (negative) prompt preservation

    # === Textual Inversion Specific ===
    ti_placeholder_token: str = "<new>"
    ti_init_token: str = ""
    ti_num_vectors: int = 1

    # === Zero Terminal SNR ===
    zero_terminal_snr: bool = False

    # === Flow Matching (SDXL / Rectified Flow) ===
    flow_model: str = ""  # "" | "rectified_flow" | "cfm"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_uniform_shift: bool = False
    flow_uniform_base_pixels: int = 256
    flow_uniform_static_ratio: float = 0.0
    cfm_lambda: float = 1.0  # Contrastive Flow Matching weight
    flow_use_ot: bool = False  # Optimal transport coupling
    # Immiscible diffusion: reorder noise to data within the minibatch by L2.
    # Covers DDPM/standard diffusion (and flow when metric overrides cosine).
    immiscible_diffusion_enabled: bool = False
    immiscible_metric: str = "l2"  # "l2" (Immiscible) | "cosine" (flow OT)
    sdxl_flow_weighting_scheme: str = "none"  # "none" | "sigma_sqrt" | "cosmap" | "mode" | "logit_normal"
    sdxl_flow_shift: float = 1.0  # discrete flow shift for SDXL (alias for shift parameter)
    sdxl_model_prediction_type: str = "epsilon"  # "epsilon" | "velocity" | "sample"

    # === Flux / SD3 Specific ===
    discrete_flow_shift: float = 3.0
    model_prediction_type: str = ""
    timestep_sampling: str = ""
    ddpm_timestep_sampling: str = ""                # "" = default uniform; "logit_normal" = logit-normal distribution
    guidance_scale: float = 1.0
    text_encoder_batch_size: int = 1

    # === Distributed Training ===
    multi_gpu: bool = False  # enable DDP (frontend maps enable_distributed -> multi_gpu)
    # Tensor / sequence parallelism (v1 subsystem; default off, not wired into
    # the live model build yet). tensor_parallel_degree>1 requires torchrun.
    tensor_parallel_degree: int = 1
    sequence_parallel: bool = False
    parallel_backend: str = "nccl"  # "nccl" | "cuda_direct"
    num_processes: int = 1  # number of processes for distributed training
    num_machines: int = 1  # number of machines for multi-node training
    machine_rank: int = 0  # current machine rank for multi-node training
    main_process_ip: str = "localhost"  # main process IP for distributed
    main_process_port: int = 29500  # main process port for distributed
    ddp_timeout: int = 0  # distributed timeout seconds; 0 = backend/default
    nccl_socket_ifname: str = ""  # optional NCCL network interface name
    gloo_socket_ifname: str = ""  # optional Gloo network interface name
    ddp_find_unused_parameters: bool = False  # DDP find_unused_parameters
    ddp_gradient_as_bucket_view: bool = True  # DDP gradient_as_bucket_view (faster)
    ddp_static_graph: bool = False  # DDP static_graph (set True if graph doesn't change)
    sync_config_from_main: bool = True  # worker-side distributed launcher hint
    sync_config_keys_from_main: str = "*"
    sync_missing_assets_from_main: bool = True
    sync_asset_keys: str = ""
    sync_main_repo_dir: str = ""
    sync_main_toml: str = ""
    sync_ssh_user: str = ""
    sync_ssh_port: int = 22
    sync_use_password_auth: bool = False
    sync_ssh_password: str = ""
    clear_dataset_npz_before_train: bool = False

    # === Thermal Management ===
    cooldown_every_n_epochs: int = 0  # pause every N epochs to cool GPU (0=disabled)
    cooldown_minutes: int = 10  # cooldown duration in minutes
    cooldown_until_temp: int = 0  # cool to target temp then resume (0=use time)
    cooldown_poll_seconds: int = 30  # polling interval for temperature-based cooldown
    gpu_power_limit_w: int = 0  # GPU power limit in watts (0=no limit)
    gpu_duty_cycle: float = 1.0  # step-level GPU duty cycle; <1.0 sleeps between steps (1.0=off)
    gpu_target_temp_c: int = 0  # closed-loop GPU temp target in C; auto-adjusts duty cycle (0=off)
    gpu_lock_clocks_mhz: int = 0  # lock GPU core clock ceiling via nvidia-smi -lgc, needs admin (0=off)

    # === Training Engine & Control ===
    trainer_engine: str = "lulynx"
    custom_toml: str = ""  # path to custom TOML config for overrides
    optimizer_preset: str = ""  # optimizer preset name (e.g. "Fast", "Stable", "Memory-efficient")
    preset: str = ""  # general preset name for newbie workflow

    # === Lulynx Native Features ===

    # Auditor & Pilot
    enable_auditor: bool = True
    auditor_interval: int = DEFAULT_AUDITOR_INTERVAL
    enable_dynamic_rank: bool = False
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD
    min_rank: int = DEFAULT_MIN_RANK

    # Bad Sample Culling
    enable_bad_sample_culling: bool = False
    quarantine_dir: str = "quarantine"

    # Auto Controller / Pilot (MN-LoRA internal only — not read by native trainer)
    enable_pilot: bool = False
    pilot_strategy: str = "population"

    # Auto Controller (Trainer V3.0)
    auto_controller_enabled: bool = False
    auto_freeze_te: bool = False
    smart_early_stop: bool = False
    smart_lr_decay: bool = False
    clip_drift_threshold: float = 0.5
    stable_rank_threshold: float = 5.0

    # DoRA
    use_dora: bool = False
    dora_wd: bool = False  # Anima UI alias for Weight-Decomposed LoRA
    dora_mode: str = "full"
    bypass_mode: bool = False  # retained alias; DoRA disables bypass-style adapter routing

    # Sampling
    sample_every: int = 0
    sample_every_n_epochs: int = 0
    sample_prompts: Union[List[str], str] = Field(default_factory=list) # Allow string for flexible input
    preview_groups: Union[List[Dict[str, Any]], str] = Field(default_factory=list)
    sample_sampler: str = "euler_a"
    sample_scheduler: str = ""  # sampler scheduler override (empty = use default)
    sample_negative: str = ""
    sample_steps: int = 20
    sample_cfg: float = 7.5
    sample_smc_cfg: bool = False
    sample_smc_cfg_lambda: float = 5.0
    sample_smc_cfg_alpha: float = 0.2
    sample_tgate_probe: bool = False
    sample_tgate_start_step: int = 0
    sample_tgate_min_block: int = 0
    sample_tgate_skip: bool = False  # opt-in T-GATE real cross-attention reuse (default off -> parity)
    sample_spectrum_probe: bool = False
    sample_spectrum_window_size: float = 2.0
    sample_spectrum_flex_window: float = 0.25
    sample_spectrum_warmup_steps: int = 6
    sample_spectrum_stop_caching_step: int = -1
    sample_smoothcache_probe: bool = False
    sample_smoothcache_error_threshold: float = 0.08
    sample_smoothcache_warmup_steps: int = 2
    sample_cache_seam_backend: str = "none"  # none|spectrum|smoothcache|deepcache|teacache (opt-in live cache execution)
    sample_cache_seam_window_size: float = 3.0
    sample_width: int = 0  # preview width (0 = use training resolution)
    sample_height: int = 0  # preview height (0 = use training resolution)
    sample_seed: int = 0  # reserved for sampler wiring; 0 = sampler default
    sample_at_first: bool = False
    preview_device: str = "cpu"  # cpu | gpu | off
    ephemeral_preview_pipeline: bool = True
    micro_vae_preview: bool = False
    micro_vae_model: str = "auto"

    # Multi-GPU
    multi_gpu: bool = False

    training_type: str = "lora"

    # === EMA (Level 1) ===
    ema_use_ema: bool = False
    ema_decay: float = DEFAULT_EMA_DECAY
    ema_update_after_step: int = 100
    ema_update_every: int = 1
    ema_use_ema_warmup: bool = False
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.666
    ema_reset_on_resume: bool = False  # reset EMA state when resuming from checkpoint

    # === S-tier Advisor / Preflight ===
    training_advisor_enabled: bool = True
    training_advisor_report_name: str = "training_advisor_report.json"
    training_advisor_log_summary: bool = True

    # === SafeGuard (Level 1) ===
    so_enable_nan_detection: bool = True
    so_nan_check_interval: int = 10
    so_gradient_check_interval: int = 10
    so_gradient_scan_mode: str = "batched"  # legacy | batched | foreach | off
    so_max_nan_count: int = 3
    so_enable_loss_spike_detection: bool = True
    so_loss_spike_threshold: float = 10.0
    so_loss_window_size: int = 50
    so_enable_lr_deadlock_detection: bool = True
    so_lr_deadlock_threshold: float = 1e-8
    so_lr_deadlock_steps: int = 200
    so_enable_auto_recovery: bool = True
    so_lr_reduction_factor: float = 0.5
    so_enable_bad_sample_culling: bool = False
    so_bad_sample_mode: str = "report"  # report | move
    so_bad_sample_report_name: str = "safeguard_events.jsonl"
    so_bad_sample_max_reported: int = 32
    so_quarantine_dir: str = "quarantine"

    # === Resource Manager (Level 1) ===
    rm_vram_warning_threshold: float = 0.85
    rm_vram_critical_threshold: float = 0.92
    rm_vram_emergency_threshold: float = 0.97
    rm_enable_adaptive_batch: bool = False
    rm_min_batch_size: int = 1
    rm_max_batch_size: int = 8
    rm_enable_adaptive_accumulation: bool = False
    rm_min_accumulation: int = 1
    rm_max_accumulation: int = 16
    rm_cache_clear_interval: int = 100
    rm_enable_adaptive_checkpointing: bool = False
    rm_checkpointing_vram_threshold: float = 0.90
    rm_enable_adaptive_cpu_offload: bool = False
    rm_cpu_offload_vram_threshold: float = 0.95
    rm_enable_peak_tracking: bool = True
    rm_peak_tracking_window: int = 50

    # === LISA (Level 1) ===
    lisa_enabled: bool = False
    lisa_active_ratio: float = 0.2
    lisa_interval: int = 1

    # Wrapper-exposed training modifiers
    lora_fa_enabled: bool = False
    lora_activation_recompute: bool = False  # recompute LoRA down projection during backward to reduce native DiT activation tape
    lora_activation_recompute_mode: str = "auto"  # auto | on | off; auto enables the DiT default while preserving benchmark A/B controls
    flexrank_lora_enabled: bool = False             # FlexRank LoRA: randomly sample active rank each step for multi-rank-compatible checkpoint
    flexrank_lora_rank_range_min: int = 1           # minimum rank to sample during FlexRank training
    vera_enabled: bool = False
    vera_d_initial: float = 0.1
    vera_prng_key: int = 0
    lora_plus_enabled: bool = False
    lora_plus_lr_ratio: float = 16.0
    # Muon optimizer (OptimizerType.MUON): Newton-Schulz orthogonalized momentum
    # on 2D LoRA factors; 1D/scalar params fall back to a built-in AdamW.
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_lr_ratio: float = 1.0  # AdamW-fallback group lr = base_lr * this
    rs_lora_enabled: bool = False  # rank-stabilized LoRA scaling: alpha / sqrt(rank)
    hutchinson_auto_freeze: bool = False
    hutchinson_freeze_ratio: float = 0.5
    tome_enabled: bool = False
    tome_ratio: float = 0.5
    tuneqdm_enabled: bool = False
    tuneqdm_warmup_steps: int = 500

    # === Coreset (Level 2) ===
    coreset_enabled: bool = False
    coreset_easy_weight: float = 1.0
    coreset_hard_weight: float = 1.0
    coreset_toxic_weight: float = 0.0
    # Legacy compatibility: used by LulynxTrainer
    coreset_classify_after: int = 500
    coreset_auto_classify_after: int = 50
    coreset_easy_threshold: float = 0.1
    coreset_hard_loss_threshold: float = 1.5
    coreset_toxic_std_threshold: float = 3.0
    coreset_report_enabled: bool = True
    coreset_report_top_k: int = 20
    coreset_report_every_n_epochs: int = 1

    # PiSSA (V2.1 Core)
    adapter_init_strategy: str = "default"  # default | pissa | olora | loftq; future: eva | lora_ga | corda | qalora
    pissa_enabled: bool = False
    pissa_init_iters: int = 1
    pissa_svd_algo: str = "rsvd"
    pissa_oversample: int = 8
    pissa_apply_conv2d: bool = False
    pissa_export_mode: str = "lora_compatible"
    loftq_bits: int = 4
    loftq_quant_type: str = "rowwise"
    adapter_init_export_mode: str = "auto"  # auto/raw/lora_compatible/approximate; raw is exact resume state
    use_pissa: bool = False  # Frontend alias?
    pissa_cache_mode: PissaCacheMode = PissaCacheMode.INIT_LORA
    auto_convert_to_slora: bool = True

    # SmartRank / Auditor (V2.1 Core)
    smart_rank_enabled: bool = False
    smart_rank_interval: int = 50
    smart_rank_min: int = 4
    smart_rank_max: int = 128
    monitor_svd_algo: str = "full"
    smart_rank: bool = False # Frontend alias?

    # DoRA (V3.0 Frontier)
    dora_enabled: bool = False

    # === Block Weight (Level 2) ===
    bw_enable: bool = False
    bw_preset: str = ""
    bw_in_weights: str = ""
    bw_mid_weight: str = ""
    bw_out_weights: str = ""
    bw_te_weight: float = 1.0
    bw_te2_weight: float = 1.0
    block_lr_zero_threshold: float = 0.0

    # === SVD Callback (Level 2) ===
    svd_enable_callback: bool = False
    svd_check_interval: int = 500
    svd_log_interval: int = 100   # Config uses this
    svd_log_full: bool = False
    svd_log_matrix: bool = False

    # === Optimizer Specific (Level 3) ===
    opt_prodigy_decouple: bool = True
    opt_prodigy_safeguard_warmup: bool = True
    opt_prodigy_use_bias_correction: bool = True
    opt_prodigy_d0: float = 1e-6
    opt_prodigy_d_coef: float = 1.0
    opt_prodigy_growth_rate: float = float("inf")
    opt_prodigy_lr_lower_bound: float = 1e-8
    opt_prodigy_lr_upper_bound: float = 1.0

    # === Advanced Network (Level 3) ===
    dora_init_scale: float = 1.0
    dora_use_scalar_magnitude: bool = False
    dora_normalize_magnitude: bool = True
    lycoris_use_effective_conv: bool = False
    lycoris_loha_use_effective: bool = False
    lycoris_lokr_factor: int = -1
    lycoris_factor: int = -1  # LyCORIS factor (alias for lycoris_lokr_factor)
    lycoris_train_norm: bool = False  # inject into LayerNorm/RMSNorm layers
    lycoris_conv_dim: int = 0  # LoRA/LyCORIS rank for Conv2d layers (0 = same as network_dim)
    lycoris_conv_alpha: float = 0.0  # alpha for Conv2d layers (0 = same as network_alpha)
    lycoris_presets: str = ""  # named preset: "full", "attn-only", "attn-mlp", custom target module list
    lokr_rank_dropout: float = 0.0  # LoKr rank-level dropout; consumed by LyCORIS LoKrLayer.
    lokr_module_dropout: float = 0.0  # LoKr module-level dropout; consumed by LyCORIS LoKrLayer.
    lokr_train_norm: bool = False  # LoKr alias for LyCORIS norm adapter injection.
    lokr_full_matrix: bool = False  # LoKr direct full-matrix branch mode.
    lokr_decompose_both: bool = False  # LoKr decomposes both Kronecker branches when rank allows.
    lokr_unbalanced_factorization: bool = False  # LoKr swaps output factor ordering for unbalanced layouts.
    lokr_no_materialize_forward: bool = False  # Experimental: apply Kronecker factors without materializing full delta weight.
    lokr_no_materialize_strategy: str = "legacy"  # legacy | matmul | auto for the experimental no-materialize path.
    lokr_export_mode: str = "native"  # LoKr export mode: native or lora_compatible.
    # GLoRA (Generalized LoRA: ΔW = W·A + B). Phase 1 standard + Phase 2 extras, all defaults off.
    glora_rank_dropout: float = 0.0  # per-output-row dropout on the materialized ΔW
    glora_module_dropout: float = 0.0  # whole-layer skip probability (returns 0)
    glora_no_materialize_forward: bool = False  # fast path: chain matmuls without materializing A,B
    glora_use_tucker: bool = False              # tucker B path for Conv2d when kernel > 1×1
    glora_train_bias: bool = True               # adapt bias when the base module has one
    glora_export_mode: str = "native"  # GLoRA export mode: native or lora_compatible
    # GLoKr (Kronecker-parameterized Generalized adapter; project-original research).
    glokr_factor: int = -1                       # Kronecker factor (-1 = auto)
    glokr_rank_dropout: float = 0.0
    glokr_module_dropout: float = 0.0
    glokr_no_materialize_forward: bool = False
    glokr_train_bias: bool = True
    glokr_export_mode: str = "native"            # native | lora_compatible (bake delta)
    base_weight: float = 0.0  # base weight scalar for network merging
    base_weight_path: str = ""  # path to base weight file (LoRA base); comma-separated for multiple
    base_weights_multiplier: str = ""  # comma-separated multipliers matching base_weight_path order
    prefuse_adapter_path: str = ""     # path to existing LoRA to merge into base model before new training
    prefuse_adapter_scale: float = 1.0 # scaling factor for the pre-fused adapter weights

    # === AutoController (Level 4) ===
    ac_enabled: bool = False
    ac_enable_smart_early_stopping: bool = False
    ac_early_stopping_patience: int = 5
    ac_enable_smart_lr_decay: bool = False
    ac_lr_decay_factor: float = 0.5
    ac_enable_auto_te_freeze: bool = False
    ac_te_freeze_step: int = 0
    ac_enable_dynamic_loss_scaling: bool = False
    ac_enable_auto_lr_adjustment: bool = False
    ac_auto_lr_scale_factor: float = 1.0
    ac_early_stopping_threshold: float = 0.001
    # Hidden AC Params (Audit Fix)
    ac_clip_drift_warning: float = 0.03
    ac_clip_drift_danger: float = 0.05
    ac_clip_drift_consecutive: int = 5
    ac_stable_rank_collapse_threshold: float = 0.3
    ac_stable_rank_consecutive: int = 10
    ac_loss_plateau_window: int = 50
    ac_gradient_rank_plateau_window: int = 30
    ac_max_decays: int = 3
    ac_target_gsnr: float = 5.0
    ac_target_loss: float = 0.0  # target loss for AutoController
    ac_batch_size_step: int = 1
    ac_warmup_steps: int = 100

    # === MN-LoRA (Level 4) ===
    mn_lora_enabled: bool = False
    mn_lora_preset: str = "slim"  # slim | fast | balanced | quality
    mn_lora_gsp_enabled: bool = True
    mn_lora_tgwd_enabled: bool = True
    mn_lora_k_ratio: float = 0.5
    mn_lora_update_interval: int = 20
    mn_lora_adaptive_k: bool = True
    mn_lora_lazy_update: bool = True
    mn_lora_svd_mode: str = "incremental"
    mn_lora_v_cache_mode: str = "tiered"
    mn_lora_v_cache_threshold: int = 64
    mn_lora_precondition_mode: str = "grad_ema"  # none | svd | grad_ema | hybrid
    mn_lora_pilot_strategy: str = "population"
    mn_lora_pilot_aggressiveness: float = 0.5
    # Hidden MN-LoRA Params
    mn_lora_residual_threshold: float = 0.3
    mn_lora_min_k_ratio: float = 0.2
    mn_lora_max_k_ratio: float = 0.8
    mn_lora_lazy_threshold: float = 0.5
    mn_lora_tgwd_alpha: float = 1.0
    mn_lora_tgwd_n_probes: int = 1
    mn_lora_tgwd_probe_interval: int = 50
    mn_lora_tgwd_finite_diff_eps: float = 1e-3
    mn_lora_svd_precond_beta: float = 0.5
    mn_lora_precond_min_scale: float = 0.25
    mn_lora_precond_max_scale: float = 4.0
    mn_lora_coord_curv_beta: float = 0.95
    mn_lora_precond_clip: float = 3.0
    mn_lora_precond_eps: float = 1e-6
    mn_lora_adaptive_sparse_enabled: bool = True
    mn_lora_adaptive_sparse_warmup_steps: int = 10
    mn_lora_adaptive_sparse_refresh_interval: int = 20
    mn_lora_adaptive_sparse_hot_ratio: float = 0.20
    mn_lora_adaptive_sparse_warm_ratio: float = 0.0
    mn_lora_adaptive_sparse_warm_interval: int = 4
    mn_lora_adaptive_sparse_cold_interval: int = 16
    mn_lora_adaptive_sparse_min_hot_layers: int = 16
    mn_lora_adaptive_sparse_zero_cold_after: int = 3
    mn_lora_plus_plus_enabled: bool = False
    mn_lora_plus_plus_profile: str = "balanced"  # safe | balanced | aggressive | custom
    mn_lora_plus_plus_rank_adapt: bool = True
    mn_lora_plus_plus_module_adapt: bool = True
    mn_lora_plus_plus_lr_up: float = 1.01
    mn_lora_plus_plus_lr_down: float = 0.95
    mn_lora_plus_plus_min_mult: float = 0.25
    mn_lora_plus_plus_max_mult: float = 2.0
    mn_lora_plus_plus_lora_up_max_mult: float = 1.25
    mn_lora_plus_plus_protected_max_mult: float = 1.0
    mn_lora_plus_plus_update_rms_cap: float = 0.01
    mn_lora_kfac_lite_enabled: bool = False
    mn_lora_kfac_lite_ema_decay: float = 0.95
    mn_lora_kfac_lite_damping: float = 1e-3
    mn_lora_kfac_lite_update_interval: int = 1
    mn_lora_kfac_lite_precondition_interval: int = 1
    mn_lora_kfac_lite_max_samples: int = 2048
    mn_lora_kfac_lite_grad_clip: float = 3.0
    mn_lora_kfac_lite_stacked_grad_clip: float = 2.0
    mn_lora_kfac_lite_active_ratio: float = 0.40
    mn_lora_kfac_lite_warmup_steps: int = 10
    mn_lora_kfac_lite_refresh_interval: int = 10
    mn_lora_kfac_lite_min_active_modules: int = 16
    mn_lora_trust_region_enabled: bool = True
    mn_lora_trust_region_max_update_rms_ratio: float = 0.01
    mn_lora_trust_region_max_update_norm_ratio: float = 0.10
    mn_lora_trust_region_hotspot_only: bool = False
    mn_lora_effective_delta_enabled: bool = True
    mn_lora_effective_delta_clip_enabled: bool = True
    mn_lora_effective_delta_max_norm_ratio: float = 0.25
    mn_lora_effective_delta_max_rms_ratio: float = 0.05
    mn_lora_effective_delta_fisher_weighted: bool = True
    mn_lora_effective_delta_fisher_beta: float = 0.95
    mn_lora_effective_delta_fisher_strength: float = 1.0
    mn_lora_effective_delta_fisher_max_weight: float = 4.0
    mn_lora_fisher_ewc_enabled: bool = True
    mn_lora_fisher_ewc_lambda: float = 1e-4
    mn_lora_fisher_ewc_beta: float = 0.95
    mn_lora_fisher_ewc_start_step: int = 1
    mn_lora_fisher_ewc_update_interval: int = 5
    mn_lora_fisher_ewc_max_penalty_norm_ratio: float = 0.25
    mn_lora_gradient_conflict_enabled: bool = False
    mn_lora_gradient_conflict_threshold: float = 0.0
    mn_lora_gradient_conflict_protect_main: bool = True

    # === Training Pilot (Level 4) ===
    pilot_threshold_high_norm: float = 2.0
    pilot_threshold_low_grad: float = 1e-6
    pilot_threshold_high_grad: float = 1.0
    pilot_lr_multiplier_cool: float = 0.5
    pilot_lr_multiplier_boost: float = 2.0
    pilot_lr_multiplier_normal: float = 1.0
    pilot_max_lr_change_per_step: float = 0.1
    pilot_history_length: int = 10

    # === Smart Caption (Level 2/3) ===
    sc_trigger_dropout: float = 0.0
    sc_style_dropout: float = 0.05
    sc_quality_dropout: float = 0.3
    sc_content_dropout: float = 0.15
    sc_modifier_dropout: float = 0.2
    sc_locked_tags: str = "" # Comma separated

    # === Auditor & Watchdog (Level 4) ===
    auditor_rsvd_k_pro: int = 50
    auditor_rsvd_k_lite: int = 10
    auditor_sample_interval_pro: int = 50
    auditor_sample_interval_lite: int = 100
    auditor_dead_neuron_epsilon: float = 1e-5
    auditor_dead_neuron_stride_pro: int = 10
    auditor_dead_neuron_stride_lite: int = 100
    watchdog_learning_steps: int = 50
    watchdog_vram_threshold_stop: float = 0.98
    watchdog_vram_threshold_lite: float = 0.95

    # === Newbie AI ===
    # Newbie uses a diffusers-format model with Gemma3 prompting.
    # The lora_target preset controls which transformer layers receive adapters.
    newbie_diffusers_path: str = ""
    newbie_lora_target: NewbieLoraTarget = NewbieLoraTarget.BALANCED
    newbie_gemma3_prompt: str = ""
    newbie_use_flash_attn2: bool = True
    newbie_dataloader_workers: int = 4
    prune_warmup_ratio: float = 0.15

    # Newbie native component paths (for non-diffusers bundle layout)
    newbie_transformer_path: str = ""
    newbie_gemma_model_path: str = ""
    newbie_clip_model_path: str = ""
    newbie_vae_path: str = ""
    newbie_target_scope: str = "layer0_attention"
    newbie_target_modules: str = ""  # comma-separated custom target module list
    newbie_gemma_max_token_length: int = 512
    newbie_clip_max_token_length: int = 2048
    newbie_caption_length_bucket_size: int = 0  # 0 = disabled
    use_cache: bool = False  # legacy Newbie cache toggle; distinct from force-cache-only
    newbie_force_cache_only: bool = False
    newbie_rebuild_cache: bool = False
    newbie_cached_latent_crop_size: int = 0
    newbie_cached_text_token_limit: int = 0
    newbie_fixed_text_tokens: int = 0  # 0 = dynamic batch pad / cached length
    newbie_fixed_visual_tokens: int = 0  # 0 = keep cached spatial tokens
    newbie_block_residency: str = "resident"  # resident | streaming_offload | block_cpu_pinned
    newbie_block_residency_min_params: int = 0
    newbie_block_checkpointing: bool = False  # recompute native Newbie DiT blocks during backward
    newbie_block_checkpointing_mode: str = "block"  # block
    newbie_block_prefetch: bool = False  # async prefetch CPU-pinned DiT Linear weights for streaming_offload
    newbie_block_prefetch_depth: int = 1  # number of future DiT blocks to prefetch
    newbie_auto_swap_release: bool = False
    newbie_safe_fallback: bool = True  # Clear CUDA state and fail safely on OOM
    newbie_run_native_smoke: bool = False  # explicit tiny transformer smoke gate; off by default
    newbie_adapter_type: str = ""  # "lora", "lora_fa", "vera", "lokr" (empty = default lora)
    trust_remote_code: bool = False  # retained for native component loaders that need HF custom code
    pytorch_cuda_expandable_segments: bool = False  # launcher/runtime env hint

    # === Anima ===
    # Anima's native route loads a primary checkpoint plus an auxiliary
    # Qwen3 text encoder and a dedicated T5 tokenizer.
    anima_model_path: str = ""
    anima_qwen3_path: str = ""
    anima_t5_tokenizer_path: str = ""
    gemma2: str = ""
    clip_g: str = ""
    anima_attn_mode: str = ""
    anima_cache_text_encoder_outputs_to_disk: bool = False
    anima_qwen3_max_token_length: int = 0  # Qwen3 max token length (0 = default)
    anima_t5_max_token_length: int = 0  # T5 max token length (0 = default)
    gemma2_max_token_length: int = 0
    anima_llm_adapter_path: str = ""  # LLM adapter checkpoint path (applied to Qwen3)
    anima_dit_adapter_path: str = ""  # DiT adapter checkpoint path (applied to Anima DiT before training)
    anima_split_attn: bool = False  # split attention for memory savings
    anima_vae_chunk_size: int = 0  # VAE chunk size (0 = disabled)
    anima_text_token_limit: int = 0  # text token limit for cache generation (0 = no truncation)
    anima_vae_disable_cache: bool = False  # disable VAE caching
    anima_unsloth_offload: bool = False  # Unsloth-style CPU offloading
    anima_sigmoid_scale: float = 1.0  # sigmoid scale for timestep sampling
    anima_weighting_scheme: str = ""  # loss weighting scheme
    anima_guidance_scale: float = 1.0  # CFG guidance scale for Anima flow training
    anima_model_prediction_type: str = "velocity"  # velocity, noise, epsilon, sample
    anima_mode_scale: float = 1.0  # scale for "mode" weighting scheme
    # JLT-style EMA feature self-distillation alignment (default-off reserve,
    # arXiv:2605.27102). Teacher = EMA-of-LoRA shadow evaluated at a smaller
    # (cleaner) timestep; selected DiT block features are cosine-aligned.
    # Layers are comma-separated block indices (teacher/student counts must
    # match). Forces an eager forward when enabled (bypasses cudagraph/offload).
    #
    # STATUS: TECHNICAL RESERVE — intentionally NOT exposed in the webui.
    # No positive-benefit evidence for LoRA fine-tuning (JLT's own headline
    # result does not use it; it only appears in an async-teacher side script
    # with no ablation). The cost is certain (≈2x forward + loses
    # cudagraph/offload). Keep runtime-only until a real anima-LoRA A/B proves
    # a gain; do not surface in UI before then. (2026-06-11 user decision)
    anima_ema_feat_align_enabled: bool = False
    anima_ema_feat_align_weight: float = 0.0
    anima_ema_feat_align_teacher_layers: str = ""  # e.g. "9"
    anima_ema_feat_align_student_layers: str = ""  # e.g. "4"
    anima_ema_feat_align_decay: float = 0.9999
    # EDM2-style learned per-sigma loss balancing (arXiv:2312.02696 eq.17) for
    # the anima flow route. Default off; when enabled a tiny Fourier+linear
    # head learns u(sigma) and the loss becomes loss/exp(u)+u (adaptive
    # timestep weighting, no manual scheme tuning).
    flow_uncertainty_weighting_enabled: bool = False
    flow_uncertainty_weighting_lr: float = 1e-2  # separate no-decay param group lr
    flow_uncertainty_weighting_channels: int = 128  # Fourier feature bank size
    anima_cached_training: bool = True  # cache-first native DiT training path
    anima_online_cache: bool = False  # generate missing Anima cache files from raw sidecars and persist them
    anima_cached_latent_crop_size: int = 0  # 0 = full cached latent, >0 = smoke/debug crop
    anima_cached_text_token_limit: int = 0  # 0 = full cached text, >0 = smoke/debug prefix
    anima_native_block_count: int = 28  # preview2 full DiT block count
    # Opt-in faithful native forward for training (#147): feeds t=sigma in [0,1]
    # (not sigma*1000), runs the frozen llm_adapter to build cross-attn context
    # (not raw Qwen3 hidden), and enables 3D-RoPE self-attention. Default False ->
    # the legacy (#132) path is bitwise-unchanged. Requires block-checkpoint /
    # cache / reducer seams off and a cache carrying t5_input_ids.
    anima_faithful_forward: bool = False
    anima_full_finetune_phase: str = "dit_only_cache_first"
    anima_full_finetune_train_text_encoder_requested: bool = False
    anima_full_finetune_text_encoder_policy: str = "dit_only"
    anima_block_residency: str = "resident"  # resident | streaming_offload | block_cpu_pinned
    anima_block_residency_min_params: int = 0  # minimum frozen Linear parameter count per base layer
    anima_block_checkpointing: bool = False  # recompute native Anima DiT blocks during backward
    anima_block_checkpointing_mode: str = "block"  # block | selective (op-level SAC: keep matmul/SDPA, recompute elementwise)
    anima_block_checkpointing_interval: int = 1  # checkpoint every Nth block (1 = all blocks); N>1 trades VRAM for less recompute
    anima_block_prefetch: bool = False  # async prefetch CPU-pinned DiT Linear weights for streaming_offload
    anima_block_prefetch_depth: int = 1  # number of future DiT blocks to prefetch
    anima_block_prefetch_mode: str = "original"  # original (fixed-depth) | adaptive (blockskip-aware + online depth-adaptive); default = byte-parity
    anima_train_llm_adapter: bool = False  # ordinary Anima LoRA keeps LLM adapter frozen/cached unless explicitly requested
    anima_fixed_text_tokens: int = 0  # 0 = dynamic batch pad, e.g. 512 for Anima fast profile
    anima_fixed_visual_tokens: int = 0  # 0 = keep cached size, e.g. 4096 for static-shape profile
    anima_cache_llm_adapter_outputs: bool = False
    anima_caption_variant_cache: bool = False
    anima_compile_scope: str = ""  # empty | per_block | full_cudagraph
    anima_fused_qkv: bool = False  # fuse Q/K/V projections in Anima DiT self-attention
    anima_merge_export: bool = False
    reft_enabled: bool = False
    reft_target_modules: str = ""
    reft_rank: int = 8
    reft_init_scale: float = 0.0
    hydralora_enabled: bool = False
    hydralora_num_experts: int = 4
    hydralora_routing: str = "top_k"
    hydralora_top_k: int = 2
    hydralora_sparse_top_k: bool = False  # Experimental: compute only selected top-k experts; benchmark before enabling.
    hydralora_balance_loss_weight: float = 0.0
    easy_control_enabled: bool = False
    control_image_dir: str = ""
    control_suffix: str = ""
    easy_control_scale: float = 1.0
    easy_control_channels: int = 3
    easycontrol_v2_enabled: bool = False
    easycontrol_v2_task_id: str = "generic"
    easycontrol_v2_control_kind: str = "reference_latent"
    easycontrol_v2_target_family: str = ""  # empty = infer from model_type
    easycontrol_v2_cond_cache_dir: str = ""
    easycontrol_v2_text_cache_dir: str = ""
    easycontrol_v2_control_image_dir: str = ""
    easycontrol_v2_control_suffix: str = ""
    easycontrol_v2_drop_p: float = 0.1
    easycontrol_v2_cond_noise_max: float = 0.0
    easycontrol_v2_scale: float = 1.0
    easycontrol_v2_match_target_bucket: bool = False
    ip_adapter_enabled: bool = False
    ip_adapter_encoder_dim: int = 1024
    ip_adapter_cond_dim: int = 1152
    ip_adapter_num_image_tokens: int = 16
    ip_adapter_scale: float = 1.0
    ip_adapter_cond_mode: str = "concat"
    fera_enabled: bool = False
    fera_gate_init: float = 0.0

    # === Merged Checkpoint Export (#121) ===
    merge_export: bool = False  # generic merge-export toggle (anima_merge_export takes precedence for Anima)

    # === Prefix/Postfix Tuning (#113) ===
    prefix_tuning_length: int = 0  # number of learnable prefix soft-prompt tokens (0 = disabled)
    postfix_tuning_length: int = 0  # number of learnable postfix soft-prompt tokens (0 = disabled)
    prefix_tuning_init: str = "normal"  # initialisation: "normal", "uniform", "zeros"

    # AdaLN Guidance (#114) — learnable bias on DiT AdaLN modulation
    adaln_guidance: bool = False
    adaln_guidance_init_scale: float = 0.0  # 0 = no-op at start

    # Anima finetune grouped learning rates
    anima_self_attn_lr: float = 0.0
    anima_cross_attn_lr: float = 0.0
    anima_mlp_lr: float = 0.0
    anima_mod_lr: float = 0.0
    anima_llm_adapter_lr: float = 0.0

    # === Neutral Native Plumbing ===
    # Generic fields that any future model family can use without adding
    # family-prefixed entries.  The per-family fields above take precedence
    # when both are set for the same purpose.
    native_secondary_model_path: str = ""
    native_tokenizer_path: str = ""
    native_attn_mode: str = ""
    native_cache_te_to_disk: bool = False
    hf_token_env: str = ""
    gpu_ids: str = ""
    prefer_json_caption: bool = False
    vae_batch_size: int = 0
    training_shift: float = 0.0
    unsloth_offload_checkpointing: bool = False
    freeze_extractors: bool = False
    lulynx_auto_check_every: int = 0
    lulynx_resource_log_interval: int = 0
    lulynx_smart_rank_keep_ratio: float = 0.0
    lulynx_block_lr_zero_threshold: float = 0.0
    lulynx_anima_block_lr_weights: str = ""
    lulynx_anima_llm_adapter_lr_weight: float = 1.0
    network_args: Union[str, List[str]] = Field(default_factory=list)
    lulynx_anima_final_layer_lr_weight: float = 1.0
    lulynx_anima_norm_lr_weight: float = 1.0
    lowram: bool = False
    no_half_vae: bool = False
    text_encoder_cpu: bool = False
    train_t5xxl: bool = False
    use_flash_attn: bool = False
    use_sage_attn: bool = False
    mem_eff_save: bool = False
    sample_batch_size: int = 0
    clip_l_dropout_rate: float = 0.0
    clip_g_dropout_rate: float = 0.0
    t5_dropout_rate: float = 0.0
    apply_lg_attn_mask: bool = False
    apply_t5_attn_mask: bool = False
    peak_vram_control_enabled: bool = False
    peak_vram_target_effective_batch: int = 0
    peak_vram_startup_guard_enabled: bool = False
    peak_vram_startup_guard_mode: str = "auto"
    peak_vram_startup_guard_steps: int = 0
    peak_vram_micro_batch_enabled: bool = False
    peak_vram_micro_batch_size: int = 1
    peak_vram_diagnostics_enabled: bool = False
    peak_vram_auto_protection_enabled: bool = False

    # === T-LoRA (Added) ===
    t_lora_enabled: bool = False
    t_lora_max_t: int = 1000
    tlora_min_rank: int = 1
    tlora_rank_schedule: str = "constant"  # "constant", "linear", "geometric"
    tlora_orthogonal_init: bool = False

    # === Thermal Management (Added) ===
    pause_enabled: bool = False
    pause_every_epochs: int = 3
    pause_duration_minutes: int = 5

    # === PCGrad (Level 4) ===
    pcgrad_enabled: bool = False
    pcgrad_conflict_threshold: float = 0.0
    pcgrad_reduction: str = "mean"

    # === Lulynx LAB (Level 4) ===
    lulynx_geometric_lock: bool = False
    lulynx_proj_dim: int = 128
    lulynx_manifold_weight: float = 0.01
    lulynx_manifold_sparse_freq: int = 1
    lulynx_anchor_layers: str = ""
    lulynx_ln_guard: bool = False
    lulynx_ln_lambda: float = 0.01
    lulynx_anti_fry: bool = False
    lulynx_ghost_replay: bool = False
    lulynx_ghost_path: str = ""
    lulynx_ghost_interval: int = 100
    lulynx_ghost_weight: float = 0.05
    lulynx_hutchinson_probes: int = 30
    pruning_enabled: bool = False
    dynamic_pruning_enabled: bool = False
    pruning_interval: int = 100
    pruning_target_ratio: float = 0.5

    # Vortex Memory (R&D)
    vortex_enabled: bool = False
    vortex_strategy: str = "standard"
    vortex_profile: str = "standard"  # "standard" | "low_vram" | "extreme"

    # === Monitoring (Added) ===
    advanced_stats_enabled: bool = False
    svd_algorithm: str = "rsvd"
    svd_sample_interval: int = 50
    ui_refresh_rate: int = 2

    # === 高级训练监控 ===
    advanced_monitoring_enabled: bool = False
    peak_vram_diagnostics_interval: int = 25
    cuda_cache_release_strategy: str = "oom_only"
    cuda_cache_release_interval: int = 1
    audit_mode_override: str = ""
    attn_entropy_interval: int = 100
    act_drift_interval: int = 100
    act_drift_anchor_layers: str = ""
    lr_finder_enabled: bool = False
    lr_finder_start_lr: float = 1e-7
    lr_finder_end_lr: float = 1e-1
    lr_finder_num_steps: int = 100

    # === Deep Diagnostics ===
    deep_diagnostics_enabled: bool = False
    hessian_trace_interval: int = 200
    grad_cosine_enabled: bool = False

    # === Forgetting Probe ===
    forgetting_probe_enabled: bool = False
    forgetting_probe_interval: int = 50
    forgetting_probe_num_anchors: int = 4

    # === Manifold Tracker ===
    manifold_enabled: bool = False
    manifold_snapshot_interval: int = 20

    # === Advanced Training Methods (Store Aliases) ===
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    # === Semantic Base-Tuner (Level 5) ===
    semantic_tuner_enabled: bool = False
    semantic_llm_path: str = ""
    semantic_projector_path: str = ""
    architecture_mode: str = "hybrid" # hybrid | pure

    model_config = ConfigDict(
        use_enum_values=True,
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTrainingConfig':
        """Legacy compatibility method for dataclass-like from_dict"""
        return cls.model_validate(data)

    @model_validator(mode="before")
    @classmethod
    def _sync_concept_geometry_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        optimizer_type = str(normalized.get("optimizer_type", "") or "").strip()
        optimizer_key = optimizer_type.lower().replace(" ", "")
        plugin_optimizer_name = ""
        if optimizer_key.startswith("pytorch_optimizer."):
            plugin_optimizer_name = optimizer_type.split(".", 1)[1].strip()
        elif optimizer_key.startswith("pytorchoptimizer:") or optimizer_key.startswith("pytorchoptimizer/"):
            plugin_optimizer_name = optimizer_type.split(optimizer_type[16], 1)[1].strip()
        generic_optimizer_name = ""
        if optimizer_key.startswith("genericoptimizer:") or optimizer_key.startswith("genericoptimizer/"):
            generic_optimizer_name = optimizer_type.split(optimizer_type[16], 1)[1].strip()
        elif optimizer_key.startswith("bitsandbytes.optim."):
            generic_optimizer_name = optimizer_type
        if plugin_optimizer_name:
            normalized["optimizer_type"] = OptimizerType.PYTORCH_OPTIMIZER.value
            raw_args = normalized.get("optimizer_args", "")
            if isinstance(raw_args, (list, tuple, set, frozenset)):
                arg_parts = [str(item).strip() for item in raw_args if str(item).strip()]
            else:
                arg_parts = [
                    part.strip()
                    for part in str(raw_args or "").replace("\n", " ").replace(",", " ").split(" ")
                    if part.strip()
                ]
            has_name_arg = any(
                part.split("=", 1)[0].strip().lower() in {"name", "optimizer_name", "optimizer"}
                for part in arg_parts
                if "=" in part
            )
            if not has_name_arg:
                arg_parts.insert(0, f"name={plugin_optimizer_name}")
            normalized["optimizer_args"] = " ".join(arg_parts)
        elif generic_optimizer_name:
            normalized["optimizer_type"] = OptimizerType.GENERIC.value
            raw_args = normalized.get("optimizer_args", "")
            if isinstance(raw_args, (list, tuple, set, frozenset)):
                arg_parts = [str(item).strip() for item in raw_args if str(item).strip()]
            else:
                arg_parts = [
                    part.strip()
                    for part in str(raw_args or "").replace("\n", " ").replace(",", " ").split(" ")
                    if part.strip()
                ]
            has_name_arg = any(
                part.split("=", 1)[0].strip().lower() in {"name", "optimizer_name", "optimizer"}
                for part in arg_parts
                if "=" in part
            )
            if not has_name_arg:
                arg_parts.insert(0, f"name={generic_optimizer_name}")
            normalized["optimizer_args"] = " ".join(arg_parts)
        elif optimizer_key in {"automagic", "automagic++"}:
            normalized["optimizer_type"] = OptimizerType.AUTOMAGIC_PLUS_PLUS.value
        elif optimizer_key in {"autoprodigy", "auto_prodigy"}:
            normalized["optimizer_type"] = OptimizerType.AUTO_PRODIGY.value
        elif optimizer_key in {"adafactor", "ada_factor"}:
            normalized["optimizer_type"] = OptimizerType.ADAFACTOR.value
        elif optimizer_key in {"animafactoredadamw", "anima_factored_adamw", "anima_factored"}:
            normalized["optimizer_type"] = OptimizerType.ANIMA_FACTORED_ADAMW.value
        elif optimizer_key == "pytorchoptimizer":
            normalized["optimizer_type"] = OptimizerType.PYTORCH_OPTIMIZER.value
        elif optimizer_key == "dadaptadampreprint":
            normalized["optimizer_type"] = OptimizerType.DADAPT_ADAM_PREPRINT.value
        elif optimizer_key in {
            "prodigyplus.prodigyplusschedulefree",
            "prodigyplusschedulefree",
            "prodigyschedulefree",
            "prodigy_schedule_free",
        }:
            normalized["optimizer_type"] = OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE.value
        optimizer_backend = str(normalized.get("optimizer_backend", "auto") or "auto").strip().lower().replace("-", "_")
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
            "torchao": "ao_8bit",
            "torchao_8bit": "ao_8bit",
            "ao": "ao_8bit",
            "compile": "compiled_step",
            "compiled": "compiled_step",
            "lulynx": "lulynx_fused",
            "lulynx_fused_adamw": "lulynx_fused",
        }
        optimizer_backend = optimizer_backend_aliases.get(optimizer_backend.replace(" ", ""), optimizer_backend)
        if optimizer_backend not in {"auto", "torch_adamw", "foreach_adamw", "torch_fused", "bnb_8bit", "ao_8bit", "compiled_step", "apex", "lulynx_fused"}:
            optimizer_backend = "auto"
        normalized["optimizer_backend"] = optimizer_backend
        advanced_optimizer_strategy = str(normalized.get("advanced_optimizer_strategy", "auto") or "auto").strip().lower().replace("-", "_").replace("+", "_plus")
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
        advanced_optimizer_strategy = advanced_optimizer_strategy_aliases.get(advanced_optimizer_strategy.replace(" ", ""), advanced_optimizer_strategy)
        if advanced_optimizer_strategy not in {"auto", "off", "profile_only", "lora_plus", "rs_lora", "galore"}:
            advanced_optimizer_strategy = "auto"
        normalized["advanced_optimizer_strategy"] = advanced_optimizer_strategy

        network_module = str(normalized.get("network_module", "") or "").strip().lower()
        network_module = network_module.replace("_", "-") if network_module in {"diag_oft"} else network_module
        if network_module == "lora":
            normalized["network_module"] = NetworkType.LORA.value
        elif network_module in {"lycoris", "lycoris.kohya"}:
            normalized["network_module"] = NetworkType.LYCORIS.value
        elif network_module in {"networks.oft", "oft", "diag-oft"}:
            normalized["network_module"] = NetworkType.LYCORIS.value
            normalized["lycoris_algo"] = LyCORISAlgo.DIAG_OFT.value

        if "lycoris_algo" in normalized:
            lycoris_algo = str(normalized.get("lycoris_algo") or LyCORISAlgo.LOHA.value).strip().lower().replace("_", "-")
            lycoris_aliases = {
                "lycoris-lokr": "lokr",
                "lycoris-loha": "loha",
                "lycoris-locon": "locon",
                "lycoris-ia3": "ia3",
                "lycoris-full": "full",
                "lycoris-diag-oft": "diag-oft",
                "lycoris-glora": "glora",
                "lycoris-glokr": "glokr",
                "oft": "diag-oft",
            }
            normalized["lycoris_algo"] = lycoris_aliases.get(lycoris_algo, lycoris_algo)

        adapter_init_strategy = str(
            normalized.get("adapter_init_strategy", normalized.get("init_lora_weights", "default")) or "default"
        ).strip().lower().replace("-", "_")
        adapter_init_aliases = {
            "": "default",
            "none": "default",
            "off": "default",
            "disabled": "default",
            "standard": "default",
            "kaiming": "default",
            "default_lora": "default",
            "pissa_init": "pissa",
            "pissa": "pissa",
            "olora": "olora",
            "o_lora": "olora",
            "orthogonal_lora": "olora",
            "loftq": "loftq",
            "loft_q": "loftq",
            "loftq_init": "loftq",
        }
        adapter_init_strategy = adapter_init_aliases.get(adapter_init_strategy.replace(" ", ""), adapter_init_strategy)
        if adapter_init_strategy not in {item.value for item in AdapterInitStrategy}:
            adapter_init_strategy = "default"
        pissa_requested = (
            _config_boolish(normalized.get("pissa_enabled"))
            or _config_boolish(normalized.get("use_pissa"))
            or _config_boolish(normalized.get("pissa_init"))
        )
        if adapter_init_strategy == "default" and pissa_requested:
            adapter_init_strategy = "pissa"
        normalized["adapter_init_strategy"] = adapter_init_strategy
        normalized["pissa_enabled"] = adapter_init_strategy == "pissa"
        normalized["use_pissa"] = adapter_init_strategy == "pissa"

        try:
            normalized["loftq_bits"] = min(max(int(normalized.get("loftq_bits", 4) or 4), 2), 8)
        except (TypeError, ValueError):
            normalized["loftq_bits"] = 4
        loftq_quant_type = str(normalized.get("loftq_quant_type", "rowwise") or "rowwise").strip().lower().replace("-", "_")
        loftq_quant_aliases = {
            "": "rowwise",
            "default": "rowwise",
            "uniform": "rowwise",
            "symmetric": "rowwise",
            "per_channel": "rowwise",
            "per_output": "rowwise",
            "global": "tensorwise",
            "per_tensor": "tensorwise",
        }
        loftq_quant_type = loftq_quant_aliases.get(loftq_quant_type.replace(" ", "_"), loftq_quant_type)
        normalized["loftq_quant_type"] = loftq_quant_type if loftq_quant_type in {"rowwise", "tensorwise"} else "rowwise"

        pissa_svd_algo = str(normalized.get("pissa_svd_algo", normalized.get("pissa_method", "rsvd")) or "rsvd").strip().lower()
        pissa_svd_aliases = {"svd": "full", "full_svd": "full", "lowrank": "rsvd", "randomized": "rsvd"}
        pissa_svd_algo = pissa_svd_aliases.get(pissa_svd_algo.replace("-", "_"), pissa_svd_algo)
        normalized["pissa_svd_algo"] = pissa_svd_algo if pissa_svd_algo in {"rsvd", "full"} else "rsvd"
        try:
            normalized["pissa_oversample"] = max(0, int(normalized.get("pissa_oversample", 8) or 0))
        except (TypeError, ValueError):
            normalized["pissa_oversample"] = 8
        normalized["pissa_apply_conv2d"] = _config_boolish(normalized.get("pissa_apply_conv2d"))
        export_mode = str(normalized.get("pissa_export_mode", "lora_compatible") or "lora_compatible").strip().lower()
        export_aliases = {
            "lora无损兼容导出": "lora_compatible",
            "lora_compatible_export": "lora_compatible",
            "compatible": "lora_compatible",
            "native": "lora_compatible",
            "lora快速近似导出": "approximate",
            "fast": "approximate",
            "quick": "approximate",
        }
        export_mode = export_aliases.get(export_mode.replace(" ", "_"), export_mode)
        normalized["pissa_export_mode"] = export_mode if export_mode in {"lora_compatible", "approximate"} else "lora_compatible"
        adapter_init_export_mode = str(
            normalized.get("adapter_init_export_mode", normalized.get("init_lora_weights_export_mode", "auto")) or "auto"
        ).strip().lower().replace("-", "_")
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
        adapter_init_export_mode = init_export_aliases.get(adapter_init_export_mode.replace(" ", "_"), adapter_init_export_mode)
        normalized["adapter_init_export_mode"] = (
            adapter_init_export_mode if adapter_init_export_mode in {"auto", "raw", "lora_compatible", "approximate"} else "auto"
        )

        acceleration_profile = normalize_acceleration_profile(
            normalized.get("acceleration_profile", normalized.get("speed_profile", "off"))
        )
        normalized["acceleration_profile"] = acceleration_profile
        normalized["speed_profile"] = acceleration_profile
        if acceleration_profile != "off":
            acceleration = apply_model_acceleration_policy_to_config(
                normalized,
                schema_id=str(normalized.get("schema_id", "") or ""),
                training_type=str(normalized.get("training_type", "") or ""),
            )
            normalized = acceleration.config

        runtime_resolution = resolve_runtime_optimization_payload(normalized)
        normalized.update(runtime_resolution.fields)
        data_backend = str(normalized.get("data_backend", "auto") or "auto").strip().lower().replace("-", "_")
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
            data_backend = "auto"
        normalized["data_backend"] = data_backend
        image_decode_backend = str(normalized.get("image_decode_backend", "pil") or "pil").strip().lower().replace("-", "_")
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
            image_decode_backend = "pil"
        normalized["image_decode_backend"] = image_decode_backend
        try:
            normalized["image_decode_cache_size"] = max(int(normalized.get("image_decode_cache_size", 0) or 0), 0)
        except (TypeError, ValueError):
            normalized["image_decode_cache_size"] = 0
        cached_collate_mode = str(normalized.get("cached_collate_mode", "auto") or "auto").strip().lower().replace("-", "_")
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
            cached_collate_mode = "auto"
        normalized["cached_collate_mode"] = cached_collate_mode
        flux_transformer_offload = str(normalized.get("flux_transformer_offload", "auto") or "auto").strip().lower().replace("-", "_")
        flux_transformer_offload = {
            "": "auto",
            "default": "auto",
            "none": "off",
            "false": "off",
            "0": "off",
            "disabled": "off",
            "low_vram": "aggressive",
            "sequential": "aggressive",
            "streaming": "aggressive",
        }.get(flux_transformer_offload.replace(" ", ""), flux_transformer_offload)
        if flux_transformer_offload not in {"auto", "off", "aggressive"}:
            flux_transformer_offload = "auto"
        normalized["flux_transformer_offload"] = flux_transformer_offload
        checkpoint_policy = str(normalized.get("checkpoint_policy", "auto") or "auto").strip().lower().replace("-", "_")
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
            checkpoint_policy = "auto"
        normalized["checkpoint_policy"] = checkpoint_policy
        fused_projection_memory_mode = str(
            normalized.get("fused_projection_memory_mode", "keep_original") or "keep_original"
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
            fused_projection_memory_mode = "keep_original"
        normalized["fused_projection_memory_mode"] = fused_projection_memory_mode
        pairs = (
            ("concept_geometry_enabled", "h_lora_enabled"),
            ("concept_geometry_path", "h_lora_geometry_path"),
            ("concept_geometry_sampler_mode", "h_lora_sampler_mode"),
            ("concept_geometry_loss_weighting", "h_lora_loss_weighting"),
            ("concept_geometry_density_power", "h_lora_density_power"),
        )
        for modern, legacy in pairs:
            if modern in normalized and legacy not in normalized:
                normalized[legacy] = normalized[modern]
            elif legacy in normalized and modern not in normalized:
                normalized[modern] = normalized[legacy]
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Legacy compatibility method for dataclass-like to_dict"""
        return self.model_dump()

    @property
    def base_model_path(self) -> str:
        """Alias for Lulynx compatibility"""
        return self.pretrained_model_name_or_path

    @base_model_path.setter
    def base_model_path(self, value: str):
        self.pretrained_model_name_or_path = value

    # === Compatibility Property Aliases (Lulynx -> Unified) ===
    @property
    def model_arch(self) -> ModelArch:
        value = self.model_type
        return value if isinstance(value, ModelArch) else ModelArch(value)

    @property
    def network_type(self) -> NetworkType:
        value = self.network_module
        return value if isinstance(value, NetworkType) else NetworkType(value)

    @property
    def epochs(self) -> int: return self.max_train_epochs

    @property
    def batch_size(self) -> int: return self.train_batch_size

    @property
    def gradient_accumulation(self) -> int: return self.gradient_accumulation_steps

    @property
    def use_ema(self) -> bool: return self.ema_use_ema

    @use_ema.setter
    def use_ema(self, value: bool):
        self.ema_use_ema = bool(value)

    @property
    def optimizer(self) -> OptimizerType:
        value = self.optimizer_type
        return value if isinstance(value, OptimizerType) else OptimizerType(value)

    @property
    def scheduler(self) -> SchedulerType:
        value = self.lr_scheduler
        return value if isinstance(value, SchedulerType) else SchedulerType(value)

    @property
    def train_unet(self) -> bool:
        # "text_encoder_only" means UNet should NOT be trained.
        return not self.network_train_text_encoder_only

    @property
    def train_text_encoder(self) -> bool:
        # "unet_only" means text encoder should NOT be trained.
        return not self.network_train_unet_only

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        # NOTE: Pydantic handles type validation naturally.
        # This method is for business logic validation.

        model_arch = self.model_arch

        def _validate_optional_model_ref(value: str, label: str) -> None:
            if not value:
                return
            if "/" not in value and "\\" not in value:
                return
            try:
                if not Path(value).exists():
                    errors.append(f"{label}不存在: {value}")
            except OSError:
                errors.append(f"{label}路径格式无效: {value}")

        if model_arch == ModelArch.NEWBIE:
            has_diffusers = bool(self.newbie_diffusers_path)
            has_native_bundle = bool(self.newbie_transformer_path or self.pretrained_model_name_or_path)
            if not has_diffusers and not has_native_bundle:
                errors.append('Newbie 模式需要指定 Diffusers 模型路径或 Transformer 路径')
        elif model_arch == ModelArch.ANIMA:
            if not self.anima_model_path and not self.pretrained_model_name_or_path:
                errors.append('Anima 原生路线需要指定主模型路径')
        elif not self.pretrained_model_name_or_path:
            errors.append('未指定底模路径')

        _validate_optional_model_ref(self.pretrained_model_name_or_path, "底模文件")
        _validate_optional_model_ref(self.newbie_diffusers_path, "Newbie Diffusers 模型路径")
        _validate_optional_model_ref(self.newbie_transformer_path, "Newbie Transformer 路径")
        _validate_optional_model_ref(self.newbie_gemma_model_path, "Newbie Gemma 模型路径")
        _validate_optional_model_ref(self.newbie_clip_model_path, "Newbie CLIP 模型路径")
        _validate_optional_model_ref(self.anima_model_path, "Anima 主模型路径")
        _validate_optional_model_ref(self.anima_qwen3_path, "Anima Qwen3 路径")
        _validate_optional_model_ref(self.anima_t5_tokenizer_path, "Anima T5 tokenizer 路径")
        _validate_optional_model_ref(self.anima_llm_adapter_path, "Anima LLM Adapter 路径")

        if not self.train_data_dir:
            errors.append('未指定训练数据目录')
        elif not Path(self.train_data_dir).exists():
            errors.append(f"训练数据目录不存在: {self.train_data_dir}")

        if not self.output_dir:
            errors.append("未指定输出目录")

        if self.network_alpha > self.network_dim and self.network_dim > 0:
            warnings.append(f'network_alpha({self.network_alpha}) > network_dim({self.network_dim}) 可能导致训练异常，建议 alpha <= dim')

        if self.mixed_precision == MixedPrecision.FP16 and self.save_precision == 'bf16':
            warnings.append('mixed_precision=fp16 但 save_precision=bf16，在旧显卡上可能崩溃')

        return (len(errors) == 0, errors, warnings)
