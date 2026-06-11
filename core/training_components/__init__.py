"""
Training Components — modular, importable training features.

All training-related features extracted from lulynx_trainer/ and lulynx/
into a single importable package. Each module is self-contained and can
be used independently.

Tier 1 (standalone, no refactoring needed):
  - ema: EMA / EMAStateTracker weight averaging
  - auto_controller: AutoController (TE freeze, early stop, LR decay)
  - smart_rank: SmartRankController dynamic rank adjustment
  - coreset: CoresetManager smart curriculum sampling
  - resource_manager: DynamicResourceManager OOM protection
  - block_weight: BlockWeightManager per-layer training weights
  - hutchinson_scan: HutchinsonScanner model X-ray diagnostics
  - ln_guard: LNGuard LayerNorm anti-fry regularization
  - manifold_constraint: ManifoldConstraint geometric lock
  - ghost_replay: GhostRecorder / GhostReplayer knowledge distillation
  - hyperparam_manager: LulynxHyperparamManager preset management
  - dora_layer: DoRALinear / DoRAInjector weight-decomposed LoRA

Tier 2 (minor import fixes):
  - safe_guard: TrainingSafeGuard NaN / loss spike / LR deadlock
  - lisa: LISAScheduler layerwise importance sampling (lulynx_trainer version)
  - lisa_scheduler: LISAScheduler (lulynx version)
  - pissa_injector: PissaInjector SVD-initialized LoRA

Tier 3 (refactored from coupled code):
  - mn_lora/: MN-LoRA optimizer subsystem (GSP, TG-WD, TrainingPilot,
    proxy regularizer, SVD engine, V-matrix cache)
"""

# ── Tier 1: Standalone ──────────────────────────────────────────────────────

from .ema import EMAModel, EMACallback, EMAStateTracker, create_ema, create_ema_callback

from .auto_controller import (
    AutoController,
    AutoControlConfig,
    AutoControllerCallback,
    AutoEvent,
    MetricsTracker,
    create_auto_controller,
    create_auto_controller_callback,
)

from .smart_rank import SmartRankController

from .coreset import (
    CoresetManager,
    WeightedBatchSampler,
    SampleStats,
    create_coreset_manager,
    create_weighted_sampler,
)

from .resource_manager import (
    DynamicResourceManager,
    ResourceConfig,
    oom_safe_execute,
)

from .block_weight import (
    BlockWeightManager,
    BlockWeightConfig,
    BlockWeightPreset,
    get_preset_list,
    create_block_weight_manager,
    create_block_weight_manager_from_settings,
)

from .hutchinson_scan import HutchinsonScanner, LayerScanResult

from .ln_guard import LNGuard

from .manifold_constraint import ManifoldConstraint, LogMethod

from .ghost_replay import GhostRecorder, GhostReplayer, inspect_ghost_fingerprint

from .hyperparam_manager import (
    LulynxHyperparamManager,
    MNLoraConfig,
    MNPreset,
    get_mn_config,
)

from .dora_layer import DoRALinear, DoRAInjector

# ── Tier 2: Minor fixes ────────────────────────────────────────────────────

from .safe_guard import (
    TrainingSafeGuard,
    SafeGuardConfig,
    SafeGuardAction,
    ProdigySafeGuardPreset,
    create_safe_prodigy_optimizer,
    reset_prodigy_state,
)

from .lisa import LISAScheduler as LISASchedulerTrainer

from .lisa_scheduler import LISAScheduler

from .pissa_injector import PissaInjector

from .noise_utils import (
    pyramid_noise_like,
    apply_adaptive_noise_scale,
    apply_ip_noise,
    rescale_zero_terminal_snr,
)

# ── Tier 3: Refactored ─────────────────────────────────────────────────────

__all__ = [
    # EMA
    "EMAModel", "EMACallback", "EMAStateTracker", "create_ema", "create_ema_callback",
    # Auto Controller
    "AutoController", "AutoControlConfig", "AutoControllerCallback",
    "AutoEvent", "MetricsTracker", "create_auto_controller", "create_auto_controller_callback",
    # Smart Rank
    "SmartRankController",
    # Coreset
    "CoresetManager", "WeightedBatchSampler", "SampleStats",
    "create_coreset_manager", "create_weighted_sampler",
    # Resource Manager
    "DynamicResourceManager", "ResourceConfig", "oom_safe_execute",
    # Block Weight
    "BlockWeightManager", "BlockWeightConfig", "BlockWeightPreset",
    "get_preset_list", "create_block_weight_manager", "create_block_weight_manager_from_settings",
    # Hutchinson
    "HutchinsonScanner", "LayerScanResult",
    # LN Guard
    "LNGuard",
    # Manifold
    "ManifoldConstraint", "LogMethod",
    # Ghost Replay
    "GhostRecorder", "GhostReplayer", "inspect_ghost_fingerprint",
    # Hyperparam
    "LulynxHyperparamManager", "MNLoraConfig", "MNPreset", "get_mn_config",
    # DoRA
    "DoRALinear", "DoRAInjector",
    # Safe Guard
    "TrainingSafeGuard", "SafeGuardConfig", "SafeGuardAction",
    "ProdigySafeGuardPreset", "create_safe_prodigy_optimizer", "reset_prodigy_state",
    # LISA
    "LISAScheduler", "LISASchedulerTrainer",
    # PiSSA
    "PissaInjector",
    # Noise Utils
    "pyramid_noise_like", "apply_adaptive_noise_scale", "apply_ip_noise", "rescale_zero_terminal_snr",
]
