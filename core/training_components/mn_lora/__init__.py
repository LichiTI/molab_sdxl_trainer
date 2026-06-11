from .trace_guided_wd import TraceGuidedWeightDecay
from .gradient_subspace import GradientSubspaceProjection, GSPLayerStats, GSPMonitor
from .mn_optimizer import MNLoRAOptimizer
from .mn_lora_plus_plus import MNLoRAPlusPlusController
from .mn_lora_trust_region import MNLoRATrustRegionController
from .effective_delta import MNLoRAEffectiveDeltaController
from .lora_kfac_lite import LoRAKFACLiteController
from .fisher_ewc import MNLoRAFisherEWCController
from .gradient_conflict import MNLoRAGradientConflictController
from .proxy_regularizer import ProxyRegularizer, AdaptiveProxyRegularizer, create_proxy_regularizer
from .ghost_replay import (
    ReferenceOutputCache,
    GhostReplayRegularizer,
    GhostReplayCallback,
    create_ghost_replay,
)
from .hijacker import wrap_optimizer
from .svd_utils import compute_effective_rank_cpu, randomized_svd
from .svd_engine import SVDMode, SVDEngine, FullSVDEngine, IncrementalSVDEngine, RandomizedSVDEngine, create_svd_engine
from .v_matrix_cache import CacheMode, VMatrixCache, CacheConfig, get_v_cache, set_cache_mode
from .mn_presets import SDXL_PRESET, FLUX_PRESET, SD15_PRESET, get_preset_for_model, select_mnlora_preset, split_mnlora_preset

__all__ = [
    # V2.1 Core
    "TraceGuidedWeightDecay",
    "GradientSubspaceProjection",
    "MNLoRAOptimizer",
    "MNLoRAPlusPlusController",
    "MNLoRATrustRegionController",
    "MNLoRAEffectiveDeltaController",
    "LoRAKFACLiteController",
    "MNLoRAFisherEWCController",
    "MNLoRAGradientConflictController",
    # V2.3 Regularization
    "ProxyRegularizer",
    "AdaptiveProxyRegularizer",
    "create_proxy_regularizer",
    # V2.4 Ghost Replay
    "ReferenceOutputCache",
    "GhostReplayRegularizer",
    "GhostReplayCallback",
    "create_ghost_replay",
    # V2.5 Monitoring
    "GSPLayerStats",
    "GSPMonitor",
    # V2.6 SVD Engine
    "SVDMode",
    "SVDEngine",
    "FullSVDEngine",
    "IncrementalSVDEngine",
    "RandomizedSVDEngine",
    "create_svd_engine",
    # V2.6 V Matrix Cache
    "CacheMode",
    "VMatrixCache",
    "CacheConfig",
    "get_v_cache",
    "set_cache_mode",
    # Optimizer Wrapper (explicit API, no monkey-patching)
    "wrap_optimizer",
    # SVD Utils
    "compute_effective_rank_cpu",
    "randomized_svd",
    # V2.6 Smart Presets
    "SDXL_PRESET",
    "FLUX_PRESET",
    "SD15_PRESET",
    "get_preset_for_model",
    "select_mnlora_preset",
    "split_mnlora_preset",
]
