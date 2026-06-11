"""Built-in request-native runners.

These runners are orchestration adapters. They must stay lightweight at import
time and must not import torch/diffusers unless a concrete heavy execution path
explicitly needs them inside ``run``.
"""

from .base_model_tensorrt import BaseModelTensorRtRuntimeRunner, create_base_model_tensorrt_runtime_registry
from .generation import DryRunGenerationRunner, SubprocessGenerationRunner, create_generation_registry
from .lab import LabSubprocessRunner, create_lab_registry
from .training import TrainingRequestRunner, TrainingRunner, create_training_registry
from .turbocore import TurboCoreBridgeRunner, create_turbocore_bridge_registry

__all__ = [
    "DryRunGenerationRunner",
    "BaseModelTensorRtRuntimeRunner",
    "SubprocessGenerationRunner",
    "LabSubprocessRunner",
    "TrainingRequestRunner",
    "TrainingRunner",
    "TurboCoreBridgeRunner",
    "create_base_model_tensorrt_runtime_registry",
    "create_generation_registry",
    "create_lab_registry",
    "create_training_registry",
    "create_turbocore_bridge_registry",
]
