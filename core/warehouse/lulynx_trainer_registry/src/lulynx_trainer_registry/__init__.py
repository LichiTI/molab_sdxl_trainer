"""Data-driven trainer definition registry with hardware runtime detection.

Provides a frozen dataclass ``TrainerSpec`` that binds a training type
to its script path, launch mode, route metadata, and optional callback
hooks for validation and warning generation.
"""

from lulynx_trainer_registry.registry import (
    TrainerSpec,
    TrainerRegistry,
)
from lulynx_trainer_registry.runtime_detect import detect_hardware_backend

__all__ = [
    "TrainerRegistry",
    "TrainerSpec",
    "detect_hardware_backend",
]
