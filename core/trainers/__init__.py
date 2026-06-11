"""
Trainers Package - Plugin system for training modules
"""

from .base import BaseTrainer, TrainerConfig, TrainerResult, TrainerProgress
from .registry import (
    register_trainer,
    get_trainer,
    get_trainer_class,
    list_trainers,
    get_trainer_ids,
    unregister_trainer,
    clear_registry,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "TrainerConfig",
    "TrainerResult",
    "TrainerProgress",
    # Registry functions
    "register_trainer",
    "get_trainer",
    "get_trainer_class",
    "list_trainers",
    "get_trainer_ids",
    "unregister_trainer",
    "clear_registry",
]

# Auto-load built-in trainers
# Third-party trainers should be imported in their own packages
try:
    from . import lora_trainer  # Will be created
except ImportError:
    pass
