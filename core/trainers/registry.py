"""
Trainer Registry - Register and discover trainer plugins
"""

from typing import Dict, Type, List, Optional
from .base import BaseTrainer


import logging

import threading

# Global registry of trainers
_TRAINERS: Dict[str, Type[BaseTrainer]] = {}
_REGISTRY_LOCK = threading.Lock()

logger = logging.getLogger("TrainerRegistry")


def register_trainer(trainer_class: Type[BaseTrainer]) -> Type[BaseTrainer]:
    """
    Decorator to register a trainer class.
    
    Usage:
        @register_trainer
        class LoRATrainer(BaseTrainer):
            id = "lora"
            ...
    """
    if not trainer_class.id:
        raise ValueError(f"Trainer {trainer_class.__name__} must have an 'id' attribute")
    
    with _REGISTRY_LOCK:
        if trainer_class.id in _TRAINERS:
            logger.warning(f"[Trainers] Overwriting existing trainer '{trainer_class.id}'")
        
        _TRAINERS[trainer_class.id] = trainer_class
        logger.info(f"[Trainers] Registered: {trainer_class.id} ({trainer_class.name})")
    return trainer_class


def get_trainer(trainer_id: str) -> Optional[BaseTrainer]:
    """
    Get an instance of a registered trainer by ID.
    
    Returns:
        Trainer instance or None if not found
    """
    trainer_class = _TRAINERS.get(trainer_id)
    if trainer_class:
        return trainer_class()
    return None


def get_trainer_class(trainer_id: str) -> Optional[Type[BaseTrainer]]:
    """Get the trainer class (not instance) by ID."""
    return _TRAINERS.get(trainer_id)


def list_trainers() -> List[dict]:
    """
    List all registered trainers with their metadata.
    
    Returns:
        List of dicts with id, name, version, author, description
    """
    return [
        {
            "id": t.id,
            "name": t.name,
            "version": t.version,
            "author": t.author,
            "description": t.description,
        }
        for t in _TRAINERS.values()
    ]


def get_trainer_ids() -> List[str]:
    """Get list of all registered trainer IDs."""
    return list(_TRAINERS.keys())


def unregister_trainer(trainer_id: str) -> bool:
    """
    Unregister a trainer by ID.
    
    Returns:
        True if trainer was found and removed
    """
    if trainer_id in _TRAINERS:
        del _TRAINERS[trainer_id]
        logger.info(f"[Trainers] Unregistered: {trainer_id}")
        return True
    return False


def clear_registry() -> None:
    """Clear all registered trainers. Mainly for testing."""
    _TRAINERS.clear()
