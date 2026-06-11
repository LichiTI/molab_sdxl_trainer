"""
Trainer Plugin Base - Abstract interface for all trainers
Third-party developers should inherit from BaseTrainer
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, Any, Optional, Generator, List, Tuple
from pathlib import Path


class TrainerConfig(BaseModel):
    """Base configuration all trainers must support"""
    data_dir: str
    output_dir: str
    output_name: str = "model"
    
    class Config:
        extra = "allow"  # Allow subclasses to add fields


class TrainerResult(BaseModel):
    """Result returned after training completes"""
    success: bool
    message: str
    output_path: Optional[str] = None
    metrics: Dict[str, Any] = {}


class TrainerProgress(BaseModel):
    """Progress update during training"""
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    message: str = ""
    progress_percent: float = 0.0


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Third-party developers must implement this interface to create
    a new trainer plugin.
    
    Example:
        @register_trainer
        class MyTrainer(BaseTrainer):
            id = "my-trainer"
            name = "My Custom Trainer"
            ...
    """
    
    # === Metadata (required) ===
    id: str = ""           # Unique ID: "lora", "yolo", "newbie"
    name: str = ""         # Display name: "LoRA Trainer"
    version: str = "1.0.0" # Version
    author: str = ""       # Author name
    description: str = ""  # Brief description
    
    # === Abstract Methods (required) ===
    
    @abstractmethod
    def get_config_schema(self) -> type[TrainerConfig]:
        """
        Return the Pydantic model for this trainer's configuration.
        This is used for validation and UI generation.
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: dict) -> Tuple[bool, str]:
        """
        Validate the configuration before training.
        
        Returns:
            (is_valid, error_message) - if valid, error_message is empty
        """
        pass
    
    @abstractmethod
    def estimate_vram(self, config: dict) -> float:
        """
        Estimate VRAM usage in GB.
        Used to warn users if they may run out of memory.
        """
        pass
    
    @abstractmethod
    def train(self, config: dict) -> Generator[TrainerProgress, None, TrainerResult]:
        """
        Execute training and yield progress updates.
        
        Args:
            config: Configuration dict matching get_config_schema()
            
        Yields:
            TrainerProgress objects with step/loss/message updates
            
        Returns:
            TrainerResult with final status
            
        Example:
            def train(self, config):
                for epoch in range(10):
                    for step in range(100):
                        yield TrainerProgress(step=step, epoch=epoch, loss=0.05)
                return TrainerResult(success=True, message="Done")
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the current training run.
        
        Returns:
            True if stopped successfully
        """
        pass
    
    # === Optional Methods ===
    
    def get_default_config(self) -> dict:
        """Return default configuration values."""
        return {}
    
    def get_dependencies(self) -> List[str]:
        """
        Return list of pip packages required by this trainer.
        Used by Extensions Center to check/install dependencies.
        """
        return []
    
    def on_load(self) -> None:
        """Called when trainer is first loaded. Use for initialization."""
        pass
    
    def on_unload(self) -> None:
        """Called when trainer is unloaded. Use for cleanup."""
        pass
