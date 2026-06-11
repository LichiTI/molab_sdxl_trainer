"""
LoRA Trainer Plugin - Main LoRA/LoCon training implementation

This is an example of how to implement a trainer plugin.
"""

from typing import Generator, Tuple, List
from pydantic import BaseModel

from .base import BaseTrainer, TrainerConfig, TrainerResult, TrainerProgress
from .registry import register_trainer
import logging
import os
from typing import Type

logger = logging.getLogger(__name__)


class LoRAConfig(TrainerConfig):
    """LoRA-specific training configuration"""
    # Network
    network_type: str = "lora"  # lora, locon, loha
    network_dim: int = 32
    network_alpha: int = 16
    # network_alpha: int = 16
    # conv_dim: int = 16 (Unused by standard LoRA, kept for LoCon compat)
    # conv_alpha: int = 8
    
    # Training
    learning_rate: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    train_batch_size: int = 1
    max_train_epochs: int = 10
    save_every_n_epochs: int = 1
    
    # Optimizer
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_ratio: float = 0.05
    
    # Base model
    pretrained_model: str = ""
    
    # Advanced
    use_pissa: bool = False
    cache_latents: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"


@register_trainer
class LoRATrainer(BaseTrainer):
    """LoRA/LoCon trainer for Stable Diffusion models"""
    
    id = "lora"
    name = "LoRA/LoCon 训练器"
    version = "1.0.0"
    author = "Lulynx"
    description = "标准 SDXL/SD1.5 LoRA/LoCon 训练，支持多种优化器和调度器"
    
    _is_training = False
    _should_stop = False
    
    def get_config_schema(self) -> Type[TrainerConfig]:
        return LoRAConfig
    
    def validate_config(self, config: dict) -> Tuple[bool, str]:
        """Validate LoRA configuration"""
        # Check required fields
        if not config.get("pretrained_model"):
            return False, "请选择基础模型"
        if not os.path.exists(config.get("pretrained_model", "")):
             return False, f"基础模型路径不存在: {config.get('pretrained_model')}"

        if not config.get("data_dir"):
            return False, "请选择训练数据目录"
        if not os.path.exists(config.get("data_dir", "")):
             return False, f"训练数据目录不存在: {config.get('data_dir')}"
        if not config.get("output_dir"):
            return False, "请选择输出目录"
        
        # Validate ranges
        dim = config.get("network_dim", 32)
        if dim < 1 or dim > 512:
            return False, f"network_dim ({dim}) 必须在 1-512 之间"
        
        lr = config.get("learning_rate", 1e-4)
        if lr <= 0 or lr > 1:
            return False, f"learning_rate ({lr}) 必须在 0-1 之间"
        
        return True, ""
    
    def estimate_vram(self, config: dict) -> float:
        """Estimate VRAM usage based on config"""
        # Base VRAM for SDXL
        base_vram = 8.0
        
        # Add for batch size
        batch_size = config.get("train_batch_size", 1)
        base_vram += batch_size * 1.5
        
        # Add for network dim
        dim = config.get("network_dim", 32)
        base_vram += (dim / 64) * 1.0
        
        # Subtract if using gradient checkpointing
        if config.get("gradient_checkpointing", True):
            base_vram *= 0.85
        
        # Add for TE training
        if config.get("text_encoder_lr", 0) > 0:
            base_vram += 1.5
        
        return round(base_vram, 1)
    
    def train(self, config: dict) -> Generator[TrainerProgress, None, TrainerResult]:
        """
        Execute LoRA training.

        NOTE: This is a MOCK implementation for demonstration/testing purposes.
        Real production training uses the native lulynx engine via core/entry_train.py.
        """
        self._is_training = True
        self._should_stop = False
        
        total_epochs = config.get("max_train_epochs", 10)
        
        # TODO: Calculate actual steps based on dataset size / batch size
        # currently using placeholder for demonstration
        steps_per_epoch = 100 
        total_steps = total_epochs * steps_per_epoch
        
        try:
            for epoch in range(total_epochs):
                if self._should_stop:
                    return TrainerResult(
                        success=False,
                        message="训练被用户停止"
                    )
                
                for step in range(steps_per_epoch):
                    if self._should_stop:
                        return TrainerResult(
                            success=False,
                            message="训练被用户停止"
                        )
                    
                    current_step = epoch * steps_per_epoch + step + 1
                    progress = (current_step / total_steps) * 100
                    
                    # Simulate loss
                    loss = 0.1 / (1 + current_step * 0.01)
                    
                    yield TrainerProgress(
                        step=current_step,
                        total_steps=total_steps,
                        epoch=epoch + 1,
                        total_epochs=total_epochs,
                        loss=loss,
                        progress_percent=progress,
                        message=f"Epoch {epoch+1}/{total_epochs}, Step {step+1}/{steps_per_epoch}"
                    )
            
            return TrainerResult(
                success=True,
                message="训练完成!",
                output_path=f"{config.get('output_dir')}/{config.get('output_name')}.safetensors",
                metrics={
                    "final_loss": 0.05,
                    "total_steps": total_steps,
                    "epochs": total_epochs
                }
            )
            
        except Exception as e:
            return TrainerResult(
                success=False,
                message=f"训练出错: {e}"
            )
        finally:
            self._is_training = False
    
    def stop(self) -> bool:
        """Stop the current training run"""
        if self._is_training:
            self._should_stop = True
            return True
        return False
    
    def get_default_config(self) -> dict:
        """Return default configuration"""
        return LoRAConfig().model_dump()
    
    def get_dependencies(self) -> List[str]:
        """Required packages"""
        return [
            "torch",
            "safetensors",
            "accelerate",
            "transformers",
            "diffusers",
        ]
