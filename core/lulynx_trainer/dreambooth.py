"""
DreamBooth 训练器

支持角色定制训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DreamBoothConfig:
    """DreamBooth 配置"""
    # 模型
    model_path: str = ""
    model_type: str = "sdxl"
    
    # 实例
    instance_prompt: str = "sks person"
    instance_data_dir: str = ""
    
    # 类别 (Prior Preservation)
    class_prompt: str = "a person"
    class_data_dir: str = ""
    num_class_images: int = 100
    prior_loss_weight: float = 1.0
    
    # 训练
    learning_rate: float = 1e-6
    train_text_encoder: bool = True
    train_unet: bool = True
    
    # LoRA (可选)
    use_lora: bool = False
    lora_rank: int = 16
    
    # 优化
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True


class PriorPreservationLoss:
    """
    Prior Preservation Loss
    
    防止语言漂移 (Language Drift)
    保持模型对一般概念的理解
    """
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def __call__(
        self,
        instance_loss: torch.Tensor,
        class_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            instance_loss: 实例损失 (训练目标)
            class_loss: 类别损失 (保持能力)
        
        Returns:
            加权总损失
        """
        return instance_loss + self.weight * class_loss


class ClassImageGenerator:
    """
    类别图像生成器
    
    为 Prior Preservation 生成类别参考图像
    """
    
    def __init__(
        self,
        pipeline,
        class_prompt: str,
        num_images: int = 100,
        output_dir: str = "./class_images",
    ):
        self.pipeline = pipeline
        self.class_prompt = class_prompt
        self.num_images = num_images
        self.output_dir = Path(output_dir)
    
    def generate(self) -> List[str]:
        """生成类别图像"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        generated = []
        
        for i in range(self.num_images):
            output_path = self.output_dir / f"class_{i:04d}.png"
            
            if output_path.exists():
                generated.append(str(output_path))
                continue
            
            image = self.pipeline(
                self.class_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            
            image.save(output_path)
            generated.append(str(output_path))
            
            if (i + 1) % 10 == 0:
                logger.info(f"[ClassImageGenerator] Generated {i + 1}/{self.num_images}")
        
        return generated


class DreamBoothTrainer:
    """
    DreamBooth 训练器
    
    特点:
    - Prior Preservation Loss 防止遗忘
    - 可选 LoRA 微调
    - 支持文本编码器训练
    """
    
    def __init__(self, config: DreamBoothConfig):
        self.config = config
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.optimizer = None
        self.prior_loss = PriorPreservationLoss(config.prior_loss_weight)
    
    def load_model(self):
        """加载模型"""
        try:
            if self.config.model_type == "sdxl":
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16,
                )
            else:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16,
                )
            
            self.model = pipe.unet
            self.text_encoder = pipe.text_encoder
            self.vae = pipe.vae
            self.scheduler = pipe.scheduler
            self.tokenizer = pipe.tokenizer
            
            # 梯度检查点
            if self.config.gradient_checkpointing:
                self.model.enable_gradient_checkpointing()
            
            logger.info("[DreamBoothTrainer] Model loaded")
            
        except Exception as e:
            logger.error(f"[DreamBoothTrainer] Load failed: {e}")
            raise
    
    def prepare_optimizer(self):
        """准备优化器"""
        params = []
        
        if self.config.train_unet:
            params.extend(self.model.parameters())
        
        if self.config.train_text_encoder:
            params.extend(self.text_encoder.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        logger.info(f"[DreamBoothTrainer] Optimizer prepared")
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """编码文本提示"""
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            embeddings = self.text_encoder(inputs.input_ids)[0]
        
        return embeddings
    
    def train_step(
        self,
        instance_batch: Dict[str, torch.Tensor],
        class_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            instance_batch: 实例数据
            class_batch: 类别数据 (Prior Preservation)
        
        Returns:
            损失字典
        """
        self.optimizer.zero_grad()
        
        # 实例损失
        instance_loss = self._compute_loss(
            instance_batch["latents"],
            instance_batch["text_embeddings"],
        )
        
        # 类别损失 (Prior Preservation)
        if class_batch is not None and self.config.prior_loss_weight > 0:
            class_loss = self._compute_loss(
                class_batch["latents"],
                class_batch["text_embeddings"],
            )
            total_loss = self.prior_loss(instance_loss, class_loss)
        else:
            class_loss = torch.tensor(0.0)
            total_loss = instance_loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )
        
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "instance_loss": instance_loss.item(),
            "class_loss": class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss,
        }
    
    def _compute_loss(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算噪声预测损失"""
        batch_size = latents.shape[0]
        device = latents.device
        
        # 采样时间步
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device,
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample
        
        # MSE 损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def save(self, output_dir: str):
        """保存模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 UNet
        self.model.save_pretrained(output_path / "unet")
        
        # 保存 Text Encoder
        if self.config.train_text_encoder:
            self.text_encoder.save_pretrained(output_path / "text_encoder")
        
        logger.info(f"[DreamBoothTrainer] Saved to {output_dir}")


# ========== 便捷函数 ==========

def create_dreambooth_trainer(
    model_path: str,
    instance_prompt: str,
    class_prompt: str = "",
) -> DreamBoothTrainer:
    """创建 DreamBooth 训练器"""
    config = DreamBoothConfig(
        model_path=model_path,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt or f"a {instance_prompt.split()[-1]}",
    )
    return DreamBoothTrainer(config)
