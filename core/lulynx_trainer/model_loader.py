"""
模型加载器

加载 SDXL/SD1.5 底座模型
"""

import torch
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from .single_file_loader import SDXLSingleFileLoader
from .sd15_single_file_loader import SD15SingleFileLoader
from .runtime_optimizations import (
    RuntimeOptimizationPlan,
    apply_attention_backend,
    apply_torch_compile_if_requested,
    apply_per_block_compile,
)
from .device_state import ModuleDeviceState, apply_loaded_model_training_states, build_loaded_model_training_states

try:
    from huggingface_hub.errors import LocalEntryNotFoundError
except Exception:
    LocalEntryNotFoundError = None

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """加载的模型组件"""
    unet: Any
    text_encoder_1: Any
    text_encoder_2: Optional[Any]
    vae: Any
    tokenizer_1: Any
    tokenizer_2: Optional[Any]
    noise_scheduler: Any
    model_arch: str


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
    def load(self, model_path: str, model_arch: str = "sdxl", vae_path: Optional[str] = None) -> LoadedModel:
        """加载模型"""
        path = Path(model_path)
        normalized_arch = str(model_arch or "sdxl").strip().lower()

        if normalized_arch == "sdxl":
            model = self._load_sdxl(path)
        elif normalized_arch == "sd15":
            model = self._load_sd15(path)
        elif normalized_arch == "anima":
            model = self._load_anima(path)
        elif normalized_arch == "newbie":
            model = self._load_newbie(path)
        elif normalized_arch == "flux":
            model = self._load_flux(path)
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")

        return self._apply_custom_vae(model, vae_path)

    def _apply_custom_vae(self, model: LoadedModel, vae_path: Optional[str]) -> LoadedModel:
        if not vae_path:
            return model

        from diffusers import AutoencoderKL

        vae_ref = Path(vae_path)
        logger.info(f"Loading custom VAE from {vae_ref}")

        if vae_ref.is_file():
            model.vae = AutoencoderKL.from_single_file(
                str(vae_ref),
                torch_dtype=self.dtype,
            )
        else:
            model.vae = AutoencoderKL.from_pretrained(
                str(vae_ref),
                torch_dtype=self.dtype,
            )

        return model
    
    def _load_sdxl(self, path: Path) -> LoadedModel:
        """加载 SDXL 模型"""
        from diffusers import (
            UNet2DConditionModel,
            AutoencoderKL,
            DDPMScheduler,
        )
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )
        
        logger.info(f"Loading SDXL model from {path}")
        
        # 检测模型格式
        if path.suffix in (".safetensors", ".ckpt", ".pt"):
            return self._load_sdxl_single_file(path)
        else:
            return self._load_sdxl_diffusers(path)
    
    def _load_sdxl_diffusers(self, path: Path) -> LoadedModel:
        """从 Diffusers 格式加载 SDXL"""
        from diffusers import (
            UNet2DConditionModel,
            AutoencoderKL,
            DDPMScheduler,
        )
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )
        
        # 加载各组件
        unet = UNet2DConditionModel.from_pretrained(
            path, subfolder="unet", torch_dtype=self.dtype
        )
        
        text_encoder_1 = CLIPTextModel.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=self.dtype
        )
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            path, subfolder="text_encoder_2", torch_dtype=self.dtype
        )
        
        vae = AutoencoderKL.from_pretrained(
            path, subfolder="vae", torch_dtype=self.dtype
        )
        
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer"
        )
        
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer_2"
        )
        
        scheduler = DDPMScheduler.from_pretrained(
            path, subfolder="scheduler"
        )
        
        return LoadedModel(
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            noise_scheduler=scheduler,
            model_arch="sdxl",
        )
    
    def _load_sdxl_single_file(self, path: Path) -> LoadedModel:
        """从单文件 safetensors 加载 SDXL"""
        logger.info(f"Loading SDXL from single file: {path}")

        components = SDXLSingleFileLoader(dtype=self.dtype).load(path)

        return LoadedModel(
            unet=components.unet,
            text_encoder_1=components.text_encoder_1,
            text_encoder_2=components.text_encoder_2,
            vae=components.vae,
            tokenizer_1=components.tokenizer_1,
            tokenizer_2=components.tokenizer_2,
            noise_scheduler=components.noise_scheduler,
            model_arch="sdxl",
        )
    
    def _load_sd15(self, path: Path) -> LoadedModel:
        """加载 SD 1.5 模型"""
        from diffusers import (
            UNet2DConditionModel,
            AutoencoderKL,
            DDPMScheduler,
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        
        logger.info(f"Loading SD1.5 model from {path}")
        
        if path.suffix in (".safetensors", ".ckpt", ".pt"):
            return self._load_sd15_single_file(path)
        
        unet = UNet2DConditionModel.from_pretrained(
            path, subfolder="unet", torch_dtype=self.dtype
        )
        
        text_encoder = CLIPTextModel.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=self.dtype
        )
        
        vae = AutoencoderKL.from_pretrained(
            path, subfolder="vae", torch_dtype=self.dtype
        )
        
        tokenizer = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer"
        )
        
        scheduler = DDPMScheduler.from_pretrained(
            path, subfolder="scheduler"
        )
        
        return LoadedModel(
            unet=unet,
            text_encoder_1=text_encoder,
            text_encoder_2=None,
            vae=vae,
            tokenizer_1=tokenizer,
            tokenizer_2=None,
            noise_scheduler=scheduler,
            model_arch="sd15",
        )
    
    def _load_sd15_single_file(self, path: Path) -> LoadedModel:
        """从单文件加载 SD1.5"""
        components = SD15SingleFileLoader(dtype=self.dtype).load(path)
        return components.to_loaded_model("sd15")
    
    def _load_flux(self, path: Path) -> LoadedModel:
        """加载 Flux 模型 (阶段3实现)"""
        raise NotImplementedError("Flux support coming in Phase 3")

    def _load_anima(self, path: Path) -> LoadedModel:
        """加载 Anima 模型。"""
        from .anima_loader import load_anima_model

        model, report = load_anima_model(
            model_path=str(path),
            device=self.device,
            dtype=self.dtype,
        )
        model.anima_load_report = report
        return model

    def _load_newbie(self, path: Path) -> LoadedModel:
        """加载 Newbie 模型。"""
        from .newbie_loader import load_newbie

        return load_newbie(
            diffusers_path=str(path),
            device=self.device,
            dtype=self.dtype,
        )
    
    def prepare_for_training(
        self,
        model: LoadedModel,
        gradient_checkpointing: bool = True,
        xformers: bool = False,
        runtime_plan: Optional[RuntimeOptimizationPlan] = None,
        *,
        train_text_encoder: bool = True,
        keep_text_encoders_on_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        preserve_unet_residency: bool = False,
        defer_per_block_compile: bool = False,
    ):
        """准备模型进行训练"""
        training_states = build_loaded_model_training_states(
            device=self.device,
            train_text_encoder=bool(train_text_encoder),
            keep_text_encoders_on_cpu=bool(keep_text_encoders_on_cpu),
            keep_vae_on_cpu=bool(keep_vae_on_cpu),
        )
        if preserve_unet_residency:
            training_states["unet"] = ModuleDeviceState(
                name="unet",
                training=True,
                device=None,
            )
        apply_loaded_model_training_states(model, training_states)
        
        # 启用梯度检查点
        if gradient_checkpointing:
            if hasattr(model.unet, "enable_gradient_checkpointing"):
                model.unet.enable_gradient_checkpointing()
            if (
                train_text_encoder
                and model.text_encoder_1 is not None
                and hasattr(model.text_encoder_1, "gradient_checkpointing_enable")
            ):
                model.text_encoder_1.gradient_checkpointing_enable()
            if (
                train_text_encoder
                and model.text_encoder_2
                and hasattr(model.text_encoder_2, "gradient_checkpointing_enable")
            ):
                model.text_encoder_2.gradient_checkpointing_enable()
        
        effective_plan = runtime_plan
        if effective_plan is None:
            from .runtime_optimizations import RuntimeOptimizationPlan

            effective_plan = RuntimeOptimizationPlan(
                attention_backend="xformers" if xformers else "torch",
                requested_attention_backend="xformers" if xformers else "torch",
            )

        apply_attention_backend(model, effective_plan)

        if model.unet is not None:
            model.unet = apply_torch_compile_if_requested(model.unet, effective_plan, label="UNet")

        if effective_plan.torch_compile and effective_plan.torch_compile_scope == "per_block" and not defer_per_block_compile:
            target = model.unet or getattr(model, "dit", None)
            if target is not None:
                apply_per_block_compile(target, effective_plan, route=model.model_arch)

        logger.info(f"Model prepared for training on {self.device}")
