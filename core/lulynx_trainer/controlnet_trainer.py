"""
ControlNet Trainer for Lulynx

Implements training for ControlNet models by freezing the base model 
and training a copy of the UNet encoder with zero-convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from .trainer import LulynxTrainer
from .config import LulynxConfig
from .model_loader import ModelLoader
from .dataset_loader import CaptionDataset, create_dataloader
from .training_loop import TrainingLoop
from .turbocore_native_update_route_binding import build_turbocore_native_update_training_loop_kwargs
from .runtime_optimizations import build_runtime_optimization_plan
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


class _ControlNetParamWrapper:
    """Wraps ControlNet parameters to satisfy the lora_injector interface
    expected by TrainingLoop gradient clipping and optimizer creation."""

    def __init__(self, controlnet: torch.nn.Module):
        self._controlnet = controlnet

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self._controlnet.parameters() if p.requires_grad]

    def get_param_groups(self, base_lr: float = 1e-4, weight_decay: float = 0.0):
        return [{"params": self.get_trainable_params(), "lr": base_lr, "weight_decay": weight_decay}]


class ControlNetTrainer(LulynxTrainer):
    """ControlNet 专门训练器"""
    
    def __init__(self, config: Optional[LulynxConfig] = None):
        super().__init__(config)
        self.controlnet = None
        
    def prepare(self):
        """准备 ControlNet 训练"""
        if not self.config:
            raise ValueError("Config not set")
            
        self._log("Loading base model...")
        loader = ModelLoader(device=self.device, dtype=self.dtype)
        self.runtime_optimization_plan = self.runtime_optimization_plan or build_runtime_optimization_plan(self.config)
        for line in self.runtime_optimization_plan.log_lines():
            self._log(line)
        self.model = loader.load(
            self.config.base_model_path,
            getattr(self.config.model_arch, "value", self.config.model_arch),
        )
        
        loader.prepare_for_training(
            self.model,
            gradient_checkpointing=self.config.gradient_checkpointing,
            xformers=self.config.xformers,
            runtime_plan=self.runtime_optimization_plan,
        )
        
        controlnet_model_path = str(getattr(self.config, "controlnet_model", "") or "").strip()
        try:
            from diffusers import ControlNetModel
            if controlnet_model_path:
                self._log(f"Loading pretrained ControlNet: {controlnet_model_path}")
                self.controlnet = ControlNetModel.from_pretrained(
                    controlnet_model_path,
                    torch_dtype=self.dtype,
                )
            else:
                self._log("Initializing ControlNet from UNet...")
                self.controlnet = ControlNetModel.from_unet(self.model.unet)
        except ImportError:
            self._log("diffusers.ControlNetModel not available, using placeholder")
            raise
            
        self.controlnet.to(self.device, dtype=self.dtype)
        
        # Freeze everything except ControlNet
        self.model.unet.requires_grad_(False)
        self.model.vae.requires_grad_(False)
        self.model.text_encoder_1.requires_grad_(False)
        if self.model.text_encoder_2:
            self.model.text_encoder_2.requires_grad_(False)

        # Gradient checkpointing on the frozen UNet blocks gradient flow
        # to ControlNet residuals — disable it since the UNet doesn't need it.
        self.model.unet.disable_gradient_checkpointing()
            
        self.controlnet.train()

        self._load_resume_or_network_weights()
        self._disable_unsupported_preview()

        # Set lora_injector so _create_optimizer() and TrainingLoop
        # gradient clipping can access trainable params.
        self.lora_injector = _ControlNetParamWrapper(self.controlnet)

        trainable_params = self.lora_injector.get_trainable_params()
        total_params = sum(p.numel() for p in trainable_params)
        self._log(f"ControlNet trainable parameters: {total_params:,}")

        return self

    def _disable_unsupported_preview(self):
        if (
            getattr(self.config, "sample_every", 0) > 0
            or getattr(self.config, "sample_every_n_epochs", 0) > 0
            or self._get_sample_prompts_list()
        ):
            self._log(
                "ControlNet preview is not wired to a control-image sampler yet; skipping preview generation."
            )
            self.config.sample_every = 0
            self.config.sample_every_n_epochs = 0
        self._sampler = None

    def _load_resume_or_network_weights(self):
        weight_path = (
            getattr(self.config, "network_weights_path", "")
            or getattr(self.config, "resume_path", "")
        )
        if not weight_path:
            return

        path = Path(weight_path)
        if not path.exists():
            self._log(f"ControlNet resume/network_weights path not found: {path}")
            return

        if path.is_dir():
            self._log(f"Loading ControlNet weights from directory: {path}")
            from diffusers import ControlNetModel

            self.controlnet = ControlNetModel.from_pretrained(
                str(path),
                torch_dtype=self.dtype,
            ).to(self.device, dtype=self.dtype)
            self.controlnet.train()
            return

        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        elif path.suffix.lower() in {".pt", ".ckpt", ".bin"}:
            state = safe_torch_load(str(path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        else:
            self._log(f"Unsupported ControlNet weights file extension: {path.suffix}")
            return

        if not isinstance(state, dict):
            self._log(f"ControlNet weights did not contain a state dict: {path}")
            return

        missing, unexpected = self.controlnet.load_state_dict(state, strict=False)
        self._log(
            f"Loaded ControlNet weights from {path} ({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )

    def _run_training(self):
        """执行 ControlNet 训练循环"""
        self._log("Creating dataset for ControlNet...")
        
        dataset = CaptionDataset(
            data_dir=self.config.train_data_dir,
            resolution=self.config.resolution,
            caption_extension=self.config.caption_extension,
            enable_bucket=self.config.enable_bucket,
            shuffle_caption=getattr(self.config, "shuffle_caption", True),
            shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
            keep_tokens=getattr(self.config, "keep_tokens", 0),
            keep_tokens_separator=getattr(self.config, "keep_tokens_separator", ""),
            caption_dropout_rate=getattr(self.config, "caption_dropout_rate", 0.0),
            caption_dropout_every_n_epochs=getattr(self.config, "caption_dropout_every_n_epochs", 0),
            tag_dropout_rate=getattr(self.config, "tag_dropout_rate", 0.0),
            caption_tag_dropout_targets=getattr(self.config, "caption_tag_dropout_targets", ""),
            caption_tag_dropout_target_mode=getattr(self.config, "caption_tag_dropout_target_mode", "drop_all"),
            caption_tag_dropout_target_count=getattr(self.config, "caption_tag_dropout_target_count", 1),
            conditioning_data_dir=getattr(self.config, "conditioning_data_dir", ""),
            image_decode_backend=getattr(self.config, "image_decode_backend", "pil"),
            image_decode_cache_size=getattr(self.config, "image_decode_cache_size", 0),
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=getattr(self.config, "dataloader_num_workers", 0),
        )
        
        grad_accum = max(int(self.config.gradient_accumulation), 1)
        steps_per_epoch = max((len(dataloader) + grad_accum - 1) // grad_accum, 1)
        total_steps = steps_per_epoch * self.config.epochs
        
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, total_steps)
        
        # Specialized Training Loop for ControlNet
        from .training_loop import TrainingLoop
        
        class ControlNetTrainingLoop(TrainingLoop):
            def __init__(self, *args, **kwargs):
                self.controlnet = kwargs.pop("controlnet")
                super().__init__(*args, **kwargs)

            def train_step(self, batch: Dict, accumulation_steps: Optional[int] = None) -> float:
                accumulation_steps = max(int(accumulation_steps or self.gradient_accumulation_steps), 1)
                images = batch["images"].to(self.device, dtype=self.dtype)
                # ControlNet requires control images (conditioning)
                # We expect "control_images" in batch
                control_images = batch.get("control_images", images).to(self.device, dtype=self.dtype)
                captions = batch["captions"]
                
                # Encode target images
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Encode text
                prompt_embeds = self._encode_prompt(captions)
                time_embeds = self._get_timestep_embedding(
                    latents.shape[0],
                    batch.get("original_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("target_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("crop_coords", [(0, 0, 1024, 1024)] * latents.shape[0]),
                )
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
                ).long()
                
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                added_cond_kwargs = None
                pooled = prompt_embeds.get("pooled_prompt_embeds")
                time_cond = time_embeds.get("added_cond_kwargs", {}) if time_embeds else {}
                if time_cond or pooled is not None:
                    added_cond_kwargs = dict(time_cond)
                    if pooled is not None:
                        added_cond_kwargs["text_embeds"] = pooled

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds["encoder_hidden_states"],
                    controlnet_cond=control_images,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                # Predict noise with ControlNet residuals
                unet_kwargs = {
                    "sample": noisy_latents,
                    "timestep": timesteps,
                    "encoder_hidden_states": prompt_embeds["encoder_hidden_states"],
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
                }
                if added_cond_kwargs is not None:
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                noise_pred = self.unet(**unet_kwargs).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss / accumulation_steps
                loss.backward()

                return loss.item() * accumulation_steps

        self.training_loop = ControlNetTrainingLoop(
            unet=self.model.unet,
            text_encoder_1=self.model.text_encoder_1,
            text_encoder_2=self.model.text_encoder_2,
            vae=self.model.vae,
            tokenizer_1=self.model.tokenizer_1,
            tokenizer_2=self.model.tokenizer_2,
            noise_scheduler=self.model.noise_scheduler,
            lora_injector=self.lora_injector,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            device=self.device,
            dtype=self.dtype,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            max_grad_norm=self.config.max_grad_norm,
            module_offload_enabled=getattr(self.config, "module_offload_enabled", False),
            module_offload_ratio=getattr(self.config, "module_offload_ratio", 0),
            module_offload_backbone_ratio=getattr(self.config, "module_offload_backbone_ratio", None),
            module_offload_text_encoder_ratio=getattr(self.config, "module_offload_text_encoder_ratio", None),
            gradient_checkpointing=getattr(self.config, "gradient_checkpointing", False),
            vram_swap_to_ram=getattr(self.config, "vram_swap_to_ram", False),
            torch_compile=getattr(self.config, "torch_compile", False),
            cpu_offload_checkpointing=getattr(self.config, "cpu_offload_checkpointing", False),
            safe_fallback=getattr(self.config, "newbie_safe_fallback", False),
            multi_gpu=bool(getattr(self.config, "multi_gpu", False)),
            num_processes=int(getattr(self.config, "num_processes", 1) or 1),
            num_machines=int(getattr(self.config, "num_machines", 1) or 1),
            training_type="controlnet",
            deepspeed=bool(getattr(self.config, "deepspeed", False)),
            model_arch=getattr(self.config.model_arch, "value", self.config.model_arch),
            **build_turbocore_native_update_training_loop_kwargs(self.config),
            controlnet=self.controlnet,
        )
        self.training_loop.steps_per_epoch = steps_per_epoch
        self.training_loop.total_steps = total_steps
        
        self.training_loop.on_step_end = self._on_step_end
        self.training_loop.on_epoch_end = self._on_epoch_end
        
        # Run loop
        self._log("Starting ControlNet training...")
        for epoch in range(self.config.epochs):
            if self._should_stop: break
            if hasattr(dataset, "set_current_epoch"):
                dataset.set_current_epoch(epoch)
            self.training_loop.train_epoch(dataloader, epoch)
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_model(epoch + 1)
        
        if not self._should_stop:
            self._save_model(self.config.epochs, final=True)

    def _save_model(self, epoch: int, final: bool = False):
        """Save ControlNet weights."""
        output_dir = Path(self.config.output_dir)
        suffix = "final" if final else f"epoch{epoch:02d}"
        filename = f"{self.config.output_name}_{suffix}"
        save_path = output_dir / filename
        
        self.controlnet.save_pretrained(str(save_path))
        self._log(f"ControlNet saved to {save_path}")

    def get_trainable_params(self):
        return list(self.controlnet.parameters())
