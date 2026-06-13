"""
IP-Adapter Trainer

Specialized trainer for image-conditioned adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from .trainer import LulynxTrainer
from .config import LulynxConfig, NetworkType
from .model_loader import ModelLoader
from .ip_adapter_injector import IPAdapterInjector
from .ip_adapter_layers import ImageProjModel, Resampler
from .dataset_loader import CaptionDataset, create_dataloader
from .training_loop import TrainingLoop
from .turbocore_native_update_route_binding import build_turbocore_native_update_training_loop_kwargs
from .runtime_optimizations import build_runtime_optimization_plan

logger = logging.getLogger(__name__)

class IPAdapterTrainer(LulynxTrainer):
    """IP-Adapter 专门训练器"""
    
    def __init__(self, config: Optional[LulynxConfig] = None):
        super().__init__(config)
        self.image_encoder = None
        self.image_proj_model = None
        self.ip_injector = None
        
        # IP-Adapter specific settings (configurable via schema, with defaults)
        self.num_tokens = 4
        self.image_encoder_path = "openai/clip-vit-large-patch14"
    
    def prepare(self):
        """准备 IP-Adapter 训练"""
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
        
        image_encoder_path = getattr(self.config, "ip_image_encoder_path", None) or self.image_encoder_path
        self._log(f"Loading image encoder: {image_encoder_path}...")
        from transformers import CLIPVisionModelWithProjection
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
            torch_dtype=self.dtype
        ).to(self.device)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        
        self._log("Initializing IP-Adapter components...")
        
        # Decide projection model type
        clip_dim = self.image_encoder.config.hidden_size # Usually 1024
        unet_cross_attention_dim = self.model.unet.config.cross_attention_dim
        
        config_num_tokens = getattr(self.config, "ip_num_tokens", None)
        if getattr(self.config.model_arch, "value", self.config.model_arch) == "sdxl":
            # SDXL typically uses Resampler
            self.num_tokens = config_num_tokens or 16
            self.image_proj_model = Resampler(
                dim=unet_cross_attention_dim,
                depth=4,
                heads=12,
                num_queries=self.num_tokens,
                embedding_dim=clip_dim,
                output_dim=unet_cross_attention_dim,
            ).to(self.device, dtype=self.dtype)
        else:
            # SD1.5 typically uses Linear projection
            self.num_tokens = config_num_tokens or 4
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=unet_cross_attention_dim,
                clip_embeddings_dim=clip_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=self.dtype)
            
        self._log(f"Injecting IP-Adapter attention layers (tokens: {self.num_tokens})...")
        self.ip_injector = IPAdapterInjector(self.model.unet, num_tokens=self.num_tokens)
        self.ip_injector.inject()

        # SDPA/xformers bypass the processor-based forward path.
        # Force processor-based attention so IPAdapterAttnProcessor is actually called.
        for name, module in self.model.unet.named_modules():
            if hasattr(module, 'use_sdpa'):
                module.use_sdpa = False
            if hasattr(module, 'use_xformers'):
                module.use_xformers = False

        # Enable training for new layers
        self.image_proj_model.train()
        for processor in self.ip_injector.processors.values():
            processor.train()

        self._load_resume_or_network_weights()
        self._disable_unsupported_preview()
            
        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in trainable_params)
        self._log(f"IP-Adapter trainable parameters: {total_params:,}")
        
        return self

    def _disable_unsupported_preview(self):
        if (
            getattr(self.config, "sample_every", 0) > 0
            or getattr(self.config, "sample_every_n_epochs", 0) > 0
            or self._get_sample_prompts_list()
        ):
            self._log(
                "IP-Adapter preview is not wired to an image-conditioned sampler yet; skipping preview generation."
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
        if not path.is_file():
            self._log(f"IP-Adapter resume/network_weights path is not a supported file: {path}")
            return

        if path.suffix.lower() not in {".safetensors", ".pt", ".ckpt", ".bin"}:
            self._log(f"Unsupported IP-Adapter weights file extension: {path.suffix}")
            return

        missing, unexpected = self.ip_injector.load_ip_adapter(
            str(path),
            image_proj_model=self.image_proj_model,
        )
        self._log(
            f"Loaded IP-Adapter weights from {path} ({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )

    def get_trainable_params(self):
        """Get parameters to be optimized."""
        params = list(self.image_proj_model.parameters())
        params.extend(self.ip_injector.get_trainable_params())
        return params

    def _create_optimizer(self):
        """Create optimizer for IP-Adapter layers."""
        trainable_params = self.get_trainable_params()
        
        # Using AdamW as default for IP-Adapter
        from .config import OptimizerType
        if self.config.optimizer == OptimizerType.ADAMW_8BIT:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            except ImportError:
                pass
        
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _run_training(self):
        """Execute IP-Adapter training loop."""
        self._log("Creating dataset for IP-Adapter...")
        
        dataset = CaptionDataset(
            data_dir=self.config.train_data_dir,
            resolution=self.config.resolution,
            caption_extension=self.config.caption_extension,
            enable_bucket=self.config.enable_bucket,
            shuffle_caption=self.config.shuffle_caption,
            shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
            keep_tokens=getattr(self.config, "keep_tokens", 0),
            keep_tokens_separator=getattr(self.config, "keep_tokens_separator", ""),
            caption_dropout_rate=getattr(self.config, "caption_dropout_rate", 0.0),
            caption_dropout_every_n_epochs=getattr(self.config, "caption_dropout_every_n_epochs", 0),
            tag_dropout_rate=getattr(self.config, "tag_dropout_rate", 0.0),
            caption_tag_dropout_targets=getattr(self.config, "caption_tag_dropout_targets", ""),
            caption_tag_dropout_target_mode=getattr(self.config, "caption_tag_dropout_target_mode", "drop_all"),
            caption_tag_dropout_target_count=getattr(self.config, "caption_tag_dropout_target_count", 1),
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
        
        # We need a specialized training loop or override the step
        # Since TrainingLoop class is quite generic, we might need to patch it 
        # or implement a subclass if needed. For now let's use the generic one 
        # and handle the IP-Adapter logic in the trainer's step callback if possible, 
        # but TrainingLoop.train_step is internal.
        
        # Better: Implementation of IPAdapterTrainingLoop
        from .training_loop import TrainingLoop
        
        class IPAdapterTrainingLoop(TrainingLoop):
            def __init__(self, *args, **kwargs):
                self.image_encoder = kwargs.pop("image_encoder")
                self.image_proj_model = kwargs.pop("image_proj_model")
                super().__init__(*args, **kwargs)

            def train_step(self, batch: Dict, accumulation_steps: Optional[int] = None) -> float:
                accumulation_steps = max(int(accumulation_steps or self.gradient_accumulation_steps), 1)
                images = batch["images"].to(self.device, dtype=self.dtype)
                guidance_images = batch["guidance_images"].to(self.device, dtype=self.dtype)
                captions = batch["captions"]
                
                # 1. Encode guidance images — use hidden states (hidden_size dim),
                #    not image_embeds (projection_dim, smaller).
                with torch.no_grad():
                    clip_out = self.image_encoder(guidance_images)
                    image_embeds = clip_out.last_hidden_state  # [B, seq_len, hidden_size]

                # 2. Project image embeddings to UNet attention space
                ip_tokens = self.image_proj_model(image_embeds) # [B, num_tokens, cross_attention_dim]
                
                # 3. Standard Diffusion Step
                # Encode target images to latent space
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Encode text
                prompt_embeds = self._encode_prompt(captions)
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
                ).long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                # Route ip_adapter_image_embeds through cross_attention_kwargs
                # so diffusers forwards it to custom IPAdapterAttnProcessor
                unet_kwargs = {
                    "sample": noisy_latents,
                    "timestep": timesteps,
                    "encoder_hidden_states": prompt_embeds["encoder_hidden_states"],
                    "cross_attention_kwargs": {"ip_adapter_image_embeds": ip_tokens},
                }
                
                # Handle SDXL additional cond
                time_embeds = self._get_timestep_embedding(
                    latents.shape[0],
                    batch.get("original_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("target_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("crop_coords", [(0, 0, 1024, 1024)] * latents.shape[0]),
                )
                if time_embeds:
                    unet_kwargs["added_cond_kwargs"] = {
                        **time_embeds.get("added_cond_kwargs", {}),
                        "text_embeds": prompt_embeds.get("pooled_prompt_embeds"),
                    }
                
                noise_pred = self.unet(**unet_kwargs).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                loss = loss / accumulation_steps
                loss.backward()

                return loss.item() * accumulation_steps

        self.training_loop = IPAdapterTrainingLoop(
            unet=self.model.unet,
            text_encoder_1=self.model.text_encoder_1,
            text_encoder_2=self.model.text_encoder_2,
            vae=self.model.vae,
            tokenizer_1=self.model.tokenizer_1,
            tokenizer_2=self.model.tokenizer_2,
            noise_scheduler=self.model.noise_scheduler,
            lora_injector=self.ip_injector, # Not actually LoRA but shares parameter method
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
            training_type="ip-adapter",
            deepspeed=bool(getattr(self.config, "deepspeed", False)),
            model_arch=getattr(self.config.model_arch, "value", self.config.model_arch),
            **build_turbocore_native_update_training_loop_kwargs(self.config),
            image_encoder=self.image_encoder,
            image_proj_model=self.image_proj_model,
        )
        self.training_loop.steps_per_epoch = steps_per_epoch
        self.training_loop.total_steps = total_steps
        
        self.training_loop.on_step_end = self._on_step_end
        self.training_loop.on_epoch_end = self._on_epoch_end
        
        # Run loop
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._log("Starting IP-Adapter training...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            if self._should_stop: break
            if hasattr(dataset, "set_current_epoch"):
                dataset.set_current_epoch(epoch)
            self.training_loop.train_epoch(dataloader, epoch)
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_model(epoch + 1)
        
        if not self._should_stop:
            self._save_model(self.config.epochs, final=True)
            
        self._log(f"Training completed in {time.time() - start_time:.1f}s")

    def _save_model(self, epoch: int, final: bool = False):
        """Save IP-Adapter weights."""
        output_dir = Path(self.config.output_dir)
        suffix = "final" if final else f"epoch{epoch:02d}"
        filename = f"{self.config.output_name}_{suffix}.safetensors"
        save_path = output_dir / filename
        
        self.ip_injector.save_ip_adapter(str(save_path), self.image_proj_model)
        self._log(f"IP-Adapter saved to {save_path}")
