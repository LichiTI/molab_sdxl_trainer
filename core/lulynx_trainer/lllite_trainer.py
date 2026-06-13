"""ControlNet-LLLite Trainer for Lulynx.

Warehouse implementation of LLLite training — lightweight LoRA-like
ControlNet adapter that injects per-layer modules into the frozen UNet
instead of training a full ControlNet copy.

Only the LLLite adapter modules and the conditioning encoder are trained;
the base UNet, VAE, and text encoders remain frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .trainer import LulynxTrainer
from .config import LulynxConfig
from .model_loader import ModelLoader
from .dataset_loader import CaptionDataset, create_dataloader
from .training_loop import TrainingLoop
from .turbocore_native_update_route_binding import build_turbocore_native_update_training_loop_kwargs
from .runtime_optimizations import build_runtime_optimization_plan
from .lllite import (
    inject_lllite,
    remove_lllite,
    set_lllite_conditioning,
    get_lllite_state_dict,
    load_lllite_state_dict,
    ConditioningEncoder,
)
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


class _LLLiteParamWrapper:
    """Wraps LLLite adapter + encoder params to satisfy the lora_injector
    interface used by TrainingLoop for gradient clipping / optimizer."""

    def __init__(self, encoder: ConditioningEncoder, unet: nn.Module):
        self._encoder = encoder
        self._unet = unet
        self.injected_layers: Dict[str, Any] = {}

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        params = list(self._encoder.parameters())
        for name in getattr(self._unet, "_lllite_adapters", []):
            from .lllite import _get_adapter
            adapter = _get_adapter(self._unet, name)
            params.extend(p for p in adapter.parameters() if p.requires_grad)
        return params

    def get_param_groups(self, base_lr: float = 1e-4, weight_decay: float = 0.0):
        return [{"params": self.get_trainable_params(), "lr": base_lr, "weight_decay": weight_decay}]


class LLLiteTrainer(LulynxTrainer):
    """LLLite ControlNet trainer — lightweight adapter approach."""

    def __init__(self, config: Optional[LulynxConfig] = None):
        super().__init__(config)
        self._lllite_encoder: Optional[ConditioningEncoder] = None

    def prepare(self):
        """Prepare LLLite training."""
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

        # Freeze the entire base model
        self.model.unet.requires_grad_(False)
        self.model.vae.requires_grad_(False)
        self.model.text_encoder_1.requires_grad_(False)
        if self.model.text_encoder_2:
            self.model.text_encoder_2.requires_grad_(False)

        # Inject LLLite adapters into the UNet
        cond_emb_dim = int(getattr(self.config, "lllite_cond_emb_dim", 32))
        mlp_dim = int(getattr(self.config, "lllite_mlp_dim", 64))
        lllite_dropout = float(getattr(self.config, "lllite_dropout", 0.0))
        skip_input = bool(getattr(self.config, "lllite_skip_input_blocks", False))
        skip_output = bool(getattr(self.config, "lllite_skip_output_blocks", True))

        self._lllite_encoder, injected = inject_lllite(
            self.model.unet,
            cond_emb_dim=cond_emb_dim,
            mlp_dim=mlp_dim,
            dropout=lllite_dropout,
            multiplier=1.0,
            skip_input_blocks=skip_input,
            skip_output_blocks=skip_output,
            transformer_only=True,
            attn_qkv_only=True,
        )

        # Move encoder to device
        self._lllite_encoder.to(self.device, dtype=self.dtype)

        # Load resume weights if provided
        self._load_lllite_weights()

        # Disable preview (no LLLite sampler wired yet)
        if (
            getattr(self.config, "sample_every", 0) > 0
            or getattr(self.config, "sample_every_n_epochs", 0) > 0
            or self._get_sample_prompts_list()
        ):
            self._log("LLLite preview is not wired to a sampler yet; skipping preview generation.")
            self.config.sample_every = 0
            self.config.sample_every_n_epochs = 0
        self._sampler = None

        # Set up param wrapper
        self.lora_injector = _LLLiteParamWrapper(self._lllite_encoder, self.model.unet)

        trainable = self.lora_injector.get_trainable_params()
        total_params = sum(p.numel() for p in trainable)
        self._log(f"LLLite trainable parameters: {total_params:,} "
                  f"(cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}, "
                  f"{len(injected)} adapter modules)")

        return self

    def _load_lllite_weights(self):
        weight_path = (
            getattr(self.config, "network_weights_path", "")
            or getattr(self.config, "resume_path", "")
        )
        if not weight_path:
            return
        path = Path(weight_path)
        if not path.exists():
            self._log(f"LLLite weights path not found: {path}")
            return

        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path), device="cpu")
        elif path.suffix.lower() in {".pt", ".ckpt", ".bin"}:
            state = safe_torch_load(str(path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        else:
            self._log(f"Unsupported LLLite weights file extension: {path.suffix}")
            return

        if not isinstance(state, dict):
            self._log(f"LLLite weights did not contain a state dict: {path}")
            return

        load_lllite_state_dict(self.model.unet, state)
        self._log(f"Loaded LLLite weights from {path}")

    def _run_training(self):
        """Run LLLite training loop."""
        self._log("Creating dataset for LLLite...")
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

        # Specialized training loop for LLLite
        class LLLiteTrainingLoop(TrainingLoop):
            def __init__(self, *args, lllite_encoder=None, **kwargs):
                self._lllite_encoder = lllite_encoder
                super().__init__(*args, **kwargs)

            def train_step(self, batch: Dict, accumulation_steps: Optional[int] = None) -> float:
                accumulation_steps = max(int(accumulation_steps or self.gradient_accumulation_steps), 1)
                images = batch["images"].to(self.device, dtype=self.dtype)
                control_images = batch.get("control_images", images).to(self.device, dtype=self.dtype)
                captions = batch["captions"]

                # Encode control image → conditioning embeddings (once per step)
                with torch.no_grad():
                    cond_embs = self._lllite_encoder(control_images.to(self.dtype))
                set_lllite_conditioning(self.unet, cond_embs)

                # Encode target images
                with torch.no_grad():
                    latents = self.vae.encode(images.to(torch.float32)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents = latents.to(dtype=self.dtype)

                # Encode text
                prompt_embeds = self._encode_prompt(captions)
                time_embeds = self._get_timestep_embedding(
                    latents.shape[0],
                    batch.get("original_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("target_sizes", [(1024, 1024)] * latents.shape[0]),
                    batch.get("crop_coords", [(0, 0, 1024, 1024)] * latents.shape[0]),
                )

                # Sample noise & timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device,
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Build UNet kwargs
                unet_kwargs = {
                    "sample": noisy_latents,
                    "timestep": timesteps,
                    "encoder_hidden_states": prompt_embeds["encoder_hidden_states"],
                }
                pooled = prompt_embeds.get("pooled_prompt_embeds")
                time_cond = time_embeds.get("added_cond_kwargs", {}) if time_embeds else {}
                if time_cond or pooled is not None:
                    added_cond_kwargs = dict(time_cond)
                    if pooled is not None:
                        added_cond_kwargs["text_embeds"] = pooled
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                # Forward with LLLite adapters active
                noise_pred = self.unet(**unet_kwargs).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss / accumulation_steps
                loss.backward()
                return loss.item() * accumulation_steps

        self.training_loop = LLLiteTrainingLoop(
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
            training_type="lllite",
            deepspeed=bool(getattr(self.config, "deepspeed", False)),
            model_arch=getattr(self.config.model_arch, "value", self.config.model_arch),
            **build_turbocore_native_update_training_loop_kwargs(self.config),
            lllite_encoder=self._lllite_encoder,
        )
        self.training_loop.steps_per_epoch = steps_per_epoch
        self.training_loop.total_steps = total_steps

        self.training_loop.on_step_end = self._on_step_end
        self.training_loop.on_epoch_end = self._on_epoch_end

        # Run training
        self._log("Starting LLLite training...")
        for epoch in range(self.config.epochs):
            if self._should_stop:
                break
            if hasattr(dataset, "set_current_epoch"):
                dataset.set_current_epoch(epoch)
            self.training_loop.train_epoch(dataloader, epoch)
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_model(epoch + 1)

        if not self._should_stop:
            self._save_model(self.config.epochs, final=True)

        # Cleanup
        remove_lllite(self.model.unet)

    def _save_model(self, epoch: int, final: bool = False):
        """Save LLLite adapter weights as safetensors."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "final" if final else f"epoch{epoch:02d}"
        filename = f"{self.config.output_name}_{suffix}.safetensors"
        save_path = output_dir / filename

        state = get_lllite_state_dict(self.model.unet)
        try:
            from safetensors.torch import save_file
            save_file(state, str(save_path))
            self._log(f"LLLite saved to {save_path} ({len(state)} tensors)")
        except ImportError:
            torch.save(state, str(save_path.with_suffix(".pt")))
            self._log(f"LLLite saved to {save_path.with_suffix('.pt')} (safetensors not available)")

    def get_trainable_params(self):
        params = list(self._lllite_encoder.parameters())
        for name in getattr(self.model.unet, "_lllite_adapters", []):
            from .lllite import _get_adapter
            adapter = _get_adapter(self.model.unet, name)
            params.extend(p for p in adapter.parameters() if p.requires_grad)
        return params

