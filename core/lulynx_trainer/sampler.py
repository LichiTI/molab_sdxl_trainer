"""
训练中采样器

在训练过程中生成样本图像，实时查看训练效果
"""

import enum
import torch
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path
from PIL import Image
from .model_family import get_model_family, ModelFamily
from .sampler_capabilities import create_diffusers_scheduler, runtime_sampler_name

logger = logging.getLogger(__name__)


class PreviewState(enum.Enum):
    """Preview capability state for a training session."""
    UNSUPPORTED = "unsupported"
    ADAPTER_INSPECT = "adapter_inspect"
    DRY_PREVIEW = "dry_preview"
    REAL_PREVIEW = "real_preview"


def get_preview_state(trainer) -> PreviewState:
    """Determine the current preview capability for a trainer."""
    if not trainer.model:
        return PreviewState.UNSUPPORTED
    preview_device = str(getattr(getattr(trainer, "config", None), "preview_device", "gpu") or "gpu").strip().lower()
    if preview_device == "off":
        return PreviewState.UNSUPPORTED
    if preview_device == "cpu":
        return PreviewState.ADAPTER_INSPECT

    model_arch = (
        getattr(trainer.model, "model_arch", None)
        or getattr(getattr(trainer, "config", None), "model_arch", None)
    )
    family = get_model_family(model_arch)

    if family.default_sampler_pipeline is None:
        return PreviewState.UNSUPPORTED

    unet = getattr(trainer.model, "unet", None) or getattr(trainer.model, "dit", None)
    vae = getattr(trainer.model, "vae", None)
    te = getattr(trainer.model, "text_encoder_1", None)
    tok = getattr(trainer.model, "tokenizer_1", None)

    if unet is not None and vae is not None and te is not None and tok is not None:
        return PreviewState.REAL_PREVIEW

    if unet is not None:
        return PreviewState.ADAPTER_INSPECT

    return PreviewState.UNSUPPORTED


def get_adapter_metadata(trainer) -> Optional[Dict[str, Any]]:
    """Return adapter inspection metadata even when full preview is unavailable.

    Useful in cache-first mode where TE/VAE are released but the DiT with
    injected adapters is still available.
    """
    injector = getattr(trainer, "lora_injector", None)
    if injector is None:
        return None

    metadata: Dict[str, Any] = {}
    metadata["adapter_type"] = type(injector).__name__

    if hasattr(injector, "get_trainable_param_count"):
        metadata["trainable_params"] = injector.get_trainable_param_count()
    elif hasattr(injector, "trainable_params"):
        params = injector.trainable_params
        if isinstance(params, (list, tuple)):
            count = 0
            for group in params:
                if isinstance(group, dict):
                    for p in group.get("params", []):
                        count += p.numel()
                else:
                    count += group.numel()
            metadata["trainable_params"] = count

    if hasattr(injector, "network_type"):
        metadata["network_type"] = str(injector.network_type)
    if hasattr(injector, "rank"):
        metadata["rank"] = injector.rank
    if hasattr(injector, "alpha"):
        metadata["alpha"] = injector.alpha

    target_modules = getattr(injector, "target_modules", None) or getattr(injector, "targets", None)
    if target_modules is not None:
        metadata["target_modules"] = list(target_modules) if not isinstance(target_modules, list) else target_modules

    return metadata


class TrainingSampler:
    """
    训练中采样器
    
    在训练过程中临时应用 LoRA 权重生成样本图像
    """
    
    def __init__(
        self,
        unet,
        text_encoder_1,
        text_encoder_2,  # None for SD1.5
        vae,
        tokenizer_1,
        tokenizer_2,  # None for SD1.5
        noise_scheduler,
        lora_injector=None,
        sampler_name: str = "euler_a",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_arch: Optional[str] = None,
        sample_width: int = 0,
        sample_height: int = 0,
        sample_seed: int = 0,
        preview_device: str = "gpu",
        ephemeral_pipeline: bool = True,
        weight_residency_mode: str = "resident",
        weight_residency_min_params: int = 0,
        dit_block_residency_mode: str = "resident",
        dit_block_residency_min_params: int = 0,
        dit_block_prefetch_enabled: bool = False,
        dit_block_prefetch_depth: int = 1,
        sample_smc_cfg: bool = False,
        sample_smc_cfg_lambda: float = 5.0,
        sample_smc_cfg_alpha: float = 0.2,
        sample_tgate_probe: bool = False,
        sample_tgate_start_step: int = 0,
        sample_tgate_min_block: int = 0,
        sample_spectrum_probe: bool = False,
        sample_spectrum_window_size: float = 2.0,
        sample_spectrum_flex_window: float = 0.25,
        sample_spectrum_warmup_steps: int = 6,
        sample_spectrum_stop_caching_step: int = -1,
        sample_cache_seam_backend: str = "none",
        sample_cache_seam_window_size: float = 3.0,
        sample_smoothcache_probe: bool = False,
        sample_smoothcache_error_threshold: float = 0.08,
        sample_smoothcache_warmup_steps: int = 2,
        sample_algorithm: str = "sde",
        sample_sde_eta: float = 1.0,
        sample_cns_enabled: bool = False,
        sample_cns_gamma_path: str = "",
        sample_cns_strength: float = 1.0,
        sample_cns_eta: float = 1.0,
    ):
        self.unet = unet
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.noise_scheduler = noise_scheduler
        self.lora_injector = lora_injector
        self.sampler_name = sampler_name
        self.device = device
        self.dtype = dtype
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.sample_seed = sample_seed
        self.preview_device = str(preview_device or "gpu").strip().lower()
        self.ephemeral_pipeline = bool(ephemeral_pipeline)
        self.weight_residency_mode = str(weight_residency_mode or "resident").strip().lower()
        self.weight_residency_min_params = max(int(weight_residency_min_params or 0), 0)
        self.dit_block_residency_mode = str(dit_block_residency_mode or "resident").strip().lower()
        self.dit_block_residency_min_params = max(int(dit_block_residency_min_params or 0), 0)
        self.dit_block_prefetch_enabled = bool(dit_block_prefetch_enabled)
        self.dit_block_prefetch_depth = max(int(dit_block_prefetch_depth or 0), 0)
        self.sample_smc_cfg = bool(sample_smc_cfg)
        self.sample_smc_cfg_lambda = float(5.0 if sample_smc_cfg_lambda is None else sample_smc_cfg_lambda)
        self.sample_smc_cfg_alpha = float(0.2 if sample_smc_cfg_alpha is None else sample_smc_cfg_alpha)
        self.sample_tgate_probe = bool(sample_tgate_probe)
        self.sample_tgate_start_step = max(int(sample_tgate_start_step or 0), 0)
        self.sample_tgate_min_block = max(int(sample_tgate_min_block or 0), 0)
        self.sample_spectrum_probe = bool(sample_spectrum_probe)
        self.sample_spectrum_window_size = max(float(2.0 if sample_spectrum_window_size is None else sample_spectrum_window_size), 1.0)
        self.sample_spectrum_flex_window = max(float(0.0 if sample_spectrum_flex_window is None else sample_spectrum_flex_window), 0.0)
        self.sample_spectrum_warmup_steps = max(int(sample_spectrum_warmup_steps or 0), 0)
        self.sample_spectrum_stop_caching_step = int(
            -1 if sample_spectrum_stop_caching_step is None else sample_spectrum_stop_caching_step
        )
        self.sample_cache_seam_backend = str(sample_cache_seam_backend or "none").strip().lower().replace("-", "_")
        self.sample_cache_seam_window_size = max(int(3 if sample_cache_seam_window_size is None else sample_cache_seam_window_size), 2)
        self.sample_smoothcache_probe = bool(sample_smoothcache_probe)
        self.sample_smoothcache_error_threshold = max(
            float(0.08 if sample_smoothcache_error_threshold is None else sample_smoothcache_error_threshold), 0.0
        )
        self.sample_smoothcache_warmup_steps = max(int(sample_smoothcache_warmup_steps or 0), 0)
        self.sample_algorithm = str(sample_algorithm or "sde").strip().lower()
        self.sample_sde_eta = max(0.0, float(1.0 if sample_sde_eta is None else sample_sde_eta))
        self.sample_cns_enabled = bool(sample_cns_enabled)
        self.sample_cns_gamma_path = str(sample_cns_gamma_path or "").strip()
        self.sample_cns_strength = max(0.0, min(1.0, float(1.0 if sample_cns_strength is None else sample_cns_strength)))
        self.sample_cns_eta = max(0.0, float(1.0 if sample_cns_eta is None else sample_cns_eta))
        self._micro_vae_decoder = None

        self._family: ModelFamily = get_model_family(model_arch)

        # Pipeline (延迟创建)
        self._pipeline = None
    
    def _get_pipeline(self):
        """获取或创建 Pipeline"""
        if self._pipeline is not None:
            return self._pipeline

        pipeline_type = self._family.default_sampler_pipeline

        try:
            if pipeline_type == "sdxl":
                from diffusers import StableDiffusionXLPipeline
                self._pipeline = StableDiffusionXLPipeline(
                    vae=self.vae,
                    text_encoder=self.text_encoder_1,
                    text_encoder_2=self.text_encoder_2,
                    tokenizer=self.tokenizer_1,
                    tokenizer_2=self.tokenizer_2,
                    unet=self.unet,
                    scheduler=self.noise_scheduler,
                )
            elif pipeline_type == "sd15":
                from diffusers import StableDiffusionPipeline
                self._pipeline = StableDiffusionPipeline(
                    vae=self.vae,
                    text_encoder=self.text_encoder_1,
                    tokenizer=self.tokenizer_1,
                    unet=self.unet,
                    scheduler=self.noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                )
            elif pipeline_type == "anima":
                # Anima uses flow-matching DiT sampling, not a diffusers pipeline
                self._pipeline = "anima_flow"
            elif pipeline_type == "newbie":
                # Newbie uses flow-matching DiT sampling with dual encoders
                self._pipeline = "newbie_flow"
            else:
                logger.warning(f"No default sampler pipeline for family '{self._family}'; sampling disabled.")
                return None

            if isinstance(self._pipeline, str):
                # Flow-matching pipelines don't need scheduler setup
                return self._pipeline

            self._pipeline.scheduler = self._create_scheduler(self._pipeline.scheduler)
            self._pipeline = self._pipeline.to(device=self.device, dtype=self.dtype)
            return self._pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return None

    def _create_scheduler(self, scheduler):
        try:
            return create_diffusers_scheduler(
                self.sampler_name,
                scheduler,
                family=self._family.default_sampler_pipeline,
            )
        except Exception as exc:
            logger.warning(f"Failed to switch sampler to {self.sampler_name}: {exc}")

        return scheduler
    
    @torch.no_grad()
    def _generate_anima(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """Generate preview using Anima flow-matching sampler."""
        # Choose between deterministic ODE (Euler) and stochastic SDE (ER-SDE) based on user preference
        use_sde = (self.sample_algorithm == "sde")

        if use_sde:
            # Use ER-SDE sampler (optionally with CNS colored noise)
            from .anima_sampler import sample_anima_ersde
            return sample_anima_ersde(
                dit_model=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder_1,
                tokenizer=self.tokenizer_1,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
                latent_channels=self._family.latent_channels,
                vae_scaling_factor=self._family.vae_scaling_factor,
                eta=self.sample_sde_eta,
                cns_gamma_path=self.sample_cns_gamma_path if self.sample_cns_enabled else "",
                cns_strength=self.sample_cns_strength if self.sample_cns_enabled else 1.0,
                smc_cfg=self.sample_smc_cfg,
                smc_cfg_lambda=self.sample_smc_cfg_lambda,
                smc_cfg_alpha=self.sample_smc_cfg_alpha,
                tgate_probe=self.sample_tgate_probe,
                tgate_start_step=self.sample_tgate_start_step,
                tgate_min_block=self.sample_tgate_min_block,
                spectrum_probe=self.sample_spectrum_probe,
                spectrum_window_size=self.sample_spectrum_window_size,
                spectrum_flex_window=self.sample_spectrum_flex_window,
                spectrum_warmup_steps=self.sample_spectrum_warmup_steps,
                spectrum_stop_caching_step=self.sample_spectrum_stop_caching_step,
                smoothcache_probe=self.sample_smoothcache_probe,
                smoothcache_error_threshold=self.sample_smoothcache_error_threshold,
                smoothcache_warmup_steps=self.sample_smoothcache_warmup_steps,
            )
        else:
            # Use deterministic ODE sampler (Euler/DPM-Solver)
            from .anima_sampler import sample_anima
            sampler_name = runtime_sampler_name(self.sampler_name, family="anima")
            return sample_anima(
                dit_model=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder_1,
                tokenizer=self.tokenizer_1,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
                sampler_name=sampler_name,
                latent_channels=self._family.latent_channels,
                vae_scaling_factor=self._family.vae_scaling_factor,
                smc_cfg=self.sample_smc_cfg,
                smc_cfg_lambda=self.sample_smc_cfg_lambda,
                smc_cfg_alpha=self.sample_smc_cfg_alpha,
                tgate_probe=self.sample_tgate_probe,
                tgate_start_step=self.sample_tgate_start_step,
                tgate_min_block=self.sample_tgate_min_block,
                spectrum_probe=self.sample_spectrum_probe,
                spectrum_window_size=self.sample_spectrum_window_size,
                spectrum_flex_window=self.sample_spectrum_flex_window,
                spectrum_warmup_steps=self.sample_spectrum_warmup_steps,
                spectrum_stop_caching_step=self.sample_spectrum_stop_caching_step,
                smoothcache_probe=self.sample_smoothcache_probe,
                smoothcache_error_threshold=self.sample_smoothcache_error_threshold,
                smoothcache_warmup_steps=self.sample_smoothcache_warmup_steps,
            )

    @torch.no_grad()
    def _generate_newbie(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """Generate preview using Newbie flow-matching sampler."""
        # Choose between deterministic ODE (Euler) and stochastic SDE (ER-SDE with CNS)
        if self.sample_cns_enabled and self.sample_cns_gamma_path:
            from .newbie_sampler import sample_newbie_ersde
            return sample_newbie_ersde(
                dit_model=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder_1,
                tokenizer=self.tokenizer_1,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
                latent_channels=self._family.latent_channels,
                vae_scaling_factor=self._family.vae_scaling_factor,
                cns_gamma_path=self.sample_cns_gamma_path,
                cns_strength=self.sample_cns_strength,
                eta=self.sample_cns_eta,
                smc_cfg=self.sample_smc_cfg,
                smc_cfg_lambda=self.sample_smc_cfg_lambda,
                smc_cfg_alpha=self.sample_smc_cfg_alpha,
            )
        else:
            from .newbie_sampler import sample_newbie
            sampler_name = runtime_sampler_name(self.sampler_name, family="newbie")
            return sample_newbie(
                dit_model=self.unet,
                vae=self.vae,
                text_encoder_1=self.text_encoder_1,
                text_encoder_2=self.text_encoder_2,
                tokenizer_1=self.tokenizer_1,
                tokenizer_2=self.tokenizer_2,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
                sampler_name=sampler_name,
                latent_channels=self._family.latent_channels,
                vae_scaling_factor=self._family.vae_scaling_factor,
                smc_cfg=self.sample_smc_cfg,
                smc_cfg_lambda=self.sample_smc_cfg_lambda,
                smc_cfg_alpha=self.sample_smc_cfg_alpha,
                tgate_probe=self.sample_tgate_probe,
                tgate_start_step=self.sample_tgate_start_step,
                tgate_min_block=self.sample_tgate_min_block,
                spectrum_probe=self.sample_spectrum_probe,
                spectrum_window_size=self.sample_spectrum_window_size,
                spectrum_flex_window=self.sample_spectrum_flex_window,
                spectrum_warmup_steps=self.sample_spectrum_warmup_steps,
                spectrum_stop_caching_step=self.sample_spectrum_stop_caching_step,
                smoothcache_probe=self.sample_smoothcache_probe,
                smoothcache_error_threshold=self.sample_smoothcache_error_threshold,
                smoothcache_warmup_steps=self.sample_smoothcache_warmup_steps,
            )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """
        生成样本图像

        Args:
            prompt: 正向提示词
            negative_prompt: 反向提示词
            num_inference_steps: 推理步数
            guidance_scale: CFG 强度
            width: 图像宽度
            height: 图像高度
            seed: 随机种子

        Returns:
            PIL.Image 或 None
        """
        # Override with config-specified preview dimensions/seed when set
        if self.sample_width > 0:
            width = self.sample_width
        if self.sample_height > 0:
            height = self.sample_height
        if self.sample_seed != 0 and seed is None:
            seed = self.sample_seed
        pipeline = self._get_pipeline()
        if pipeline is None:
            return None

        # Opt-in unified cache execution seam (default off -> bitwise parity).
        from .unified_cache_seam import build_cache_seam, cache_seam_context
        _seam_backend = str(getattr(self, "sample_cache_seam_backend", "none") or "none")
        cache_seam = build_cache_seam(
            enabled=_seam_backend.lower() not in {"", "none"},
            backend=_seam_backend,
            spectrum_window_size=int(getattr(self, "sample_cache_seam_window_size", 3) or 3),
        )

        # Flow-matching DiT pipelines
        if pipeline == "anima_flow":
            try:
                with cache_seam_context(cache_seam):
                    return self._generate_anima(
                        prompt=prompt, negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        width=width, height=height, seed=seed,
                    )
            finally:
                self._restore_dit_block_residency_after_preview()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if pipeline == "newbie_flow":
            try:
                with cache_seam_context(cache_seam):
                    return self._generate_newbie(
                        prompt=prompt, negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        width=width, height=height, seed=seed,
                    )
            finally:
                self._restore_dit_block_residency_after_preview()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Standard diffusers pipelines (SDXL / SD1.5)
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None

        try:
            # 生成图像
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

            image = result.images[0]
            return image

        except Exception as e:
            logger.warning(f"Preview image generation failed: {e}")
            return None
        finally:
            if self.ephemeral_pipeline and not isinstance(pipeline, str):
                self._pipeline = None
                try:
                    del pipeline
                except Exception:
                    pass
                self._restore_native_weight_residency_after_preview()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _restore_native_weight_residency_after_preview(self) -> None:
        if self._family.default_sampler_pipeline != "sdxl" or self.weight_residency_mode == "resident":
            return
        try:
            from .native_unet.weight_residency import apply_weight_residency

            apply_weight_residency(
                self.unet,
                mode=self.weight_residency_mode,
                min_parameter_count=self.weight_residency_min_params,
            )
        except Exception as exc:
            logger.warning("Failed to restore native weight residency after preview: %s", exc)

    def _restore_dit_block_residency_after_preview(self) -> None:
        pipeline = self._family.default_sampler_pipeline
        if pipeline not in {"anima", "newbie"} or self.dit_block_residency_mode == "resident":
            return
        try:
            if pipeline == "anima":
                from .anima_block_residency import apply_anima_block_residency

                apply_anima_block_residency(
                    self.unet,
                    mode=self.dit_block_residency_mode,
                    min_parameter_count=self.dit_block_residency_min_params,
                    device=self.device,
                    dtype=self.dtype,
                    prefetch_enabled=self.dit_block_prefetch_enabled,
                    prefetch_depth=self.dit_block_prefetch_depth,
                )
                return
            if pipeline == "newbie":
                from .newbie_block_residency import apply_newbie_block_residency

                apply_newbie_block_residency(
                    self.unet,
                    mode=self.dit_block_residency_mode,
                    min_parameter_count=self.dit_block_residency_min_params,
                    device=self.device,
                    dtype=self.dtype,
                    prefetch_enabled=self.dit_block_prefetch_enabled,
                    prefetch_depth=self.dit_block_prefetch_depth,
                )
        except Exception as exc:
            logger.warning("Failed to restore %s block residency after preview: %s", pipeline, exc)
    
    def generate_grid(
        self,
        prompts: List[str],
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = 42,
    ) -> Optional[Image.Image]:
        """
        生成图像网格
        
        Args:
            prompts: 提示词列表
            其他参数同 generate
        
        Returns:
            合并后的 PIL.Image
        """
        images = []
        
        for i, prompt in enumerate(prompts[:4]):  # 最多 4 个
            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed + i,
            )
            if image:
                images.append(image)
        
        if not images:
            return None
        
        # 创建 2x2 网格
        return self._create_grid(images, cols=2)
    
    def _create_grid(self, images: List[Image.Image], cols: int = 2) -> Image.Image:
        """创建图像网格"""
        if not images:
            return None
        
        n = len(images)
        rows = (n + cols - 1) // cols
        
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        
        for i, img in enumerate(images):
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste(img, (x, y))
        
        return grid

    def load_micro_decoder(self, model_type: str = "auto"):
        """Load a micro VAE decoder for fast latent preview."""
        try:
            from .micro_vae import load_micro_decoder
            arch = model_type
            if arch == "auto":
                arch = "sdxl" if self._family.name in ("sdxl",) else "sd15"
            self._micro_vae_decoder = load_micro_decoder(arch)
            self._micro_vae_decoder = self._micro_vae_decoder.to(device=self.device, dtype=self.dtype)
        except Exception as e:
            logger.warning(f"Failed to load micro VAE decoder: {e}")
            self._micro_vae_decoder = None

    def micro_preview(self, latents: "torch.Tensor") -> "Optional[Image.Image]":
        """Quick latent→image decode using micro VAE (no full pipeline)."""
        if self._micro_vae_decoder is None:
            return None
        try:
            from .micro_vae import micro_decode
            return micro_decode(latents, self._micro_vae_decoder)
        except Exception:
            return None


def create_sampler_from_trainer(trainer) -> Optional[TrainingSampler]:
    """从 LulynxTrainer 创建采样器"""
    if not trainer.model:
        return None
    preview_device = str(getattr(trainer.config, "preview_device", "gpu") or "gpu").strip().lower()
    if preview_device == "off":
        logger.info("Preview sampling disabled: preview_device=off")
        return None
    if preview_device == "cpu":
        from .cpu_preview_worker import CPUPreviewWorker

        return CPUPreviewWorker(
            output_dir=str(getattr(trainer.config, "output_dir", "outputs") or "outputs"),
            model_arch=str(getattr(trainer.model, "model_arch", None) or getattr(trainer.config, "model_arch", "") or ""),
            model_path=str(getattr(trainer.config, "pretrained_model_name_or_path", "") or ""),
        )

    model_arch = getattr(trainer.model, "model_arch", None) or getattr(getattr(trainer, "config", None), "model_arch", None)
    family = get_model_family(model_arch)
    if family.default_sampler_pipeline is None:
        return None

    # Anima / Newbie: the "unet" attribute holds the DiT model
    unet = getattr(trainer.model, "unet", None) or getattr(trainer.model, "dit", None)

    # For anima/newbie, text_encoder_2 and tokenizer_2 may not exist
    text_encoder_1 = getattr(trainer.model, "text_encoder_1", None)
    text_encoder_2 = getattr(trainer.model, "text_encoder_2", None)
    tokenizer_1 = getattr(trainer.model, "tokenizer_1", None)
    tokenizer_2 = getattr(trainer.model, "tokenizer_2", None)
    vae = getattr(trainer.model, "vae", None)
    noise_scheduler = getattr(trainer.model, "noise_scheduler", None)

    # For anima, create a dummy scheduler if none exists (flow matching doesn't use it)
    if noise_scheduler is None and model_arch in ("anima", "newbie"):
        noise_scheduler = None  # flow-matching samplers don't need a scheduler

    # Guard: cache-first / native modes release TE/VAE after cache build.
    # Without these components, the sampler cannot generate images.
    pipeline_type = family.default_sampler_pipeline
    if pipeline_type in ("anima", "newbie"):
        missing = []
        if unet is None:
            missing.append("DiT model")
        if vae is None:
            missing.append("VAE")
        if text_encoder_1 is None:
            missing.append("text encoder")
        if tokenizer_1 is None:
            missing.append("tokenizer")
        if missing:
            logger.info(
                "Preview sampling disabled for %s: %s unavailable "
                "(cache-first training releases these components after cache build).",
                model_arch, ", ".join(missing),
            )
            return None

    # Read sampling algorithm configuration from UI
    sample_algorithm = str(getattr(trainer.config, "sample_algorithm", "sde") or "sde").strip().lower()
    sample_sde_eta = float(getattr(trainer.config, "sample_sde_eta", 1.0))

    # Read CNS configuration
    sample_cns_enabled = bool(getattr(trainer.config, "sample_cns_enabled", False))
    sample_cns_gamma_path = str(getattr(trainer.config, "sample_cns_gamma_path", "") or "").strip()
    sample_cns_strength = float(getattr(trainer.config, "sample_cns_strength", 1.0))

    # Inference-accel coordination: a chosen scheme (sample_cache_seam_backend) must
    # also enable its decision probe, since a real block skip needs probe + seam both
    # on.  Observe-only single use (probe on, seam=none) stays intact via the OR below.
    from .unified_cache_seam import resolve_inference_accel_scheme
    _accel = resolve_inference_accel_scheme(
        str(getattr(trainer.config, "sample_cache_seam_backend", "none") or "none")
    )

    return TrainingSampler(
        unet=unet,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        vae=vae,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        lora_injector=getattr(trainer, "lora_injector", None),
        sampler_name=getattr(trainer.config, "sample_sampler", "euler_a"),
        device=trainer.device,
        dtype=trainer.dtype,
        model_arch=model_arch,
        sample_width=int(getattr(trainer.config, "sample_width", 0) or 0),
        sample_height=int(getattr(trainer.config, "sample_height", 0) or 0),
        sample_seed=int(getattr(trainer.config, "sample_seed", 0) or 0),
        preview_device=str(getattr(trainer.config, "preview_device", "gpu") or "gpu"),
        ephemeral_pipeline=bool(getattr(trainer.config, "ephemeral_preview_pipeline", True)),
        weight_residency_mode=str(getattr(trainer.config, "lulynx_weight_residency", "resident") or "resident"),
        weight_residency_min_params=int(getattr(trainer.config, "lulynx_weight_residency_min_params", 0) or 0),
        dit_block_residency_mode=(
            str(getattr(trainer.config, "anima_block_residency", "resident") or "resident")
            if family.default_sampler_pipeline == "anima"
            else str(getattr(trainer.config, "newbie_block_residency", "resident") or "resident")
            if family.default_sampler_pipeline == "newbie"
            else "resident"
        ),
        dit_block_residency_min_params=(
            int(getattr(trainer.config, "anima_block_residency_min_params", 0) or 0)
            if family.default_sampler_pipeline == "anima"
            else int(getattr(trainer.config, "newbie_block_residency_min_params", 0) or 0)
            if family.default_sampler_pipeline == "newbie"
            else 0
        ),
        dit_block_prefetch_enabled=(
            bool(getattr(trainer.config, "anima_block_prefetch", False))
            if family.default_sampler_pipeline == "anima"
            else bool(getattr(trainer.config, "newbie_block_prefetch", False))
            if family.default_sampler_pipeline == "newbie"
            else False
        ),
        dit_block_prefetch_depth=(
            int(getattr(trainer.config, "anima_block_prefetch_depth", 1) or 0)
            if family.default_sampler_pipeline == "anima"
            else int(getattr(trainer.config, "newbie_block_prefetch_depth", 1) or 0)
            if family.default_sampler_pipeline == "newbie"
            else 0
        ),
        sample_algorithm=sample_algorithm,
        sample_sde_eta=sample_sde_eta,
        sample_smc_cfg=bool(getattr(trainer.config, "sample_smc_cfg", False)),
        sample_smc_cfg_lambda=float(getattr(trainer.config, "sample_smc_cfg_lambda", 5.0)),
        sample_smc_cfg_alpha=float(getattr(trainer.config, "sample_smc_cfg_alpha", 0.2)),
        sample_tgate_probe=bool(getattr(trainer.config, "sample_tgate_probe", False)),
        sample_tgate_start_step=int(getattr(trainer.config, "sample_tgate_start_step", 0) or 0),
        sample_tgate_min_block=int(getattr(trainer.config, "sample_tgate_min_block", 0) or 0),
        sample_spectrum_probe=bool(getattr(trainer.config, "sample_spectrum_probe", False)) or _accel.spectrum_probe,
        sample_spectrum_window_size=float(getattr(trainer.config, "sample_spectrum_window_size", 2.0)),
        sample_spectrum_flex_window=float(getattr(trainer.config, "sample_spectrum_flex_window", 0.25)),
        sample_spectrum_warmup_steps=int(getattr(trainer.config, "sample_spectrum_warmup_steps", 6) or 0),
        sample_spectrum_stop_caching_step=int(getattr(trainer.config, "sample_spectrum_stop_caching_step", -1)),
        sample_cache_seam_backend=str(getattr(trainer.config, "sample_cache_seam_backend", "none") or "none"),
        sample_cache_seam_window_size=float(getattr(trainer.config, "sample_cache_seam_window_size", 3.0) or 3.0),
        sample_smoothcache_probe=bool(getattr(trainer.config, "sample_smoothcache_probe", False)) or _accel.smoothcache_probe,
        sample_smoothcache_error_threshold=float(getattr(trainer.config, "sample_smoothcache_error_threshold", 0.08)),
        sample_smoothcache_warmup_steps=int(getattr(trainer.config, "sample_smoothcache_warmup_steps", 2) or 0),
        sample_cns_enabled=sample_cns_enabled,
        sample_cns_gamma_path=sample_cns_gamma_path,
        sample_cns_strength=sample_cns_strength,
        sample_cns_eta=float(getattr(trainer.config, "sample_cns_eta", 1.0)),
    )
