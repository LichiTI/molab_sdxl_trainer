"""Generation request contracts."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import ArtifactManifest, BaseRequest, RunResult


class GenerationRequest(BaseRequest):
    """API-ready request for image generation and training preview generation.

    The contract is intentionally model-agnostic. Runners may specialize it for
    Anima, Newbie, SDXL, or future generation backends after validation.
    """

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "generation.image"
    model_path: str = ""
    adapter_path: str = ""
    vae_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 5.0
    sampler: str = "auto"
    cns: str = ""
    cns_strength: float = 1.0
    smc_cfg: bool = False
    smc_cfg_lambda: float = 5.0
    smc_cfg_alpha: float = 0.2
    tgate_probe: bool = False
    tgate_start_step: int = 0
    tgate_min_block: int = 0
    spectrum_probe: bool = False
    spectrum_window_size: float = 2.0
    spectrum_flex_window: float = 0.25
    spectrum_warmup_steps: int = 6
    spectrum_stop_caching_step: int = -1
    smoothcache_probe: bool = False
    smoothcache_error_threshold: float = 0.08
    smoothcache_warmup_steps: int = 2
    cache_seam_backend: str = "none"
    cache_seam_window_size: float = 3.0
    # High-level inference-accel scheme. Resolves (via resolve_inference_accel_scheme)
    # to the spectrum/smoothcache probe + seam pair; the low-level cache_seam_backend /
    # *_probe fields above stay for backward compatibility and advanced/CLI use.
    inference_accel_scheme: str = "none"
    seed: int = -1
    dtype: str = "auto"
    device: str = "auto"
    batch_size: int = 1
    arch: str = "auto"
    output_dir: str = ""
    output_name: str = ""
    input_image_path: str = ""
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    dry_run: bool = False

    @field_validator("width", "height")
    @classmethod
    def _validate_dimension(cls, value: int) -> int:
        value = int(value)
        if value < 64:
            raise ValueError("generation dimensions must be >= 64")
        if value % 8 != 0:
            raise ValueError("generation dimensions must be divisible by 8")
        return value

    @field_validator("steps", "batch_size")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    @field_validator("guidance_scale")
    @classmethod
    def _validate_guidance(cls, value: float) -> float:
        value = float(value)
        if value < 0:
            raise ValueError("guidance_scale must be >= 0")
        return value

    @field_validator("cns_strength")
    @classmethod
    def _validate_cns_strength(cls, value: float) -> float:
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError("cns_strength must be within [0, 1]")
        return value

    @field_validator("smc_cfg_lambda")
    @classmethod
    def _validate_smc_cfg_lambda(cls, value: float) -> float:
        value = float(value)
        if value < 0:
            raise ValueError("smc_cfg_lambda must be >= 0")
        return value

    @field_validator("smc_cfg_alpha")
    @classmethod
    def _validate_smc_cfg_alpha(cls, value: float) -> float:
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError("smc_cfg_alpha must be within [0, 1]")
        return value

    @field_validator("tgate_start_step", "tgate_min_block")
    @classmethod
    def _validate_nonnegative_probe_int(cls, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("T-GATE probe indices must be >= 0")
        return value

    @field_validator("spectrum_window_size")
    @classmethod
    def _validate_spectrum_window_size(cls, value: float) -> float:
        value = float(value)
        if value < 1:
            raise ValueError("spectrum_window_size must be >= 1")
        return value

    @field_validator("spectrum_flex_window")
    @classmethod
    def _validate_spectrum_flex_window(cls, value: float) -> float:
        value = float(value)
        if value < 0:
            raise ValueError("spectrum_flex_window must be >= 0")
        return value

    @field_validator("spectrum_warmup_steps")
    @classmethod
    def _validate_spectrum_warmup_steps(cls, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("spectrum_warmup_steps must be >= 0")
        return value

    @field_validator("smoothcache_error_threshold")
    @classmethod
    def _validate_smoothcache_error_threshold(cls, value: float) -> float:
        value = float(value)
        if value < 0:
            raise ValueError("smoothcache_error_threshold must be >= 0")
        return value

    @field_validator("smoothcache_warmup_steps")
    @classmethod
    def _validate_smoothcache_warmup_steps(cls, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("smoothcache_warmup_steps must be >= 0")
        return value

    @field_validator("cache_seam_backend")
    @classmethod
    def _validate_cache_seam_backend(cls, value: str) -> str:
        token = str(value or "none").strip().lower().replace("-", "").replace("_", "")
        if token not in {"none", "spectrum", "smoothcache", "tgate"}:
            raise ValueError("cache_seam_backend must be one of none|spectrum|smoothcache|tgate")
        return token

    @field_validator("cache_seam_window_size")
    @classmethod
    def _validate_cache_seam_window_size(cls, value: float) -> float:
        value = float(value)
        if value < 2:
            raise ValueError("cache_seam_window_size must be >= 2")
        return value

    @field_validator("inference_accel_scheme")
    @classmethod
    def _validate_inference_accel_scheme(cls, value: str) -> str:
        token = str(value or "none").strip().lower().replace("-", "").replace("_", "")
        if token not in {"none", "spectrum", "smoothcache"}:
            raise ValueError("inference_accel_scheme must be one of none|spectrum|smoothcache")
        return token

    @model_validator(mode="after")
    def _validate_prompt_for_real_run(self) -> "GenerationRequest":
        if not self.dry_run and not self.prompt.strip():
            raise ValueError("prompt is required unless dry_run is true")
        return self

    def required_paths(self) -> Dict[str, str]:
        """Return non-empty path fields that a runner may resolve/check."""

        keys = ("model_path", "adapter_path", "vae_path", "input_image_path", "output_dir")
        return {key: str(getattr(self, key) or "") for key in keys if str(getattr(self, key) or "").strip()}


class GenerationResult(RunResult):
    """RunResult specialization for generation outputs."""

    images: List[ArtifactManifest] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mirror_images_into_artifacts(self) -> "GenerationResult":
        if self.images and not self.artifacts:
            self.artifacts = list(self.images)
        return self
