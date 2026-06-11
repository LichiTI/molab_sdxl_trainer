"""Generic diffusers Attention processor adapters.

This module owns the reusable ``hidden_states -> QKV -> kernel -> output``
path for U-Net style diffusers models. Runtime-specific attention backends
plug in as small kernel adapters.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

from .attention_kernel_adapters import (
    flash2_attention_bhnd,
    sage_attention_bhnd,
    sdpa_attention_bhnd,
    sparge2_attention_bhnd,
    torch_attention_bhnd,
    xformers_attention_bhnd,
)


class DiffusersAttentionKernel(Protocol):
    backend_id: str

    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        ...


class _FunctionDiffusersAttentionKernel:
    def __init__(self, backend_id: str, run_fn: Callable[..., torch.Tensor]):
        if not callable(run_fn):
            raise RuntimeError(f"{backend_id} diffusers attention kernel is not callable")
        self.backend_id = backend_id
        self._run_fn = run_fn

    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        return self._run_fn(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=0.0,
            scale=scale,
            causal=causal,
        )


class SdpaDiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    def __init__(self):
        super().__init__("sdpa", sdpa_attention_bhnd)


class TorchDiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    def __init__(self):
        super().__init__("torch", torch_attention_bhnd)


class XformersDiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    def __init__(self):
        super().__init__("xformers", xformers_attention_bhnd)

    @classmethod
    def create(cls) -> "XformersDiffusersAttentionKernel":
        try:
            from xformers.ops import memory_efficient_attention
        except Exception as exc:
            raise RuntimeError(f"xformers requested but xformers.ops is unavailable: {exc}") from exc
        if not callable(memory_efficient_attention):
            raise RuntimeError("xformers requested but memory_efficient_attention is not callable")
        return cls()


class SageDiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    def __init__(self):
        super().__init__("sageattn", sage_attention_bhnd)

    @classmethod
    def create(cls) -> "SageDiffusersAttentionKernel":
        try:
            from sageattention import sageattn
        except Exception as exc:
            raise RuntimeError(f"sageattention requested but package is unavailable: {exc}") from exc
        if not callable(sageattn):
            raise RuntimeError("sageattention requested but sageattn is not callable")
        return cls()


class Sparge2DiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    def __init__(self):
        super().__init__("spargeattn2", sparge2_attention_bhnd)

    @classmethod
    def create(cls) -> "Sparge2DiffusersAttentionKernel":
        try:
            import spas_sage_attn
        except Exception as exc:
            raise RuntimeError(f"spargeattn2 requested but spas_sage_attn is unavailable: {exc}") from exc
        if not any(
            callable(getattr(spas_sage_attn, name, None))
            for name in (
                "spas_sage2_attn_meansim_cuda",
                "spas_sage_attn_meansim_cuda",
                "block_sparse_sage2_attn_cuda",
            )
        ):
            raise RuntimeError("spargeattn2 requested but no callable Sparge/Sage2 attention kernel was found")
        return cls()


class Flash2DiffusersAttentionKernel(_FunctionDiffusersAttentionKernel):
    backend_id = "flash2"

    def __init__(self, flash_attn_func: Any):
        if not callable(flash_attn_func):
            raise RuntimeError("flash2 requested but flash_attn_func is not callable")
        self._flash_attn_func = flash_attn_func
        super().__init__("flash2", self._run_flash2)

    @classmethod
    def create(cls) -> "Flash2DiffusersAttentionKernel":
        if not torch.cuda.is_available():
            raise RuntimeError("flash2 requested for diffusers attention, but CUDA is not available")
        if bool(getattr(torch.version, "hip", None)):
            raise RuntimeError("flash2 requested for diffusers attention, but this runtime is ROCm/HIP")
        try:
            capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        except Exception:
            capability = None
        if capability is not None and capability < (8, 0):
            raise RuntimeError(
                f"flash2 requested for diffusers attention, but GPU capability {capability} is below SM80"
            )

        try:
            from flash_attn import flash_attn_func
        except Exception as exc:
            raise RuntimeError(f"flash2 requested but flash_attn is not importable: {exc}") from exc
        return cls(flash_attn_func)

    def _run_flash2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        dropout_p: float = 0.0,
        scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        return flash2_attention_bhnd(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
            causal=causal,
            flash_attn_func=self._flash_attn_func,
        )


class SlidingWindowDiffusersAttentionKernel:
    """Windowed self-attention kernel for SDXL/U-Net diffusers processors."""

    backend_id = "sliding_window"
    self_attention_only = True

    def __init__(self, profile: Any, fallback_kernel: DiffusersAttentionKernel | None = None):
        self.profile = profile
        self.fallback_kernel = fallback_kernel or SdpaDiffusersAttentionKernel()

    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        if attention_mask is not None:
            return self.fallback_kernel.run(
                query,
                key,
                value,
                attention_mask=attention_mask,
                scale=scale,
                causal=causal,
            )
        from .runtime_optimizations import sliding_window_attention

        return sliding_window_attention(
            query,
            key,
            value,
            window_size=int(getattr(self.profile, "window_size", 0) or 0),
            scale=scale,
            backend=str(getattr(self.profile, "backend", "auto") or "auto"),
            torch_fallback_max_tokens=int(getattr(self.profile, "torch_fallback_max_tokens", 2048) or 2048),
            launcher_attention_backend=str(getattr(self.profile, "launcher_attention_backend", "auto") or "auto"),
            flex_runtime_active=bool(getattr(self.profile, "flex_runtime_active", False)),
            causal=bool(causal),
        )


class GenericDiffusersAttnProcessor:
    """Common diffusers Attention processor with a pluggable kernel."""

    def __init__(self, kernel: DiffusersAttentionKernel):
        self.kernel = kernel

    @property
    def backend_id(self) -> str:
        return self.kernel.backend_id

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        is_self_attention = encoder_hidden_states is None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None

        sequence_length = (
            hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        fused_kv = getattr(attn, "_fused_kv", None)
        if fused_kv is not None:
            key, value = fused_kv(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        kernel = self.kernel
        if getattr(kernel, "self_attention_only", False) and not is_self_attention:
            kernel = getattr(kernel, "fallback_kernel", None) or SdpaDiffusersAttentionKernel()

        hidden_states = kernel.run(
            query,
            key,
            value,
            attention_mask=attention_mask,
            scale=getattr(attn, "scale", None),
            causal=bool(getattr(attn, "is_causal", False)),
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def build_diffusers_attention_kernel(backend_id: str) -> DiffusersAttentionKernel:
    normalized = str(backend_id or "").strip().lower()
    if normalized in {"sdpa"}:
        return SdpaDiffusersAttentionKernel()
    if normalized in {"torch", "native"}:
        return TorchDiffusersAttentionKernel()
    if normalized in {"xformers"}:
        return XformersDiffusersAttentionKernel.create()
    if normalized in {"flash", "flash2", "flashattn", "flashattention", "flashattention2", "fa2"}:
        return Flash2DiffusersAttentionKernel.create()
    if normalized in {"sage", "sageattn", "sageattention"}:
        return SageDiffusersAttentionKernel.create()
    if normalized in {"sparge", "spargeattn", "spargeattn2"}:
        return Sparge2DiffusersAttentionKernel.create()
    raise RuntimeError(f"Diffusers attention backend '{backend_id}' is not implemented yet")


def _is_standard_processor(processor: Any) -> bool:
    if isinstance(processor, GenericDiffusersAttnProcessor):
        return True
    module_name = processor.__class__.__module__
    return module_name.startswith("diffusers.models.attention_processor")


def install_diffusers_attention_processor(
    unet: Any,
    kernel: DiffusersAttentionKernel,
    *,
    allow_custom_processors: bool = False,
) -> int:
    try:
        from diffusers.models.attention_processor import Attention
    except Exception as exc:
        raise RuntimeError(f"diffusers Attention is unavailable: {exc}") from exc

    patched = 0
    for module in unet.modules():
        if not isinstance(module, Attention):
            continue
        current_processor = getattr(module, "processor", None)
        if (
            current_processor is not None
            and not allow_custom_processors
            and not _is_standard_processor(current_processor)
        ):
            raise RuntimeError(
                "cannot replace custom diffusers attention processor "
                f"{current_processor.__class__.__module__}.{current_processor.__class__.__name__}"
            )
        module._lulynx_original_attn_processor = current_processor
        module.set_processor(GenericDiffusersAttnProcessor(kernel))
        patched += 1

    if patched <= 0:
        raise RuntimeError("no diffusers Attention modules were found on the U-Net")
    return patched
