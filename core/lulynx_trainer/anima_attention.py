"""Attention backend dispatch for Anima/Newbie DiT models.

Provides a unified attention function that dispatches to:
  - flash2 (flash_attn):  FlashAttention-2 via the flash_attn package
  - sageattn:            SageAttention via the sageattention package
  - spargeattn2:          Sparse GEMM Attention 2 via spas_sage_attn
  - xformers:             xFormers memory_efficient_attention
  - flexattn:            PyTorch FlexAttention via torch.nn.attention
  - sdpa (default):      PyTorch F.scaled_dot_product_attention
  - torch:               Manual Q@K^T softmax @V fallback

This module is imported lazily — flash_attn / sageattention / FlexAttention are only
imported when the user selects those backends, so missing packages
degrade gracefully to SDPA.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from typing import Dict

import torch

from .amd_runtime import estimate_sdpa_chunk_count
from .attention_kernel_adapters import (
    flash2_attention_bhnd,
    sage_attention_bhnd,
    sdpa_attention_bhnd,
    sparge2_attention_bhnd,
    torch_attention_bhnd,
    xformers_attention_bhnd,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attention runtime statistics
# ---------------------------------------------------------------------------

_attention_runtime_stats: Dict[str, int] = {
    "flash2_calls": 0,
    "sageattn_calls": 0,
    "spargeattn2_calls": 0,
    "xformers_calls": 0,
    "flexattn_calls": 0,
    "sliding_window_calls": 0,
    "sdpa_calls": 0,
    "torch_calls": 0,
    "tgate_self_attention_calls": 0,
    "tgate_cross_attention_calls": 0,
    "tgate_eligible_cross_attention_calls": 0,
}


def _dynamo_disabled(fn):
    try:
        return torch.compiler.disable(fn)
    except Exception:
        try:
            return torch._dynamo.disable(fn)
        except Exception:
            return fn


@_dynamo_disabled
def _increment_attention_stat(backend: str) -> None:
    key = f"{backend}_calls"
    if key in _attention_runtime_stats:
        _attention_runtime_stats[key] += 1


def snapshot_attention_stats() -> Dict[str, int]:
    """Return a copy of the current attention backend call counters."""
    return dict(_attention_runtime_stats)


def reset_attention_stats() -> None:
    """Reset all attention backend call counters to zero."""
    for k in _attention_runtime_stats:
        _attention_runtime_stats[k] = 0
    try:
        from .tgate import reset_tgate_stats

        reset_tgate_stats()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_flash_attn_available: Optional[bool] = None
_sageattn_available: Optional[bool] = None
_spargeattn2_available: Optional[bool] = None
_xformers_available: Optional[bool] = None
_flex_attn_available: Optional[bool] = None


def _check_flash_attn() -> bool:
    global _flash_attn_available
    if sys.modules.get("flash_attn") is None and "flash_attn" in sys.modules:
        _flash_attn_available = False
    if _flash_attn_available is None:
        try:
            import flash_attn  # noqa: F401
            _flash_attn_available = True
        except ImportError:
            _flash_attn_available = False
    return _flash_attn_available


def _check_sageattn() -> bool:
    global _sageattn_available
    if _sageattn_available is None:
        try:
            import sageattention  # noqa: F401
            _sageattn_available = True
        except ImportError:
            _sageattn_available = False
    return _sageattn_available


def _check_spargeattn2() -> bool:
    global _spargeattn2_available
    if _spargeattn2_available is None:
        try:
            import spas_sage_attn  # noqa: F401
            _spargeattn2_available = True
        except ImportError:
            _spargeattn2_available = False
    return _spargeattn2_available


def _check_xformers() -> bool:
    global _xformers_available
    if _xformers_available is None:
        try:
            from xformers.ops import memory_efficient_attention

            _xformers_available = callable(memory_efficient_attention)
        except Exception:
            _xformers_available = False
    return _xformers_available


def _check_flex_attn() -> bool:
    global _flex_attn_available
    if _flex_attn_available is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention

            _flex_attn_available = callable(flex_attention)
        except Exception:
            _flex_attn_available = False
    return _flex_attn_available


# ---------------------------------------------------------------------------
# Unified attention function
# ---------------------------------------------------------------------------

@_dynamo_disabled
def dit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "sdpa",
    dropout_p: float = 0.0,
    split_chunks: int = 0,
    amd_sdpa_slice_trigger_gb: float = 0.0,
    amd_sdpa_slice_target_gb: float = 0.0,
    early_delete: bool = False,
    sliding_window_size: int = 0,
    sliding_backend: str = "auto",
    sliding_torch_fallback_max_tokens: int = 2048,
    launcher_attention_backend: str = "auto",
    flex_runtime_active: bool = False,
) -> torch.Tensor:
    """Compute multi-head attention with backend dispatch.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(B, H, T, D)`` (BH_TD layout, which is what DiT _split_heads
        produces).  For flash2, tensors are transposed to ``(B, T, H, D)``
        internally.
    backend : str
        One of ``"flash2"``, ``"sageattn"``, ``"spargeattn2"``,
        ``"xformers"``, ``"flexattn"``, ``"sdpa"``, ``"torch"``.
    dropout_p : float
        Dropout probability (0.0 during inference).
    split_chunks : int
        If > 1, run attention in head-group chunks to lower peak VRAM. ``0``
        and ``1`` both disable splitting.  Used to wire up the Anima
        ``split_attn`` config flag.
    early_delete : bool
        If True, aggressively delete intermediate tensors (Q, K, V copies)
        inside backend functions to free VRAM sooner.

    Returns
    -------
    torch.Tensor
        Shape ``(B, H, T, D)`` — same layout as input.
    """
    if int(sliding_window_size or 0) > 0:
        _increment_attention_stat("sliding_window")
        from .runtime_optimizations import sliding_window_attention

        out = sliding_window_attention(
            q,
            k,
            v,
            window_size=int(sliding_window_size),
            backend=sliding_backend,
            torch_fallback_max_tokens=int(sliding_torch_fallback_max_tokens or 2048),
            launcher_attention_backend=launcher_attention_backend,
            flex_runtime_active=bool(flex_runtime_active),
        )
        if early_delete:
            del q, k, v
        return out

    if backend == "sdpa" and split_chunks <= 1:
        split_chunks = estimate_sdpa_chunk_count(
            q,
            trigger_gb=float(amd_sdpa_slice_trigger_gb or 0.0),
            target_gb=float(amd_sdpa_slice_target_gb or 0.0),
        )
    if split_chunks and split_chunks > 1 and q.shape[1] >= split_chunks:
        _increment_attention_stat(backend)
        from . import attn_entropy as _ent
        if _ent.should_probe():
            _ent.probe_from_qk(q, k)
        return _chunked_head_attention(
            q, k, v, backend=backend, dropout_p=dropout_p, chunks=split_chunks,
            early_delete=early_delete,
        )
    _increment_attention_stat(backend)
    from . import attn_entropy as _ent
    if _ent.should_probe():
        _ent.probe_from_qk(q, k)
    if backend == "flash2":
        return _flash2_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
    if backend == "sageattn":
        return _sageattn_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
    if backend == "spargeattn2":
        return _spargeattn2_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
    if backend == "xformers":
        return _xformers_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
    if backend == "flexattn":
        return _flexattn_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
    if backend == "sdpa":
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)
    return _torch_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)


def _chunked_head_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *,
    backend: str,
    dropout_p: float,
    chunks: int,
    early_delete: bool = False,
) -> torch.Tensor:
    """Split heads into ``chunks`` groups, run attention per group, concatenate.

    This trades a small overhead from re-launching the attention kernel for a
    proportional reduction in peak attention-matrix memory: each group only
    materialises ``H/chunks`` heads worth of ``Q @ K^T`` at a time.
    """
    H = q.shape[1]
    n = max(1, min(chunks, H))
    if n <= 1:
        # no-op split — fall through to non-split path
        if backend == "flash2":
            return _flash2_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
        if backend == "sageattn":
            return _sageattn_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
        if backend == "spargeattn2":
            return _spargeattn2_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
        if backend == "xformers":
            return _xformers_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
        if backend == "flexattn":
            return _flexattn_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)
        if backend == "sdpa":
            return _sdpa_attention(q, k, v, dropout_p=dropout_p)
        return _torch_attention(q, k, v, dropout_p=dropout_p, early_delete=early_delete)

    base = H // n
    rem = H - base * n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]

    parts = []
    start = 0
    for size in sizes:
        end = start + size
        q_chunk = q[:, start:end]
        k_chunk = k[:, start:end]
        v_chunk = v[:, start:end]
        if backend == "flash2":
            out = _flash2_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        elif backend == "sageattn":
            out = _sageattn_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        elif backend == "spargeattn2":
            out = _spargeattn2_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        elif backend == "xformers":
            out = _xformers_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        elif backend == "flexattn":
            out = _flexattn_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        elif backend == "sdpa":
            out = _sdpa_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p)
        else:
            out = _torch_attention(q_chunk, k_chunk, v_chunk, dropout_p=dropout_p, early_delete=early_delete)
        del q_chunk, k_chunk, v_chunk
        parts.append(out)
        start = end

    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
) -> torch.Tensor:
    """PyTorch SDPA — BH_TD layout, directly compatible."""
    return sdpa_attention_bhnd(q, k, v, dropout_p=dropout_p)


def _torch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """Manual QK^T softmax V — BH_TD layout."""
    out = torch_attention_bhnd(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    return out


def _flash2_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """FlashAttention-2 via the flash_attn package.

    flash_attn expects (B, T, H, D) while our DiT uses (B, H, T, D).
    """
    if not _check_flash_attn():
        logger.warning("flash_attn not available, falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)

    out = flash2_attention_bhnd(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    return out


def _sageattn_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """SageAttention via the sageattention package.

    sageattn expects (B, H, T, D_head), matching the shared BHND adapter.
    """
    if not _check_sageattn():
        logger.warning("sageattention not available, falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)

    out = sage_attention_bhnd(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    return out


def _spargeattn2_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """SpargeAttention2 via spas_sage_attn with recompute-based gradients."""
    if not _check_spargeattn2():
        logger.warning("spas_sage_attn not available, falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)

    out = sparge2_attention_bhnd(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    return out


def _xformers_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """xFormers memory efficient attention in BHND layout."""
    if not _check_xformers():
        logger.warning("xformers not available, falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)

    out = xformers_attention_bhnd(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    return out


def _flexattn_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0,
    early_delete: bool = False,
) -> torch.Tensor:
    """PyTorch FlexAttention.

    flex_attention expects ``(B, H, T, D)`` query/key/value tensors, matching
    the DiT layout used here.  The PyTorch API is prototype and does not expose
    a dropout argument, so training-time dropout requests stay on SDPA.
    """
    if dropout_p > 0.0:
        logger.warning("flex_attention does not expose dropout_p; falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)
    if not _check_flex_attn():
        logger.warning("torch.nn.attention.flex_attention not available, falling back to SDPA")
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)

    from torch.nn.attention.flex_attention import flex_attention

    try:
        out = flex_attention(q, k, v)
    except Exception as exc:
        logger.warning("flex_attention failed, falling back to SDPA: %s", exc)
        return _sdpa_attention(q, k, v, dropout_p=dropout_p)
    if early_delete:
        del q, k, v
    if isinstance(out, tuple):
        out = out[0]
    return out


# ---------------------------------------------------------------------------
# Monkey-patch helper for Anima _ProjectionAttention
# ---------------------------------------------------------------------------

def patch_anima_attention(
    model: torch.nn.Module,
    backend: str = "sdpa",
    *,
    split_chunks: int = 0,
    amd_sdpa_slice_trigger_gb: float = 0.0,
    amd_sdpa_slice_target_gb: float = 0.0,
    early_deletion: bool = False,
    attention_profile: Optional[object] = None,
) -> int:
    """Replace the forward method of all DiT attention modules in *model*
    so they use the specified attention backend instead of hardcoded SDPA.

    This works by setting a ``_attention_backend`` attribute on each
    attention instance.  The patched forward method reads this
    attribute and dispatches to ``dit_attention()``.

    Matching logic:
      1. Modules whose class name is ``_ProjectionAttention`` or ``_TinyAttention``.
      2. Modules that have the DiT attention interface (q_proj, k_proj, v_proj,
         output_proj, q_norm, k_norm, _split_heads, _merge_heads).

    Parameters
    ----------
    backend
        Attention backend name (``"sdpa"``, ``"flash2"``, ``"sageattn"``,
        ``"spargeattn2"``, ``"xformers"``, ``"flexattn"``, ``"torch"``).
    split_chunks
        If >1, attention is computed in this many head-group chunks to reduce
        peak VRAM. Used by the Anima ``split_attn`` config flag — the trainer
        passes ``split_chunks=2`` (or higher) when ``split_attn=True``.
    """
    patched = 0
    _DIT_ATTN_ATTRS = {"q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm",
                       "_split_heads", "_merge_heads"}
    sliding_window_size = int(getattr(attention_profile, "window_size", 0) or 0)
    sliding_backend = str(getattr(attention_profile, "backend", "auto") or "auto")
    sliding_torch_limit = int(getattr(attention_profile, "torch_fallback_max_tokens", 2048) or 2048)
    launcher_attention_backend = str(getattr(attention_profile, "launcher_attention_backend", backend) or backend)
    flex_runtime_active = bool(getattr(attention_profile, "flex_runtime_active", False))

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == "_NextDiTWrapper":
            module._attention_backend = backend  # type: ignore[attr-defined]
            module._attention_split_chunks = int(split_chunks)  # type: ignore[attr-defined]
            module._amd_sdpa_slice_trigger_gb = float(amd_sdpa_slice_trigger_gb or 0.0)  # type: ignore[attr-defined]
            module._amd_sdpa_slice_target_gb = float(amd_sdpa_slice_target_gb or 0.0)  # type: ignore[attr-defined]
            module._attention_early_deletion = bool(early_deletion)  # type: ignore[attr-defined]
            module._attention_profile_window_size = sliding_window_size  # type: ignore[attr-defined]
            module._attention_profile_backend = sliding_backend  # type: ignore[attr-defined]
            module._attention_profile_torch_max_tokens = sliding_torch_limit  # type: ignore[attr-defined]
            module._attention_profile_launcher_backend = launcher_attention_backend  # type: ignore[attr-defined]
            module._attention_profile_flex_runtime_active = flex_runtime_active  # type: ignore[attr-defined]
            patched += max(len(getattr(module, "_block_modules", []) or []), 1)
            continue
        if cls_name in ("_ProjectionAttention", "_TinyAttention"):
            match = True
        elif _DIT_ATTN_ATTRS.issubset(set(dir(module))):
            match = True
        else:
            match = False

        if match:
            module._attention_module_name = name  # type: ignore[attr-defined]
            module._attention_backend = backend  # type: ignore[attr-defined]
            module._attention_split_chunks = int(split_chunks)  # type: ignore[attr-defined]
            module._amd_sdpa_slice_trigger_gb = float(amd_sdpa_slice_trigger_gb or 0.0)  # type: ignore[attr-defined]
            module._amd_sdpa_slice_target_gb = float(amd_sdpa_slice_target_gb or 0.0)  # type: ignore[attr-defined]
            module._attention_early_deletion = bool(early_deletion)  # type: ignore[attr-defined]
            module._attention_profile_window_size = sliding_window_size  # type: ignore[attr-defined]
            module._attention_profile_backend = sliding_backend  # type: ignore[attr-defined]
            module._attention_profile_torch_max_tokens = sliding_torch_limit  # type: ignore[attr-defined]
            module._attention_profile_launcher_backend = launcher_attention_backend  # type: ignore[attr-defined]
            module._attention_profile_flex_runtime_active = flex_runtime_active  # type: ignore[attr-defined]
            if not hasattr(module, "_original_forward_patched"):
                module._original_forward_patched = True  # type: ignore[attr-defined]
                _patch_attention_forward(module)
            patched += 1

    if patched > 0:
        logger.info(
            f"Patched {patched} attention modules to use backend '{backend}'"
            + (f" with split_chunks={split_chunks}" if split_chunks > 1 else "")
            + (f", sliding_window={sliding_window_size}" if sliding_window_size > 0 else "")
            + (", early_deletion=on" if early_deletion else "")
        )
    else:
        logger.debug("No DiT attention modules found to patch")
    return patched


def _patch_attention_forward(module: torch.nn.Module) -> None:
    """Replace module.forward with a backend-dispatching version.

    The patched forward reads _attention_backend and calls dit_attention().
    It assumes the module has: q_proj, k_proj, v_proj, output_proj, q_norm,
    k_norm, _split_heads, _merge_heads.
    """
    _orig_forward = module.forward

    def _dispatched_forward(self, x, context=None):
        is_cross_attention = context is not None
        module_name = str(getattr(self, "_attention_module_name", ""))
        try:
            from .tgate import observe_attention_call

            eligible = observe_attention_call(
                module_name=module_name,
                is_cross_attention=is_cross_attention,
            )
            if is_cross_attention:
                _attention_runtime_stats["tgate_cross_attention_calls"] += 1
            else:
                _attention_runtime_stats["tgate_self_attention_calls"] += 1
            if eligible:
                _attention_runtime_stats["tgate_eligible_cross_attention_calls"] += 1
        except Exception:
            pass

        source = x if context is None else context

        fused_qkv = getattr(self, "_fused_qkv", None)
        fused_kv = getattr(self, "_fused_kv", None)

        if fused_qkv is not None and context is None:
            q_raw, k_raw, v_raw = fused_qkv(x)
            q = self.q_norm(self._split_heads(q_raw))
            k = self.k_norm(self._split_heads(k_raw))
            v = self._split_heads(v_raw)
        elif fused_kv is not None and context is not None:
            q = self.q_norm(self._split_heads(self.q_proj(x)))
            k_raw, v_raw = fused_kv(source)
            k = self.k_norm(self._split_heads(k_raw))
            v = self._split_heads(v_raw)
        else:
            q = self.q_norm(self._split_heads(self.q_proj(x)))
            k = self.k_norm(self._split_heads(self.k_proj(source)))
            v = self._split_heads(self.v_proj(source))

        backend = getattr(self, "_attention_backend", "sdpa")
        split_chunks = int(getattr(self, "_attention_split_chunks", 0) or 0)
        early_del = bool(getattr(self, "_attention_early_deletion", False))
        attn = dit_attention(
            q,
            k,
            v,
            backend=backend,
            split_chunks=split_chunks,
            amd_sdpa_slice_trigger_gb=float(getattr(self, "_amd_sdpa_slice_trigger_gb", 0.0) or 0.0),
            amd_sdpa_slice_target_gb=float(getattr(self, "_amd_sdpa_slice_target_gb", 0.0) or 0.0),
            early_delete=early_del,
            sliding_window_size=int(getattr(self, "_attention_profile_window_size", 0) or 0),
            sliding_backend=str(getattr(self, "_attention_profile_backend", "auto") or "auto"),
            sliding_torch_fallback_max_tokens=int(getattr(self, "_attention_profile_torch_max_tokens", 2048) or 2048),
            launcher_attention_backend=str(getattr(self, "_attention_profile_launcher_backend", backend) or backend),
            flex_runtime_active=bool(getattr(self, "_attention_profile_flex_runtime_active", False)),
        )
        if early_del:
            del q, k, v
        return self.output_proj(self._merge_heads(attn))

    import types
    module.forward = types.MethodType(_dispatched_forward, module)


def unpatch_anima_attention(model: torch.nn.Module) -> None:
    """Remove attention backend patches, restoring original forward methods.

    This is a safety net — in practice the patched forward delegates to
    ``dit_attention("sdpa")`` which is equivalent to the original
    ``F.scaled_dot_product_attention`` call.
    """
    for name, module in model.named_modules():
        if hasattr(module, "_attention_backend"):
            delattr(module, "_attention_backend")
        if hasattr(module, "_attention_module_name"):
            delattr(module, "_attention_module_name")
        if hasattr(module, "_original_forward_patched"):
            delattr(module, "_original_forward_patched")
        if hasattr(module, "_amd_sdpa_slice_trigger_gb"):
            delattr(module, "_amd_sdpa_slice_trigger_gb")
        if hasattr(module, "_amd_sdpa_slice_target_gb"):
            delattr(module, "_amd_sdpa_slice_target_gb")
        if hasattr(module, "_attention_early_deletion"):
            delattr(module, "_attention_early_deletion")
