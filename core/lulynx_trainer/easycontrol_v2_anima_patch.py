# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""EasyControl v2 two-stream patch for the real Anima executable-subset DiT.

The v2 attention math, the synthetic-block patch and the Anima *tiny* trainable
patch already live in ``easycontrol_v2_adapter``.  The remaining gap this module
closes is the **real executable-subset** wiring: patching the genuine
``AnimaNativeExecutableSubset`` ``_Block.forward`` so the target stream's
self-attention can attend to the condition stream (EasyControl v2's "two-stream"
attention), while keeping the cross-attention and MLP paths byte-for-byte the
original.

Design red-lines (mirroring every other Lulynx default-off reserve)
------------------------------------------------------------------
* **No-condition bitwise parity.** When the adapter has no active condition
  tokens (``adapter.current_cond_tokens is None``) the patched ``forward``
  delegates to the captured original ``forward`` unchanged, so an installed but
  un-conditioned patch is bitwise identical to today.
* **Faithful target math.** The patched path reuses the block's own
  ``_apply_adaln`` / ``_with_adaln_lora`` / ``adaln_modulation_*`` and the
  attention module's ``q_norm`` / ``k_norm`` / ``_split_heads`` / ``_merge_heads``
  so the target stream is computed by exactly the same operations as the
  original; the only addition is the extra condition keys/values (gated by the
  learned ``b_cond`` bias) in the self-attention.
* **Zero-init near-identity.** With the adapter's default zero-init LoRA deltas
  and very negative ``b_cond``, the condition columns carry ~``exp(b_cond)``
  attention mass, so a freshly-built adapter perturbs the target output only
  negligibly — the standard "start from identity" behaviour for trainable
  adapters.  Real-model training quality remains the operator's job.

Clean-room Lulynx module; references no external EasyControl source.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import
    from .easycontrol_v2_adapter import (
        EasyControlV2Adapter,
        easycontrol_v2_extended_attention,
    )
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.easycontrol_v2_adapter import (
        EasyControlV2Adapter,
        easycontrol_v2_extended_attention,
    )


def easycontrol_v2_anima_executable_subset_patch_supported() -> bool:
    """Whether the real executable-subset two-stream patch is implemented.

    This module *is* that implementation, so it always returns True.  The
    adapter-side readiness report imports this lazily to flip ``patch_supported``
    without a circular import.
    """
    return True


def easycontrol_v2_subset_two_stream_self_attention(
    attn: nn.Module,
    target_norm: torch.Tensor,
    cond_norm: Optional[torch.Tensor],
    block_deltas: Optional[Mapping[str, torch.Tensor]],
    b_cond: torch.Tensor | float,
    rope_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Two-stream self-attention for a real Anima ``_ProjectionAttention``.

    ``attn`` must expose ``q_proj``/``k_proj``/``v_proj``/``output_proj`` plus the
    per-head ``q_norm``/``k_norm`` and the ``_split_heads``/``_merge_heads``
    helpers (the executable-subset attention contract).  ``target_norm`` /
    ``cond_norm`` are already AdaLN-normalised ``[B, S, D]`` tensors.

    Returns ``(target_out, cond_out)``.  When ``cond_norm is None`` this is
    exactly the original ``attn.forward(target_norm)`` and ``cond_out is None``.

    ``rope_emb`` (faithful mode) is applied to the **target** stream's q/k via
    the same ``AnimaRope3D.apply`` the original self-attention uses, so the target
    math stays bit-faithful. The **condition** stream carries no rope: it is a
    separate reference sequence with no anima positional grid, matching the
    original (which only applies rope to the target's self-attention q/k).
    """
    target_q = attn.q_norm(attn._split_heads(attn.q_proj(target_norm)))
    target_k = attn.k_norm(attn._split_heads(attn.k_proj(target_norm)))
    target_v = attn._split_heads(attn.v_proj(target_norm))
    if rope_emb is not None:
        _apply_rope = _resolve_apply_rope()
        if _apply_rope is not None:
            target_q = _apply_rope(target_q, rope_emb)
            target_k = _apply_rope(target_k, rope_emb)

    if cond_norm is None:
        attn_out = F.scaled_dot_product_attention(target_q, target_k, target_v, dropout_p=0.0)
        return attn.output_proj(attn._merge_heads(attn_out)), None

    cond_q_lin = attn.q_proj(cond_norm)
    cond_k_lin = attn.k_proj(cond_norm)
    cond_v_lin = attn.v_proj(cond_norm)
    if block_deltas is not None:
        qkv_delta = block_deltas.get("qkv_delta")
        if qkv_delta is not None:
            dq, dk, dv = qkv_delta.chunk(3, dim=-1)
            cond_q_lin = cond_q_lin + dq
            cond_k_lin = cond_k_lin + dk
            cond_v_lin = cond_v_lin + dv
    cond_q = attn.q_norm(attn._split_heads(cond_q_lin))
    cond_k = attn.k_norm(attn._split_heads(cond_k_lin))
    cond_v = attn._split_heads(cond_v_lin)

    # Target queries see target keys plus the condition keys (b_cond-gated).
    target_out = easycontrol_v2_extended_attention(
        target_q, target_k, target_v, cond_k, cond_v, b_cond
    )
    target_out = attn.output_proj(attn._merge_heads(target_out))

    # Condition stream self-attention so the condition can evolve across blocks.
    cond_attn = F.scaled_dot_product_attention(cond_q, cond_k, cond_v, dropout_p=0.0)
    cond_out = attn.output_proj(attn._merge_heads(cond_attn))
    if block_deltas is not None:
        out_delta = block_deltas.get("out_delta")
        if out_delta is not None:
            cond_out = cond_out + out_delta
    return target_out, cond_out


def _is_executable_subset_block(block: nn.Module) -> bool:
    if not all(
        hasattr(block, attr)
        for attr in ("self_attn", "cross_attn", "mlp", "_apply_adaln", "_with_adaln_lora")
    ):
        return False
    if not all(
        hasattr(block, f"adaln_modulation_{branch}")
        for branch in ("self_attn", "cross_attn", "mlp")
    ):
        return False
    self_attn = getattr(block, "self_attn", None)
    return all(
        hasattr(self_attn, attr)
        for attr in ("q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm")
    )


def _resolve_apply_rope():
    """Return ``AnimaRope3D.apply`` (faithful 3D RoPE) or None if unavailable.

    Mirrors the import the executable-subset attention uses so the two-stream
    target stream is rotated by the exact same operator.
    """
    try:
        from .anima_native_faithful import AnimaRope3D
    except ImportError:  # pragma: no cover - direct-file usage
        try:
            from core.lulynx_trainer.anima_native_faithful import AnimaRope3D
        except Exception:
            return None
    except Exception:
        return None
    return AnimaRope3D.apply


def _make_patched_forward(block: nn.Module, adapter: EasyControlV2Adapter, original_forward: Any, block_index: int):
    def patched_forward(x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor,
                        adaln_lora: Optional[torch.Tensor] = None,
                        rope_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond_tokens = adapter.current_cond_tokens
        if cond_tokens is None:
            # Default-off red-line: identical to the original block forward
            # (faithful rope_emb is threaded through unchanged).
            return original_forward(x, emb, context, adaln_lora, rope_emb)

        deltas = adapter.block_deltas(block_index, cond_tokens)
        b_cond = deltas.get("b_cond", -10.0) if deltas is not None else -10.0
        emb_use = emb[:, 0] if emb.dim() == 3 else emb

        # --- self-attention: two-stream (target + condition) ---
        shift, scale, gate = block._with_adaln_lora(block.adaln_modulation_self_attn(emb_use), adaln_lora)
        target_norm = block._apply_adaln(x, shift, scale)
        cond_norm = block._apply_adaln(cond_tokens, shift, scale)
        self_out, cond_self_out = easycontrol_v2_subset_two_stream_self_attention(
            block.self_attn, target_norm, cond_norm, deltas, b_cond, rope_emb
        )
        x = x + gate.unsqueeze(1) * self_out
        if cond_self_out is not None:
            cond_tokens = cond_tokens + gate.unsqueeze(1) * cond_self_out

        # --- cross-attention: target only, identical to the original path ---
        shift, scale, gate = block._with_adaln_lora(block.adaln_modulation_cross_attn(emb_use), adaln_lora)
        x = x + gate.unsqueeze(1) * block.cross_attn(block._apply_adaln(x, shift, scale), context)

        # --- MLP: target only, identical to the original path ---
        shift, scale, gate = block._with_adaln_lora(block.adaln_modulation_mlp(emb_use), adaln_lora)
        x = x + gate.unsqueeze(1) * block.mlp(block._apply_adaln(x, shift, scale))

        # Publish the evolved condition stream for the next patched block.
        adapter._cond_tokens = cond_tokens
        return x

    return patched_forward


class EasyControlV2AnimaSubsetPatchHandle:
    """Restores every executable-subset block patched by the v2 installer."""

    def __init__(self, patched: List[Tuple[nn.Module, Any]]) -> None:
        self._patched = patched
        self.active = True

    @property
    def block_count(self) -> int:
        return len(self._patched)

    def remove(self) -> None:
        if not self.active:
            return
        for block, original_forward in self._patched:
            block.forward = original_forward
        self.active = False


def install_easycontrol_v2_anima_executable_subset_patch(
    module_or_block: nn.Module,
    adapter: EasyControlV2Adapter,
    *,
    block_indices: Optional[Tuple[int, ...]] = None,
) -> EasyControlV2AnimaSubsetPatchHandle:
    """Patch the real Anima executable-subset blocks for EasyControl v2.

    ``module_or_block`` may be an ``AnimaNativeExecutableSubset`` (its
    ``net.blocks`` are patched) or a single executable-subset ``_Block``.  Each
    patched block's ``forward`` runs the two-stream self-attention when the
    adapter has active condition tokens, and is bitwise-identical to the original
    otherwise.  Returns a handle whose ``remove()`` restores every block.

    ``block_indices`` optionally restricts which blocks are patched (and selects
    the adapter block used for each); by default all discovered blocks are
    patched 1:1 with adapter blocks ``0..N-1``.
    """
    if not isinstance(adapter, EasyControlV2Adapter):
        raise TypeError("adapter must be an EasyControlV2Adapter")

    if bool(getattr(module_or_block, "is_anima_executable_subset", False)):
        blocks = list(getattr(getattr(module_or_block, "net", None), "blocks", []) or [])
    else:
        blocks = [module_or_block]
    if not blocks:
        raise ValueError("no executable-subset blocks found to patch")

    selected: List[Tuple[int, nn.Module]]
    if block_indices is None:
        selected = list(enumerate(blocks))
    else:
        selected = []
        for adapter_index, block_index in enumerate(block_indices):
            if block_index < 0 or block_index >= len(blocks):
                raise IndexError(f"block index {block_index} out of range for {len(blocks)} blocks")
            selected.append((adapter_index, blocks[block_index]))

    hidden = adapter.config.hidden_size
    patched: List[Tuple[nn.Module, Any]] = []
    for adapter_index, block in selected:
        if not _is_executable_subset_block(block):
            raise ValueError("target is not an Anima executable-subset block (missing required submodules)")
        attn_hidden = int(getattr(block.self_attn, "hidden_dim", hidden))
        if attn_hidden != hidden:
            raise ValueError(
                f"adapter hidden_size {hidden} != block self_attn hidden_dim {attn_hidden}"
            )
        if adapter_index >= len(adapter.blocks):
            raise IndexError(
                f"adapter has {len(adapter.blocks)} blocks; cannot patch block #{adapter_index}"
            )
        original_forward = block.forward
        block.forward = _make_patched_forward(block, adapter, original_forward, adapter_index)
        patched.append((block, original_forward))

    return EasyControlV2AnimaSubsetPatchHandle(patched)


__all__ = [
    "easycontrol_v2_anima_executable_subset_patch_supported",
    "easycontrol_v2_subset_two_stream_self_attention",
    "install_easycontrol_v2_anima_executable_subset_patch",
    "EasyControlV2AnimaSubsetPatchHandle",
]
