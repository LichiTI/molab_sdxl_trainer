# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Faithful-forward cross-attention context resolver for Anima training (#147).

The legacy (#132) training path feeds the *raw* Qwen3 hidden states straight
into the native DiT as cross-attention context, bypassing the ``llm_adapter``.
That is one of the two bugs that make Anima "fail to converge" — the velocity
field the model is asked to match is conditioned on the wrong context.

The faithful path (opt-in, ``anima_faithful_forward=True``) instead runs the
*frozen* ``llm_adapter`` once per prompt to map (Qwen3 hidden + T5 token ids)
into the cross-attention context the native DiT was actually trained against.
The adapter is run under ``no_grad`` — the text side never trains (Anima
guidance); gradients flow only into the LoRA on the DiT blocks.

This module is a thin, cleanroom resolver: it locates the adapter inputs in the
conditioning / batch dicts, runs the adapter, and returns the context tensor.
The forward-input handler calls it and overrides ``encoder_hidden_states``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


class AnimaFaithfulDecision(NamedTuple):
    """Outcome of resolving the EFFECTIVE faithful-forward flag at prepare time.

    * ``effective`` — whether the run actually takes the faithful path.
    * ``degrade_reason`` — a stable enum-like tag when faithful was *requested*
      but had to be turned off (``compute_reducer_conflict`` / ``cache_missing_t5``);
      ``None`` when faithful was not requested, or when it stays on.
    * ``degrade_detail`` — operator-facing prose explaining the degrade; ``None``
      unless ``degrade_reason`` is set.

    ``degrade_reason is not None`` is the signal to warn loudly + emit a runtime
    event (a degrade is NOT an error — the run continues as the #132 baseline).
    """

    effective: bool
    degrade_reason: Optional[str]
    degrade_detail: Optional[str]


def resolve_anima_faithful_decision(
    *,
    faithful_requested: bool,
    reducer_strategy: Any,
    has_t5_checker: Callable[[], bool],
) -> AnimaFaithfulDecision:
    """Decide whether the faithful native forward can run, gracefully degrading.

    Pure policy (no I/O, no side effects) so it is CPU-testable in isolation.
    The faithful forward (3D RoPE + frozen llm_adapter context + ``t in [0,1]``)
    is mutually exclusive with two things, checked in short-circuit order:

      1. **token-count-changing compute-reducer seams** (TREAD / DiffCR) — they
         route / merge tokens, so the per-position 3D ``rope_emb`` would no longer
         line up with the surviving tokens. They are an explicit training feature,
         so the explicit choice wins: degrade faithful off, keep the reducer.
         **BlockSkip is exempt**: it skips whole blocks via token-preserving
         identity passthrough, so ``rope_emb`` threads cleanly and it now runs
         *under* faithful (and coexists with block-checkpointing) — see
         ``anima_native_dit._run_blocks``.
      2. **a cache built without the T5 tokenizer** (no ``t5_input_ids``) — the frozen
         llm_adapter needs T5 ids, so degrade to the legacy path.

    ``has_t5_checker`` is a thunk (it touches disk) and is only invoked once the
    cheaper reducer check has passed — so a reducer conflict short-circuits before
    any cache I/O. Returns :class:`AnimaFaithfulDecision`; the caller applies the
    config mutation / logging / runtime event.
    """
    if not faithful_requested:
        return AnimaFaithfulDecision(False, None, None)

    reducer = str(reducer_strategy or "none").strip().lower()
    # BlockSkip preserves token count (identity skip), so it threads rope_emb and
    # stays faithful-compatible; only the token-routing reducers conflict.
    if reducer not in ("", "none", "blockskip"):
        return AnimaFaithfulDecision(
            False,
            "compute_reducer_conflict",
            f"the compute reducer dit_compute_reducer_strategy={reducer!r} "
            "(TREAD/DiffCR) routes/merges tokens and is incompatible with "
            "faithful 3D RoPE (the explicit reducer wins)",
        )

    if not has_t5_checker():
        return AnimaFaithfulDecision(
            False,
            "cache_missing_t5",
            "the cached training data carries no t5_input_ids (cache built without "
            "the T5 tokenizer) and the frozen llm_adapter needs T5 ids -- rebuild the "
            "Anima cache with the T5 tokenizer to use faithful",
        )

    return AnimaFaithfulDecision(True, None, None)


def anima_text_cache_needs_padding_rebuild(
    *,
    qwen3_seq: Optional[int],
    t5_seq: Optional[int],
    expected_qwen3: int = 512,
    expected_t5: int = 512,
) -> bool:
    """Decide whether a cached text-conditioning sample is too short for faithful.

    Pure policy (no I/O) so it is CPU-testable. The faithful native forward
    conditions on the pretrained model's *max-padded* cross-attention context:
    per the Anima reference invariant the text encoders must be padded to
    ``MAX_CROSSATTN_TOKENS = 512`` at **every** stage (training and inference),
    because zero-padded positions act as attention sinks in the cross-attn
    softmax — trimming to the real text length shifts the conditioning manifold
    and renders flat/gray. Inference always pads to 512; a faithful LoRA trained
    on an un-padded cache (e.g. seq 66/74 from an older cache builder) learns the
    wrong manifold and the two no longer match.

    Returns ``True`` when a *known* sequence length is shorter than its expected
    pad length (→ the caller should force-rebuild the cache at full padding).
    Fail-open: when a stream's length is unknown (``None`` — nothing was peeked,
    or the key is absent and handled by the separate ``cache_missing_t5`` degrade)
    it does not by itself trigger a rebuild.
    """
    if qwen3_seq is None and t5_seq is None:
        return False
    if qwen3_seq is not None and int(qwen3_seq) < int(expected_qwen3):
        return True
    if t5_seq is not None and int(t5_seq) < int(expected_t5):
        return True
    return False


# One-shot, process-local marker so the production worker log carries a single
# decisive line proving fix③ actually fired on the hot path (vs silently
# degrading to the raw-Qwen3 context when the adapter is unreachable).
_FIRST_CONTEXT_LOGGED = False


def resolve_anima_faithful_context(
    prompt_embeds: Mapping[str, Any],
    batch: Mapping[str, Any],
    adapter: Any,
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """Return the frozen-``llm_adapter`` cross-attention context for one batch.

    Inputs (already produced by the conditioning/collate stages):
      * source hidden = raw Qwen3 hidden. In the native qwen3-primary cache it
        is ``prompt_embeds["encoder_hidden_states"]``; when Qwen3 is secondary
        it is ``prompt_embeds["qwen3_hidden_states"]``.
      * ``batch["t5_input_ids"]`` = T5 token ids (the adapter's target stream).

    Fails loud when ``t5_input_ids`` is missing — that means the cache was built
    without the T5 tokenizer and must be rebuilt for the faithful path, rather
    than silently degrading to the (wrong) raw-Qwen3 context.
    """
    source = prompt_embeds.get("qwen3_hidden_states")
    if source is None:
        source = prompt_embeds.get("encoder_hidden_states")
    if source is None:
        raise ValueError(
            "anima_faithful_forward=True needs Qwen3 hidden states as the llm_adapter "
            "source, but neither qwen3_hidden_states nor encoder_hidden_states is present."
        )

    t5_ids = batch.get("t5_input_ids")
    if t5_ids is None:
        raise ValueError(
            "anima_faithful_forward=True requires t5_input_ids in the cached batch, "
            "but it is missing — rebuild the Anima cache with the T5 tokenizer enabled "
            "(native caches carry t5_input_ids by default; old/clip-primary caches do not)."
        )

    source_mask = prompt_embeds.get("qwen3_attention_mask")
    if source_mask is None:
        source_mask = prompt_embeds.get("attention_mask")
    target_mask = batch.get("t5_attention_mask")

    source = source.to(device=device, dtype=dtype)
    t5_ids = t5_ids.to(device=device).long()
    if source_mask is not None:
        source_mask = source_mask.to(device=device)
    if target_mask is not None:
        target_mask = target_mask.to(device=device)

    with torch.no_grad():
        context = adapter(
            source_hidden_states=source,
            target_input_ids=t5_ids,
            target_attention_mask=target_mask,
            source_attention_mask=source_mask,
        )
    context = context.to(device=device, dtype=dtype)

    global _FIRST_CONTEXT_LOGGED
    if not _FIRST_CONTEXT_LOGGED:
        _FIRST_CONTEXT_LOGGED = True
        logger.info(
            "[anima-faithful] #147 fix3 ACTIVE on training hot path: cross-attn context "
            "resolved via FROZEN llm_adapter under no_grad (source%s + t5_ids%s -> "
            "context%s); raw-Qwen3 context bypassed.",
            tuple(source.shape), tuple(t5_ids.shape), tuple(context.shape),
        )
    return context


__all__ = [
    "resolve_anima_faithful_context",
    "resolve_anima_faithful_decision",
    "anima_text_cache_needs_padding_rebuild",
    "AnimaFaithfulDecision",
]
