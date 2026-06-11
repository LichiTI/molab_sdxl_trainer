# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Embedding Inversion (Textual Inversion) — Phase 8.8 / #116.

Train a single token embedding to capture a concept while freezing all
other model weights.  This is the classical Textual Inversion technique:

1. Reserve N placeholder tokens (e.g. ``<concept_0>``, ``<concept_1>``).
2. Add their embeddings to the text encoder's embedding table.
3. Freeze all model weights *except* the new rows.
4. Train: any caption containing the placeholders updates only those rows.

This module is model-agnostic — it operates on any HF-style embedding
table that exposes ``num_embeddings`` and ``embedding_dim``.

Typical wiring::

    handler = TextualInversionHandler(text_encoder, tokenizer)
    handler.add_placeholder_tokens(["<my_concept>"], num_vectors=4)
    optimizer.add_param_group({"params": handler.get_trainable_params()})
    # ... training ...
    handler.save("learned_embeddings.safetensors")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class _PlaceholderRecord:
    base_token: str
    expanded_tokens: List[str]
    token_ids: List[int]


class TextualInversionHandler:
    """Manages placeholder tokens, embedding rows, and serialisation.

    Parameters
    ----------
    text_encoder : nn.Module
        Object exposing ``get_input_embeddings()`` (HF text encoder) or
        a direct ``nn.Embedding`` reference via ``input_embeddings``.
    tokenizer : Any
        Tokenizer with ``add_tokens`` and ``convert_tokens_to_ids``.
    init_strategy : str
        Initialisation strategy for new rows: ``"random"``, ``"zeros"``,
        or ``"copy_token"`` (copies the embedding of the optional
        ``init_token`` parameter passed to :meth:`add_placeholder_tokens`).
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer: Any,
        init_strategy: str = "random",
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.init_strategy = init_strategy

        self._placeholders: Dict[str, _PlaceholderRecord] = {}
        self._original_num_embeddings = self._embedding_table().num_embeddings

    # ------------------------------------------------------------------
    # Embedding table helpers
    # ------------------------------------------------------------------

    def _embedding_table(self) -> nn.Embedding:
        if hasattr(self.text_encoder, "get_input_embeddings"):
            emb = self.text_encoder.get_input_embeddings()
            if isinstance(emb, nn.Embedding):
                return emb
        if hasattr(self.text_encoder, "input_embeddings"):
            emb = self.text_encoder.input_embeddings
            if isinstance(emb, nn.Embedding):
                return emb
        if isinstance(self.text_encoder, nn.Embedding):
            return self.text_encoder
        raise RuntimeError("Cannot locate embedding table on text_encoder")

    # ------------------------------------------------------------------
    # Token registration
    # ------------------------------------------------------------------

    def add_placeholder_tokens(
        self,
        base_tokens: List[str],
        *,
        num_vectors: int = 1,
        init_token: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """Add placeholder tokens (each expanded to ``num_vectors`` slots).

        Returns a dict mapping each base token to its list of token ids.
        """
        if num_vectors < 1:
            raise ValueError("num_vectors must be >= 1")

        # Expand base tokens into vector slots
        added: Dict[str, List[int]] = {}
        new_tokens_to_add: List[str] = []

        for base in base_tokens:
            if num_vectors == 1:
                expanded = [base]
            else:
                expanded = [f"{base}_{i}" for i in range(num_vectors)]
            for tok in expanded:
                if tok not in new_tokens_to_add:
                    new_tokens_to_add.append(tok)
            self._placeholders[base] = _PlaceholderRecord(
                base_token=base,
                expanded_tokens=expanded,
                token_ids=[],
            )

        added_count = self.tokenizer.add_tokens(new_tokens_to_add)
        if added_count == 0:
            logger.warning("textual_inversion: tokenizer reported 0 new tokens added")

        # Resize the embedding table
        emb = self._embedding_table()
        old_num = emb.num_embeddings
        new_num = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else old_num + added_count

        if new_num > old_num:
            self._resize_embedding(emb, new_num)

        # Resolve token ids and initialise new rows
        for base, rec in self._placeholders.items():
            ids = [self.tokenizer.convert_tokens_to_ids(t) for t in rec.expanded_tokens]
            rec.token_ids = ids
            self._init_rows(emb, ids, init_token=init_token)
            added[base] = ids

        return added

    def _resize_embedding(self, emb: nn.Embedding, new_num: int) -> None:
        """Extend the embedding table to ``new_num`` rows, preserving old rows."""
        old_weight = emb.weight.data
        old_num, dim = old_weight.shape
        if new_num <= old_num:
            return
        new_weight = torch.empty((new_num, dim), dtype=old_weight.dtype, device=old_weight.device)
        new_weight[:old_num] = old_weight
        nn.init.normal_(new_weight[old_num:], mean=0.0, std=0.02)
        emb.weight = nn.Parameter(new_weight)
        emb.num_embeddings = new_num

    def _init_rows(
        self,
        emb: nn.Embedding,
        token_ids: List[int],
        init_token: Optional[str] = None,
    ) -> None:
        """Initialise newly-added rows according to the configured strategy."""
        with torch.no_grad():
            for tid in token_ids:
                if self.init_strategy == "zeros":
                    emb.weight.data[tid].zero_()
                elif self.init_strategy == "copy_token" and init_token:
                    src_id = self.tokenizer.convert_tokens_to_ids(init_token)
                    if src_id is not None and src_id < emb.weight.shape[0]:
                        emb.weight.data[tid] = emb.weight.data[src_id].clone()
                # Otherwise the row keeps its random initialisation from resize

    # ------------------------------------------------------------------
    # Trainable param exposure
    # ------------------------------------------------------------------

    def freeze_all_but_placeholders(self) -> None:
        """Freeze all text encoder parameters; only placeholder rows train.

        Since nn.Embedding stores all rows in a single Parameter, we use a
        gradient mask hook to zero out gradients for non-placeholder rows.
        """
        emb = self._embedding_table()

        # Freeze everything else in the text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        emb.weight.requires_grad = True

        placeholder_ids = self._all_placeholder_ids()
        if not placeholder_ids:
            return

        mask = torch.zeros(emb.weight.shape[0], dtype=torch.bool, device=emb.weight.device)
        mask[placeholder_ids] = True
        emb.weight._ti_mask = mask  # type: ignore[attr-defined]

        def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
            row_mask = emb.weight._ti_mask.unsqueeze(-1)  # type: ignore[attr-defined]
            return grad * row_mask.to(grad.dtype)

        if not hasattr(emb.weight, "_ti_grad_handle"):
            handle = emb.weight.register_hook(_grad_hook)
            emb.weight._ti_grad_handle = handle  # type: ignore[attr-defined]

    def _all_placeholder_ids(self) -> List[int]:
        ids: List[int] = []
        for rec in self._placeholders.values():
            ids.extend(rec.token_ids)
        return ids

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return the embedding weight tensor for optimizer registration.

        Note: the gradient mask installed by :meth:`freeze_all_but_placeholders`
        ensures only placeholder rows actually receive non-zero gradients.
        """
        emb = self._embedding_table()
        return [emb.weight] if emb.weight.requires_grad else []

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def export_learned_rows(self) -> Dict[str, torch.Tensor]:
        """Return a dict ``{base_token: tensor[num_vectors, dim]}``."""
        emb = self._embedding_table()
        out: Dict[str, torch.Tensor] = {}
        for base, rec in self._placeholders.items():
            if not rec.token_ids:
                continue
            rows = emb.weight.data[rec.token_ids].detach().cpu().clone()
            out[base] = rows
        return out

    def save(self, output_path: str, *, format: str = "auto") -> str:
        """Save learned embedding rows to disk.

        Format ``"auto"`` picks ``safetensors`` when available, else ``pt``.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = self.export_learned_rows()

        target = format
        if format == "auto":
            target = "safetensors" if out.suffix.lower() == ".safetensors" else "pt"

        if target == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(rows, str(out))
            except ImportError:
                logger.warning("safetensors not available, falling back to torch.save")
                torch.save(rows, str(out.with_suffix(".pt")))
        else:
            torch.save(rows, str(out))
        return str(out)

    def load(self, input_path: str) -> None:
        """Load learned rows from disk and apply them to the embedding table."""
        path = Path(input_path)
        if path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file
                rows = load_file(str(path))
            except ImportError:
                rows = torch.load(str(path), weights_only=True, map_location="cpu")
        else:
            rows = torch.load(str(path), weights_only=True, map_location="cpu")

        emb = self._embedding_table()
        with torch.no_grad():
            for base, tensor in rows.items():
                rec = self._placeholders.get(base)
                if rec is None or not rec.token_ids:
                    logger.warning("textual_inversion: load skipping unknown token '%s'", base)
                    continue
                if tensor.shape[0] != len(rec.token_ids):
                    logger.warning(
                        "textual_inversion: row count mismatch for '%s' (file=%d, expected=%d)",
                        base, tensor.shape[0], len(rec.token_ids),
                    )
                    continue
                for i, tid in enumerate(rec.token_ids):
                    emb.weight.data[tid] = tensor[i].to(
                        device=emb.weight.device, dtype=emb.weight.dtype,
                    )
