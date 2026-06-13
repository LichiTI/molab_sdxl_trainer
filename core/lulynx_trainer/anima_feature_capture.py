# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Opt-in DiT block feature capture seam (default-off -> bitwise parity).

A small module-level active-context, mirroring ``spectrum_probe`` /
``unified_cache_seam``: when no capture is active ``get_active_feature_capture``
returns ``None`` and the ``_run_blocks`` loop runs untouched, so the native
forward stays bitwise identical. When a capture *is* active (only while the
EMA feature-alignment reserve is enabled, see ``anima_ema_feature_align``), the
block loop records the output of selected blocks so a teacher/student feature
cosine-alignment loss can be computed.

Student features are captured during the main forward and therefore must keep
their autograd graph (``observe`` does NOT detach); teacher features are
captured under ``torch.no_grad`` so they carry no graph regardless.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, List, Optional

import torch


class FeatureCapture:
    """Collect the output of selected DiT blocks, keyed by block index."""

    def __init__(self, layers: List[int]) -> None:
        # Store as a set for O(1) membership, keep the original list for report.
        self._layers = [int(v) for v in layers]
        self._wanted = set(self._layers)
        self._features: Dict[int, torch.Tensor] = {}

    def observe(self, block_index: int, x: torch.Tensor) -> None:
        """Record block output ``x`` if ``block_index`` is requested.

        No detach: the student forward needs the graph for backprop. The
        teacher forward calls this under ``torch.no_grad`` so it is graph-free
        on that path anyway.
        """
        if block_index in self._wanted:
            self._features[int(block_index)] = x

    @property
    def features(self) -> Dict[int, torch.Tensor]:
        return self._features

    @property
    def layers(self) -> List[int]:
        return list(self._layers)


_CURRENT_CAPTURE: ContextVar[Optional[FeatureCapture]] = ContextVar(
    "lulynx_anima_feature_capture", default=None
)


def get_active_feature_capture() -> Optional[FeatureCapture]:
    """Return the active capture, or ``None`` (default) -> parity path."""
    return _CURRENT_CAPTURE.get()


def set_active_feature_capture(layers: List[int]) -> FeatureCapture:
    """Activate a fresh capture for ``layers`` and return it.

    Caller is responsible for ``clear_active_feature_capture()`` afterwards;
    prefer ``feature_capture_scope`` which guarantees cleanup.
    """
    capture = FeatureCapture(layers)
    _CURRENT_CAPTURE.set(capture)
    return capture


def clear_active_feature_capture() -> None:
    _CURRENT_CAPTURE.set(None)


@contextmanager
def feature_capture_scope(layers: List[int]) -> Iterator[FeatureCapture]:
    """Scope an active capture; always restores the previous context."""
    capture = FeatureCapture(layers)
    token = _CURRENT_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_CAPTURE.reset(token)


def parse_layer_list(spec: object) -> List[int]:
    """Parse a comma-separated (or list) layer spec into a list of ints.

    Mirrors the JLT ``_parse_layer_list`` contract: empty -> ``[]``.
    """
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return [int(v) for v in spec]
    return [int(v.strip()) for v in str(spec).split(",") if str(v).strip()]
