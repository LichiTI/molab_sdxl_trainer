# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""APEX Distillation loss helpers (Phase 8.7 / #115).

Distillation routes a frozen teacher model alongside the trainable
student.  Each step the student receives the same input as the teacher
and the loss is augmented with:

  1. **Output distillation** — KL divergence (or soft MSE) between
     student and teacher noise / velocity predictions.
  2. **Feature matching** — MSE between student / teacher intermediate
     hidden states at user-selected layers.

The actual training loop already computes a primary loss; the helpers
here return *additional* loss terms that get added with a weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Distillation hyper-parameters."""

    enabled: bool = False
    output_kl_weight: float = 0.0       # KL weight on output prediction
    output_mse_weight: float = 0.0      # plain MSE weight on output prediction
    feature_match_weight: float = 0.0   # MSE on hidden states
    feature_match_layers: tuple = field(default_factory=tuple)
    temperature: float = 1.0            # temperature for KL distillation
    detach_teacher: bool = True         # gradient never flows into the teacher


# ---------------------------------------------------------------------------
# Loss primitives
# ---------------------------------------------------------------------------

def output_kl_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    temperature: float = 1.0,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """KL divergence between flattened student and teacher distributions.

    Predictions are softmaxed along the last dimension after flattening
    spatial dims.  Useful when the model output can be interpreted as a
    score map (e.g. class logits, attention maps).
    """
    if detach_teacher:
        teacher_pred = teacher_pred.detach()
    s = student_pred.flatten(start_dim=1) / temperature
    t = teacher_pred.flatten(start_dim=1) / temperature
    log_p = F.log_softmax(s, dim=-1)
    q = F.softmax(t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def output_mse_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """Plain element-wise MSE between student and teacher predictions."""
    if detach_teacher:
        teacher_pred = teacher_pred.detach()
    return F.mse_loss(student_pred, teacher_pred)


def feature_match_loss(
    student_features: List[torch.Tensor],
    teacher_features: List[torch.Tensor],
    *,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """Sum of MSE losses between paired feature tensors.

    Lengths must match.  Returns a zero tensor if either list is empty.
    """
    if not student_features or not teacher_features:
        return torch.zeros((), device=(student_features[0].device if student_features else "cpu"))
    if len(student_features) != len(teacher_features):
        raise ValueError(
            f"student_features ({len(student_features)}) and teacher_features "
            f"({len(teacher_features)}) length mismatch"
        )
    total = student_features[0].new_zeros(())
    for s_feat, t_feat in zip(student_features, teacher_features):
        if detach_teacher:
            t_feat = t_feat.detach()
        if s_feat.shape != t_feat.shape:
            # Bilinear interpolate teacher to match student shape if 4D
            if s_feat.dim() == 4:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
            else:
                continue
        total = total + F.mse_loss(s_feat, t_feat)
    return total / max(len(student_features), 1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    config: DistillationConfig,
    *,
    student_features: Optional[List[torch.Tensor]] = None,
    teacher_features: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Combine the configured distillation terms into a single scalar loss."""
    if not config.enabled:
        return student_pred.new_zeros(())

    total = student_pred.new_zeros(())
    if config.output_kl_weight > 0:
        total = total + config.output_kl_weight * output_kl_loss(
            student_pred, teacher_pred,
            temperature=config.temperature,
            detach_teacher=config.detach_teacher,
        )
    if config.output_mse_weight > 0:
        total = total + config.output_mse_weight * output_mse_loss(
            student_pred, teacher_pred,
            detach_teacher=config.detach_teacher,
        )
    if config.feature_match_weight > 0 and student_features and teacher_features:
        total = total + config.feature_match_weight * feature_match_loss(
            student_features, teacher_features,
            detach_teacher=config.detach_teacher,
        )
    return total


# ---------------------------------------------------------------------------
# Feature capture helper
# ---------------------------------------------------------------------------

class FeatureCapture:
    """Register forward hooks that capture outputs of named modules.

    Usage::

        capture = FeatureCapture(model, layer_names=["block.5", "block.10"])
        model(input)
        feats = capture.features  # list[Tensor]
        capture.clear()           # call once per step to drop references
        capture.remove_hooks()    # call when done
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self._features: Dict[str, torch.Tensor] = {}
        self._handles = []
        self._install_hooks()

    @property
    def features(self) -> List[torch.Tensor]:
        return [self._features[name] for name in self.layer_names if name in self._features]

    def clear(self) -> None:
        self._features.clear()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

    def _install_hooks(self) -> None:
        for name in self.layer_names:
            module = self._resolve(name)
            if module is None:
                logger.warning("FeatureCapture: could not find module '%s'", name)
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) and output else output
            if isinstance(tensor, torch.Tensor):
                self._features[name] = tensor
        return _hook

    def _resolve(self, dotted_name: str) -> Optional[nn.Module]:
        parts = dotted_name.split(".")
        obj: nn.Module = self.model
        for p in parts:
            if hasattr(obj, p):
                child = getattr(obj, p)
                if isinstance(child, nn.Module):
                    obj = child
                    continue
            if isinstance(obj, nn.ModuleList) and p.isdigit():
                obj = obj[int(p)]
                continue
            if isinstance(obj, nn.ModuleDict) and p in obj:
                obj = obj[p]
                continue
            return None
        return obj
