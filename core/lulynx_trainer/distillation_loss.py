"""Distillation loss computation for unified framework.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationLoss(ABC):
    """Base class for distillation losses."""

    @abstractmethod
    def compute(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute distillation loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student model output.
        teacher_output : torch.Tensor
            Teacher model output.
        **kwargs
            Additional mode-specific parameters.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Loss value and metrics dict.
        """
        pass


class ConsistencyDistillationLoss(DistillationLoss):
    """Consistency distillation loss (LCM-style, Mode D).

    Encourages student to produce similar outputs to teacher
    using simple MSE + optional perceptual loss.
    """

    def __init__(
        self,
        use_perceptual: bool = False,
        perceptual_weight: float = 0.1,
    ):
        """
        Parameters
        ----------
        use_perceptual : bool, default False
            Whether to add perceptual loss.
        perceptual_weight : float, default 0.1
            Weight for perceptual loss component.
        """
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight

        if self.use_perceptual:
            # Lazy-load perceptual loss (e.g., LPIPS)
            self._perceptual_fn = None
            logger.info("Perceptual loss enabled for consistency distillation")

    def compute(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute consistency loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student latents or images [batch, channels, height, width].
        teacher_output : torch.Tensor
            Teacher latents or images [batch, channels, height, width].

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Loss and metrics.
        """
        # MSE loss in latent space
        mse_loss = F.mse_loss(student_output, teacher_output)

        total_loss = mse_loss
        metrics = {
            "distill_mse": mse_loss.item(),
        }

        # Optional perceptual loss
        if self.use_perceptual:
            perceptual_loss = self._compute_perceptual_loss(
                student_output, teacher_output
            )
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            metrics["distill_perceptual"] = perceptual_loss.item()

        metrics["distill_consistency_total"] = total_loss.item()

        return total_loss, metrics

    def _compute_perceptual_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss (e.g., LPIPS).

        Parameters
        ----------
        student_output : torch.Tensor
            Student output.
        teacher_output : torch.Tensor
            Teacher output.

        Returns
        -------
        torch.Tensor
            Perceptual loss.
        """
        if self._perceptual_fn is None:
            # Lazy initialization
            try:
                import lpips
                self._perceptual_fn = lpips.LPIPS(net="vgg").to(student_output.device)
                self._perceptual_fn.eval()
                logger.info("Loaded LPIPS perceptual loss")
            except ImportError:
                logger.warning(
                    "lpips not installed, falling back to L1 loss. "
                    "Install with: pip install lpips"
                )
                self._perceptual_fn = "fallback"

        if self._perceptual_fn == "fallback":
            # Fallback to L1 loss
            return F.l1_loss(student_output, teacher_output)

        # LPIPS expects [-1, 1] range
        with torch.no_grad():
            loss = self._perceptual_fn(student_output, teacher_output)

        return loss.mean()


class FeatureMatchingLoss(DistillationLoss):
    """Feature matching loss for adversarial distillation (Mode A).

    Matches intermediate features from critic network.
    """

    def __init__(
        self,
        feature_layers: Optional[list] = None,
        weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        feature_layers : list of str, optional
            Which critic layers to match.
        weight : float, default 1.0
            Weight for feature matching loss.
        """
        self.feature_layers = feature_layers or []
        self.weight = weight

    def compute(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: Optional[dict] = None,
        teacher_features: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute feature matching loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student images.
        teacher_output : torch.Tensor
            Teacher images (real).
        student_features : dict, optional
            Student intermediate features from critic.
        teacher_features : dict, optional
            Teacher intermediate features from critic.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Loss and metrics.
        """
        if not student_features or not teacher_features:
            # No features provided, return zero loss
            logger.warning(
                "Feature matching enabled but no features provided, returning zero loss"
            )
            return torch.tensor(0.0, device=student_output.device), {}

        total_loss = 0.0
        num_matched = 0

        for layer_name in self.feature_layers:
            if layer_name in student_features and layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]

                # L1 loss between features
                feat_loss = F.l1_loss(student_feat, teacher_feat)
                total_loss = total_loss + feat_loss
                num_matched += 1

        if num_matched > 0:
            total_loss = total_loss / num_matched

        total_loss = self.weight * total_loss

        metrics = {
            "distill_feature_matching": total_loss.item(),
            "feature_layers_matched": num_matched,
        }

        return total_loss, metrics


class GuidedDatasetLoss(DistillationLoss):
    """Loss for guided dataset mode (Mode C).

    Simple MSE loss between student output and pre-generated teacher output.
    """

    def compute(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute guided dataset loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student output.
        teacher_output : torch.Tensor
            Pre-generated teacher output from dataset.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Loss and metrics.
        """
        mse_loss = F.mse_loss(student_output, teacher_output)

        metrics = {
            "distill_guided_mse": mse_loss.item(),
        }

        return mse_loss, metrics


def create_distillation_loss(
    mode: str,
    use_perceptual: bool = False,
    perceptual_weight: float = 0.1,
    feature_matching_layers: Optional[list] = None,
    feature_matching_weight: float = 1.0,
) -> DistillationLoss:
    """Create distillation loss for specified mode.

    Parameters
    ----------
    mode : str
        Distillation mode: "consistency", "guided_dataset", "online_adv", "hybrid".
    use_perceptual : bool, default False
        Enable perceptual loss (for consistency mode).
    perceptual_weight : float, default 0.1
        Perceptual loss weight.
    feature_matching_layers : list of str, optional
        Feature layers for matching (for adversarial mode).
    feature_matching_weight : float, default 1.0
        Feature matching weight.

    Returns
    -------
    DistillationLoss
        Loss instance for the specified mode.
    """
    mode = str(mode).strip().lower()

    if mode == "consistency":
        return ConsistencyDistillationLoss(
            use_perceptual=use_perceptual,
            perceptual_weight=perceptual_weight,
        )
    elif mode == "guided_dataset":
        return GuidedDatasetLoss()
    elif mode == "online_adv":
        # For adversarial mode, use consistency + feature matching
        # (can be combined in training loop)
        return ConsistencyDistillationLoss(
            use_perceptual=use_perceptual,
            perceptual_weight=perceptual_weight,
        )
    elif mode == "hybrid":
        # Hybrid uses consistency loss
        return ConsistencyDistillationLoss(
            use_perceptual=use_perceptual,
            perceptual_weight=perceptual_weight,
        )
    else:
        logger.warning(f"Unknown distillation mode '{mode}', using consistency")
        return ConsistencyDistillationLoss()


def compute_distillation_loss(
    loss_fn: DistillationLoss,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    weight: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    """Convenience function to compute weighted distillation loss.

    Parameters
    ----------
    loss_fn : DistillationLoss
        Loss function instance.
    student_output : torch.Tensor
        Student model output.
    teacher_output : torch.Tensor
        Teacher model output.
    weight : float, default 1.0
        Global weight for distillation loss.
    **kwargs
        Additional parameters passed to loss_fn.compute().

    Returns
    -------
    tuple of (torch.Tensor, dict)
        Weighted loss and metrics.
    """
    loss, metrics = loss_fn.compute(student_output, teacher_output, **kwargs)

    weighted_loss = weight * loss

    # Add weighted loss to metrics
    metrics["distill_loss_weighted"] = weighted_loss.item()
    metrics["distill_weight"] = weight

    return weighted_loss, metrics
