"""TREAD token routing observation wrapper for training telemetry.

This module provides a lightweight observation layer for TREAD token routing
that records metrics without affecting training logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from .tread_token_routing import TreadTokenRoutePolicy, TreadTokenRoutePlan, build_tread_token_route_plan

logger = logging.getLogger(__name__)


class TreadObserver:
    """Observes TREAD token routing decisions during training.

    This observer records token pruning metrics without actually applying
    the routing. It's useful for collecting data to inform future decisions
    about enabling TREAD in production.

    Example
    -------
    >>> observer = TreadObserver(keep_ratio=0.75)
    >>> # In training loop:
    >>> observer.observe_tokens(hidden_states, step=100)
    >>> metrics = observer.get_metrics()
    """

    def __init__(
        self,
        keep_ratio: float = 1.0,
        min_keep_tokens: int = 1,
        score_mode: str = "l2",
    ):
        """
        Parameters
        ----------
        keep_ratio : float, default 1.0
            Fraction of tokens to keep (1.0 = no pruning).
        min_keep_tokens : int, default 1
            Minimum number of tokens to keep per sample.
        score_mode : str, default "l2"
            Token scoring method ("l2", "abs_mean", "provided").
        """
        self.policy = TreadTokenRoutePolicy(
            enabled=True,
            keep_ratio=keep_ratio,
            min_keep_tokens=min_keep_tokens,
            score_mode=score_mode,
        ).normalized()

        # Metrics accumulators
        self._total_observations = 0
        self._total_tokens = 0
        self._total_kept = 0
        self._total_dropped = 0
        self._plans: List[Dict] = []

    def observe_tokens(
        self,
        tokens: torch.Tensor,
        step: Optional[int] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> TreadTokenRoutePlan:
        """Observe token routing decision without applying it.

        Parameters
        ----------
        tokens : torch.Tensor
            Token embeddings of shape [batch, tokens, hidden].
        step : int, optional
            Current training step.
        scores : torch.Tensor, optional
            Pre-computed token importance scores.

        Returns
        -------
        TreadTokenRoutePlan
            Routing plan with statistics.
        """
        # Generate routing plan (doesn't modify tokens)
        plan = build_tread_token_route_plan(tokens, self.policy, scores=scores)

        # Record metrics
        self._total_observations += 1
        self._total_tokens += plan.token_count * plan.batch_size
        self._total_kept += plan.keep_count * plan.batch_size
        self._total_dropped += plan.drop_count * plan.batch_size

        # Store plan details
        plan_dict = plan.as_dict()
        if step is not None:
            plan_dict["step"] = step
        self._plans.append(plan_dict)

        return plan

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated observation metrics.

        Returns
        -------
        dict
            Metrics including average keep ratio, attention savings, etc.
        """
        if self._total_observations == 0:
            return {
                "total_observations": 0,
                "avg_keep_ratio": 1.0,
                "avg_attention_savings": 0.0,
                "total_tokens_observed": 0,
            }

        avg_keep_ratio = self._total_kept / max(self._total_tokens, 1)
        # Attention is O(n^2), so savings = 1 - (keep_ratio)^2
        avg_attention_savings = 1.0 - (avg_keep_ratio ** 2)

        return {
            "total_observations": self._total_observations,
            "avg_keep_ratio": avg_keep_ratio,
            "avg_drop_ratio": 1.0 - avg_keep_ratio,
            "avg_attention_savings": avg_attention_savings,
            "total_tokens_observed": self._total_tokens,
            "total_tokens_kept": self._total_kept,
            "total_tokens_dropped": self._total_dropped,
        }

    def get_recent_plans(self, n: int = 10) -> List[Dict]:
        """Get the most recent routing plans.

        Parameters
        ----------
        n : int, default 10
            Number of recent plans to return.

        Returns
        -------
        list of dict
            Recent routing plans with statistics.
        """
        return self._plans[-n:]

    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self._total_observations = 0
        self._total_tokens = 0
        self._total_kept = 0
        self._total_dropped = 0
        self._plans.clear()

    def summary(self) -> str:
        """Get a human-readable summary of observations.

        Returns
        -------
        str
            Summary text.
        """
        metrics = self.get_metrics()
        return (
            f"TREAD Observer Summary:\n"
            f"  Observations: {metrics['total_observations']}\n"
            f"  Avg Keep Ratio: {metrics['avg_keep_ratio']:.2%}\n"
            f"  Avg Drop Ratio: {metrics['avg_drop_ratio']:.2%}\n"
            f"  Estimated Attention Savings: {metrics['avg_attention_savings']:.2%}\n"
            f"  Total Tokens: {metrics['total_tokens_observed']}"
        )


def create_tread_observer_from_config(config) -> Optional[TreadObserver]:
    """Create TREAD observer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    TreadObserver or None
        Observer if enabled, None otherwise.
    """
    if not getattr(config, "tread_probe_enabled", False):
        return None

    keep_ratio = float(getattr(config, "tread_probe_keep_ratio", 1.0))
    min_keep = int(getattr(config, "tread_probe_min_keep_tokens", 1))
    score_mode = str(getattr(config, "tread_probe_score_mode", "l2"))

    observer = TreadObserver(
        keep_ratio=keep_ratio,
        min_keep_tokens=min_keep,
        score_mode=score_mode,
    )

    logger.info(
        f"TREAD observer initialized: keep_ratio={keep_ratio:.2%}, "
        f"min_keep={min_keep}, score_mode={score_mode}"
    )

    return observer
