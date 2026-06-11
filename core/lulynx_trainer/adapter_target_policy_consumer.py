"""Adapter target policy consumer for LoRA injector integration.

This module applies AdapterTargetPolicy to select which layers to inject
LoRA adapters based on profiled metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .adapter_target_policy import (
    AdapterTargetPolicyConfig,
    AdapterLayerMetric,
    build_adapter_target_policy_plan,
)

logger = logging.getLogger(__name__)


class AdapterTargetPolicyConsumer:
    """Consumes AdapterTargetPolicy to select injection targets.

    Example
    -------
    >>> consumer = AdapterTargetPolicyConsumer.from_profile("profile.json")
    >>> targets, rank_map = consumer.select_targets(
    ...     available_modules=["to_q", "to_k", "to_v", "to_out"],
    ...     base_rank=16
    ... )
    """

    def __init__(
        self,
        metrics: List[AdapterLayerMetric],
        config: AdapterTargetPolicyConfig,
    ):
        """
        Parameters
        ----------
        metrics : list of AdapterLayerMetric
            Profiled layer metrics.
        config : AdapterTargetPolicyConfig
            Policy configuration.
        """
        self.metrics = metrics
        self.config = config.normalized()

    @classmethod
    def from_profile(
        cls,
        profile_path: str | Path,
        config: Optional[AdapterTargetPolicyConfig] = None,
    ) -> "AdapterTargetPolicyConsumer":
        """Create consumer from saved profile JSON.

        Parameters
        ----------
        profile_path : str or Path
            Path to profile JSON file.
        config : AdapterTargetPolicyConfig, optional
            Policy configuration. If None, uses defaults.

        Returns
        -------
        AdapterTargetPolicyConsumer
        """
        with open(profile_path, "r") as f:
            profile = json.load(f)

        metrics = [
            AdapterLayerMetric.from_mapping(layer)
            for layer in profile.get("layers", [])
        ]

        if config is None:
            config = AdapterTargetPolicyConfig()

        return cls(metrics, config)

    def select_targets(
        self,
        available_modules: List[str],
        base_rank: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Select target modules and rank assignments.

        Parameters
        ----------
        available_modules : list of str
            Available module names to choose from.
        base_rank : int, optional
            Base rank to use. If None, uses config.base_rank.

        Returns
        -------
        tuple of (list of str, dict)
            Selected module names and rank map {name: rank}.
        """
        if base_rank is not None:
            config = AdapterTargetPolicyConfig(
                policy=self.config.policy,
                base_rank=base_rank,
                min_rank=self.config.min_rank,
                max_rank=self.config.max_rank,
                target_fraction=self.config.target_fraction,
                top_k=self.config.top_k,
                min_score=self.config.min_score,
            )
        else:
            config = self.config

        # Apply policy
        plan = build_adapter_target_policy_plan(self.metrics, config)

        # Filter to available modules. A per-type profile stores the target name
        # directly ("to_q", "to_out.0", "ff.net.0.proj"); a per-layer profile
        # stores a full module path ("net.blocks.0.attn.to_q") whose suffix is the
        # target. Try a full match first, then suffix / leaf matching, so dotted
        # targets ("to_out.0", "ff.net.0.proj") are not silently dropped (which
        # would force the fallback-to-all path and make gradient_selected a no-op).
        rank_map = {}

        for row in plan.rows:
            if not row.selected:
                continue

            name = row.name
            if name in available_modules:
                matched = name
            else:
                matched = next(
                    (t for t in available_modules if name.endswith(t) or name.split(".")[-1] == t),
                    None,
                )

            if matched is not None and matched not in rank_map:
                rank_map[matched] = row.rank

        selected_names = list(rank_map.keys())

        if not selected_names:
            logger.warning(
                f"Policy selected {plan.selected_count} layers, but none matched available modules. "
                f"Falling back to all modules."
            )
            selected_names = available_modules
            rank_map = {name: config.base_rank for name in available_modules}

        logger.info(
            f"Policy '{config.policy}' selected {len(selected_names)}/{len(available_modules)} modules"
        )

        return selected_names, rank_map

    def get_plan(self) -> dict:
        """Get full policy plan as dict.

        Returns
        -------
        dict
            Full policy plan details.
        """
        plan = apply_adapter_target_policy(self.metrics, self.config)

        return {
            "policy": plan.policy,
            "selected_count": plan.selected_count,
            "total_count": plan.total_count,
            "rows": [row.as_dict() for row in plan.rows],
        }


def load_policy_consumer_from_config(
    config,
) -> Optional[AdapterTargetPolicyConsumer]:
    """Load policy consumer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    AdapterTargetPolicyConsumer or None
        Consumer if policy is enabled and profile exists, None otherwise.
    """
    profile_path = getattr(config, "adapter_target_policy_profile_path", None)
    if not profile_path:
        return None

    profile_path = Path(profile_path)
    if not profile_path.exists():
        logger.warning(f"Adapter target policy profile not found: {profile_path}")
        return None

    # Build policy config from training config
    policy_config = AdapterTargetPolicyConfig(
        policy=getattr(config, "adapter_target_policy", "all"),
        base_rank=getattr(config, "network_dim", 16),
        min_rank=getattr(config, "adapter_target_policy_min_rank", 1),
        max_rank=getattr(config, "adapter_target_policy_max_rank", 64),
        target_fraction=getattr(config, "adapter_target_policy_fraction", 1.0),
        top_k=getattr(config, "adapter_target_policy_top_k", 0),
        min_score=getattr(config, "adapter_target_policy_min_score", 0.0),
    )

    consumer = AdapterTargetPolicyConsumer.from_profile(profile_path, policy_config)

    logger.info(
        f"Loaded adapter target policy consumer: policy={policy_config.policy}, "
        f"layers={len(consumer.metrics)}"
    )

    return consumer
