"""Adapter target profiler for collecting gradient metrics.

This module profiles model layers during training to collect gradient norms
and other metrics for use with AdapterTargetPolicy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .adapter_target_policy import AdapterLayerMetric

logger = logging.getLogger(__name__)


class AdapterTargetProfiler:
    """Profiles model layers to collect metrics for adapter target selection.

    This profiler records gradient norms and parameter counts for all
    potential adapter target layers during training.

    Example
    -------
    >>> profiler = AdapterTargetProfiler()
    >>> profiler.attach_to_model(model, target_names=["to_q", "to_k", "to_v"])
    >>> # After backward pass:
    >>> profiler.collect_gradients(step=100)
    >>> profiler.save_profile("profile.json")
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.layer_info: Dict[str, Dict] = {}
        self._hooks = []

    def attach_to_model(
        self,
        model: nn.Module,
        target_names: Optional[List[str]] = None,
        prefix: str = "",
    ):
        """Attach profiler to model layers.

        Parameters
        ----------
        model : nn.Module
            Model to profile.
        target_names : list of str, optional
            Names of layers to profile. If None, profiles all Linear layers.
        prefix : str, optional
            Prefix to add to layer names.
        """
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Check if this layer matches target names
            layer_name = name.split(".")[-1] if name else ""
            if target_names is not None and layer_name not in target_names:
                continue

            full_name = f"{prefix}.{name}" if prefix else name

            # Record layer info
            param_count = sum(p.numel() for p in module.parameters())
            self.layer_info[full_name] = {
                "name": full_name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "parameter_count": param_count,
            }

            # Initialize metrics list
            self.metrics[full_name] = []

            logger.debug(f"Attached profiler to {full_name}")

    def collect_gradients(self, model: nn.Module, step: Optional[int] = None):
        """Collect gradient norms from attached layers.

        Parameters
        ----------
        model : nn.Module
            Model to collect gradients from.
        step : int, optional
            Current training step (for logging).
        """
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Find matching metric key
            metric_key = None
            for key in self.metrics.keys():
                if key.endswith(name):
                    metric_key = key
                    break

            if metric_key is None:
                continue

            # Collect gradient norm
            grad_norm = 0.0
            for param in module.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2

            grad_norm = grad_norm ** 0.5
            self.metrics[metric_key].append(grad_norm)

    def get_average_metrics(self) -> List[AdapterLayerMetric]:
        """Get averaged metrics for all profiled layers.

        Returns
        -------
        list of AdapterLayerMetric
            Metrics with averaged gradient norms.
        """
        results = []

        for name, grad_norms in self.metrics.items():
            if not grad_norms:
                continue

            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            param_count = self.layer_info[name]["parameter_count"]

            metric = AdapterLayerMetric(
                name=name,
                parameter_count=param_count,
                gradient_norm=avg_grad_norm,
            )
            results.append(metric)

        # Sort by gradient norm (descending)
        results.sort(key=lambda m: m.gradient_norm, reverse=True)
        return results

    def save_profile(self, path: str | Path):
        """Save profiler results to JSON file.

        Parameters
        ----------
        path : str or Path
            Path to save profile JSON.
        """
        metrics = self.get_average_metrics()

        profile = {
            "layers": [
                {
                    "name": m.name,
                    "parameter_count": m.parameter_count,
                    "gradient_norm": m.gradient_norm,
                }
                for m in metrics
            ]
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Saved adapter target profile to {path}")

    def load_profile(self, path: str | Path) -> List[AdapterLayerMetric]:
        """Load profile from JSON file.

        Parameters
        ----------
        path : str or Path
            Path to profile JSON.

        Returns
        -------
        list of AdapterLayerMetric
            Loaded metrics.
        """
        with open(path, "r") as f:
            profile = json.load(f)

        metrics = [
            AdapterLayerMetric.from_mapping(layer)
            for layer in profile.get("layers", [])
        ]

        return metrics

    def summary(self) -> str:
        """Get human-readable summary of profiling results.

        Returns
        -------
        str
            Summary text.
        """
        metrics = self.get_average_metrics()

        lines = [
            "Adapter Target Profile Summary:",
            f"  Total Layers: {len(metrics)}",
        ]

        if metrics:
            lines.append(f"  Top 5 by Gradient Norm:")
            for i, m in enumerate(metrics[:5], 1):
                lines.append(f"    {i}. {m.name}: {m.gradient_norm:.4f}")

        return "\n".join(lines)


def create_profiler_from_config(config) -> Optional[AdapterTargetProfiler]:
    """Create profiler from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    AdapterTargetProfiler or None
        Profiler if enabled, None otherwise.
    """
    if not getattr(config, "adapter_target_profile_enabled", False):
        return None

    profiler = AdapterTargetProfiler()

    logger.info("Adapter target profiler initialized")

    return profiler
