"""CDM-QTA LoRA quantization observation wrapper for training telemetry.

This module provides observation of LoRA weight quantization errors without
actually quantizing weights during training.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .cdm_qta_lora_probe import fake_quantize_symmetric_ste

logger = logging.getLogger(__name__)


class CDMQTAObserver:
    """Observes LoRA weight quantization characteristics during training.

    This observer computes quantization error metrics for LoRA weights
    without actually quantizing them. Useful for understanding the impact
    of potential quantization before enabling it in production.

    Example
    -------
    >>> observer = CDMQTAObserver(quant_bits=8)
    >>> # In training loop:
    >>> observer.observe_lora_weights(lora_down, lora_up, step=100)
    >>> metrics = observer.get_metrics()
    """

    def __init__(
        self,
        quant_bits: int = 8,
        observe_down: bool = True,
        observe_up: bool = True,
    ):
        """
        Parameters
        ----------
        quant_bits : int, default 8
            Target quantization bits (4 or 8).
        observe_down : bool, default True
            Whether to observe lora_down quantization.
        observe_up : bool, default True
            Whether to observe lora_up quantization.
        """
        if quant_bits not in {4, 8}:
            raise ValueError(f"quant_bits must be 4 or 8, got {quant_bits}")

        self.quant_bits = quant_bits
        self.observe_down = observe_down
        self.observe_up = observe_up

        # Metrics accumulators
        self._total_observations = 0
        self._down_errors = []
        self._up_errors = []
        self._down_snr = []
        self._up_snr = []

    def observe_lora_weights(
        self,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Observe quantization error for LoRA weight pair.

        Parameters
        ----------
        lora_down : torch.Tensor
            LoRA down projection weights.
        lora_up : torch.Tensor
            LoRA up projection weights.
        step : int, optional
            Current training step.

        Returns
        -------
        dict
            Quantization error metrics for this observation.
        """
        metrics = {"step": step} if step is not None else {}

        if self.observe_down:
            down_metrics = self._compute_quant_error(lora_down, "down")
            metrics.update(down_metrics)
            self._down_errors.append(down_metrics["down_mse"])
            self._down_snr.append(down_metrics["down_snr_db"])

        if self.observe_up:
            up_metrics = self._compute_quant_error(lora_up, "up")
            metrics.update(up_metrics)
            self._up_errors.append(up_metrics["up_mse"])
            self._up_snr.append(up_metrics["up_snr_db"])

        self._total_observations += 1
        return metrics

    def _compute_quant_error(
        self,
        weight: torch.Tensor,
        prefix: str,
    ) -> Dict[str, float]:
        """Compute quantization error metrics for a weight tensor.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to analyze.
        prefix : str
            Prefix for metric keys ("down" or "up").

        Returns
        -------
        dict
            Error metrics including MSE, SNR, max error.
        """
        with torch.no_grad():
            # Fake quantize
            quantized = fake_quantize_symmetric_ste(weight, bits=self.quant_bits)

            # Compute error
            error = weight - quantized.dequantized
            mse = (error ** 2).mean().item()
            max_error = error.abs().max().item()

            # Compute SNR (Signal-to-Noise Ratio)
            signal_power = (weight ** 2).mean().item()
            noise_power = mse
            snr = 10.0 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-10))).item()

            # Weight statistics
            weight_mean = weight.mean().item()
            weight_std = weight.std().item()
            weight_max_abs = weight.abs().max().item()

            return {
                f"{prefix}_mse": mse,
                f"{prefix}_max_error": max_error,
                f"{prefix}_snr_db": snr,
                f"{prefix}_weight_mean": weight_mean,
                f"{prefix}_weight_std": weight_std,
                f"{prefix}_weight_max_abs": weight_max_abs,
                f"{prefix}_scale": quantized.scale.item(),
            }

    def observe_lora_layer(
        self,
        layer: nn.Module,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """Observe quantization error for a LoRA layer.

        Parameters
        ----------
        layer : nn.Module
            LoRA layer with lora_down and lora_up attributes/parameters.
        step : int, optional
            Current training step.

        Returns
        -------
        dict or None
            Quantization error metrics, or None if layer doesn't have LoRA weights.
        """
        # Try to find lora_down and lora_up
        lora_down = None
        lora_up = None

        if hasattr(layer, "lora_down"):
            lora_down = layer.lora_down
            if isinstance(lora_down, nn.Parameter):
                lora_down = lora_down.data
            elif isinstance(lora_down, nn.Module):
                lora_down = lora_down.weight.data

        if hasattr(layer, "lora_up"):
            lora_up = layer.lora_up
            if isinstance(lora_up, nn.Parameter):
                lora_up = lora_up.data
            elif isinstance(lora_up, nn.Module):
                lora_up = lora_up.weight.data

        if lora_down is None or lora_up is None:
            return None

        return self.observe_lora_weights(lora_down, lora_up, step=step)

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated quantization error metrics.

        Returns
        -------
        dict
            Aggregated metrics including average MSE, SNR, etc.
        """
        if self._total_observations == 0:
            return {
                "total_observations": 0,
                "quant_bits": self.quant_bits,
            }

        metrics = {
            "total_observations": self._total_observations,
            "quant_bits": self.quant_bits,
        }

        if self._down_errors:
            metrics["down_avg_mse"] = sum(self._down_errors) / len(self._down_errors)
            metrics["down_avg_snr_db"] = sum(self._down_snr) / len(self._down_snr)
            metrics["down_min_snr_db"] = min(self._down_snr)
            metrics["down_max_snr_db"] = max(self._down_snr)

        if self._up_errors:
            metrics["up_avg_mse"] = sum(self._up_errors) / len(self._up_errors)
            metrics["up_avg_snr_db"] = sum(self._up_snr) / len(self._up_snr)
            metrics["up_min_snr_db"] = min(self._up_snr)
            metrics["up_max_snr_db"] = max(self._up_snr)

        return metrics

    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self._total_observations = 0
        self._down_errors.clear()
        self._up_errors.clear()
        self._down_snr.clear()
        self._up_snr.clear()

    def summary(self) -> str:
        """Get a human-readable summary of observations.

        Returns
        -------
        str
            Summary text.
        """
        metrics = self.get_metrics()
        lines = [
            f"CDM-QTA Observer Summary (INT{self.quant_bits}):",
            f"  Observations: {metrics['total_observations']}",
        ]

        if "down_avg_snr_db" in metrics:
            lines.append(f"  LoRA Down Avg SNR: {metrics['down_avg_snr_db']:.2f} dB")
            lines.append(f"  LoRA Down Avg MSE: {metrics['down_avg_mse']:.6f}")

        if "up_avg_snr_db" in metrics:
            lines.append(f"  LoRA Up Avg SNR: {metrics['up_avg_snr_db']:.2f} dB")
            lines.append(f"  LoRA Up Avg MSE: {metrics['up_avg_mse']:.6f}")

        return "\n".join(lines)


def create_cdm_qta_observer_from_config(config) -> Optional[CDMQTAObserver]:
    """Create CDM-QTA observer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    CDMQTAObserver or None
        Observer if enabled, None otherwise.
    """
    if not getattr(config, "cdm_qta_probe_enabled", False):
        return None

    quant_bits = int(getattr(config, "cdm_qta_probe_bits", 8))
    observe_down = bool(getattr(config, "cdm_qta_probe_down", True))
    observe_up = bool(getattr(config, "cdm_qta_probe_up", True))

    observer = CDMQTAObserver(
        quant_bits=quant_bits,
        observe_down=observe_down,
        observe_up=observe_up,
    )

    logger.info(
        f"CDM-QTA observer initialized: bits={quant_bits}, "
        f"observe_down={observe_down}, observe_up={observe_up}"
    )

    return observer
