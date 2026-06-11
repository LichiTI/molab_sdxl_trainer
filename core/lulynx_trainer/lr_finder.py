# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Learning Rate Finder — pre-training LR sweep.

Runs a short training phase with exponentially increasing LR, records loss
at each step, and finds the LR at which loss decreases fastest (steepest
negative gradient of the smoothed loss curve).

Uses checkpoint-based save/restore so the model returns to its original
state after the sweep.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LRFinderResult:
    lr_values: List[float] = field(default_factory=list)
    loss_values: List[float] = field(default_factory=list)
    smoothed_losses: List[float] = field(default_factory=list)
    suggested_lr: float = 0.0
    suggested_max_lr: float = 0.0
    num_steps_run: int = 0
    diverged_at_step: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suggested_lr": self.suggested_lr,
            "suggested_max_lr": self.suggested_max_lr,
            "num_steps_run": self.num_steps_run,
            "diverged_at_step": self.diverged_at_step,
            "lr_loss_pairs": list(zip(self.lr_values, self.loss_values)),
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"  Steps run: {self.num_steps_run}",
            f"  Suggested LR: {self.suggested_lr:.2e}",
            f"  Max LR (divergence): {self.suggested_max_lr:.2e}",
        ]
        if self.diverged_at_step >= 0:
            lines.append(f"  Diverged at step: {self.diverged_at_step}")
        return lines


class LRFinder:
    """Exponential LR sweep with checkpoint-based state restoration.

    Parameters
    ----------
    model : nn.Module
        The trainable model (or LoRA-injected model).
    optimizer : torch.optim.Optimizer
        The optimizer instance (will be modified during sweep, then restored).
    step_fn : Callable[[], float]
        A callable that executes one training step and returns the scalar loss.
        Should handle forward, backward, and optimizer.step() internally.
    start_lr : float
        Starting learning rate (default 1e-7).
    end_lr : float
        Ending learning rate (default 1e-1).
    num_steps : int
        Number of sweep steps (default 100).
    smooth_factor : float
        EMA smoothing factor for loss curve (default 0.05).
    diverge_threshold : float
        Stop early if smoothed loss exceeds best_loss × this factor (default 4.0).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fn: Callable[[], float],
        *,
        start_lr: float = 1e-7,
        end_lr: float = 1e-1,
        num_steps: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._step_fn = step_fn
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._num_steps = max(num_steps, 2)
        self._smooth_factor = smooth_factor
        self._diverge_threshold = diverge_threshold

    def run(self) -> LRFinderResult:
        """Execute the LR sweep, then restore model and optimizer state."""
        checkpoint_path = self._save_checkpoint()
        result = LRFinderResult()

        try:
            result = self._sweep()
        except Exception as e:
            logger.warning("LR Finder sweep failed: %s", e)
        finally:
            self._restore_checkpoint(checkpoint_path)
            try:
                checkpoint_path.unlink(missing_ok=True)
                checkpoint_path.parent.rmdir()
            except Exception:
                pass

        return result

    def _sweep(self) -> LRFinderResult:
        result = LRFinderResult()

        lr_mult = (self._end_lr / self._start_lr) ** (1.0 / max(self._num_steps - 1, 1))
        current_lr = self._start_lr

        for pg in self._optimizer.param_groups:
            pg["lr"] = current_lr

        best_loss = float("inf")
        avg_loss = 0.0

        for step in range(self._num_steps):
            self._optimizer.zero_grad()
            loss_val = self._step_fn()
            result.num_steps_run = step + 1

            if step == 0:
                avg_loss = loss_val
            else:
                avg_loss = self._smooth_factor * loss_val + (1 - self._smooth_factor) * avg_loss

            result.lr_values.append(current_lr)
            result.loss_values.append(loss_val)
            result.smoothed_losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if avg_loss > self._diverge_threshold * best_loss and step > 5:
                result.diverged_at_step = step
                result.suggested_max_lr = current_lr
                break

            current_lr *= lr_mult
            for pg in self._optimizer.param_groups:
                pg["lr"] = current_lr

        if result.suggested_max_lr <= 0 and result.lr_values:
            result.suggested_max_lr = result.lr_values[-1]

        result.suggested_lr = self._find_steepest_descent(
            result.lr_values, result.smoothed_losses
        )

        return result

    @staticmethod
    def _find_steepest_descent(lrs: List[float], losses: List[float]) -> float:
        """Find the LR at the steepest negative gradient of the smoothed loss curve."""
        if len(lrs) < 3:
            return lrs[0] if lrs else 1e-4

        log_lrs = [math.log10(lr) for lr in lrs]
        gradients = []
        for i in range(1, len(losses) - 1):
            denom = log_lrs[i + 1] - log_lrs[i - 1]
            if abs(denom) < 1e-12:
                continue
            grad = (losses[i + 1] - losses[i - 1]) / denom
            gradients.append((i, grad))

        if not gradients:
            return lrs[0]

        best_idx, _ = min(gradients, key=lambda x: x[1])
        return lrs[best_idx]

    def _save_checkpoint(self) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="lr_finder_"))
        tmp = tmp_dir / "checkpoint.pt"
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(state, tmp)
        logger.info("LR Finder: saved checkpoint (%s)", tmp)
        return tmp

    def _restore_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            logger.warning("LR Finder: checkpoint not found at %s", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state["model"], strict=False)
        self._optimizer.load_state_dict(state["optimizer"])
        logger.info("LR Finder: restored checkpoint")
