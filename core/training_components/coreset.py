"""
Coreset: loss-driven sample diagnostics and optional weighted sampling.

The manager keeps a compact rolling history per training sample, classifies
samples into easy/normal/hard/toxic buckets, and can emit JSON reports for UI
or offline dataset cleanup. Sampling integration remains opt-in through the
existing WeightedBatchSampler helper.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SampleStats:
    """Rolling statistics for one training sample."""

    filename: str
    loss_history: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)

    category: str = "normal"
    weight: float = 1.0

    loss_mean: float = 0.0
    loss_std: float = 0.0
    loss_trend: float = 0.0
    gradient_mean: float = 0.0

    def update(self, loss: float, gradient_norm: Optional[float] = None) -> None:
        self.loss_history.append(float(loss))
        if gradient_norm is not None:
            self.gradient_norms.append(float(gradient_norm))

        if len(self.loss_history) > 20:
            self.loss_history.pop(0)
        if len(self.gradient_norms) > 20:
            self.gradient_norms.pop(0)

        if self.loss_history:
            self.loss_mean = float(np.mean(self.loss_history))
            self.loss_std = float(np.std(self.loss_history))

        if len(self.loss_history) >= 3:
            x = np.arange(len(self.loss_history), dtype=np.float32)
            self.loss_trend = float(np.polyfit(x, np.asarray(self.loss_history, dtype=np.float32), 1)[0])

        if self.gradient_norms:
            self.gradient_mean = float(np.mean(self.gradient_norms))

    def classify(
        self,
        *,
        easy_threshold: float = 0.01,
        hard_loss_threshold: float = 0.1,
        toxic_std_threshold: float = 0.05,
        easy_weight: float = 0.2,
        hard_weight: float = 2.0,
        toxic_weight: float = 0.0,
        normal_weight: float = 1.0,
        min_history: int = 5,
    ) -> None:
        if len(self.loss_history) < max(int(min_history), 1):
            self.category = "normal"
            self.weight = float(normal_weight)
            return

        if self.loss_std > float(toxic_std_threshold):
            self.category = "toxic"
            self.weight = float(toxic_weight)
            return

        if self.loss_trend < -float(easy_threshold):
            self.category = "easy"
            self.weight = float(easy_weight)
            return

        if self.loss_mean > float(hard_loss_threshold) and abs(self.loss_trend) < max(float(easy_threshold) * 0.1, 1e-6):
            self.category = "hard"
            self.weight = float(hard_weight)
            return

        self.category = "normal"
        self.weight = float(normal_weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "weight": float(self.weight),
            "loss_mean": float(self.loss_mean),
            "loss_std": float(self.loss_std),
            "loss_trend": float(self.loss_trend),
            "gradient_mean": float(self.gradient_mean),
            "history_len": len(self.loss_history),
        }


class CoresetManager:
    """Tracks sample statistics and produces coreset diagnostics."""

    def __init__(
        self,
        easy_weight: float = 0.2,
        hard_weight: float = 2.0,
        toxic_weight: float = 0.0,
        auto_classify_after: int = 5,
        easy_threshold: float = 0.01,
        hard_loss_threshold: float = 0.1,
        toxic_std_threshold: float = 0.05,
        min_history: int = 5,
    ):
        self.easy_weight = float(easy_weight)
        self.hard_weight = float(hard_weight)
        self.toxic_weight = float(toxic_weight)
        self.normal_weight = 1.0
        self.auto_classify_after = max(int(auto_classify_after), 0)
        self.easy_threshold = float(easy_threshold)
        self.hard_loss_threshold = float(hard_loss_threshold)
        self.toxic_std_threshold = float(toxic_std_threshold)
        self.min_history = max(int(min_history), 1)

        self._samples: Dict[str, SampleStats] = {}
        self._epoch = 0
        self._total_updates = 0
        self._weights_cache: Optional[Dict[str, float]] = None
        self._cache_valid = False

    def update_sample(self, filename: str, loss: float, gradient_norm: Optional[float] = None) -> None:
        key = str(filename)
        if key not in self._samples:
            self._samples[key] = SampleStats(filename=key)

        self._samples[key].update(float(loss), gradient_norm)
        self._total_updates += 1
        self._cache_valid = False

    def update_batch(
        self,
        filenames: List[str],
        losses: List[float],
        gradient_norms: Optional[List[float]] = None,
    ) -> None:
        if gradient_norms is None:
            gradient_norms = [None] * len(filenames)  # type: ignore[list-item]

        for filename, loss, grad_norm in zip(filenames, losses, gradient_norms):
            self.update_sample(filename, float(loss), grad_norm)

    def on_epoch_end(self) -> None:
        self._epoch += 1
        if self.auto_classify_after <= 0 or self._epoch >= self.auto_classify_after:
            self.classify_all()

    def classify_all(self) -> None:
        for sample in self._samples.values():
            sample.classify(
                easy_threshold=self.easy_threshold,
                hard_loss_threshold=self.hard_loss_threshold,
                toxic_std_threshold=self.toxic_std_threshold,
                easy_weight=self.easy_weight,
                hard_weight=self.hard_weight,
                toxic_weight=self.toxic_weight,
                normal_weight=self.normal_weight,
                min_history=self.min_history,
            )

        self._cache_valid = False
        logger.info("[Coreset] Classification: %s", self.get_statistics().get("categories", {}))

    def get_weights(self) -> Dict[str, float]:
        if self._cache_valid and self._weights_cache is not None:
            return self._weights_cache

        self._weights_cache = {filename: sample.weight for filename, sample in self._samples.items()}
        self._cache_valid = True
        return self._weights_cache

    def get_weight(self, filename: str) -> float:
        sample = self._samples.get(str(filename))
        return float(sample.weight) if sample else 1.0

    def get_sample_list(self) -> List[Tuple[str, float]]:
        return [(filename, float(sample.weight)) for filename, sample in self._samples.items()]

    def get_statistics(self) -> Dict[str, Any]:
        categories = {"easy": 0, "normal": 0, "hard": 0, "toxic": 0}
        total_weight = 0.0
        for sample in self._samples.values():
            categories[sample.category] = categories.get(sample.category, 0) + 1
            total_weight += max(float(sample.weight), 0.0)

        return {
            "total": len(self._samples),
            "categories": categories,
            "epoch": self._epoch,
            "updates": self._total_updates,
            "effective_size": sum(1 for sample in self._samples.values() if sample.weight > 0),
            "total_weight": total_weight,
        }

    def _ranked_samples(self, category: str, top_k: int, *, reverse: bool = True) -> List[Dict[str, Any]]:
        items = [sample for sample in self._samples.values() if sample.category == category]
        if category == "toxic":
            items.sort(key=lambda sample: (sample.loss_std, sample.loss_mean), reverse=True)
        elif category == "easy":
            items.sort(key=lambda sample: (sample.loss_trend, sample.loss_mean), reverse=False)
        else:
            items.sort(key=lambda sample: sample.loss_mean, reverse=reverse)
        return [dict({"filename": sample.filename}, **sample.to_dict()) for sample in items[: max(int(top_k), 0)]]

    def get_toxic_samples(self) -> List[str]:
        return [filename for filename, sample in self._samples.items() if sample.category == "toxic"]

    def get_easy_samples(self, top_k: int = 10) -> List[Tuple[str, float]]:
        return [(item["filename"], item["loss_mean"]) for item in self._ranked_samples("easy", top_k, reverse=False)]

    def get_hard_samples(self, top_k: int = 10) -> List[Tuple[str, float]]:
        return [(item["filename"], item["loss_mean"]) for item in self._ranked_samples("hard", top_k, reverse=True)]

    def to_report(self, top_k: int = 20, *, include_samples: bool = True) -> Dict[str, Any]:
        summary = self.get_statistics()
        report: Dict[str, Any] = {
            "epoch": self._epoch,
            "updates": self._total_updates,
            "summary": summary,
            "settings": {
                "easy_weight": self.easy_weight,
                "hard_weight": self.hard_weight,
                "toxic_weight": self.toxic_weight,
                "easy_threshold": self.easy_threshold,
                "hard_loss_threshold": self.hard_loss_threshold,
                "toxic_std_threshold": self.toxic_std_threshold,
                "auto_classify_after": self.auto_classify_after,
                "min_history": self.min_history,
            },
            "top_hard": self._ranked_samples("hard", top_k, reverse=True),
            "top_toxic": self._ranked_samples("toxic", top_k, reverse=True),
            "top_easy": self._ranked_samples("easy", top_k, reverse=False),
        }
        if include_samples:
            report["samples"] = {filename: sample.to_dict() for filename, sample in sorted(self._samples.items())}
        return report

    def save_report(self, path: str, top_k: int = 20, *, include_samples: bool = True) -> Dict[str, Any]:
        report = self.to_report(top_k=top_k, include_samples=include_samples)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[Coreset] Saved report to %s", target)
        return report

    def save(self, path: str) -> None:
        self.save_report(path, include_samples=True)

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._epoch = int(data.get("epoch", 0) or 0)
        self._total_updates = int(data.get("updates", data.get("summary", {}).get("updates", 0)) or 0)

        for filename, info in data.get("samples", {}).items():
            sample = SampleStats(filename=str(filename))
            sample.category = info.get("category", "normal")
            sample.weight = float(info.get("weight", 1.0))
            sample.loss_mean = float(info.get("loss_mean", 0.0))
            sample.loss_std = float(info.get("loss_std", 0.0))
            sample.loss_trend = float(info.get("loss_trend", 0.0))
            sample.gradient_mean = float(info.get("gradient_mean", 0.0))
            self._samples[str(filename)] = sample

        self._cache_valid = False
        logger.info("[Coreset] Loaded %d samples from %s", len(self._samples), path)


class WeightedBatchSampler(torch.utils.data.Sampler):
    """Weighted batch sampler backed by CoresetManager weights."""

    def __init__(
        self,
        coreset: CoresetManager,
        filenames: List[str],
        batch_size: int,
        drop_last: bool = True,
    ):
        self.coreset = coreset
        self.filenames = filenames
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        weights = [self.coreset.get_weight(filename) for filename in self.filenames]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0] * len(weights)
            total_weight = float(len(weights))

        probs = [weight / total_weight for weight in weights]
        indices = np.random.choice(len(self.filenames), size=len(self.filenames), replace=True, p=probs)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size].tolist()
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.filenames) // self.batch_size
        return (len(self.filenames) + self.batch_size - 1) // self.batch_size


def create_coreset_manager(**kwargs) -> CoresetManager:
    return CoresetManager(**kwargs)


def create_weighted_sampler(coreset: CoresetManager, dataset, batch_size: int) -> WeightedBatchSampler:
    filenames = getattr(dataset, "filenames", None)
    if filenames is None:
        filenames = [str(i) for i in range(len(dataset))]

    return WeightedBatchSampler(coreset=coreset, filenames=filenames, batch_size=batch_size)
