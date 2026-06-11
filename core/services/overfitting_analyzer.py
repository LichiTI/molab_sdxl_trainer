# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Advisory-only overfitting risk analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class OverfittingSignal:
    code: str
    severity: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OverfittingReport:
    schema_version: int
    run_id: str
    status: str
    signals: List[Dict[str, Any]]
    risk_level: str
    confidence: float
    recommendations: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class OverfittingAnalyzer:
    def analyze(
        self,
        *,
        run_id: str = "",
        train_loss: Sequence[float] | None = None,
        validation_loss: Sequence[float] | None = None,
        dataset_health: Dict[str, Any] | None = None,
        fixed_prompt_validation_set: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        train = self._numbers(train_loss or [])
        validation = self._numbers(validation_loss or [])
        dataset = dataset_health if isinstance(dataset_health, dict) else {}
        prompt_set = fixed_prompt_validation_set if isinstance(fixed_prompt_validation_set, dict) else {}
        signals: List[OverfittingSignal] = []

        if len(train) >= 4 and len(validation) >= 4:
            train_delta = train[-1] - train[0]
            validation_delta = validation[-1] - validation[0]
            gap_start = validation[0] - train[0]
            gap_end = validation[-1] - train[-1]
            if train_delta < 0 and validation_delta > 0:
                signals.append(OverfittingSignal("loss.train_down_validation_up", "critical", "Training loss is falling while validation loss is rising.", {"train_delta": round(train_delta, 6), "validation_delta": round(validation_delta, 6)}))
            if gap_start > 0 and gap_end > gap_start * 1.5:
                signals.append(OverfittingSignal("loss.gap_widening", "warning", "The train/validation loss gap is widening.", {"gap_start": round(gap_start, 6), "gap_end": round(gap_end, 6)}))
        if self._has_loss_spikes(train):
            signals.append(OverfittingSignal("loss.train_spikes", "warning", "Training loss has sustained spikes.", {"recent_losses": train[-6:]}))

        image_count = int(dataset.get("summary", {}).get("image_count") or dataset.get("image_count") or 0) if dataset else 0
        unique_tags = int(dataset.get("caption_health", {}).get("unique_tag_count") or 0) if dataset else 0
        if 0 < image_count < 10:
            signals.append(OverfittingSignal("dataset.small_prior", "warning", "The dataset is small enough to raise prior overfitting risk.", {"image_count": image_count}))
        if image_count > 0 and unique_tags > 0 and unique_tags / max(1, image_count) < 1.0:
            signals.append(OverfittingSignal("dataset.low_tag_diversity", "info", "Tag diversity is low relative to dataset size.", {"image_count": image_count, "unique_tag_count": unique_tags}))
        prompt_count = len(prompt_set.get("prompts", []) or []) if prompt_set else 0
        if prompt_count <= 0:
            signals.append(OverfittingSignal("validation.fixed_prompts_missing", "info", "Fixed validation prompts are unavailable, reducing comparison confidence.", {}))

        report = OverfittingReport(
            schema_version=1,
            run_id=run_id,
            status="ready",
            signals=[asdict(signal) for signal in signals],
            risk_level=self._risk_level(signals),
            confidence=self._confidence(train, validation, prompt_count),
            recommendations=self._recommendations(signals),
            metrics={
                "train_loss_count": len(train),
                "validation_loss_count": len(validation),
                "latest_train_loss": train[-1] if train else None,
                "latest_validation_loss": validation[-1] if validation else None,
                "fixed_prompt_count": prompt_count,
                "dataset_image_count": image_count,
            },
        )
        return asdict(report)

    def _numbers(self, values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                continue
        return result

    def _has_loss_spikes(self, train: List[float]) -> bool:
        if len(train) < 6:
            return False
        recent = train[-6:]
        baseline = sum(recent[:3]) / 3
        return sum(1 for value in recent[3:] if value > baseline * 1.5) >= 2

    def _risk_level(self, signals: List[OverfittingSignal]) -> str:
        severities = {signal.severity for signal in signals}
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        if signals:
            return "low"
        return "none"

    def _confidence(self, train: List[float], validation: List[float], prompt_count: int) -> float:
        score = 0.35
        if len(train) >= 4:
            score += 0.2
        if len(validation) >= 4:
            score += 0.3
        if prompt_count > 0:
            score += 0.15
        return min(0.95, round(score, 2))

    def _recommendations(self, signals: List[OverfittingSignal]) -> List[Dict[str, Any]]:
        codes = {signal.code for signal in signals}
        recommendations: List[Dict[str, Any]] = []
        if "loss.train_down_validation_up" in codes or "loss.gap_widening" in codes:
            recommendations.append({"code": "training.review_lr_or_stop", "message": "Review learning rate, validation samples, or consider stopping earlier."})
        if "dataset.small_prior" in codes:
            recommendations.append({"code": "dataset.add_or_augment", "message": "Add more images or increase regularization before long training runs."})
        if "validation.fixed_prompts_missing" in codes:
            recommendations.append({"code": "validation.enable_fixed_prompts", "message": "Enable a fixed validation prompt set for comparable samples."})
        return recommendations
