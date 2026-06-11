"""T-LoRA single/few-image benchmark result gate.

The real benchmark should run through the trainer. This module only ingests
small result summaries and decides whether T-LoRA improved overfitting and
prompt generalization enough to justify trainer integration work.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TLoRASingleImageMetrics:
    method: str
    train_loss: float
    holdout_loss: float
    prompt_score: float = 0.0
    steps: int = 0
    image_count: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_method: str = "") -> "TLoRASingleImageMetrics":
        return cls(
            method=str(value.get("method") or value.get("adapter_method") or default_method),
            train_loss=float(value.get("train_loss", value.get("final_train_loss", 0.0)) or 0.0),
            holdout_loss=float(value.get("holdout_loss", value.get("validation_loss", 0.0)) or 0.0),
            prompt_score=float(value.get("prompt_score", value.get("generalization_score", 0.0)) or 0.0),
            steps=max(int(value.get("steps", value.get("steps_completed", 0)) or 0), 0),
            image_count=max(int(value.get("image_count", value.get("train_image_count", 1)) or 1), 1),
        )

    @property
    def overfit_gap(self) -> float:
        return float(self.holdout_loss - self.train_loss)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "train_loss": float(self.train_loss),
            "holdout_loss": float(self.holdout_loss),
            "overfit_gap": float(self.overfit_gap),
            "prompt_score": float(self.prompt_score),
            "steps": int(self.steps),
            "image_count": int(self.image_count),
        }


@dataclass(frozen=True)
class TLoRASingleImageBenchmarkThresholds:
    min_gap_reduction: float = 0.05
    min_prompt_score_delta: float = 0.02
    max_train_loss_regression: float = 0.05
    min_steps: int = 1

    def normalized(self) -> "TLoRASingleImageBenchmarkThresholds":
        return TLoRASingleImageBenchmarkThresholds(
            min_gap_reduction=max(float(self.min_gap_reduction or 0.0), 0.0),
            min_prompt_score_delta=max(float(self.min_prompt_score_delta or 0.0), 0.0),
            max_train_loss_regression=max(float(self.max_train_loss_regression or 0.0), 0.0),
            min_steps=max(int(self.min_steps or 1), 1),
        )


def build_tlora_single_image_benchmark_scorecard(
    *,
    baseline: Mapping[str, Any] | TLoRASingleImageMetrics | None = None,
    tlora: Mapping[str, Any] | TLoRASingleImageMetrics | None = None,
    result_payload: Mapping[str, Any] | None = None,
    result_path: str | Path | None = None,
    thresholds: TLoRASingleImageBenchmarkThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_payload(result_payload, result_path)
    baseline_metrics = _metrics(baseline or payload.get("baseline"), "baseline")
    tlora_metrics = _metrics(tlora or payload.get("tlora"), "tlora")
    limits = _thresholds(thresholds)

    if baseline_metrics is None or tlora_metrics is None:
        return _missing_scorecard(payload, baseline_metrics, tlora_metrics)

    gap_reduction = baseline_metrics.overfit_gap - tlora_metrics.overfit_gap
    train_loss_delta = tlora_metrics.train_loss - baseline_metrics.train_loss
    prompt_score_delta = tlora_metrics.prompt_score - baseline_metrics.prompt_score
    blockers: list[str] = []
    if baseline_metrics.steps < limits.min_steps or tlora_metrics.steps < limits.min_steps:
        blockers.append("benchmark_steps_below_minimum")
    if gap_reduction < limits.min_gap_reduction:
        blockers.append("overfit_gap_reduction_below_threshold")
    if prompt_score_delta < limits.min_prompt_score_delta:
        blockers.append("prompt_generalization_delta_below_threshold")
    if train_loss_delta > limits.max_train_loss_regression:
        blockers.append("train_loss_regression_above_threshold")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_single_image_benchmark_scorecard_v0",
        "ok": ready,
        "benchmark_result_ready": True,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "baseline": baseline_metrics.as_dict(),
        "tlora": tlora_metrics.as_dict(),
        "thresholds": limits.__dict__,
        "deltas": {
            "overfit_gap_reduction": float(gap_reduction),
            "train_loss_delta": float(train_loss_delta),
            "prompt_score_delta": float(prompt_score_delta),
        },
        "progress_gates": {
            "has_baseline": True,
            "has_tlora": True,
            "steps_sufficient": baseline_metrics.steps >= limits.min_steps and tlora_metrics.steps >= limits.min_steps,
            "overfit_gap_improved": gap_reduction >= limits.min_gap_reduction,
            "prompt_generalization_improved": prompt_score_delta >= limits.min_prompt_score_delta,
            "train_loss_not_regressed": train_loss_delta <= limits.max_train_loss_regression,
        },
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run representative Anima/Newbie single-image trainer A/B"
            if ready
            else "collect stronger T-LoRA single-image benchmark evidence"
        ),
        "notes": [
            "This scorecard ingests benchmark results only; it does not run training.",
            "Promotion remains false until representative real trainer A/B and quality review pass.",
        ],
    }


def _load_payload(result_payload: Mapping[str, Any] | None, result_path: str | Path | None) -> dict[str, Any]:
    if isinstance(result_payload, Mapping):
        return dict(result_payload)
    if result_path:
        path = Path(result_path)
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                return dict(loaded)
    return {}


def _metrics(value: Any, default_method: str) -> TLoRASingleImageMetrics | None:
    if isinstance(value, TLoRASingleImageMetrics):
        return value
    if isinstance(value, Mapping):
        return TLoRASingleImageMetrics.from_mapping(value, default_method=default_method)
    return None


def _thresholds(
    thresholds: TLoRASingleImageBenchmarkThresholds | Mapping[str, Any] | None,
) -> TLoRASingleImageBenchmarkThresholds:
    if isinstance(thresholds, Mapping):
        return TLoRASingleImageBenchmarkThresholds(**thresholds).normalized()
    return (thresholds or TLoRASingleImageBenchmarkThresholds()).normalized()


def _missing_scorecard(
    payload: Mapping[str, Any],
    baseline: TLoRASingleImageMetrics | None,
    tlora: TLoRASingleImageMetrics | None,
) -> dict[str, Any]:
    blockers = []
    if baseline is None:
        blockers.append("baseline_result_missing")
    if tlora is None:
        blockers.append("tlora_result_missing")
    return {
        "schema_version": 1,
        "scorecard": "tlora_single_image_benchmark_scorecard_v0",
        "ok": False,
        "benchmark_result_ready": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "payload_keys": sorted(str(key) for key in payload.keys()),
        "blocked_reasons": blockers,
        "recommended_next_step": "run baseline and T-LoRA single-image benchmark cases",
    }


__all__ = [
    "TLoRASingleImageBenchmarkThresholds",
    "TLoRASingleImageMetrics",
    "build_tlora_single_image_benchmark_scorecard",
]
