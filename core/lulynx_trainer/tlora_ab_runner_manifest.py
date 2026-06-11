"""Default-off T-LoRA real trainer A/B manifest and result gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tlora_single_image_benchmark import (
    TLoRASingleImageBenchmarkThresholds,
    build_tlora_single_image_benchmark_scorecard,
)


@dataclass(frozen=True)
class TLoRAABCase:
    case_id: str
    family: str = "anima"
    image_count: int = 1
    max_train_steps: int = 100
    resolution: int = 1024
    baseline_network_module: str = "networks.lora"
    tlora_network_module: str = "networks.tlora"
    tlora_min_rank: int = 1
    tlora_rank_schedule: str = "linear"
    tlora_orthogonal_init: bool = False
    seed: int = 0

    def normalized(self) -> "TLoRAABCase":
        return TLoRAABCase(
            case_id=str(self.case_id or "case").strip() or "case",
            family=str(self.family or "anima").strip().lower() or "anima",
            image_count=max(int(self.image_count or 1), 1),
            max_train_steps=max(int(self.max_train_steps or 1), 1),
            resolution=max(int(self.resolution or 1), 1),
            baseline_network_module=str(self.baseline_network_module or "networks.lora"),
            tlora_network_module=str(self.tlora_network_module or "networks.tlora"),
            tlora_min_rank=max(int(self.tlora_min_rank or 1), 1),
            tlora_rank_schedule=str(self.tlora_rank_schedule or "linear"),
            tlora_orthogonal_init=bool(self.tlora_orthogonal_init),
            seed=max(int(self.seed or 0), 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "image_count": int(self.image_count),
            "max_train_steps": int(self.max_train_steps),
            "resolution": int(self.resolution),
            "baseline_network_module": self.baseline_network_module,
            "tlora_network_module": self.tlora_network_module,
            "tlora_min_rank": int(self.tlora_min_rank),
            "tlora_rank_schedule": self.tlora_rank_schedule,
            "tlora_orthogonal_init": bool(self.tlora_orthogonal_init),
            "seed": int(self.seed),
        }


def build_tlora_ab_runner_manifest(
    cases: Sequence[TLoRAABCase | Mapping[str, Any]],
    *,
    output_root: str | Path = "temp/tlora_ab",
    thresholds: TLoRASingleImageBenchmarkThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_cases = tuple(_case(item) for item in cases)
    blockers: list[str] = []
    if not normalized_cases:
        blockers.append("ab_cases_missing")
    duplicate_ids = _duplicates([case.case_id for case in normalized_cases])
    blockers.extend(f"duplicate_case_id:{item}" for item in duplicate_ids)
    for case in normalized_cases:
        if not case.tlora_network_module.endswith("tlora"):
            blockers.append(f"{case.case_id}:tlora_network_module_not_tlora")
        if case.baseline_network_module == case.tlora_network_module:
            blockers.append(f"{case.case_id}:baseline_and_tlora_modules_match")
    root = str(output_root)
    return {
        "schema_version": 1,
        "manifest": "tlora_ab_runner_manifest_v0",
        "ok": not blockers,
        "runner_ready": not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "output_root": root,
        "thresholds": _threshold_dict(thresholds),
        "case_count": len(normalized_cases),
        "cases": [
            {
                **case.as_dict(),
                "baseline_result_path": f"{root}/{case.case_id}/baseline_result.json",
                "tlora_result_path": f"{root}/{case.case_id}/tlora_result.json",
                "scorecard_path": f"{root}/{case.case_id}/scorecard.json",
            }
            for case in normalized_cases
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": "run manifest cases through baseline and T-LoRA trainer paths",
    }


def build_tlora_ab_result_gate(
    manifest: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    *,
    thresholds: TLoRASingleImageBenchmarkThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = {str(case.get("case_id")): dict(case) for case in manifest.get("cases", []) if isinstance(case, Mapping)}
    results = {str(item.get("case_id")): dict(item) for item in case_results if item.get("case_id")}
    scorecards = []
    blockers: list[str] = []
    for case_id in sorted(cases):
        result = results.get(case_id)
        if result is None:
            blockers.append(f"{case_id}:result_missing")
            continue
        scorecard = build_tlora_single_image_benchmark_scorecard(
            baseline=result.get("baseline"),
            tlora=result.get("tlora"),
            result_payload=result,
            thresholds=thresholds or manifest.get("thresholds"),
        )
        scorecard["case_id"] = case_id
        scorecards.append(scorecard)
        if not scorecard.get("ok"):
            blockers.extend(f"{case_id}:{reason}" for reason in scorecard.get("blocked_reasons", []))
    extra_results = sorted(case_id for case_id in results if case_id not in cases)
    blockers.extend(f"unexpected_result:{case_id}" for case_id in extra_results)
    ready = bool(cases) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_result_gate_v0",
        "ok": ready,
        "ab_result_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "case_count": len(cases),
        "result_count": len(results),
        "scorecards": scorecards,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "review representative T-LoRA quality before trainer promotion"
            if ready
            else "complete all manifest cases and resolve benchmark blockers"
        ),
    }


def _case(value: TLoRAABCase | Mapping[str, Any]) -> TLoRAABCase:
    if isinstance(value, TLoRAABCase):
        return value.normalized()
    if isinstance(value, Mapping):
        return TLoRAABCase(**value).normalized()
    raise TypeError("T-LoRA A/B case must be TLoRAABCase or Mapping")


def _threshold_dict(thresholds: TLoRASingleImageBenchmarkThresholds | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(thresholds, TLoRASingleImageBenchmarkThresholds):
        return thresholds.normalized().__dict__
    if isinstance(thresholds, Mapping):
        return TLoRASingleImageBenchmarkThresholds(**thresholds).normalized().__dict__
    return TLoRASingleImageBenchmarkThresholds().normalized().__dict__


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


__all__ = [
    "TLoRAABCase",
    "build_tlora_ab_result_gate",
    "build_tlora_ab_runner_manifest",
]
