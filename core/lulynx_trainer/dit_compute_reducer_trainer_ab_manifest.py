"""Real-trainer A/B manifest for DiT compute reducer candidates.

The manifest is the handoff between cached-token reducer evidence and a later
operator-run trainer A/B. It records concrete cases and result expectations,
but does not dispatch runs, register request fields, or enable any runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REDUCERS = ("tread", "diffcr", "blockskip", "local_window_attention")
REQUIRED_RESULT_FIELDS = (
    "case_id",
    "reducer_id",
    "family",
    "baseline_step_time_ms",
    "candidate_step_time_ms",
    "baseline_peak_vram_mb",
    "candidate_peak_vram_mb",
    "quality_drift",
    "loss_delta",
    "steady_state_steps",
    "cache_first",
    "native_dit",
)


@dataclass(frozen=True)
class DiTComputeReducerTrainerABCase:
    case_id: str
    reducer_id: str
    family: str = "anima"
    max_train_steps: int = 40
    warmup_steps: int = 5
    resolution: int = 1024
    batch_size: int = 1
    seed: int = 20260605
    baseline_runtime_profile: str = "native_dit_cache_first"
    candidate_runtime_profile: str = "native_dit_cache_first_reducer_probe"

    def normalized(self) -> "DiTComputeReducerTrainerABCase":
        reducer = str(self.reducer_id or "").strip().lower()
        family = str(self.family or "anima").strip().lower()
        if family not in {"anima", "newbie"}:
            family = "anima"
        return DiTComputeReducerTrainerABCase(
            case_id=str(self.case_id or f"{family}-{reducer}-trainer-ab").strip(),
            reducer_id=reducer,
            family=family,
            max_train_steps=max(int(self.max_train_steps or 1), 1),
            warmup_steps=max(int(self.warmup_steps or 0), 0),
            resolution=max(int(self.resolution or 1), 1),
            batch_size=max(int(self.batch_size or 1), 1),
            seed=max(int(self.seed or 0), 0),
            baseline_runtime_profile=str(self.baseline_runtime_profile or "native_dit_cache_first"),
            candidate_runtime_profile=str(
                self.candidate_runtime_profile or "native_dit_cache_first_reducer_probe"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reducer_id": self.reducer_id,
            "family": self.family,
            "max_train_steps": int(self.max_train_steps),
            "warmup_steps": int(self.warmup_steps),
            "resolution": int(self.resolution),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "baseline_runtime_profile": self.baseline_runtime_profile,
            "candidate_runtime_profile": self.candidate_runtime_profile,
        }


def build_dit_compute_reducer_trainer_ab_manifest(
    *,
    cached_token_matrix: Mapping[str, Any],
    output_root: str | Path = "temp/dit_compute_reducer_trainer_ab",
    families: Sequence[str] | None = None,
    max_reducers: int = 3,
    max_train_steps: int = 40,
    warmup_steps: int = 5,
    resolution: int = 1024,
    batch_size: int = 1,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    matrix = dict(cached_token_matrix)
    root = str(output_root)
    selected = _selected_reducers(matrix, max_reducers=max_reducers)
    requested_families = _families(families, matrix.get("family"))
    blockers = _matrix_blockers(matrix)
    if not selected:
        blockers.append("selected_reducers_missing")
    if not requested_families:
        blockers.append("families_missing")

    cases = [
        _case(
            reducer_id=reducer,
            family=family,
            max_train_steps=max_train_steps,
            warmup_steps=warmup_steps,
            resolution=resolution,
            batch_size=batch_size,
        )
        for family in requested_families
        for reducer in selected
    ]
    duplicate_ids = _duplicates(case.case_id for case in cases)
    blockers.extend(f"duplicate_case_id:{item}" for item in duplicate_ids)
    for case in cases:
        if case.reducer_id not in REDUCERS:
            blockers.append(f"{case.case_id}:unsupported_reducer")
        if case.warmup_steps >= case.max_train_steps:
            blockers.append(f"{case.case_id}:warmup_steps_must_be_less_than_max_train_steps")

    ready = not blockers
    return {
        "schema_version": 1,
        "manifest": "dit_compute_reducer_trainer_ab_manifest_v0",
        "scorecard": "dit_compute_reducer_trainer_ab_manifest_v0",
        "ok": ready,
        "runner_ready": ready,
        "trainer_ab_manifest_ready": ready,
        "source_scorecard": str(matrix.get("scorecard") or ""),
        "selected_reducers": list(selected),
        "families": list(requested_families),
        "case_count": len(cases),
        "output_root": root,
        "required_metrics": [
            "step_time_ms",
            "peak_vram_mb",
            "quality_drift",
            "loss_delta",
            "steady_state_steps",
            "cache_first",
            "native_dit",
        ],
        "thresholds": _thresholds(thresholds),
        "cases": [_case_payload(case, root) for case in cases],
        "trainer_ab_execution_allowed": False,
        "trainer_ab_execution_started": False,
        "trainer_ab_execution_completed": False,
        "ab_execution_allowed": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run these cases through the normal trainer and ingest result summaries"
            if ready
            else "complete cached-token matrix prerequisites before trainer A/B"
        ),
    }


def build_dit_compute_reducer_trainer_ab_result_gate(
    *,
    manifest: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(manifest)
    results = [dict(item) for item in case_results if isinstance(item, Mapping)]
    limits = _thresholds(thresholds or plan.get("thresholds"))
    manifest_cases = {
        str(case.get("case_id") or ""): dict(case)
        for case in plan.get("cases", ())
        if isinstance(case, Mapping) and str(case.get("case_id") or "")
    }
    result_by_case = {
        str(item.get("case_id") or ""): item
        for item in results
        if str(item.get("case_id") or "")
    }
    blockers: list[str] = []
    if plan.get("scorecard") != "dit_compute_reducer_trainer_ab_manifest_v0":
        blockers.append("unexpected_trainer_ab_manifest")
    if not bool(plan.get("trainer_ab_manifest_ready", plan.get("ok", False))):
        blockers.append("trainer_ab_manifest_not_ready")
    if _unsafe_flags(plan):
        blockers.append("unsafe_manifest_flag")
    if not manifest_cases:
        blockers.append("manifest_cases_missing")
    if not results:
        blockers.append("case_results_missing")

    rows = [_result_row(case, result_by_case.get(case_id), limits) for case_id, case in manifest_cases.items()]
    blockers.extend(f"{row['case_id']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    unexpected = sorted(case_id for case_id in result_by_case if case_id not in manifest_cases)
    blockers.extend(f"unexpected_result:{case_id}" for case_id in unexpected)
    if any(_unsafe_flags(item) for item in results):
        blockers.append("unsafe_result_summary_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_result_gate_v0",
        "ok": ready,
        "trainer_ab_result_ready": ready,
        "case_count": len(manifest_cases),
        "result_count": len(results),
        "passed_case_ids": [row["case_id"] for row in rows if row["ok"]],
        "passed_reducers": sorted({row["reducer_id"] for row in rows if row["ok"]}),
        "result_rows": rows,
        "result_summaries_for_ingestion": [_ingestion_summary(row) for row in rows if row["ok"]],
        "thresholds": limits,
        "trainer_ab_execution_allowed": False,
        "ab_execution_allowed": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed passed reducer summaries into compute-reducer A/B result ingestion"
            if ready
            else "collect complete passing trainer A/B result summaries"
        ),
    }


def _case(
    *,
    reducer_id: str,
    family: str,
    max_train_steps: int,
    warmup_steps: int,
    resolution: int,
    batch_size: int,
) -> DiTComputeReducerTrainerABCase:
    return DiTComputeReducerTrainerABCase(
        case_id=f"{family}-{reducer_id}-trainer-ab",
        reducer_id=reducer_id,
        family=family,
        max_train_steps=max_train_steps,
        warmup_steps=warmup_steps,
        resolution=resolution,
        batch_size=batch_size,
    ).normalized()


def _case_payload(case: DiTComputeReducerTrainerABCase, root: str) -> dict[str, Any]:
    payload = case.as_dict()
    case_root = f"{root}/{case.case_id}"
    payload.update(
        {
            "baseline_result_path": f"{case_root}/baseline_result.json",
            "candidate_result_path": f"{case_root}/candidate_result.json",
            "scorecard_path": f"{case_root}/trainer_ab_scorecard.json",
            "cache_first_required": True,
            "native_dit_required": True,
            "baseline_request_overrides": {
                "native_runtime_profile": case.baseline_runtime_profile,
                "cache_first": True,
                "compute_reducer": "disabled",
            },
            "candidate_request_overrides": {
                "native_runtime_profile": case.candidate_runtime_profile,
                "cache_first": True,
                "compute_reducer": case.reducer_id,
                "compute_reducer_mode": "report_only_candidate",
            },
        }
    )
    return payload


def _result_row(case: Mapping[str, Any], result: Mapping[str, Any] | None, limits: Mapping[str, float]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "")
    reducer_id = str(case.get("reducer_id") or "")
    family = str(case.get("family") or "")
    if result is None:
        return {
            "case_id": case_id,
            "reducer_id": reducer_id,
            "family": family,
            "ok": False,
            "step_time_improvement": 0.0,
            "vram_delta_fraction": 0.0,
            "quality_drift": None,
            "loss_delta": None,
            "blocked_reasons": ["result_missing"],
        }
    blockers: list[str] = []
    for name in REQUIRED_RESULT_FIELDS:
        if name not in result:
            blockers.append(f"result_field_missing:{name}")
    if str(result.get("reducer_id") or "") != reducer_id:
        blockers.append("reducer_id_mismatch")
    if str(result.get("family") or "") != family:
        blockers.append("family_mismatch")
    baseline_step = _positive_float(result.get("baseline_step_time_ms"))
    candidate_step = _positive_float(result.get("candidate_step_time_ms"))
    baseline_vram = _positive_float(result.get("baseline_peak_vram_mb"))
    candidate_vram = _positive_float(result.get("candidate_peak_vram_mb"))
    quality_drift = _float_or_none(result.get("quality_drift"))
    loss_delta = _float_or_none(result.get("loss_delta"))
    steady_steps = int(_positive_float(result.get("steady_state_steps")))
    step_improvement = 0.0 if baseline_step <= 0 else (baseline_step - candidate_step) / baseline_step
    vram_delta = 0.0 if baseline_vram <= 0 else (candidate_vram - baseline_vram) / baseline_vram
    min_steps = max(int(case.get("max_train_steps") or 1) - int(case.get("warmup_steps") or 0), 1)
    if baseline_step <= 0 or candidate_step <= 0:
        blockers.append("step_time_invalid")
    elif step_improvement < limits["min_step_time_improvement"]:
        blockers.append("step_time_improvement_below_threshold")
    if baseline_vram <= 0 or candidate_vram <= 0:
        blockers.append("peak_vram_invalid")
    elif vram_delta > limits["max_vram_regression"]:
        blockers.append("vram_regression_above_threshold")
    if quality_drift is None:
        blockers.append("quality_drift_missing")
    elif quality_drift > limits["max_quality_drift"]:
        blockers.append("quality_drift_above_threshold")
    if loss_delta is None:
        blockers.append("loss_delta_missing")
    elif loss_delta > limits["max_loss_delta"]:
        blockers.append("loss_delta_above_threshold")
    if steady_steps < min_steps:
        blockers.append("steady_state_steps_below_manifest")
    if result.get("cache_first") is not True:
        blockers.append("cache_first_missing")
    if result.get("native_dit") is not True:
        blockers.append("native_dit_missing")
    if _unsafe_flags(result):
        blockers.append("unsafe_result_flag")
    return {
        "case_id": case_id,
        "reducer_id": reducer_id,
        "family": family,
        "ok": not blockers,
        "baseline_step_time_ms": baseline_step,
        "candidate_step_time_ms": candidate_step,
        "step_time_improvement": float(step_improvement),
        "baseline_peak_vram_mb": baseline_vram,
        "candidate_peak_vram_mb": candidate_vram,
        "vram_delta_fraction": float(vram_delta),
        "quality_drift": quality_drift,
        "loss_delta": loss_delta,
        "steady_state_steps": steady_steps,
        "blocked_reasons": blockers,
    }


def _ingestion_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reducer_id": str(row.get("reducer_id") or ""),
        "baseline_step_time_ms": float(row.get("baseline_step_time_ms") or 0.0),
        "candidate_step_time_ms": float(row.get("candidate_step_time_ms") or 0.0),
        "baseline_peak_vram_mb": float(row.get("baseline_peak_vram_mb") or 0.0),
        "candidate_peak_vram_mb": float(row.get("candidate_peak_vram_mb") or 0.0),
        "quality_drift": float(row.get("quality_drift") or 0.0),
        "loss_delta": float(row.get("loss_delta") or 0.0),
    }


def _selected_reducers(matrix: Mapping[str, Any], *, max_reducers: int) -> tuple[str, ...]:
    rows = {
        str(row.get("reducer_id") or ""): row
        for row in matrix.get("rows", ())
        if isinstance(row, Mapping) and str(row.get("reducer_id") or "")
    }
    ranked = tuple(str(item) for item in matrix.get("ranked_candidate_ids", ()) if str(item) in rows)
    selected = [
        reducer for reducer in ranked
        if reducer in REDUCERS
        and bool(rows[reducer].get("cached_token_ab_ready", rows[reducer].get("ok", False)))
        and not list(rows[reducer].get("blocked_reasons") or [])
    ]
    return tuple(selected[:max(max_reducers, 0)])


def _families(requested: Sequence[str] | None, fallback: Any) -> tuple[str, ...]:
    values = requested if requested is not None else (fallback or "anima",)
    result: list[str] = []
    for value in values:
        family = str(value or "").strip().lower()
        if family in {"anima", "newbie"} and family not in result:
            result.append(family)
    return tuple(result)


def _matrix_blockers(matrix: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if matrix.get("scorecard") != "dit_compute_reducer_cached_token_ab_matrix_v0":
        blockers.append("unexpected_cached_token_ab_matrix")
    if not bool(matrix.get("ok", False)):
        blockers.append("cached_token_ab_matrix_not_ready")
    if _unsafe_flags(matrix):
        blockers.append("unsafe_cached_token_ab_matrix_flag")
    return blockers


def _thresholds(value: Mapping[str, Any] | None) -> dict[str, float]:
    payload = dict(value or {})
    return {
        "min_step_time_improvement": _float_or_default(payload.get("min_step_time_improvement"), 0.05),
        "max_vram_regression": _float_or_default(payload.get("max_vram_regression"), 0.05),
        "max_quality_drift": _float_or_default(payload.get("max_quality_drift"), 0.01),
        "max_loss_delta": _float_or_default(payload.get("max_loss_delta"), 0.01),
    }


def _duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return sorted(dupes)


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "request_fields_emitted",
        "request_adapter_registered",
        "runtime_activation_enabled",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "trainer_ab_execution_allowed",
        "trainer_ab_execution_started",
        "trainer_ab_execution_completed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "DiTComputeReducerTrainerABCase",
    "build_dit_compute_reducer_trainer_ab_manifest",
    "build_dit_compute_reducer_trainer_ab_result_gate",
]
