"""Plan native multi-batch long-window stability probes for Lulynx.

This module builds a dry-run plan from manifest/runtime evidence. It does not
start training; commands are intended for a protected runner or manual review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LULYNX_MULTI_BATCH_STABILITY_MATRIX_PLAN = "lulynx_multi_batch_stability_matrix_plan_v0"
DEFAULT_BATCH_CANDIDATES = (2, 4, 8)


def build_lulynx_multi_batch_stability_matrix_plan(
    runtime_features: Mapping[str, Any],
    *,
    repo_root: Path | str,
    out_root: Path | str,
    python_executable: Path | str | None = None,
    family: str | None = None,
    batch_candidates: Sequence[int] = DEFAULT_BATCH_CANDIDATES,
    steps: int = 80,
    warmup: int = 10,
    samples: int = 64,
    resolution: int = 1024,
) -> dict[str, Any]:
    features = runtime_features if isinstance(runtime_features, Mapping) else {}
    gate = _mapping(features.get("multi_batch_promotion_gate"))
    execution_strategy = _mapping(gate.get("execution_strategy"))
    execution_strategy_gate = _mapping(gate.get("execution_strategy_gate"))
    dataloader = _mapping(features.get("multi_batch_dataloader"))
    training_loop_runtime = _mapping(features.get("training_loop_runtime"))
    trace = _mapping(training_loop_runtime.get("training_pipeline_trace") or features.get("training_pipeline_trace"))
    candidate_from_gate = _as_int(gate.get("candidate_physical_batch_size"), 0)
    family_key = _resolve_family(family, features)
    repo = Path(repo_root)
    out = Path(out_root)
    python_path = Path(python_executable) if python_executable is not None else repo / "backend" / "env" / "python-flashattention" / "python.exe"
    candidates = [_candidate_plan(
        batch_size=batch,
        family=family_key,
        repo_root=repo,
        out_root=out,
        python_executable=python_path,
        gate=gate,
        execution_strategy=execution_strategy,
        execution_strategy_gate=execution_strategy_gate,
        dataloader=dataloader,
        candidate_from_gate=candidate_from_gate,
        steps=steps,
        warmup=warmup,
        samples=samples,
        resolution=resolution,
    ) for batch in _normalize_candidates(batch_candidates)]
    ready_count = sum(1 for item in candidates if item["status"] == "ready_to_schedule")
    blocked_count = sum(1 for item in candidates if item["status"] != "ready_to_schedule")
    return {
        "schema_version": 1,
        "report": LULYNX_MULTI_BATCH_STABILITY_MATRIX_PLAN,
        "status": "ready_with_candidates" if ready_count else "blocked_no_ready_candidates",
        "safe_to_auto_start": False,
        "requires_gpu": True,
        "release_claim_allowed": False,
        "family": family_key,
        "candidate_count": len(candidates),
        "ready_candidate_count": ready_count,
        "blocked_candidate_count": blocked_count,
        "promotion_gate_status": str(gate.get("status") or "missing"),
        "trace_status": str(trace.get("status") or "missing"),
        "dataloader_contract_status": str(gate.get("dataloader_contract_status") or ("ok" if dataloader.get("ok") else "missing")),
        "execution_strategy": dict(execution_strategy),
        "execution_strategy_gate": dict(execution_strategy_gate),
        "candidates": candidates,
        "next_actions": _next_actions(ready_count=ready_count, blocked_count=blocked_count),
    }


def _candidate_plan(
    *,
    batch_size: int,
    family: str,
    repo_root: Path,
    out_root: Path,
    python_executable: Path,
    gate: Mapping[str, Any],
    execution_strategy: Mapping[str, Any],
    execution_strategy_gate: Mapping[str, Any],
    dataloader: Mapping[str, Any],
    candidate_from_gate: int,
    steps: int,
    warmup: int,
    samples: int,
    resolution: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    cautions: list[str] = []
    if not bool(gate.get("ready_for_long_window_probe")):
        blockers.append("promotion_gate_not_ready")
        blockers.extend(_string_list(gate.get("blockers")))
    strategy_name = str(execution_strategy.get("strategy") or "")
    if not strategy_name:
        blockers.append("missing_multi_batch_execution_strategy")
    elif strategy_name != "native_batch_forward":
        blockers.append("execution_strategy_not_native_batch_forward")
    if bool(execution_strategy.get("diagnostic_only")):
        blockers.append("diagnostic_execution_strategy_cannot_schedule_release_probe")
    gate_status = str(execution_strategy_gate.get("status") or "")
    if not gate_status:
        blockers.append("missing_multi_batch_execution_strategy_gate")
    elif gate_status not in {"ready_behind_disabled_internal_gate", "ready_to_route_strategy"}:
        blockers.append("execution_strategy_gate_not_ready")
    if candidate_from_gate and batch_size != candidate_from_gate:
        blockers.append("needs_fresh_gate_evidence_for_candidate_batch_size")
    if dataloader and _as_int(dataloader.get("physical_batch_size"), 0) not in {0, batch_size}:
        blockers.append("dataloader_contract_batch_size_mismatch")
    if dataloader and not bool(dataloader.get("drop_last")):
        blockers.append("drop_last_required_for_static_tail_batch_gate")
    cautions.extend(_string_list(gate.get("cautions")))
    case_id = f"{family}_batch{batch_size}_long_window_stability_probe"
    case_out = out_root / case_id
    command = _benchmark_command(
        python_executable=python_executable,
        repo_root=repo_root,
        family=family,
        batch_size=batch_size,
        steps=steps,
        warmup=warmup,
        samples=samples,
        resolution=resolution,
        out_dir=case_out,
    )
    return {
        "id": case_id,
        "family": family,
        "physical_batch_size": batch_size,
        "status": "ready_to_schedule" if not blockers else "blocked",
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "requires_gpu": True,
        "blockers": _dedupe(blockers),
        "cautions": _dedupe(cautions),
        "execution_strategy": dict(execution_strategy),
        "execution_strategy_gate": dict(execution_strategy_gate),
        "config_overlay": {"train_batch_size": batch_size, "compile_static_shape_drop_last": True},
        "dry_run_command": [*command, "--dry-run"],
        "protected_command": command,
        "expected_reports": {
            "gpu_bubble_report": str(case_out / "gpu_bubble_experiment_report.json"),
            "benchmark_summary": str(case_out / f"{family}_summary.json"),
            "run_manifest_glob": str(case_out / family / "standard" / "output" / "run_manifest.json"),
        },
        "evidence_requirements": [
            "completed_run_manifest",
            "training_pipeline_trace_completed",
            "multi_batch_promotion_gate_ready_before_run",
            "steady_samples_per_second",
            "loss_stability_guard",
            "peak_vram_guard",
            "active_gpu_window",
            "failure_stage_if_failed",
        ],
    }


def _benchmark_command(
    *,
    python_executable: Path,
    repo_root: Path,
    family: str,
    batch_size: int,
    steps: int,
    warmup: int,
    samples: int,
    resolution: int,
    out_dir: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "gpu_bubble_experiment.py"),
        "--out-dir",
        str(out_dir),
        "--",
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "native_runtime_profile_benchmark.py"),
        "--family",
        family,
        "--profiles",
        "standard",
        "--steps",
        str(max(int(steps), 1)),
        "--steady-warmup",
        str(max(int(warmup), 0)),
        "--samples",
        str(max(int(samples), 1)),
        "--resolution",
        str(max(int(resolution), 1)),
        "--network-dim",
        "1",
        "--train-batch-size",
        str(max(int(batch_size), 1)),
        "--phase-profile",
        "--out",
        str(out_dir),
    ]


def _resolve_family(family: str | None, features: Mapping[str, Any]) -> str:
    if family:
        return _family_key(family)
    acceleration = _mapping(features.get("model_acceleration"))
    if acceleration.get("family"):
        return _family_key(acceleration.get("family"))
    return "sdxl"


def _next_actions(*, ready_count: int, blocked_count: int) -> list[str]:
    if ready_count:
        actions = ["review_ready_batchN_candidates", "run_ready_candidates_via_protected_runner"]
        if blocked_count:
            actions.append("collect_fresh_gate_evidence_for_blocked_candidates")
        return actions
    return [
        "fix_multi_batch_promotion_gate_blockers",
        "collect_fresh_batchN_pipeline_trace_before_matrix",
        "require_native_batch_forward_strategy_before_long_window_probe",
    ]


def _normalize_candidates(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        batch = max(_as_int(value, 0), 0)
        if batch > 1 and batch not in result:
            result.append(batch)
    return result or list(DEFAULT_BATCH_CANDIDATES)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]


def _family_key(raw: Any) -> str:
    family = str(raw or "").strip().lower().replace("-", "_")
    return "newbie" if family == "dit" else family or "sdxl"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "DEFAULT_BATCH_CANDIDATES",
    "LULYNX_MULTI_BATCH_STABILITY_MATRIX_PLAN",
    "build_lulynx_multi_batch_stability_matrix_plan",
]
