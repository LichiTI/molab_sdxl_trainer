from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is None or value == "":
            continue
        result[key] = value
    return result


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item is not None] if isinstance(value, list) else []


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _add_manifest_candidate(candidates: list[Path], value: Any) -> None:
    raw = str(value or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    candidates.append(path if path.name == "run_manifest.json" else path / "run_manifest.json")


def _collect_output_dir_candidates(*sources: Mapping[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for source in sources:
        mapped = _mapping(source)
        _add_manifest_candidate(candidates, mapped.get("output_dir"))
        _add_manifest_candidate(candidates, _mapping(mapped.get("paths")).get("output_dir"))
        _add_manifest_candidate(candidates, _mapping(mapped.get("raw_config")).get("output_dir"))
        normalized = _mapping(mapped.get("normalized_config"))
        _add_manifest_candidate(candidates, normalized.get("output_dir"))
    return candidates


def find_run_manifest(
    reader: Any,
    run_id: str,
    run_dir: Path,
    state: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    get_config_snapshot = getattr(reader, "get_config_snapshot", None)
    config_snapshot = get_config_snapshot(run_id) if callable(get_config_snapshot) else {}
    config_json = _read_json_file(run_dir / "config.json")
    candidates = _collect_output_dir_candidates(state, launch, config_snapshot, config_json)
    candidates.append(run_dir / "run_manifest.json")
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        manifest = _read_json_file(path)
        if manifest:
            return manifest, path
    return {}, None


def _compact_bubble_run(run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(run.get("metrics"))
    return {
        **_copy_present(run, ("source_kind", "case_id", "family", "status", "steps_completed")),
        "metrics": _copy_present(
            metrics,
            (
                "steady_samples_per_second",
                "throughput_estimated",
                "final_loss",
                "dominant_bottleneck",
                "mean_step_ms",
                "active_gpu_util_pct_mean",
                "active_gpu_saturated_sample_ratio",
                "peak_vram_mb",
                "memory_ratio",
            ),
        ),
    }


def compact_bubble_advisor_ab_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not evidence:
        return {}
    decision = _mapping(evidence.get("decision"))
    action = _mapping(evidence.get("action"))
    comparison = _mapping(evidence.get("comparison"))
    before = _mapping(evidence.get("before"))
    after = _mapping(evidence.get("after"))
    reasons = decision.get("reasons")
    return {
        "schema_version": int(_float_or_none(evidence.get("schema_version")) or 1),
        "report": str(evidence.get("report") or "bubble_advisor_ab_evidence_v0"),
        "status": str(evidence.get("status") or decision.get("status") or ""),
        "action": _copy_present(
            action,
            ("action_id", "status", "phase", "domain", "action_kind", "apply_scope", "applied_overlay"),
        ),
        "decision": {
            **_copy_present(decision, ("status", "recommended_action", "loss_regression_ratio")),
            "reasons": [str(item) for item in reasons[:8]] if isinstance(reasons, list) else [],
            "rollback_overlay": dict(_mapping(decision.get("rollback_overlay"))),
        },
        "comparison": _copy_present(
            comparison,
            (
                "steady_samples_per_second_before",
                "steady_samples_per_second_after",
                "steady_samples_per_second_delta",
                "steady_samples_per_second_gain_pct",
                "active_gpu_util_pct_delta",
                "peak_vram_mb_delta",
                "memory_ratio_delta",
                "final_loss_delta",
            ),
        ),
        "before": _compact_bubble_run(before),
        "after": _compact_bubble_run(after),
        "auto_pair": dict(_mapping(evidence.get("auto_pair"))),
    }


def _compact_bubble_closed_loop_action(action: Mapping[str, Any]) -> dict[str, Any]:
    if not action:
        return {}
    evaluation = _mapping(action.get("evaluation"))
    before = _mapping(evaluation.get("before"))
    after = _mapping(evaluation.get("after"))
    return {
        **_copy_present(
            action,
            (
                "schema_version",
                "ledger",
                "action_id",
                "status",
                "domain",
                "action_kind",
                "applied_step",
                "cooldown_until_step",
                "closed_step",
            ),
        ),
        "applied_overlay": dict(_mapping(action.get("applied_overlay"))),
        "rollback_restore": dict(_mapping(action.get("rollback_restore"))),
        "rollback_adapter": dict(_mapping(action.get("rollback_adapter"))),
        "rollback_applied_overlay": dict(_mapping(action.get("rollback_applied_overlay"))),
        "evaluation": {
            **_copy_present(
                evaluation,
                (
                    "steady_samples_per_second_gain_ratio",
                    "steady_samples_per_second_gain_pct",
                    "evaluated_step",
                ),
            ),
            "before": _copy_present(
                before,
                (
                    "steady_samples_per_second",
                    "active_gpu_util_pct_mean",
                    "host_gap_share",
                    "logging_checkpoint_share",
                    "final_loss",
                ),
            ),
            "after": _copy_present(
                after,
                (
                    "steady_samples_per_second",
                    "active_gpu_util_pct_mean",
                    "host_gap_share",
                    "logging_checkpoint_share",
                    "final_loss",
                ),
            ),
        },
    }


def _compact_bubble_closed_loop_runtime_apply(runtime_apply: Mapping[str, Any]) -> dict[str, Any]:
    if not runtime_apply:
        return {}
    return {
        **_copy_present(
            runtime_apply,
            (
                "adapter_id",
                "action_id",
                "status",
                "step",
                "cooldown_until_step",
                "diagnosis_kind",
            ),
        ),
        "applied_overlay": dict(_mapping(runtime_apply.get("applied_overlay"))),
        "rollback_restore": dict(_mapping(runtime_apply.get("rollback_restore"))),
        "rollback_adapter": dict(_mapping(runtime_apply.get("rollback_adapter"))),
        "skipped_mutations": [
            dict(item)
            for item in runtime_apply.get("skipped_mutations", [])
            if isinstance(item, Mapping)
        ][:8]
        if isinstance(runtime_apply.get("skipped_mutations"), list)
        else [],
    }


def compact_bubble_closed_loop_state(state: Mapping[str, Any], controller: Mapping[str, Any]) -> dict[str, Any]:
    closed_loop = _mapping(controller.get("closed_loop"))
    executor = _mapping(closed_loop.get("executor"))
    runtime_adapter = dict(_mapping(executor.get("runtime_adapter")))
    runtime_apply = _compact_bubble_closed_loop_runtime_apply(_mapping(executor.get("runtime_apply")))
    active_action = _compact_bubble_closed_loop_action(
        _mapping(state.get("active_action")) or _mapping(executor.get("active_action"))
    )
    raw_history = state.get("action_history")
    if not isinstance(raw_history, list):
        raw_history = closed_loop.get("action_history")
    if not isinstance(raw_history, list):
        raw_history = []
    history = [
        _compact_bubble_closed_loop_action(_mapping(item))
        for item in raw_history
        if isinstance(item, Mapping)
    ]
    history = [item for item in history if item]
    latest = history[-1] if history else active_action
    if not state and not history and not active_action and not executor:
        return {}
    return {
        "schema_version": 1,
        "report": "bubble_runtime_closed_loop_state_v0",
        "status": str(state.get("status") or closed_loop.get("status") or executor.get("status") or ""),
        "mode": str(closed_loop.get("mode") or executor.get("mode") or ""),
        "safe_to_auto_apply": bool(closed_loop.get("safe_to_auto_apply", executor.get("safe_to_auto_apply", False))),
        "can_apply_during_current_run": bool(
            closed_loop.get("can_apply_during_current_run", executor.get("can_apply_during_current_run", False))
        ),
        "candidate_action": str(closed_loop.get("candidate_action") or executor.get("candidate_action") or ""),
        "domain": str(executor.get("domain") or ""),
        "reason": str(executor.get("reason") or ""),
        "blocked_reasons": [
            str(item)
            for item in executor.get("blocked_reasons", [])
            if item is not None
        ][:8]
        if isinstance(executor.get("blocked_reasons"), list)
        else [],
        "active_action": active_action,
        "latest_action": latest,
        "runtime_adapter": runtime_adapter,
        "runtime_apply": runtime_apply,
        "action_history_count": len(history),
        "action_history": history[-5:],
        "evaluation": dict(_mapping(executor.get("evaluation"))),
        "rollback": dict(_mapping(executor.get("rollback"))),
    }


def _release_closed_copy(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result = _copy_present(source, keys)
    result["release_claim_allowed"] = False
    return result


def _first_runtime_mapping(key: str, *sources: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in sources:
        mapped = _mapping(source)
        for value in (
            mapped.get(key),
            _mapping(mapped.get("extra")).get(key),
            _mapping(_mapping(mapped.get("extra")).get("runtime_features")).get(key),
            _mapping(mapped.get("runtime_features")).get(key),
        ):
            if isinstance(value, Mapping):
                return value
    return {}


def _find_multi_batch_stability_candidate_evidence(*sources: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = (
        "multi_batch_stability_candidate_evidence",
        "multi_batch_stability_runner_candidate_evidence",
        "stability_candidate_evidence",
    )
    for key in keys:
        found = _first_runtime_mapping(key, *sources)
        if found:
            return found
    for source in sources:
        if str(_mapping(source).get("report") or "") == "lulynx_multi_batch_stability_candidate_evidence_v0":
            return source
        runner = _first_runtime_mapping("multi_batch_stability_matrix_runner", source)
        if not runner and str(_mapping(source).get("report") or "") == "lulynx_multi_batch_stability_matrix_runner_v0":
            runner = source
        candidates = runner.get("candidates") if isinstance(runner, Mapping) else []
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            evidence = _mapping(_mapping(candidate).get("evidence"))
            if evidence:
                return evidence
    return {}


def _compact_multi_batch_promotion_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    if not gate:
        return {}
    strategy = _mapping(gate.get("execution_strategy"))
    compact = _release_closed_copy(
        gate,
        ("status", "ready_for_long_window_probe", "dataloader_contract_status"),
    )
    compact.update(
        {
            "schema_version": int(_float_or_none(gate.get("schema_version")) or 1),
            "gate": str(gate.get("gate") or "lulynx_multi_batch_promotion_gate_v0"),
            "candidate_physical_batch_size": _int_or_none(gate.get("candidate_physical_batch_size")),
            "execution_strategy": _release_closed_copy(strategy, ("contract", "strategy", "diagnostic_only")),
            "missing_required_stage_ids": _string_list(gate.get("missing_required_stage_ids"))[:8],
            "missing_required_stage_plans": _string_list(gate.get("missing_required_stage_plans"))[:8],
            "blockers": _string_list(gate.get("blockers"))[:10],
            "cautions": _string_list(gate.get("cautions"))[:10],
            "recommended_next_actions": _string_list(gate.get("recommended_next_actions"))[:8],
        }
    )
    compact["execution_strategy"]["reasons"] = _string_list(strategy.get("reasons"))[:8]
    return compact


def _compact_multi_batch_dataloader(dataloader: Mapping[str, Any]) -> dict[str, Any]:
    if not dataloader:
        return {}
    compact = _release_closed_copy(
        dataloader,
        (
            "schema_version",
            "contract",
            "ok",
            "physical_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_world_size",
            "effective_batch_size",
            "uses_bucket_batch_sampler",
            "has_bucket_manager",
            "drop_last",
            "dataloader_route",
            "descriptor",
        ),
    )
    compact["warnings"] = _string_list(dataloader.get("warnings"))[:8]
    compact["recommended_next_checks"] = _string_list(dataloader.get("recommended_next_checks"))[:8]
    return compact


def _compact_multi_batch_stability_candidate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not evidence:
        return {}
    keys = (
        "schema_version",
        "report",
        "gpu_bubble_report_exists",
        "benchmark_summary_exists",
        "run_manifest_exists",
        "benchmark_summary_parsed",
        "run_manifest_parsed",
        "fresh_promotion_gate_status",
        "fresh_promotion_gate_ready",
        "fresh_dataloader_contract_status",
        "fresh_dataloader_drop_last",
        "gpu_bubble_return_code",
        "benchmark_summary_missing",
        "run_summary_count",
        "first_run_success",
        "steps_completed",
        "steady_samples_per_second",
        "peak_vram_mb",
        "final_loss",
        "classification_status",
        "active_gpu_util_pct_mean",
        "evidence_complete_for_review",
    )
    compact = _release_closed_copy(evidence, keys)
    compact["report"] = str(evidence.get("report") or "lulynx_multi_batch_stability_candidate_evidence_v0")
    compact["fresh_promotion_gate_blockers"] = _string_list(evidence.get("fresh_promotion_gate_blockers"))[:10]
    return compact


def compact_multi_batch_metadata(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    gate = _compact_multi_batch_promotion_gate(_first_runtime_mapping("multi_batch_promotion_gate", manifest, state))
    dataloader = _compact_multi_batch_dataloader(_first_runtime_mapping("multi_batch_dataloader", manifest, state))
    stability = _compact_multi_batch_stability_candidate_evidence(
        _find_multi_batch_stability_candidate_evidence(manifest, state)
    )
    result: dict[str, Any] = {}
    if gate:
        result["multi_batch_promotion_gate"] = gate
    if dataloader:
        result["multi_batch_dataloader"] = dataloader
    if stability:
        result["multi_batch_stability_candidate_evidence"] = stability
    return result


def compact_training_runtime_summary(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    controller = _first_runtime_mapping("bubble_controller", manifest, state)
    diagnosis = _mapping(controller.get("diagnosis"))
    snapshot = _mapping(controller.get("snapshot"))
    step_phase = _mapping(snapshot.get("step_phase"))
    gpu = _mapping(snapshot.get("gpu"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    action = _mapping(diagnosis.get("recommended_action"))

    if controller or step_phase:
        result = {
            "schema_version": 1,
            "report": "training_runtime_task_summary_v0",
            "source": "bubble_controller",
            "controller_mode": str(controller.get("mode") or ""),
            "controller_status": str(controller.get("status") or ""),
            "phase": str(controller.get("phase") or ""),
            "diagnosis_kind": str(diagnosis.get("kind") or step_phase.get("dominant_bottleneck") or "unknown"),
            "dominant_bottleneck": str(
                diagnosis.get("dominant_bottleneck") or step_phase.get("dominant_bottleneck") or "unknown"
            ),
            "confidence": _float_or_none(diagnosis.get("confidence")),
            "recommended_action_kind": str(action.get("kind") or ""),
            "recommended_action_reason": str(action.get("reason") or ""),
            "throughput_priority": True,
            "no_gpu_99_claim": True,
            "step_phase": _copy_present(
                step_phase,
                (
                    "bubble_ratio_estimate",
                    "data_wait_share",
                    "h2d_transfer_share",
                    "optimizer_share",
                    "host_gap_share",
                    "logging_checkpoint_share",
                    "train_step_share",
                    "mean_step_ms",
                    "steady_samples_per_second",
                    "throughput_estimated",
                    "window_step_count",
                    "train_step_ms",
                ),
            ),
            "gpu": _copy_present(
                gpu,
                (
                    "active_gpu_util_pct_mean",
                    "active_gpu_saturated_sample_ratio",
                    "active_gpu_idle_sample_ratio",
                    "memory_used_mb_max",
                    "memory_total_mb",
                ),
            ),
            "runtime": _copy_present(
                runtime,
                (
                    "workers",
                    "prefetch_factor",
                    "pin_memory",
                    "train_batch_size",
                    "gradient_accumulation_steps",
                    "data_transfer_non_blocking",
                    "offload_active",
                    "prefetch_enabled",
                    "prefetch_depth",
                    "prefetch_missed",
                    "optimizer_backend",
                ),
            ),
            "safety": _copy_present(safety, ("memory_ratio", "max_vram_ratio", "vram_safe")),
        }
        top_phases = step_phase.get("top_phases")
        if isinstance(top_phases, list):
            result["step_phase"]["top_phases"] = [
                dict(item) for item in top_phases if isinstance(item, Mapping)
            ][:6]
        return result

    summary = _first_runtime_mapping("runtime_feature_summary", manifest, state)
    loop = _mapping(summary.get("training_loop_runtime"))
    step_profile = _mapping(loop.get("step_phase_profile"))
    if not step_profile:
        return {}
    transfer = _mapping(step_profile.get("transfer"))
    return {
        "schema_version": 1,
        "report": "training_runtime_task_summary_v0",
        "source": "runtime_feature_summary",
        "diagnosis_kind": str(step_profile.get("dominant_bottleneck") or "unknown"),
        "dominant_bottleneck": str(step_profile.get("dominant_bottleneck") or "unknown"),
        "throughput_priority": True,
        "no_gpu_99_claim": True,
        "step_phase": {
            **_copy_present(
                step_profile,
                (
                    "bubble_ratio_estimate",
                    "data_wait_share",
                    "h2d_transfer_share",
                    "optimizer_share",
                    "host_gap_share",
                ),
            ),
            "transfer": _copy_present(
                transfer,
                ("step_share", "transfer_seconds", "mib", "ops", "recommendation"),
            ),
        },
    }


def build_training_runtime_task_metadata(
    reader: Any,
    run_id: str,
    run_dir: Path,
    state: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, manifest_path = find_run_manifest(reader, run_id, run_dir, state, launch)
    extra = _mapping(manifest.get("extra"))
    result: dict[str, Any] = {}

    compact_ab = compact_bubble_advisor_ab_evidence(_mapping(extra.get("bubble_advisor_ab_evidence")))
    if compact_ab:
        result["bubble_advisor_ab_evidence"] = compact_ab

    compact_closed_loop = compact_bubble_closed_loop_state(
        _mapping(extra.get("bubble_closed_loop_state")),
        _mapping(extra.get("bubble_controller")),
    )
    if compact_closed_loop:
        result["bubble_closed_loop_state"] = compact_closed_loop

    runtime_summary = compact_training_runtime_summary(manifest, state)
    if runtime_summary:
        result["training_runtime_summary"] = runtime_summary

    result.update(compact_multi_batch_metadata(manifest, state))
    if result and manifest_path is not None:
        result["run_manifest_path"] = str(manifest_path)
    return result


__all__ = [
    "build_training_runtime_task_metadata",
    "compact_bubble_advisor_ab_evidence",
    "compact_bubble_closed_loop_state",
    "compact_multi_batch_metadata",
    "compact_training_runtime_summary",
    "find_run_manifest",
]
