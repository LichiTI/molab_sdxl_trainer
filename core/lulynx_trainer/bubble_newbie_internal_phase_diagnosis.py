"""JSON-only Newbie internal phase diagnosis for GPU bubble evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_internal_phase_diagnosis_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
DEFAULT_DATA_WAIT_THRESHOLD = 0.08
DEFAULT_TRAIN_STEP_COMPUTE_THRESHOLD = 0.9
DEFAULT_MIN_EXHAUSTED_PROBES = 2


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if text in {"dit", "newbie_dit"} else text


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _selected_run(summary: Mapping[str, Any], natural_evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    runs = _mapping(summary.get("runs"))
    preferred = str(natural_evidence.get("profile_label") or natural_evidence.get("run_label") or "standard")
    for key in (preferred, "standard"):
        run = _mapping(runs.get(key))
        if run:
            return run
    for value in runs.values():
        run = _mapping(value)
        if run:
            return run
    return {}


def _steady_bubble_profile(run: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = _mapping(run.get("runtime_feature_summary"))
    loop_runtime = _mapping(runtime.get("training_loop_runtime"))
    loop_phase = _mapping(loop_runtime.get("step_phase_profile"))
    direct_phase = _mapping(run.get("step_phase_profile"))
    return _first_mapping(
        run.get("steady_bubble_profile"),
        run.get("bubble_profile"),
        loop_phase.get("gpu_bubble_profile"),
        direct_phase.get("gpu_bubble_profile"),
        loop_phase,
        direct_phase,
    )


def _top_phase_rows(profile: Mapping[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    evidence = _mapping(profile.get("evidence"))
    rows = [_mapping(item) for item in _list(evidence.get("top_phases")) if _mapping(item)]
    if rows:
        return [
            {
                "label": str(row.get("label") or ""),
                "mean_ms": _round(row.get("mean_ms"), 4),
                "share": _round(row.get("share")),
            }
            for row in rows[:limit]
        ]

    phase_share = _mapping(profile.get("phase_share"))
    phase_mean = _mapping(profile.get("phase_mean_ms"))
    derived = [
        {
            "label": str(label),
            "mean_ms": _round(phase_mean.get(label), 4),
            "share": _round(share),
        }
        for label, share in phase_share.items()
    ]
    derived.sort(key=lambda item: item["share"], reverse=True)
    return derived[:limit]


def _train_step_compute_breakdown(profile: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "train_step_compute_substage."
    phase_mean = _mapping(profile.get("phase_mean_ms"))
    phase_share = _mapping(profile.get("phase_share"))
    substage_ms = {
        str(label)[len(prefix) :]: _safe_float(value)
        for label, value in phase_mean.items()
        if str(label).startswith(prefix)
    }
    substage_share = {
        str(label)[len(prefix) :]: _safe_float(value)
        for label, value in phase_share.items()
        if str(label).startswith(prefix)
    }
    total_ms = sum(substage_ms.values())
    dominant = max(substage_ms.items(), key=lambda item: item[1])[0] if substage_ms else ""
    return {
        "profile": "newbie_train_step_compute_substage_projection_v0",
        "source": "steady_bubble_profile.phase_mean_ms",
        "label_prefix": prefix,
        "available": bool(substage_ms),
        "profiled_substage_count": len(substage_ms),
        "profiled_substage_labels": list(substage_ms),
        "profiled_substage_ms": {label: _round(value, 4) for label, value in substage_ms.items()},
        "profiled_substage_share": {label: _round(value) for label, value in substage_share.items()},
        "profiled_substage_total_ms": _round(total_ms, 4),
        "dominant_profiled_substage": dominant,
        "dominant_profiled_substage_share": _round(substage_share.get(dominant)),
        "runtime_default_change": False,
    }


def _newbie_backward_op_profile(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(run.get("runtime_feature_summary"))
    loop_runtime = _mapping(runtime.get("training_loop_runtime"))
    profile = _mapping(loop_runtime.get("newbie_backward_op_profile"))
    latest = _mapping(profile.get("latest"))
    top_ops = [_mapping(item) for item in _list(latest.get("top_ops")) if _mapping(item)]
    top_matmul_shapes = [
        _mapping(item) for item in _list(latest.get("top_matmul_shape_groups")) if _mapping(item)
    ]
    top_op = top_ops[0] if top_ops else {}
    top_matmul_shape = top_matmul_shapes[0] if top_matmul_shapes else {}
    return {
        "profile": "newbie_backward_op_profile_projection_v0",
        "available": bool(profile and latest and top_ops),
        "shape_profile_available": bool(latest.get("record_shapes", False) and top_matmul_shapes),
        "sample_count": _safe_int(profile.get("sample_count")),
        "latest_status": str(latest.get("status") or ""),
        "sort_key": str(latest.get("sort_key") or ""),
        "top_op_key": str(top_op.get("key") or ""),
        "shape_group_count": _safe_int(latest.get("shape_group_count")),
        "top_matmul_shape_key": str(top_matmul_shape.get("key") or ""),
        "top_matmul_shape": str(top_matmul_shape.get("input_shapes") or ""),
        "top_ops": [
            {
                "key": str(row.get("key") or ""),
                "count": _safe_int(row.get("count")),
                "self_cuda_ms": _round(row.get("self_cuda_ms"), 4),
                "cuda_ms": _round(row.get("cuda_ms"), 4),
                "self_cpu_ms": _round(row.get("self_cpu_ms"), 4),
                "cpu_ms": _round(row.get("cpu_ms"), 4),
            }
            for row in top_ops[:8]
        ],
        "top_matmul_shape_groups": [
            {
                "key": str(row.get("key") or ""),
                "input_shapes": str(row.get("input_shapes") or ""),
                "count": _safe_int(row.get("count")),
                "self_cuda_ms": _round(row.get("self_cuda_ms"), 4),
                "cuda_ms": _round(row.get("cuda_ms"), 4),
                "self_cpu_ms": _round(row.get("self_cpu_ms"), 4),
                "cpu_ms": _round(row.get("cpu_ms"), 4),
            }
            for row in top_matmul_shapes[:8]
        ],
        "runtime_default_change": False,
    }


def _newbie_module_timing_profile(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(run.get("runtime_feature_summary"))
    loop_runtime = _mapping(runtime.get("training_loop_runtime"))
    profile = _mapping(loop_runtime.get("newbie_module_timing_profile"))
    latest = _mapping(profile.get("latest"))
    top_groups = [_mapping(item) for item in _list(latest.get("top_groups")) if _mapping(item)]
    top_group = top_groups[0] if top_groups else {}
    return {
        "profile": "newbie_module_timing_profile_projection_v0",
        "available": bool(profile and latest and top_groups),
        "sample_count": _safe_int(profile.get("sample_count")),
        "latest_status": str(latest.get("status") or ""),
        "tracked_module_count": _safe_int(latest.get("tracked_module_count")),
        "group_count": _safe_int(latest.get("group_count")),
        "top_group": str(top_group.get("group") or ""),
        "top_group_backward_cuda_ms": _round(top_group.get("backward_cuda_ms"), 4),
        "top_group_forward_cuda_ms": _round(top_group.get("forward_cuda_ms"), 4),
        "top_groups": [
            {
                "group": str(row.get("group") or ""),
                "module_count": _safe_int(row.get("module_count")),
                "forward_count": _safe_int(row.get("forward_count")),
                "backward_count": _safe_int(row.get("backward_count")),
                "forward_cuda_ms": _round(row.get("forward_cuda_ms"), 4),
                "backward_cuda_ms": _round(row.get("backward_cuda_ms"), 4),
                "forward_cpu_ms": _round(row.get("forward_cpu_ms"), 4),
                "backward_cpu_ms": _round(row.get("backward_cpu_ms"), 4),
                "module_name_examples": [str(item) for item in _list(row.get("module_name_examples"))[:5]],
            }
            for row in top_groups[:8]
        ],
        "runtime_default_change": False,
    }


def _triton_ops_runtime(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(run.get("runtime_feature_summary"))
    profile = _mapping(runtime.get("triton_ops_runtime"))
    return {
        "profile": "triton_ops_runtime_projection_v0",
        "available": bool(profile),
        "enabled": bool(profile.get("enabled", False)),
        "requested": bool(profile.get("requested", False)),
        "status": str(profile.get("status") or ""),
        "dtype": str(profile.get("dtype") or ""),
        "patched_lora_layers": _safe_int(profile.get("patched_lora_layers")),
        "patched_qkv_blocks": _safe_int(profile.get("patched_qkv_blocks")),
        "patched_adaln_blocks": _safe_int(profile.get("patched_adaln_blocks")),
        "inject_lora": bool(profile.get("inject_lora", False)),
        "inject_qkv": bool(profile.get("inject_qkv", False)),
        "inject_adaln": bool(profile.get("inject_adaln", False)),
        "fp32_backward": bool(profile.get("fp32_backward", False)),
        "runtime_default_change": False,
    }


def extract_newbie_phase_probe(
    summary: Mapping[str, Any],
    natural_evidence: Mapping[str, Any] | None = None,
    *,
    summary_path: str = "",
    natural_evidence_path: str = "",
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
    train_step_compute_threshold: float = DEFAULT_TRAIN_STEP_COMPUTE_THRESHOLD,
) -> dict[str, Any]:
    """Extract a compact Newbie phase probe row from benchmark summary JSON."""

    evidence = _mapping(natural_evidence)
    benchmark = _mapping(summary.get("benchmark"))
    run = _selected_run(summary, evidence)
    profile = _steady_bubble_profile(run)
    profile_evidence = _mapping(profile.get("evidence"))
    phase_share = _mapping(profile.get("phase_share"))
    phase_mean = _mapping(profile.get("phase_mean_ms"))
    metrics = _mapping(evidence.get("metrics"))
    matrix_axes = _mapping(_mapping(evidence.get("analysis")).get("matrix_axes"))
    decision = _mapping(evidence.get("decision"))

    data_wait_share = _safe_float(
        metrics.get("data_wait_share"),
        _safe_float(profile_evidence.get("data_wait_share"), _safe_float(phase_share.get("data_wait"))),
    )
    train_step_share = _safe_float(profile_evidence.get("train_step_share"), _safe_float(phase_share.get("train_step_total")))
    forward_share = _safe_float(phase_share.get("forward_total"))
    backward_share = _safe_float(phase_share.get("backward_total"))
    forward_backward_share = forward_share + backward_share
    train_step_breakdown = _train_step_compute_breakdown(profile)
    backward_op_profile = _newbie_backward_op_profile(run)
    module_timing_profile = _newbie_module_timing_profile(run)
    triton_ops_runtime = _triton_ops_runtime(run)
    runtime = _mapping(run.get("runtime_feature_summary"))
    adapter_runtime = _mapping(runtime.get("adapter_runtime"))
    newbie_target_scope = str(
        adapter_runtime.get("newbie_target_scope") or benchmark.get("newbie_target_scope") or ""
    ).strip().lower().replace("-", "_")
    dominant_bottleneck = str(metrics.get("dominant_bottleneck") or profile.get("dominant_bottleneck") or "")
    family = _family(evidence.get("family") or benchmark.get("family"))
    has_phase_profile = bool(profile)
    compute_dominates = bool(
        dominant_bottleneck == "compute_bound"
        or train_step_share >= train_step_compute_threshold
        or forward_backward_share >= train_step_compute_threshold
    )

    return {
        "case_id": str(evidence.get("case_id") or benchmark.get("case_id") or ""),
        "family": family,
        "summary_path": summary_path,
        "natural_evidence_path": natural_evidence_path,
        "status": str(evidence.get("status") or ""),
        "steps_completed": _safe_int(evidence.get("steps_completed"), _safe_int(run.get("steps_completed"))),
        "train_batch_size": _safe_int(
            matrix_axes.get("train_batch_size"),
            _safe_int(benchmark.get("train_batch_size")),
        ),
        "native_cache_mode": str(matrix_axes.get("native_cache_mode") or benchmark.get("native_cache_mode") or ""),
        "source_fixture": str(matrix_axes.get("source_fixture") or ""),
        "has_phase_profile": has_phase_profile,
        "dominant_bottleneck": dominant_bottleneck,
        "data_wait_share": _round(data_wait_share),
        "data_wait_below_threshold": bool(data_wait_share < float(data_wait_threshold)),
        "mean_step_ms": _round(profile.get("mean_step_ms"), 4),
        "steady_mean_step_ms": _round(run.get("steady_mean_step_ms"), 4),
        "step_count": _safe_int(profile.get("step_count")),
        "train_step_share": _round(train_step_share),
        "forward_total_share": _round(forward_share),
        "backward_total_share": _round(backward_share),
        "forward_backward_share": _round(forward_backward_share),
        "optimizer_share": _round(profile_evidence.get("optimizer_share"), 6),
        "h2d_transfer_share": _round(profile_evidence.get("h2d_transfer_share"), 6),
        "host_gap_share": _round(profile_evidence.get("host_gap_share"), 6),
        "data_wait_mean_ms": _round(phase_mean.get("data_wait"), 4),
        "forward_total_mean_ms": _round(phase_mean.get("forward_total"), 4),
        "backward_total_mean_ms": _round(phase_mean.get("backward_total"), 4),
        "train_step_total_mean_ms": _round(phase_mean.get("train_step_total"), 4),
        "compute_dominates": compute_dominates,
        "train_step_compute_substage_profile_available": bool(train_step_breakdown.get("available")),
        "train_step_compute_breakdown": train_step_breakdown,
        "newbie_backward_op_profile_available": bool(backward_op_profile.get("available")),
        "newbie_backward_op_profile": backward_op_profile,
        "newbie_module_timing_profile_available": bool(module_timing_profile.get("available")),
        "newbie_module_timing_profile": module_timing_profile,
        "newbie_target_scope": newbie_target_scope,
        "newbie_target_module_count": _safe_int(adapter_runtime.get("newbie_target_module_count")),
        "newbie_injected_layer_count": _safe_int(adapter_runtime.get("injected_layer_count")),
        "triton_ops_runtime_available": bool(triton_ops_runtime.get("available")),
        "triton_ops_runtime": triton_ops_runtime,
        "dataloader_rebuild_observed": bool(decision.get("dataloader_rebuild_observed")),
        "natural_candidate": bool(decision.get("natural_candidate")),
        "top_phases": _top_phase_rows(profile),
    }


def build_newbie_internal_phase_diagnosis(
    probes: Sequence[Mapping[str, Any]],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
    train_step_compute_threshold: float = DEFAULT_TRAIN_STEP_COMPUTE_THRESHOLD,
    min_exhausted_probes: int = DEFAULT_MIN_EXHAUSTED_PROBES,
) -> dict[str, Any]:
    """Classify Newbie natural-load blockers from existing phase probes."""

    rows = [dict(_mapping(row)) for row in probes if _family(_mapping(row).get("family")) == "newbie"]
    analyzed = [row for row in rows if bool(row.get("has_phase_profile"))]
    low_data_wait = [row for row in analyzed if bool(row.get("data_wait_below_threshold"))]
    compute_bound = [row for row in analyzed if bool(row.get("compute_dominates"))]
    dataloader_rebuild = [row for row in analyzed if bool(row.get("dataloader_rebuild_observed"))]
    natural_candidates = [row for row in analyzed if bool(row.get("natural_candidate"))]
    substage_profiled = [row for row in analyzed if bool(row.get("train_step_compute_substage_profile_available"))]
    backward_op_profiled = [row for row in analyzed if bool(row.get("newbie_backward_op_profile_available"))]
    backward_shape_profiled = [
        row
        for row in backward_op_profiled
        if bool(_mapping(row.get("newbie_backward_op_profile")).get("shape_profile_available"))
    ]
    module_timing_profiled = [row for row in analyzed if bool(row.get("newbie_module_timing_profile_available"))]
    triton_ops_profiled = [row for row in analyzed if bool(row.get("triton_ops_runtime_available"))]
    triton_lora_patched = [
        row
        for row in triton_ops_profiled
        if _safe_int(_mapping(row.get("triton_ops_runtime")).get("patched_lora_layers")) > 0
    ]
    exhausted = bool(
        len(low_data_wait) >= int(min_exhausted_probes)
        and len(compute_bound) >= int(min_exhausted_probes)
        and not dataloader_rebuild
        and not natural_candidates
    )
    status = "newbie_train_step_compute_bound" if exhausted else "insufficient_newbie_internal_phase_evidence"

    max_data_wait = max((_safe_float(row.get("data_wait_share")) for row in analyzed), default=0.0)
    min_train_step = min((_safe_float(row.get("train_step_share")) for row in analyzed), default=0.0)
    max_forward_backward = max((_safe_float(row.get("forward_backward_share")) for row in analyzed), default=0.0)

    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": "newbie",
        "status": status,
        "classification": status,
        "not_release_evidence": True,
        "fail_closed": True,
        "publishable": False,
        "claimable": False,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
        "data_wait_route_exhausted": exhausted,
        "release_claim": {
            "eligible": False,
            "reason": "diagnosis only; Newbie natural-load gate remains blocked pending semantic review or train-step follow-up evidence",
            "scope": "not_eligible",
        },
        "thresholds": {
            "data_wait_share": float(data_wait_threshold),
            "train_step_compute_share": float(train_step_compute_threshold),
            "min_exhausted_probes": int(min_exhausted_probes),
        },
        "probe_count": len(rows),
        "analyzed_probe_count": len(analyzed),
        "low_data_wait_probe_count": len(low_data_wait),
        "compute_bound_probe_count": len(compute_bound),
        "dataloader_rebuild_observed_count": len(dataloader_rebuild),
        "natural_candidate_count": len(natural_candidates),
        "train_step_compute_substage_profile_available_count": len(substage_profiled),
        "newbie_backward_op_profile_available_count": len(backward_op_profiled),
        "newbie_backward_shape_profile_available_count": len(backward_shape_profiled),
        "newbie_module_timing_profile_available_count": len(module_timing_profiled),
        "triton_ops_runtime_available_count": len(triton_ops_profiled),
        "triton_lora_patched_probe_count": len(triton_lora_patched),
        "summary": {
            "max_data_wait_share": _round(max_data_wait),
            "min_train_step_share": _round(min_train_step),
            "max_forward_backward_share": _round(max_forward_backward),
            "dominant_bottleneck_counts": _count_by(analyzed, "dominant_bottleneck"),
            "dominant_train_step_substage_counts": _dominant_substage_counts(analyzed),
            "newbie_backward_top_op_counts": _backward_top_op_counts(backward_op_profiled),
            "newbie_backward_top_matmul_shape_counts": _backward_top_matmul_shape_counts(backward_shape_profiled),
            "newbie_module_timing_top_group_counts": _module_timing_top_group_counts(module_timing_profiled),
            "newbie_target_scope_counts": _count_by(analyzed, "newbie_target_scope"),
            "triton_ops_status_counts": _triton_ops_status_counts(triton_ops_profiled),
        },
        "route_boundary": {
            "status": "newbie_steady_loop_uses_cached_payload_boundary",
            "data_wait_boundary": "steady_training_loader_iteration",
            "interpretation": (
                "Newbie raw decode and cache materialization are pre-step work; the observed steady loop "
                "is dominated by forward/backward train-step phases."
            ),
        },
        "next_actions": [
            {
                "id": "split_newbie_train_step_internal_phases",
                "kind": "diagnostic_followup",
                "requires_gpu_heavy_run": True,
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "reason": "forward_total and backward_total dominate the steady step while data_wait remains below threshold",
            },
            {
                "id": "review_newbie_natural_load_gate_semantics",
                "kind": "gate_semantics_review",
                "requires_gpu_heavy_run": False,
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "reason": "cache-first Newbie route may not be able to produce a natural DataLoader wait release canary",
            },
            {
                "id": "separate_newbie_cache_materialization_timing_boundary",
                "kind": "measurement_boundary_review",
                "requires_gpu_heavy_run": False,
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "reason": "raw/cache pressure happens before steady data_wait measurement",
            },
        ],
        "probes": rows,
    }


def _count_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dominant_substage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        breakdown = _mapping(row.get("train_step_compute_breakdown"))
        key = str(breakdown.get("dominant_profiled_substage") or "unprofiled")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _backward_top_op_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        profile = _mapping(row.get("newbie_backward_op_profile"))
        key = str(profile.get("top_op_key") or "unprofiled")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _backward_top_matmul_shape_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        profile = _mapping(row.get("newbie_backward_op_profile"))
        key = str(profile.get("top_matmul_shape") or "unprofiled")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _module_timing_top_group_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        profile = _mapping(row.get("newbie_module_timing_profile"))
        key = str(profile.get("top_group") or "unprofiled")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _triton_ops_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        profile = _mapping(row.get("triton_ops_runtime"))
        status = str(profile.get("status") or "unknown")
        patched = _safe_int(profile.get("patched_lora_layers"))
        key = f"{status}:lora_patched" if patched > 0 else status
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_diagnosis_from_json_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], str, str]],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
    train_step_compute_threshold: float = DEFAULT_TRAIN_STEP_COMPUTE_THRESHOLD,
    min_exhausted_probes: int = DEFAULT_MIN_EXHAUSTED_PROBES,
) -> dict[str, Any]:
    probes = [
        extract_newbie_phase_probe(
            summary,
            evidence,
            summary_path=summary_path,
            natural_evidence_path=evidence_path,
            data_wait_threshold=data_wait_threshold,
            train_step_compute_threshold=train_step_compute_threshold,
        )
        for summary, evidence, summary_path, evidence_path in pairs
    ]
    return build_newbie_internal_phase_diagnosis(
        probes,
        data_wait_threshold=data_wait_threshold,
        train_step_compute_threshold=train_step_compute_threshold,
        min_exhausted_probes=min_exhausted_probes,
    )


__all__ = [
    "REPORT",
    "ROADMAP",
    "build_diagnosis_from_json_pairs",
    "build_newbie_internal_phase_diagnosis",
    "extract_newbie_phase_probe",
]
