"""JSON-only rerun preflight for the Newbie tail8 seed2027 follow-up."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any


REPORT = "bubble_newbie_tail8_seed2027_rerun_preflight_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"
DEFAULT_MIN_DISK_FREE_GB = 20.0
TAIL8_SEED2027_CASE_ID = (
    "newbie_cache_first_full_latent_batch1_tail8_attention_long_window_seed2027_compute_probe"
)
TAIL8_SEED1337_REFERENCE_CASE_ID = (
    "newbie_cache_first_full_latent_batch1_tail8_attention_long_window_compute_probe"
)
TAIL8_SEED2027_BASELINE_CASE_ID = (
    "newbie_cache_first_full_latent_batch1_checkpoint_off_long_window_seed2027_compute_probe"
)
TAIL8_RERUN_OUT_DIR = (
    "devtools/benchmark_evidence/bubble_runtime/"
    "newbie_tail8_attention_long_window_seed2027_rerun_manual"
)
POST_RERUN_REFRESH_SEQUENCE = [
    "refresh_newbie_tail8_attention_compute_review",
    "refresh_newbie_tail8_forward_anomaly_review",
    "refresh_newbie_tail8_seed2027_rerun_preflight",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
]
ENVIRONMENT_SNAPSHOT_REQUIRED_KEYS = [
    "python_version",
    "platform",
    "cwd",
    "repo_root",
    "nvidia_smi_path",
    "timestamp_utc",
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _known_status(value: Any) -> str:
    return _clean_text(value or "missing")


def _clean_text(value: Any) -> str:
    text = str(value or "")
    return "".join(
        char if char in ("\n", "\r", "\t") or 32 <= ord(char) < 127 else "?"
        for char in text.replace("\ufffd", "?")
    )


def _clean_text_list(value: Any) -> list[str]:
    return [_clean_text(item) for item in _list(value)]


def _clean_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_clean_text(key): _clean_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_jsonish(item) for item in value]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _default_environment_snapshot() -> dict[str, Any]:
    return {key: "" for key in ENVIRONMENT_SNAPSHOT_REQUIRED_KEYS}


def _gpu_summary_from_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_clean_jsonish(dict(item)) for item in samples if isinstance(item, Mapping)]
    if not rows:
        return {
            "available": False,
            "sample_count": 0,
            "valid_sample_count": 0,
            "sample_error_count": 0,
            "reason": "no_samples",
        }
    valid_gpu_rows = [row for row in rows if any(key in row for key in ("gpu_util_pct", "memory_util_pct", "memory_used_mb", "temperature_gpu_c", "power_draw_w"))]
    sample_error_count = len(rows) - len(valid_gpu_rows)
    gpu_util = [_safe_float(row.get("gpu_util_pct")) for row in valid_gpu_rows]
    if not valid_gpu_rows or not any(math.isfinite(value) for value in gpu_util):
        return {
            "available": False,
            "sample_count": len(rows),
            "valid_sample_count": 0,
            "sample_error_count": sample_error_count,
            "reason": "no_valid_gpu_metrics",
            "raw_samples": rows,
        }
    memory_util = [_safe_float(row.get("memory_util_pct")) for row in valid_gpu_rows]
    memory_used = [_safe_float(row.get("memory_used_mb")) for row in valid_gpu_rows]
    temperature = [_safe_float(row.get("temperature_gpu_c")) for row in valid_gpu_rows]
    power = [_safe_float(row.get("power_draw_w")) for row in valid_gpu_rows]
    start = _safe_float(rows[0].get("monotonic_seconds"))
    end = _safe_float(rows[-1].get("monotonic_seconds"), start)
    active = [value for value in gpu_util if value >= 70.0]
    idle = [value for value in gpu_util if value < 20.0]
    return {
        "available": True,
        "sample_count": len(rows),
        "valid_sample_count": len(valid_gpu_rows),
        "sample_error_count": sample_error_count,
        "duration_seconds": _round(max(end - start, 0.0), 4),
        "gpu_name": _clean_text(valid_gpu_rows[-1].get("name")),
        "gpu_util_pct_mean": _round(sum(gpu_util) / len(gpu_util), 4),
        "gpu_util_pct_p95": _round(
            sorted(gpu_util)[max(min(int(round(len(gpu_util) * 0.95)) - 1, len(gpu_util) - 1), 0)],
            4,
        ),
        "gpu_util_pct_max": _round(max(gpu_util), 4),
        "gpu_active_sample_ratio": _round(len(active) / len(gpu_util), 6),
        "gpu_idle_sample_ratio": _round(len(idle) / len(gpu_util), 6),
        "memory_util_pct_mean": _round(sum(memory_util) / len(memory_util), 4),
        "memory_used_mb_max": _round(max(memory_used), 4),
        "temperature_gpu_c_max": _round(max(temperature), 4),
        "power_draw_w_mean": _round(sum(power) / len(power), 4),
        "raw_samples": rows,
    }


def _resource_summary(resource_report: Any) -> dict[str, Any]:
    if isinstance(resource_report, Mapping):
        report = dict(resource_report)
    else:
        disk = getattr(resource_report, "disk", None)
        report = {
            "disk": {
                "free_gb": getattr(disk, "free_gb", None),
                "total_gb": getattr(disk, "total_gb", None),
            }
            if disk is not None
            else None,
            "gpus": [
                {
                    "name": getattr(gpu, "name", ""),
                    "vram_total_mb": getattr(gpu, "vram_total_mb", 0),
                    "vram_free_mb": getattr(gpu, "vram_free_mb", 0),
                }
                for gpu in list(getattr(resource_report, "gpus", []) or [])
            ],
            "warnings": list(getattr(resource_report, "warnings", []) or []),
            "notes": list(getattr(resource_report, "notes", []) or []),
            "errors": list(getattr(resource_report, "errors", []) or []),
        }
    disk = _mapping(report.get("disk"))
    gpus = report.get("gpus") if isinstance(report.get("gpus"), list) else []
    return {
        "disk_free_gb": _round(disk.get("free_gb"), 2),
        "disk_total_gb": _round(disk.get("total_gb"), 2),
        "gpu_count": len(gpus),
        "warning_count": len(report.get("warnings") or []),
        "note_count": len(report.get("notes") or []),
        "error_count": len(report.get("errors") or []),
    }


def _compute_apps_probe_summary(compute_apps_probe: Any) -> dict[str, Any]:
    probe = _mapping(compute_apps_probe)
    rows = probe.get("rows") if isinstance(probe.get("rows"), list) else []
    row_count = _safe_int(probe.get("row_count"), len(rows))
    raw_line_count = _safe_int(probe.get("raw_line_count"), row_count)
    query_returncode = _safe_int(probe.get("query_returncode"))
    permission_denied = bool(probe.get("permission_denied"))
    query_ok = bool(probe.get("query_ok"))
    stderr = _clean_text(probe.get("stderr"))
    stdout = _clean_text(probe.get("stdout"))
    if not probe:
        query_ok = False
    inspection_ready = query_ok and not permission_denied
    return {
        "present": bool(probe),
        "command": _clean_text_list(probe.get("command")),
        "query_ok": query_ok,
        "inspection_ready": inspection_ready,
        "permission_denied": permission_denied,
        "query_returncode": query_returncode,
        "row_count": row_count,
        "raw_line_count": raw_line_count,
        "stdout": stdout,
        "stderr": stderr,
        "timestamp_utc": _clean_text(probe.get("timestamp_utc")),
        "duration_ms": _round(probe.get("duration_ms"), 3),
    }


def _compute_app_class(process_name: str) -> str:
    name = process_name.lower()
    if "insufficient permissions" in name or name.startswith("["):
        return "permission_unknown"
    if any(token in name for token in ("python", "torch", "accelerate", "train", "lulynx")):
        return "compute_or_training_client"
    if any(
        token in name
        for token in (
            "explorer.exe",
            "textinputhost.exe",
            "msedgewebview2.exe",
            "chrome.exe",
            "telegram.exe",
            "systemsettings.exe",
            "windowsterminal.exe",
            "bongocat.exe",
            "dwm.exe",
        )
    ):
        return "background_gpu_client"
    return "unknown_gpu_client"


def _compute_apps_classification_summary(
    compute_rows: Iterable[Mapping[str, Any]],
    compute_probe: Mapping[str, Any],
    process_details: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    details_by_pid = {
        _clean_text(item.get("pid")): _clean_jsonish(dict(item))
        for item in (process_details or [])
        if isinstance(item, Mapping)
    }
    permission_denied_row_count = 0
    permission_denied_resolved_count = 0
    for item in compute_rows:
        pid = _clean_text(item.get("pid"))
        process_name = _clean_text(item.get("process_name"))
        detail = _mapping(details_by_pid.get(pid))
        resolved_process_name = _clean_text(
            detail.get("image_name") or detail.get("process_name") or ""
        )
        permission_denied_row = _compute_app_class(process_name) == "permission_unknown"
        if permission_denied_row:
            permission_denied_row_count += 1
        classification_source = "nvidia_smi_process_name"
        classified_name = process_name
        if permission_denied_row and resolved_process_name:
            classified_name = resolved_process_name
            classification_source = "windows_process_detail"
            permission_denied_resolved_count += 1
        row_class = _compute_app_class(classified_name)
        class_counts[row_class] = class_counts.get(row_class, 0) + 1
        rows.append(
            {
                "pid": pid,
                "process_name": process_name,
                "resolved_process_name": resolved_process_name,
                "used_memory": _clean_text(item.get("used_memory")),
                "classification": row_class,
                "classification_source": classification_source,
                "permission_denied_row": permission_denied_row,
                "process_detail_available": bool(detail),
                "process_detail_status": _clean_text(detail.get("status")),
                "process_detail_window_title": _clean_text(detail.get("window_title")),
                "blocks_explicit_empty_result": True,
            }
        )
    if (
        bool(compute_probe.get("permission_denied"))
        and not class_counts.get("permission_unknown")
        and permission_denied_resolved_count == 0
    ):
        class_counts["permission_unknown"] = 1
    blocking_compute_like_count = class_counts.get("compute_or_training_client", 0)
    permission_unknown_count = class_counts.get("permission_unknown", 0)
    background_gpu_client_count = class_counts.get("background_gpu_client", 0)
    unknown_gpu_client_count = class_counts.get("unknown_gpu_client", 0)
    actionable_blockers: list[str] = []
    if permission_unknown_count:
        actionable_blockers.append("rerun_compute_apps_probe_with_sufficient_permissions")
    if blocking_compute_like_count:
        actionable_blockers.append("stop_training_or_python_gpu_clients_before_tail8_rerun")
    if background_gpu_client_count or unknown_gpu_client_count:
        actionable_blockers.append("close_or_idle_background_gpu_clients_before_tail8_rerun")
    if not rows and bool(compute_probe.get("inspection_ready")):
        actionable_blockers.append("compute_apps_probe_explicit_empty_result_ready")
    return {
        "row_count": len(rows),
        "blocking_row_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "permission_denied_row_count": permission_denied_row_count,
        "permission_denied_resolved_count": permission_denied_resolved_count,
        "permission_unknown_count": permission_unknown_count,
        "blocking_compute_like_count": blocking_compute_like_count,
        "background_gpu_client_count": background_gpu_client_count,
        "unknown_gpu_client_count": unknown_gpu_client_count,
        "explicit_empty_result": not rows and bool(compute_probe.get("inspection_ready")),
        "actionable_blockers": actionable_blockers,
        "rows": rows[:20],
    }


def _compute_apps_probe_proof(
    compute_probe: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    sanitized_rows = [
        {
            "pid": _clean_text(row.get("pid")),
            "classification": _clean_text(row.get("classification")),
            "classification_source": _clean_text(row.get("classification_source")),
            "resolved_process_name": _clean_text(row.get("resolved_process_name")),
            "blocks_explicit_empty_result": bool(row.get("blocks_explicit_empty_result")),
        }
        for row in (_mapping(item) for item in _list(classification.get("rows")))
    ]
    explicit_empty_result = bool(classification.get("explicit_empty_result"))
    permission_unknown_count = _safe_int(classification.get("permission_unknown_count"))
    probe_present = bool(compute_probe.get("present"))
    query_command = _clean_text_list(compute_probe.get("command"))
    probe_timestamp = _clean_text(compute_probe.get("timestamp_utc"))
    proof_ready = (
        probe_present
        and bool(compute_probe.get("inspection_ready"))
        and bool(query_command)
        and bool(probe_timestamp)
        and explicit_empty_result
        and permission_unknown_count == 0
        and not sanitized_rows
    )
    return {
        "schema_version": 1,
        "proof": "compute_apps_probe_proof_v1",
        "probe_present": probe_present,
        "query_command": query_command,
        "query_exit_code": _safe_int(compute_probe.get("query_returncode")),
        "query_ok": bool(compute_probe.get("query_ok")),
        "inspection_ready": bool(compute_probe.get("inspection_ready")),
        "permission_denied": bool(compute_probe.get("permission_denied")),
        "permission_unknown_count": permission_unknown_count,
        "explicit_empty_result_required": True,
        "explicit_empty_result": explicit_empty_result,
        "proof_ready": proof_ready,
        "raw_line_count": _safe_int(compute_probe.get("raw_line_count")),
        "row_count": _safe_int(compute_probe.get("row_count")),
        "sanitized_rows": sanitized_rows[:20],
        "probe_timestamp_utc": probe_timestamp,
        "probe_duration_ms": _round(compute_probe.get("duration_ms"), 3),
    }


def _tail8_request_config_without_seed() -> dict[str, Any]:
    return {
        "family": "newbie",
        "profiles": "standard",
        "steps": 48,
        "steady_warmup": 8,
        "samples": 8,
        "resolution": 64,
        "network_dim": 1,
        "train_batch_size": 1,
        "dataloader_workers": 0,
        "checkpoint_policy": "off",
        "dataloader_prefetch_factor": 2,
        "phase_profile": True,
        "data_transfer_profile": True,
        "data_transfer_profile_mode": "event",
        "allow_dataloader_rebuild_current_run": True,
        "max_actions_per_run": 2,
        "native_cache_mode": "cache_first",
        "newbie_latent_crop_size": 0,
        "newbie_target_scope": "tail8_attention",
    }


def _tail8_case_config_contract(config_without_seed: Mapping[str, Any]) -> dict[str, Any]:
    digest = _digest(config_without_seed)
    return {
        "schema_version": 1,
        "contract": "tail8_seed2027_case_config_contract_v1",
        "case_id": TAIL8_SEED2027_CASE_ID,
        "reference_run_id": TAIL8_SEED1337_REFERENCE_CASE_ID,
        "baseline_run_id": TAIL8_SEED2027_BASELINE_CASE_ID,
        "required_seed": 2027,
        "allowed_config_diff_keys": ["seed"],
        "family": _clean_text(config_without_seed.get("family")),
        "profiles": _clean_text(config_without_seed.get("profiles")),
        "newbie_target_scope": _clean_text(config_without_seed.get("newbie_target_scope")),
        "steps": _safe_int(config_without_seed.get("steps")),
        "steady_warmup": _safe_int(config_without_seed.get("steady_warmup")),
        "samples": _safe_int(config_without_seed.get("samples")),
        "resolution": _safe_int(config_without_seed.get("resolution")),
        "network_dim": _safe_int(config_without_seed.get("network_dim")),
        "train_batch_size": _safe_int(config_without_seed.get("train_batch_size")),
        "checkpoint_policy": _clean_text(config_without_seed.get("checkpoint_policy")),
        "native_cache_mode": _clean_text(config_without_seed.get("native_cache_mode")),
        "newbie_latent_crop_size": _safe_int(
            config_without_seed.get("newbie_latent_crop_size")
        ),
        "planned_out_dir": TAIL8_RERUN_OUT_DIR,
        "manual_runner": "devtools/run_bubble_closed_loop_matrix.py",
        "config_digest_excluding_seed": digest,
        "contract_ready": True,
    }


def _tail8_manual_rerun_envelope(
    *,
    manual_rerun_ready: bool,
    forward_anomaly_gate: Mapping[str, Any],
    gpu_idle_ready: bool,
    compute_apps_probe_proof: Mapping[str, Any],
    environment_snapshot_ready: bool,
    disk_space_ready: bool,
) -> dict[str, Any]:
    python_executable = "backend/env/python-flashattention/python.exe"
    runner = "devtools/run_bubble_closed_loop_matrix.py"
    base_command = [
        python_executable,
        runner,
        "--case",
        TAIL8_SEED2027_CASE_ID,
        "--out-dir",
        TAIL8_RERUN_OUT_DIR,
    ]
    config_without_seed = _tail8_request_config_without_seed()
    case_config_contract = _tail8_case_config_contract(config_without_seed)
    expected_outputs = [
        f"{TAIL8_RERUN_OUT_DIR}/closed_loop_matrix_results.json",
        (
            f"{TAIL8_RERUN_OUT_DIR}/{TAIL8_SEED2027_CASE_ID}/"
            "natural_data_wait_evidence.json"
        ),
        f"{TAIL8_RERUN_OUT_DIR}/{TAIL8_SEED2027_CASE_ID}/run/newbie_summary.json",
        (
            f"{TAIL8_RERUN_OUT_DIR}/{TAIL8_SEED2027_CASE_ID}/run/newbie/standard/"
            "output/run_manifest.json"
        ),
        "devtools/benchmark_evidence/bubble_runtime/newbie_tail8_attention_compute_review.json",
        "devtools/benchmark_evidence/bubble_runtime/newbie_tail8_forward_anomaly_review.json",
        "devtools/benchmark_evidence/bubble_runtime/newbie_tail8_seed2027_rerun_preflight.json",
        "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_experiment_readiness_next_actions.json",
        "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_readiness_terminal_self_check.json",
        "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_release_readiness_guard_report.json",
    ]
    ready_gate = {
        "candidate_incomplete_forward_anomaly_ready": bool(
            forward_anomaly_gate.get("candidate_incomplete_forward_anomaly_ready")
        ),
        "forward_anomaly_review_ready": bool(
            forward_anomaly_gate.get("forward_anomaly_review_ready")
        ),
        "forward_anomaly_comparison_source_ready": bool(
            forward_anomaly_gate.get("comparison_source_ready")
        ),
        "forward_anomaly_comparison_source_missing_count": _safe_int(
            forward_anomaly_gate.get("comparison_source_missing_count")
        ),
        "forward_anomaly_missing_source_manifest_ids": _strings(
            forward_anomaly_gate.get("missing_source_manifest_ids")
        ),
        "gpu_idle_ready": bool(gpu_idle_ready),
        "compute_apps_probe_proof_ready": bool(compute_apps_probe_proof.get("proof_ready")),
        "environment_snapshot_ready": bool(environment_snapshot_ready),
        "disk_space_ready": bool(disk_space_ready),
        "manual_rerun_ready": bool(manual_rerun_ready),
    }
    return {
        "schema_version": 1,
        "envelope": "tail8_manual_rerun_envelope_v1",
        "status": "protected_manual_ready" if manual_rerun_ready else "blocked_by_preflight",
        "reference_run_id": TAIL8_SEED1337_REFERENCE_CASE_ID,
        "baseline_run_id": TAIL8_SEED2027_BASELINE_CASE_ID,
        "candidate_run_id": TAIL8_SEED2027_CASE_ID,
        "planned_out_dir": TAIL8_RERUN_OUT_DIR,
        "manual_execute_command": base_command,
        "manual_dry_run_command": [*base_command, "--dry-run"],
        "request_config_digest_excluding_seed": case_config_contract[
            "config_digest_excluding_seed"
        ],
        "request_config_excluding_seed": config_without_seed,
        "case_config_contract": case_config_contract,
        "allowed_config_diff_keys": ["seed"],
        "required_seed": 2027,
        "post_rerun_refresh_sequence": POST_RERUN_REFRESH_SEQUENCE,
        "expected_post_rerun_outputs": expected_outputs,
        "ready_gate": ready_gate,
        "manual_start_required_after_ready": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "not_release_evidence": True,
    }


def _recommended_next_actions(
    *,
    missing_source_manifest_ids: Sequence[str],
    manual_rerun_ready: bool,
    gpu_idle_ready: bool,
    compute_apps_probe_ready: bool,
    compute_apps_probe_proof_ready: bool,
    compute_apps_present: bool,
    environment_snapshot_ready: bool,
    disk_space_ready: bool,
) -> list[str]:
    actions: list[str] = []
    missing_source_actions = {
        "seed2027_tail8_candidate_manifest": (
            "restore_or_rebuild_seed2027_tail8_candidate_manifest_before_rerun"
        ),
        "seed1337_tail8_reference_manifest": (
            "restore_seed1337_tail8_reference_manifest_before_comparison"
        ),
        "seed2027_layer0_baseline_manifest": (
            "restore_seed2027_layer0_baseline_manifest_before_comparison"
        ),
    }
    for source_id in missing_source_manifest_ids:
        action_id = missing_source_actions.get(str(source_id))
        if action_id:
            actions.append(action_id)
    if (
        compute_apps_present
        or not compute_apps_probe_ready
        or not compute_apps_probe_proof_ready
    ):
        actions.append(
            "requery_compute_apps_until_the_inspection_returns_an_explicit_empty_result"
        )
    if not gpu_idle_ready or not environment_snapshot_ready:
        actions.append("capture_gpu_telemetry_and_environment_snapshot_until_gpu_is_idle")
    if not disk_space_ready:
        actions.append("free_or_move_disk_space_before_tail8_seed2027_manual_rerun")
    if manual_rerun_ready:
        actions.append(
            "rerun_seed2027_tail8_long_window_once_after_gpu_idle_and_snapshot_ready"
        )
    actions.extend(
        [
            "treat_any_new_partial_as_diagnostic_only_until_full_pair_completes",
            "compare_tail4_tail8_tail12_only_after_seed2027_tail8_repeat_is_complete",
        ]
    )
    return actions


def _run_present(row: Mapping[str, Any]) -> bool:
    if "present" in row:
        return bool(row.get("present"))
    status = _known_status(row.get("status"))
    return bool(status and status != "missing")


def _forward_anomaly_gate_summary(
    anomaly: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
) -> dict[str, Any]:
    reference_runs = _mapping(anomaly.get("reference_runs"))
    tail8_reference = _mapping(reference_runs.get("tail8_seed1337"))
    layer0_baseline = _mapping(reference_runs.get("layer0_seed2027_baseline"))
    reference_declared = bool(reference_runs)
    candidate_present = _run_present(candidate_run)
    tail8_reference_present = (
        _run_present(tail8_reference) if reference_declared else True
    )
    layer0_baseline_present = (
        _run_present(layer0_baseline) if reference_declared else True
    )
    missing_source_manifest_ids: list[str] = []
    if not candidate_present:
        missing_source_manifest_ids.append("seed2027_tail8_candidate_manifest")
    if not tail8_reference_present:
        missing_source_manifest_ids.append("seed1337_tail8_reference_manifest")
    if not layer0_baseline_present:
        missing_source_manifest_ids.append("seed2027_layer0_baseline_manifest")
    review_ready = bool(anomaly.get("review_ready", bool(anomaly)))
    comparison_source_ready = not missing_source_manifest_ids
    candidate_incomplete = bool(diagnosis.get("candidate_incomplete"))
    forward_anomaly = bool(diagnosis.get("forward_runtime_anomaly"))
    low_data_wait = bool(diagnosis.get("low_data_wait"))
    gate_ready = (
        review_ready
        and comparison_source_ready
        and candidate_incomplete
        and forward_anomaly
        and low_data_wait
    )
    return {
        "forward_anomaly_review_ready": review_ready,
        "candidate_run_present": candidate_present,
        "tail8_seed1337_reference_present": tail8_reference_present,
        "layer0_seed2027_baseline_present": layer0_baseline_present,
        "comparison_source_ready": comparison_source_ready,
        "comparison_source_present_count": 3 - len(missing_source_manifest_ids),
        "comparison_source_missing_count": len(missing_source_manifest_ids),
        "missing_source_manifest_ids": missing_source_manifest_ids,
        "candidate_incomplete_forward_anomaly_ready": gate_ready,
    }


def build_newbie_tail8_seed2027_rerun_preflight(
    *,
    forward_anomaly_review: Mapping[str, Any] | None,
    gpu_samples: Iterable[Mapping[str, Any]],
    resource_report: Any,
    compute_apps: Iterable[Mapping[str, Any]] | None = None,
    compute_apps_probe: Mapping[str, Any] | None = None,
    process_details: Iterable[Mapping[str, Any]] | None = None,
    environment_snapshot: Mapping[str, Any] | None = None,
    min_disk_free_gb: float = DEFAULT_MIN_DISK_FREE_GB,
) -> dict[str, Any]:
    anomaly = _mapping(forward_anomaly_review)
    diagnosis = _mapping(anomaly.get("diagnosis"))
    comparison = _mapping(anomaly.get("comparison"))
    candidate_run = _mapping(anomaly.get("candidate_run"))
    gpu_summary = _gpu_summary_from_samples(gpu_samples)
    resources = _resource_summary(resource_report)
    compute_probe = _compute_apps_probe_summary(compute_apps_probe)
    env = _default_environment_snapshot()
    env.update({_clean_text(k): _clean_text(v) for k, v in _mapping(environment_snapshot).items()})
    compute_rows = [
        {_clean_text(key): _clean_text(value) for key, value in item.items()}
        for item in (compute_apps or [])
        if isinstance(item, Mapping)
    ]
    disk_free_gb = _safe_float(resources.get("disk_free_gb"))
    disk_space_ready = disk_free_gb >= float(min_disk_free_gb)

    candidate_incomplete = bool(diagnosis.get("candidate_incomplete"))
    forward_anomaly = bool(diagnosis.get("forward_runtime_anomaly"))
    low_data_wait = bool(diagnosis.get("low_data_wait"))
    forward_anomaly_gate = _forward_anomaly_gate_summary(
        anomaly,
        diagnosis,
        candidate_run,
    )
    gpu_idle_ready = (
        bool(gpu_summary.get("available"))
        and _safe_int(gpu_summary.get("valid_sample_count")) >= 3
        and _safe_float(gpu_summary.get("gpu_util_pct_p95")) <= 20.0
        and _safe_float(gpu_summary.get("gpu_active_sample_ratio")) <= 0.0
        and _safe_float(gpu_summary.get("memory_util_pct_mean")) <= 35.0
    )
    compute_apps_count = len(compute_rows)
    compute_apps_present = compute_apps_count > 0
    compute_apps_probe_present = bool(compute_probe.get("present"))
    compute_apps_probe_ready = bool(compute_probe.get("inspection_ready"))
    compute_apps_classification = _compute_apps_classification_summary(
        compute_rows,
        compute_probe,
        process_details,
    )
    compute_probe_proof = _compute_apps_probe_proof(compute_probe, compute_apps_classification)
    compute_apps_probe_proof_ready = bool(compute_probe_proof.get("proof_ready"))
    missing_environment_snapshot_keys = [
        key for key in ENVIRONMENT_SNAPSHOT_REQUIRED_KEYS if not _clean_text(env.get(key))
    ]
    environment_snapshot_ready = not missing_environment_snapshot_keys
    manual_rerun_ready = (
        bool(forward_anomaly_gate.get("candidate_incomplete_forward_anomaly_ready"))
        and gpu_idle_ready
        and compute_apps_probe_ready
        and compute_apps_probe_proof_ready
        and environment_snapshot_ready
        and disk_space_ready
        and not compute_apps_present
    )

    blockers: list[str] = []
    if not candidate_incomplete:
        blockers.append("candidate_already_complete")
    if not forward_anomaly:
        blockers.append("forward_runtime_anomaly_not_proven")
    if not bool(forward_anomaly_gate.get("comparison_source_ready")):
        blockers.append("forward_anomaly_comparison_source_missing")
    if not low_data_wait:
        blockers.append("low_data_wait_not_proven")
    if not gpu_idle_ready:
        blockers.append("gpu_not_idle")
    if not compute_apps_probe_present:
        blockers.append("gpu_compute_apps_probe_missing")
    elif not compute_apps_probe_ready:
        blockers.append("gpu_compute_apps_probe_failed")
    if compute_apps_probe_ready and not compute_apps_probe_proof_ready:
        blockers.append("gpu_compute_apps_explicit_empty_result_not_proven")
    if compute_apps_present:
        blockers.append("gpu_compute_apps_present")
    if not environment_snapshot_ready:
        blockers.append("environment_snapshot_incomplete")
    if not disk_space_ready:
        blockers.append("disk_free_below_tail8_manual_rerun_threshold")
    blockers = sorted(set(blockers))

    status = (
        "seed2027_tail8_manual_rerun_ready"
        if manual_rerun_ready
        else "seed2027_tail8_rerun_preflight_blocked"
    )
    manual_rerun_envelope = _tail8_manual_rerun_envelope(
        manual_rerun_ready=manual_rerun_ready,
        forward_anomaly_gate=forward_anomaly_gate,
        gpu_idle_ready=gpu_idle_ready,
        compute_apps_probe_proof=compute_probe_proof,
        environment_snapshot_ready=environment_snapshot_ready,
        disk_space_ready=disk_space_ready,
    )

    digest_payload = {
        "forward_anomaly_review_status": _known_status(anomaly.get("status")),
        "forward_anomaly_gate": forward_anomaly_gate,
        "gpu_summary": gpu_summary,
        "compute_apps_probe": compute_probe,
        "compute_apps_probe_proof": compute_probe_proof,
        "resource_summary": resources,
        "disk_space_ready": disk_space_ready,
        "min_disk_free_gb": float(min_disk_free_gb),
        "compute_apps_classification_summary": compute_apps_classification,
        "tail8_manual_rerun_envelope": manual_rerun_envelope,
        "environment_snapshot": env,
        "missing_environment_snapshot_keys": missing_environment_snapshot_keys,
        "compute_apps_count": compute_apps_count,
        "manual_rerun_ready": manual_rerun_ready,
        "blockers": blockers,
    }
    recommended_next_actions = _recommended_next_actions(
        missing_source_manifest_ids=_strings(
            forward_anomaly_gate.get("missing_source_manifest_ids")
        ),
        manual_rerun_ready=manual_rerun_ready,
        gpu_idle_ready=gpu_idle_ready,
        compute_apps_probe_ready=compute_apps_probe_ready,
        compute_apps_probe_proof_ready=compute_apps_probe_proof_ready,
        compute_apps_present=compute_apps_present,
        environment_snapshot_ready=environment_snapshot_ready,
        disk_space_ready=disk_space_ready,
    )
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_newbie_tail8_seed2027_rerun_preflight",
        "family": FAMILY,
        "candidate": "newbie_target_scope:tail8_attention seed:2027",
        "status": status,
        "review_ready": True,
        "candidate_incomplete": candidate_incomplete,
        "forward_runtime_anomaly": forward_anomaly,
        "low_data_wait": low_data_wait,
        "forward_anomaly_review_status": _known_status(anomaly.get("status")),
        "forward_anomaly_review_ready": bool(
            forward_anomaly_gate.get("forward_anomaly_review_ready")
        ),
        "forward_anomaly_candidate_run_present": bool(
            forward_anomaly_gate.get("candidate_run_present")
        ),
        "forward_anomaly_tail8_seed1337_reference_present": bool(
            forward_anomaly_gate.get("tail8_seed1337_reference_present")
        ),
        "forward_anomaly_layer0_seed2027_baseline_present": bool(
            forward_anomaly_gate.get("layer0_seed2027_baseline_present")
        ),
        "forward_anomaly_comparison_source_ready": bool(
            forward_anomaly_gate.get("comparison_source_ready")
        ),
        "forward_anomaly_comparison_source_present_count": _safe_int(
            forward_anomaly_gate.get("comparison_source_present_count")
        ),
        "forward_anomaly_comparison_source_missing_count": _safe_int(
            forward_anomaly_gate.get("comparison_source_missing_count")
        ),
        "forward_anomaly_missing_source_manifest_ids": _strings(
            forward_anomaly_gate.get("missing_source_manifest_ids")
        ),
        "candidate_incomplete_forward_anomaly_ready": bool(
            forward_anomaly_gate.get("candidate_incomplete_forward_anomaly_ready")
        ),
        "manual_rerun_ready": manual_rerun_ready,
        "gpu_idle_ready": gpu_idle_ready,
        "compute_apps_present": compute_apps_present,
        "gpu_compute_apps_present": compute_apps_present,
        "gpu_compute_apps_count": compute_apps_count,
        "environment_snapshot_ready": environment_snapshot_ready,
        "environment_snapshot_required_keys": list(ENVIRONMENT_SNAPSHOT_REQUIRED_KEYS),
        "missing_environment_snapshot_keys": missing_environment_snapshot_keys,
        "disk_space_ready": disk_space_ready,
        "min_disk_free_gb": float(min_disk_free_gb),
        "disk_free_gb": _round(disk_free_gb, 2),
        "candidate_status": _known_status(anomaly.get("status")),
        "candidate_release_claim_allowed": bool(anomaly.get("release_claim_allowed")),
        "candidate_safe_to_auto_start": bool(anomaly.get("safe_to_auto_start")),
        "candidate_diagnosis": diagnosis,
        "candidate_comparison": comparison,
        "candidate_run": {
            "status": _known_status(candidate_run.get("status")),
            "global_step": _safe_int(candidate_run.get("global_step")),
            "total_steps": _safe_int(candidate_run.get("total_steps")),
            "newbie_transformer_smoke_seconds": _round(
                candidate_run.get("newbie_transformer_smoke_seconds"),
                4,
            ),
            "forward_model_execution_mean_ms": _round(
                candidate_run.get("forward_model_execution_mean_ms"),
                4,
            ),
            "data_wait_share": _round(candidate_run.get("data_wait_share"), 6),
        },
        "gpu_summary": gpu_summary,
        "compute_apps_probe": compute_probe,
        "compute_apps_probe_proof": compute_probe_proof,
        "compute_apps_classification_summary": compute_apps_classification,
        "tail8_manual_rerun_envelope": manual_rerun_envelope,
        "resource_summary": resources,
        "environment_snapshot": env,
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
        "artifact_digest": _digest(digest_payload),
        "not_release_evidence": True,
        "fail_closed": True,
        "publishable": False,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_tail8_seed2027_rerun_preflight",
]
