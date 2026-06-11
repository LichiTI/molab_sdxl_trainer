"""Build guarded SDXL non-DataLoader probe command manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SDXL_NON_DATALOADER_PROBE_COMMAND_PLAN_REPORT = "bubble_sdxl_non_dataloader_probe_command_plan_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _path_text(path: Path) -> str:
    return str(path)


def _item_index(probe_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("id") or ""): _mapping(item)
        for item in probe_plan.get("items", [])
        if _mapping(item).get("id")
    }


def _benchmark_command(
    *,
    repo_root: Path,
    out_dir: Path,
    source_data: Path,
    batch: int,
    fused_adamw: bool = False,
    attention_backend: str = "auto",
    sdpa_backend_policy: str = "cutlass",
) -> list[str]:
    command = [
        _path_text(repo_root / "backend" / "env" / "python-flashattention" / "python.exe"),
        _path_text(repo_root / "backend" / "core" / "lulynx_trainer" / "native_runtime_profile_benchmark.py"),
        "--family",
        "sdxl",
        "--profiles",
        "standard",
        "--steps",
        "16",
        "--steady-warmup",
        "4",
        "--samples",
        "8",
        "--resolution",
        "1024",
        "--network-dim",
        "1",
        "--train-batch-size",
        str(max(int(batch), 1)),
        "--attention-backend",
        str(attention_backend or "auto"),
        "--sdpa-backend-policy",
        str(sdpa_backend_policy or "cutlass"),
        "--dataloader-workers",
        "0",
        "--dataloader-prefetch-factor",
        "2",
        "--phase-profile",
        "--data-transfer-profile",
        "--data-transfer-profile-mode",
        "event",
        "--bubble-controller-enabled",
        "--bubble-controller-mode",
        "report_only",
        "--bubble-controller-warmup-steps",
        "4",
        "--bubble-controller-tune-interval-steps",
        "4",
        "--bubble-controller-max-actions-per-run",
        "0",
        "--source-data",
        _path_text(source_data),
        "--out",
        _path_text(out_dir),
    ]
    if fused_adamw:
        command.append("--fused-adamw")
    return command


def _command_entry(
    *,
    command_id: str,
    role: str,
    axis: str,
    command: Sequence[str],
    out_dir: Path,
    candidate_overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": command_id,
        "role": role,
        "axis": axis,
        "manual_start_required": True,
        "requires_gpu_if_executed": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "command": [str(part) for part in command],
        "out_dir": str(out_dir),
        "expected_summary": str(out_dir / "sdxl_summary.json"),
        "candidate_overlay": dict(candidate_overlay or {}),
    }


def _workload_group(
    *,
    repo_root: Path,
    source_data: Path,
    out_root: Path,
    probe_item: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_out = out_root / "sdxl_workload_batch1_baseline"
    candidate_out = out_root / "sdxl_workload_batch2_candidate"
    return {
        "id": "sdxl_workload_batch1_vs_batch2_probe",
        "family": "sdxl",
        "track": str(probe_item.get("track") or "batch_or_microbatch_shape_probe"),
        "category": "manual_gpu_probe_pair",
        "status": "manual_gpu_commands_ready",
        "priority": _safe_int(probe_item.get("priority"), 20),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "source_item_id": str(probe_item.get("id") or ""),
        "source_data": str(source_data),
        "request_boundary": str(probe_item.get("request_boundary") or "advisor_patch_next_request_only"),
        "required_gates": _string_list(probe_item.get("required_gates")),
        "comparison_required": {
            "primary_metric": "steady_samples_per_second",
            "secondary_metrics": ["active_window_gpu_util", "peak_vram_ratio", "loss_stability"],
            "manual_evidence_builder_needed": True,
        },
        "commands": [
            _command_entry(
                command_id="sdxl_workload_batch1_baseline",
                role="baseline",
                axis="train_batch_size",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=baseline_out,
                    source_data=source_data,
                    batch=1,
                ),
                out_dir=baseline_out,
                candidate_overlay={"train_batch_size": 1},
            ),
            _command_entry(
                command_id="sdxl_workload_batch2_candidate",
                role="candidate",
                axis="train_batch_size",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=candidate_out,
                    source_data=source_data,
                    batch=2,
                ),
                out_dir=candidate_out,
                candidate_overlay={"train_batch_size": 2},
            ),
        ],
        "rationale": str(probe_item.get("rationale") or "Probe SDXL workload shape with a batch-size A/B."),
    }


def _optimizer_group(
    *,
    repo_root: Path,
    source_data: Path,
    out_root: Path,
    probe_item: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_out = out_root / "sdxl_optimizer_default_adamw_baseline"
    candidate_out = out_root / "sdxl_optimizer_fused_adamw_candidate"
    return {
        "id": "sdxl_optimizer_default_vs_fused_adamw_probe",
        "family": "sdxl",
        "track": str(probe_item.get("track") or "optimizer_or_attention_backend_probe"),
        "category": "manual_gpu_probe_pair",
        "status": "manual_gpu_commands_ready",
        "priority": _safe_int(probe_item.get("priority"), 30),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "source_item_id": str(probe_item.get("id") or ""),
        "source_data": str(source_data),
        "request_boundary": "manual_benchmark_existing_runner",
        "required_gates": _string_list(probe_item.get("required_gates")),
        "comparison_required": {
            "primary_metric": "steady_samples_per_second",
            "secondary_metrics": ["optimizer_share", "loss_stability"],
            "manual_evidence_builder_needed": True,
        },
        "commands": [
            _command_entry(
                command_id="sdxl_optimizer_default_adamw_baseline",
                role="baseline",
                axis="optimizer_backend",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=baseline_out,
                    source_data=source_data,
                    batch=1,
                ),
                out_dir=baseline_out,
                candidate_overlay={"optimizer_args": ""},
            ),
            _command_entry(
                command_id="sdxl_optimizer_fused_adamw_candidate",
                role="candidate",
                axis="optimizer_backend",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=candidate_out,
                    source_data=source_data,
                    batch=1,
                    fused_adamw=True,
                ),
                out_dir=candidate_out,
                candidate_overlay={"optimizer_args": "fused=True"},
            ),
        ],
        "rationale": str(probe_item.get("rationale") or "Probe SDXL optimizer/backend with a guarded fused AdamW A/B."),
    }


def _attention_group(
    *,
    repo_root: Path,
    source_data: Path,
    out_root: Path,
    probe_item: Mapping[str, Any],
    availability: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_out = out_root / "sdxl_attention_sdpa_baseline"
    candidate_out = out_root / "sdxl_attention_flash2_candidate"
    summary = _mapping(availability.get("summary"))
    availability_known = str(availability.get("report") or "") == "bubble_sdxl_attention_backend_availability_v0"
    sdpa_available = bool(summary.get("sdpa_available")) if availability_known else True
    flash2_available = (
        bool(summary.get("flash2_available"))
        if "flash2_available" in summary
        else bool(summary.get("flash_attn_available")) and bool(summary.get("cuda_available", True))
    )
    if not availability_known:
        flash2_available = True
    reason_codes: list[str] = []
    if not sdpa_available:
        reason_codes.append("sdpa_unavailable")
    if not flash2_available:
        relevant = [
            code
            for code in _string_list(summary.get("blockers"))
            if code
            in {
                "flash2_unavailable",
                "flash_attn_unavailable",
                "flash_attn_func_unavailable",
                "flash2_cuda_unavailable",
                "flash2_hip_runtime",
                "flash2_sm80_unavailable",
            }
        ]
        reason_codes.extend(relevant or ["flash2_unavailable"])
    status = "manual_gpu_commands_ready" if flash2_available and sdpa_available else "blocked_backend_unavailable"
    return {
        "id": "sdxl_attention_sdpa_vs_flash2_probe",
        "family": "sdxl",
        "track": str(probe_item.get("track") or "optimizer_or_attention_backend_probe"),
        "category": "manual_gpu_probe_pair",
        "status": status,
        "priority": _safe_int(probe_item.get("priority"), 30) + 1,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "blocker_reason_codes": [] if status == "manual_gpu_commands_ready" else reason_codes,
        "availability_report": str(availability.get("report") or ""),
        "availability_summary": dict(summary),
        "source_item_id": str(probe_item.get("id") or ""),
        "source_data": str(source_data),
        "request_boundary": "manual_benchmark_existing_runner",
        "required_gates": [
            *_string_list(probe_item.get("required_gates")),
            "actual_attention_backend_must_change",
            "fallback_to_sdpa_blocks_promotion",
            "flash2_preflight_must_pass",
            "flash2_attention_mask_none_required",
        ],
        "comparison_required": {
            "primary_metric": "steady_samples_per_second",
            "secondary_metrics": ["active_window_gpu_util", "loss_stability", "actual_attention_backend"],
            "manual_evidence_builder_needed": True,
        },
        "commands": [
            _command_entry(
                command_id="sdxl_attention_sdpa_baseline",
                role="baseline",
                axis="attention_backend",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=baseline_out,
                    source_data=source_data,
                    batch=1,
                    attention_backend="sdpa",
                    sdpa_backend_policy="cutlass",
                ),
                out_dir=baseline_out,
                candidate_overlay={"attention_backend": "sdpa", "sdpa_backend_policy": "cutlass"},
            ),
            _command_entry(
                command_id="sdxl_attention_flash2_candidate",
                role="candidate",
                axis="attention_backend",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=candidate_out,
                    source_data=source_data,
                    batch=1,
                    attention_backend="flash2",
                    sdpa_backend_policy="cutlass",
                ),
                out_dir=candidate_out,
                candidate_overlay={
                    "attention_backend": "flash2",
                    "fallback_backend": "sdpa",
                    "requires_attention_mask_none": True,
                },
            ),
        ],
        "safety_checks": [
            "runner_default_manifest_only",
            "requires_explicit_execute",
            "workers_prefetch_fixed",
            "candidate_actual_backend_recorded",
            "fallback_to_sdpa_blocks_promotion",
            "flash2_preflight_must_pass",
            "flash2_attention_mask_none_required",
            "no_release_claim_from_runner_manifest",
        ],
        "rationale": (
            "Probe SDXL attention backend with explicit sdpa baseline vs flash2 candidate after xformers was blocked; "
            "if flash2 is unavailable or falls back to sdpa, the result is recorded and blocks promotion."
        ),
    }


def _sdpa_policy_group(
    *,
    repo_root: Path,
    source_data: Path,
    out_root: Path,
    probe_item: Mapping[str, Any],
    availability: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_out = out_root / "sdxl_attention_sdpa_baseline"
    candidate_out = out_root / "sdxl_attention_sdpa_flash_policy_candidate"
    summary = _mapping(availability.get("summary"))
    availability_known = str(availability.get("report") or "") == "bubble_sdxl_attention_backend_availability_v0"
    sdpa_available = bool(summary.get("sdpa_available")) if availability_known else True
    status = "manual_gpu_commands_ready" if sdpa_available else "blocked_backend_unavailable"
    return {
        "id": "sdxl_attention_sdpa_cutlass_vs_flash_policy_probe",
        "family": "sdxl",
        "track": str(probe_item.get("track") or "optimizer_or_attention_backend_probe"),
        "category": "manual_gpu_probe_pair",
        "status": status,
        "priority": _safe_int(probe_item.get("priority"), 30) + 2,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "blocker_reason_codes": [] if status == "manual_gpu_commands_ready" else ["sdpa_unavailable"],
        "availability_report": str(availability.get("report") or ""),
        "availability_summary": dict(summary),
        "source_item_id": str(probe_item.get("id") or ""),
        "source_data": str(source_data),
        "request_boundary": "manual_benchmark_existing_runner",
        "required_gates": [
            *_string_list(probe_item.get("required_gates")),
            "actual_attention_backend_must_remain_sdpa",
            "sdpa_backend_policy_must_change",
            "fallback_to_torch_blocks_promotion",
        ],
        "comparison_required": {
            "primary_metric": "steady_samples_per_second",
            "secondary_metrics": ["loss_stability", "phase_profile", "actual_attention_backend"],
            "manual_evidence_builder_needed": True,
        },
        "commands": [
            _command_entry(
                command_id="sdxl_attention_sdpa_cutlass_policy_baseline",
                role="baseline",
                axis="sdpa_backend_policy",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=baseline_out,
                    source_data=source_data,
                    batch=1,
                    attention_backend="sdpa",
                    sdpa_backend_policy="cutlass",
                ),
                out_dir=baseline_out,
                candidate_overlay={"attention_backend": "sdpa", "sdpa_backend_policy": "cutlass"},
            ),
            _command_entry(
                command_id="sdxl_attention_sdpa_flash_policy_candidate",
                role="candidate",
                axis="sdpa_backend_policy",
                command=_benchmark_command(
                    repo_root=repo_root,
                    out_dir=candidate_out,
                    source_data=source_data,
                    batch=1,
                    attention_backend="sdpa",
                    sdpa_backend_policy="flash",
                ),
                out_dir=candidate_out,
                candidate_overlay={"attention_backend": "sdpa", "sdpa_backend_policy": "flash"},
            ),
        ],
        "safety_checks": [
            "runner_default_manifest_only",
            "requires_explicit_execute",
            "workers_prefetch_fixed",
            "actual_backend_must_remain_sdpa",
            "sdpa_backend_policy_must_change",
            "no_release_claim_from_runner_manifest",
        ],
        "rationale": (
            "Probe SDXL SDPA backend policy after flash2 regressed on the current axis; compare existing "
            "sdpa/cutlass baseline with an sdpa/flash policy candidate without repeating the flash2 axis."
        ),
    }


def build_sdxl_non_dataloader_probe_command_plan(
    probe_plan: Mapping[str, Any],
    *,
    repo_root: Path,
    source_data: Path | None = None,
    out_root: Path | None = None,
    attention_backend_availability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest-only command plan for supported SDXL non-DataLoader probes."""

    repo = Path(repo_root)
    source = Path(source_data) if source_data is not None else repo / "sucai" / "6_lulu"
    out_base = out_root or repo / "devtools" / "benchmark_evidence" / "bubble_runtime" / "sdxl_non_dataloader_probe_runs"
    availability = _mapping(attention_backend_availability)
    items = _item_index(probe_plan)
    groups: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    workload = items.get("sdxl_batch_or_microbatch_shape_probe")
    if _mapping(workload).get("status") == "manual_review_ready":
        groups.append(_workload_group(repo_root=repo, source_data=source, out_root=out_base, probe_item=_mapping(workload)))

    optimizer = items.get("sdxl_optimizer_or_attention_backend_probe")
    if _mapping(optimizer).get("status") == "manual_review_ready":
        group = _optimizer_group(repo_root=repo, source_data=source, out_root=out_base, probe_item=_mapping(optimizer))
        groups.append(group)
        blocked.extend(group.get("blocked_subaxes") or [])
        attention = _attention_group(
            repo_root=repo,
            source_data=source,
            out_root=out_base,
            probe_item=_mapping(optimizer),
            availability=availability,
        )
        if str(attention.get("status") or "") == "manual_gpu_commands_ready":
            groups.append(attention)
        else:
            blocked.append(
                {
                    "id": str(attention.get("id") or "sdxl_attention_sdpa_vs_flash2_probe"),
                    "family": "sdxl",
                    "status": str(attention.get("status") or "blocked_backend_unavailable"),
                    "reason_codes": _string_list(attention.get("blocker_reason_codes")) or ["attention_backend_unavailable"],
                    "availability_report": str(attention.get("availability_report") or ""),
                    "availability_summary": dict(_mapping(attention.get("availability_summary"))),
                    "blocked_actions": [f"run_{attention.get('id')}_via_protected_runner"],
                    "rationale": "Do not run the SDXL flash2 attention A/B when the runner environment cannot apply the requested backend.",
                }
            )
        sdpa_policy = _sdpa_policy_group(
            repo_root=repo,
            source_data=source,
            out_root=out_base,
            probe_item=_mapping(optimizer),
            availability=availability,
        )
        if str(sdpa_policy.get("status") or "") == "manual_gpu_commands_ready":
            groups.append(sdpa_policy)
        else:
            blocked.append(
                {
                    "id": str(sdpa_policy.get("id") or "sdxl_attention_sdpa_cutlass_vs_flash_policy_probe"),
                    "family": "sdxl",
                    "status": str(sdpa_policy.get("status") or "blocked_backend_unavailable"),
                    "reason_codes": _string_list(sdpa_policy.get("blocker_reason_codes")) or ["sdpa_unavailable"],
                    "availability_report": str(sdpa_policy.get("availability_report") or ""),
                    "availability_summary": dict(_mapping(sdpa_policy.get("availability_summary"))),
                    "blocked_actions": [f"run_{sdpa_policy.get('id')}_via_protected_runner"],
                    "rationale": "Do not run the SDXL SDPA policy A/B when SDPA is unavailable in the runner environment.",
                }
            )

    if not groups and str(probe_plan.get("status") or "") == "needs_sdxl_real_material_ab_evidence":
        blocked.append(
            {
                "id": "collect_sdxl_real_material_ab_evidence_first",
                "status": "blocked_missing_prerequisite",
                "reason": "SDXL non-DataLoader probe commands require the aggregate investigation report first.",
            }
        )

    command_count = sum(len(group.get("commands") or []) for group in groups)
    return {
        "schema_version": 1,
        "report": SDXL_NON_DATALOADER_PROBE_COMMAND_PLAN_REPORT,
        "status": "manual_gpu_commands_ready" if command_count else "blocked_or_no_supported_commands",
        "family": "sdxl",
        "source_probe_plan_report": str(probe_plan.get("report") or ""),
        "source_probe_plan_status": str(probe_plan.get("status") or ""),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "source_data": str(source),
        "out_root": str(out_base),
        "summary": {
            "group_count": len(groups),
            "manual_gpu_command_count": command_count,
            "blocked_subaxis_count": len(blocked),
            "supported_axes": [str(group.get("id") or "") for group in groups],
        },
        "groups": groups,
        "blocked_subaxes": blocked,
        "notes": [
            "This manifest does not execute GPU work.",
            "Commands reuse existing native/runtime benchmark entrypoints; no new training entrypoint is introduced.",
            "Workers/prefetch stay fixed at workers=0/prefetch=2 so these probes do not repeat the exhausted DataLoader axis.",
        ],
    }


__all__ = [
    "SDXL_NON_DATALOADER_PROBE_COMMAND_PLAN_REPORT",
    "build_sdxl_non_dataloader_probe_command_plan",
]
