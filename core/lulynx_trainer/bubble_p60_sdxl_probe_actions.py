"""Build P60 SDXL non-DataLoader probe actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.core.lulynx_trainer.bubble_p60_sdxl_compile_actions import (
    build_sdxl_compile_followup_actions,
)
from backend.core.lulynx_trainer.bubble_p60_sdxl_workload_actions import (
    build_sdxl_workload_followup_actions,
)


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


def _base_action(
    *,
    action_id: str,
    family: str,
    priority: int,
    action_type: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "family": family,
        "priority": int(priority),
        "action_type": action_type,
        "status": status,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "manual_start_required": False,
        "requires_gpu_if_executed": False,
        "requires_external_input": False,
        "diagnostic_only": False,
        "release_relevant": False,
        "reasons": [],
        "warnings": [],
    }


def _sdxl_pivot_action(sdxl_investigation: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(sdxl_investigation.get("status") or "") != "sdxl_dataloader_axis_negative_evidence":
        return []
    summary = _mapping(sdxl_investigation.get("summary"))
    action = _base_action(
        action_id="sdxl_pivot_non_dataloader_or_workload_probe",
        family="sdxl",
        priority=30,
        action_type="manual_probe_design",
        status="manual_review_ready",
    )
    action.update(
        {
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "case_count": _safe_int(summary.get("case_count")),
            "baseline_compute_bound_count": _safe_int(summary.get("baseline_compute_bound_count")),
            "baseline_data_wait_below_threshold_count": _safe_int(
                summary.get("baseline_data_wait_below_threshold_count")
            ),
            "after_data_wait_worse_count": _safe_int(summary.get("after_data_wait_worse_count")),
            "next_actions": _string_list(summary.get("next_actions")),
            "candidate_tracks": [
                "batch_or_microbatch_shape_probe",
                "host_scheduling_closed_loop_probe",
                "transfer_or_h2d_profile_review",
                "attention_compile_optimizer_guardrail_review",
            ],
            "required_gates": [
                "throughput_gain",
                "loss_stability",
                "vram_ratio",
                "phase_profile_boundary",
                "case_specific_release_wording",
            ],
            "rationale": (
                "SDXL workers/prefetch evidence on the current source axis is negative; pivot to compute, "
                "host, transfer, or workload-shape hypotheses instead of repeating the same data-supply canary."
            ),
        }
    )
    return [action]


def _sdxl_probe_plan_actions(sdxl_probe_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(sdxl_probe_plan.get("report") or "") != "bubble_sdxl_non_dataloader_probe_plan_v0":
        return []
    actions: list[dict[str, Any]] = []
    for raw in sdxl_probe_plan.get("items", []):
        item = _mapping(raw)
        category = str(item.get("category") or "")
        status = str(item.get("status") or "")
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        if category == "guardrail" and status == "active":
            action = _base_action(
                action_id=f"guardrail_{item_id}",
                family="sdxl",
                priority=_safe_int(item.get("priority"), 10),
                action_type="guardrail",
                status="active",
            )
            action.update(
                {
                    "source_item_id": item_id,
                    "reasons": _string_list(item.get("reason_codes")),
                    "blocked_actions": _string_list(item.get("blocked_actions")),
                    "rationale": str(item.get("rationale") or "Keep this SDXL probe guardrail active."),
                }
            )
            actions.append(action)
            continue
        if category != "manual_probe" or status != "manual_review_ready":
            continue
        action = _base_action(
            action_id=f"review_{item_id}",
            family="sdxl",
            priority=_safe_int(item.get("priority"), 40),
            action_type="manual_probe_design",
            status="manual_review_ready",
        )
        action.update(
            {
                "source_item_id": item_id,
                "manual_start_required": True,
                "requires_gpu_if_executed": bool(item.get("requires_gpu_if_executed")),
                "probe_profile": str(item.get("probe_profile") or ""),
                "request_boundary": str(item.get("request_boundary") or ""),
                "candidate_axes": _string_list(item.get("candidate_axes")),
                "required_inputs": _string_list(item.get("required_inputs")),
                "required_gates": _string_list(item.get("required_gates")),
                "reasons": _string_list(item.get("reason_codes")),
                "rationale": str(
                    item.get("rationale") or "Review this SDXL non-DataLoader probe before running GPU work."
                ),
            }
        )
        actions.append(action)
    return actions


def _sdxl_probe_command_actions(sdxl_probe_command_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(sdxl_probe_command_plan.get("report") or "") != "bubble_sdxl_non_dataloader_probe_command_plan_v0":
        return []
    actions: list[dict[str, Any]] = []
    for raw in sdxl_probe_command_plan.get("blocked_subaxes", []):
        blocked = _mapping(raw)
        blocked_id = str(blocked.get("id") or "")
        if not blocked_id:
            continue
        action = _base_action(
            action_id=f"blocked_{blocked_id}",
            family="sdxl",
            priority=31,
            action_type="blocked_probe_subaxis",
            status="blocked",
        )
        action.update(
            {
                "source_blocked_subaxis_id": blocked_id,
                "reasons": _string_list(blocked.get("reason_codes")) or _string_list(blocked.get("reasons")),
                "blocked_actions": _string_list(blocked.get("blocked_actions")),
                "availability_report": str(blocked.get("availability_report") or ""),
                "availability_summary": dict(_mapping(blocked.get("availability_summary"))),
                "rationale": str(blocked.get("rationale") or "This SDXL probe sub-axis is blocked by preflight evidence."),
            }
        )
        actions.append(action)
    for raw in sdxl_probe_command_plan.get("groups", []):
        group = _mapping(raw)
        if str(group.get("status") or "") != "manual_gpu_commands_ready":
            continue
        group_id = str(group.get("id") or "")
        commands = [_mapping(item) for item in group.get("commands", []) if _mapping(item)]
        if not group_id or not commands:
            continue
        action = _base_action(
            action_id=f"prepare_{group_id}",
            family="sdxl",
            priority=_safe_int(group.get("priority"), 35),
            action_type="manual_probe_command_plan",
            status="manual_review_ready",
        )
        action.update(
            {
                "source_group_id": group_id,
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "command_count": len(commands),
                "source_data": str(group.get("source_data") or sdxl_probe_command_plan.get("source_data") or ""),
                "request_boundary": str(group.get("request_boundary") or ""),
                "required_gates": _string_list(group.get("required_gates")),
                "comparison_required": dict(_mapping(group.get("comparison_required"))),
                "commands": [
                    {
                        "id": str(command.get("id") or ""),
                        "role": str(command.get("role") or ""),
                        "axis": str(command.get("axis") or ""),
                        "out_dir": str(command.get("out_dir") or ""),
                        "expected_summary": str(command.get("expected_summary") or ""),
                        "command": list(command.get("command") or []),
                    }
                    for command in commands
                ],
                "rationale": str(
                    group.get("rationale") or "Review this supported SDXL probe command pair before GPU execution."
                ),
            }
        )
        actions.append(action)
    return actions


def _sdxl_probe_runner_actions(
    sdxl_probe_runner_manifest: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if str(sdxl_probe_runner_manifest.get("report") or "") != "bubble_sdxl_non_dataloader_probe_command_runner_v0":
        return []
    if not bool(sdxl_probe_runner_manifest.get("ok")):
        return []
    rows = [_mapping(row) for row in sdxl_probe_runner_manifest.get("commands", []) if _mapping(row)]
    pending_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("output_exists_before") or not row.get("ready_to_execute"):
            continue
        group_id = str(row.get("group_id") or "")
        if group_id:
            pending_by_group.setdefault(group_id, []).append(row)
    if not pending_by_group:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    actions: list[dict[str, Any]] = []
    for group_id, pending_rows in sorted(pending_by_group.items()):
        action = _base_action(
            action_id=f"run_{group_id}_via_protected_runner",
            family="sdxl",
            priority=32,
            action_type="protected_manual_probe_runner",
            status="manual_review_ready",
        )
        command: list[str] = []
        if repo is not None:
            command = [
                _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
                _path_text(repo / "devtools" / "run_bubble_sdxl_non_dataloader_probe_commands.py"),
                "--execute",
                "--only-missing",
                "--group",
                group_id,
            ]
        action.update(
            {
                "source_group_id": group_id,
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "runner_report": str(sdxl_probe_runner_manifest.get("report") or ""),
                "runner_manifest_only": bool(sdxl_probe_runner_manifest.get("manifest_only")),
                "runner_execute_requested": bool(sdxl_probe_runner_manifest.get("execute_requested")),
                "pending_ready_command_count": len(pending_rows),
                "pending_command_ids": [str(row.get("command_id") or "") for row in pending_rows],
                "expected_summaries": [str(row.get("expected_summary") or "") for row in pending_rows],
                "command": command,
                "safety_checks": [
                    "runner_default_manifest_only",
                    "requires_explicit_execute",
                    "only_missing_supported",
                    "workers_prefetch_fixed",
                    "controller_report_only_max_actions_zero",
                    "no_release_claim_from_runner_manifest",
                ],
                "rationale": "Use the protected runner to execute only missing SDXL non-DataLoader probe summaries after manual GPU authorization.",
            }
        )
        actions.append(action)
    return actions


def _sdxl_probe_state_refresh_action(
    sdxl_probe_runner_manifest: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if str(sdxl_probe_runner_manifest.get("report") or "") != "bubble_sdxl_non_dataloader_probe_command_runner_v0":
        return []
    summary = _mapping(sdxl_probe_runner_manifest.get("summary"))
    pending_ready = _safe_int(summary.get("pending_ready_command_count"))
    if pending_ready <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command: list[str] = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "refresh_bubble_sdxl_non_dataloader_probe_state.py"),
        ]
    action = _base_action(
        action_id="refresh_sdxl_non_dataloader_probe_state_after_runner",
        family="sdxl",
        priority=34,
        action_type="non_gpu_probe_state_refresh",
        status="manual_review_ready",
    )
    action.update(
        {
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "runner_pending_ready_command_count": pending_ready,
            "runner_executed_count": _safe_int(summary.get("executed_count")),
            "command": command,
            "refreshes": [
                "sdxl_non_dataloader_probe_results.json",
                "sdxl_non_dataloader_probe_command_runner_manifest.json",
                "p60_next_action_plan.json",
            ],
            "safety_checks": [
                "non_gpu_only",
                "does_not_pass_execute_to_runner",
                "evaluates_existing_summaries_only",
                "refreshes_top_level_p60_plan",
            ],
            "rationale": "After manually authorized SDXL probe commands finish, refresh results and the P60 action plan without starting GPU work.",
        }
    )
    return [action]


def _sdxl_probe_result_actions(sdxl_probe_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(sdxl_probe_results.get("report") or "") != "bubble_sdxl_non_dataloader_probe_results_v0":
        return []
    actions: list[dict[str, Any]] = []
    for raw in sdxl_probe_results.get("groups", []):
        group = _mapping(raw)
        group_id = str(group.get("id") or "")
        status = str(group.get("status") or "")
        if not group_id:
            continue
        if status == "pending_manual_gpu_commands":
            action = _base_action(
                action_id=f"pending_{group_id}_results",
                family="sdxl",
                priority=35,
                action_type="pending_manual_probe_results",
                status="manual_review_ready",
            )
            action.update(
                {
                    "source_group_id": group_id,
                    "manual_start_required": True,
                    "requires_gpu_if_executed": True,
                    "missing_summary_count": _safe_int(group.get("missing_summary_count")),
                    "missing_summaries": _string_list(group.get("missing_summaries")),
                    "rationale": "Run or reuse the listed SDXL probe commands before evidence can be evaluated.",
                }
            )
            actions.append(action)
            continue
        if status in {"keep_candidate_review", "needs_review", "rollback_recommended", "insufficient_evidence"}:
            action = _base_action(
                action_id=f"review_{group_id}_results",
                family="sdxl",
                priority=25 if status == "keep_candidate_review" else 35,
                action_type="manual_probe_result_review",
                status="manual_review_ready" if status != "rollback_recommended" else "blocked",
            )
            decision = _mapping(group.get("decision"))
            comparison = _mapping(group.get("comparison"))
            action.update(
                {
                    "source_group_id": group_id,
                    "probe_result_status": status,
                    "recommended_action": str(decision.get("recommended_action") or ""),
                    "reasons": _string_list(decision.get("reasons")),
                    "comparison": dict(comparison),
                    "release_claim_allowed": False,
                    "manual_start_required": False,
                    "requires_gpu_if_executed": False,
                    "rationale": "Review SDXL probe evidence with release gates before promoting any setting.",
                }
            )
            actions.append(action)
            if group_id == "sdxl_attention_sdpa_vs_flash2_probe" and status == "rollback_recommended":
                guardrail = _base_action(
                    action_id="guardrail_sdxl_do_not_repeat_attention_flash2_current_axis",
                    family="sdxl",
                    priority=12,
                    action_type="guardrail",
                    status="active",
                )
                guardrail.update(
                    {
                        "source_group_id": group_id,
                        "reasons": [
                            *_string_list(decision.get("reasons")),
                            "attention_flash2_throughput_regressed_current_axis",
                        ],
                        "blocked_actions": [
                            "repeat_sdxl_attention_sdpa_vs_flash2_current_axis",
                            "promote_sdxl_flash2_default_enablement",
                            "write_sdxl_flash2_as_release_gain",
                        ],
                        "current_axis_scope": {
                            "family": "sdxl",
                            "resolution": 1024,
                            "train_batch_size": 1,
                            "baseline_attention_backend": "sdpa",
                            "candidate_attention_backend": "flash2",
                            "dataloader_workers": 0,
                            "dataloader_prefetch_factor": 2,
                        },
                        "summary": dict(comparison),
                        "allowed_next_axes": [
                            "different_attention_backend",
                            "different_sdpa_backend_policy",
                            "heavier_workload_with_stable_eager_anchor",
                            "gpu_telemetry_wrapped_attention_probe",
                        ],
                        "rationale": (
                            "Current SDXL 1024 batch1 flash2 attention probe regressed throughput; do not repeat or "
                            "promote this exact axis without a new hypothesis and gates."
                        ),
                    }
                )
                actions.append(guardrail)
            if (
                group_id == "sdxl_attention_sdpa_cutlass_vs_flash_policy_probe"
                and status == "rollback_recommended"
            ):
                guardrail = _base_action(
                    action_id="guardrail_sdxl_do_not_repeat_sdpa_flash_policy_current_axis",
                    family="sdxl",
                    priority=12,
                    action_type="guardrail",
                    status="active",
                )
                guardrail.update(
                    {
                        "source_group_id": group_id,
                        "reasons": [
                            *_string_list(decision.get("reasons")),
                            "sdpa_flash_policy_throughput_regressed_current_axis",
                        ],
                        "blocked_actions": [
                            "repeat_sdxl_attention_sdpa_cutlass_vs_flash_policy_current_axis",
                            "promote_sdxl_sdpa_flash_policy_default_enablement",
                            "write_sdxl_sdpa_flash_policy_as_release_gain",
                        ],
                        "current_axis_scope": {
                            "family": "sdxl",
                            "resolution": 1024,
                            "train_batch_size": 1,
                            "attention_backend": "sdpa",
                            "baseline_sdpa_backend_policy": "cutlass",
                            "candidate_sdpa_backend_policy": "flash",
                            "dataloader_workers": 0,
                            "dataloader_prefetch_factor": 2,
                        },
                        "summary": dict(comparison),
                        "allowed_next_axes": [
                            "different_sdpa_backend_policy",
                            "different_attention_backend",
                            "heavier_workload_with_stable_eager_anchor",
                            "gpu_telemetry_wrapped_attention_probe",
                        ],
                        "rationale": (
                            "Current SDXL 1024 batch1 SDPA flash policy probe regressed throughput; do not repeat "
                            "or promote this exact policy axis without a new hypothesis and gates."
                        ),
                    }
                )
                actions.append(guardrail)
    return actions


def build_sdxl_probe_actions(
    *,
    sdxl_investigation: Mapping[str, Any],
    sdxl_probe_plan: Mapping[str, Any],
    sdxl_probe_command_plan: Mapping[str, Any],
    sdxl_probe_runner_manifest: Mapping[str, Any] | None = None,
    sdxl_probe_results: Mapping[str, Any],
    sdxl_workload_telemetry_evidence: Mapping[str, Any] | None = None,
    sdxl_workload_stability_manifest: Mapping[str, Any] | None = None,
    sdxl_workload_stability_evidence: Mapping[str, Any] | None = None,
    sdxl_workload_loss_guard_evidence: Mapping[str, Any] | None = None,
    sdxl_alternate_workload_shape_manifest: Mapping[str, Any] | None = None,
    sdxl_alternate_workload_shape_evidence: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_manifest: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_evidence: Mapping[str, Any] | None = None,
    sdxl_batch2_failure_diagnostic_manifest: Mapping[str, Any] | None = None,
    sdxl_batch2_failure_diagnostic_evidence: Mapping[str, Any] | None = None,
    sdxl_batch2_cuda_failure_debug_manifest: Mapping[str, Any] | None = None,
    sdxl_batch2_cuda_failure_debug_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_ab_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_ab_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_evidence: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build guarded SDXL probe actions for the P60 next-action plan."""

    probe_plan_actions = _sdxl_probe_plan_actions(_mapping(sdxl_probe_plan))
    actions: list[dict[str, Any]] = []
    actions.extend(probe_plan_actions or _sdxl_pivot_action(_mapping(sdxl_investigation)))
    actions.extend(_sdxl_probe_command_actions(_mapping(sdxl_probe_command_plan)))
    actions.extend(_sdxl_probe_runner_actions(_mapping(sdxl_probe_runner_manifest), repo_root=repo_root))
    actions.extend(_sdxl_probe_state_refresh_action(_mapping(sdxl_probe_runner_manifest), repo_root=repo_root))
    actions.extend(_sdxl_probe_result_actions(_mapping(sdxl_probe_results)))
    actions.extend(
        build_sdxl_workload_followup_actions(
            sdxl_workload_telemetry_evidence=sdxl_workload_telemetry_evidence,
            sdxl_workload_stability_manifest=sdxl_workload_stability_manifest,
            sdxl_workload_stability_evidence=sdxl_workload_stability_evidence,
            sdxl_workload_loss_guard_evidence=sdxl_workload_loss_guard_evidence,
            sdxl_alternate_workload_shape_manifest=sdxl_alternate_workload_shape_manifest,
            sdxl_alternate_workload_shape_evidence=sdxl_alternate_workload_shape_evidence,
            sdxl_longer_window_batch2_repeat_manifest=sdxl_longer_window_batch2_repeat_manifest,
            sdxl_longer_window_batch2_repeat_evidence=sdxl_longer_window_batch2_repeat_evidence,
            sdxl_batch2_failure_diagnostic_manifest=sdxl_batch2_failure_diagnostic_manifest,
            sdxl_batch2_failure_diagnostic_evidence=sdxl_batch2_failure_diagnostic_evidence,
            sdxl_batch2_cuda_failure_debug_manifest=sdxl_batch2_cuda_failure_debug_manifest,
            sdxl_batch2_cuda_failure_debug_evidence=sdxl_batch2_cuda_failure_debug_evidence,
            repo_root=repo_root,
        )
    )
    actions.extend(
        build_sdxl_compile_followup_actions(
            sdxl_alternate_workload_shape_evidence=sdxl_alternate_workload_shape_evidence,
            sdxl_longer_window_batch2_repeat_manifest=sdxl_longer_window_batch2_repeat_manifest,
            sdxl_longer_window_batch2_repeat_evidence=sdxl_longer_window_batch2_repeat_evidence,
            sdxl_compile_ab_manifest=sdxl_compile_ab_manifest,
            sdxl_compile_ab_evidence=sdxl_compile_ab_evidence,
            sdxl_compile_stable_anchor_manifest=sdxl_compile_stable_anchor_manifest,
            sdxl_compile_stable_anchor_evidence=sdxl_compile_stable_anchor_evidence,
            sdxl_compile_warm_cache_manifest=sdxl_compile_warm_cache_manifest,
            sdxl_compile_warm_cache_evidence=sdxl_compile_warm_cache_evidence,
            repo_root=repo_root,
        )
    )
    return actions


__all__ = ["build_sdxl_probe_actions"]
