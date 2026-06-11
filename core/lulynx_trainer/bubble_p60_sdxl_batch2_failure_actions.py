"""Build P60 actions for SDXL batch2 failure diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


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


def _base_action(*, action_id: str, priority: int, action_type: str, status: str) -> dict[str, Any]:
    return {
        "id": action_id,
        "family": "sdxl",
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


def _report_missing(row: Mapping[str, Any]) -> bool:
    expected = str(row.get("expected_report") or "")
    return not expected or not Path(expected).is_file()


def _pending_rows_and_count(manifest: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    rows = [_mapping(row) for row in manifest.get("commands", []) if _mapping(row)]
    pending_rows = [row for row in rows if _report_missing(row)]
    summary = _mapping(manifest.get("summary"))
    selected_count = _safe_int(summary.get("selected_command_count"))
    summary_pending = _safe_int(summary.get("pending_output_count"))
    if rows and (selected_count <= 0 or len(rows) >= selected_count):
        return pending_rows, len(pending_rows)
    return pending_rows, summary_pending


def _repeat_failed_for_batch2_diagnostic(repeat_evidence: Mapping[str, Any]) -> bool:
    if str(repeat_evidence.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0":
        return False
    summary = _mapping(repeat_evidence.get("summary"))
    return (
        str(repeat_evidence.get("status") or "") in {"execution_failed", "execution_failed_needs_review"}
        or _safe_int(summary.get("candidate_execution_failure_count")) > 0
        or _safe_int(summary.get("missing_summary_count")) > 0
    )


def build_batch2_failure_diagnostic_actions(
    *,
    diagnostic_manifest: Mapping[str, Any],
    diagnostic_evidence: Mapping[str, Any],
    cuda_debug_manifest: Mapping[str, Any] | None = None,
    cuda_debug_evidence: Mapping[str, Any] | None = None,
    repeat_evidence: Mapping[str, Any],
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    actions.extend(
        _runner_actions(diagnostic_manifest, repeat_evidence=repeat_evidence, repo_root=repo_root)
    )
    actions.extend(_result_actions(diagnostic_evidence))
    actions.extend(
        _cuda_debug_runner_actions(
            _mapping(cuda_debug_manifest),
            diagnostic_evidence=diagnostic_evidence,
            repo_root=repo_root,
        )
    )
    actions.extend(_cuda_debug_result_actions(_mapping(cuda_debug_evidence), repo_root=repo_root))
    return actions


def _runner_actions(
    diagnostic_manifest: Mapping[str, Any],
    *,
    repeat_evidence: Mapping[str, Any],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not _repeat_failed_for_batch2_diagnostic(repeat_evidence):
        return []
    if str(diagnostic_manifest.get("report") or "") != "bubble_sdxl_batch2_failure_diagnostic_followup_v0":
        return []
    pending_rows, pending_count = _pending_rows_and_count(diagnostic_manifest)
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_batch2_failure_diagnostic_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_batch2_failure_diagnostic_followup_via_protected_runner",
        priority=24,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(diagnostic_manifest.get("report") or ""),
            "source_repeat_evidence_report": str(repeat_evidence.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "diagnostic_only": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "triggered_only_after_batch2_repeat_failure",
                "activation_recompute_and_microbatch_diagnostics_only",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": (
                "Diagnose the failed SDXL 1024 batch2 backward path with activation recompute and microbatch "
                "accumulation before opening another promotion-oriented repeat axis."
            ),
        }
    )
    return [action]


def _result_actions(diagnostic_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(diagnostic_evidence.get("report") or "") != "bubble_sdxl_batch2_failure_diagnostic_followup_v0":
        return []
    status = str(diagnostic_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(diagnostic_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_batch2_failure_diagnostic_followup_results",
            priority=25,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "diagnostic_only": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_summary_count": _safe_int(summary.get("missing_summary_count")),
                "missing_reports": _string_list(diagnostic_evidence.get("missing_reports")),
                "missing_summaries": _string_list(diagnostic_evidence.get("missing_summaries")),
                "rationale": "Run or reuse missing SDXL batch2 failure diagnostic reports before review.",
            }
        )
        return [action]
    decision = _mapping(diagnostic_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_batch2_failure_diagnostic_followup_results",
        priority=23 if status == "diagnostic_candidate_review" else 32,
        action_type="manual_probe_result_review",
        status="manual_review_ready" if status != "rollback_recommended" else "blocked",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(diagnostic_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "diagnostic_only": True,
            "rationale": "Review batch2 failure diagnostics; positive rows must seed a new protected repeat axis before promotion.",
        }
    )
    return [action, *_cuda_failure_debug_followup_actions(diagnostic_evidence)]


def _cuda_failure_debug_followup_actions(diagnostic_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not _diagnostic_exhausted_current_axis(diagnostic_evidence):
        return []

    decision = _mapping(diagnostic_evidence.get("decision"))
    reasons = _string_list(decision.get("reasons"))
    summary = _mapping(diagnostic_evidence.get("summary"))
    action = _base_action(
        action_id="plan_sdxl_batch2_cuda_failure_debug_followup",
        priority=22,
        action_type="debug_followup_plan",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_evidence_report": str(diagnostic_evidence.get("report") or ""),
            "probe_result_status": str(diagnostic_evidence.get("status") or ""),
            "recommended_action": "build_protected_sdxl_cuda_failure_debug_queue",
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "followup_requires_gpu_if_executed": True,
            "diagnostic_only": True,
            "case_specific_only": True,
            "reasons": [
                "batch2_failure_diagnostic_exhausted_current_axis",
                *reasons,
            ],
            "blocked_actions": [
                "promote_sdxl_batch2_on_sucai_6_lulu",
                "repeat_sdxl_batch2_recompute_current_axis",
                "promote_microbatch_accumulation_as_speedup",
                "write_sdxl_batch2_or_microbatch_release_gain",
            ],
            "candidate_debug_axes": [
                "cuda_launch_blocking_failure_repro",
                "batch2_smaller_resolution_or_bucket_shape_repro",
                "batch2_attention_backend_stability_axis",
                "batch1_control_with_same_debug_flags",
            ],
            "required_next_artifact": "protected_manifest_only_cuda_failure_debug_runner",
            "required_gates": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "same_request_runtime_boundary",
                "no_new_training_entrypoint",
                "no_release_claim_from_debug_evidence",
            ],
            "summary": dict(summary),
            "rationale": (
                "Activation recompute and microbatch accumulation did not clear the current batch2 failure axis; "
                "the next step is a protected CUDA/backward debug queue or smaller-shape repro, not promotion."
            ),
        }
    )
    return [action]


def _diagnostic_exhausted_current_axis(diagnostic_evidence: Mapping[str, Any]) -> bool:
    status = str(diagnostic_evidence.get("status") or "")
    decision = _mapping(diagnostic_evidence.get("decision"))
    reasons = _string_list(decision.get("reasons"))
    summary = _mapping(diagnostic_evidence.get("summary"))
    failure_count = _safe_int(summary.get("execution_failure_count"))
    candidate_count = _safe_int(summary.get("diagnostic_candidate_count"))
    return (
        status in {"execution_failed_needs_review", "needs_review"}
        and candidate_count <= 0
        and (
            failure_count > 0
            or "diagnostic_execution_failed" in reasons
            or "no_diagnostic_candidate_passed_gates" in reasons
        )
    )


def _cuda_debug_runner_actions(
    cuda_debug_manifest: Mapping[str, Any],
    *,
    diagnostic_evidence: Mapping[str, Any],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not _diagnostic_exhausted_current_axis(diagnostic_evidence):
        return []
    if str(cuda_debug_manifest.get("report") or "") != "bubble_sdxl_batch2_cuda_failure_debug_followup_v0":
        return []
    pending_rows, pending_count = _pending_rows_and_count(cuda_debug_manifest)
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_batch2_cuda_failure_debug_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_batch2_cuda_failure_debug_followup_via_protected_runner",
        priority=23,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(cuda_debug_manifest.get("report") or ""),
            "source_diagnostic_evidence_report": str(diagnostic_evidence.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "diagnostic_only": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "triggered_only_after_batch2_diagnostic_exhausted",
                "cuda_launch_blocking_debug_env_only",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": (
                "Run only missing SDXL CUDA/backward debug rows after manual GPU authorization; this is "
                "diagnostic evidence, not batch2 promotion."
            ),
        }
    )
    return [action]


def _cuda_debug_result_actions(cuda_debug_evidence: Mapping[str, Any], *, repo_root: Path | None) -> list[dict[str, Any]]:
    if str(cuda_debug_evidence.get("report") or "") != "bubble_sdxl_batch2_cuda_failure_debug_followup_v0":
        return []
    status = str(cuda_debug_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(cuda_debug_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_batch2_cuda_failure_debug_followup_results",
            priority=24,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "diagnostic_only": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_summary_count": _safe_int(summary.get("missing_summary_count")),
                "missing_reports": _string_list(cuda_debug_evidence.get("missing_reports")),
                "missing_summaries": _string_list(cuda_debug_evidence.get("missing_summaries")),
                "rationale": "Run or reuse missing SDXL CUDA/backward debug reports before review.",
            }
        )
        return [action]
    decision = _mapping(cuda_debug_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_batch2_cuda_failure_debug_followup_results",
        priority=21 if status == "debug_candidate_review" else 30,
        action_type="manual_probe_result_review",
        status="manual_review_ready" if status != "rollback_recommended" else "blocked",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(cuda_debug_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "diagnostic_only": True,
            "rationale": "Review CUDA/backward debug diagnostics; positive rows must seed a new protected repeat axis before promotion.",
        }
    )
    actions = [action]
    if status == "debug_candidate_review":
        repo = Path(repo_root) if repo_root is not None else None
        command: list[str] = []
        expected_outputs: list[str] = []
        if repo is not None:
            command = [
                _path_text(repo / "backend" / "env" / "python-flashattention" / "python.exe"),
                _path_text(repo / "devtools" / "run_bubble_sdxl_batch2_cuda_debug_repeat_followup.py"),
                "--execute",
                "--only-missing",
            ]
            expected_outputs = [
                _path_text(
                    repo
                    / "devtools"
                    / "benchmark_evidence"
                    / "bubble_runtime"
                    / "sdxl_batch2_cuda_debug_repeat_followup_manifest.json"
                ),
                _path_text(
                    repo
                    / "devtools"
                    / "benchmark_evidence"
                    / "bubble_runtime"
                    / "sdxl_batch2_cuda_debug_repeat_evidence.json"
                ),
            ]
        repeat = _base_action(
            action_id="run_sdxl_batch2_cuda_debug_repeat_followup_via_protected_runner",
            priority=22,
            action_type="protected_manual_probe_runner",
            status="manual_review_ready",
        )
        repeat.update(
            {
                "source_debug_evidence_report": str(cuda_debug_evidence.get("report") or ""),
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "diagnostic_only": True,
                "release_claim_allowed": False,
                "command": command,
                "expected_outputs": expected_outputs,
                "safety_checks": [
                    "runner_default_manifest_only",
                    "requires_explicit_execute",
                    "only_missing_supported",
                    "triggered_only_after_cuda_debug_candidate_review",
                    "cuda_launch_blocking_debug_env_only",
                    "same_request_runtime_boundary",
                    "controller_report_only_max_actions_zero",
                    "no_release_claim_from_repeat_manifest",
                ],
                "blocked_actions": [
                    "promote_sdxl_batch2_from_single_debug_run",
                    "write_sdxl_batch2_release_gain_without_repeat_axis",
                    "use_cuda_debug_candidate_as_compile_anchor_before_repeat",
                ],
                "rationale": (
                    "Repeat positive SDXL CUDA/backward debug candidates under a protected axis before "
                    "any batch2, microbatch, or compile-anchor promotion review."
                ),
            }
        )
        actions.append(repeat)
    return actions


def build_cuda_debug_repeat_evidence_actions(repeat_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    repeat_evidence = _mapping(repeat_evidence)
    if str(repeat_evidence.get("report") or "") != "bubble_sdxl_batch2_cuda_debug_repeat_followup_v0":
        return []
    status = str(repeat_evidence.get("status") or "")
    if not status:
        return []
    if status == "repeat_candidate_review":
        return [_cuda_debug_repeat_candidate_review_action(repeat_evidence)]
    return [_cuda_debug_repeat_fail_closed_action(repeat_evidence)]


def _cuda_debug_repeat_fail_closed_action(repeat_evidence: Mapping[str, Any]) -> dict[str, Any]:
    status = str(repeat_evidence.get("status") or "")
    summary = _mapping(repeat_evidence.get("summary"))
    decision = _mapping(repeat_evidence.get("decision"))
    pending = status in {"pending", "pending_manual_gpu_commands", "pending_manual_review"}
    action = _base_action(
        action_id=(
            "pending_sdxl_batch2_cuda_debug_repeat_followup_results"
            if pending
            else "blocked_sdxl_batch2_cuda_debug_repeat_followup_results"
        ),
        priority=22 if pending else 20,
        action_type="pending_manual_probe_results" if pending else "blocked_debug_repeat_evidence",
        status="manual_review_ready" if pending else "blocked",
    )
    action.update(
        {
            "source_evidence_report": str(repeat_evidence.get("report") or ""),
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")) or ["cuda_debug_repeat_not_promotion_ready"],
            "summary": dict(summary),
            "thresholds": dict(_mapping(repeat_evidence.get("thresholds"))),
            "missing_report_count": _safe_int(summary.get("missing_report_count")),
            "missing_summary_count": _safe_int(summary.get("missing_summary_count")),
            "missing_reports": _string_list(repeat_evidence.get("missing_reports")),
            "missing_summaries": _string_list(repeat_evidence.get("missing_summaries")),
            "manual_start_required": pending,
            "requires_gpu_if_executed": pending,
            "diagnostic_only": True,
            "fail_closed": True,
            "blocked_actions": [
                "promote_sdxl_batch2_from_cuda_debug_repeat",
                "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
                "use_cuda_debug_repeat_as_compile_anchor",
                "enable_sdxl_batch2_by_default",
            ],
            "rationale": (
                "SDXL CUDA debug repeat evidence is pending or failed; keep batch2 promotion, release claims, "
                "compile anchoring, and default enablement closed."
            ),
        }
    )
    return action


def _cuda_debug_repeat_candidate_review_action(repeat_evidence: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(repeat_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_batch2_cuda_debug_repeat_followup_results",
        priority=20,
        action_type="manual_promotion_review",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_evidence_report": str(repeat_evidence.get("report") or ""),
            "probe_result_status": "repeat_candidate_review",
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(_mapping(repeat_evidence.get("summary"))),
            "thresholds": dict(_mapping(repeat_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "diagnostic_only": True,
            "case_specific_only": True,
            "allowed_followup_actions": [
                "manual_promotion_review",
                "protected_followup_axis_review",
            ],
            "blocked_actions": [
                "promote_sdxl_batch2_from_cuda_debug_repeat_without_manual_review",
                "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
                "use_cuda_debug_repeat_as_compile_anchor",
                "enable_sdxl_batch2_by_default",
            ],
            "required_gates": [
                "manual_promotion_review_required",
                "protected_followup_axis_review_only",
                "no_release_claim_from_debug_repeat_evidence",
                "no_compile_anchor_from_debug_repeat_evidence",
                "no_batch2_default_enablement_from_debug_repeat_evidence",
            ],
            "rationale": (
                "A positive CUDA debug repeat can only enter manual promotion/protected-followup review; "
                "it must not become release wording, compile-anchor evidence, or default batch2 enablement."
            ),
        }
    )
    return action


__all__ = ["build_batch2_failure_diagnostic_actions", "build_cuda_debug_repeat_evidence_actions"]
