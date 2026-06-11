"""Build P60 SDXL workload-shape follow-up actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.core.lulynx_trainer.bubble_p60_sdxl_batch2_failure_actions import (
    build_batch2_failure_diagnostic_actions,
)
from backend.core.lulynx_trainer.bubble_p60_sdxl_workload_guardrails import (
    batch2_repeat_cuda_failure_guardrail,
    workload_telemetry_no_gpu_98_claim_guardrail,
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


def _base_action(*, action_id: str, family: str, priority: int, action_type: str, status: str) -> dict[str, Any]:
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


def _is_workload_telemetry_evidence(report: Mapping[str, Any]) -> bool:
    return str(report.get("report") or "") == "bubble_advisor_ab_evidence_v0" and (
        str(report.get("source_followup_report") or "") == "bubble_sdxl_workload_telemetry_followup_v0"
        or str(report.get("evidence_profile") or "") == "sdxl_workload_telemetry_followup_v0"
    )


def _telemetry_actions(telemetry_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not _is_workload_telemetry_evidence(telemetry_evidence):
        return []
    status = str(telemetry_evidence.get("status") or "")
    action = _base_action(
        action_id="review_sdxl_workload_telemetry_followup_results",
        family="sdxl",
        priority=26,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    decision = _mapping(telemetry_evidence.get("decision"))
    action.update(
        {
            "source_evidence_report": str(telemetry_evidence.get("report") or ""),
            "source_followup_report": str(telemetry_evidence.get("source_followup_report") or ""),
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "comparison": dict(_mapping(telemetry_evidence.get("comparison"))),
            "loss_regression_ratio": decision.get("loss_regression_ratio"),
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "required_followup": "sdxl_workload_stability_followup",
            "required_gates": [
                "repeat_stability_required",
                "longer_window_required",
                "loss_stability_required",
                "active_gpu_telemetry_required",
                "vram_ratio_within_limit",
                "case_specific_release_wording_only",
            ],
            "rationale": "Telemetry follow-up shows the SDXL workload-shape signal, but release promotion needs repeated stability evidence.",
        }
    )
    return [action, *workload_telemetry_no_gpu_98_claim_guardrail(telemetry_evidence)]


def _stability_runner_actions(
    telemetry_evidence: Mapping[str, Any],
    stability_manifest: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    manifest_valid = str(stability_manifest.get("report") or "") == "bubble_sdxl_workload_stability_followup_v0"
    telemetry_valid = _is_workload_telemetry_evidence(telemetry_evidence)
    telemetry_status = str(telemetry_evidence.get("status") or "")
    if not manifest_valid and telemetry_status not in {"needs_review", "keep_recommended"}:
        return []
    if not manifest_valid and not telemetry_valid:
        return []
    pending_rows, pending_count = _pending_rows_and_count(stability_manifest)
    if not manifest_valid and pending_count <= 0:
        pending_count = 4
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_workload_stability_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_workload_stability_followup_via_protected_runner",
        family="sdxl",
        priority=27,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_evidence_report": str(telemetry_evidence.get("report") or ""),
            "source_manifest_report": str(stability_manifest.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "repeat_stability_pairing",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": "Run the longer repeated SDXL batch1 vs batch2 telemetry follow-up before promoting workload-shape settings.",
        }
    )
    return [action]


def _stability_result_actions(stability_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(stability_evidence.get("report") or "") != "bubble_sdxl_workload_stability_followup_v0":
        return []
    status = str(stability_evidence.get("status") or "")
    if not status:
        return []
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_workload_stability_followup_results",
            family="sdxl",
            priority=28,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        summary = _mapping(stability_evidence.get("summary"))
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_reports": _string_list(stability_evidence.get("missing_reports")),
                "rationale": "Run or reuse the missing repeated SDXL workload telemetry reports before stability can be evaluated.",
            }
        )
        return [action]
    action = _base_action(
        action_id="review_sdxl_workload_stability_followup_results",
        family="sdxl",
        priority=24 if status == "stable_candidate_review" else 29,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    decision = _mapping(stability_evidence.get("decision"))
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(_mapping(stability_evidence.get("summary"))),
            "thresholds": dict(_mapping(stability_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "rationale": "Review repeated SDXL workload-shape stability evidence before any case-specific promotion.",
        }
    )
    return [action]


def _loss_guard_actions(loss_guard_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(loss_guard_evidence.get("report") or "") != "bubble_sdxl_workload_loss_guard_followup_v0":
        return []
    status = str(loss_guard_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(loss_guard_evidence.get("summary"))
    actions: list[dict[str, Any]] = []
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_workload_loss_guard_followup_results",
            family="sdxl",
            priority=28,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_reports": _string_list(loss_guard_evidence.get("missing_reports")),
                "rationale": "Run or reuse the missing SDXL workload loss-guard reports before batch2 can be reviewed.",
            }
        )
        return [action]
    decision = _mapping(loss_guard_evidence.get("decision"))
    review = _base_action(
        action_id="review_sdxl_workload_loss_guard_followup_results",
        family="sdxl",
        priority=23,
        action_type="manual_probe_result_review",
        status="manual_review_ready" if status != "rollback_recommended" else "blocked",
    )
    review.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "rationale": "Review SDXL workload loss-guard evidence before any batch-size promotion.",
        }
    )
    actions.append(review)
    if _safe_int(summary.get("loss_guard_candidate_count")) <= 0 and status in {"needs_review", "rollback_recommended"}:
        guardrail = _base_action(
            action_id="guardrail_sdxl_do_not_promote_batch2_current_axis",
            family="sdxl",
            priority=6,
            action_type="guardrail",
            status="active",
        )
        guardrail.update(
            {
                "blocked_actions": [
                    "promote_sdxl_batch2_on_sucai_6_lulu",
                    "write_batch2_as_release_gain",
                    "promote_sdxl_alternate_workload_shape_without_loss_gate",
                ],
                "reasons": [
                    "loss_guard_candidate_count_zero",
                    "batch2_loss_guard_failed_current_axis",
                    "alternate_workload_shape_requires_case_specific_review",
                    *_string_list(decision.get("reasons")),
                ],
                "rationale": "Batch2 improved throughput but failed loss-guard follow-up; keep it out of release matrix on this source axis.",
            }
        )
        actions.append(guardrail)
        pivot = _base_action(
            action_id="review_sdxl_alternate_workload_shape_axes_after_batch2_loss_guard",
            family="sdxl",
            priority=24,
            action_type="manual_probe_design",
            status="manual_review_ready",
        )
        pivot.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "candidate_axes": [
                    "resolution_or_bucket_shape_control",
                    "token_length_or_caption_bucket_stability_control",
                    "loss_metric_fairness_review",
                    "longer_window_same_effective_samples_review",
                ],
                "required_gates": [
                    "throughput_gain_over_baseline",
                    "loss_stability_required",
                    "active_gpu_telemetry_required",
                    "vram_ratio_within_limit",
                    "case_specific_release_wording_only",
                ],
                "rationale": "Batch2/LR/effective-batch probes did not pass loss gates; pivot to other workload-shape or loss-metric review axes.",
            }
        )
        actions.append(pivot)
    return actions


def _alternate_runner_actions(alternate_manifest: Mapping[str, Any], loss_guard_evidence: Mapping[str, Any], *, repo_root: Path | None) -> list[dict[str, Any]]:
    if str(alternate_manifest.get("report") or "") != "bubble_sdxl_alternate_workload_shape_followup_v0":
        return []
    pending_rows, pending_count = _pending_rows_and_count(alternate_manifest)
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_alternate_workload_shape_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_alternate_workload_shape_followup_via_protected_runner",
        family="sdxl",
        priority=25,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(alternate_manifest.get("report") or ""),
            "source_loss_guard_report": str(loss_guard_evidence.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
                "alternate_workload_shape_not_batch2_promotion",
            ],
            "rationale": "Run missing SDXL alternate workload-shape follow-up commands after manual GPU authorization.",
        }
    )
    return [action]


def _alternate_result_actions(alternate_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(alternate_evidence.get("report") or "") != "bubble_sdxl_alternate_workload_shape_followup_v0":
        return []
    status = str(alternate_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(alternate_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_alternate_workload_shape_followup_results",
            family="sdxl",
            priority=26,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_reports": _string_list(alternate_evidence.get("missing_reports")),
                "rationale": "Run or reuse missing SDXL alternate workload-shape reports before evidence can be reviewed.",
            }
        )
        return [action]
    decision = _mapping(alternate_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_alternate_workload_shape_followup_results",
        family="sdxl",
        priority=24 if status == "alternate_candidate_review" else 30,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(alternate_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "rationale": "Review SDXL alternate workload-shape evidence without promoting batch2 or diagnostic resolution rows.",
        }
    )
    return [action]


def _alternate_has_longer_window_candidate(alternate_evidence: Mapping[str, Any]) -> bool:
    if str(alternate_evidence.get("report") or "") != "bubble_sdxl_alternate_workload_shape_followup_v0":
        return False
    if str(alternate_evidence.get("status") or "") != "alternate_candidate_review":
        return False
    summary = _mapping(alternate_evidence.get("summary"))
    return _safe_int(summary.get("alternate_candidate_count")) > 0


def _repeat_runner_actions(
    repeat_manifest: Mapping[str, Any],
    alternate_evidence: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not _alternate_has_longer_window_candidate(alternate_evidence):
        return []
    pending_rows, pending_count = _pending_rows_and_count(repeat_manifest)
    manifest_valid = str(repeat_manifest.get("report") or "") == "bubble_sdxl_longer_window_batch2_repeat_followup_v0"
    if not manifest_valid:
        pending_count = 4
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_longer_window_batch2_repeat_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_longer_window_batch2_repeat_followup_via_protected_runner",
        family="sdxl",
        priority=25,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(repeat_manifest.get("report") or ""),
            "source_alternate_evidence_report": str(alternate_evidence.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "repeat_longer_window_pairing",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
                "case_specific_promotion_review_only",
            ],
            "rationale": "Repeat the SDXL 1024 batch2 longer-window candidate before any case-specific promotion review.",
        }
    )
    return [action]


def _repeat_result_actions(repeat_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(repeat_evidence.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0":
        return []
    status = str(repeat_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(repeat_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_longer_window_batch2_repeat_followup_results",
            family="sdxl",
            priority=26,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_reports": _string_list(repeat_evidence.get("missing_reports")),
                "rationale": "Run or reuse missing repeated SDXL 1024 batch2 reports before repeat evidence can be reviewed.",
            }
        )
        return [action]
    decision = _mapping(repeat_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_longer_window_batch2_repeat_followup_results",
        family="sdxl",
        priority=22 if status == "repeat_candidate_review" else 31,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(repeat_evidence.get("thresholds"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "rationale": "Review repeated SDXL 1024 batch2 longer-window evidence before any case-specific promotion.",
        }
    )
    return [action, *batch2_repeat_cuda_failure_guardrail(repeat_evidence)]


def build_sdxl_workload_followup_actions(
    *,
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
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    telemetry = _mapping(sdxl_workload_telemetry_evidence)
    stability_manifest = _mapping(sdxl_workload_stability_manifest)
    loss_guard = _mapping(sdxl_workload_loss_guard_evidence)
    actions: list[dict[str, Any]] = []
    actions.extend(_telemetry_actions(telemetry))
    actions.extend(_stability_runner_actions(telemetry, stability_manifest, repo_root=repo_root))
    actions.extend(_stability_result_actions(_mapping(sdxl_workload_stability_evidence)))
    actions.extend(_loss_guard_actions(loss_guard))
    actions.extend(_alternate_runner_actions(_mapping(sdxl_alternate_workload_shape_manifest), loss_guard, repo_root=repo_root))
    actions.extend(_alternate_result_actions(_mapping(sdxl_alternate_workload_shape_evidence)))
    actions.extend(
        _repeat_runner_actions(
            _mapping(sdxl_longer_window_batch2_repeat_manifest),
            _mapping(sdxl_alternate_workload_shape_evidence),
            repo_root=repo_root,
        )
    )
    repeat_evidence = _mapping(sdxl_longer_window_batch2_repeat_evidence)
    actions.extend(_repeat_result_actions(repeat_evidence))
    actions.extend(
        build_batch2_failure_diagnostic_actions(
            diagnostic_manifest=_mapping(sdxl_batch2_failure_diagnostic_manifest),
            diagnostic_evidence=_mapping(sdxl_batch2_failure_diagnostic_evidence),
            cuda_debug_manifest=_mapping(sdxl_batch2_cuda_failure_debug_manifest),
            cuda_debug_evidence=_mapping(sdxl_batch2_cuda_failure_debug_evidence),
            repeat_evidence=repeat_evidence,
            repo_root=repo_root,
        )
    )
    return actions


__all__ = ["build_sdxl_workload_followup_actions"]
