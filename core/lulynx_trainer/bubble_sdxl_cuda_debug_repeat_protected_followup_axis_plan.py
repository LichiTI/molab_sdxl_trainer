"""Protected follow-up axis plan for SDXL CUDA debug repeat review.

This artifact is deliberately JSON-only. It turns a signed, non-release
manual review record into a protected follow-up axis preparation record, while
keeping GPU execution and release claims fail-closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT = "bubble_sdxl_cuda_debug_repeat_protected_followup_axis_plan_v0"
MANUAL_REVIEW_RECORD_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_manual_review_record_v0"
PROMOTION_REVIEW_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_promotion_review_v0"
POST_MANUAL_REBUILD_REPORT = "bubble_post_manual_evidence_rebuild_plan_v0"
DEBUG_REPEAT_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_followup_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _repo_path(repo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def _critical_gates(record: Mapping[str, Any], promotion: Mapping[str, Any]) -> dict[str, bool]:
    has_promotion = bool(promotion)
    return {
        "manual_review_record_present": bool(record),
        "manual_review_record_identity_valid": str(record.get("report") or "") == MANUAL_REVIEW_RECORD_REPORT,
        "manual_review_record_ready": bool(record.get("decision_record_ready"))
        and str(record.get("status") or "") == "decision_record_ready",
        "protected_axis_preparation_approved": bool(record.get("approved_for_protected_followup_axis_preparation")),
        "manual_review_record_release_closed": (
            not bool(record.get("release_claim_allowed"))
            and not bool(record.get("publishable"))
            and not bool(record.get("safe_to_auto_start"))
        ),
        "manual_review_record_diagnostic_only": bool(record.get("diagnostic_only"))
        and bool(record.get("case_specific_only")),
        "promotion_review_identity_valid_if_present": (
            not has_promotion or str(promotion.get("report") or "") == PROMOTION_REVIEW_REPORT
        ),
        "promotion_review_release_closed_if_present": (
            not has_promotion
            or (
                not bool(promotion.get("release_claim_allowed"))
                and not bool(promotion.get("publishable"))
                and not bool(promotion.get("safe_to_auto_start"))
            )
        ),
    }


def _gate_blockers(gates: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _axis_row(*, ready: bool, record: Mapping[str, Any], promotion: Mapping[str, Any]) -> dict[str, Any]:
    promotion_summary = _mapping(promotion.get("summary")) or _mapping(record.get("promotion_review_summary"))
    return {
        "id": "sdxl_cuda_debug_repeat_case_specific_followup_axis",
        "family": "sdxl",
        "axis_kind": "cuda_debug_repeat_case_specific_followup",
        "status": "prepared_manual_execution_required" if ready else "blocked_waiting_for_manual_review_record",
        "ready": ready,
        "manual_start_required": True,
        "requires_gpu_if_executed": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "diagnostic_only": True,
        "case_specific_only": True,
        "source_decision": str(record.get("decision") or ""),
        "comparison_count": _safe_int(promotion_summary.get("comparison_count")),
        "repeat_candidate_pass_count": _safe_int(promotion_summary.get("repeat_candidate_pass_count")),
        "fully_repeated_candidate_count": _safe_int(promotion_summary.get("fully_repeated_candidate_count")),
        "acceptance_gates": [
            "explicit_manual_gpu_execution_only",
            "rebuild_current_combined_evidence_pack_after_any_manual_output",
            "rebuild_natural_load_canary_after_any_manual_output",
            "rebuild_release_claims_after_any_manual_output",
            "keep_sdxl_cuda_debug_repeat_case_specific",
        ],
        "blocked_actions": [
            "auto_start_sdxl_cuda_debug_repeat_followup_axis",
            "enable_sdxl_batch2_by_default",
            "use_cuda_debug_repeat_as_compile_anchor",
            "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
            "skip_sd15_or_natural_load_release_gates",
        ],
    }


def _command_template(
    *,
    command_id: str,
    description: str,
    command: list[str],
    expected_outputs: Sequence[str],
    depends_on: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": command_id,
        "command_id": command_id,
        "description": description,
        "command": command,
        "expected_outputs": list(expected_outputs),
        "depends_on_command_ids": list(depends_on),
        "requires_gpu_if_executed": False,
        "manual_start_required": False,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "status": "json_template_ready",
    }


def _command_templates(repo_root: Path, python_exe: str) -> list[dict[str, Any]]:
    py = _repo_path(repo_root, python_exe)
    runtime = "devtools/benchmark_evidence/bubble_runtime"
    return [
        _command_template(
            command_id="refresh_sdxl_cuda_debug_repeat_protected_followup_axis_plan",
            description="Refresh this JSON-only protected follow-up axis plan after manual review changes.",
            command=[
                py,
                _repo_path(repo_root, "devtools/build_bubble_sdxl_cuda_debug_repeat_protected_followup_axis_plan.py"),
            ],
            expected_outputs=[
                _repo_path(repo_root, f"{runtime}/sdxl_cuda_debug_repeat_protected_followup_axis_plan.json"),
            ],
        ),
        _command_template(
            command_id="refresh_post_manual_evidence_rebuild_plan",
            description="Refresh the post-manual evidence rebuild plan; it still does not run GPU work.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_post_manual_evidence_rebuild_plan.py")],
            expected_outputs=[
                _repo_path(repo_root, f"{runtime}/post_manual_evidence_rebuild_plan.json"),
            ],
            depends_on=["refresh_sdxl_cuda_debug_repeat_protected_followup_axis_plan"],
        ),
        _command_template(
            command_id="refresh_gpu_bubble_readiness_next_actions",
            description="Refresh top-level readiness so release claims remain fail-closed.",
            command=[py, _repo_path(repo_root, "devtools/build_gpu_bubble_experiment_readiness_next_actions.py")],
            expected_outputs=[
                _repo_path(repo_root, f"{runtime}/gpu_bubble_experiment_readiness_next_actions.json"),
            ],
            depends_on=["refresh_post_manual_evidence_rebuild_plan"],
        ),
    ]


def _expected_output_status(outputs: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in outputs:
        path = Path(str(raw))
        rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "state": "already_present" if path.is_file() else "missing_until_manual_execute",
            }
        )
    return rows


def _debug_repeat_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(evidence.get("summary"))
    return {
        "present": bool(evidence),
        "report": str(evidence.get("report") or ""),
        "status": str(evidence.get("status") or ""),
        "source_candidate_count": _safe_int(summary.get("source_candidate_count")),
        "comparison_count": _safe_int(summary.get("comparison_count")),
        "repeat_candidate_pass_count": _safe_int(summary.get("repeat_candidate_pass_count")),
        "fully_repeated_candidate_count": _safe_int(summary.get("fully_repeated_candidate_count")),
        "missing_report_count": _safe_int(summary.get("missing_report_count")),
        "missing_summary_count": _safe_int(summary.get("missing_summary_count")),
        "execution_failure_count": _safe_int(summary.get("execution_failure_count")),
    }


def _debug_repeat_outputs_complete(evidence: Mapping[str, Any]) -> bool:
    summary = _debug_repeat_summary(evidence)
    return (
        summary["report"] == DEBUG_REPEAT_REPORT
        and summary["status"] == "repeat_candidate_review"
        and summary["missing_report_count"] == 0
        and summary["missing_summary_count"] == 0
        and summary["execution_failure_count"] == 0
    )


def _manual_gpu_command_status(*, ready: bool, debug_repeat_evidence: Mapping[str, Any]) -> tuple[str, list[str]]:
    if not ready:
        return "blocked_waiting_for_manual_review_record", ["manual_review_record_required"]
    if _debug_repeat_outputs_complete(debug_repeat_evidence):
        return "not_recommended_outputs_already_present", ["existing_repeat_outputs_already_complete"]
    if str(debug_repeat_evidence.get("report") or "") != DEBUG_REPEAT_REPORT:
        return "blocked_missing_debug_repeat_evidence", ["sdxl_debug_repeat_evidence_required"]
    return "manual_execute_available", []


def _manual_gpu_command_templates(repo_root: Path, python_exe: str, *, ready: bool, debug_repeat_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    py = _repo_path(repo_root, python_exe)
    runtime = "devtools/benchmark_evidence/bubble_runtime"
    debug_evidence = _repo_path(repo_root, f"{runtime}/sdxl_batch2_cuda_failure_debug_evidence.json")
    out_root = _repo_path(repo_root, f"{runtime}/sdxl_batch2_cuda_debug_repeat_followup_runs")
    manifest = _repo_path(repo_root, f"{runtime}/sdxl_batch2_cuda_debug_repeat_followup_manifest.json")
    evidence = _repo_path(repo_root, f"{runtime}/sdxl_batch2_cuda_debug_repeat_evidence.json")
    source_data = _repo_path(repo_root, "sucai/6_lulu")
    base_command = [
        py,
        _repo_path(repo_root, "devtools/run_bubble_sdxl_batch2_cuda_debug_repeat_followup.py"),
        "--debug-evidence",
        debug_evidence,
        "--out-root",
        out_root,
        "--json-out",
        manifest,
        "--evidence-out",
        evidence,
        "--source-data",
        source_data,
        "--repeat-count",
        "2",
        "--only-missing",
    ]
    status, blockers = _manual_gpu_command_status(ready=ready, debug_repeat_evidence=debug_repeat_evidence)
    outputs = [manifest, evidence]
    return [
        {
            "id": "dry_run_sdxl_cuda_debug_repeat_protected_followup_axis_manifest",
            "command_id": "dry_run_sdxl_cuda_debug_repeat_protected_followup_axis_manifest",
            "description": "Render the protected repeat manifest without running GPU work.",
            "status": "json_template_ready" if ready else status,
            "ready": ready,
            "command": base_command,
            "expected_outputs": outputs,
            "expected_output_status": _expected_output_status(outputs),
            "requires_gpu_if_executed": False,
            "manual_start_required": False,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_success": False,
            "diagnostic_only": True,
            "case_specific_only": True,
            "blockers": [] if ready else blockers,
        },
        {
            "id": "manual_execute_sdxl_cuda_debug_repeat_protected_followup_axis_only_missing",
            "command_id": "manual_execute_sdxl_cuda_debug_repeat_protected_followup_axis_only_missing",
            "description": "Explicit manual GPU execution template; keep --only-missing to avoid repeating completed rows.",
            "status": status,
            "ready": status == "manual_execute_available",
            "command": [*base_command, "--execute"],
            "expected_outputs": outputs,
            "expected_output_status": _expected_output_status(outputs),
            "requires_gpu_if_executed": True,
            "manual_start_required": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_success": False,
            "diagnostic_only": True,
            "case_specific_only": True,
            "blockers": blockers,
            "blocked_actions": [
                "auto_start_sdxl_cuda_debug_repeat_followup_axis",
                "repeat_completed_rows_without_missing_outputs",
                "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
                "skip_post_manual_evidence_rebuild",
            ],
        },
    ]


def _rebuild_plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(plan),
        "report": str(plan.get("report") or ""),
        "status": str(plan.get("status") or ""),
        "command_count": _safe_int(plan.get("command_count")),
        "ready_command_count": _safe_int(plan.get("ready_command_count")),
        "blockers": _strings(plan.get("blockers")),
        "next_rebuild_stage_id": str(_mapping(plan.get("next_rebuild_stage")).get("stage_id") or ""),
    }


def build_sdxl_cuda_debug_repeat_protected_followup_axis_plan(
    *,
    repo_root: Path,
    manual_review_record: Mapping[str, Any] | None = None,
    promotion_review: Mapping[str, Any] | None = None,
    sdxl_debug_repeat_evidence: Mapping[str, Any] | None = None,
    post_manual_evidence_rebuild_plan: Mapping[str, Any] | None = None,
    python_exe: str = "backend/env/python-flashattention/python.exe",
) -> dict[str, Any]:
    """Build a fail-closed protected-axis preparation artifact."""

    repo = Path(repo_root)
    record = _mapping(manual_review_record)
    promotion = _mapping(promotion_review)
    debug_repeat = _mapping(sdxl_debug_repeat_evidence)
    rebuild_plan = _mapping(post_manual_evidence_rebuild_plan)
    gates = _critical_gates(record, promotion)
    blockers = _gate_blockers(gates)
    ready = not blockers
    axes = [_axis_row(ready=ready, record=record, promotion=promotion)]
    commands = _command_templates(repo, python_exe)
    manual_commands = _manual_gpu_command_templates(
        repo,
        python_exe,
        ready=ready,
        debug_repeat_evidence=debug_repeat,
    )
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "protected_followup_axis_plan_ready" if ready else "blocked_waiting_for_manual_review_record",
        "ok": ready,
        "protected_followup_axis_plan_ready": ready,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "diagnostic_only": True,
        "case_specific_only": True,
        "manual_start_required": True,
        "requires_gpu_if_executed": False,
        "followup_requires_gpu_if_executed": True,
        "prepared_axis_count": len(axes),
        "ready_axis_count": sum(1 for item in axes if bool(item.get("ready"))),
        "gpu_heavy_axis_count": sum(1 for item in axes if bool(item.get("requires_gpu_if_executed"))),
        "auto_startable_axis_count": sum(1 for item in axes if bool(item.get("safe_to_auto_start"))),
        "command_template_count": len(commands),
        "manual_gpu_command_template_count": len(manual_commands),
        "manual_gpu_execute_command_count": sum(1 for item in manual_commands if bool(item.get("requires_gpu_if_executed"))),
        "manual_gpu_execute_ready_count": sum(1 for item in manual_commands if bool(item.get("ready")) and bool(item.get("requires_gpu_if_executed"))),
        "critical_gates": gates,
        "blockers": blockers,
        "protected_axes": axes,
        "command_templates": commands,
        "manual_gpu_command_templates": manual_commands,
        "sdxl_debug_repeat_evidence_summary": _debug_repeat_summary(debug_repeat),
        "post_manual_evidence_rebuild_plan_summary": _rebuild_plan_summary(rebuild_plan),
        "allowed_followup_actions": [
            "refresh_post_manual_evidence_rebuild_plan",
            "refresh_gpu_bubble_readiness_next_actions",
            "manual_execute_protected_followup_axis_only_after_explicit_start",
        ]
        if ready
        else [],
        "blocked_actions": [
            "auto_start_sdxl_cuda_debug_repeat_followup_axis",
            "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
            "enable_sdxl_batch2_by_default",
            "use_cuda_debug_repeat_as_compile_anchor",
            "promote_microbatch_or_universal_gpu_utilization_claim",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "acceptance_gates": [
            "manual_review_record_ready_and_approved",
            "protected_axis_plan_is_json_only",
            "manual_gpu_output_requires_post_manual_evidence_rebuild",
            "release_claim_allowed_remains_false",
            "sd15_and_natural_load_release_gates_cannot_be_skipped",
        ],
        "recommended_next_action": (
            "refresh_post_manual_evidence_rebuild_plan_without_release_claim"
            if ready
            else "record_signed_manual_review_before_preparing_axis"
        ),
        "notes": [
            "This artifact prepares a protected follow-up axis but never starts it.",
            "Any manual GPU output must be rebuilt through current_combined evidence before release review.",
            "SDXL CUDA debug repeat remains case-specific diagnostic evidence, not a release claim.",
        ],
    }


__all__ = [
    "MANUAL_REVIEW_RECORD_REPORT",
    "DEBUG_REPEAT_REPORT",
    "POST_MANUAL_REBUILD_REPORT",
    "PROMOTION_REVIEW_REPORT",
    "REPORT",
    "build_sdxl_cuda_debug_repeat_protected_followup_axis_plan",
]
