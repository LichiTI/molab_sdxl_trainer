"""Build P60 SDXL compile A/B follow-up actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.core.lulynx_trainer.compile_randomization_contract import (
    sdxl_compile_randomization_contract,
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


def _positive(value: Any) -> bool:
    return _safe_int(value) > 0


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


def _repeat_manifest_block_reasons(repeat_manifest: Mapping[str, Any]) -> list[str]:
    if str(repeat_manifest.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0":
        return []
    reasons: list[str] = []
    if str(repeat_manifest.get("status") or "") == "execution_failed":
        reasons.append("repeat_manifest_execution_failed")
    if repeat_manifest.get("ok") is False:
        reasons.append("repeat_manifest_not_ok")
    summary = _mapping(repeat_manifest.get("summary"))
    if _positive(summary.get("execution_failure_count")):
        reasons.append("repeat_manifest_execution_failure_count_positive")
    for row in repeat_manifest.get("commands", []):
        item = _mapping(row)
        execution = _mapping(item.get("execution"))
        if not execution or not execution.get("executed"):
            continue
        command_id = str(item.get("id") or "unknown")
        if execution.get("ok") is False:
            reasons.append(f"repeat_command_failed:{command_id}")
        if execution.get("summary_exists_after") is False:
            reasons.append(f"repeat_summary_missing:{command_id}")
    return reasons


def _repeat_evidence_block_reasons(repeat_evidence: Mapping[str, Any]) -> list[str]:
    if str(repeat_evidence.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0":
        return []
    reasons: list[str] = []
    status = str(repeat_evidence.get("status") or "")
    summary = _mapping(repeat_evidence.get("summary"))
    thresholds = _mapping(repeat_evidence.get("thresholds"))
    required = max(_safe_int(thresholds.get("required_repeat_count"), 2), 1)
    if status != "repeat_candidate_review":
        reasons.append(f"repeat_evidence_status_not_valid_anchor:{status or 'missing'}")
    if _safe_int(summary.get("completed_pair_count")) < required:
        reasons.append("repeat_completed_pair_count_below_required")
    if _positive(summary.get("missing_report_count")):
        reasons.append("repeat_missing_report_count_positive")
    if _positive(summary.get("missing_summary_count")):
        reasons.append("repeat_missing_summary_count_positive")
    if _positive(summary.get("execution_failure_count")):
        reasons.append("repeat_execution_failure_count_positive")
    if _positive(summary.get("candidate_execution_failure_count")):
        reasons.append("repeat_candidate_execution_failure_count_positive")
    decision_reasons = set(_string_list(_mapping(repeat_evidence.get("decision")).get("reasons")))
    bad_tokens = ("missing", "failed", "insufficient", "threshold_not_met", "regressed", "rollback")
    for reason in sorted(decision_reasons):
        if any(token in reason for token in bad_tokens):
            reasons.append(f"repeat_decision_reason_blocked:{reason}")
    return reasons


def _has_valid_repeat_candidate_anchor(
    *,
    repeat_evidence: Mapping[str, Any],
    repeat_manifest: Mapping[str, Any],
) -> bool:
    if str(repeat_evidence.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0":
        return False
    return not _repeat_evidence_block_reasons(repeat_evidence) and not _repeat_manifest_block_reasons(repeat_manifest)


def _has_candidate_anchor(
    *,
    alternate_evidence: Mapping[str, Any],
    repeat_evidence: Mapping[str, Any],
    repeat_manifest: Mapping[str, Any],
) -> bool:
    if _has_valid_repeat_candidate_anchor(repeat_evidence=repeat_evidence, repeat_manifest=repeat_manifest):
        return True
    if (
        str(alternate_evidence.get("report") or "") == "bubble_sdxl_alternate_workload_shape_followup_v0"
        and str(alternate_evidence.get("status") or "") == "alternate_candidate_review"
        and _safe_int(_mapping(alternate_evidence.get("summary")).get("alternate_candidate_count")) > 0
        and str(repeat_evidence.get("report") or "") != "bubble_sdxl_longer_window_batch2_repeat_followup_v0"
    ):
        return True
    return False


def _blocked_anchor_actions(
    *,
    repeat_evidence: Mapping[str, Any],
    repeat_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence_reasons = _repeat_evidence_block_reasons(repeat_evidence)
    manifest_reasons = _repeat_manifest_block_reasons(repeat_manifest)
    if not evidence_reasons and not manifest_reasons:
        return []
    action = _base_action(
        action_id="blocked_sdxl_compile_ab_followup_until_repeat_evidence_valid",
        priority=24,
        action_type="blocked_compile_anchor",
        status="blocked",
    )
    action.update(
        {
            "blocked_actions": ["run_sdxl_compile_ab_followup_via_protected_runner"],
            "reasons": [*evidence_reasons, *manifest_reasons],
            "source_repeat_evidence_report": str(repeat_evidence.get("report") or ""),
            "source_repeat_manifest_report": str(repeat_manifest.get("report") or ""),
            "randomization_compat_contract": sdxl_compile_randomization_contract(),
            "rationale": (
                "SDXL compile A/B needs a stable eager batch2 anchor first; current repeat evidence has "
                "execution, summary, or threshold problems, so compile would mix anchor instability with compile effects."
            ),
        }
    )
    return [action]


def _compile_runner_actions(
    compile_manifest: Mapping[str, Any],
    *,
    alternate_evidence: Mapping[str, Any],
    repeat_evidence: Mapping[str, Any],
    repeat_manifest: Mapping[str, Any],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not _has_candidate_anchor(
        alternate_evidence=alternate_evidence,
        repeat_evidence=repeat_evidence,
        repeat_manifest=repeat_manifest,
    ):
        return []
    pending_rows, pending_count = _pending_rows_and_count(compile_manifest)
    manifest_valid = str(compile_manifest.get("report") or "") == "bubble_sdxl_compile_ab_followup_v0"
    if not manifest_valid:
        pending_count = 2
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_compile_ab_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_compile_ab_followup_via_protected_runner",
        priority=26,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(compile_manifest.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "randomization_compat_contract": dict(
                _mapping(compile_manifest.get("randomization_compat_contract"))
                or sdxl_compile_randomization_contract()
            ),
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "baseline_eager_candidate_compile",
                "per_block_compile_only",
                "randomization_dynamic_path_contract_required",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": "Run SDXL eager vs per-block torch.compile A/B only after a workload-shape candidate exists.",
        }
    )
    return [action]


def _compile_result_actions(compile_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(compile_evidence.get("report") or "") != "bubble_sdxl_compile_ab_followup_v0":
        return []
    status = str(compile_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(compile_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_compile_ab_followup_results",
            priority=27,
            action_type="pending_manual_probe_results",
            status="manual_review_ready",
        )
        action.update(
            {
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "missing_report_count": _safe_int(summary.get("missing_report_count")),
                "missing_reports": _string_list(compile_evidence.get("missing_reports")),
                "rationale": "Run or reuse missing SDXL compile A/B reports before compile evidence can be reviewed.",
            }
        )
        return [action]
    decision = _mapping(compile_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_compile_ab_followup_results",
        priority=23 if status == "compile_candidate_review" else 32,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(compile_evidence.get("thresholds"))),
            "randomization_compat_contract": dict(_mapping(compile_evidence.get("randomization_compat_contract"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "rationale": "Review SDXL compile evidence with throughput, active GPU, loss, VRAM and randomization compatibility gates.",
        }
    )
    return [action]


def _stable_anchor_runner_actions(
    stable_manifest: Mapping[str, Any],
    *,
    blocked_by_batch2_anchor: bool,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not blocked_by_batch2_anchor:
        return []
    pending_rows, pending_count = _pending_rows_and_count(stable_manifest)
    manifest_valid = str(stable_manifest.get("report") or "") == "bubble_sdxl_compile_stable_anchor_followup_v0"
    if not manifest_valid:
        pending_count = 2
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_compile_stable_anchor_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_compile_stable_anchor_followup_via_protected_runner",
        priority=28,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(stable_manifest.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "diagnostic_only": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "randomization_compat_contract": dict(
                _mapping(stable_manifest.get("randomization_compat_contract"))
                or sdxl_compile_randomization_contract()
            ),
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "diagnostic_only_not_release_anchor",
                "batch1_stable_anchor_only",
                "does_not_validate_batch2",
                "per_block_compile_only",
                "randomization_dynamic_path_contract_required",
                "workers_prefetch_fixed",
                "controller_report_only_max_actions_zero",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": (
                "Batch2 compile A/B is blocked by unstable eager repeat evidence; run batch1 eager vs "
                "per-block compile only as a stable-anchor diagnostic."
            ),
        }
    )
    return [action]


def _stable_anchor_result_actions(stable_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(stable_evidence.get("report") or "") != "bubble_sdxl_compile_stable_anchor_followup_v0":
        return []
    status = str(stable_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(stable_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_compile_stable_anchor_followup_results",
            priority=29,
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
                "missing_reports": _string_list(stable_evidence.get("missing_reports")),
                "missing_summaries": _string_list(stable_evidence.get("missing_summaries")),
                "rationale": "Run or reuse missing SDXL stable-anchor compile diagnostic reports before review.",
            }
        )
        return [action]
    decision = _mapping(stable_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_compile_stable_anchor_followup_results",
        priority=27 if status == "stable_anchor_compile_candidate_review" else 33,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(stable_evidence.get("thresholds"))),
            "randomization_compat_contract": dict(_mapping(stable_evidence.get("randomization_compat_contract"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "diagnostic_only": True,
            "rationale": (
                "Review stable batch1 compile evidence as compile-path signal only; it does not validate batch2 or release acceleration."
            ),
        }
    )
    return [action]


def _stable_anchor_suggests_warm_cache_probe(stable_evidence: Mapping[str, Any]) -> bool:
    if str(stable_evidence.get("report") or "") != "bubble_sdxl_compile_stable_anchor_followup_v0":
        return False
    status = str(stable_evidence.get("status") or "")
    if status not in {"needs_review", "stable_anchor_compile_candidate_review"}:
        return False
    reasons = set(_string_list(_mapping(stable_evidence.get("decision")).get("reasons")))
    return bool(
        reasons
        & {
            "compile_cold_start_overhead_observed",
            "stable_anchor_compile_threshold_not_met",
            "steady_throughput_gain_below_threshold",
            "active_gpu_delta_below_threshold",
        }
    )


def _warm_cache_runner_actions(
    warm_manifest: Mapping[str, Any],
    *,
    stable_evidence: Mapping[str, Any],
    warm_evidence: Mapping[str, Any],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if not _stable_anchor_suggests_warm_cache_probe(stable_evidence):
        return []
    warm_status = str(warm_evidence.get("status") or "")
    if str(warm_evidence.get("report") or "") == "bubble_sdxl_compile_warm_cache_followup_v0" and warm_status not in {
        "",
        "pending_manual_gpu_commands",
    }:
        return []
    command_rows = [
        _mapping(row)
        for row in warm_manifest.get("commands", [])
        if _mapping(row) and not _mapping(row).get("reference_only")
    ]
    pending_rows = [row for row in command_rows if _report_missing(row)]
    pending_count = len(pending_rows)
    manifest_valid = str(warm_manifest.get("report") or "") == "bubble_sdxl_compile_warm_cache_followup_v0"
    if not manifest_valid:
        pending_count = 1
    if pending_count <= 0:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "run_bubble_sdxl_compile_warm_cache_followup.py"),
            "--execute",
            "--only-missing",
        ]
    action = _base_action(
        action_id="run_sdxl_compile_warm_cache_followup_via_protected_runner",
        priority=34,
        action_type="protected_manual_probe_runner",
        status="manual_review_ready",
    )
    action.update(
        {
            "source_manifest_report": str(warm_manifest.get("report") or ""),
            "manual_start_required": True,
            "requires_gpu_if_executed": True,
            "diagnostic_only": True,
            "pending_ready_command_count": pending_count,
            "pending_command_ids": [str(row.get("id") or "") for row in pending_rows],
            "expected_reports": [str(row.get("expected_report") or "") for row in pending_rows],
            "command": command,
            "randomization_compat_contract": dict(
                _mapping(warm_manifest.get("randomization_compat_contract"))
                or sdxl_compile_randomization_contract()
            ),
            "safety_checks": [
                "runner_default_manifest_only",
                "requires_explicit_execute",
                "only_missing_supported",
                "diagnostic_only_not_release_anchor",
                "batch1_stable_anchor_only",
                "does_not_validate_batch2",
                "per_block_compile_only",
                "compile_cache_reuse_probe",
                "stable_eager_baseline_reference_only",
                "no_release_claim_from_runner_manifest",
            ],
            "rationale": (
                "Stable-anchor compile ran but did not clear gates and showed large cold-start overhead; "
                "run a warm-cache per-block compile candidate to separate steady compile benefit from cold-start cost."
            ),
        }
    )
    return [action]


def _warm_cache_result_actions(warm_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(warm_evidence.get("report") or "") != "bubble_sdxl_compile_warm_cache_followup_v0":
        return []
    status = str(warm_evidence.get("status") or "")
    if not status:
        return []
    summary = _mapping(warm_evidence.get("summary"))
    if status == "pending_manual_gpu_commands":
        action = _base_action(
            action_id="pending_sdxl_compile_warm_cache_followup_results",
            priority=35,
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
                "missing_reports": _string_list(warm_evidence.get("missing_reports")),
                "missing_summaries": _string_list(warm_evidence.get("missing_summaries")),
                "rationale": "Run or reuse missing SDXL warm-cache compile diagnostic reports before review.",
            }
        )
        return [action]
    decision = _mapping(warm_evidence.get("decision"))
    action = _base_action(
        action_id="review_sdxl_compile_warm_cache_followup_results",
        priority=30 if status == "warm_cache_compile_candidate_review" else 36,
        action_type="manual_probe_result_review",
        status="blocked" if status == "rollback_recommended" else "manual_review_ready",
    )
    action.update(
        {
            "probe_result_status": status,
            "recommended_action": str(decision.get("recommended_action") or ""),
            "reasons": _string_list(decision.get("reasons")),
            "summary": dict(summary),
            "thresholds": dict(_mapping(warm_evidence.get("thresholds"))),
            "randomization_compat_contract": dict(_mapping(warm_evidence.get("randomization_compat_contract"))),
            "release_claim_allowed": False,
            "manual_start_required": False,
            "requires_gpu_if_executed": False,
            "diagnostic_only": True,
            "rationale": (
                "Review warm-cache compile evidence as steady compile-path signal only; it does not validate batch2 or release acceleration."
            ),
        }
    )
    return [action]


def _compile_current_axis_guardrail_actions(
    *,
    stable_evidence: Mapping[str, Any],
    warm_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if str(stable_evidence.get("report") or "") != "bubble_sdxl_compile_stable_anchor_followup_v0":
        return []
    if str(warm_evidence.get("report") or "") != "bubble_sdxl_compile_warm_cache_followup_v0":
        return []
    if str(warm_evidence.get("status") or "") not in {"needs_review", "rollback_recommended"}:
        return []

    stable_decision = _mapping(stable_evidence.get("decision"))
    warm_decision = _mapping(warm_evidence.get("decision"))
    stable_reasons = set(_string_list(stable_decision.get("reasons")))
    warm_reasons = set(_string_list(warm_decision.get("reasons")))
    threshold_reasons = {
        "stable_anchor_compile_threshold_not_met",
        "warm_compile_threshold_not_met",
        "steady_throughput_gain_below_threshold",
        "active_gpu_delta_below_threshold",
        "throughput_regressed",
    }
    if not (stable_reasons | warm_reasons) & threshold_reasons:
        return []

    summary = _mapping(warm_evidence.get("summary"))
    action = _base_action(
        action_id="guardrail_sdxl_do_not_repeat_compile_batch1_current_axis",
        priority=6,
        action_type="guardrail",
        status="active",
    )
    action.update(
        {
            "blocked_actions": [
                "repeat_sdxl_compile_stable_anchor_batch1_current_axis",
                "repeat_sdxl_compile_warm_cache_batch1_current_axis",
                "promote_sdxl_compile_default_enablement",
                "promote_sdxl_compile_release_claim_current_axis",
            ],
            "reasons": sorted(stable_reasons | warm_reasons),
            "summary": {
                "warm_cache_steady_samples_per_second_gain_pct": summary.get(
                    "steady_samples_per_second_gain_pct"
                ),
                "warm_cache_active_gpu_util_pct_delta": summary.get("active_gpu_util_pct_delta"),
                "warm_cache_candidate_first_step_ms": summary.get("candidate_first_step_ms"),
                "warm_cache_loss_regression_ratio": summary.get("loss_regression_ratio"),
                "warm_cache_after_memory_ratio": summary.get("after_memory_ratio"),
            },
            "current_axis_scope": {
                "family": "sdxl",
                "source_data": str(warm_evidence.get("source_data") or ""),
                "resolution": 1024,
                "train_batch_size": 1,
                "compile_scope": "per_block",
                "cache_policy": "stable_anchor_cache_reuse",
            },
            "allowed_next_axes": [
                "different_attention_backend_or_sdpa_policy",
                "heavier_workload_with_valid_eager_anchor",
                "longer_steady_repeat_with_new_hypothesis",
                "different_compile_mode_or_backend",
            ],
            "rationale": (
                "Stable-anchor and warm-cache diagnostics did not clear throughput/active GPU gates on the "
                "current SDXL batch1 per-block compile axis, so the planner should stop repeating this axis "
                "unless a materially different hypothesis is introduced."
            ),
        }
    )
    return [action]


def build_sdxl_compile_followup_actions(
    *,
    sdxl_alternate_workload_shape_evidence: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_manifest: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_ab_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_ab_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_evidence: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    alternate = _mapping(sdxl_alternate_workload_shape_evidence)
    repeat_manifest = _mapping(sdxl_longer_window_batch2_repeat_manifest)
    repeat = _mapping(sdxl_longer_window_batch2_repeat_evidence)
    actions: list[dict[str, Any]] = []
    blocked = _blocked_anchor_actions(repeat_evidence=repeat, repeat_manifest=repeat_manifest)
    actions.extend(blocked)
    actions.extend(
        _compile_runner_actions(
            _mapping(sdxl_compile_ab_manifest),
            alternate_evidence=alternate,
            repeat_manifest=repeat_manifest,
            repeat_evidence=repeat,
            repo_root=repo_root,
        )
    )
    compile_evidence = _mapping(sdxl_compile_ab_evidence)
    if not blocked or str(compile_evidence.get("status") or "") != "pending_manual_gpu_commands":
        actions.extend(_compile_result_actions(compile_evidence))
    stable_manifest = _mapping(sdxl_compile_stable_anchor_manifest)
    actions.extend(
        _stable_anchor_runner_actions(
            stable_manifest,
            blocked_by_batch2_anchor=bool(blocked),
            repo_root=repo_root,
        )
    )
    stable_evidence = _mapping(sdxl_compile_stable_anchor_evidence)
    actions.extend(_stable_anchor_result_actions(stable_evidence))
    warm_evidence = _mapping(sdxl_compile_warm_cache_evidence)
    actions.extend(
        _warm_cache_runner_actions(
            _mapping(sdxl_compile_warm_cache_manifest),
            stable_evidence=stable_evidence,
            warm_evidence=warm_evidence,
            repo_root=repo_root,
        )
    )
    actions.extend(_warm_cache_result_actions(warm_evidence))
    actions.extend(
        _compile_current_axis_guardrail_actions(
            stable_evidence=stable_evidence,
            warm_evidence=warm_evidence,
        )
    )
    return actions


__all__ = ["build_sdxl_compile_followup_actions"]
