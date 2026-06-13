"""Readiness checks for Bubble Runtime follow-up run plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FOLLOWUP_RUN_READINESS_REPORT = "bubble_runtime_followup_run_readiness_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _path_exists(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).exists()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _ab_evidence_summary(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "load_error": type(exc).__name__,
            "status": "unreadable_ab_evidence",
            "release_claim_eligible": False,
        }
    comparison = _mapping(report.get("comparison"))
    decision = _mapping(report.get("decision"))
    release_claim = _mapping(report.get("release_claim"))
    loss_stability = _mapping(report.get("loss_stability"))
    before = _mapping(report.get("before"))
    after = _mapping(report.get("after"))
    before_metrics = _mapping(before.get("metrics"))
    after_metrics = _mapping(after.get("metrics"))
    data_wait_before = comparison.get("data_wait_share_before", before_metrics.get("data_wait_share"))
    data_wait_after = comparison.get("data_wait_share_after", after_metrics.get("data_wait_share"))
    return {
        "path": str(path),
        "case_id": str(report.get("case_id") or ""),
        "family": str(report.get("family") or ""),
        "status": str(report.get("status") or ""),
        "decision_status": str(decision.get("status") or ""),
        "decision_reasons": _string_list(decision.get("reasons")),
        "release_claim_eligible": bool(release_claim.get("eligible")),
        "release_claim_scope": str(release_claim.get("scope") or ""),
        "data_wait_share_before": _safe_float(data_wait_before),
        "data_wait_share_after": _safe_float(data_wait_after),
        "data_wait_share_delta": _safe_float(comparison.get("data_wait_share_delta")),
        "steady_samples_per_second_gain_pct": _safe_float(
            comparison.get("steady_samples_per_second_gain_pct")
        ),
        "steady_samples_per_second_gain_ratio": _safe_float(
            comparison.get("steady_samples_per_second_gain_ratio")
        ),
        "loss_regression_ratio": _safe_float(comparison.get("loss_regression_ratio")),
        "max_loss_regression_ratio": _safe_float(
            loss_stability.get("max_loss_regression_ratio"), 0.05
        ),
        "loss_stability_status": str(loss_stability.get("status") or ""),
    }


def _existing_completion(out_dir: Any) -> dict[str, Any]:
    text = str(out_dir or "").strip()
    if not text:
        return {
            "completed": False,
            "evidence_paths": [],
            "missing_paths": ["out_dir_missing"],
            "ab_evidence_summaries": [],
        }
    base = Path(text)
    required = [
        base / "real_material_canary_results.json",
        base / "evidence_pack" / "evidence_pack.json",
        base / "evidence_pack" / "natural_load_canary.json",
        base / "evidence_pack" / "release_claims.json",
    ]
    ab_files = sorted(base.glob("*_real_material_canary_ab_evidence.json")) if base.exists() else []
    missing = [str(path) for path in required if not path.exists()]
    if not ab_files:
        missing.append(str(base / "*_real_material_canary_ab_evidence.json"))
    evidence_paths = [str(path) for path in required if path.exists()]
    evidence_paths.extend(str(path) for path in ab_files)
    return {
        "completed": not missing,
        "evidence_paths": evidence_paths,
        "missing_paths": missing,
        "ab_evidence_summaries": [_ab_evidence_summary(path) for path in ab_files],
    }


def _command_has(command: Sequence[Any], flag: str) -> bool:
    return flag in [str(item) for item in command]


def _command_readiness(item: Mapping[str, Any]) -> dict[str, Any]:
    command = [str(part) for part in item.get("command", []) if part is not None]
    dry_run_command = [str(part) for part in item.get("dry_run_command", []) if part is not None]
    family = str(item.get("family") or "")
    profile = str(item.get("profile") or "")
    diagnostic_only = bool(item.get("diagnostic_only"))
    release_relevant = bool(item.get("release_relevant"))
    reasons: list[str] = []
    warnings: list[str] = []
    existing_completion = _existing_completion(item.get("out_dir"))

    if not command:
        reasons.append("missing_command")
    if not dry_run_command or dry_run_command[-1:] != ["--dry-run"]:
        reasons.append("missing_dry_run_command")
    if not _path_exists(item.get("source_data")):
        reasons.append("source_data_missing")
    if not str(item.get("out_dir") or ""):
        reasons.append("out_dir_missing")
    if str(item.get("apply_mode") or "") == "auto_apply":
        reasons.append("unexpected_auto_apply_plan")
    if profile == "aggressive_scaffold_blocked":
        reasons.append("aggressive_scaffold_blocked")
    if profile == "diagnostic_only_probe" or diagnostic_only:
        warnings.append("diagnostic_only_not_release_evidence")
    if release_relevant and diagnostic_only:
        reasons.append("release_relevant_cannot_be_diagnostic_only")
    if family == "sdxl" and release_relevant and not _command_has(command, "--sdxl-resolution"):
        reasons.append("sdxl_release_command_missing_resolution")
    if family == "newbie" and release_relevant:
        warnings.append("newbie_release_recheck_still_requires_cache_and_baseline_data_wait_review")

    status = "manual_ready" if not reasons else "blocked"
    manual_start_required = status in {"manual_ready", "diagnostic_manual_ready"}
    if status == "manual_ready" and existing_completion["completed"]:
        status = "completed_existing_evidence"
        manual_start_required = False
        warnings.append("out_dir_already_has_formal_evidence")
    elif status == "manual_ready" and diagnostic_only:
        status = "diagnostic_manual_ready"
        manual_start_required = True
    return {
        "id": str(item.get("id") or ""),
        "family": family,
        "profile": profile,
        "release_relevant": release_relevant,
        "diagnostic_only": diagnostic_only,
        "status": status,
        "manual_start_required": manual_start_required,
        "safe_to_auto_start": False,
        "reasons": reasons,
        "warnings": warnings,
        "source_data": str(item.get("source_data") or ""),
        "out_dir": str(item.get("out_dir") or ""),
        "sample_offset": item.get("sample_offset"),
        "existing_evidence": existing_completion,
    }


def build_followup_run_readiness(run_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Classify follow-up plan commands before a human starts a GPU run."""

    commands = [_mapping(item) for item in run_plan.get("commands", []) if _mapping(item)]
    command_readiness = [_command_readiness(item) for item in commands]
    aggressive_scaffolds = [
        {
            "id": str(_mapping(item).get("id") or ""),
            "family": str(_mapping(item).get("family") or ""),
            "status": "blocked",
            "safe_to_auto_start": False,
            "reasons": _string_list(_mapping(item).get("blocked_by")) or ["aggressive_scaffold_blocked"],
        }
        for item in run_plan.get("aggressive_scaffolds", [])
        if _mapping(item)
    ]
    manual_ready = [item for item in command_readiness if item["status"] == "manual_ready"]
    diagnostic_ready = [item for item in command_readiness if item["status"] == "diagnostic_manual_ready"]
    completed = [item for item in command_readiness if item["status"] == "completed_existing_evidence"]
    blocked = [item for item in command_readiness if item["status"] == "blocked"]
    return {
        "schema_version": 1,
        "report": FOLLOWUP_RUN_READINESS_REPORT,
        "status": "manual_ready" if manual_ready else "blocked_or_diagnostic_only",
        "source_run_plan_report": str(run_plan.get("report") or ""),
        "source_command_count": len(commands),
        "manual_ready_count": len(manual_ready),
        "diagnostic_manual_ready_count": len(diagnostic_ready),
        "completed_command_count": len(completed),
        "blocked_command_count": len(blocked),
        "aggressive_scaffold_blocked_count": len(aggressive_scaffolds),
        "safe_to_auto_start": False,
        "recommended_manual_command_ids": [item["id"] for item in manual_ready],
        "diagnostic_command_ids": [item["id"] for item in diagnostic_ready],
        "completed_command_ids": [item["id"] for item in completed],
        "blocked_command_ids": [item["id"] for item in blocked],
        "commands": command_readiness,
        "aggressive_scaffolds": aggressive_scaffolds,
        "notes": [
            "This readiness report does not start GPU work.",
            "manual_ready means a human can choose to run the dry-run-verified command; it is not release evidence.",
            "diagnostic_manual_ready commands must stay out of release claims.",
            "completed_existing_evidence means the command out_dir already contains formal canary evidence and should not be rerun by default.",
        ],
    }


__all__ = ["FOLLOWUP_RUN_READINESS_REPORT", "build_followup_run_readiness"]
