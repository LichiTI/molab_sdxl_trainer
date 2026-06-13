"""JSON-only terminal self-check for GPU-bubble readiness.

This sidecar proves whether the current GPU-bubble artifact chain is wired and
waiting on external inputs/manual GPU evidence. It is not release evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_gpu_bubble_readiness_terminal_self_check_v0"
READINESS_REPORT = "gpu_bubble_experiment_readiness_next_actions_v0"
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


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


DETECTED_UNACCEPTED_ROLLUP_FIELDS = (
    "detected_unaccepted_input_ids",
    "detected_unaccepted_next_focus_ids",
    "detected_unaccepted_blocked_criteria_ids",
    "detected_unaccepted_blocked_reason_ids",
)


def _detected_unaccepted_projection(source: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        field: _strings(source.get(field))[:50]
        for field in DETECTED_UNACCEPTED_ROLLUP_FIELDS
    }


def _project_detected_unaccepted(
    *,
    target: dict[str, Any],
    source: Mapping[str, Any],
    row_key: str | None = None,
) -> None:
    target.update(_detected_unaccepted_projection(source))
    if row_key is None:
        return
    source_rows = {
        str(row.get(row_key) or ""): _detected_unaccepted_projection(row)
        for row in (_mapping(item) for item in _list(source.get("rows")))
        if str(row.get(row_key) or "")
    }
    empty_projection = {field: [] for field in DETECTED_UNACCEPTED_ROLLUP_FIELDS}
    for row in target.get("rows", []):
        if isinstance(row, dict):
            row.update(source_rows.get(str(row.get(row_key) or ""), empty_projection))


def _source_cache_stage_lineage_summary(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    lineage = _mapping(pipeline.get("stage_roadmap_lineage"))
    return {
        "expected_roadmap": str(lineage.get("expected_roadmap") or ROADMAP),
        "lineage_ok": bool(lineage.get("lineage_ok")),
        "stage_count": _safe_int(lineage.get("stage_count")),
        "roadmap_mismatch_count": _safe_int(lineage.get("roadmap_mismatch_count")),
        "roadmap_mismatch_stage_ids": _strings(lineage.get("roadmap_mismatch_stage_ids"))[:20],
        "safe_to_auto_start": bool(lineage.get("safe_to_auto_start")),
        "release_claim_allowed": bool(lineage.get("release_claim_allowed")),
        "not_release_evidence": bool(lineage.get("not_release_evidence")),
    }


def _source_cache_stage_lineage_complete(
    summary: Mapping[str, Any],
    *,
    expected_stage_count: int,
) -> bool:
    return bool(
        str(summary.get("expected_roadmap") or "") == ROADMAP
        and bool(summary.get("lineage_ok"))
        and _safe_int(summary.get("stage_count")) > 0
        and _safe_int(summary.get("stage_count")) == expected_stage_count
        and _safe_int(summary.get("roadmap_mismatch_count")) == 0
        and not _strings(summary.get("roadmap_mismatch_stage_ids"))
        and not bool(summary.get("safe_to_auto_start"))
        and not bool(summary.get("release_claim_allowed"))
        and bool(summary.get("not_release_evidence"))
    )


def _sd15_manual_ab_contract_summary(sd15_readiness: Mapping[str, Any]) -> dict[str, Any]:
    envelope = _mapping(sd15_readiness.get("manual_ab_envelope"))
    contract = _mapping(envelope.get("case_config_contract"))
    post_success = _mapping(envelope.get("post_success_rebuild_contract"))
    before = _mapping(contract.get("before"))
    after = _mapping(contract.get("after"))
    common = _mapping(contract.get("common"))
    post_success_command_ids = _strings(
        post_success.get("required_post_manual_command_ids")
    )
    post_success_artifact_ids = _strings(
        post_success.get("required_rebuild_artifact_ids")
    )
    post_success_ready = bool(
        str(post_success.get("contract") or "")
        == "sd15_lora512_post_success_rebuild_contract_v1"
        and "rebuild_current_combined_evidence_pack" in post_success_command_ids
        and "refresh_post_manual_evidence_rebuild_plan" in post_success_command_ids
        and "current_combined/release_claims.json" in post_success_artifact_ids
        and bool(post_success.get("release_claims_rebuild_required"))
        and bool(post_success.get("natural_load_rebuild_required"))
        and not bool(post_success.get("safe_to_auto_start"))
        and not bool(post_success.get("release_claim_allowed_after_success"))
        and bool(post_success.get("not_release_evidence"))
    )
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_sd15_manual_ab_contract_summary",
        "envelope": str(envelope.get("envelope") or ""),
        "contract": str(contract.get("contract") or ""),
        "post_success_rebuild_contract": str(post_success.get("contract") or ""),
        "post_success_rebuild_contract_ready": post_success_ready,
        "post_success_first_rebuild_command_id": str(
            post_success.get("first_rebuild_command_id") or ""
        ),
        "post_success_required_command_ids": post_success_command_ids,
        "post_success_required_artifact_ids": post_success_artifact_ids,
        "post_success_release_claims_rebuild_required": bool(
            post_success.get("release_claims_rebuild_required")
        ),
        "post_success_natural_load_rebuild_required": bool(
            post_success.get("natural_load_rebuild_required")
        ),
        "post_success_safe_to_auto_start": bool(post_success.get("safe_to_auto_start")),
        "post_success_release_claim_allowed_after_success": bool(
            post_success.get("release_claim_allowed_after_success")
        ),
        "post_success_not_release_evidence": bool(
            post_success.get("not_release_evidence")
        ),
        "status": str(envelope.get("status") or ""),
        "case_id": str(contract.get("case_id") or envelope.get("case_id") or ""),
        "release_case_id": str(contract.get("release_case_id") or envelope.get("release_case_id") or ""),
        "family": str(contract.get("family") or envelope.get("family") or ""),
        "contract_ready": bool(contract.get("contract_ready")),
        "manual_start_required_after_ready": bool(envelope.get("manual_start_required_after_ready")),
        "safe_to_auto_start": bool(envelope.get("safe_to_auto_start")),
        "release_claim_allowed_after_success": bool(envelope.get("release_claim_allowed_after_success")),
        "release_claim_allowed": False,
        "requires_gpu_if_executed": bool(envelope.get("requires_gpu_if_executed")),
        "allowed_side_diff_keys": _strings(contract.get("allowed_side_diff_keys")),
        "before_dataloader_workers": _safe_int(before.get("dataloader_workers")),
        "after_dataloader_workers": _safe_int(after.get("dataloader_workers")),
        "resolution": _safe_int(common.get("resolution")),
        "train_batch_size": _safe_int(common.get("train_batch_size")),
        "steps": _safe_int(common.get("steps")),
        "before_config_digest_present": str(before.get("config_digest") or "").startswith("sha256:"),
        "after_config_digest_present": str(after.get("config_digest") or "").startswith("sha256:"),
        "not_release_evidence": True,
    }


def _source_path_rows(readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(readiness.get("source_paths"))]


def _json_only_progress_audit(
    *,
    readiness: Mapping[str, Any],
    remaining: Mapping[str, Any],
    chain_complete: bool,
) -> dict[str, Any]:
    source_rows = _source_path_rows(readiness)
    missing_sources = [
        str(row.get("path") or "")
        for row in source_rows
        if row and not bool(row.get("exists"))
    ]
    load_error_sources = [
        str(row.get("path") or "")
        for row in source_rows
        if row and str(row.get("load_error") or "")
    ]
    json_ready_count = _safe_int(remaining.get("json_ready_action_count"))
    json_ready_ids = _strings(remaining.get("json_ready_action_ids"))
    progress_available = bool(
        json_ready_count > 0
        or missing_sources
        or load_error_sources
        or not chain_complete
    )
    reason_ids: list[str] = []
    if json_ready_count > 0:
        reason_ids.append("json_ready_actions_remaining")
    if missing_sources:
        reason_ids.append("source_json_missing")
    if load_error_sources:
        reason_ids.append("source_json_load_error")
    if not chain_complete:
        reason_ids.append("json_chain_incomplete")
    if not reason_ids:
        reason_ids = [
            "json_ready_action_count_zero",
            "source_reports_loaded",
            "json_chain_complete",
            "external_input_or_manual_gpu_required",
        ]
    return {
        "json_only_substantive_progress_available": progress_available,
        "reason_ids": reason_ids,
        "json_ready_action_count": json_ready_count,
        "json_ready_action_ids": json_ready_ids[:20],
        "json_closed_action_count": _safe_int(remaining.get("json_closed_action_count")),
        "source_path_count": len(source_rows),
        "source_path_missing_count": len(missing_sources),
        "source_path_missing_paths": missing_sources[:20],
        "source_path_load_error_count": len(load_error_sources),
        "source_path_load_error_paths": load_error_sources[:20],
        "chain_complete": chain_complete,
    }


def _intake_missing_inputs(intake: Mapping[str, Any]) -> list[str]:
    missing = _strings(intake.get("missing_external_inputs"))
    if missing:
        return missing
    return [
        str(item.get("id") or "")
        for item in (_mapping(raw) for raw in _list(intake.get("intake_items")))
        if str(item.get("status") or "") in {"missing", "pending_external_input"}
        and str(item.get("id") or "")
    ]


def _external_input_filesystem_audit(
    *,
    expected_missing_inputs: Sequence[str],
    external_input_intake_registry: Mapping[str, Any],
    live_external_input_intake_registry: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _mapping(external_input_intake_registry)
    live = _mapping(live_external_input_intake_registry)
    source = live or registry
    registry_missing = _intake_missing_inputs(registry)
    live_missing = _intake_missing_inputs(live)
    expected_missing = _strings(expected_missing_inputs)
    live_or_registry_missing = live_missing or registry_missing
    expected_set = set(expected_missing)
    observed_set = set(live_or_registry_missing)
    registry_set = set(registry_missing)
    live_set = set(live_missing)
    sd15 = _mapping(source.get("sd15"))
    source_axis = _mapping(source.get("source_axis"))
    checkpoint_exists = bool(sd15.get("checkpoint_exists"))
    checkpoint_count = _safe_int(sd15.get("checkpoint_count"), 1 if checkpoint_exists else 0)
    new_source_root_count = _safe_int(source_axis.get("new_source_root_count"))
    new_source_root_required = bool(source_axis.get("new_source_root_required"))
    drift_reason_ids: list[str] = []
    if not registry:
        drift_reason_ids.append("external_input_intake_registry_missing")
    if not live:
        drift_reason_ids.append("live_external_input_intake_scan_missing")
    if registry and live and registry_set != live_set:
        drift_reason_ids.append("external_input_intake_registry_does_not_match_live_scan")
    if expected_set and observed_set != expected_set:
        drift_reason_ids.append("external_input_missing_inputs_do_not_match_release_unblocker")
    if "sd15_checkpoint" in expected_set and (checkpoint_exists or checkpoint_count > 0):
        drift_reason_ids.append("sd15_checkpoint_present_but_still_marked_missing")
    if "sd15_checkpoint" not in expected_set and not checkpoint_exists:
        drift_reason_ids.append("sd15_checkpoint_missing_but_not_marked_missing")
    if "new_source_root" in expected_set and new_source_root_count > 0:
        drift_reason_ids.append("new_source_root_present_but_still_marked_missing")
    if "new_source_root" not in expected_set and new_source_root_required:
        drift_reason_ids.append("new_source_root_missing_but_not_marked_missing")

    return {
        "summary_version": 1,
        "registry_report": str(registry.get("report") or ""),
        "live_report": str(live.get("report") or ""),
        "registry_status": str(registry.get("status") or ""),
        "live_status": str(live.get("status") or ""),
        "registry_available": bool(registry),
        "live_scan_available": bool(live),
        "registry_missing_external_inputs": registry_missing[:20],
        "live_missing_external_inputs": live_missing[:20],
        "expected_missing_external_inputs": expected_missing[:20],
        "registry_matches_live_scan": bool(registry and live and registry_set == live_set),
        "live_or_registry_matches_expected_missing_inputs": bool(expected_set and observed_set == expected_set),
        "filesystem_external_input_detected": bool(source.get("external_input_detected")),
        "filesystem_external_input_required": bool(source.get("external_input_required")),
        "sd15_model_dir": str(sd15.get("model_dir") or ""),
        "sd15_model_dir_exists": bool(sd15.get("model_dir_exists")),
        "sd15_checkpoint_exists": checkpoint_exists,
        "sd15_checkpoint_count": checkpoint_count,
        "sd15_checkpoint_path": str(sd15.get("checkpoint_path") or ""),
        "source_root": str(source_axis.get("source_root") or ""),
        "source_root_exists": bool(source_axis.get("source_root_exists")),
        "source_root_count": _safe_int(source_axis.get("source_root_count")),
        "new_source_root_count": new_source_root_count,
        "new_source_root_required": new_source_root_required,
        "new_source_roots": [
            str(item.get("root") or "")
            for item in (_mapping(raw) for raw in _list(source_axis.get("roots")))
            if str(item.get("intake_status") or "") == "new_root_available"
        ][:20],
        "artifact_matches_filesystem_and_release_blockers": not drift_reason_ids,
        "drift_reason_ids": drift_reason_ids,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
    }


def _source_cache_negative_evidence_summary(
    *,
    external_input_intake_registry: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any],
    source_axis_freshness_dedupe_audit: Mapping[str, Any],
    newbie_warm_cache_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    intake = _mapping(external_input_intake_registry)
    requirement = _mapping(source_axis_requirement)
    freshness = _mapping(source_axis_freshness_dedupe_audit)
    warm_cache = _mapping(newbie_warm_cache_inventory)
    source_axis = _mapping(intake.get("source_axis"))
    root_rows = [_mapping(item) for item in _list(source_axis.get("roots"))]
    duplicate_roots = [
        str(item.get("root") or "")
        for item in root_rows
        if str(item.get("intake_status") or "") == "current_axis_duplicate"
    ]
    families: list[dict[str, Any]] = []
    completed_out_dirs: set[str] = set()
    for raw in _list(requirement.get("families")):
        item = _mapping(raw)
        run_readiness = _mapping(item.get("run_readiness"))
        out_dirs = _strings(run_readiness.get("completed_out_dirs"))
        completed_out_dirs.update(out_dirs)
        families.append(
            {
                "family": str(item.get("family") or ""),
                "status": str(item.get("status") or ""),
                "requirement": str(item.get("requirement") or ""),
                "source_axis_state": str(item.get("source_axis_state") or ""),
                "requires_external_input": bool(item.get("requires_external_input")),
                "do_not_rerun_current_axis": bool(item.get("do_not_rerun_current_axis")),
                "blocked_by_natural_load_canary": bool(item.get("blocked_by_natural_load_canary")),
                "candidate_count": _safe_int(item.get("candidate_count")),
                "ready_axis_count": _safe_int(item.get("ready_axis_count")),
                "unattempted_high_quality_ready_axis_count": _safe_int(
                    item.get("unattempted_high_quality_ready_axis_count")
                ),
                "completed_out_dir_count": len(out_dirs),
                "blocked_actions": _strings(item.get("blocked_actions"))[:20],
            }
        )
    axes = [_mapping(item) for item in _list(warm_cache.get("axes"))]
    cache_ready_axes = [item for item in axes if bool(item.get("cache_ready"))]
    claimable_axes = [item for item in axes if bool(item.get("claimable"))]
    completed_canary_count = sum(_safe_int(item.get("completed_canary_command_count")) for item in axes)
    new_source_root_count = _safe_int(
        source_axis.get("new_source_root_count"),
        _safe_int(freshness.get("new_source_root_count")),
    )
    current_duplicate_count = len([root for root in duplicate_roots if root])
    warm_cache_status = str(warm_cache.get("status") or "")
    warm_cache_release_ready = bool(claimable_axes) or bool(warm_cache.get("release_claim_allowed"))
    blocker_reason_ids: list[str] = []
    if new_source_root_count <= 0 and current_duplicate_count > 0:
        blocker_reason_ids.append("current_source_root_is_duplicate_not_new_axis")
    if _safe_int(requirement.get("external_input_required_count")) > 0:
        blocker_reason_ids.append("source_axis_families_require_external_input")
    if warm_cache_status and not warm_cache_release_ready:
        blocker_reason_ids.append("warm_cache_inventory_not_release_ready")
    if completed_out_dirs:
        blocker_reason_ids.append("completed_followup_out_dirs_must_not_be_rerun_without_new_axis")
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "source_axis_requirement_report": str(requirement.get("report") or ""),
        "source_axis_requirement_status": str(requirement.get("status") or ""),
        "source_axis_freshness_report": str(freshness.get("report") or ""),
        "source_axis_freshness_status": str(freshness.get("status") or ""),
        "newbie_warm_cache_report": str(warm_cache.get("report") or ""),
        "newbie_warm_cache_status": warm_cache_status,
        "current_source_roots": _strings(source_axis.get("current_source_roots"))
        or _strings(freshness.get("current_source_roots")),
        "current_source_root_duplicate_count": current_duplicate_count,
        "current_source_root_duplicates": duplicate_roots[:20],
        "new_source_root_count": new_source_root_count,
        "new_source_roots": _strings(freshness.get("new_source_roots"))[:20],
        "external_input_required_family_count": _safe_int(requirement.get("external_input_required_count")),
        "candidate_available_family_count": _safe_int(requirement.get("candidate_available_family_count")),
        "exhausted_family_count": _safe_int(requirement.get("exhausted_family_count")),
        "no_ready_source_axis_family_count": _safe_int(requirement.get("no_ready_source_axis_family_count")),
        "families": families,
        "cache_ready_axis_count": len(cache_ready_axes),
        "claimable_cache_axis_count": len(claimable_axes),
        "completed_canary_command_count": completed_canary_count,
        "completed_out_dir_count": len(completed_out_dirs),
        "cannot_clear_new_source_root_blocker_from_current_axis": bool(
            new_source_root_count <= 0 and current_duplicate_count > 0
        ),
        "cannot_clear_warm_cache_axis_from_inventory": not warm_cache_release_ready,
        "negative_evidence_reason_ids": blocker_reason_ids,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
    }


def _compact_summary_mirror(
    summary: Mapping[str, Any],
    *,
    string_fields: Sequence[str] = (),
    bool_fields: Sequence[str] = (),
    int_fields: Sequence[str] = (),
    number_fields: Sequence[str] = (),
    list_fields: Sequence[str] = (),
    nested_fields: Sequence[str] = (),
) -> dict[str, Any]:
    source = _mapping(summary)
    mirrored: dict[str, Any] = {
        "summary_version": _safe_int(source.get("summary_version"), 1),
        "roadmap": str(source.get("roadmap") or ""),
        "artifact_role": str(source.get("artifact_role") or ""),
    }
    for field in string_fields:
        mirrored[field] = str(source.get(field) or "")
    for field in bool_fields:
        mirrored[field] = bool(source.get(field))
    for field in int_fields:
        mirrored[field] = _safe_int(source.get(field))
    for field in number_fields:
        mirrored[field] = _safe_number(source.get(field))
    for field in list_fields:
        mirrored[field] = _strings(source.get(field))[:20]
    for field in nested_fields:
        mirrored[field] = dict(_mapping(source.get(field)))
    for field in [
        "fail_closed",
        "not_release_evidence",
        "safe_to_auto_start",
        "release_claim_allowed",
        "does_not_run_training",
        "does_not_run_cuda",
    ]:
        mirrored[field] = bool(source.get(field))
    if "publishable" in source:
        mirrored["publishable"] = bool(source.get("publishable"))
    return mirrored


def _guard_report_lineage_summary(release_readiness_guard_report: Mapping[str, Any]) -> dict[str, Any]:
    guard = _mapping(release_readiness_guard_report)
    input_summary = _mapping(guard.get("input_artifact_summary"))
    readiness_summary = _mapping(input_summary.get("readiness"))
    terminal_summary = _mapping(input_summary.get("terminal"))
    freshness_summary = _mapping(input_summary.get("freshness"))
    available = bool(guard)
    return {
        "summary_version": 1,
        "lineage_role": "optional_previous_or_latest_guard_report",
        "available": available,
        "does_not_gate_terminal_self_check": True,
        "not_release_evidence": bool(guard.get("not_release_evidence")) if available else True,
        "report": str(guard.get("report") or ""),
        "roadmap": str(guard.get("roadmap") or ""),
        "artifact_role": str(guard.get("artifact_role") or ""),
        "status": str(guard.get("status") or ""),
        "ok": bool(guard.get("ok")),
        "failure_count": _safe_int(guard.get("failure_count")),
        "safe_to_auto_start": bool(guard.get("safe_to_auto_start")),
        "release_claim_allowed": bool(guard.get("release_claim_allowed")),
        "checked_readiness_artifact_status": str(readiness_summary.get("artifact_status") or ""),
        "checked_readiness_release_readiness": str(readiness_summary.get("release_readiness") or ""),
        "checked_terminal_status": str(terminal_summary.get("terminal_status") or ""),
        "checked_chain_integrity_status": str(terminal_summary.get("chain_integrity_status") or ""),
        "checked_freshness_ok": bool(freshness_summary.get("freshness_ok")),
        "guard_report_fail_closed": bool(
            available
            and bool(guard.get("not_release_evidence"))
            and not bool(guard.get("safe_to_auto_start"))
            and not bool(guard.get("release_claim_allowed"))
        ),
        "blocked_actions": [
            "do_not_use_guard_report_lineage_as_release_evidence",
            "do_not_make_terminal_self_check_depend_on_latest_guard_report",
            "do_not_auto_start_gpu_heavy_from_guard_report_lineage",
        ],
    }


def _roadmap_lineage_audit(*, artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for artifact_id, artifact in artifacts.items():
        item = _mapping(artifact)
        report = str(item.get("report") or "")
        roadmap = str(item.get("roadmap") or "")
        available = bool(item)
        rows.append(
            {
                "id": artifact_id,
                "report": report,
                "available": available,
                "roadmap": roadmap,
                "expected_roadmap": ROADMAP,
                "roadmap_matches_expected": available and roadmap == ROADMAP,
                "required_for_lineage": True,
            }
        )
    missing = [row["id"] for row in rows if not row["available"]]
    mismatched = [row["id"] for row in rows if row["available"] and not row["roadmap_matches_expected"]]
    return {
        "summary_version": 1,
        "expected_roadmap": ROADMAP,
        "audited_artifact_count": len(rows),
        "required_artifact_count": len(rows),
        "missing_required_artifact_count": len(missing),
        "missing_required_artifact_ids": missing[:20],
        "mismatched_artifact_count": len(mismatched),
        "mismatched_artifact_ids": mismatched[:20],
        "lineage_ok": not missing and not mismatched,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
        "audited_artifacts": rows,
        "blocked_actions": [
            "do_not_mix_gpu_bubble_artifacts_from_other_roadmaps",
            "do_not_treat_roadmap_lineage_as_release_evidence",
            "do_not_auto_start_gpu_heavy_from_roadmap_lineage",
        ],
    }


def _source_cache_axis_identity_registry_summary(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary_version": _safe_int(registry.get("summary_version"), 1),
        "report": str(registry.get("report") or ""),
        "roadmap": str(registry.get("roadmap") or ""),
        "status": str(registry.get("status") or ""),
        "axis_state": str(registry.get("axis_state") or ""),
        "artifact_role": str(registry.get("artifact_role") or ""),
        "identity_schema_version": _safe_int(registry.get("identity_schema_version"), 1),
        "row_count": _safe_int(registry.get("row_count")),
        "root_identity_row_count": _safe_int(registry.get("root_identity_row_count")),
        "full_axis_identity_row_count": _safe_int(registry.get("full_axis_identity_row_count")),
        "current_source_root_count": _safe_int(registry.get("current_source_root_count")),
        "new_source_root_count": _safe_int(registry.get("new_source_root_count")),
        "duplicate_or_stale_axis_count": _safe_int(registry.get("duplicate_or_stale_axis_count")),
        "fresh_axis_candidate_count": _safe_int(registry.get("fresh_axis_candidate_count")),
        "unsafe_row_count": _safe_int(registry.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(registry.get("unsafe_row_ids"))[:20],
        "fail_closed": bool(registry.get("fail_closed")),
        "not_release_evidence": bool(registry.get("not_release_evidence")),
        "does_not_run_training": bool(registry.get("does_not_run_training")),
        "does_not_run_cuda": bool(registry.get("does_not_run_cuda")),
        "publishable": bool(registry.get("publishable")),
        "safe_to_auto_start": bool(registry.get("safe_to_auto_start")),
        "release_claim_allowed": bool(registry.get("release_claim_allowed")),
    }


def build_gpu_bubble_readiness_terminal_self_check(
    *,
    readiness_next_actions: Mapping[str, Any] | None = None,
    external_input_handoff_packet: Mapping[str, Any] | None = None,
    external_input_intake_registry: Mapping[str, Any] | None = None,
    live_external_input_intake_registry: Mapping[str, Any] | None = None,
    artifact_freshness_audit: Mapping[str, Any] | None = None,
    source_axis_requirement: Mapping[str, Any] | None = None,
    source_axis_freshness_dedupe_audit: Mapping[str, Any] | None = None,
    source_cache_axis_identity_registry: Mapping[str, Any] | None = None,
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
    source_cache_axis_pipeline_readiness: Mapping[str, Any] | None = None,
    post_manual_evidence_rebuild_plan: Mapping[str, Any] | None = None,
    sdxl_non_dataloader_manual_gpu_queue: Mapping[str, Any] | None = None,
    sd15_readiness: Mapping[str, Any] | None = None,
    release_readiness_guard_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = _mapping(readiness_next_actions)
    handoff = _mapping(external_input_handoff_packet)
    intake = _mapping(external_input_intake_registry)
    live_intake = _mapping(live_external_input_intake_registry)
    freshness = _mapping(artifact_freshness_audit)
    source_requirement = _mapping(source_axis_requirement)
    source_freshness = _mapping(source_axis_freshness_dedupe_audit)
    identity_registry = _mapping(source_cache_axis_identity_registry) or _mapping(
        source_freshness.get("source_cache_axis_identity_registry")
    )
    warm_cache = _mapping(newbie_warm_cache_inventory)
    pipeline = _mapping(source_cache_axis_pipeline_readiness)
    post_manual = _mapping(post_manual_evidence_rebuild_plan)
    sdxl_manual_gpu_queue = _mapping(sdxl_non_dataloader_manual_gpu_queue)
    sd15 = _mapping(sd15_readiness)
    guard_report = _mapping(release_readiness_guard_report)
    remaining = _mapping(readiness.get("remaining_work_summary"))
    manual_gpu_execution = _mapping(readiness.get("manual_gpu_execution_summary"))
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    manual_blocking = _mapping(post_manual.get("manual_evidence_blocking_summary"))
    input_resolution = _mapping(unblocker.get("input_resolution_summary")) or _mapping(
        handoff.get("input_resolution_summary")
    )
    manual_evidence_blocking = _mapping(unblocker.get("manual_evidence_blocking_summary")) or manual_blocking
    first_release_scope = _mapping(readiness.get("first_release_scope"))
    first_release_policy = _mapping(readiness.get("first_release_policy_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("first_release_policy_summary")
    )
    source_axis_freshness_summary = _mapping(
        readiness.get("source_axis_freshness_dedupe_audit")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("source_axis_freshness_dedupe_audit"))
    source_axis_requirement_summary = _mapping(readiness.get("source_axis_requirement_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_axis_requirement")
    )
    if not source_axis_requirement_summary and source_requirement:
        source_axis_requirement_summary = {
            "summary_version": 1,
            "roadmap": ROADMAP,
            "artifact_role": "gpu_bubble_source_axis_requirement_summary",
            "report": str(source_requirement.get("report") or ""),
            "status": str(source_requirement.get("status") or ""),
            "family_count": _safe_int(source_requirement.get("family_count")),
            "external_input_required_count": _safe_int(
                source_requirement.get("external_input_required_count")
            ),
            "candidate_available_family_count": _safe_int(
                source_requirement.get("candidate_available_family_count")
            ),
            "exhausted_family_count": _safe_int(source_requirement.get("exhausted_family_count")),
            "no_ready_source_axis_family_count": _safe_int(
                source_requirement.get("no_ready_source_axis_family_count")
            ),
            "completed_existing_command_count": _safe_int(
                source_requirement.get("completed_existing_command_count")
            ),
            "external_input_required": _safe_int(source_requirement.get("external_input_required_count")) > 0,
            "release_claim_allowed": False,
            "safe_to_auto_start": False,
            "does_not_run_training": True,
            "does_not_run_cuda": True,
            "not_release_evidence": True,
            "fail_closed": True,
        }
    source_cache_axis_pipeline_readiness_summary = _mapping(
        readiness.get("source_cache_axis_pipeline_readiness_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("source_cache_axis_pipeline_readiness"))
    if not source_cache_axis_pipeline_readiness_summary and pipeline:
        axis_readiness = _mapping(pipeline.get("axis_readiness"))
        stage_freshness = _mapping(pipeline.get("stage_freshness"))
        source_cache_axis_pipeline_readiness_summary = {
            "summary_version": 1,
            "roadmap": ROADMAP,
            "artifact_role": "gpu_bubble_source_cache_axis_pipeline_readiness_summary",
            "report": str(pipeline.get("report") or ""),
            "status": str(pipeline.get("status") or ""),
            "axis_readiness_status": str(
                pipeline.get("axis_readiness_status") or axis_readiness.get("status") or ""
            ),
            "pipeline_complete": bool(pipeline.get("pipeline_complete")),
            "external_input_required": bool(pipeline.get("external_input_required")),
            "preflight_admitted": bool(pipeline.get("preflight_admitted")),
            "manual_canary_plan_ready": bool(pipeline.get("manual_canary_plan_ready")),
            "waiting_external_input": bool(axis_readiness.get("waiting_external_input")),
            "duplicate_or_stale_axis_blocked": bool(axis_readiness.get("duplicate_or_stale_axis_blocked")),
            "cache_axis_not_ready": bool(axis_readiness.get("cache_axis_not_ready")),
            "stage_count": _safe_int(pipeline.get("stage_count")),
            "stage_ok_count": _safe_int(pipeline.get("stage_ok_count")),
            "stage_freshness_checked": bool(stage_freshness.get("freshness_checked")),
            "stage_freshness_ok": bool(stage_freshness.get("freshness_ok")),
            "stale_stage_count": _safe_int(stage_freshness.get("stale_stage_count")),
            "stale_stage_ids": _strings(stage_freshness.get("stale_stage_ids"))[:20],
            "stale_ready_stage_count": _safe_int(stage_freshness.get("stale_ready_stage_count")),
            "stale_ready_stage_ids": _strings(stage_freshness.get("stale_ready_stage_ids"))[:20],
            "ready_stage_freshness_ok": bool(stage_freshness.get("ready_stage_freshness_ok")),
            "blocker_count": len(_strings(pipeline.get("blockers"))),
            "next_action_count": len(_list(pipeline.get("next_actions"))),
            "release_claim_allowed": False,
            "safe_to_auto_start": False,
            "does_not_run_training": True,
            "does_not_run_cuda": True,
            "not_release_evidence": True,
            "fail_closed": True,
        }
    external_input_admission_summary = _mapping(
        readiness.get("external_input_admission_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("external_input_admission"))
    external_input_intake_registry_summary = _mapping(
        readiness.get("external_input_intake_registry_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("external_input_intake_registry"))
    external_input_replay_plan_summary = _mapping(
        readiness.get("external_input_replay_plan_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("external_input_replay_plan"))
    external_input_handoff_packet_summary = _mapping(
        readiness.get("external_input_handoff_packet_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("external_input_handoff_packet"))
    newbie_warm_cache_inventory_summary = _mapping(
        readiness.get("newbie_warm_cache_inventory_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("newbie_warm_cache_inventory"))
    source_cache_axis_admission_preflight_summary = _mapping(
        readiness.get("source_cache_axis_admission_preflight_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("source_cache_axis_admission_preflight"))
    source_axis_unblock_recommendation_summary = _mapping(
        readiness.get("source_axis_unblock_recommendation_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_axis_unblock_recommendation_summary")
    )
    source_cache_axis_manual_canary_plan_summary = _mapping(
        readiness.get("source_cache_axis_manual_canary_plan_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("source_cache_axis_manual_canary_plan"))
    post_manual_evidence_rebuild_plan_summary = _mapping(
        readiness.get("post_manual_evidence_rebuild_plan_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("post_manual_evidence_rebuild_plan"))
    recommended_release_policy = str(
        first_release_scope.get("recommended_release_policy")
        or remaining.get("recommended_release_policy")
        or ""
    )
    claim_wording_audit = _mapping(readiness.get("forbidden_claim_wording_audit"))
    claim_wording_policy = str(readiness.get("claim_wording_policy") or claim_wording_audit.get("claim_wording_policy") or "")
    forbidden_claim_wording_hit_count = _safe_int(
        readiness.get("forbidden_claim_wording_hit_count"),
        _safe_int(claim_wording_audit.get("forbidden_claim_wording_hit_count")),
    )
    roadmap_acceptance = _mapping(readiness.get("roadmap_acceptance_gate_summary"))
    roadmap_acceptance_fail_closed = (
        bool(roadmap_acceptance)
        and str(roadmap_acceptance.get("roadmap") or "") == ROADMAP
        and bool(roadmap_acceptance.get("fail_closed"))
        and bool(roadmap_acceptance.get("not_release_evidence"))
        and not bool(roadmap_acceptance.get("safe_to_auto_start"))
        and not bool(roadmap_acceptance.get("release_claim_allowed"))
    )
    roadmap_execution = _mapping(readiness.get("roadmap_execution_contract_summary"))
    roadmap_execution_fail_closed = (
        bool(roadmap_execution)
        and str(roadmap_execution.get("roadmap") or "") == ROADMAP
        and bool(roadmap_execution.get("fail_closed"))
        and bool(roadmap_execution.get("not_release_evidence"))
        and not bool(roadmap_execution.get("safe_to_auto_start"))
        and not bool(roadmap_execution.get("release_claim_allowed"))
    )
    matrix_readiness = _mapping(readiness.get("experiment_matrix_readiness"))
    matrix_readiness_fail_closed = (
        bool(matrix_readiness)
        and str(matrix_readiness.get("roadmap") or "") == ROADMAP
        and bool(matrix_readiness.get("fail_closed"))
        and bool(matrix_readiness.get("not_release_evidence"))
        and not bool(matrix_readiness.get("safe_to_auto_start"))
        and not bool(matrix_readiness.get("release_claim_allowed"))
    )
    identity_registry_fail_closed = (
        bool(identity_registry)
        and str(identity_registry.get("roadmap") or "") == ROADMAP
        and bool(identity_registry.get("fail_closed"))
        and bool(identity_registry.get("not_release_evidence"))
        and _safe_int(identity_registry.get("unsafe_row_count")) == 0
        and _safe_int(identity_registry.get("row_count")) > 0
        and not bool(identity_registry.get("safe_to_auto_start"))
        and not bool(identity_registry.get("release_claim_allowed"))
    )
    normalized_gate_mapping = _mapping(readiness.get("normalized_evidence_gate_mapping"))
    normalized_gate_mapping_fail_closed = (
        bool(normalized_gate_mapping)
        and str(normalized_gate_mapping.get("roadmap") or "") == ROADMAP
        and bool(normalized_gate_mapping.get("fail_closed"))
        and bool(normalized_gate_mapping.get("not_release_evidence"))
        and _safe_int(normalized_gate_mapping.get("unmapped_row_count")) == 0
        and _safe_int(normalized_gate_mapping.get("unsafe_row_count")) == 0
        and not bool(normalized_gate_mapping.get("safe_to_auto_start"))
        and not bool(normalized_gate_mapping.get("release_claim_allowed"))
    )
    normalized_gate_explanation = _mapping(
        readiness.get("normalized_evidence_gate_explanation_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "normalized_evidence_gate_explanation_summary"
        )
    )
    normalized_gate_explanation_fail_closed = (
        bool(normalized_gate_explanation)
        and str(normalized_gate_explanation.get("roadmap") or "") == ROADMAP
        and bool(normalized_gate_explanation.get("fail_closed"))
        and bool(normalized_gate_explanation.get("not_release_evidence"))
        and _safe_int(normalized_gate_explanation.get("unsafe_row_count")) == 0
        and _safe_int(normalized_gate_explanation.get("mapped_row_count"))
        == _safe_int(normalized_gate_mapping.get("mapped_row_count"))
        and not bool(normalized_gate_explanation.get("safe_to_auto_start"))
        and not bool(normalized_gate_explanation.get("release_claim_allowed"))
    )
    next_action_machine = _mapping(readiness.get("next_action_machine_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("next_action_machine_summary")
    )
    next_action_contract = _mapping(readiness.get("next_action_contract_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("next_action_contract_summary")
    )
    manual_review_queue = _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_review_queue_summary")
    )
    manual_review_artifact_chain = _mapping(
        readiness.get("manual_review_artifact_chain_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_review_artifact_chain_summary")
    )
    sdxl_diagnostic_artifact_chain = _mapping(
        readiness.get("sdxl_diagnostic_artifact_chain_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("sdxl_diagnostic_artifact_chain_summary")
    )
    newbie_blockskip_compute_bound_chain = _mapping(
        readiness.get("newbie_blockskip_compute_bound_artifact_chain_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "newbie_blockskip_compute_bound_artifact_chain_summary"
        )
    )
    newbie_compute_diagnosis_chain = _mapping(
        readiness.get("newbie_compute_diagnosis_artifact_chain_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "newbie_compute_diagnosis_artifact_chain_summary"
        )
    )
    newbie_tail8_attention_compute_review = _mapping(
        readiness.get("newbie_tail8_attention_compute_review_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "newbie_tail8_attention_compute_review_summary"
        )
    )
    newbie_tail8_forward_anomaly_review = _mapping(
        readiness.get("newbie_tail8_forward_anomaly_review_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "newbie_tail8_forward_anomaly_review_summary"
        )
    )
    newbie_tail8_seed2027_rerun_preflight = _mapping(
        readiness.get("newbie_tail8_seed2027_rerun_preflight_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "newbie_tail8_seed2027_rerun_preflight_summary"
        )
    )
    protected_followup_gpu_queue = _mapping(
        readiness.get("protected_followup_gpu_queue_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("protected_followup_gpu_queue_summary"))
    blocker_matrix = _mapping(readiness.get("remaining_release_blocker_matrix_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_release_blocker_matrix_summary")
    )
    blocker_handoff = _mapping(readiness.get("remaining_blocker_resolution_handoff_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_blocker_resolution_handoff_summary")
    )
    action_dependency_graph = _mapping(readiness.get("remaining_action_dependency_graph_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_action_dependency_graph_summary")
    )
    action_unblock_sequence = _mapping(readiness.get("remaining_action_unblock_sequence_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_action_unblock_sequence_summary")
    )
    blocker_presence = _mapping(readiness.get("remaining_blocker_artifact_presence_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_blocker_artifact_presence_summary")
    )
    release_exit = _mapping(readiness.get("release_claim_exit_criteria_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("release_claim_exit_criteria_summary")
    )
    release_gate_input_dependency = _mapping(
        readiness.get("release_gate_input_dependency_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("release_gate_input_dependency_summary"))
    release_gate_post_input_refresh_plan = _mapping(
        readiness.get("release_gate_post_input_refresh_plan_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_plan_summary"
        )
    )
    release_gate_input_detection_source = _mapping(
        readiness.get("release_gate_input_detection_source_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_detection_source_summary"
        )
    )
    release_gate_input_acceptance_criteria = _mapping(
        readiness.get("release_gate_input_acceptance_criteria_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_acceptance_criteria_summary"
        )
    )
    release_gate_input_refresh_readiness = _mapping(
        readiness.get("release_gate_input_refresh_readiness_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_refresh_readiness_summary"
        )
    )
    release_gate_input_refresh_blocker = _mapping(
        readiness.get("release_gate_input_refresh_blocker_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_refresh_blocker_summary"
        )
    )
    release_gate_input_lifecycle = _mapping(
        readiness.get("release_gate_input_lifecycle_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_lifecycle_summary"
        )
    )
    external_input_release_gate_alignment = _mapping(
        readiness.get("external_input_release_gate_alignment_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "external_input_release_gate_alignment_summary"
        )
    )
    release_gate_post_input_refresh_command_surface = _mapping(
        readiness.get("release_gate_post_input_refresh_command_surface_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_command_surface_summary"
        )
    )
    release_gate_post_input_refresh_sequence_integrity = _mapping(
        readiness.get("release_gate_post_input_refresh_sequence_integrity_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_sequence_integrity_summary"
        )
    )
    release_gate_post_input_refresh_terminal_guard_dependency = _mapping(
        readiness.get("release_gate_post_input_refresh_terminal_guard_dependency_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_terminal_guard_dependency_summary"
        )
    )
    release_gate_post_input_refresh_artifact_coverage = _mapping(
        readiness.get("release_gate_post_input_refresh_artifact_coverage_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_artifact_coverage_summary"
        )
    )
    release_gate_post_input_refresh_command_artifact_link = _mapping(
        readiness.get("release_gate_post_input_refresh_command_artifact_link_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_command_artifact_link_summary"
        )
    )
    release_gate_post_input_refresh_guard_consumption = _mapping(
        readiness.get("release_gate_post_input_refresh_guard_consumption_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_guard_consumption_summary"
        )
    )
    release_gate_post_input_refresh_guard_report_acceptance = _mapping(
        readiness.get("release_gate_post_input_refresh_guard_report_acceptance_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_guard_report_acceptance_summary"
        )
    )
    external_input_json_refresh_runner_manifest_summary = _mapping(
        readiness.get("external_input_json_refresh_runner_manifest_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "external_input_json_refresh_runner_manifest_summary"
        )
    )
    command_surface = _mapping(readiness.get("manual_protected_gpu_command_surface_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_protected_gpu_command_surface_summary")
    )
    protected_run_plan_chain = _mapping(
        readiness.get("protected_followup_run_plan_artifact_chain_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "protected_followup_run_plan_artifact_chain_summary"
        )
    )
    source_artifact_inventory = _mapping(readiness.get("source_artifact_inventory_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_artifact_inventory_summary")
    )
    evidence_summary_inventory = _mapping(readiness.get("evidence_summary_inventory_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("evidence_summary_inventory_summary")
    )
    next_action_machine_fail_closed = (
        bool(next_action_machine)
        and str(next_action_machine.get("roadmap") or "") == ROADMAP
        and bool(next_action_machine.get("fail_closed"))
        and bool(next_action_machine.get("not_release_evidence"))
        and _safe_int(next_action_machine.get("unique_action_count")) == len(_list(readiness.get("next_actions")))
        and _safe_int(next_action_machine.get("unsafe_action_count")) == 0
        and _safe_int(next_action_machine.get("missing_machine_field_action_count")) == 0
        and not bool(next_action_machine.get("safe_to_auto_start"))
        and not bool(next_action_machine.get("release_claim_allowed"))
    )
    next_action_contract_fail_closed = (
        bool(next_action_contract)
        and str(next_action_contract.get("expected_roadmap") or "") == ROADMAP
        and bool(next_action_contract.get("contract_ok"))
        and bool(next_action_contract.get("not_release_evidence"))
        and _safe_int(next_action_contract.get("action_count")) == len(_list(readiness.get("next_actions")))
        and _safe_int(next_action_contract.get("contract_complete_action_count"))
        == len(_list(readiness.get("next_actions")))
        and _safe_int(next_action_contract.get("missing_contract_action_count")) == 0
        and _safe_int(next_action_contract.get("release_or_auto_start_unsafe_action_count")) == 0
        and not bool(next_action_contract.get("safe_to_auto_start"))
        and not bool(next_action_contract.get("release_claim_allowed"))
    )
    manual_review_queue_fail_closed = (
        bool(manual_review_queue)
        and str(manual_review_queue.get("roadmap") or "") == ROADMAP
        and bool(manual_review_queue.get("fail_closed"))
        and bool(manual_review_queue.get("not_release_evidence"))
        and _safe_int(manual_review_queue.get("unsafe_action_count")) == 0
        and not bool(manual_review_queue.get("safe_to_auto_start"))
        and not bool(manual_review_queue.get("release_claim_allowed"))
    )
    manual_review_artifact_chain_fail_closed = (
        bool(manual_review_artifact_chain)
        and str(manual_review_artifact_chain.get("roadmap") or "") == ROADMAP
        and bool(manual_review_artifact_chain.get("fail_closed"))
        and bool(manual_review_artifact_chain.get("not_release_evidence"))
        and _safe_int(manual_review_artifact_chain.get("unsafe_artifact_count")) == 0
        and not bool(manual_review_artifact_chain.get("safe_to_auto_start"))
        and not bool(manual_review_artifact_chain.get("release_claim_allowed"))
    )
    sdxl_diagnostic_artifact_chain_fail_closed = (
        bool(sdxl_diagnostic_artifact_chain)
        and str(sdxl_diagnostic_artifact_chain.get("roadmap") or "") == ROADMAP
        and bool(sdxl_diagnostic_artifact_chain.get("fail_closed"))
        and bool(sdxl_diagnostic_artifact_chain.get("not_release_evidence"))
        and _safe_int(sdxl_diagnostic_artifact_chain.get("unsafe_artifact_count")) == 0
        and not bool(sdxl_diagnostic_artifact_chain.get("safe_to_auto_start"))
        and not bool(sdxl_diagnostic_artifact_chain.get("release_claim_allowed"))
    )
    newbie_blockskip_compute_bound_chain_fail_closed = (
        bool(newbie_blockskip_compute_bound_chain)
        and str(newbie_blockskip_compute_bound_chain.get("roadmap") or "") == ROADMAP
        and bool(newbie_blockskip_compute_bound_chain.get("fail_closed"))
        and bool(newbie_blockskip_compute_bound_chain.get("not_release_evidence"))
        and _safe_int(newbie_blockskip_compute_bound_chain.get("missing_artifact_count")) == 0
        and _safe_int(newbie_blockskip_compute_bound_chain.get("unsafe_artifact_count")) == 0
        and bool(newbie_blockskip_compute_bound_chain.get("quality_gate_blocked"))
        and bool(newbie_blockskip_compute_bound_chain.get("quality_drift_review_blocked"))
        and bool(newbie_blockskip_compute_bound_chain.get("loss_curve_nonrelease_ready"))
        and bool(newbie_blockskip_compute_bound_chain.get("quality_semantic_nonrelease_ready"))
        and bool(newbie_blockskip_compute_bound_chain.get("semantics_reviewed_blocked"))
        and bool(newbie_blockskip_compute_bound_chain.get("policy_defined_blocked"))
        and not bool(newbie_blockskip_compute_bound_chain.get("policy_compute_bound_exception_allowed"))
        and not bool(newbie_blockskip_compute_bound_chain.get("safe_to_auto_start"))
        and not bool(newbie_blockskip_compute_bound_chain.get("release_claim_allowed"))
    )
    newbie_compute_diagnosis_chain_fail_closed = (
        bool(newbie_compute_diagnosis_chain)
        and str(newbie_compute_diagnosis_chain.get("roadmap") or "") == ROADMAP
        and bool(newbie_compute_diagnosis_chain.get("fail_closed"))
        and bool(newbie_compute_diagnosis_chain.get("not_release_evidence"))
        and bool(newbie_compute_diagnosis_chain.get("data_wait_route_exhausted"))
        and _safe_int(newbie_compute_diagnosis_chain.get("unsafe_artifact_count")) == 0
        and _safe_int(newbie_compute_diagnosis_chain.get("compute_bound_probe_count")) >= 2
        and _safe_int(newbie_compute_diagnosis_chain.get("low_data_wait_probe_count")) >= 2
        and _safe_int(newbie_compute_diagnosis_chain.get("dataloader_rebuild_observed_count")) == 0
        and _safe_int(newbie_compute_diagnosis_chain.get("natural_candidate_count")) == 0
        and not bool(newbie_compute_diagnosis_chain.get("safe_to_auto_start"))
        and not bool(newbie_compute_diagnosis_chain.get("release_claim_allowed"))
    )
    newbie_tail8_attention_compute_review_fail_closed = (
        bool(newbie_tail8_attention_compute_review)
        and str(newbie_tail8_attention_compute_review.get("roadmap") or "") == ROADMAP
        and bool(newbie_tail8_attention_compute_review.get("fail_closed"))
        and bool(newbie_tail8_attention_compute_review.get("not_release_evidence"))
        and str(newbie_tail8_attention_compute_review.get("candidate") or "")
        == "newbie_target_scope:tail8_attention"
        and not bool(newbie_tail8_attention_compute_review.get("release_claim_eligible"))
        and _safe_int(newbie_tail8_attention_compute_review.get("unsafe_artifact_count")) == 0
        and not bool(newbie_tail8_attention_compute_review.get("publishable"))
        and not bool(newbie_tail8_attention_compute_review.get("safe_to_auto_start"))
        and not bool(newbie_tail8_attention_compute_review.get("release_claim_allowed"))
    )
    newbie_tail8_target_depth_progression = _mapping(
        newbie_tail8_attention_compute_review.get("target_depth_progression_summary")
    )
    newbie_tail8_target_depth_progression_fail_closed = (
        newbie_tail8_attention_compute_review_fail_closed
        and str(newbie_tail8_target_depth_progression.get("progression") or "")
        == "newbie_target_depth_progression_v1"
        and str(newbie_tail8_target_depth_progression.get("current_candidate_scope") or "")
        == "tail8_attention"
        and bool(newbie_tail8_target_depth_progression.get("fail_closed"))
        and bool(newbie_tail8_target_depth_progression.get("not_release_evidence"))
        and bool(newbie_tail8_target_depth_progression.get("does_not_run_training"))
        and bool(newbie_tail8_target_depth_progression.get("does_not_run_cuda"))
        and bool(newbie_tail8_target_depth_progression.get("does_not_run_gpu_heavy"))
        and not bool(newbie_tail8_target_depth_progression.get("publishable"))
        and not bool(newbie_tail8_target_depth_progression.get("safe_to_auto_start"))
        and not bool(newbie_tail8_target_depth_progression.get("release_claim_allowed"))
        and (
            not bool(
                newbie_tail8_target_depth_progression.get(
                    "alternate_target_depth_comparison_allowed"
                )
            )
            or (
                bool(newbie_tail8_target_depth_progression.get("tail8_repeat_gate_ready"))
                and bool(newbie_tail8_target_depth_progression.get("tail8_quality_gate_ready"))
                and bool(
                    newbie_tail8_target_depth_progression.get(
                        "tail8_stable_for_depth_comparison"
                    )
                )
            )
        )
    )
    newbie_tail8_source_file_count = _safe_int(
        newbie_tail8_attention_compute_review.get("source_file_count")
    )
    newbie_tail8_existing_source_file_count = _safe_int(
        newbie_tail8_attention_compute_review.get("existing_source_file_count")
    )
    newbie_tail8_missing_source_file_count = _safe_int(
        newbie_tail8_attention_compute_review.get("missing_source_file_count")
    )
    newbie_tail8_source_file_presence_status = str(
        newbie_tail8_attention_compute_review.get("source_file_presence_status") or ""
    )
    newbie_tail8_source_file_presence_fail_closed = (
        newbie_tail8_attention_compute_review_fail_closed
        and newbie_tail8_source_file_count > 0
        and newbie_tail8_existing_source_file_count >= 0
        and newbie_tail8_missing_source_file_count >= 0
        and newbie_tail8_source_file_count
        == newbie_tail8_existing_source_file_count + newbie_tail8_missing_source_file_count
        and (
            (
                newbie_tail8_missing_source_file_count == 0
                and newbie_tail8_source_file_presence_status == "source_files_present"
            )
            or (
                newbie_tail8_missing_source_file_count > 0
                and newbie_tail8_source_file_presence_status
                == "source_files_missing_fail_closed"
                and bool(
                    _strings(
                        newbie_tail8_attention_compute_review.get(
                            "missing_source_file_pair_ids"
                        )
                    )
                )
                and bool(
                    _strings(
                        newbie_tail8_attention_compute_review.get(
                            "missing_source_file_ids"
                        )
                    )
                )
            )
        )
    )
    newbie_tail8_forward_anomaly_review_fail_closed = (
        bool(newbie_tail8_forward_anomaly_review)
        and str(newbie_tail8_forward_anomaly_review.get("roadmap") or "") == ROADMAP
        and bool(newbie_tail8_forward_anomaly_review.get("fail_closed"))
        and bool(newbie_tail8_forward_anomaly_review.get("not_release_evidence"))
        and str(newbie_tail8_forward_anomaly_review.get("candidate") or "")
        == "newbie_target_scope:tail8_attention seed:2027"
        and bool(newbie_tail8_forward_anomaly_review.get("review_ready"))
        and _safe_int(newbie_tail8_forward_anomaly_review.get("unsafe_artifact_count")) == 0
        and _safe_int(newbie_tail8_forward_anomaly_review.get("comparison_source_present_count"))
        + _safe_int(newbie_tail8_forward_anomaly_review.get("comparison_source_missing_count"))
        == 3
        and (
            _safe_int(newbie_tail8_forward_anomaly_review.get("comparison_source_missing_count"))
            == 0
            or bool(
                _strings(
                    newbie_tail8_forward_anomaly_review.get(
                        "missing_source_manifest_ids"
                    )
                )
            )
        )
        and not bool(newbie_tail8_forward_anomaly_review.get("publishable"))
        and not bool(newbie_tail8_forward_anomaly_review.get("safe_to_auto_start"))
        and not bool(newbie_tail8_forward_anomaly_review.get("release_claim_allowed"))
    )
    tail8_preflight_forward_missing_count = _safe_int(
        newbie_tail8_seed2027_rerun_preflight.get(
            "forward_anomaly_comparison_source_missing_count"
        )
    )
    tail8_preflight_forward_present_count = _safe_int(
        newbie_tail8_seed2027_rerun_preflight.get(
            "forward_anomaly_comparison_source_present_count"
        )
    )
    tail8_preflight_forward_gate_fail_closed = (
        tail8_preflight_forward_present_count + tail8_preflight_forward_missing_count
        == 3
        and bool(
            newbie_tail8_seed2027_rerun_preflight.get(
                "forward_anomaly_comparison_source_ready"
            )
        )
        == (tail8_preflight_forward_missing_count == 0)
        and (
            tail8_preflight_forward_missing_count == 0
            or bool(
                _strings(
                    newbie_tail8_seed2027_rerun_preflight.get(
                        "forward_anomaly_missing_source_manifest_ids"
                    )
                )
            )
        )
    )
    newbie_tail8_seed2027_rerun_preflight_fail_closed = (
        bool(newbie_tail8_seed2027_rerun_preflight)
        and str(newbie_tail8_seed2027_rerun_preflight.get("roadmap") or "") == ROADMAP
        and bool(newbie_tail8_seed2027_rerun_preflight.get("fail_closed"))
        and bool(newbie_tail8_seed2027_rerun_preflight.get("not_release_evidence"))
        and str(newbie_tail8_seed2027_rerun_preflight.get("candidate") or "")
        == "newbie_target_scope:tail8_attention seed:2027"
        and bool(newbie_tail8_seed2027_rerun_preflight.get("review_ready"))
        and tail8_preflight_forward_gate_fail_closed
        and (
            not bool(newbie_tail8_seed2027_rerun_preflight.get("manual_rerun_ready"))
            or (
                bool(newbie_tail8_seed2027_rerun_preflight.get("gpu_idle_ready"))
                and bool(newbie_tail8_seed2027_rerun_preflight.get("environment_snapshot_ready"))
                and bool(newbie_tail8_seed2027_rerun_preflight.get("disk_space_ready"))
                and not bool(newbie_tail8_seed2027_rerun_preflight.get("gpu_compute_apps_present"))
                and _safe_int(
                    newbie_tail8_seed2027_rerun_preflight.get("gpu_compute_apps_count")
                )
                == 0
            )
        )
        and not bool(newbie_tail8_seed2027_rerun_preflight.get("safe_to_auto_start"))
        and not bool(newbie_tail8_seed2027_rerun_preflight.get("release_claim_allowed"))
    )
    protected_followup_gpu_queue_fail_closed = (
        bool(protected_followup_gpu_queue)
        and str(protected_followup_gpu_queue.get("roadmap") or "") == ROADMAP
        and bool(protected_followup_gpu_queue.get("fail_closed"))
        and bool(protected_followup_gpu_queue.get("not_release_evidence"))
        and str(protected_followup_gpu_queue.get("execution_policy") or "")
        == "manual_protected_followup_only"
        and _safe_int(protected_followup_gpu_queue.get("unsafe_action_count")) == 0
        and not bool(protected_followup_gpu_queue.get("safe_to_auto_start"))
        and not bool(protected_followup_gpu_queue.get("release_claim_allowed"))
    )
    blocker_matrix_fail_closed = (
        bool(blocker_matrix)
        and str(blocker_matrix.get("roadmap") or "") == ROADMAP
        and bool(blocker_matrix.get("fail_closed"))
        and bool(blocker_matrix.get("not_release_evidence"))
        and _safe_int(blocker_matrix.get("unsafe_action_count")) == 0
        and not bool(blocker_matrix.get("safe_to_auto_start"))
        and not bool(blocker_matrix.get("release_claim_allowed"))
    )
    blocker_handoff_fail_closed = (
        bool(blocker_handoff)
        and str(blocker_handoff.get("roadmap") or "") == ROADMAP
        and bool(blocker_handoff.get("fail_closed"))
        and bool(blocker_handoff.get("not_release_evidence"))
        and _safe_int(blocker_handoff.get("unsafe_row_count")) == 0
        and bool(blocker_handoff.get("resolution_contract_ok"))
        and _safe_int(blocker_handoff.get("resolution_contract_bad_count")) == 0
        and not bool(blocker_handoff.get("release_claim_after_resolution_allowed"))
        and str(blocker_handoff.get("execution_policy") or "")
        == "manual_or_external_input_handoff_only"
        and not bool(blocker_handoff.get("safe_to_auto_start"))
        and not bool(blocker_handoff.get("release_claim_allowed"))
    )
    action_dependency_graph_fail_closed = (
        bool(action_dependency_graph)
        and str(action_dependency_graph.get("roadmap") or "") == ROADMAP
        and bool(action_dependency_graph.get("fail_closed"))
        and bool(action_dependency_graph.get("not_release_evidence"))
        and _safe_int(action_dependency_graph.get("unsafe_action_count")) == 0
        and _safe_int(action_dependency_graph.get("action_node_count"))
        == sum(
            1
            for action in _list(readiness.get("next_actions"))
            if str(_mapping(action).get("readiness_state") or "") != "json_closed"
        )
        and bool(action_dependency_graph.get("refresh_sequence_terminal_guard_ok"))
        and str(action_dependency_graph.get("execution_policy") or "")
        == "dependency_graph_only_manual_or_external_input"
        and not bool(action_dependency_graph.get("safe_to_auto_start"))
        and not bool(action_dependency_graph.get("release_claim_allowed"))
    )
    action_unblock_sequence_fail_closed = (
        bool(action_unblock_sequence)
        and str(action_unblock_sequence.get("roadmap") or "") == ROADMAP
        and bool(action_unblock_sequence.get("fail_closed"))
        and bool(action_unblock_sequence.get("not_release_evidence"))
        and _safe_int(action_unblock_sequence.get("stage_count")) > 0
        and _safe_int(action_unblock_sequence.get("unsafe_stage_count")) == 0
        and bool(action_unblock_sequence.get("refresh_sequence_terminal_guard_ok"))
        and bool(action_unblock_sequence.get("terminal_guard_required"))
        and str(action_unblock_sequence.get("execution_policy") or "")
        == "ordered_handoff_only_manual_or_external_input"
        and not bool(action_unblock_sequence.get("safe_to_auto_start"))
        and not bool(action_unblock_sequence.get("release_claim_allowed"))
    )
    blocker_presence_fail_closed = (
        bool(blocker_presence)
        and str(blocker_presence.get("roadmap") or "") == ROADMAP
        and bool(blocker_presence.get("fail_closed"))
        and bool(blocker_presence.get("not_release_evidence"))
        and _safe_int(blocker_presence.get("unsafe_row_count")) == 0
        and str(blocker_presence.get("execution_policy") or "")
        == "read_only_artifact_presence_audit"
        and not bool(blocker_presence.get("safe_to_auto_start"))
        and not bool(blocker_presence.get("release_claim_allowed"))
    )
    release_exit_fail_closed = (
        bool(release_exit)
        and str(release_exit.get("roadmap") or "") == ROADMAP
        and bool(release_exit.get("fail_closed"))
        and bool(release_exit.get("not_release_evidence"))
        and _safe_int(release_exit.get("unsafe_gate_count")) == 0
        and _safe_int(release_exit.get("json_only_exit_available_count")) == 0
        and str(release_exit.get("execution_policy") or "")
        == "release_claim_exit_criteria_only"
        and not bool(release_exit.get("safe_to_auto_start"))
        and not bool(release_exit.get("release_claim_allowed"))
    )
    release_gate_input_dependency_fail_closed = (
        bool(release_gate_input_dependency)
        and str(release_gate_input_dependency.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_dependency.get("fail_closed"))
        and bool(release_gate_input_dependency.get("not_release_evidence"))
        and _safe_int(release_gate_input_dependency.get("unsafe_input_count")) == 0
        and _safe_int(release_gate_input_dependency.get("json_only_resolution_available_count")) == 0
        and str(release_gate_input_dependency.get("execution_policy") or "")
        == "release_gate_input_dependency_only"
        and not bool(release_gate_input_dependency.get("safe_to_auto_start"))
        and not bool(release_gate_input_dependency.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_plan_fail_closed = (
        bool(release_gate_post_input_refresh_plan)
        and str(release_gate_post_input_refresh_plan.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_plan.get("fail_closed"))
        and bool(release_gate_post_input_refresh_plan.get("not_release_evidence"))
        and _safe_int(release_gate_post_input_refresh_plan.get("unsafe_plan_count")) == 0
        and "refresh_gpu_bubble_readiness_next_actions"
        in _strings(release_gate_post_input_refresh_plan.get("required_refresh_command_ids"))
        and "refresh_gpu_bubble_terminal_self_check"
        in _strings(release_gate_post_input_refresh_plan.get("required_refresh_command_ids"))
        and "run_gpu_bubble_release_readiness_guard"
        in _strings(release_gate_post_input_refresh_plan.get("required_refresh_command_ids"))
        and str(release_gate_post_input_refresh_plan.get("execution_policy") or "")
        == "post_input_json_refresh_plan_only"
        and not bool(release_gate_post_input_refresh_plan.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_plan.get("release_claim_allowed"))
    )
    release_gate_input_detection_source_fail_closed = (
        bool(release_gate_input_detection_source)
        and str(release_gate_input_detection_source.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_detection_source.get("fail_closed"))
        and bool(release_gate_input_detection_source.get("not_release_evidence"))
        and _safe_int(release_gate_input_detection_source.get("unsafe_detector_count")) == 0
        and str(release_gate_input_detection_source.get("execution_policy") or "")
        == "release_gate_input_detection_source_only"
        and not bool(release_gate_input_detection_source.get("safe_to_auto_start"))
        and not bool(release_gate_input_detection_source.get("release_claim_allowed"))
    )
    release_gate_input_acceptance_criteria_fail_closed = (
        bool(release_gate_input_acceptance_criteria)
        and str(release_gate_input_acceptance_criteria.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_acceptance_criteria.get("fail_closed"))
        and bool(release_gate_input_acceptance_criteria.get("not_release_evidence"))
        and _safe_int(release_gate_input_acceptance_criteria.get("unsafe_acceptance_count")) == 0
        and _safe_int(
            release_gate_input_acceptance_criteria.get(
                "accepted_with_blocked_criteria_count"
            )
        )
        == 0
        and str(release_gate_input_acceptance_criteria.get("execution_policy") or "")
        == "release_gate_input_acceptance_criteria_only"
        and not bool(release_gate_input_acceptance_criteria.get("safe_to_auto_start"))
        and not bool(release_gate_input_acceptance_criteria.get("release_claim_allowed"))
    )
    release_gate_input_refresh_readiness_fail_closed = (
        bool(release_gate_input_refresh_readiness)
        and str(release_gate_input_refresh_readiness.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_refresh_readiness.get("fail_closed"))
        and bool(release_gate_input_refresh_readiness.get("not_release_evidence"))
        and _safe_int(release_gate_input_refresh_readiness.get("unsafe_refresh_count")) == 0
        and str(release_gate_input_refresh_readiness.get("execution_policy") or "")
        == "release_gate_input_refresh_readiness_only"
        and not bool(release_gate_input_refresh_readiness.get("safe_to_auto_start"))
        and not bool(release_gate_input_refresh_readiness.get("release_claim_allowed"))
    )
    release_gate_input_refresh_blocker_fail_closed = (
        bool(release_gate_input_refresh_blocker)
        and str(release_gate_input_refresh_blocker.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_refresh_blocker.get("fail_closed"))
        and bool(release_gate_input_refresh_blocker.get("not_release_evidence"))
        and _safe_int(release_gate_input_refresh_blocker.get("unsafe_blocker_count")) == 0
        and str(release_gate_input_refresh_blocker.get("execution_policy") or "")
        == "release_gate_input_refresh_blocker_only"
        and not bool(release_gate_input_refresh_blocker.get("safe_to_auto_start"))
        and not bool(release_gate_input_refresh_blocker.get("release_claim_allowed"))
    )
    release_gate_input_lifecycle_fail_closed = (
        bool(release_gate_input_lifecycle)
        and str(release_gate_input_lifecycle.get("roadmap") or "") == ROADMAP
        and bool(release_gate_input_lifecycle.get("fail_closed"))
        and bool(release_gate_input_lifecycle.get("not_release_evidence"))
        and _safe_int(release_gate_input_lifecycle.get("unsafe_input_count")) == 0
        and str(release_gate_input_lifecycle.get("execution_policy") or "")
        == "release_gate_input_lifecycle_summary_only"
        and not bool(release_gate_input_lifecycle.get("safe_to_auto_start"))
        and not bool(release_gate_input_lifecycle.get("release_claim_allowed"))
    )
    external_input_release_gate_alignment_fail_closed = (
        bool(external_input_release_gate_alignment)
        and str(external_input_release_gate_alignment.get("roadmap") or "") == ROADMAP
        and bool(external_input_release_gate_alignment.get("fail_closed"))
        and bool(external_input_release_gate_alignment.get("not_release_evidence"))
        and bool(external_input_release_gate_alignment.get("alignment_ok"))
        and _safe_int(
            external_input_release_gate_alignment.get("unsafe_alignment_count")
        )
        == 0
        and _safe_int(
            external_input_release_gate_alignment.get(
                "external_missing_from_release_gate_count"
            )
        )
        == 0
        and _safe_int(
            external_input_release_gate_alignment.get(
                "release_external_missing_from_transition_count"
            )
        )
        == 0
        and str(external_input_release_gate_alignment.get("execution_policy") or "")
        == "external_input_release_gate_alignment_only"
        and not bool(external_input_release_gate_alignment.get("safe_to_auto_start"))
        and not bool(external_input_release_gate_alignment.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_command_surface_fail_closed = (
        bool(release_gate_post_input_refresh_command_surface)
        and str(release_gate_post_input_refresh_command_surface.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_command_surface.get("fail_closed"))
        and bool(release_gate_post_input_refresh_command_surface.get("not_release_evidence"))
        and _safe_int(
            release_gate_post_input_refresh_command_surface.get("unsafe_command_count")
        )
        == 0
        and str(release_gate_post_input_refresh_command_surface.get("execution_policy") or "")
        == "post_input_refresh_command_surface_only"
        and not bool(release_gate_post_input_refresh_command_surface.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_command_surface.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_sequence_integrity_fail_closed = (
        bool(release_gate_post_input_refresh_sequence_integrity)
        and str(release_gate_post_input_refresh_sequence_integrity.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_sequence_integrity.get("fail_closed"))
        and bool(release_gate_post_input_refresh_sequence_integrity.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_sequence_integrity.get("sequence_ok"))
        and _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("unsafe_sequence_count")
        )
        == 0
        and str(release_gate_post_input_refresh_sequence_integrity.get("execution_policy") or "")
        == "post_input_refresh_sequence_integrity_only"
        and not bool(release_gate_post_input_refresh_sequence_integrity.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_sequence_integrity.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_terminal_guard_dependency_fail_closed = (
        bool(release_gate_post_input_refresh_terminal_guard_dependency)
        and str(release_gate_post_input_refresh_terminal_guard_dependency.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("fail_closed"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("dependency_ok"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_required"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_self_check_required"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("release_guard_required"))
        and bool(release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_tail_ok"))
        and bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "all_json_refresh_commands_before_terminal_guard"
            )
        )
        and _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "unsafe_dependency_count"
            )
        )
        == 0
        and str(release_gate_post_input_refresh_terminal_guard_dependency.get("execution_policy") or "")
        == "post_input_refresh_terminal_guard_dependency_only"
        and not bool(release_gate_post_input_refresh_terminal_guard_dependency.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_terminal_guard_dependency.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_artifact_coverage_fail_closed = (
        bool(release_gate_post_input_refresh_artifact_coverage)
        and str(release_gate_post_input_refresh_artifact_coverage.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_artifact_coverage.get("fail_closed"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("coverage_ok"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("readiness_artifact_required"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("terminal_artifact_required"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("release_guard_artifact_required"))
        and bool(release_gate_post_input_refresh_artifact_coverage.get("terminal_guard_dependency_ok"))
        and _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get(
                "missing_coverage_input_count"
            )
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get(
                "unsafe_artifact_coverage_count"
            )
        )
        == 0
        and str(release_gate_post_input_refresh_artifact_coverage.get("execution_policy") or "")
        == "post_input_refresh_artifact_coverage_only"
        and not bool(release_gate_post_input_refresh_artifact_coverage.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_artifact_coverage.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_command_artifact_link_fail_closed = (
        bool(release_gate_post_input_refresh_command_artifact_link)
        and str(release_gate_post_input_refresh_command_artifact_link.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_command_artifact_link.get("fail_closed"))
        and bool(release_gate_post_input_refresh_command_artifact_link.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_command_artifact_link.get("link_ok"))
        and bool(release_gate_post_input_refresh_command_artifact_link.get("artifact_coverage_ok"))
        and bool(release_gate_post_input_refresh_command_artifact_link.get("blocked_until_input_acceptance"))
        and _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("missing_link_artifact_count")
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("extra_link_artifact_count")
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("unsafe_link_count")
        )
        == 0
        and str(release_gate_post_input_refresh_command_artifact_link.get("execution_policy") or "")
        == "post_input_refresh_command_artifact_link_only"
        and not bool(release_gate_post_input_refresh_command_artifact_link.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_command_artifact_link.get("release_claim_allowed"))
    )
    external_input_json_refresh_runner_expected_count = _safe_int(
        external_input_json_refresh_runner_manifest_summary.get("expected_command_count")
    )
    release_gate_post_input_refresh_guard_consumption_runner_row = next(
        (
            _mapping(item)
            for item in _list(release_gate_post_input_refresh_guard_consumption.get("rows"))
            if str(_mapping(item).get("summary_id") or "")
            == "external_input_json_refresh_runner_manifest_summary"
        ),
        {},
    )
    release_gate_post_input_refresh_guard_consumption_runner_row_fail_closed = (
        bool(release_gate_post_input_refresh_guard_consumption_runner_row)
        and bool(release_gate_post_input_refresh_guard_consumption_runner_row.get("present"))
        and bool(release_gate_post_input_refresh_guard_consumption_runner_row.get("fail_closed"))
        and bool(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "not_release_evidence"
            )
        )
        and bool(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "safety_contract_ok"
            )
        )
        and not bool(release_gate_post_input_refresh_guard_consumption_runner_row.get("unsafe"))
        and not bool(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "safe_to_auto_start"
            )
        )
        and not bool(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "release_claim_allowed"
            )
        )
        and bool(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "row_execution_consistent"
            )
        )
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "expected_command_count"
            )
        )
        == external_input_json_refresh_runner_expected_count
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption_runner_row.get("row_count")
        )
        == external_input_json_refresh_runner_expected_count
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "missing_output_row_count"
            )
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "row_forbidden_heavy_flag_count"
            )
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption_runner_row.get(
                "unsafe_row_count"
            )
        )
        == 0
    )
    release_gate_post_input_refresh_guard_consumption_fail_closed = (
        bool(release_gate_post_input_refresh_guard_consumption)
        and str(release_gate_post_input_refresh_guard_consumption.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_guard_consumption.get("fail_closed"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("consumption_ok"))
        and str(release_gate_post_input_refresh_guard_consumption.get("guard_command_id") or "")
        == "run_gpu_bubble_release_readiness_guard"
        and bool(release_gate_post_input_refresh_guard_consumption.get("input_artifacts_consumed"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("guard_artifact_produced"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("command_artifact_link_ok"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("artifact_coverage_ok"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("terminal_guard_dependency_ok"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("terminal_lineage_required"))
        and bool(release_gate_post_input_refresh_guard_consumption.get("blocked_until_input_acceptance"))
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "missing_consumed_summary_count"
            )
        )
        == 0
        and _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "unsafe_consumed_summary_count"
            )
        )
        == 0
        and str(release_gate_post_input_refresh_guard_consumption.get("execution_policy") or "")
        == "post_input_refresh_guard_consumption_only"
        and release_gate_post_input_refresh_guard_consumption_runner_row_fail_closed
        and not bool(release_gate_post_input_refresh_guard_consumption.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_guard_consumption.get("release_claim_allowed"))
    )
    release_gate_post_input_refresh_guard_report_acceptance_fail_closed = (
        bool(release_gate_post_input_refresh_guard_report_acceptance)
        and str(release_gate_post_input_refresh_guard_report_acceptance.get("roadmap") or "") == ROADMAP
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("fail_closed"))
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("not_release_evidence"))
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("acceptance_ok"))
        and str(release_gate_post_input_refresh_guard_report_acceptance.get("guard_command_id") or "")
        == "run_gpu_bubble_release_readiness_guard"
        and str(release_gate_post_input_refresh_guard_report_acceptance.get("guard_report_artifact_id") or "")
        == "gpu_bubble_release_readiness_guard_report"
        and str(release_gate_post_input_refresh_guard_report_acceptance.get("expected_report_status") or "")
        == "guard_passed_blocked_release_claim"
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("expected_ok"))
        and _safe_int(release_gate_post_input_refresh_guard_report_acceptance.get("expected_failure_count")) == 0
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("guard_consumption_ok"))
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("input_artifacts_consumed"))
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("guard_artifact_produced"))
        and bool(release_gate_post_input_refresh_guard_report_acceptance.get("blocked_until_input_acceptance"))
        and _safe_int(release_gate_post_input_refresh_guard_report_acceptance.get("unsafe_acceptance_count")) == 0
        and str(release_gate_post_input_refresh_guard_report_acceptance.get("execution_policy") or "")
        == "post_input_refresh_guard_report_acceptance_only"
        and not bool(release_gate_post_input_refresh_guard_report_acceptance.get("safe_to_auto_start"))
        and not bool(release_gate_post_input_refresh_guard_report_acceptance.get("release_claim_allowed"))
    )
    external_input_json_refresh_runner_manifest_fail_closed = (
        bool(external_input_json_refresh_runner_manifest_summary)
        and str(external_input_json_refresh_runner_manifest_summary.get("roadmap") or "") == ROADMAP
        and bool(external_input_json_refresh_runner_manifest_summary.get("fail_closed"))
        and bool(external_input_json_refresh_runner_manifest_summary.get("safety_contract_ok"))
        and bool(external_input_json_refresh_runner_manifest_summary.get("not_release_evidence"))
        and str(external_input_json_refresh_runner_manifest_summary.get("execution_policy") or "")
        == "external_input_json_refresh_runner_manifest_acceptance_only"
        and bool(
            external_input_json_refresh_runner_manifest_summary.get(
                "row_execution_consistent"
            )
        )
        and bool(external_input_json_refresh_runner_manifest_summary.get("stage_manifest_ok"))
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("stage_manifest_issue_count")
        )
        == 0
        and _safe_int(external_input_json_refresh_runner_manifest_summary.get("row_count"))
        == external_input_json_refresh_runner_expected_count
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "missing_output_row_count"
            )
        )
        == 0
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "row_forbidden_heavy_flag_count"
            )
        )
        == 0
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("unsafe_row_count")
        )
        == 0
        and _safe_int(external_input_json_refresh_runner_manifest_summary.get("stage_count"))
        == external_input_json_refresh_runner_expected_count
        and _safe_int(external_input_json_refresh_runner_manifest_summary.get("script_count"))
        == external_input_json_refresh_runner_expected_count
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("expected_output_count")
        )
        == external_input_json_refresh_runner_expected_count
        and _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "stage_manifest_forbidden_heavy_flag_count"
            )
        )
        == 0
        and not bool(external_input_json_refresh_runner_manifest_summary.get("safe_to_auto_start"))
        and not bool(external_input_json_refresh_runner_manifest_summary.get("release_claim_allowed"))
    )
    command_surface_fail_closed = (
        bool(command_surface)
        and str(command_surface.get("roadmap") or "") == ROADMAP
        and bool(command_surface.get("fail_closed"))
        and bool(command_surface.get("not_release_evidence"))
        and _safe_int(command_surface.get("unsafe_command_count")) == 0
        and _safe_int(command_surface.get("release_claim_allowed_after_success_count")) == 0
        and str(command_surface.get("execution_policy") or "")
        == "manual_protected_or_external_input_only"
        and not bool(command_surface.get("safe_to_auto_start"))
        and not bool(command_surface.get("release_claim_allowed"))
    )
    protected_run_plan_chain_fail_closed = (
        bool(protected_run_plan_chain)
        and str(protected_run_plan_chain.get("roadmap") or "") == ROADMAP
        and bool(protected_run_plan_chain.get("fail_closed"))
        and bool(protected_run_plan_chain.get("not_release_evidence"))
        and _safe_int(protected_run_plan_chain.get("missing_artifact_count")) == 0
        and _safe_int(protected_run_plan_chain.get("unsafe_artifact_count")) == 0
        and _safe_int(protected_run_plan_chain.get("unsafe_command_count")) == 0
        and _safe_int(protected_run_plan_chain.get("unsafe_scaffold_count")) == 0
        and _safe_int(protected_run_plan_chain.get("release_claim_allowed_after_success_command_count")) == 0
        and str(protected_run_plan_chain.get("execution_policy") or "")
        == "manual_protected_followup_only"
        and not bool(protected_run_plan_chain.get("safe_to_auto_start"))
        and not bool(protected_run_plan_chain.get("release_claim_allowed"))
    )
    source_downstream_contract = _mapping(readiness.get("source_and_downstream_artifact_contract_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_and_downstream_artifact_contract_summary")
    )
    source_downstream_contract_fail_closed = (
        bool(source_downstream_contract)
        and str(source_downstream_contract.get("roadmap") or "") == ROADMAP
        and bool(source_downstream_contract.get("fail_closed"))
        and bool(source_downstream_contract.get("not_release_evidence"))
        and _safe_int(source_downstream_contract.get("source_path_count")) == len(_list(readiness.get("source_paths")))
        and _safe_int(source_downstream_contract.get("downstream_artifact_count"))
        == len(_list(readiness.get("downstream_artifacts")))
        and _safe_int(source_downstream_contract.get("unsafe_artifact_count")) == 0
        and not bool(source_downstream_contract.get("safe_to_auto_start"))
        and not bool(source_downstream_contract.get("release_claim_allowed"))
    )
    source_artifact_inventory_fail_closed = (
        bool(source_artifact_inventory)
        and str(source_artifact_inventory.get("roadmap") or "") == ROADMAP
        and bool(source_artifact_inventory.get("fail_closed"))
        and bool(source_artifact_inventory.get("not_release_evidence"))
        and bool(source_artifact_inventory.get("does_not_run_training"))
        and bool(source_artifact_inventory.get("does_not_run_cuda"))
        and _safe_int(source_artifact_inventory.get("source_artifact_count")) == len(_list(readiness.get("source_paths")))
        and _safe_int(source_artifact_inventory.get("unsafe_source_artifact_count")) == 0
        and _safe_int(source_artifact_inventory.get("missing_source_artifact_count")) == 0
        and _safe_int(source_artifact_inventory.get("load_error_count")) == 0
        and _safe_int(source_artifact_inventory.get("roadmap_mismatch_count")) == 0
        and _safe_int(source_artifact_inventory.get("release_unsafe_count")) == 0
        and str(source_artifact_inventory.get("execution_policy") or "")
        == "read_only_source_artifact_inventory"
        and not bool(source_artifact_inventory.get("publishable"))
        and not bool(source_artifact_inventory.get("safe_to_auto_start"))
        and not bool(source_artifact_inventory.get("release_claim_allowed"))
    )
    evidence_summary_inventory_fail_closed = (
        bool(evidence_summary_inventory)
        and str(evidence_summary_inventory.get("roadmap") or "") == ROADMAP
        and bool(evidence_summary_inventory.get("fail_closed"))
        and bool(evidence_summary_inventory.get("not_release_evidence"))
        and bool(evidence_summary_inventory.get("does_not_run_training"))
        and bool(evidence_summary_inventory.get("does_not_run_cuda"))
        and _safe_int(evidence_summary_inventory.get("evidence_key_count")) > 0
        and _safe_int(evidence_summary_inventory.get("unsafe_evidence_entry_count")) == 0
        and _safe_int(evidence_summary_inventory.get("roadmap_mismatch_entry_count")) == 0
        and _safe_int(evidence_summary_inventory.get("release_unsafe_entry_count")) == 0
        and str(evidence_summary_inventory.get("execution_policy") or "")
        == "read_only_evidence_summary_inventory"
        and not bool(evidence_summary_inventory.get("publishable"))
        and not bool(evidence_summary_inventory.get("safe_to_auto_start"))
        and not bool(evidence_summary_inventory.get("release_claim_allowed"))
    )
    external_input_transition = _mapping(readiness.get("external_input_transition_table"))
    transition_rows = [_mapping(item) for item in _list(external_input_transition.get("rows"))]
    external_input_transition_fail_closed = (
        bool(external_input_transition)
        and str(external_input_transition.get("roadmap") or "") == ROADMAP
        and bool(external_input_transition.get("fail_closed"))
        and bool(external_input_transition.get("not_release_evidence"))
        and _safe_int(external_input_transition.get("row_count")) == len(transition_rows)
        and _safe_int(external_input_transition.get("unsafe_row_count")) == 0
        and not bool(external_input_transition.get("safe_to_auto_start"))
        and not bool(external_input_transition.get("release_claim_allowed"))
    )
    first_release_policy_fail_closed = (
        bool(first_release_policy)
        and str(first_release_policy.get("roadmap") or "") == ROADMAP
        and bool(first_release_policy.get("fail_closed"))
        and bool(first_release_policy.get("not_release_evidence"))
        and _safe_int(first_release_policy.get("unsafe_policy_count")) == 0
        and str(first_release_policy.get("scope") or "") == "gpu_bubble_readiness_only"
        and not bool(first_release_policy.get("stable_first_release_blocked_by_this_artifact"))
        and not bool(first_release_policy.get("gpu_bubble_release_claim_allowed"))
        and bool(first_release_policy.get("gpu_bubble_release_claim_blocked"))
        and str(first_release_policy.get("recommended_release_policy") or "")
        == "ship_stable_baseline_without_gpu_bubble_gain_claim"
        and str(first_release_policy.get("claim_publication_scope") or "")
        == "non_release_benchmark_claims"
        and bool(first_release_policy.get("does_not_prove_global_product_release"))
        and not bool(first_release_policy.get("safe_to_auto_start"))
        and not bool(first_release_policy.get("release_claim_allowed"))
    )
    source_axis_freshness_fail_closed = (
        bool(source_axis_freshness_summary)
        and str(source_axis_freshness_summary.get("roadmap") or "") == ROADMAP
        and bool(source_axis_freshness_summary.get("fail_closed"))
        and bool(source_axis_freshness_summary.get("not_release_evidence"))
        and bool(source_axis_freshness_summary.get("does_not_run_training"))
        and bool(source_axis_freshness_summary.get("does_not_run_cuda"))
        and _safe_int(source_axis_freshness_summary.get("unsafe_audit_count")) == 0
        and not bool(source_axis_freshness_summary.get("publishable"))
        and not bool(source_axis_freshness_summary.get("safe_to_auto_start"))
        and not bool(source_axis_freshness_summary.get("release_claim_allowed"))
    )
    remaining_work_fail_closed = (
        bool(remaining)
        and str(remaining.get("roadmap") or "") == ROADMAP
        and bool(remaining.get("fail_closed"))
        and bool(remaining.get("not_release_evidence"))
        and _safe_int(remaining.get("total_action_count")) == len(_list(readiness.get("next_actions")))
        and _safe_int(remaining.get("unsafe_action_count")) == 0
        and bool(remaining.get("gpu_bubble_release_claim_blocked"))
        and not bool(remaining.get("stable_first_release_blocked_by_this_artifact"))
        and str(remaining.get("recommended_release_policy") or "")
        == "ship_stable_baseline_without_gpu_bubble_gain_claim"
        and not bool(remaining.get("safe_to_auto_start"))
        and not bool(remaining.get("release_claim_allowed"))
    )

    hard_gates = _strings(
        unblocker.get("gpu_bubble_release_hard_gate_ids")
    ) or _strings(remaining.get("gpu_bubble_release_hard_gate_ids"))
    missing_inputs = (
        _strings(unblocker.get("missing_external_inputs"))
        or _strings(input_resolution.get("missing_external_inputs"))
        or _strings(handoff.get("missing_external_inputs"))
    )
    manual_inputs = _strings(manual_evidence_blocking.get("next_required_inputs")) or _strings(
        unblocker.get("post_manual_next_required_inputs")
    )
    pipeline_complete = bool(pipeline.get("pipeline_complete")) or bool(
        unblocker.get("source_cache_axis_pipeline_complete")
    )
    stage_ok = _safe_int(
        pipeline.get("stage_ok_count"),
        _safe_int(unblocker.get("source_cache_axis_stage_ok_count")),
    )
    stage_count = _safe_int(
        pipeline.get("stage_count"),
        _safe_int(unblocker.get("source_cache_axis_stage_count")),
    )
    stage_lineage = _source_cache_stage_lineage_summary(pipeline)
    stage_lineage_complete = _source_cache_stage_lineage_complete(
        stage_lineage,
        expected_stage_count=stage_count,
    )
    chain_complete = (
        pipeline_complete
        and stage_count > 0
        and stage_ok == stage_count
        and stage_lineage_complete
        and roadmap_acceptance_fail_closed
        and roadmap_execution_fail_closed
        and matrix_readiness_fail_closed
        and identity_registry_fail_closed
        and normalized_gate_mapping_fail_closed
        and normalized_gate_explanation_fail_closed
        and next_action_machine_fail_closed
        and next_action_contract_fail_closed
        and manual_review_queue_fail_closed
        and manual_review_artifact_chain_fail_closed
        and sdxl_diagnostic_artifact_chain_fail_closed
        and newbie_blockskip_compute_bound_chain_fail_closed
        and newbie_compute_diagnosis_chain_fail_closed
        and newbie_tail8_attention_compute_review_fail_closed
        and newbie_tail8_target_depth_progression_fail_closed
        and newbie_tail8_source_file_presence_fail_closed
        and newbie_tail8_forward_anomaly_review_fail_closed
        and newbie_tail8_seed2027_rerun_preflight_fail_closed
        and remaining_work_fail_closed
        and protected_followup_gpu_queue_fail_closed
        and blocker_matrix_fail_closed
        and blocker_handoff_fail_closed
        and action_dependency_graph_fail_closed
        and action_unblock_sequence_fail_closed
        and blocker_presence_fail_closed
        and release_exit_fail_closed
        and release_gate_input_dependency_fail_closed
        and release_gate_post_input_refresh_plan_fail_closed
        and release_gate_input_detection_source_fail_closed
        and release_gate_input_acceptance_criteria_fail_closed
        and release_gate_input_refresh_readiness_fail_closed
        and release_gate_input_refresh_blocker_fail_closed
        and release_gate_input_lifecycle_fail_closed
        and external_input_release_gate_alignment_fail_closed
        and release_gate_post_input_refresh_command_surface_fail_closed
        and release_gate_post_input_refresh_sequence_integrity_fail_closed
        and release_gate_post_input_refresh_terminal_guard_dependency_fail_closed
        and release_gate_post_input_refresh_artifact_coverage_fail_closed
        and release_gate_post_input_refresh_command_artifact_link_fail_closed
        and release_gate_post_input_refresh_guard_consumption_fail_closed
        and release_gate_post_input_refresh_guard_report_acceptance_fail_closed
        and external_input_json_refresh_runner_manifest_fail_closed
        and command_surface_fail_closed
        and protected_run_plan_chain_fail_closed
        and source_artifact_inventory_fail_closed
        and evidence_summary_inventory_fail_closed
        and source_downstream_contract_fail_closed
        and external_input_transition_fail_closed
        and first_release_policy_fail_closed
        and source_axis_freshness_fail_closed
    )
    terminal_blocked = bool(hard_gates) and bool(missing_inputs or manual_inputs)
    progress_audit = _json_only_progress_audit(
        readiness=readiness,
        remaining=remaining,
        chain_complete=chain_complete,
    )
    filesystem_audit = _external_input_filesystem_audit(
        expected_missing_inputs=missing_inputs,
        external_input_intake_registry=intake,
        live_external_input_intake_registry=live_intake,
    )
    source_cache_negative = _source_cache_negative_evidence_summary(
        external_input_intake_registry=intake,
        source_axis_requirement=source_requirement,
        source_axis_freshness_dedupe_audit=source_freshness,
        newbie_warm_cache_inventory=warm_cache,
    )
    terminal_input_resolution_summary = {
        "summary_version": _safe_int(input_resolution.get("summary_version"), 1),
        "roadmap": str(input_resolution.get("roadmap") or ROADMAP),
        "external_input_required": bool(input_resolution.get("external_input_required", bool(missing_inputs))),
        "missing_external_inputs": (_strings(input_resolution.get("missing_external_inputs")) or missing_inputs)[:20],
        "sd15_checkpoint_exists": bool(input_resolution.get("sd15_checkpoint_exists")),
        "sd15_checkpoint_required": bool(input_resolution.get("sd15_checkpoint_required")),
        "sd15_checkpoint_path": str(input_resolution.get("sd15_checkpoint_path") or ""),
        "new_source_root_count": _safe_int(input_resolution.get("new_source_root_count")),
        "new_source_root_required": bool(input_resolution.get("new_source_root_required")),
        "source_or_cache_axis_required": bool(input_resolution.get("source_or_cache_axis_required")),
        "warm_cache_or_caption_repair_required": bool(
            input_resolution.get("warm_cache_or_caption_repair_required")
        ),
        "anima_source_or_cache_axis_required": bool(
            input_resolution.get(
                "anima_source_or_cache_axis_required",
                "anima_source_or_cache_axis" in set(missing_inputs),
            )
        ),
        "external_input_detected": bool(input_resolution.get("external_input_detected")),
        "json_replay_ready": bool(input_resolution.get("json_replay_ready")),
        "preflight_admitted": bool(input_resolution.get("preflight_admitted")),
        "manual_canary_plan_ready": bool(input_resolution.get("manual_canary_plan_ready")),
        "handoff_step_ids": _strings(input_resolution.get("handoff_step_ids"))[:20],
        "next_json_refresh_sequence": _strings(input_resolution.get("next_json_refresh_sequence"))[:50],
        "next_manual_gpu_gate": str(input_resolution.get("next_manual_gpu_gate") or ""),
        "release_gate_blockers": _strings(input_resolution.get("release_gate_blockers"))[:20],
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }
    terminal_manual_evidence_blocking_summary = {
        "summary_version": _safe_int(manual_evidence_blocking.get("summary_version"), 1),
        "roadmap": str(manual_evidence_blocking.get("roadmap") or ROADMAP),
        "manual_gpu_evidence_ready": bool(manual_evidence_blocking.get("manual_gpu_evidence_ready")),
        "manual_gpu_evidence_required": bool(manual_evidence_blocking.get("manual_gpu_evidence_required")),
        "source_cache_axis_manual_canary_plan_ready": bool(
            manual_evidence_blocking.get("source_cache_axis_manual_canary_plan_ready")
        ),
        "source_cache_axis_manual_canary_plan_required": bool(
            manual_evidence_blocking.get("source_cache_axis_manual_canary_plan_required")
        ),
        "sd15_checkpoint_required": bool(manual_evidence_blocking.get("sd15_checkpoint_required")),
        "natural_load_canary_pending": bool(manual_evidence_blocking.get("natural_load_canary_pending")),
        "release_claims_rebuild_required": bool(
            manual_evidence_blocking.get("release_claims_rebuild_required")
        ),
        "release_gate_blockers": _strings(manual_evidence_blocking.get("release_gate_blockers"))[:20],
        "next_required_inputs": _strings(manual_evidence_blocking.get("next_required_inputs"))[:20],
        "next_json_rebuild_stage_id": str(manual_evidence_blocking.get("next_json_rebuild_stage_id") or ""),
        "blocked_actions": _strings(manual_evidence_blocking.get("blocked_actions"))[:20],
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }
    terminal_sd15_manual_ab_contract_summary = _sd15_manual_ab_contract_summary(sd15)
    natural_load_gate = _mapping(unblocker.get("natural_load_gate_summary"))
    terminal_natural_load_gate_summary = {
        "summary_version": _safe_int(natural_load_gate.get("summary_version"), 1),
        "roadmap": str(natural_load_gate.get("roadmap") or ROADMAP),
        "gate_id": str(natural_load_gate.get("gate_id") or "natural_load_canary_pending"),
        "gate_blocked": bool(natural_load_gate.get("gate_blocked")),
        "blocked_families": _strings(natural_load_gate.get("blocked_families"))[:20],
        "blocked_family_count": _safe_int(
            natural_load_gate.get("blocked_family_count"),
            len(_strings(natural_load_gate.get("blocked_families"))),
        ),
        "blocked_family_statuses": dict(_mapping(natural_load_gate.get("blocked_family_statuses"))),
        "missing_families": _strings(natural_load_gate.get("missing_families"))[:20],
        "blocker_summary": dict(_mapping(natural_load_gate.get("blocker_summary"))),
        "reason": str(natural_load_gate.get("reason") or ""),
        "requires_new_source_or_cache_axis": bool(natural_load_gate.get("requires_new_source_or_cache_axis")),
        "requires_warm_cache_or_caption_repair": bool(
            natural_load_gate.get("requires_warm_cache_or_caption_repair")
        ),
        "manual_canary_plan_ready": bool(natural_load_gate.get("manual_canary_plan_ready")),
        "preflight_admitted": bool(natural_load_gate.get("preflight_admitted")),
        "source_cache_preflight_status": str(
            natural_load_gate.get("source_cache_preflight_status") or ""
        ),
        "source_cache_preflight_blockers": _strings(
            natural_load_gate.get("source_cache_preflight_blockers")
        )[:20],
        "source_cache_preflight_blocker_count": _safe_int(
            natural_load_gate.get("source_cache_preflight_blocker_count")
        ),
        "source_cache_candidate_source": str(
            natural_load_gate.get("source_cache_candidate_source") or ""
        ),
        "source_cache_candidate_family": str(
            natural_load_gate.get("source_cache_candidate_family") or ""
        ),
        "source_cache_candidate_root": str(
            natural_load_gate.get("source_cache_candidate_root") or ""
        ),
        "source_cache_candidate_sample_offset": natural_load_gate.get(
            "source_cache_candidate_sample_offset"
        ),
        "source_cache_matched_axis_found": bool(
            natural_load_gate.get("source_cache_matched_axis_found")
        ),
        "source_cache_matched_axis_cache_ready": bool(
            natural_load_gate.get("source_cache_matched_axis_cache_ready")
        ),
        "source_cache_matched_axis_quality_ok": bool(
            natural_load_gate.get("source_cache_matched_axis_quality_ok")
        ),
        "source_cache_matched_axis_candidate_rank_score": _safe_number(
            natural_load_gate.get("source_cache_matched_axis_candidate_rank_score")
        ),
        "source_cache_matched_axis_caption_sample_coverage": _safe_number(
            natural_load_gate.get("source_cache_matched_axis_caption_sample_coverage")
        ),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }
    terminal_external_input_transition_table = {
        "summary_version": _safe_int(external_input_transition.get("summary_version"), 1),
        "roadmap": str(external_input_transition.get("roadmap") or ""),
        "artifact_role": str(external_input_transition.get("artifact_role") or ""),
        "transition_status": str(external_input_transition.get("transition_status") or ""),
        "external_input_required": bool(external_input_transition.get("external_input_required")),
        "missing_external_inputs": _strings(external_input_transition.get("missing_external_inputs"))[:20],
        "row_count": _safe_int(external_input_transition.get("row_count")),
        "required_row_count": _safe_int(external_input_transition.get("required_row_count")),
        "blocked_row_count": _safe_int(external_input_transition.get("blocked_row_count")),
        "pending_not_missing_input_count": _safe_int(
            external_input_transition.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            external_input_transition.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            external_input_transition.get("pending_not_missing_reason_ids")
        )[:50],
        "detected_row_count": _safe_int(external_input_transition.get("detected_row_count")),
        "detected_unaccepted_input_count": _safe_int(
            external_input_transition.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            external_input_transition.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            external_input_transition.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            external_input_transition.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            external_input_transition.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "admitted_row_count": _safe_int(external_input_transition.get("admitted_row_count")),
        "manual_plan_ready_row_count": _safe_int(
            external_input_transition.get("manual_plan_ready_row_count")
        ),
        "unsafe_row_count": _safe_int(external_input_transition.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(external_input_transition.get("unsafe_row_ids"))[:20],
        "transition_state_counts": dict(_mapping(external_input_transition.get("transition_state_counts"))),
        "next_json_refresh_sequence": _strings(external_input_transition.get("next_json_refresh_sequence"))[:50],
        "replay_command_count": _safe_int(external_input_transition.get("replay_command_count")),
        "replay_ready_command_count": _safe_int(external_input_transition.get("replay_ready_command_count")),
        "handoff_step_ids": _strings(external_input_transition.get("handoff_step_ids"))[:20],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "required": bool(row.get("required")),
                "external_input_missing": bool(row.get("external_input_missing")),
                "pending_not_missing": bool(row.get("pending_not_missing")),
                "pending_not_missing_reason_ids": _strings(
                    row.get("pending_not_missing_reason_ids")
                )[:20],
                "detected": bool(row.get("detected")),
                "detected_count": _safe_int(row.get("detected_count")),
                "admitted": bool(row.get("admitted")),
                "manual_plan_ready": bool(row.get("manual_plan_ready")),
                "transition_state": str(row.get("transition_state") or ""),
                "handoff_step_id": str(row.get("handoff_step_id") or ""),
                "handoff_step_present": bool(row.get("handoff_step_present")),
                "admission_stage": str(row.get("admission_stage") or ""),
                "blocker_ids": _strings(row.get("blocker_ids"))[:10],
                "upstream_artifact_ids": _strings(row.get("upstream_artifact_ids"))[:10],
                "blocked_criteria_ids": _strings(row.get("blocked_criteria_ids"))[:20],
                "blocked_reason_ids": _strings(row.get("blocked_reason_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                "replay_command_ids": _strings(row.get("replay_command_ids"))[:20],
                "next_manual_gpu_gate": str(row.get("next_manual_gpu_gate") or ""),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in transition_rows
        ],
        "fail_closed": bool(external_input_transition.get("fail_closed")),
        "safe_to_auto_start": bool(external_input_transition.get("safe_to_auto_start")),
        "release_claim_allowed": bool(external_input_transition.get("release_claim_allowed")),
        "not_release_evidence": bool(external_input_transition.get("not_release_evidence")),
    }
    terminal_next_action_machine_summary = {
        "summary_version": _safe_int(next_action_machine.get("summary_version"), 1),
        "roadmap": str(next_action_machine.get("roadmap") or ""),
        "artifact_role": str(next_action_machine.get("artifact_role") or ""),
        "unique_action_count": _safe_int(next_action_machine.get("unique_action_count")),
        "deduped_duplicate_action_count": _safe_int(
            next_action_machine.get("deduped_duplicate_action_count")
        ),
        "readiness_state_counts": dict(_mapping(next_action_machine.get("readiness_state_counts"))),
        "readiness_blocker_kind_counts": dict(
            _mapping(next_action_machine.get("readiness_blocker_kind_counts"))
        ),
        "json_ready_action_count": _safe_int(next_action_machine.get("json_ready_action_count")),
        "json_closed_action_count": _safe_int(next_action_machine.get("json_closed_action_count")),
        "unsafe_action_count": _safe_int(next_action_machine.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(next_action_machine.get("unsafe_action_ids"))[:50],
        "missing_machine_field_action_count": _safe_int(
            next_action_machine.get("missing_machine_field_action_count")
        ),
        "missing_machine_field_action_ids": _strings(
            next_action_machine.get("missing_machine_field_action_ids")
        )[:50],
        "fail_closed": bool(next_action_machine.get("fail_closed")),
        "safe_to_auto_start": bool(next_action_machine.get("safe_to_auto_start")),
        "release_claim_allowed": bool(next_action_machine.get("release_claim_allowed")),
        "not_release_evidence": bool(next_action_machine.get("not_release_evidence")),
    }
    terminal_next_action_contract_summary = {
        "summary_version": _safe_int(next_action_contract.get("summary_version"), 1),
        "expected_roadmap": str(next_action_contract.get("expected_roadmap") or ""),
        "artifact_role": "gpu_bubble_next_action_contract_summary",
        "action_count": _safe_int(next_action_contract.get("action_count")),
        "contract_complete_action_count": _safe_int(
            next_action_contract.get("contract_complete_action_count")
        ),
        "missing_contract_action_count": _safe_int(
            next_action_contract.get("missing_contract_action_count")
        ),
        "missing_contract_action_ids": _strings(
            next_action_contract.get("missing_contract_action_ids")
        )[:50],
        "missing_contract_fields_by_action": [
            {
                "id": str(_mapping(item).get("id") or ""),
                "missing_keys": _strings(_mapping(item).get("missing_keys"))[:20],
            }
            for item in _list(next_action_contract.get("missing_contract_fields_by_action"))[:50]
        ],
        "release_or_auto_start_unsafe_action_count": _safe_int(
            next_action_contract.get("release_or_auto_start_unsafe_action_count")
        ),
        "release_or_auto_start_unsafe_action_ids": _strings(
            next_action_contract.get("release_or_auto_start_unsafe_action_ids")
        )[:50],
        "contract_ok": bool(next_action_contract.get("contract_ok")),
        "fail_closed": next_action_contract_fail_closed,
        "safe_to_auto_start": bool(next_action_contract.get("safe_to_auto_start")),
        "release_claim_allowed": bool(next_action_contract.get("release_claim_allowed")),
        "not_release_evidence": bool(next_action_contract.get("not_release_evidence")),
    }
    terminal_remaining_work_summary = {
        "summary_version": _safe_int(remaining.get("summary_version"), 1),
        "roadmap": str(remaining.get("roadmap") or ""),
        "artifact_role": str(remaining.get("artifact_role") or ""),
        "total_action_count": _safe_int(remaining.get("total_action_count")),
        "stable_first_release_blocked_by_this_artifact": bool(
            remaining.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_blocked": bool(
            remaining.get("gpu_bubble_release_claim_blocked")
        ),
        "gpu_bubble_release_hard_gate_count": _safe_int(
            remaining.get("gpu_bubble_release_hard_gate_count")
        ),
        "gpu_bubble_release_hard_gate_ids": _strings(
            remaining.get("gpu_bubble_release_hard_gate_ids")
        )[:20],
        "json_ready_action_count": _safe_int(remaining.get("json_ready_action_count")),
        "json_ready_action_ids": _strings(remaining.get("json_ready_action_ids"))[:50],
        "json_closed_action_count": _safe_int(remaining.get("json_closed_action_count")),
        "json_closed_action_ids": _strings(remaining.get("json_closed_action_ids"))[:50],
        "external_input_action_count": _safe_int(remaining.get("external_input_action_count")),
        "external_input_action_ids": _strings(remaining.get("external_input_action_ids"))[:50],
        "missing_prerequisite_action_count": _safe_int(
            remaining.get("missing_prerequisite_action_count")
        ),
        "missing_prerequisite_action_ids": _strings(
            remaining.get("missing_prerequisite_action_ids")
        )[:50],
        "manual_gpu_evidence_action_count": _safe_int(
            remaining.get("manual_gpu_evidence_action_count")
        ),
        "manual_gpu_evidence_action_ids": _strings(
            remaining.get("manual_gpu_evidence_action_ids")
        )[:50],
        "protected_manual_gpu_ready_action_count": _safe_int(
            remaining.get("protected_manual_gpu_ready_action_count")
        ),
        "protected_manual_gpu_ready_action_ids": _strings(
            remaining.get("protected_manual_gpu_ready_action_ids")
        )[:50],
        "followup_gpu_required_action_count": _safe_int(
            remaining.get("followup_gpu_required_action_count")
        ),
        "followup_gpu_required_action_ids": _strings(
            remaining.get("followup_gpu_required_action_ids")
        )[:50],
        "current_gpu_heavy_action_count": _safe_int(
            remaining.get("current_gpu_heavy_action_count")
        ),
        "current_gpu_heavy_action_ids": _strings(
            remaining.get("current_gpu_heavy_action_ids")
        )[:50],
        "cache_axis_not_ready_action_count": _safe_int(
            remaining.get("cache_axis_not_ready_action_count")
        ),
        "cache_axis_not_ready_action_ids": _strings(
            remaining.get("cache_axis_not_ready_action_ids")
        )[:50],
        "duplicate_or_stale_axis_action_count": _safe_int(
            remaining.get("duplicate_or_stale_axis_action_count")
        ),
        "duplicate_or_stale_axis_action_ids": _strings(
            remaining.get("duplicate_or_stale_axis_action_ids")
        )[:50],
        "release_gate_related_action_count": _safe_int(
            remaining.get("release_gate_related_action_count")
        ),
        "release_gate_related_action_ids": _strings(
            remaining.get("release_gate_related_action_ids")
        )[:50],
        "external_input_lifecycle_status": str(
            remaining.get("external_input_lifecycle_status") or ""
        ),
        "detected_input_count": _safe_int(remaining.get("detected_input_count")),
        "detected_input_ids": _strings(remaining.get("detected_input_ids"))[:50],
        "accepted_input_count": _safe_int(remaining.get("accepted_input_count")),
        "accepted_input_ids": _strings(remaining.get("accepted_input_ids"))[:50],
        "still_missing_input_count": _safe_int(
            remaining.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            remaining.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            remaining.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            remaining.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            remaining.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            remaining.get("detected_but_not_accepted_input_count")
        ),
        "detected_but_not_accepted_input_ids": _strings(
            remaining.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            remaining.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            remaining.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            remaining.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            remaining.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            remaining.get("pending_not_missing_reason_ids")
        )[:50],
        "pending_input_count": _safe_int(remaining.get("pending_input_count")),
        "pending_input_ids": _strings(remaining.get("pending_input_ids"))[:50],
        "recommended_external_input_focus": str(
            remaining.get("recommended_external_input_focus") or ""
        ),
        "recommended_release_policy": str(remaining.get("recommended_release_policy") or ""),
        "recommended_next_non_gpu_focus": str(
            remaining.get("recommended_next_non_gpu_focus") or ""
        ),
        "unsafe_action_count": _safe_int(remaining.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(remaining.get("unsafe_action_ids"))[:50],
        "fail_closed": remaining_work_fail_closed,
        "safe_to_auto_start": bool(remaining.get("safe_to_auto_start")),
        "release_claim_allowed": bool(remaining.get("release_claim_allowed")),
        "not_release_evidence": bool(remaining.get("not_release_evidence")),
    }
    terminal_first_release_policy_summary = {
        "summary_version": _safe_int(first_release_policy.get("summary_version"), 1),
        "roadmap": str(first_release_policy.get("roadmap") or ""),
        "artifact_role": str(first_release_policy.get("artifact_role") or ""),
        "scope": str(first_release_policy.get("scope") or ""),
        "stable_first_release_blocked_by_this_artifact": bool(
            first_release_policy.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_allowed": bool(
            first_release_policy.get("gpu_bubble_release_claim_allowed")
        ),
        "gpu_bubble_release_claim_blocked": bool(
            first_release_policy.get("gpu_bubble_release_claim_blocked")
        ),
        "gpu_bubble_release_hard_gate_count": _safe_int(
            first_release_policy.get("gpu_bubble_release_hard_gate_count")
        ),
        "gpu_bubble_release_hard_gate_ids": _strings(
            first_release_policy.get("gpu_bubble_release_hard_gate_ids")
        )[:20],
        "recommended_release_policy": str(
            first_release_policy.get("recommended_release_policy") or ""
        ),
        "claim_publication_scope": str(first_release_policy.get("claim_publication_scope") or ""),
        "does_not_prove_global_product_release": bool(
            first_release_policy.get("does_not_prove_global_product_release")
        ),
        "unsafe_policy_count": _safe_int(first_release_policy.get("unsafe_policy_count")),
        "unsafe_policy_ids": _strings(first_release_policy.get("unsafe_policy_ids"))[:20],
        "fail_closed": first_release_policy_fail_closed,
        "not_release_evidence": bool(first_release_policy.get("not_release_evidence")),
        "safe_to_auto_start": bool(first_release_policy.get("safe_to_auto_start")),
        "release_claim_allowed": bool(first_release_policy.get("release_claim_allowed")),
    }
    terminal_source_axis_freshness_summary = {
        "summary_version": _safe_int(source_axis_freshness_summary.get("summary_version"), 1),
        "roadmap": str(source_axis_freshness_summary.get("roadmap") or ""),
        "artifact_role": str(source_axis_freshness_summary.get("artifact_role") or ""),
        "report": str(source_axis_freshness_summary.get("report") or ""),
        "status": str(source_axis_freshness_summary.get("status") or ""),
        "axis_state": str(source_axis_freshness_summary.get("axis_state") or ""),
        "external_input_detected": bool(
            source_axis_freshness_summary.get("external_input_detected")
        ),
        "new_source_root_count": _safe_int(
            source_axis_freshness_summary.get("new_source_root_count")
        ),
        "new_source_roots": _strings(source_axis_freshness_summary.get("new_source_roots"))[:20],
        "completed_axis_count": _safe_int(
            source_axis_freshness_summary.get("completed_axis_count")
        ),
        "completed_out_dir_count": _safe_int(
            source_axis_freshness_summary.get("completed_out_dir_count")
        ),
        "candidate_status": str(source_axis_freshness_summary.get("candidate_status") or ""),
        "candidate_fresh": bool(source_axis_freshness_summary.get("candidate_fresh")),
        "candidate_duplicate_or_stale": bool(
            source_axis_freshness_summary.get("candidate_duplicate_or_stale")
        ),
        "matching_axis_count": _safe_int(
            source_axis_freshness_summary.get("matching_axis_count")
        ),
        "preflight_admitted": bool(source_axis_freshness_summary.get("preflight_admitted")),
        "manual_canary_plan_ready": bool(
            source_axis_freshness_summary.get("manual_canary_plan_ready")
        ),
        "publishable": bool(source_axis_freshness_summary.get("publishable")),
        "blocker_count": _safe_int(source_axis_freshness_summary.get("blocker_count")),
        "blockers": _strings(source_axis_freshness_summary.get("blockers"))[:20],
        "acceptance_gates": _strings(source_axis_freshness_summary.get("acceptance_gates"))[:20],
        "blocked_actions": _strings(source_axis_freshness_summary.get("blocked_actions"))[:20],
        "unsafe_audit_count": _safe_int(
            source_axis_freshness_summary.get("unsafe_audit_count")
        ),
        "unsafe_audit_ids": _strings(source_axis_freshness_summary.get("unsafe_audit_ids"))[:20],
        "fail_closed": source_axis_freshness_fail_closed,
        "not_release_evidence": bool(source_axis_freshness_summary.get("not_release_evidence")),
        "does_not_run_training": bool(
            source_axis_freshness_summary.get("does_not_run_training")
        ),
        "does_not_run_cuda": bool(source_axis_freshness_summary.get("does_not_run_cuda")),
        "safe_to_auto_start": bool(source_axis_freshness_summary.get("safe_to_auto_start")),
        "release_claim_allowed": bool(
            source_axis_freshness_summary.get("release_claim_allowed")
        ),
    }
    terminal_manual_review_queue_summary = {
        "summary_version": _safe_int(manual_review_queue.get("summary_version"), 1),
        "roadmap": str(manual_review_queue.get("roadmap") or ""),
        "artifact_role": str(manual_review_queue.get("artifact_role") or ""),
        "manual_review_ready_count": _safe_int(
            manual_review_queue.get("manual_review_ready_count")
        ),
        "closed_blocked_or_regression_count": _safe_int(
            manual_review_queue.get("closed_blocked_or_regression_count")
        ),
        "closed_diagnostic_or_promotion_count": _safe_int(
            manual_review_queue.get("closed_diagnostic_or_promotion_count")
        ),
        "review_only_action_count": _safe_int(
            manual_review_queue.get("review_only_action_count")
        ),
        "followup_gpu_action_count": _safe_int(
            manual_review_queue.get("followup_gpu_action_count")
        ),
        "current_gpu_heavy_action_count": _safe_int(
            manual_review_queue.get("current_gpu_heavy_action_count")
        ),
        "closed_blocked_or_regression_action_ids": _strings(
            manual_review_queue.get("closed_blocked_or_regression_action_ids")
        )[:50],
        "closed_diagnostic_or_promotion_action_ids": _strings(
            manual_review_queue.get("closed_diagnostic_or_promotion_action_ids")
        )[:50],
        "review_only_action_ids": _strings(
            manual_review_queue.get("review_only_action_ids")
        )[:50],
        "followup_gpu_action_ids": _strings(
            manual_review_queue.get("followup_gpu_action_ids")
        )[:50],
        "current_gpu_heavy_action_ids": _strings(
            manual_review_queue.get("current_gpu_heavy_action_ids")
        )[:50],
        "review_outcome_counts": dict(
            _mapping(manual_review_queue.get("review_outcome_counts"))
        ),
        "review_outcome_action_ids": {
            str(kind): _strings(ids)[:50]
            for kind, ids in _mapping(
                manual_review_queue.get("review_outcome_action_ids")
            ).items()
        },
        "unsafe_action_count": _safe_int(manual_review_queue.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(manual_review_queue.get("unsafe_action_ids"))[:50],
        "blocked_actions": _strings(manual_review_queue.get("blocked_actions"))[:50],
        "fail_closed": manual_review_queue_fail_closed,
        "safe_to_auto_start": bool(manual_review_queue.get("safe_to_auto_start")),
        "release_claim_allowed": bool(manual_review_queue.get("release_claim_allowed")),
        "not_release_evidence": bool(manual_review_queue.get("not_release_evidence")),
    }
    terminal_protected_followup_gpu_queue_summary = {
        "summary_version": _safe_int(protected_followup_gpu_queue.get("summary_version"), 1),
        "roadmap": str(protected_followup_gpu_queue.get("roadmap") or ""),
        "artifact_role": str(protected_followup_gpu_queue.get("artifact_role") or ""),
        "queue_status": str(protected_followup_gpu_queue.get("queue_status") or ""),
        "followup_gpu_required_action_count": _safe_int(
            protected_followup_gpu_queue.get("followup_gpu_required_action_count")
        ),
        "followup_gpu_required_action_ids": _strings(
            protected_followup_gpu_queue.get("followup_gpu_required_action_ids")
        )[:20],
        "row_ids": _strings(protected_followup_gpu_queue.get("row_ids"))[:50],
        "family_counts": dict(_mapping(protected_followup_gpu_queue.get("family_counts"))),
        "readiness_state_counts": dict(
            _mapping(protected_followup_gpu_queue.get("readiness_state_counts"))
        ),
        "readiness_blocker_kind_counts": dict(
            _mapping(protected_followup_gpu_queue.get("readiness_blocker_kind_counts"))
        ),
        "current_action_gpu_count": _safe_int(
            protected_followup_gpu_queue.get("current_action_gpu_count")
        ),
        "current_action_manual_start_count": _safe_int(
            protected_followup_gpu_queue.get("current_action_manual_start_count")
        ),
        "followup_manual_start_required_count": _safe_int(
            protected_followup_gpu_queue.get("followup_manual_start_required_count")
        ),
        "requires_external_input_count": _safe_int(
            protected_followup_gpu_queue.get("requires_external_input_count")
        ),
        "unsafe_action_count": _safe_int(protected_followup_gpu_queue.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(protected_followup_gpu_queue.get("unsafe_action_ids"))[:50],
        "rows": [
            {
                "id": str(row.get("id") or ""),
                "family": str(row.get("family") or ""),
                "readiness_state": str(row.get("readiness_state") or ""),
                "readiness_blocker_kind": str(row.get("readiness_blocker_kind") or ""),
                "current_action_requires_gpu": bool(row.get("current_action_requires_gpu")),
                "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
                "followup_manual_start_required": bool(row.get("followup_manual_start_required")),
                "current_action_manual_start_required": bool(
                    row.get("current_action_manual_start_required")
                ),
                "requires_external_input": bool(row.get("requires_external_input")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (_mapping(item) for item in _list(protected_followup_gpu_queue.get("rows")))
        ],
        "execution_policy": str(protected_followup_gpu_queue.get("execution_policy") or ""),
        "fail_closed": bool(protected_followup_gpu_queue.get("fail_closed")),
        "safe_to_auto_start": bool(protected_followup_gpu_queue.get("safe_to_auto_start")),
        "release_claim_allowed": bool(protected_followup_gpu_queue.get("release_claim_allowed")),
        "not_release_evidence": bool(protected_followup_gpu_queue.get("not_release_evidence")),
    }
    terminal_source_downstream_contract_summary = {
        "summary_version": _safe_int(source_downstream_contract.get("summary_version"), 1),
        "roadmap": str(source_downstream_contract.get("roadmap") or ""),
        "artifact_role": str(source_downstream_contract.get("artifact_role") or ""),
        "source_path_count": _safe_int(source_downstream_contract.get("source_path_count")),
        "source_path_missing_count": _safe_int(source_downstream_contract.get("source_path_missing_count")),
        "source_path_missing_paths": _strings(
            source_downstream_contract.get("source_path_missing_paths")
        )[:20],
        "source_path_load_error_count": _safe_int(
            source_downstream_contract.get("source_path_load_error_count")
        ),
        "source_path_load_error_paths": _strings(
            source_downstream_contract.get("source_path_load_error_paths")
        )[:20],
        "source_path_roadmap_mismatch_count": _safe_int(
            source_downstream_contract.get("source_path_roadmap_mismatch_count")
        ),
        "source_path_roadmap_mismatch_paths": _strings(
            source_downstream_contract.get("source_path_roadmap_mismatch_paths")
        )[:20],
        "source_path_release_unsafe_count": _safe_int(
            source_downstream_contract.get("source_path_release_unsafe_count")
        ),
        "source_path_release_unsafe_paths": _strings(
            source_downstream_contract.get("source_path_release_unsafe_paths")
        )[:20],
        "downstream_artifact_count": _safe_int(
            source_downstream_contract.get("downstream_artifact_count")
        ),
        "downstream_artifact_missing_count": _safe_int(
            source_downstream_contract.get("downstream_artifact_missing_count")
        ),
        "downstream_artifact_missing_ids": _strings(
            source_downstream_contract.get("downstream_artifact_missing_ids")
        )[:20],
        "downstream_source_path_dependency_count": _safe_int(
            source_downstream_contract.get("downstream_source_path_dependency_count")
        ),
        "downstream_source_path_dependency_ids": _strings(
            source_downstream_contract.get("downstream_source_path_dependency_ids")
        )[:20],
        "downstream_release_unsafe_count": _safe_int(
            source_downstream_contract.get("downstream_release_unsafe_count")
        ),
        "downstream_release_unsafe_ids": _strings(
            source_downstream_contract.get("downstream_release_unsafe_ids")
        )[:20],
        "downstream_in_source_path_count": _safe_int(
            source_downstream_contract.get("downstream_in_source_path_count")
        ),
        "downstream_in_source_paths": _strings(
            source_downstream_contract.get("downstream_in_source_paths")
        )[:20],
        "unsafe_artifact_count": _safe_int(source_downstream_contract.get("unsafe_artifact_count")),
        "unsafe_artifact_ids": _strings(source_downstream_contract.get("unsafe_artifact_ids"))[:50],
        "fail_closed": bool(source_downstream_contract.get("fail_closed")),
        "safe_to_auto_start": bool(source_downstream_contract.get("safe_to_auto_start")),
        "release_claim_allowed": bool(source_downstream_contract.get("release_claim_allowed")),
        "not_release_evidence": bool(source_downstream_contract.get("not_release_evidence")),
    }
    terminal_blocker_matrix_summary = {
        "summary_version": _safe_int(blocker_matrix.get("summary_version"), 1),
        "roadmap": str(blocker_matrix.get("roadmap") or ""),
        "artifact_role": str(blocker_matrix.get("artifact_role") or ""),
        "matrix_status": str(blocker_matrix.get("matrix_status") or ""),
        "total_unclosed_action_count": _safe_int(blocker_matrix.get("total_unclosed_action_count")),
        "unclosed_action_ids": _strings(blocker_matrix.get("unclosed_action_ids"))[:50],
        "json_ready_action_count": _safe_int(blocker_matrix.get("json_ready_action_count")),
        "readiness_state_counts": dict(_mapping(blocker_matrix.get("readiness_state_counts"))),
        "blocked_by_kind_counts": dict(_mapping(blocker_matrix.get("blocked_by_kind_counts"))),
        "family_count": _safe_int(blocker_matrix.get("family_count")),
        "family_ids": _strings(blocker_matrix.get("family_ids"))[:20],
        "family_action_counts": dict(_mapping(blocker_matrix.get("family_action_counts"))),
        "family_external_input_required_counts": dict(
            _mapping(blocker_matrix.get("family_external_input_required_counts"))
        ),
        "family_manual_gpu_required_counts": dict(
            _mapping(blocker_matrix.get("family_manual_gpu_required_counts"))
        ),
        "family_protected_followup_gpu_required_counts": dict(
            _mapping(blocker_matrix.get("family_protected_followup_gpu_required_counts"))
        ),
        "family_source_cache_blocked_counts": dict(
            _mapping(blocker_matrix.get("family_source_cache_blocked_counts"))
        ),
        "unsafe_family_count": _safe_int(blocker_matrix.get("unsafe_family_count")),
        "family_unsafe_action_counts": dict(_mapping(blocker_matrix.get("family_unsafe_action_counts"))),
        "release_hard_gate_ids": _strings(blocker_matrix.get("release_hard_gate_ids"))[:20],
        "external_input_required_action_count": _safe_int(
            blocker_matrix.get("external_input_required_action_count")
        ),
        "external_input_required_action_ids": _strings(
            blocker_matrix.get("external_input_required_action_ids")
        )[:50],
        "missing_external_inputs": _strings(blocker_matrix.get("missing_external_inputs"))[:20],
        "manual_gpu_required_action_count": _safe_int(
            blocker_matrix.get("manual_gpu_required_action_count")
        ),
        "manual_gpu_required_action_ids": _strings(blocker_matrix.get("manual_gpu_required_action_ids"))[:50],
        "protected_followup_gpu_required_action_count": _safe_int(
            blocker_matrix.get("protected_followup_gpu_required_action_count")
        ),
        "protected_followup_gpu_required_action_ids": _strings(
            blocker_matrix.get("protected_followup_gpu_required_action_ids")
        )[:50],
        "source_cache_blocked_action_count": _safe_int(
            blocker_matrix.get("source_cache_blocked_action_count")
        ),
        "source_cache_blocked_action_ids": _strings(
            blocker_matrix.get("source_cache_blocked_action_ids")
        )[:50],
        "sd15_checkpoint_action_count": _safe_int(blocker_matrix.get("sd15_checkpoint_action_count")),
        "sd15_checkpoint_action_ids": _strings(blocker_matrix.get("sd15_checkpoint_action_ids"))[:50],
        "duplicate_or_stale_source_axis_action_count": _safe_int(
            blocker_matrix.get("duplicate_or_stale_source_axis_action_count")
        ),
        "duplicate_or_stale_source_axis_action_ids": _strings(
            blocker_matrix.get("duplicate_or_stale_source_axis_action_ids")
        )[:50],
        "next_unlock_inputs": _strings(blocker_matrix.get("next_unlock_inputs"))[:50],
        "unsafe_action_count": _safe_int(blocker_matrix.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(blocker_matrix.get("unsafe_action_ids"))[:50],
        "execution_policy": str(blocker_matrix.get("execution_policy") or ""),
        "fail_closed": bool(blocker_matrix.get("fail_closed")),
        "safe_to_auto_start": bool(blocker_matrix.get("safe_to_auto_start")),
        "release_claim_allowed": bool(blocker_matrix.get("release_claim_allowed")),
        "not_release_evidence": bool(blocker_matrix.get("not_release_evidence")),
    }
    terminal_blocker_handoff_summary = {
        "summary_version": _safe_int(blocker_handoff.get("summary_version"), 1),
        "roadmap": str(blocker_handoff.get("roadmap") or ""),
        "artifact_role": str(blocker_handoff.get("artifact_role") or ""),
        "handoff_status": str(blocker_handoff.get("handoff_status") or ""),
        "row_count": _safe_int(blocker_handoff.get("row_count")),
        "row_ids": _strings(blocker_handoff.get("row_ids"))[:50],
        "resolution_contract_version": _safe_int(blocker_handoff.get("resolution_contract_version"), 1),
        "resolution_contract_ok": bool(blocker_handoff.get("resolution_contract_ok")),
        "resolution_contract_bad_count": _safe_int(
            blocker_handoff.get("resolution_contract_bad_count")
        ),
        "resolution_contract_bad_ids": _strings(
            blocker_handoff.get("resolution_contract_bad_ids")
        )[:50],
        "resolution_bucket_counts": dict(_mapping(blocker_handoff.get("resolution_bucket_counts"))),
        "json_only_resolution_available_count": _safe_int(
            blocker_handoff.get("json_only_resolution_available_count")
        ),
        "external_input_required_count": _safe_int(
            blocker_handoff.get("external_input_required_count")
        ),
        "manual_gpu_required_count": _safe_int(blocker_handoff.get("manual_gpu_required_count")),
        "protected_runner_required_count": _safe_int(
            blocker_handoff.get("protected_runner_required_count")
        ),
        "release_claim_after_resolution_allowed": bool(
            blocker_handoff.get("release_claim_after_resolution_allowed")
        ),
        "blocker_bucket_counts": dict(_mapping(blocker_handoff.get("blocker_bucket_counts"))),
        "next_unlock_input_ids": _strings(blocker_handoff.get("next_unlock_input_ids"))[:50],
        "still_missing_input_count": _safe_int(
            blocker_handoff.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            blocker_handoff.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            blocker_handoff.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            blocker_handoff.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            blocker_handoff.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            blocker_handoff.get("detected_but_not_accepted_input_count")
        ),
        "detected_but_not_accepted_input_ids": _strings(
            blocker_handoff.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            blocker_handoff.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            blocker_handoff.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            blocker_handoff.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            blocker_handoff.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            blocker_handoff.get("pending_not_missing_reason_ids")
        )[:50],
        "natural_load_blocked_family_ids": _strings(
            blocker_handoff.get("natural_load_blocked_family_ids")
        )[:20],
        "natural_load_next_family_focus_ids": _strings(
            blocker_handoff.get("natural_load_next_family_focus_ids")
        )[:20],
        "required_refresh_command_ids": _strings(
            blocker_handoff.get("required_refresh_command_ids")
        )[:50],
        "external_input_row_count": _safe_int(blocker_handoff.get("external_input_row_count")),
        "current_gpu_row_count": _safe_int(blocker_handoff.get("current_gpu_row_count")),
        "protected_followup_gpu_row_count": _safe_int(
            blocker_handoff.get("protected_followup_gpu_row_count")
        ),
        "unsafe_row_count": _safe_int(blocker_handoff.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(blocker_handoff.get("unsafe_row_ids"))[:50],
        "rows": [
            {
                "id": str(row.get("id") or ""),
                "family": str(row.get("family") or ""),
                "readiness_state": str(row.get("readiness_state") or ""),
                "readiness_blocker_kind": str(row.get("readiness_blocker_kind") or ""),
                "blocker_bucket": str(row.get("blocker_bucket") or ""),
                "resolution_contract": {
                    "resolution_contract_version": _safe_int(
                        _mapping(row.get("resolution_contract")).get("resolution_contract_version"),
                        1,
                    ),
                    "resolution_kind": str(
                        _mapping(row.get("resolution_contract")).get("resolution_kind") or ""
                    ),
                    "can_resolve_json_only_now": bool(
                        _mapping(row.get("resolution_contract")).get("can_resolve_json_only_now")
                    ),
                    "requires_external_input": bool(
                        _mapping(row.get("resolution_contract")).get("requires_external_input")
                    ),
                    "requires_manual_gpu": bool(
                        _mapping(row.get("resolution_contract")).get("requires_manual_gpu")
                    ),
                    "requires_protected_runner": bool(
                        _mapping(row.get("resolution_contract")).get("requires_protected_runner")
                    ),
                    "required_input_ids": _strings(
                        _mapping(row.get("resolution_contract")).get("required_input_ids")
                    )[:20],
                    "missing_input_ids": _strings(
                        _mapping(row.get("resolution_contract")).get("missing_input_ids")
                    )[:20],
                    "post_unlock_refresh_command_ids": _strings(
                        _mapping(row.get("resolution_contract")).get("post_unlock_refresh_command_ids")
                    )[:50],
                    "post_unlock_required_artifact_ids": _strings(
                        _mapping(row.get("resolution_contract")).get("post_unlock_required_artifact_ids")
                    )[:20],
                    "terminal_guard_required": bool(
                        _mapping(row.get("resolution_contract")).get("terminal_guard_required")
                    ),
                    "release_claim_after_resolution_allowed": bool(
                        _mapping(row.get("resolution_contract")).get(
                            "release_claim_after_resolution_allowed"
                        )
                    ),
                    "safe_to_auto_start_after_resolution": bool(
                        _mapping(row.get("resolution_contract")).get(
                            "safe_to_auto_start_after_resolution"
                        )
                    ),
                    "not_release_evidence": bool(
                        _mapping(row.get("resolution_contract")).get("not_release_evidence")
                    ),
                },
                "next_unlock_input_ids": _strings(row.get("next_unlock_input_ids"))[:20],
                "required_refresh_command_ids": _strings(row.get("required_refresh_command_ids"))[:50],
                "requires_external_input": bool(row.get("requires_external_input")),
                "current_action_requires_gpu": bool(row.get("current_action_requires_gpu")),
                "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (_mapping(item) for item in _list(blocker_handoff.get("rows")))
        ],
        "execution_policy": str(blocker_handoff.get("execution_policy") or ""),
        "fail_closed": bool(blocker_handoff.get("fail_closed")),
        "safe_to_auto_start": bool(blocker_handoff.get("safe_to_auto_start")),
        "release_claim_allowed": bool(blocker_handoff.get("release_claim_allowed")),
        "not_release_evidence": bool(blocker_handoff.get("not_release_evidence")),
    }
    terminal_action_dependency_graph_summary = {
        "summary_version": _safe_int(action_dependency_graph.get("summary_version"), 1),
        "roadmap": str(action_dependency_graph.get("roadmap") or ""),
        "artifact_role": str(action_dependency_graph.get("artifact_role") or ""),
        "graph_status": str(action_dependency_graph.get("graph_status") or ""),
        "action_node_count": _safe_int(action_dependency_graph.get("action_node_count")),
        "action_node_ids": _strings(action_dependency_graph.get("action_node_ids"))[:50],
        "dependency_node_count": _safe_int(action_dependency_graph.get("dependency_node_count")),
        "dependency_node_ids": _strings(action_dependency_graph.get("dependency_node_ids"))[:80],
        "edge_count": _safe_int(action_dependency_graph.get("edge_count")),
        "edge_sample": [
            {
                "from_dependency_id": str(row.get("from_dependency_id") or ""),
                "to_action_id": str(row.get("to_action_id") or ""),
                "dependency_kind": str(row.get("dependency_kind") or ""),
            }
            for row in (_mapping(item) for item in _list(action_dependency_graph.get("edge_sample"))[:80])
        ],
        "action_state_counts": dict(_mapping(action_dependency_graph.get("action_state_counts"))),
        "blocker_kind_counts": dict(_mapping(action_dependency_graph.get("blocker_kind_counts"))),
        "dependency_kind_counts": dict(_mapping(action_dependency_graph.get("dependency_kind_counts"))),
        "missing_external_inputs": _strings(action_dependency_graph.get("missing_external_inputs"))[:20],
        "release_hard_gate_ids": _strings(action_dependency_graph.get("release_hard_gate_ids"))[:20],
        "still_missing_input_count": _safe_int(
            action_dependency_graph.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            action_dependency_graph.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            action_dependency_graph.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            action_dependency_graph.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            action_dependency_graph.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            action_dependency_graph.get("detected_but_not_accepted_input_count")
        ),
        "detected_but_not_accepted_input_ids": _strings(
            action_dependency_graph.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            action_dependency_graph.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            action_dependency_graph.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            action_dependency_graph.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            action_dependency_graph.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            action_dependency_graph.get("pending_not_missing_reason_ids")
        )[:50],
        "required_refresh_command_ids": _strings(
            action_dependency_graph.get("required_refresh_command_ids")
        )[:50],
        "refresh_sequence_terminal_guard_ok": bool(
            action_dependency_graph.get("refresh_sequence_terminal_guard_ok")
        ),
        "rows": [
            {
                "action_id": str(row.get("action_id") or ""),
                "family": str(row.get("family") or ""),
                "readiness_state": str(row.get("readiness_state") or ""),
                "readiness_blocker_kind": str(row.get("readiness_blocker_kind") or ""),
                "primary_blocker": str(row.get("primary_blocker") or ""),
                "dependency_ids": _strings(row.get("dependency_ids"))[:30],
                "dependency_kinds": _strings(row.get("dependency_kinds"))[:20],
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_current_gpu": bool(row.get("requires_current_gpu")),
                "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
                "manual_start_required": bool(row.get("manual_start_required")),
                "expected_output_count": _safe_int(row.get("expected_output_count")),
                "evidence_path_count": _safe_int(row.get("evidence_path_count")),
                "post_unlock_refresh_command_ids": _strings(
                    row.get("post_unlock_refresh_command_ids")
                )[:50],
                "terminal_guard_required": bool(row.get("terminal_guard_required")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (_mapping(item) for item in _list(action_dependency_graph.get("rows")))
        ],
        "unsafe_action_count": _safe_int(action_dependency_graph.get("unsafe_action_count")),
        "unsafe_action_ids": _strings(action_dependency_graph.get("unsafe_action_ids"))[:50],
        "ready_for_release_claim": bool(action_dependency_graph.get("ready_for_release_claim")),
        "execution_policy": str(action_dependency_graph.get("execution_policy") or ""),
        "fail_closed": bool(action_dependency_graph.get("fail_closed")),
        "not_release_evidence": bool(action_dependency_graph.get("not_release_evidence")),
        "safe_to_auto_start": bool(action_dependency_graph.get("safe_to_auto_start")),
        "release_claim_allowed": bool(action_dependency_graph.get("release_claim_allowed")),
    }
    terminal_action_unblock_sequence_summary = {
        "summary_version": _safe_int(action_unblock_sequence.get("summary_version"), 1),
        "roadmap": str(action_unblock_sequence.get("roadmap") or ""),
        "artifact_role": str(action_unblock_sequence.get("artifact_role") or ""),
        "sequence_status": str(action_unblock_sequence.get("sequence_status") or ""),
        "stage_count": _safe_int(action_unblock_sequence.get("stage_count")),
        "stage_ids": _strings(action_unblock_sequence.get("stage_ids"))[:20],
        "current_stage_id": str(action_unblock_sequence.get("current_stage_id") or ""),
        "next_required_input_ids": _strings(action_unblock_sequence.get("next_required_input_ids"))[:50],
        "still_missing_input_count": _safe_int(
            action_unblock_sequence.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            action_unblock_sequence.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            action_unblock_sequence.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            action_unblock_sequence.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            action_unblock_sequence.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            action_unblock_sequence.get("detected_but_not_accepted_input_count")
        ),
        "detected_but_not_accepted_input_ids": _strings(
            action_unblock_sequence.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            action_unblock_sequence.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            action_unblock_sequence.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            action_unblock_sequence.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            action_unblock_sequence.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            action_unblock_sequence.get("pending_not_missing_reason_ids")
        )[:50],
        "natural_load_blocked_family_ids": _strings(
            action_unblock_sequence.get("natural_load_blocked_family_ids")
        )[:20],
        "natural_load_next_family_focus_ids": _strings(
            action_unblock_sequence.get("natural_load_next_family_focus_ids")
        )[:20],
        "manual_gpu_stage_count": _safe_int(action_unblock_sequence.get("manual_gpu_stage_count")),
        "external_input_stage_count": _safe_int(action_unblock_sequence.get("external_input_stage_count")),
        "protected_runner_stage_count": _safe_int(
            action_unblock_sequence.get("protected_runner_stage_count")
        ),
        "release_hard_gate_ids": _strings(action_unblock_sequence.get("release_hard_gate_ids"))[:20],
        "terminal_guard_required": bool(action_unblock_sequence.get("terminal_guard_required")),
        "required_refresh_command_ids": _strings(
            action_unblock_sequence.get("required_refresh_command_ids")
        )[:50],
        "refresh_sequence_terminal_guard_ok": bool(
            action_unblock_sequence.get("refresh_sequence_terminal_guard_ok")
        ),
        "rows": [
            {
                "stage_id": str(row.get("stage_id") or ""),
                "stage_order": _safe_int(row.get("stage_order")),
                "stage_kind": str(row.get("stage_kind") or ""),
                "status": str(row.get("status") or ""),
                "required_input_ids": _strings(row.get("required_input_ids"))[:20],
                "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_protected_runner": bool(row.get("requires_protected_runner")),
                "required_refresh_command_ids": _strings(row.get("required_refresh_command_ids"))[:50],
                "terminal_guard_required_after_stage": bool(
                    row.get("terminal_guard_required_after_stage")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_stage": bool(
                    row.get("release_claim_allowed_after_stage")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (_mapping(item) for item in _list(action_unblock_sequence.get("rows")))
        ],
        "unsafe_stage_count": _safe_int(action_unblock_sequence.get("unsafe_stage_count")),
        "unsafe_stage_ids": _strings(action_unblock_sequence.get("unsafe_stage_ids"))[:20],
        "ready_for_release_claim": bool(action_unblock_sequence.get("ready_for_release_claim")),
        "execution_policy": str(action_unblock_sequence.get("execution_policy") or ""),
        "fail_closed": bool(action_unblock_sequence.get("fail_closed")),
        "not_release_evidence": bool(action_unblock_sequence.get("not_release_evidence")),
        "safe_to_auto_start": bool(action_unblock_sequence.get("safe_to_auto_start")),
        "release_claim_allowed": bool(action_unblock_sequence.get("release_claim_allowed")),
    }
    terminal_blocker_presence_summary = {
        "summary_version": _safe_int(blocker_presence.get("summary_version"), 1),
        "roadmap": str(blocker_presence.get("roadmap") or ""),
        "artifact_role": str(blocker_presence.get("artifact_role") or ""),
        "presence_status": str(blocker_presence.get("presence_status") or ""),
        "row_count": _safe_int(blocker_presence.get("row_count")),
        "row_ids": _strings(blocker_presence.get("row_ids"))[:50],
        "expected_output_action_count": _safe_int(
            blocker_presence.get("expected_output_action_count")
        ),
        "expected_output_missing_action_count": _safe_int(
            blocker_presence.get("expected_output_missing_action_count")
        ),
        "expected_output_missing_action_ids": _strings(
            blocker_presence.get("expected_output_missing_action_ids")
        )[:50],
        "evidence_path_action_count": _safe_int(blocker_presence.get("evidence_path_action_count")),
        "evidence_path_missing_action_count": _safe_int(
            blocker_presence.get("evidence_path_missing_action_count")
        ),
        "evidence_path_missing_action_ids": _strings(
            blocker_presence.get("evidence_path_missing_action_ids")
        )[:50],
        "unsafe_row_count": _safe_int(blocker_presence.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(blocker_presence.get("unsafe_row_ids"))[:50],
        "rows": [
            {
                "id": str(row.get("id") or ""),
                "family": str(row.get("family") or ""),
                "readiness_state": str(row.get("readiness_state") or ""),
                "readiness_blocker_kind": str(row.get("readiness_blocker_kind") or ""),
                "expected_output_count": _safe_int(row.get("expected_output_count")),
                "expected_output_existing_count": _safe_int(row.get("expected_output_existing_count")),
                "expected_output_missing_count": _safe_int(row.get("expected_output_missing_count")),
                "evidence_path_count": _safe_int(row.get("evidence_path_count")),
                "evidence_path_existing_count": _safe_int(row.get("evidence_path_existing_count")),
                "evidence_path_missing_count": _safe_int(row.get("evidence_path_missing_count")),
                "requires_external_input": bool(row.get("requires_external_input")),
                "current_action_requires_gpu": bool(row.get("current_action_requires_gpu")),
                "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (_mapping(item) for item in _list(blocker_presence.get("rows")))
        ],
        "execution_policy": str(blocker_presence.get("execution_policy") or ""),
        "fail_closed": bool(blocker_presence.get("fail_closed")),
        "safe_to_auto_start": bool(blocker_presence.get("safe_to_auto_start")),
        "release_claim_allowed": bool(blocker_presence.get("release_claim_allowed")),
        "not_release_evidence": bool(blocker_presence.get("not_release_evidence")),
    }
    terminal_release_exit_summary = {
        "summary_version": _safe_int(release_exit.get("summary_version"), 1),
        "roadmap": str(release_exit.get("roadmap") or ""),
        "artifact_role": str(release_exit.get("artifact_role") or ""),
        "exit_status": str(release_exit.get("exit_status") or ""),
        "release_hard_gate_count": _safe_int(release_exit.get("release_hard_gate_count")),
        "release_hard_gate_ids": _strings(release_exit.get("release_hard_gate_ids"))[:20],
        "gate_row_count": _safe_int(release_exit.get("gate_row_count")),
        "json_only_exit_available_count": _safe_int(
            release_exit.get("json_only_exit_available_count")
        ),
        "manual_gpu_required_gate_count": _safe_int(
            release_exit.get("manual_gpu_required_gate_count")
        ),
        "protected_runner_required_gate_count": _safe_int(
            release_exit.get("protected_runner_required_gate_count")
        ),
        "missing_declared_output_gate_count": _safe_int(
            release_exit.get("missing_declared_output_gate_count")
        ),
        "unsafe_gate_count": _safe_int(release_exit.get("unsafe_gate_count")),
        "unsafe_gate_ids": _strings(release_exit.get("unsafe_gate_ids"))[:50],
        "still_missing_input_count": _safe_int(
            release_exit.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            release_exit.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            release_exit.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            release_exit.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            release_exit.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            release_exit.get("detected_but_not_accepted_input_count")
        ),
        "detected_but_not_accepted_input_ids": _strings(
            release_exit.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            release_exit.get("detected_unaccepted_blocked_criteria_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            release_exit.get("detected_unaccepted_blocked_reason_ids")
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            release_exit.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            release_exit.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            release_exit.get("pending_not_missing_reason_ids")
        )[:50],
        "rows": [
            {
                "gate_id": str(row.get("gate_id") or ""),
                "gate_status": str(row.get("gate_status") or ""),
                "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                "related_family_ids": _strings(row.get("related_family_ids"))[:20],
                "related_family_counts": {
                    str(key): _safe_int(value)
                    for key, value in _mapping(row.get("related_family_counts")).items()
                },
                "related_readiness_state_counts": {
                    str(key): _safe_int(value)
                    for key, value in _mapping(row.get("related_readiness_state_counts")).items()
                },
                "related_blocker_kind_counts": {
                    str(key): _safe_int(value)
                    for key, value in _mapping(row.get("related_blocker_kind_counts")).items()
                },
                "related_external_input_action_count": _safe_int(
                    row.get("related_external_input_action_count")
                ),
                "related_manual_gpu_action_count": _safe_int(
                    row.get("related_manual_gpu_action_count")
                ),
                "related_protected_followup_action_count": _safe_int(
                    row.get("related_protected_followup_action_count")
                ),
                "required_input_ids": _strings(row.get("required_input_ids"))[:20],
                "required_output_ids": _strings(row.get("required_output_ids"))[:20],
                "missing_declared_output_action_ids": _strings(
                    row.get("missing_declared_output_action_ids")
                )[:50],
                "detected_unaccepted_input_ids": _strings(
                    row.get("detected_unaccepted_input_ids")
                )[:50],
                "detected_unaccepted_next_focus_ids": _strings(
                    row.get("detected_unaccepted_next_focus_ids")
                )[:50],
                "detected_unaccepted_blocked_criteria_ids": _strings(
                    row.get("detected_unaccepted_blocked_criteria_ids")
                )[:50],
                "detected_unaccepted_blocked_reason_ids": _strings(
                    row.get("detected_unaccepted_blocked_reason_ids")
                )[:50],
                "manual_gpu_required": bool(row.get("manual_gpu_required")),
                "protected_runner_required": bool(row.get("protected_runner_required")),
                "json_only_exit_available": bool(row.get("json_only_exit_available")),
                "terminal_guard_required": bool(row.get("terminal_guard_required")),
                "release_claim_allowed_after_exit": bool(
                    row.get("release_claim_allowed_after_exit")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (_mapping(item) for item in _list(release_exit.get("rows")))
        ],
        "execution_policy": str(release_exit.get("execution_policy") or ""),
        "fail_closed": bool(release_exit.get("fail_closed")),
        "safe_to_auto_start": bool(release_exit.get("safe_to_auto_start")),
        "release_claim_allowed": bool(release_exit.get("release_claim_allowed")),
        "not_release_evidence": bool(release_exit.get("not_release_evidence")),
    }
    terminal_release_gate_input_dependency_summary = {
        "summary_version": _safe_int(release_gate_input_dependency.get("summary_version"), 1),
        "roadmap": str(release_gate_input_dependency.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_dependency.get("artifact_role") or ""),
        "dependency_status": str(release_gate_input_dependency.get("dependency_status") or ""),
        "release_hard_gate_count": _safe_int(
            release_gate_input_dependency.get("release_hard_gate_count")
        ),
        "release_hard_gate_ids": _strings(
            release_gate_input_dependency.get("release_hard_gate_ids")
        )[:20],
        "dependency_row_count": _safe_int(
            release_gate_input_dependency.get("dependency_row_count")
        ),
        "required_input_ids": _strings(
            release_gate_input_dependency.get("required_input_ids")
        )[:50],
        "missing_input_count": _safe_int(
            release_gate_input_dependency.get("missing_input_count")
        ),
        "missing_input_ids": _strings(
            release_gate_input_dependency.get("missing_input_ids")
        )[:50],
        "external_input_dependency_count": _safe_int(
            release_gate_input_dependency.get("external_input_dependency_count")
        ),
        "manual_gpu_dependency_count": _safe_int(
            release_gate_input_dependency.get("manual_gpu_dependency_count")
        ),
        "source_cache_refresh_dependency_count": _safe_int(
            release_gate_input_dependency.get("source_cache_refresh_dependency_count")
        ),
        "json_only_resolution_available_count": _safe_int(
            release_gate_input_dependency.get("json_only_resolution_available_count")
        ),
        "unsafe_input_count": _safe_int(
            release_gate_input_dependency.get("unsafe_input_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_dependency.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "dependency_status": str(row.get("dependency_status") or ""),
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "missing": bool(row.get("missing")),
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "json_only_resolution_available": bool(row.get("json_only_resolution_available")),
                "terminal_guard_required_after_input": bool(
                    row.get("terminal_guard_required_after_input")
                ),
                "release_claim_allowed_after_input": bool(
                    row.get("release_claim_allowed_after_input")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item) for item in _list(release_gate_input_dependency.get("rows"))
            )
        ],
        "execution_policy": str(release_gate_input_dependency.get("execution_policy") or ""),
        "fail_closed": bool(release_gate_input_dependency.get("fail_closed")),
        "safe_to_auto_start": bool(release_gate_input_dependency.get("safe_to_auto_start")),
        "release_claim_allowed": bool(release_gate_input_dependency.get("release_claim_allowed")),
        "not_release_evidence": bool(release_gate_input_dependency.get("not_release_evidence")),
    }
    terminal_release_gate_post_input_refresh_plan_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_plan.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_plan.get("roadmap") or ""),
        "artifact_role": str(release_gate_post_input_refresh_plan.get("artifact_role") or ""),
        "plan_status": str(release_gate_post_input_refresh_plan.get("plan_status") or ""),
        "plan_row_count": _safe_int(release_gate_post_input_refresh_plan.get("plan_row_count")),
        "input_ids": _strings(release_gate_post_input_refresh_plan.get("input_ids"))[:50],
        "blocked_input_count": _safe_int(
            release_gate_post_input_refresh_plan.get("blocked_input_count")
        ),
        "blocked_input_ids": _strings(
            release_gate_post_input_refresh_plan.get("blocked_input_ids")
        )[:50],
        "external_input_plan_count": _safe_int(
            release_gate_post_input_refresh_plan.get("external_input_plan_count")
        ),
        "manual_gpu_evidence_plan_count": _safe_int(
            release_gate_post_input_refresh_plan.get("manual_gpu_evidence_plan_count")
        ),
        "source_cache_refresh_plan_count": _safe_int(
            release_gate_post_input_refresh_plan.get("source_cache_refresh_plan_count")
        ),
        "required_refresh_command_ids": _strings(
            release_gate_post_input_refresh_plan.get("required_refresh_command_ids")
        )[:50],
        "terminal_guard_command_ids": _strings(
            release_gate_post_input_refresh_plan.get("terminal_guard_command_ids")
        )[:10],
        "post_refresh_required_artifact_ids": _strings(
            release_gate_post_input_refresh_plan.get("post_refresh_required_artifact_ids")
        )[:20],
        "unsafe_plan_count": _safe_int(
            release_gate_post_input_refresh_plan.get("unsafe_plan_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_post_input_refresh_plan.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "plan_status": str(row.get("plan_status") or ""),
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "input_missing": bool(row.get("input_missing")),
                "external_input_required_before_refresh": bool(
                    row.get("external_input_required_before_refresh")
                ),
                "manual_gpu_evidence_required_before_refresh": bool(
                    row.get("manual_gpu_evidence_required_before_refresh")
                ),
                "source_cache_refresh_input": bool(row.get("source_cache_refresh_input")),
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "terminal_guard_command_ids": _strings(
                    row.get("terminal_guard_command_ids")
                )[:10],
                "post_refresh_required_artifact_ids": _strings(
                    row.get("post_refresh_required_artifact_ids")
                )[:20],
                "terminal_guard_required_after_refresh": bool(
                    row.get("terminal_guard_required_after_refresh")
                ),
                "safe_to_auto_start_refresh": bool(row.get("safe_to_auto_start_refresh")),
                "release_claim_allowed_after_refresh": bool(
                    row.get("release_claim_allowed_after_refresh")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_post_input_refresh_plan.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_plan.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_post_input_refresh_plan.get("fail_closed")),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_plan.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_plan.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_plan.get("not_release_evidence")
        ),
    }
    terminal_release_gate_input_detection_source_summary = {
        "summary_version": _safe_int(
            release_gate_input_detection_source.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_input_detection_source.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_detection_source.get("artifact_role") or ""),
        "detection_status": str(release_gate_input_detection_source.get("detection_status") or ""),
        "detection_row_count": _safe_int(
            release_gate_input_detection_source.get("detection_row_count")
        ),
        "input_ids": _strings(release_gate_input_detection_source.get("input_ids"))[:50],
        "missing_or_unverified_input_count": _safe_int(
            release_gate_input_detection_source.get("missing_or_unverified_input_count")
        ),
        "missing_or_unverified_input_ids": _strings(
            release_gate_input_detection_source.get("missing_or_unverified_input_ids")
        )[:50],
        "detected_input_count": _safe_int(
            release_gate_input_detection_source.get("detected_input_count")
        ),
        "detected_input_ids": _strings(
            release_gate_input_detection_source.get("detected_input_ids")
        )[:50],
        "external_input_detector_count": _safe_int(
            release_gate_input_detection_source.get("external_input_detector_count")
        ),
        "manual_gpu_detector_count": _safe_int(
            release_gate_input_detection_source.get("manual_gpu_detector_count")
        ),
        "source_cache_refresh_detector_count": _safe_int(
            release_gate_input_detection_source.get("source_cache_refresh_detector_count")
        ),
        "unsafe_detector_count": _safe_int(
            release_gate_input_detection_source.get("unsafe_detector_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_detection_source.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "detection_status": str(row.get("detection_status") or ""),
                "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected": bool(row.get("detected")),
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "terminal_guard_required_after_detection": bool(
                    row.get("terminal_guard_required_after_detection")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_detection": bool(
                    row.get("release_claim_allowed_after_detection")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item) for item in _list(release_gate_input_detection_source.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_input_detection_source.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_input_detection_source.get("fail_closed")),
        "safe_to_auto_start": bool(release_gate_input_detection_source.get("safe_to_auto_start")),
        "release_claim_allowed": bool(
            release_gate_input_detection_source.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_input_detection_source.get("not_release_evidence")
        ),
    }
    terminal_release_gate_input_acceptance_criteria_summary = {
        "summary_version": _safe_int(
            release_gate_input_acceptance_criteria.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_input_acceptance_criteria.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_acceptance_criteria.get("artifact_role") or ""),
        "acceptance_status": str(release_gate_input_acceptance_criteria.get("acceptance_status") or ""),
        "acceptance_row_count": _safe_int(
            release_gate_input_acceptance_criteria.get("acceptance_row_count")
        ),
        "input_ids": _strings(release_gate_input_acceptance_criteria.get("input_ids"))[:50],
        "accepted_input_count": _safe_int(
            release_gate_input_acceptance_criteria.get("accepted_input_count")
        ),
        "accepted_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("accepted_input_ids")
        )[:50],
        "criteria_passed_input_count": _safe_int(
            release_gate_input_acceptance_criteria.get("criteria_passed_input_count")
        ),
        "criteria_passed_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("criteria_passed_input_ids")
        )[:50],
        "criteria_blocked_input_count": _safe_int(
            release_gate_input_acceptance_criteria.get("criteria_blocked_input_count")
        ),
        "criteria_blocked_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("criteria_blocked_input_ids")
        )[:50],
        "accepted_with_blocked_criteria_count": _safe_int(
            release_gate_input_acceptance_criteria.get(
                "accepted_with_blocked_criteria_count"
            )
        ),
        "accepted_with_blocked_criteria_ids": _strings(
            release_gate_input_acceptance_criteria.get(
                "accepted_with_blocked_criteria_ids"
            )
        )[:50],
        "unsatisfied_input_count": _safe_int(
            release_gate_input_acceptance_criteria.get("unsatisfied_input_count")
        ),
        "unsatisfied_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("unsatisfied_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            release_gate_input_acceptance_criteria.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            release_gate_input_acceptance_criteria.get(
                "detected_unaccepted_next_focus_ids"
            )
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            release_gate_input_acceptance_criteria.get(
                "detected_unaccepted_blocked_criteria_ids"
            )
        )[:50],
        "detected_unaccepted_reason_rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "detected": bool(row.get("detected")),
                "accepted": bool(row.get("accepted")),
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "refresh_ready": bool(row.get("refresh_ready")),
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "reason_ids": _strings(row.get("reason_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "detector_artifact_ids": _strings(
                    row.get("detector_artifact_ids")
                )[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item)
                for item in _list(
                    release_gate_input_acceptance_criteria.get(
                        "detected_unaccepted_reason_rows"
                    )
                )
            )
        ],
        "external_input_acceptance_count": _safe_int(
            release_gate_input_acceptance_criteria.get("external_input_acceptance_count")
        ),
        "manual_gpu_acceptance_count": _safe_int(
            release_gate_input_acceptance_criteria.get("manual_gpu_acceptance_count")
        ),
        "source_cache_refresh_acceptance_count": _safe_int(
            release_gate_input_acceptance_criteria.get("source_cache_refresh_acceptance_count")
        ),
        "unsafe_acceptance_count": _safe_int(
            release_gate_input_acceptance_criteria.get("unsafe_acceptance_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_acceptance_criteria.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "acceptance_status": str(row.get("acceptance_status") or ""),
                "criteria_status": str(row.get("criteria_status") or ""),
                "criteria_passed": bool(row.get("criteria_passed")),
                "blocked_criteria_ids": _strings(row.get("blocked_criteria_ids"))[:20],
                "criteria_flags": dict(_mapping(row.get("criteria_flags"))),
                "accepted_with_blocked_criteria": bool(
                    row.get("accepted_with_blocked_criteria")
                ),
                "acceptance_criteria_ids": _strings(row.get("acceptance_criteria_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected": bool(row.get("detected")),
                "accepted": bool(row.get("accepted")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "terminal_guard_required_after_acceptance": bool(
                    row.get("terminal_guard_required_after_acceptance")
                ),
                "release_claim_allowed_after_acceptance": bool(
                    row.get("release_claim_allowed_after_acceptance")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_input_acceptance_criteria.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_input_acceptance_criteria.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_input_acceptance_criteria.get("fail_closed")),
        "safe_to_auto_start": bool(
            release_gate_input_acceptance_criteria.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_input_acceptance_criteria.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_input_acceptance_criteria.get("not_release_evidence")
        ),
    }
    terminal_release_gate_input_refresh_readiness_summary = {
        "summary_version": _safe_int(
            release_gate_input_refresh_readiness.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_input_refresh_readiness.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_refresh_readiness.get("artifact_role") or ""),
        "refresh_readiness_status": str(
            release_gate_input_refresh_readiness.get("refresh_readiness_status") or ""
        ),
        "refresh_row_count": _safe_int(
            release_gate_input_refresh_readiness.get("refresh_row_count")
        ),
        "input_ids": _strings(release_gate_input_refresh_readiness.get("input_ids"))[:50],
        "accepted_input_count": _safe_int(
            release_gate_input_refresh_readiness.get("accepted_input_count")
        ),
        "accepted_input_ids": _strings(
            release_gate_input_refresh_readiness.get("accepted_input_ids")
        )[:50],
        "refresh_ready_input_count": _safe_int(
            release_gate_input_refresh_readiness.get("refresh_ready_input_count")
        ),
        "refresh_ready_input_ids": _strings(
            release_gate_input_refresh_readiness.get("refresh_ready_input_ids")
        )[:50],
        "blocked_refresh_input_count": _safe_int(
            release_gate_input_refresh_readiness.get("blocked_refresh_input_count")
        ),
        "blocked_refresh_input_ids": _strings(
            release_gate_input_refresh_readiness.get("blocked_refresh_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            release_gate_input_refresh_readiness.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            release_gate_input_refresh_readiness.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            release_gate_input_refresh_readiness.get(
                "detected_unaccepted_next_focus_ids"
            )
        )[:50],
        "detected_unaccepted_refresh_blocked_input_count": _safe_int(
            release_gate_input_refresh_readiness.get(
                "detected_unaccepted_refresh_blocked_input_count"
            )
        ),
        "detected_unaccepted_refresh_blocked_input_ids": _strings(
            release_gate_input_refresh_readiness.get(
                "detected_unaccepted_refresh_blocked_input_ids"
            )
        )[:50],
        "external_input_refresh_count": _safe_int(
            release_gate_input_refresh_readiness.get("external_input_refresh_count")
        ),
        "manual_gpu_refresh_count": _safe_int(
            release_gate_input_refresh_readiness.get("manual_gpu_refresh_count")
        ),
        "source_cache_refresh_count": _safe_int(
            release_gate_input_refresh_readiness.get("source_cache_refresh_count")
        ),
        "unsafe_refresh_count": _safe_int(
            release_gate_input_refresh_readiness.get("unsafe_refresh_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_refresh_readiness.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "refresh_readiness_status": str(
                    row.get("refresh_readiness_status") or ""
                ),
                "accepted": bool(row.get("accepted")),
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected": bool(row.get("detected")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "refresh_ready": bool(row.get("refresh_ready")),
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "acceptance_status": str(row.get("acceptance_status") or ""),
                "plan_status": str(row.get("plan_status") or ""),
                "acceptance_criteria_ids": _strings(row.get("acceptance_criteria_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "terminal_guard_command_ids": _strings(
                    row.get("terminal_guard_command_ids")
                )[:10],
                "post_refresh_required_artifact_ids": _strings(
                    row.get("post_refresh_required_artifact_ids")
                )[:20],
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "terminal_guard_required_after_refresh": bool(
                    row.get("terminal_guard_required_after_refresh")
                ),
                "safe_to_auto_start_refresh": bool(row.get("safe_to_auto_start_refresh")),
                "release_claim_allowed_after_refresh": bool(
                    row.get("release_claim_allowed_after_refresh")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_input_refresh_readiness.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_input_refresh_readiness.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_input_refresh_readiness.get("fail_closed")),
        "safe_to_auto_start": bool(
            release_gate_input_refresh_readiness.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_input_refresh_readiness.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_input_refresh_readiness.get("not_release_evidence")
        ),
    }
    terminal_release_gate_input_refresh_blocker_summary = {
        "summary_version": _safe_int(
            release_gate_input_refresh_blocker.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_input_refresh_blocker.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_refresh_blocker.get("artifact_role") or ""),
        "blocker_status": str(
            release_gate_input_refresh_blocker.get("blocker_status") or ""
        ),
        "blocker_row_count": _safe_int(
            release_gate_input_refresh_blocker.get("blocker_row_count")
        ),
        "input_ids": _strings(release_gate_input_refresh_blocker.get("input_ids"))[:50],
        "blocked_input_count": _safe_int(
            release_gate_input_refresh_blocker.get("blocked_input_count")
        ),
        "blocked_input_ids": _strings(
            release_gate_input_refresh_blocker.get("blocked_input_ids")
        )[:50],
        "refresh_ready_input_count": _safe_int(
            release_gate_input_refresh_blocker.get("refresh_ready_input_count")
        ),
        "refresh_ready_input_ids": _strings(
            release_gate_input_refresh_blocker.get("refresh_ready_input_ids")
        )[:50],
        "missing_input_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("missing_input_blocker_count")
        ),
        "undetected_input_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("undetected_input_blocker_count")
        ),
        "unaccepted_input_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("unaccepted_input_blocker_count")
        ),
        "detected_unaccepted_input_count": _safe_int(
            release_gate_input_refresh_blocker.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            release_gate_input_refresh_blocker.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            release_gate_input_refresh_blocker.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            release_gate_input_refresh_blocker.get(
                "detected_unaccepted_blocked_reason_ids"
            )
        )[:50],
        "detected_unaccepted_blocker_rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "detected": bool(row.get("detected")),
                "accepted": bool(row.get("accepted")),
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "refresh_ready": bool(row.get("refresh_ready")),
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "reason_ids": _strings(row.get("reason_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "detector_artifact_ids": _strings(
                    row.get("detector_artifact_ids")
                )[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
            }
            for row in (
                _mapping(item)
                for item in _list(
                    release_gate_input_refresh_blocker.get(
                        "detected_unaccepted_blocker_rows"
                    )
                )
            )
        ],
        "external_input_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("external_input_blocker_count")
        ),
        "manual_gpu_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("manual_gpu_blocker_count")
        ),
        "source_cache_refresh_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("source_cache_refresh_blocker_count")
        ),
        "terminal_guard_required_count": _safe_int(
            release_gate_input_refresh_blocker.get("terminal_guard_required_count")
        ),
        "unsafe_blocker_count": _safe_int(
            release_gate_input_refresh_blocker.get("unsafe_blocker_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_refresh_blocker.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "blocker_status": str(row.get("blocker_status") or ""),
                "blocked_reason_ids": _strings(row.get("blocked_reason_ids"))[:20],
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "refresh_ready": bool(row.get("refresh_ready")),
                "accepted": bool(row.get("accepted")),
                "detected": bool(row.get("detected")),
                "input_missing": bool(row.get("input_missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "terminal_guard_command_ids": _strings(
                    row.get("terminal_guard_command_ids")
                )[:10],
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "terminal_guard_required_after_refresh": bool(
                    row.get("terminal_guard_required_after_refresh")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_input_refresh_blocker.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_input_refresh_blocker.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_input_refresh_blocker.get("fail_closed")),
        "safe_to_auto_start": bool(
            release_gate_input_refresh_blocker.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_input_refresh_blocker.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_input_refresh_blocker.get("not_release_evidence")
        ),
    }
    terminal_release_gate_input_lifecycle_summary = {
        "summary_version": _safe_int(
            release_gate_input_lifecycle.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_input_lifecycle.get("roadmap") or ""),
        "artifact_role": str(release_gate_input_lifecycle.get("artifact_role") or ""),
        "lifecycle_status": str(
            release_gate_input_lifecycle.get("lifecycle_status") or ""
        ),
        "input_count": _safe_int(release_gate_input_lifecycle.get("input_count")),
        "input_ids": _strings(release_gate_input_lifecycle.get("input_ids"))[:50],
        "lifecycle_stage_counts": dict(
            _mapping(release_gate_input_lifecycle.get("lifecycle_stage_counts"))
        ),
        "detected_input_count": _safe_int(
            release_gate_input_lifecycle.get("detected_input_count")
        ),
        "detected_input_ids": _strings(
            release_gate_input_lifecycle.get("detected_input_ids")
        )[:50],
        "still_missing_input_count": _safe_int(
            release_gate_input_lifecycle.get("still_missing_input_count")
        ),
        "still_missing_input_ids": _strings(
            release_gate_input_lifecycle.get("still_missing_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            release_gate_input_lifecycle.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            release_gate_input_lifecycle.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            release_gate_input_lifecycle.get("detected_unaccepted_next_focus_ids")
        )[:50],
        "detected_but_not_accepted_input_count": _safe_int(
            release_gate_input_lifecycle.get(
                "detected_but_not_accepted_input_count"
            )
        ),
        "detected_but_not_accepted_input_ids": _strings(
            release_gate_input_lifecycle.get("detected_but_not_accepted_input_ids")
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            release_gate_input_lifecycle.get(
                "detected_unaccepted_blocked_criteria_ids"
            )
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            release_gate_input_lifecycle.get(
                "detected_unaccepted_blocked_reason_ids"
            )
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            release_gate_input_lifecycle.get("pending_not_missing_input_count")
        ),
        "pending_not_missing_input_ids": _strings(
            release_gate_input_lifecycle.get("pending_not_missing_input_ids")
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            release_gate_input_lifecycle.get("pending_not_missing_reason_ids")
        )[:50],
        "accepted_input_count": _safe_int(
            release_gate_input_lifecycle.get("accepted_input_count")
        ),
        "accepted_input_ids": _strings(
            release_gate_input_lifecycle.get("accepted_input_ids")
        )[:50],
        "accepted_pending_refresh_input_count": _safe_int(
            release_gate_input_lifecycle.get("accepted_pending_refresh_input_count")
        ),
        "accepted_pending_refresh_input_ids": _strings(
            release_gate_input_lifecycle.get("accepted_pending_refresh_input_ids")
        )[:50],
        "refresh_ready_input_count": _safe_int(
            release_gate_input_lifecycle.get("refresh_ready_input_count")
        ),
        "refresh_ready_input_ids": _strings(
            release_gate_input_lifecycle.get("refresh_ready_input_ids")
        )[:50],
        "blocked_input_count": _safe_int(
            release_gate_input_lifecycle.get("blocked_input_count")
        ),
        "blocked_input_ids": _strings(
            release_gate_input_lifecycle.get("blocked_input_ids")
        )[:50],
        "external_input_count": _safe_int(
            release_gate_input_lifecycle.get("external_input_count")
        ),
        "manual_gpu_input_count": _safe_int(
            release_gate_input_lifecycle.get("manual_gpu_input_count")
        ),
        "source_cache_refresh_input_count": _safe_int(
            release_gate_input_lifecycle.get("source_cache_refresh_input_count")
        ),
        "unsafe_input_count": _safe_int(
            release_gate_input_lifecycle.get("unsafe_input_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_input_lifecycle.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "lifecycle_stage": str(row.get("lifecycle_stage") or ""),
                "dependency_status": str(row.get("dependency_status") or ""),
                "detection_status": str(row.get("detection_status") or ""),
                "acceptance_status": str(row.get("acceptance_status") or ""),
                "refresh_readiness_status": str(row.get("refresh_readiness_status") or ""),
                "blocker_status": str(row.get("blocker_status") or ""),
                "missing": bool(row.get("missing")),
                "still_missing": bool(row.get("still_missing")),
                "detected": bool(row.get("detected")),
                "accepted": bool(row.get("accepted")),
                "detected_but_not_accepted": bool(row.get("detected_but_not_accepted")),
                "refresh_ready": bool(row.get("refresh_ready")),
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "blocked_criteria_ids": _strings(row.get("blocked_criteria_ids"))[:20],
                "blocked_reason_ids": _strings(row.get("blocked_reason_ids"))[:20],
                "required_evidence_artifact_ids": _strings(
                    row.get("required_evidence_artifact_ids")
                )[:20],
                "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                "required_refresh_command_ids": _strings(
                    row.get("required_refresh_command_ids")
                )[:50],
                "terminal_guard_command_ids": _strings(
                    row.get("terminal_guard_command_ids")
                )[:10],
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(row.get("requires_source_cache_refresh")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_input_lifecycle.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_input_lifecycle.get("execution_policy") or ""
        ),
        "fail_closed": bool(release_gate_input_lifecycle.get("fail_closed")),
        "safe_to_auto_start": bool(release_gate_input_lifecycle.get("safe_to_auto_start")),
        "release_claim_allowed": bool(
            release_gate_input_lifecycle.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_input_lifecycle.get("not_release_evidence")
        ),
    }
    terminal_external_input_release_gate_alignment_summary = {
        "summary_version": _safe_int(
            external_input_release_gate_alignment.get("summary_version"), 1
        ),
        "roadmap": str(external_input_release_gate_alignment.get("roadmap") or ""),
        "artifact_role": str(
            external_input_release_gate_alignment.get("artifact_role") or ""
        ),
        "alignment_status": str(
            external_input_release_gate_alignment.get("alignment_status") or ""
        ),
        "alignment_ok": bool(
            external_input_release_gate_alignment.get("alignment_ok")
        ),
        "external_input_count": _safe_int(
            external_input_release_gate_alignment.get("external_input_count")
        ),
        "external_input_ids": _strings(
            external_input_release_gate_alignment.get("external_input_ids")
        )[:50],
        "release_gate_input_count": _safe_int(
            external_input_release_gate_alignment.get("release_gate_input_count")
        ),
        "release_gate_input_ids": _strings(
            external_input_release_gate_alignment.get("release_gate_input_ids")
        )[:50],
        "external_release_gate_input_count": _safe_int(
            external_input_release_gate_alignment.get("external_release_gate_input_count")
        ),
        "manual_gpu_release_gate_input_count": _safe_int(
            external_input_release_gate_alignment.get(
                "manual_gpu_release_gate_input_count"
            )
        ),
        "source_cache_refresh_release_gate_input_count": _safe_int(
            external_input_release_gate_alignment.get(
                "source_cache_refresh_release_gate_input_count"
            )
        ),
        "non_external_release_gate_input_count": _safe_int(
            external_input_release_gate_alignment.get(
                "non_external_release_gate_input_count"
            )
        ),
        "non_external_release_gate_input_ids": _strings(
            external_input_release_gate_alignment.get(
                "non_external_release_gate_input_ids"
            )
        )[:50],
        "external_missing_from_release_gate_count": _safe_int(
            external_input_release_gate_alignment.get(
                "external_missing_from_release_gate_count"
            )
        ),
        "external_missing_from_release_gate_ids": _strings(
            external_input_release_gate_alignment.get(
                "external_missing_from_release_gate_ids"
            )
        )[:50],
        "release_external_missing_from_transition_count": _safe_int(
            external_input_release_gate_alignment.get(
                "release_external_missing_from_transition_count"
            )
        ),
        "release_external_missing_from_transition_ids": _strings(
            external_input_release_gate_alignment.get(
                "release_external_missing_from_transition_ids"
            )
        )[:50],
        "detected_input_count": _safe_int(
            external_input_release_gate_alignment.get("detected_input_count")
        ),
        "detected_input_ids": _strings(
            external_input_release_gate_alignment.get("detected_input_ids")
        )[:50],
        "accepted_input_count": _safe_int(
            external_input_release_gate_alignment.get("accepted_input_count")
        ),
        "accepted_input_ids": _strings(
            external_input_release_gate_alignment.get("accepted_input_ids")
        )[:50],
        "detected_unaccepted_input_count": _safe_int(
            external_input_release_gate_alignment.get("detected_unaccepted_input_count")
        ),
        "detected_unaccepted_input_ids": _strings(
            external_input_release_gate_alignment.get("detected_unaccepted_input_ids")
        )[:50],
        "detected_unaccepted_next_focus_ids": _strings(
            external_input_release_gate_alignment.get(
                "detected_unaccepted_next_focus_ids"
            )
        )[:50],
        "detected_unaccepted_blocked_criteria_ids": _strings(
            external_input_release_gate_alignment.get(
                "detected_unaccepted_blocked_criteria_ids"
            )
        )[:50],
        "detected_unaccepted_blocked_reason_ids": _strings(
            external_input_release_gate_alignment.get(
                "detected_unaccepted_blocked_reason_ids"
            )
        )[:50],
        "pending_not_missing_input_count": _safe_int(
            external_input_release_gate_alignment.get(
                "pending_not_missing_input_count"
            )
        ),
        "pending_not_missing_input_ids": _strings(
            external_input_release_gate_alignment.get(
                "pending_not_missing_input_ids"
            )
        )[:50],
        "pending_not_missing_reason_ids": _strings(
            external_input_release_gate_alignment.get(
                "pending_not_missing_reason_ids"
            )
        )[:50],
        "blocked_input_count": _safe_int(
            external_input_release_gate_alignment.get("blocked_input_count")
        ),
        "unsafe_alignment_count": _safe_int(
            external_input_release_gate_alignment.get("unsafe_alignment_count")
        ),
        "unsafe_input_ids": _strings(
            external_input_release_gate_alignment.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "input_kind": str(row.get("input_kind") or ""),
                "alignment_kind": str(row.get("alignment_kind") or ""),
                "requires_external_input": bool(row.get("requires_external_input")),
                "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    row.get("requires_source_cache_refresh")
                ),
                "in_external_transition_table": bool(
                    row.get("in_external_transition_table")
                ),
                "expected_in_external_transition_table": bool(
                    row.get("expected_in_external_transition_table")
                ),
                "external_input_missing": bool(row.get("external_input_missing")),
                "pending_not_missing": bool(row.get("pending_not_missing")),
                "pending_not_missing_reason_ids": _strings(
                    row.get("pending_not_missing_reason_ids")
                )[:20],
                "release_gate_input_present": bool(
                    row.get("release_gate_input_present")
                ),
                "lifecycle_stage": str(row.get("lifecycle_stage") or ""),
                "transition_state": str(row.get("transition_state") or ""),
                "missing": bool(row.get("missing")),
                "detected": bool(row.get("detected")),
                "accepted": bool(row.get("accepted")),
                "blocked_refresh": bool(row.get("blocked_refresh")),
                "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                "blocked_criteria_ids": _strings(row.get("blocked_criteria_ids"))[:20],
                "blocked_reason_ids": _strings(row.get("blocked_reason_ids"))[:20],
                "handoff_step_id": str(row.get("handoff_step_id") or ""),
                "replay_command_ids": _strings(row.get("replay_command_ids"))[:20],
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(external_input_release_gate_alignment.get("rows"))
            )
        ],
        "execution_policy": str(
            external_input_release_gate_alignment.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            external_input_release_gate_alignment.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            external_input_release_gate_alignment.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            external_input_release_gate_alignment.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            external_input_release_gate_alignment.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_command_surface_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_command_surface.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_command_surface.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_command_surface.get("artifact_role") or ""
        ),
        "command_surface_status": str(
            release_gate_post_input_refresh_command_surface.get("command_surface_status") or ""
        ),
        "command_row_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("command_row_count")
        ),
        "required_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("required_command_count")
        ),
        "required_command_ids": _strings(
            release_gate_post_input_refresh_command_surface.get("required_command_ids")
        )[:50],
        "json_refresh_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("json_refresh_command_count")
        ),
        "terminal_guard_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get(
                "terminal_guard_command_count"
            )
        ),
        "blocked_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("blocked_command_count")
        ),
        "blocked_command_ids": _strings(
            release_gate_post_input_refresh_command_surface.get("blocked_command_ids")
        )[:50],
        "ready_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("ready_command_count")
        ),
        "ready_command_ids": _strings(
            release_gate_post_input_refresh_command_surface.get("ready_command_ids")
        )[:50],
        "blocked_input_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("blocked_input_count")
        ),
        "blocked_input_ids": _strings(
            release_gate_post_input_refresh_command_surface.get("blocked_input_ids")
        )[:50],
        "unsafe_command_count": _safe_int(
            release_gate_post_input_refresh_command_surface.get("unsafe_command_count")
        ),
        "unsafe_command_ids": _strings(
            release_gate_post_input_refresh_command_surface.get("unsafe_command_ids")
        )[:50],
        "rows": [
            {
                "command_id": str(row.get("command_id") or ""),
                "command_order": _safe_int(row.get("command_order")),
                "command_kind": str(row.get("command_kind") or ""),
                "command_status": str(row.get("command_status") or ""),
                "related_input_ids": _strings(row.get("related_input_ids"))[:50],
                "blocked_input_ids": _strings(row.get("blocked_input_ids"))[:50],
                "blocked_input_count": _safe_int(row.get("blocked_input_count")),
                "refresh_ready_input_count": _safe_int(
                    row.get("refresh_ready_input_count")
                ),
                "terminal_guard_command": bool(row.get("terminal_guard_command")),
                "required_after_input_acceptance": bool(
                    row.get("required_after_input_acceptance")
                ),
                "blocked_until_input_acceptance": bool(
                    row.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(
                    release_gate_post_input_refresh_command_surface.get("rows")
                )
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_command_surface.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_command_surface.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_command_surface.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_command_surface.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_command_surface.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_sequence_integrity_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_sequence_integrity.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_sequence_integrity.get("artifact_role") or ""
        ),
        "sequence_status": str(
            release_gate_post_input_refresh_sequence_integrity.get("sequence_status") or ""
        ),
        "sequence_ok": bool(
            release_gate_post_input_refresh_sequence_integrity.get("sequence_ok")
        ),
        "expected_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("expected_command_count")
        ),
        "observed_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("observed_command_count")
        ),
        "expected_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("expected_command_ids")
        )[:50],
        "observed_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("observed_command_ids")
        )[:50],
        "missing_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("missing_command_count")
        ),
        "missing_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("missing_command_ids")
        )[:50],
        "unexpected_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("unexpected_command_count")
        ),
        "unexpected_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("unexpected_command_ids")
        )[:50],
        "duplicate_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("duplicate_command_count")
        ),
        "duplicate_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("duplicate_command_ids")
        )[:50],
        "order_matches_expected": bool(
            release_gate_post_input_refresh_sequence_integrity.get("order_matches_expected")
        ),
        "terminal_guard_tail_ok": bool(
            release_gate_post_input_refresh_sequence_integrity.get("terminal_guard_tail_ok")
        ),
        "terminal_guard_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("terminal_guard_command_ids")
        )[:10],
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_sequence_integrity.get(
                "blocked_until_input_acceptance"
            )
        ),
        "blocked_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("blocked_command_count")
        ),
        "ready_command_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("ready_command_count")
        ),
        "unsafe_sequence_count": _safe_int(
            release_gate_post_input_refresh_sequence_integrity.get("unsafe_sequence_count")
        ),
        "unsafe_command_ids": _strings(
            release_gate_post_input_refresh_sequence_integrity.get("unsafe_command_ids")
        )[:50],
        "execution_policy": str(
            release_gate_post_input_refresh_sequence_integrity.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_sequence_integrity.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_sequence_integrity.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_sequence_integrity.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_sequence_integrity.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_terminal_guard_dependency_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_terminal_guard_dependency.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_terminal_guard_dependency.get("artifact_role") or ""
        ),
        "dependency_status": str(
            release_gate_post_input_refresh_terminal_guard_dependency.get("dependency_status") or ""
        ),
        "dependency_ok": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("dependency_ok")
        ),
        "terminal_guard_required": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_required")
        ),
        "terminal_guard_command_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_command_count")
        ),
        "expected_terminal_guard_command_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "expected_terminal_guard_command_count"
            )
        ),
        "terminal_guard_command_ids": _strings(
            release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_command_ids")
        )[:10],
        "expected_terminal_guard_command_ids": _strings(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "expected_terminal_guard_command_ids"
            )
        )[:10],
        "terminal_guard_command_orders": [
            _safe_int(item)
            for item in _list(
                release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_command_orders"
                )
            )
        ][:10],
        "expected_terminal_guard_command_orders": [
            _safe_int(item)
            for item in _list(
                release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "expected_terminal_guard_command_orders"
                )
            )
        ][:10],
        "terminal_self_check_required": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_self_check_required")
        ),
        "release_guard_required": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("release_guard_required")
        ),
        "terminal_guard_tail_ok": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("terminal_guard_tail_ok")
        ),
        "all_json_refresh_commands_before_terminal_guard": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "all_json_refresh_commands_before_terminal_guard"
            )
        ),
        "json_refresh_command_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("json_refresh_command_count")
        ),
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get(
                "blocked_until_input_acceptance"
            )
        ),
        "blocked_command_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("blocked_command_count")
        ),
        "ready_command_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("ready_command_count")
        ),
        "unsafe_dependency_count": _safe_int(
            release_gate_post_input_refresh_terminal_guard_dependency.get("unsafe_dependency_count")
        ),
        "unsafe_command_ids": _strings(
            release_gate_post_input_refresh_terminal_guard_dependency.get("unsafe_command_ids")
        )[:50],
        "rows": [
            {
                "command_id": str(row.get("command_id") or ""),
                "dependency_order": _safe_int(row.get("dependency_order")),
                "command_order": _safe_int(row.get("command_order")),
                "guard_kind": str(row.get("guard_kind") or ""),
                "depends_on_json_refresh_sequence": bool(
                    row.get("depends_on_json_refresh_sequence")
                ),
                "required_after_json_refresh": bool(row.get("required_after_json_refresh")),
                "blocked_until_input_acceptance": bool(
                    row.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(
                    release_gate_post_input_refresh_terminal_guard_dependency.get("rows")
                )
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_terminal_guard_dependency.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_terminal_guard_dependency.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_artifact_coverage_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_artifact_coverage.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_artifact_coverage.get("artifact_role") or ""
        ),
        "coverage_status": str(
            release_gate_post_input_refresh_artifact_coverage.get("coverage_status") or ""
        ),
        "coverage_ok": bool(
            release_gate_post_input_refresh_artifact_coverage.get("coverage_ok")
        ),
        "required_artifact_count": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("required_artifact_count")
        ),
        "required_artifact_ids": _strings(
            release_gate_post_input_refresh_artifact_coverage.get("required_artifact_ids")
        )[:20],
        "input_row_count": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("input_row_count")
        ),
        "covered_input_count": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("covered_input_count")
        ),
        "covered_input_ids": _strings(
            release_gate_post_input_refresh_artifact_coverage.get("covered_input_ids")
        )[:50],
        "missing_coverage_input_count": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("missing_coverage_input_count")
        ),
        "missing_coverage_input_ids": _strings(
            release_gate_post_input_refresh_artifact_coverage.get("missing_coverage_input_ids")
        )[:50],
        "readiness_artifact_required": bool(
            release_gate_post_input_refresh_artifact_coverage.get("readiness_artifact_required")
        ),
        "terminal_artifact_required": bool(
            release_gate_post_input_refresh_artifact_coverage.get("terminal_artifact_required")
        ),
        "release_guard_artifact_required": bool(
            release_gate_post_input_refresh_artifact_coverage.get("release_guard_artifact_required")
        ),
        "terminal_guard_dependency_ok": bool(
            release_gate_post_input_refresh_artifact_coverage.get("terminal_guard_dependency_ok")
        ),
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_artifact_coverage.get("blocked_until_input_acceptance")
        ),
        "unsafe_artifact_coverage_count": _safe_int(
            release_gate_post_input_refresh_artifact_coverage.get("unsafe_artifact_coverage_count")
        ),
        "unsafe_input_ids": _strings(
            release_gate_post_input_refresh_artifact_coverage.get("unsafe_input_ids")
        )[:50],
        "rows": [
            {
                "input_id": str(row.get("input_id") or ""),
                "artifact_ids": _strings(row.get("artifact_ids"))[:20],
                "artifact_count": _safe_int(row.get("artifact_count")),
                "missing_artifact_ids": _strings(row.get("missing_artifact_ids"))[:20],
                "readiness_artifact_required": bool(row.get("readiness_artifact_required")),
                "terminal_artifact_required": bool(row.get("terminal_artifact_required")),
                "release_guard_artifact_required": bool(row.get("release_guard_artifact_required")),
                "covered": bool(row.get("covered")),
                "blocked_until_input_acceptance": bool(row.get("blocked_until_input_acceptance")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_post_input_refresh_artifact_coverage.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_artifact_coverage.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_artifact_coverage.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_artifact_coverage.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_artifact_coverage.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_artifact_coverage.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_command_artifact_link_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_command_artifact_link.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_command_artifact_link.get("artifact_role") or ""
        ),
        "link_status": str(
            release_gate_post_input_refresh_command_artifact_link.get("link_status") or ""
        ),
        "link_ok": bool(release_gate_post_input_refresh_command_artifact_link.get("link_ok")),
        "command_row_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("command_row_count")
        ),
        "required_artifact_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("required_artifact_count")
        ),
        "required_artifact_ids": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("required_artifact_ids")
        )[:20],
        "linked_artifact_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("linked_artifact_count")
        ),
        "linked_artifact_ids": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("linked_artifact_ids")
        )[:20],
        "missing_link_artifact_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("missing_link_artifact_count")
        ),
        "missing_link_artifact_ids": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("missing_link_artifact_ids")
        )[:20],
        "extra_link_artifact_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("extra_link_artifact_count")
        ),
        "extra_link_artifact_ids": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("extra_link_artifact_ids")
        )[:20],
        "command_artifact_link_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("command_artifact_link_count")
        ),
        "command_ids_with_artifacts": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("command_ids_with_artifacts")
        )[:20],
        "command_ids_without_artifacts": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("command_ids_without_artifacts")
        )[:50],
        "readiness_command_id": str(
            release_gate_post_input_refresh_command_artifact_link.get("readiness_command_id") or ""
        ),
        "terminal_command_id": str(
            release_gate_post_input_refresh_command_artifact_link.get("terminal_command_id") or ""
        ),
        "release_guard_command_id": str(
            release_gate_post_input_refresh_command_artifact_link.get("release_guard_command_id") or ""
        ),
        "artifact_coverage_ok": bool(
            release_gate_post_input_refresh_command_artifact_link.get("artifact_coverage_ok")
        ),
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_command_artifact_link.get(
                "blocked_until_input_acceptance"
            )
        ),
        "unsafe_link_count": _safe_int(
            release_gate_post_input_refresh_command_artifact_link.get("unsafe_link_count")
        ),
        "unsafe_command_ids": _strings(
            release_gate_post_input_refresh_command_artifact_link.get("unsafe_command_ids")
        )[:50],
        "rows": [
            {
                "command_id": str(row.get("command_id") or ""),
                "command_order": _safe_int(row.get("command_order")),
                "command_kind": str(row.get("command_kind") or ""),
                "output_artifact_ids": _strings(row.get("output_artifact_ids"))[:20],
                "output_artifact_count": _safe_int(row.get("output_artifact_count")),
                "produces_required_post_refresh_artifact": bool(
                    row.get("produces_required_post_refresh_artifact")
                ),
                "blocked_until_input_acceptance": bool(
                    row.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_post_input_refresh_command_artifact_link.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_command_artifact_link.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_command_artifact_link.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_command_artifact_link.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_command_artifact_link.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_command_artifact_link.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_guard_consumption_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_guard_consumption.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_guard_consumption.get("artifact_role") or ""
        ),
        "consumption_status": str(
            release_gate_post_input_refresh_guard_consumption.get("consumption_status") or ""
        ),
        "consumption_ok": bool(
            release_gate_post_input_refresh_guard_consumption.get("consumption_ok")
        ),
        "guard_command_id": str(
            release_gate_post_input_refresh_guard_consumption.get("guard_command_id") or ""
        ),
        "required_input_artifact_count": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "required_input_artifact_count"
            )
        ),
        "required_input_artifact_ids": _strings(
            release_gate_post_input_refresh_guard_consumption.get(
                "required_input_artifact_ids"
            )
        )[:20],
        "produced_guard_artifact_id": str(
            release_gate_post_input_refresh_guard_consumption.get(
                "produced_guard_artifact_id"
            )
            or ""
        ),
        "input_artifacts_consumed": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "input_artifacts_consumed"
            )
        ),
        "guard_artifact_produced": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "guard_artifact_produced"
            )
        ),
        "required_consumed_summary_count": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "required_consumed_summary_count"
            )
        ),
        "required_consumed_summary_ids": _strings(
            release_gate_post_input_refresh_guard_consumption.get(
                "required_consumed_summary_ids"
            )
        )[:50],
        "present_consumed_summary_count": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "present_consumed_summary_count"
            )
        ),
        "missing_consumed_summary_count": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "missing_consumed_summary_count"
            )
        ),
        "missing_consumed_summary_ids": _strings(
            release_gate_post_input_refresh_guard_consumption.get(
                "missing_consumed_summary_ids"
            )
        )[:50],
        "unsafe_consumed_summary_count": _safe_int(
            release_gate_post_input_refresh_guard_consumption.get(
                "unsafe_consumed_summary_count"
            )
        ),
        "unsafe_consumed_summary_ids": _strings(
            release_gate_post_input_refresh_guard_consumption.get(
                "unsafe_consumed_summary_ids"
            )
        )[:50],
        "terminal_lineage_required": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "terminal_lineage_required"
            )
        ),
        "terminal_lineage_summary_id": str(
            release_gate_post_input_refresh_guard_consumption.get(
                "terminal_lineage_summary_id"
            )
            or ""
        ),
        "command_artifact_link_ok": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "command_artifact_link_ok"
            )
        ),
        "artifact_coverage_ok": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "artifact_coverage_ok"
            )
        ),
        "terminal_guard_dependency_ok": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "terminal_guard_dependency_ok"
            )
        ),
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_guard_consumption.get(
                "blocked_until_input_acceptance"
            )
        ),
        "rows": [
            {
                "summary_id": str(row.get("summary_id") or ""),
                "consumption_stage": str(row.get("consumption_stage") or ""),
                "required_for_guard": bool(row.get("required_for_guard")),
                "present": bool(row.get("present")),
                "fail_closed": bool(row.get("fail_closed")),
                "terminal_only": bool(row.get("terminal_only")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "unsafe": bool(row.get("unsafe")),
                **(
                    {
                        "manifest_ok": bool(row.get("manifest_ok")),
                        "runner_ready": bool(row.get("runner_ready")),
                        "execution_ok": bool(row.get("execution_ok")),
                        "in_progress_runner_manifest_accepted": bool(
                            row.get("in_progress_runner_manifest_accepted")
                        ),
                        "safety_contract_ok": bool(row.get("safety_contract_ok")),
                        "execution_status": str(row.get("execution_status") or ""),
                        "row_execution_consistent": bool(
                            row.get("row_execution_consistent")
                        ),
                        "expected_command_count": _safe_int(
                            row.get("expected_command_count")
                        ),
                        "row_count": _safe_int(row.get("row_count")),
                        "executed_row_count": _safe_int(row.get("executed_row_count")),
                        "failed_row_count": _safe_int(row.get("failed_row_count")),
                        "missing_output_row_count": _safe_int(
                            row.get("missing_output_row_count")
                        ),
                        "row_forbidden_heavy_flag_count": _safe_int(
                            row.get("row_forbidden_heavy_flag_count")
                        ),
                        "unsafe_row_count": _safe_int(row.get("unsafe_row_count")),
                    }
                    if str(row.get("summary_id") or "")
                    == "external_input_json_refresh_runner_manifest_summary"
                    else {}
                ),
            }
            for row in (
                _mapping(item)
                for item in _list(release_gate_post_input_refresh_guard_consumption.get("rows"))
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_guard_consumption.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_guard_consumption.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_guard_consumption.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_guard_consumption.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_guard_consumption.get("not_release_evidence")
        ),
    }
    terminal_release_gate_post_input_refresh_guard_report_acceptance_summary = {
        "summary_version": _safe_int(
            release_gate_post_input_refresh_guard_report_acceptance.get("summary_version"), 1
        ),
        "roadmap": str(release_gate_post_input_refresh_guard_report_acceptance.get("roadmap") or ""),
        "artifact_role": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("artifact_role") or ""
        ),
        "acceptance_status": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("acceptance_status") or ""
        ),
        "acceptance_ok": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("acceptance_ok")
        ),
        "guard_command_id": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("guard_command_id") or ""
        ),
        "guard_report_artifact_id": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("guard_report_artifact_id") or ""
        ),
        "expected_report_status": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("expected_report_status") or ""
        ),
        "expected_ok": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("expected_ok")
        ),
        "expected_failure_count": _safe_int(
            release_gate_post_input_refresh_guard_report_acceptance.get("expected_failure_count")
        ),
        "required_guard_report_field_count": _safe_int(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "required_guard_report_field_count"
            )
        ),
        "required_guard_report_fields": _strings(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "required_guard_report_fields"
            )
        )[:50],
        "requires_input_artifact_summary": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "requires_input_artifact_summary"
            )
        ),
        "requires_not_release_evidence": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "requires_not_release_evidence"
            )
        ),
        "requires_safe_to_auto_start_false": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "requires_safe_to_auto_start_false"
            )
        ),
        "requires_release_claim_allowed_false": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "requires_release_claim_allowed_false"
            )
        ),
        "requires_blocked_actions": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "requires_blocked_actions"
            )
        ),
        "guard_consumption_ok": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("guard_consumption_ok")
        ),
        "input_artifacts_consumed": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "input_artifacts_consumed"
            )
        ),
        "guard_artifact_produced": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "guard_artifact_produced"
            )
        ),
        "blocked_until_input_acceptance": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "blocked_until_input_acceptance"
            )
        ),
        "acceptance_row_count": _safe_int(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "acceptance_row_count"
            )
        ),
        "unsafe_acceptance_count": _safe_int(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "unsafe_acceptance_count"
            )
        ),
        "unsafe_acceptance_ids": _strings(
            release_gate_post_input_refresh_guard_report_acceptance.get(
                "unsafe_acceptance_ids"
            )
        )[:50],
        "rows": [
            {
                "acceptance_id": str(row.get("acceptance_id") or ""),
                "required_field_ids": _strings(row.get("required_field_ids"))[:20],
                "required": bool(row.get("required")),
                "expected_value_summary": str(row.get("expected_value_summary") or ""),
                "present": bool(row.get("present")),
                "fail_closed": bool(row.get("fail_closed")),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed": bool(row.get("release_claim_allowed")),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (
                _mapping(item)
                for item in _list(
                    release_gate_post_input_refresh_guard_report_acceptance.get("rows")
                )
            )
        ],
        "execution_policy": str(
            release_gate_post_input_refresh_guard_report_acceptance.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("fail_closed")
        ),
        "safe_to_auto_start": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            release_gate_post_input_refresh_guard_report_acceptance.get("not_release_evidence")
        ),
    }
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_plan_summary,
        source=release_gate_post_input_refresh_plan,
        row_key="input_id",
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_command_surface_summary,
        source=release_gate_post_input_refresh_command_surface,
        row_key="command_id",
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_sequence_integrity_summary,
        source=release_gate_post_input_refresh_sequence_integrity,
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_terminal_guard_dependency_summary,
        source=release_gate_post_input_refresh_terminal_guard_dependency,
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_artifact_coverage_summary,
        source=release_gate_post_input_refresh_artifact_coverage,
        row_key="input_id",
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_command_artifact_link_summary,
        source=release_gate_post_input_refresh_command_artifact_link,
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_guard_consumption_summary,
        source=release_gate_post_input_refresh_guard_consumption,
    )
    _project_detected_unaccepted(
        target=terminal_release_gate_post_input_refresh_guard_report_acceptance_summary,
        source=release_gate_post_input_refresh_guard_report_acceptance,
    )
    terminal_external_input_json_refresh_runner_manifest_summary = {
        "summary_version": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("summary_version"), 1
        ),
        "roadmap": str(external_input_json_refresh_runner_manifest_summary.get("roadmap") or ""),
        "artifact_role": str(
            external_input_json_refresh_runner_manifest_summary.get("artifact_role") or ""
        ),
        "status": str(external_input_json_refresh_runner_manifest_summary.get("status") or ""),
        "execution_status": str(
            external_input_json_refresh_runner_manifest_summary.get("execution_status") or ""
        ),
        "manifest_available": bool(
            external_input_json_refresh_runner_manifest_summary.get("manifest_available")
        ),
        "manifest_probe": str(
            external_input_json_refresh_runner_manifest_summary.get("manifest_probe") or ""
        ),
        "manifest_ok": bool(external_input_json_refresh_runner_manifest_summary.get("manifest_ok")),
        "runner_ready": bool(external_input_json_refresh_runner_manifest_summary.get("runner_ready")),
        "execution_ok": bool(
            external_input_json_refresh_runner_manifest_summary.get("execution_ok")
        ),
        "safety_contract_ok": bool(
            external_input_json_refresh_runner_manifest_summary.get("safety_contract_ok")
        ),
        "row_execution_consistent": bool(
            external_input_json_refresh_runner_manifest_summary.get(
                "row_execution_consistent"
            )
        ),
        "expected_command_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("expected_command_count")
        ),
        "command_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("command_count")
        ),
        "row_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("row_count")
        ),
        "executed_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("executed_count")
        ),
        "executed_row_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("executed_row_count")
        ),
        "failure_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("failure_count")
        ),
        "failed_row_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("failed_row_count")
        ),
        "output_missing_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("output_missing_count")
        ),
        "missing_output_row_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "missing_output_row_count"
            )
        ),
        "forbidden_heavy_flag_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("forbidden_heavy_flag_count")
        ),
        "row_forbidden_heavy_flag_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "row_forbidden_heavy_flag_count"
            )
        ),
        "unsafe_row_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("unsafe_row_count")
        ),
        "validation_issue_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("validation_issue_count")
        ),
        "sequence_ok": bool(external_input_json_refresh_runner_manifest_summary.get("sequence_ok")),
        "canonical_stage_ids": _strings(
            external_input_json_refresh_runner_manifest_summary.get("canonical_stage_ids")
        )[:50],
        "observed_stage_ids": _strings(
            external_input_json_refresh_runner_manifest_summary.get("observed_stage_ids")
        )[:50],
        "row_stage_ids": _strings(
            external_input_json_refresh_runner_manifest_summary.get("row_stage_ids")
        )[:50],
        "stage_manifest_source": str(
            external_input_json_refresh_runner_manifest_summary.get("stage_manifest_source")
            or ""
        ),
        "stage_manifest_ok": bool(
            external_input_json_refresh_runner_manifest_summary.get("stage_manifest_ok")
        ),
        "stage_manifest_issue_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "stage_manifest_issue_count"
            )
        ),
        "stage_manifest_issue_reasons": _strings(
            external_input_json_refresh_runner_manifest_summary.get(
                "stage_manifest_issue_reasons"
            )
        )[:20],
        "stage_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("stage_count")
        ),
        "stage_ids": _strings(
            external_input_json_refresh_runner_manifest_summary.get("stage_ids")
        )[:50],
        "script_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("script_count")
        ),
        "expected_output_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get("expected_output_count")
        ),
        "stage_manifest_forbidden_heavy_flag_count": _safe_int(
            external_input_json_refresh_runner_manifest_summary.get(
                "stage_manifest_forbidden_heavy_flag_count"
            )
        ),
        "execution_policy": str(
            external_input_json_refresh_runner_manifest_summary.get("execution_policy") or ""
        ),
        "fail_closed": bool(
            external_input_json_refresh_runner_manifest_summary.get("fail_closed")
        ),
        "does_not_run_training": bool(
            external_input_json_refresh_runner_manifest_summary.get("does_not_run_training")
        ),
        "does_not_run_cuda": bool(
            external_input_json_refresh_runner_manifest_summary.get("does_not_run_cuda")
        ),
        "safe_to_auto_start": bool(
            external_input_json_refresh_runner_manifest_summary.get("safe_to_auto_start")
        ),
        "release_claim_allowed": bool(
            external_input_json_refresh_runner_manifest_summary.get("release_claim_allowed")
        ),
        "not_release_evidence": bool(
            external_input_json_refresh_runner_manifest_summary.get("not_release_evidence")
        ),
    }
    terminal_command_surface_summary = {
        "summary_version": _safe_int(command_surface.get("summary_version"), 1),
        "roadmap": str(command_surface.get("roadmap") or ""),
        "artifact_role": str(command_surface.get("artifact_role") or ""),
        "surface_status": str(command_surface.get("surface_status") or ""),
        "source_artifact_count": _safe_int(command_surface.get("source_artifact_count")),
        "source_artifact_ids": _strings(command_surface.get("source_artifact_ids"))[:20],
        "command_surface_row_count": _safe_int(command_surface.get("command_surface_row_count")),
        "row_ids": _strings(command_surface.get("row_ids"))[:50],
        "row_refs": _strings(command_surface.get("row_refs"))[:50],
        "status_counts": dict(_mapping(command_surface.get("status_counts"))),
        "family_counts": dict(_mapping(command_surface.get("family_counts"))),
        "source_artifact_counts": dict(_mapping(command_surface.get("source_artifact_counts"))),
        "manual_gpu_command_count": _safe_int(command_surface.get("manual_gpu_command_count")),
        "manual_gpu_command_ids": _strings(command_surface.get("manual_gpu_command_ids"))[:50],
        "manual_gpu_command_row_refs": _strings(command_surface.get("manual_gpu_command_row_refs"))[:50],
        "protected_gpu_command_count": _safe_int(command_surface.get("protected_gpu_command_count")),
        "protected_gpu_command_ids": _strings(command_surface.get("protected_gpu_command_ids"))[:50],
        "protected_gpu_command_row_refs": _strings(command_surface.get("protected_gpu_command_row_refs"))[:50],
        "dry_run_command_count": _safe_int(command_surface.get("dry_run_command_count")),
        "dry_run_command_ids": _strings(command_surface.get("dry_run_command_ids"))[:50],
        "dry_run_command_row_refs": _strings(command_surface.get("dry_run_command_row_refs"))[:50],
        "template_command_count": _safe_int(command_surface.get("template_command_count")),
        "template_command_ids": _strings(command_surface.get("template_command_ids"))[:50],
        "template_command_row_refs": _strings(command_surface.get("template_command_row_refs"))[:50],
        "ready_command_count": _safe_int(command_surface.get("ready_command_count")),
        "ready_command_ids": _strings(command_surface.get("ready_command_ids"))[:50],
        "ready_command_row_refs": _strings(command_surface.get("ready_command_row_refs"))[:50],
        "blocked_command_count": _safe_int(command_surface.get("blocked_command_count")),
        "blocked_command_ids": _strings(command_surface.get("blocked_command_ids"))[:50],
        "blocked_command_row_refs": _strings(command_surface.get("blocked_command_row_refs"))[:50],
        "completed_existing_command_count": _safe_int(
            command_surface.get("completed_existing_command_count")
        ),
        "completed_existing_command_ids": _strings(
            command_surface.get("completed_existing_command_ids")
        )[:50],
        "completed_existing_command_row_refs": _strings(
            command_surface.get("completed_existing_command_row_refs")
        )[:50],
        "rerun_blocked_without_new_axis_count": _safe_int(
            command_surface.get("rerun_blocked_without_new_axis_count")
        ),
        "rerun_blocked_without_new_axis_ids": _strings(
            command_surface.get("rerun_blocked_without_new_axis_ids")
        )[:50],
        "rerun_blocked_without_new_axis_row_refs": _strings(
            command_surface.get("rerun_blocked_without_new_axis_row_refs")
        )[:50],
        "requires_gpu_if_executed_count": _safe_int(
            command_surface.get("requires_gpu_if_executed_count")
        ),
        "requires_gpu_if_executed_ids": _strings(command_surface.get("requires_gpu_if_executed_ids"))[:50],
        "requires_gpu_if_executed_row_refs": _strings(
            command_surface.get("requires_gpu_if_executed_row_refs")
        )[:50],
        "manual_start_required_count": _safe_int(
            command_surface.get("manual_start_required_count")
        ),
        "manual_start_required_ids": _strings(command_surface.get("manual_start_required_ids"))[:50],
        "manual_start_required_row_refs": _strings(
            command_surface.get("manual_start_required_row_refs")
        )[:50],
        "release_relevant_command_count": _safe_int(
            command_surface.get("release_relevant_command_count")
        ),
        "release_relevant_command_ids": _strings(command_surface.get("release_relevant_command_ids"))[:50],
        "release_relevant_command_row_refs": _strings(
            command_surface.get("release_relevant_command_row_refs")
        )[:50],
        "diagnostic_only_command_count": _safe_int(
            command_surface.get("diagnostic_only_command_count")
        ),
        "diagnostic_only_command_ids": _strings(command_surface.get("diagnostic_only_command_ids"))[:50],
        "diagnostic_only_command_row_refs": _strings(
            command_surface.get("diagnostic_only_command_row_refs")
        )[:50],
        "release_claim_allowed_after_success_count": _safe_int(
            command_surface.get("release_claim_allowed_after_success_count")
        ),
        "release_claim_allowed_after_success_ids": _strings(
            command_surface.get("release_claim_allowed_after_success_ids")
        )[:50],
        "release_claim_allowed_after_success_row_refs": _strings(
            command_surface.get("release_claim_allowed_after_success_row_refs")
        )[:50],
        "unsafe_command_count": _safe_int(command_surface.get("unsafe_command_count")),
        "unsafe_command_ids": _strings(command_surface.get("unsafe_command_ids"))[:50],
        "unsafe_command_row_refs": _strings(command_surface.get("unsafe_command_row_refs"))[:50],
        "run_plan_source_artifact_id": str(command_surface.get("run_plan_source_artifact_id") or ""),
        "run_plan_command_count": _safe_int(command_surface.get("run_plan_command_count")),
        "run_plan_execution_surface_status": str(
            command_surface.get("run_plan_execution_surface_status") or ""
        ),
        "run_plan_active_release_relevant_command_count": _safe_int(
            command_surface.get("run_plan_active_release_relevant_command_count")
        ),
        "run_plan_active_release_relevant_command_ids": _strings(
            command_surface.get("run_plan_active_release_relevant_command_ids")
        )[:50],
        "run_plan_diagnostic_manual_ready_command_count": _safe_int(
            command_surface.get("run_plan_diagnostic_manual_ready_command_count")
        ),
        "run_plan_diagnostic_manual_ready_command_ids": _strings(
            command_surface.get("run_plan_diagnostic_manual_ready_command_ids")
        )[:50],
        "run_plan_completed_existing_command_count": _safe_int(
            command_surface.get("run_plan_completed_existing_command_count")
        ),
        "run_plan_completed_existing_command_ids": _strings(
            command_surface.get("run_plan_completed_existing_command_ids")
        )[:50],
        "run_plan_rerun_blocked_without_new_axis_count": _safe_int(
            command_surface.get("run_plan_rerun_blocked_without_new_axis_count")
        ),
        "run_plan_rerun_blocked_without_new_axis_command_ids": _strings(
            command_surface.get("run_plan_rerun_blocked_without_new_axis_command_ids")
        )[:50],
        "run_plan_blocked_nonrelease_command_count": _safe_int(
            command_surface.get("run_plan_blocked_nonrelease_command_count")
        ),
        "run_plan_blocked_nonrelease_command_ids": _strings(
            command_surface.get("run_plan_blocked_nonrelease_command_ids")
        )[:50],
        "rows": [
            {
                "id": str(row.get("id") or ""),
                "source_artifact": str(row.get("source_artifact") or ""),
                "source_key": str(row.get("source_key") or ""),
                "family": str(row.get("family") or ""),
                "status": str(row.get("status") or ""),
                "ready": bool(row.get("ready")),
                "blocked": bool(row.get("blocked")),
                "template": bool(row.get("template")),
                "dry_run_present": bool(row.get("dry_run_present")),
                "requires_gpu_if_executed": bool(row.get("requires_gpu_if_executed")),
                "manual_start_required": bool(row.get("manual_start_required")),
                "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    row.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(row.get("not_release_evidence")),
                "release_relevant": bool(row.get("release_relevant")),
                "diagnostic_only": bool(row.get("diagnostic_only")),
                "completed_existing": bool(row.get("completed_existing")),
                "do_not_rerun_without_new_axis": bool(
                    row.get("do_not_rerun_without_new_axis")
                ),
                "unsafe": bool(row.get("unsafe")),
            }
            for row in (_mapping(item) for item in _list(command_surface.get("rows")))
        ],
        "execution_policy": str(command_surface.get("execution_policy") or ""),
        "fail_closed": bool(command_surface.get("fail_closed")),
        "safe_to_auto_start": bool(command_surface.get("safe_to_auto_start")),
        "release_claim_allowed": bool(command_surface.get("release_claim_allowed")),
        "not_release_evidence": bool(command_surface.get("not_release_evidence")),
    }
    guard_lineage = _guard_report_lineage_summary(guard_report)
    roadmap_lineage = _roadmap_lineage_audit(
        artifacts={
            "readiness_next_actions": readiness,
            "external_input_handoff_packet": handoff,
            "external_input_intake_registry": intake,
            "live_external_input_intake_registry": live_intake,
            "source_axis_requirement": source_requirement,
            "source_axis_freshness_dedupe_audit": source_freshness,
            "source_cache_axis_identity_registry": identity_registry,
            "newbie_warm_cache_inventory": warm_cache,
            "source_cache_axis_pipeline_readiness": pipeline,
            "post_manual_evidence_rebuild_plan": post_manual,
            "sdxl_non_dataloader_manual_gpu_queue": sdxl_manual_gpu_queue,
            "sd15_readiness": sd15,
        }
    )
    terminal_release_unblocker = {
        "summary_version": _safe_int(unblocker.get("summary_version"), 1),
        "recommended_next_non_gpu_focus": str(
            unblocker.get("recommended_next_non_gpu_focus")
            or readiness.get("recommended_next_non_gpu_focus")
            or ""
        ),
        "stable_first_release_blocked_by_this_artifact": bool(
            unblocker.get("stable_first_release_blocked_by_this_artifact")
            or first_release_scope.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_allowed": False,
        "gpu_bubble_release_claim_blocked": bool(hard_gates),
        "gpu_bubble_release_hard_gate_ids": hard_gates[:20],
        "recommended_release_policy": recommended_release_policy,
        "claim_publication_scope": str(first_release_scope.get("claim_publication_scope") or ""),
        "does_not_prove_global_product_release": bool(
            first_release_scope.get("does_not_prove_global_product_release")
        ),
        "claim_wording_policy": claim_wording_policy,
        "release_gain_claim_wording_allowed": bool(readiness.get("release_gain_claim_wording_allowed")),
        "forbidden_claim_wording_hit_count": forbidden_claim_wording_hit_count,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "external_input_required": bool(missing_inputs),
        "missing_external_inputs": missing_inputs[:20],
        "external_input_handoff_status": str(handoff.get("status") or unblocker.get("external_input_handoff_status") or ""),
        "external_input_replay_status": str(unblocker.get("external_input_replay_status") or ""),
        "external_input_replay_command_count": _safe_int(unblocker.get("external_input_replay_command_count")),
        "external_input_replay_ready_command_count": _safe_int(
            unblocker.get("external_input_replay_ready_command_count")
        ),
        "external_input_replay_template_command_count": _safe_int(
            unblocker.get("external_input_replay_template_command_count")
        ),
        "source_cache_axis_pipeline_status": str(
            pipeline.get("status") or unblocker.get("source_cache_axis_pipeline_status") or ""
        ),
        "source_cache_axis_readiness_status": str(
            pipeline.get("axis_readiness_status")
            or unblocker.get("source_cache_axis_readiness_status")
            or ""
        ),
        "source_cache_axis_pipeline_complete": pipeline_complete,
        "source_cache_axis_stage_ok_count": stage_ok,
        "source_cache_axis_stage_count": stage_count,
        "post_manual_rebuild_status": str(post_manual.get("status") or unblocker.get("post_manual_rebuild_status") or ""),
        "post_manual_ready_command_count": _safe_int(
            post_manual.get("ready_command_count"),
            _safe_int(unblocker.get("post_manual_ready_command_count")),
        ),
        "post_manual_next_rebuild_stage_id": str(
            _mapping(post_manual.get("next_rebuild_stage")).get("stage_id")
            or unblocker.get("post_manual_next_rebuild_stage_id")
            or ""
        ),
        "post_manual_next_required_inputs": (
            _strings(manual_evidence_blocking.get("next_required_inputs"))
            or _strings(unblocker.get("post_manual_next_required_inputs"))
        )[:20],
        "post_manual_release_gate_blockers": (
            _strings(manual_evidence_blocking.get("release_gate_blockers"))
            or _strings(unblocker.get("post_manual_release_gate_blockers"))
        )[:20],
        "sd15_release_gap_status": str(sd15.get("status") or unblocker.get("sd15_release_gap_status") or ""),
        "sd15_release_gap_blockers": (
            _strings(sd15.get("blockers")) or _strings(unblocker.get("sd15_release_gap_blockers"))
        )[:20],
        "input_resolution_summary": terminal_input_resolution_summary,
        "external_input_transition_table": terminal_external_input_transition_table,
        "manual_evidence_blocking_summary": terminal_manual_evidence_blocking_summary,
        "natural_load_gate_summary": terminal_natural_load_gate_summary,
        "blocked_actions": _strings(unblocker.get("blocked_actions"))[:20]
        or [
            "do_not_publish_gpu_bubble_release_claim_until_unblocker_summary_clears",
            "do_not_auto_start_gpu_heavy_from_release_unblocker_summary",
            "do_not_treat_unblocker_summary_as_release_evidence",
        ],
    }

    if chain_complete and terminal_blocked:
        terminal_status = "external_input_and_manual_gpu_blocked"
        chain_status = "complete_waiting_external_input"
    elif not chain_complete:
        terminal_status = "json_chain_incomplete"
        chain_status = "json_chain_incomplete"
    else:
        terminal_status = "not_terminal_blocked"
        chain_status = "not_terminal_blocked"

    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": terminal_status,
        "terminal_status": terminal_status,
        "not_release_evidence": True,
        "chain_integrity_status": chain_status,
        "json_chain_complete_but_waiting_external_or_manual_gpu": bool(chain_complete and terminal_blocked),
        "json_only_substantive_progress_available": bool(
            progress_audit["json_only_substantive_progress_available"]
        ),
        "json_only_progress_audit": progress_audit,
        "recommended_next_non_gpu_focus": str(
            readiness.get("recommended_next_non_gpu_focus")
            or unblocker.get("recommended_next_non_gpu_focus")
            or ""
        ),
        "stable_first_release_blocked_by_this_artifact": bool(
            first_release_scope.get("stable_first_release_blocked_by_this_artifact")
            or unblocker.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_allowed": False,
        "gpu_bubble_release_claim_blocked": bool(hard_gates),
        "gpu_bubble_release_hard_gate_ids": hard_gates[:20],
        "recommended_release_policy": recommended_release_policy,
        "claim_publication_scope": str(first_release_scope.get("claim_publication_scope") or ""),
        "does_not_prove_global_product_release": bool(
            first_release_scope.get("does_not_prove_global_product_release")
        ),
        "claim_wording_policy": claim_wording_policy,
        "release_gain_claim_wording_allowed": bool(readiness.get("release_gain_claim_wording_allowed")),
        "forbidden_claim_wording_hit_count": forbidden_claim_wording_hit_count,
        "roadmap_acceptance_gate_summary": {
            "summary_version": _safe_int(roadmap_acceptance.get("summary_version"), 1),
            "roadmap": str(roadmap_acceptance.get("roadmap") or ""),
            "artifact_role": str(roadmap_acceptance.get("artifact_role") or ""),
            "metric_contract_status": str(roadmap_acceptance.get("metric_contract_status") or ""),
            "required_metric_count": _safe_int(roadmap_acceptance.get("required_metric_count")),
            "required_metric_ids": _strings(roadmap_acceptance.get("required_metric_ids"))[:30],
            "experiment_matrix_status": str(roadmap_acceptance.get("experiment_matrix_status") or ""),
            "required_experiment_batch_count": _safe_int(
                roadmap_acceptance.get("required_experiment_batch_count")
            ),
            "required_experiment_batch_ids": _strings(
                roadmap_acceptance.get("required_experiment_batch_ids")
            )[:20],
            "acceptance_gate_status": str(roadmap_acceptance.get("acceptance_gate_status") or ""),
            "required_acceptance_gate_count": _safe_int(
                roadmap_acceptance.get("required_acceptance_gate_count")
            ),
            "required_acceptance_gate_ids": _strings(
                roadmap_acceptance.get("required_acceptance_gate_ids")
            )[:30],
            "blocked_acceptance_gate_count": _safe_int(
                roadmap_acceptance.get("blocked_acceptance_gate_count")
            ),
            "blocked_acceptance_gate_ids": _strings(
                roadmap_acceptance.get("blocked_acceptance_gate_ids")
            )[:30],
            "gpu_bubble_release_hard_gate_ids": _strings(
                roadmap_acceptance.get("gpu_bubble_release_hard_gate_ids")
            )[:20],
            "missing_external_inputs": _strings(roadmap_acceptance.get("missing_external_inputs"))[:20],
            "manual_gpu_evidence_required": bool(
                roadmap_acceptance.get("manual_gpu_evidence_required")
            ),
            "release_claims_rebuild_required": bool(
                roadmap_acceptance.get("release_claims_rebuild_required")
            ),
            "ready_for_recommended_sorting": bool(
                roadmap_acceptance.get("ready_for_recommended_sorting")
            ),
            "ready_for_ui_advisor_stable_strategy": bool(
                roadmap_acceptance.get("ready_for_ui_advisor_stable_strategy")
            ),
            "next_allowed_stage": str(roadmap_acceptance.get("next_allowed_stage") or ""),
            "fail_closed": bool(roadmap_acceptance.get("fail_closed")),
            "not_release_evidence": bool(roadmap_acceptance.get("not_release_evidence")),
            "safe_to_auto_start": bool(roadmap_acceptance.get("safe_to_auto_start")),
            "release_claim_allowed": bool(roadmap_acceptance.get("release_claim_allowed")),
        },
        "roadmap_execution_contract_summary": {
            "summary_version": _safe_int(roadmap_execution.get("summary_version"), 1),
            "roadmap": str(roadmap_execution.get("roadmap") or ""),
            "artifact_role": str(roadmap_execution.get("artifact_role") or ""),
            "attribution_rule_contract_status": str(
                roadmap_execution.get("attribution_rule_contract_status") or ""
            ),
            "attribution_rule_count": _safe_int(roadmap_execution.get("attribution_rule_count")),
            "attribution_rule_ids": _strings(roadmap_execution.get("attribution_rule_ids"))[:20],
            "family_strategy_status": str(roadmap_execution.get("family_strategy_status") or ""),
            "family_strategy_count": _safe_int(roadmap_execution.get("family_strategy_count")),
            "family_strategy_ids": _strings(roadmap_execution.get("family_strategy_ids"))[:20],
            "progression_status": str(roadmap_execution.get("progression_status") or ""),
            "progression_phase_count": _safe_int(roadmap_execution.get("progression_phase_count")),
            "progression_phase_ids": _strings(roadmap_execution.get("progression_phase_ids"))[:20],
            "current_progression_phase_id": str(
                roadmap_execution.get("current_progression_phase_id") or ""
            ),
            "next_allowed_stage": str(roadmap_execution.get("next_allowed_stage") or ""),
            "gpu_bubble_release_hard_gate_ids": _strings(
                roadmap_execution.get("gpu_bubble_release_hard_gate_ids")
            )[:20],
            "missing_external_inputs": _strings(roadmap_execution.get("missing_external_inputs"))[:20],
            "ready_for_combined_strategy": bool(
                roadmap_execution.get("ready_for_combined_strategy")
            ),
            "ready_for_long_training_validation": bool(
                roadmap_execution.get("ready_for_long_training_validation")
            ),
            "ready_for_ui_advisor_stable_strategy": bool(
                roadmap_execution.get("ready_for_ui_advisor_stable_strategy")
            ),
            "fail_closed": bool(roadmap_execution.get("fail_closed")),
            "not_release_evidence": bool(roadmap_execution.get("not_release_evidence")),
            "safe_to_auto_start": bool(roadmap_execution.get("safe_to_auto_start")),
            "release_claim_allowed": bool(roadmap_execution.get("release_claim_allowed")),
        },
        "experiment_matrix_readiness": {
            "summary_version": _safe_int(matrix_readiness.get("summary_version"), 1),
            "roadmap": str(matrix_readiness.get("roadmap") or ""),
            "artifact_role": str(matrix_readiness.get("artifact_role") or ""),
            "matrix_status": str(matrix_readiness.get("matrix_status") or ""),
            "row_count": _safe_int(matrix_readiness.get("row_count")),
            "required_batch_count": _safe_int(matrix_readiness.get("required_batch_count")),
            "required_batch_ids": _strings(matrix_readiness.get("required_batch_ids"))[:20],
            "covered_batch_ids": _strings(matrix_readiness.get("covered_batch_ids"))[:20],
            "missing_batch_ids": _strings(matrix_readiness.get("missing_batch_ids"))[:20],
            "batch_row_counts": dict(_mapping(matrix_readiness.get("batch_row_counts"))),
            "family_row_counts": dict(_mapping(matrix_readiness.get("family_row_counts"))),
            "coverage_state_counts": dict(_mapping(matrix_readiness.get("coverage_state_counts"))),
            "release_hard_gate_ids": _strings(matrix_readiness.get("release_hard_gate_ids"))[:20],
            "current_progression_phase_id": str(
                matrix_readiness.get("current_progression_phase_id") or ""
            ),
            "blocked_reason_ids": _strings(matrix_readiness.get("blocked_reason_ids"))[:30],
            "unsafe_row_count": _safe_int(matrix_readiness.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(matrix_readiness.get("unsafe_row_ids"))[:20],
            "ready_for_release_claim": bool(matrix_readiness.get("ready_for_release_claim")),
            "fail_closed": bool(matrix_readiness.get("fail_closed")),
            "not_release_evidence": bool(matrix_readiness.get("not_release_evidence")),
            "safe_to_auto_start": bool(matrix_readiness.get("safe_to_auto_start")),
            "release_claim_allowed": bool(matrix_readiness.get("release_claim_allowed")),
        },
        "normalized_evidence_gate_mapping": {
            "summary_version": _safe_int(normalized_gate_mapping.get("summary_version"), 1),
            "roadmap": str(normalized_gate_mapping.get("roadmap") or ""),
            "artifact_role": str(normalized_gate_mapping.get("artifact_role") or ""),
            "source_report": str(normalized_gate_mapping.get("source_report") or ""),
            "source_normalized_evidence_count": _safe_int(
                normalized_gate_mapping.get("source_normalized_evidence_count")
            ),
            "mapped_row_count": _safe_int(normalized_gate_mapping.get("mapped_row_count")),
            "unmapped_row_count": _safe_int(normalized_gate_mapping.get("unmapped_row_count")),
            "unmapped_row_ids": _strings(normalized_gate_mapping.get("unmapped_row_ids"))[:20],
            "gate_schema_version": _safe_int(normalized_gate_mapping.get("gate_schema_version"), 1),
            "gate_ids": _strings(normalized_gate_mapping.get("gate_ids"))[:30],
            "gate_state_counts": dict(_mapping(normalized_gate_mapping.get("gate_state_counts"))),
            "release_claim_role_counts": dict(
                _mapping(normalized_gate_mapping.get("release_claim_role_counts"))
            ),
            "family_row_counts": dict(_mapping(normalized_gate_mapping.get("family_row_counts"))),
            "source_kind_counts": dict(_mapping(normalized_gate_mapping.get("source_kind_counts"))),
            "release_hard_gate_ids": _strings(
                normalized_gate_mapping.get("release_hard_gate_ids")
            )[:20],
            "blocked_release_gate_ids": _strings(
                normalized_gate_mapping.get("blocked_release_gate_ids")
            )[:30],
            "missing_metric_ids": _strings(normalized_gate_mapping.get("missing_metric_ids"))[:30],
            "experiment_matrix_row_count": _safe_int(
                normalized_gate_mapping.get("experiment_matrix_row_count")
            ),
            "unsafe_row_count": _safe_int(normalized_gate_mapping.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(normalized_gate_mapping.get("unsafe_row_ids"))[:20],
            "ready_for_release_claim": bool(normalized_gate_mapping.get("ready_for_release_claim")),
            "fail_closed": bool(normalized_gate_mapping.get("fail_closed")),
            "not_release_evidence": bool(normalized_gate_mapping.get("not_release_evidence")),
            "safe_to_auto_start": bool(normalized_gate_mapping.get("safe_to_auto_start")),
            "release_claim_allowed": bool(normalized_gate_mapping.get("release_claim_allowed")),
        },
        "normalized_evidence_gate_explanation_summary": {
            "summary_version": _safe_int(normalized_gate_explanation.get("summary_version"), 1),
            "roadmap": str(normalized_gate_explanation.get("roadmap") or ""),
            "artifact_role": str(normalized_gate_explanation.get("artifact_role") or ""),
            "source_artifact_role": str(
                normalized_gate_explanation.get("source_artifact_role") or ""
            ),
            "source_report": str(normalized_gate_explanation.get("source_report") or ""),
            "mapped_row_count": _safe_int(normalized_gate_explanation.get("mapped_row_count")),
            "gate_schema_version": _safe_int(
                normalized_gate_explanation.get("gate_schema_version"),
                1,
            ),
            "gate_ids": _strings(normalized_gate_explanation.get("gate_ids"))[:30],
            "release_hard_gate_ids": _strings(
                normalized_gate_explanation.get("release_hard_gate_ids")
            )[:20],
            "blocked_release_gate_ids": _strings(
                normalized_gate_explanation.get("blocked_release_gate_ids")
            )[:30],
            "row_outcome_counts": dict(
                _mapping(normalized_gate_explanation.get("row_outcome_counts"))
            ),
            "gate_outcome_counts": dict(
                _mapping(normalized_gate_explanation.get("gate_outcome_counts"))
            ),
            "blocker_explanation_counts": dict(
                _mapping(normalized_gate_explanation.get("blocker_explanation_counts"))
            ),
            "missing_metric_counts": dict(
                _mapping(normalized_gate_explanation.get("missing_metric_counts"))
            ),
            "release_hard_gate_row_counts": dict(
                _mapping(normalized_gate_explanation.get("release_hard_gate_row_counts"))
            ),
            "gate_outcome_row_count": _safe_int(
                normalized_gate_explanation.get("gate_outcome_row_count")
            ),
            "gate_outcome_rows": [
                dict(_mapping(item))
                for item in _list(normalized_gate_explanation.get("gate_outcome_rows"))[:30]
            ],
            "blocker_explanation_row_count": _safe_int(
                normalized_gate_explanation.get("blocker_explanation_row_count")
            ),
            "top_blocker_explanations": [
                dict(_mapping(item))
                for item in _list(normalized_gate_explanation.get("top_blocker_explanations"))[:20]
            ],
            "unsafe_row_count": _safe_int(normalized_gate_explanation.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(normalized_gate_explanation.get("unsafe_row_ids"))[:20],
            "ready_for_release_claim": bool(
                normalized_gate_explanation.get("ready_for_release_claim")
            ),
            "fail_closed": bool(normalized_gate_explanation.get("fail_closed")),
            "not_release_evidence": bool(
                normalized_gate_explanation.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(normalized_gate_explanation.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                normalized_gate_explanation.get("release_claim_allowed")
            ),
        },
        "next_action_machine_summary": terminal_next_action_machine_summary,
        "next_action_contract_summary": terminal_next_action_contract_summary,
        "remaining_work_summary": terminal_remaining_work_summary,
        "first_release_policy_summary": terminal_first_release_policy_summary,
        "source_axis_freshness_dedupe_audit": terminal_source_axis_freshness_summary,
        "source_axis_requirement_summary": {
            "summary_version": _safe_int(source_axis_requirement_summary.get("summary_version"), 1),
            "roadmap": str(source_axis_requirement_summary.get("roadmap") or ""),
            "artifact_role": str(source_axis_requirement_summary.get("artifact_role") or ""),
            "report": str(source_axis_requirement_summary.get("report") or ""),
            "status": str(source_axis_requirement_summary.get("status") or ""),
            "family_count": _safe_int(source_axis_requirement_summary.get("family_count")),
            "external_input_required_count": _safe_int(
                source_axis_requirement_summary.get("external_input_required_count")
            ),
            "candidate_available_family_count": _safe_int(
                source_axis_requirement_summary.get("candidate_available_family_count")
            ),
            "exhausted_family_count": _safe_int(source_axis_requirement_summary.get("exhausted_family_count")),
            "no_ready_source_axis_family_count": _safe_int(
                source_axis_requirement_summary.get("no_ready_source_axis_family_count")
            ),
            "completed_existing_command_count": _safe_int(
                source_axis_requirement_summary.get("completed_existing_command_count")
            ),
            "external_input_required": bool(source_axis_requirement_summary.get("external_input_required")),
            "fail_closed": bool(source_axis_requirement_summary.get("fail_closed")),
            "not_release_evidence": bool(source_axis_requirement_summary.get("not_release_evidence")),
            "safe_to_auto_start": bool(source_axis_requirement_summary.get("safe_to_auto_start")),
            "release_claim_allowed": bool(source_axis_requirement_summary.get("release_claim_allowed")),
            "does_not_run_training": bool(source_axis_requirement_summary.get("does_not_run_training")),
            "does_not_run_cuda": bool(source_axis_requirement_summary.get("does_not_run_cuda")),
        },
        "manual_review_queue_summary": terminal_manual_review_queue_summary,
        "manual_review_artifact_chain_summary": _compact_summary_mirror(
            manual_review_artifact_chain,
            string_fields=["status"],
            int_fields=[
                "expected_artifact_count",
                "present_artifact_count",
                "missing_artifact_count",
                "manual_review_ready_count",
                "closed_blocked_or_regression_count",
                "closed_diagnostic_or_promotion_count",
                "followup_gpu_required_action_count",
                "unsafe_artifact_count",
            ],
            list_fields=[
                "present_artifact_ids",
                "missing_artifact_ids",
                "unsafe_artifact_ids",
                "blocked_actions",
            ],
            nested_fields=["artifact_status_counts"],
        ),
        "sdxl_diagnostic_artifact_chain_summary": _compact_summary_mirror(
            sdxl_diagnostic_artifact_chain,
            string_fields=[
                "status",
                "probe_status",
                "debug_repeat_status",
                "manual_gpu_queue_status",
            ],
            int_fields=[
                "expected_artifact_count",
                "present_artifact_count",
                "missing_artifact_count",
                "probe_group_count",
                "probe_rollback_group_count",
                "probe_pending_group_count",
                "debug_repeat_candidate_pass_count",
                "debug_repeat_fully_repeated_candidate_count",
                "debug_repeat_missing_report_count",
                "debug_repeat_missing_summary_count",
                "debug_repeat_execution_failure_count",
                "manual_gpu_queue_item_count",
                "manual_gpu_completed_group_count",
                "manual_gpu_pending_ready_command_count",
                "manual_gpu_completed_summary_count",
                "manual_gpu_missing_summary_count",
                "unsafe_artifact_count",
            ],
            list_fields=[
                "present_artifact_ids",
                "missing_artifact_ids",
                "unsafe_artifact_ids",
                "blocked_actions",
            ],
            nested_fields=["artifact_status_counts"],
        ),
        "newbie_blockskip_compute_bound_artifact_chain_summary": _compact_summary_mirror(
            newbie_blockskip_compute_bound_chain,
            string_fields=[
                "status",
                "followup_status",
                "quality_status",
                "quality_drift_status",
                "quality_drift_quality_review_type",
                "loss_curve_status",
                "loss_curve_review_type",
                "quality_semantic_status",
                "quality_semantic_review_type",
                "semantics_status",
                "policy_status",
            ],
            bool_fields=[
                "quality_throughput_repeat_ready",
                "quality_loss_quality_ready",
                "quality_gate_blocked",
                "quality_drift_review_ready",
                "quality_drift_quality_evidence_present",
                "quality_drift_loss_curve_ready",
                "quality_drift_shape_stable",
                "quality_drift_disabled_parity_ok",
                "quality_drift_checkpoint_semantics_ok",
                "quality_drift_residual_reuse_parity_ok",
                "quality_drift_loss_gate_blocked",
                "quality_drift_semantic_ready",
                "quality_drift_review_blocked",
                "loss_curve_review_ready",
                "loss_curve_shape_stable",
                "loss_curve_disabled_parity_ok",
                "loss_curve_checkpoint_semantics_ok",
                "loss_curve_residual_reuse_parity_ok",
                "loss_curve_nonrelease_ready",
                "quality_semantic_review_ready",
                "quality_semantic_semantic_ready",
                "quality_semantic_loss_curve_ready",
                "quality_semantic_cached_token_ab_ready",
                "quality_semantic_shape_stable",
                "quality_semantic_disabled_parity_ok",
                "quality_semantic_checkpoint_semantics_ok",
                "quality_semantic_residual_reuse_parity_ok",
                "quality_semantic_cpu_replay_only",
                "quality_semantic_default_behavior_changed",
                "quality_semantic_runtime_activation_enabled",
                "quality_semantic_trainer_wiring_allowed",
                "quality_semantic_nonrelease_ready",
                "semantics_review_ready",
                "semantics_reviewed_blocked",
                "policy_ready",
                "policy_compute_bound_exception_allowed",
                "policy_natural_load_gate_exit_allowed",
                "policy_blockskip_counts_as_release_evidence",
                "policy_defined_blocked",
            ],
            int_fields=[
                "expected_artifact_count",
                "present_artifact_count",
                "missing_artifact_count",
                "followup_complete_pair_count",
                "followup_pending_pair_count",
                "followup_executed_count",
                "followup_execution_failure_count",
                "followup_manual_start_required_count",
                "quality_completed_seed_pair_count",
                "loss_curve_pair_count",
                "loss_curve_ready_pair_count",
                "quality_semantic_pair_count",
                "quality_semantic_checkpoint_semantics_pair_count",
                "unsafe_artifact_count",
                "blocker_count",
            ],
            list_fields=[
                "present_artifact_ids",
                "missing_artifact_ids",
                "semantics_blocked_families",
                "unsafe_artifact_ids",
                "blocked_actions",
            ],
            nested_fields=["artifact_status_counts"],
        ),
        "newbie_compute_diagnosis_artifact_chain_summary": _compact_summary_mirror(
            newbie_compute_diagnosis_chain,
            string_fields=["status", "diagnosis_status", "diagnosis_report", "family"],
            bool_fields=[
                "data_wait_route_exhausted",
                "fail_closed",
                "not_release_evidence",
                "does_not_run_training",
                "does_not_run_cuda",
                "does_not_run_gpu_heavy",
                "safe_to_auto_start",
                "release_claim_allowed",
            ],
            int_fields=[
                "analyzed_probe_count",
                "low_data_wait_probe_count",
                "compute_bound_probe_count",
                "dataloader_rebuild_observed_count",
                "natural_candidate_count",
                "train_step_compute_substage_profile_available_count",
                "newbie_backward_op_profile_available_count",
                "newbie_backward_shape_profile_available_count",
                "newbie_module_timing_profile_available_count",
                "unsafe_artifact_count",
            ],
            list_fields=["next_focus_ids", "unsafe_artifact_ids", "blocked_actions"],
            nested_fields=[
                "dominant_bottleneck_counts",
                "dominant_train_step_substage_counts",
                "newbie_backward_top_op_counts",
                "newbie_backward_top_matmul_shape_counts",
                "newbie_module_timing_top_group_counts",
            ],
        ),
        "newbie_tail8_attention_compute_review_summary": _compact_summary_mirror(
            newbie_tail8_attention_compute_review,
            string_fields=[
                "status",
                "review_status",
                "review_report",
                "family",
                "candidate",
                "source_file_presence_status",
            ],
            bool_fields=[
                "repeat_evidence_complete",
                "throughput_repeat_ready",
                "loss_curve_quality_ready",
                "release_claim_eligible",
                "fail_closed",
                "not_release_evidence",
                "publishable",
                "does_not_run_training",
                "does_not_run_cuda",
                "does_not_run_gpu_heavy",
                "safe_to_auto_start",
                "release_claim_allowed",
            ],
            int_fields=[
                "completed_seed_pair_count",
                "required_seed_pair_count",
                "blocker_count",
                "unsafe_artifact_count",
                "source_file_count",
                "existing_source_file_count",
                "missing_source_file_count",
            ],
            list_fields=[
                "blockers",
                "blocked_release_reasons",
                "recommended_next_actions",
                "unsafe_artifact_ids",
                "blocked_actions",
                "missing_source_file_pair_ids",
                "missing_source_file_ids",
            ],
            nested_fields=["target_depth_progression_summary"],
        ),
        "newbie_tail8_forward_anomaly_review_summary": _compact_summary_mirror(
            newbie_tail8_forward_anomaly_review,
            string_fields=[
                "status",
                "review_status",
                "report",
                "family",
                "candidate",
                "candidate_status",
                "root_cause_confidence",
            ],
            bool_fields=[
                "review_ready",
                "candidate_run_present",
                "tail8_seed1337_reference_present",
                "layer0_seed2027_baseline_present",
                "comparison_source_ready",
                "candidate_incomplete",
                "forward_runtime_anomaly",
                "low_data_wait",
                "natural_load_or_dataloader_regression_evidence",
                "counts_as_tail8_repeat_pair",
                "root_cause_proven",
                "fail_closed",
                "not_release_evidence",
                "publishable",
                "does_not_run_training",
                "does_not_run_cuda",
                "does_not_run_gpu_heavy",
                "safe_to_auto_start",
                "release_claim_allowed",
            ],
            int_fields=[
                "comparison_source_present_count",
                "comparison_source_missing_count",
                "candidate_global_step",
                "candidate_total_steps",
                "blocker_count",
                "unsafe_artifact_count",
            ],
            list_fields=[
                "missing_source_manifest_ids",
                "blockers",
                "recommended_next_actions",
                "unsafe_artifact_ids",
                "blocked_actions",
            ],
        ),
        "newbie_tail8_seed2027_rerun_preflight_summary": _compact_summary_mirror(
            newbie_tail8_seed2027_rerun_preflight,
            string_fields=[
                "report",
                "status",
                "family",
                "candidate",
                "candidate_status",
                "candidate_run_status",
                "candidate_run_id",
                "reference_run_id",
                "baseline_run_id",
                "planned_out_dir",
                "manual_runner",
                "request_config_digest_excluding_seed",
                "case_config_contract_id",
                "case_config_contract_digest_excluding_seed",
                "forward_anomaly_review_status",
            ],
            bool_fields=[
                "review_ready",
                "candidate_incomplete",
                "forward_runtime_anomaly",
                "low_data_wait",
                "forward_anomaly_review_ready",
                "forward_anomaly_candidate_run_present",
                "forward_anomaly_tail8_seed1337_reference_present",
                "forward_anomaly_layer0_seed2027_baseline_present",
                "forward_anomaly_comparison_source_ready",
                "candidate_incomplete_forward_anomaly_ready",
                "manual_rerun_ready",
                "gpu_idle_ready",
                "gpu_summary_available",
                "compute_apps_present",
                "gpu_compute_apps_present",
                "environment_snapshot_ready",
                "disk_space_ready",
                "compute_apps_probe_present",
                "compute_apps_probe_command_present",
                "compute_apps_probe_query_ok",
                "compute_apps_probe_inspection_ready",
                "compute_apps_probe_permission_denied",
                "compute_apps_probe_proof_ready",
                "compute_apps_probe_proof_probe_present",
                "compute_apps_probe_proof_inspection_ready",
                "compute_apps_probe_proof_permission_denied",
                "compute_apps_probe_proof_explicit_empty_result",
                "compute_apps_classification_explicit_empty_result",
                "candidate_release_claim_allowed",
                "candidate_safe_to_auto_start",
                "case_config_contract_ready",
                "fail_closed",
                "not_release_evidence",
                "publishable",
                "release_claim_allowed",
                "safe_to_auto_start",
                "does_not_run_training",
                "does_not_run_cuda",
                "does_not_run_gpu_heavy",
            ],
            int_fields=[
                "gpu_compute_apps_count",
                "gpu_sample_count",
                "gpu_valid_sample_count",
                "gpu_sample_error_count",
                "compute_apps_probe_row_count",
                "compute_apps_probe_raw_line_count",
                "compute_apps_probe_proof_permission_unknown_count",
                "compute_apps_probe_proof_row_count",
                "compute_apps_probe_proof_raw_line_count",
                "compute_apps_blocking_compute_like_count",
                "compute_apps_background_gpu_client_count",
                "compute_apps_unknown_gpu_client_count",
                "forward_anomaly_comparison_source_present_count",
                "forward_anomaly_comparison_source_missing_count",
                "candidate_run_global_step",
                "candidate_run_total_steps",
                "blocker_count",
                "unsafe_artifact_count",
            ],
            number_fields=[
                "gpu_util_pct_mean",
                "gpu_util_pct_p95",
                "gpu_util_pct_max",
                "gpu_active_sample_ratio",
                "gpu_idle_sample_ratio",
                "gpu_memory_util_pct_mean",
                "disk_free_gb",
                "min_disk_free_gb",
                "candidate_run_data_wait_share",
                "candidate_run_forward_model_execution_mean_ms",
                "candidate_run_newbie_transformer_smoke_seconds",
            ],
            list_fields=[
                "blockers",
                "recommended_next_actions",
                "forward_anomaly_missing_source_manifest_ids",
                "manual_execute_command",
                "manual_dry_run_command",
                "post_rerun_refresh_sequence",
                "expected_post_rerun_outputs",
                "unsafe_artifact_ids",
            ],
            nested_fields=[
                "candidate_diagnosis",
                "gpu_summary",
                "compute_apps_probe",
                "compute_apps_probe_proof",
                "compute_apps_classification_summary",
                "tail8_manual_rerun_envelope",
                "resource_summary",
                "environment_snapshot",
            ],
        ),
        "protected_followup_gpu_queue_summary": terminal_protected_followup_gpu_queue_summary,
        "source_artifact_inventory_summary": _compact_summary_mirror(
            source_artifact_inventory,
            string_fields=["status", "execution_policy"],
            int_fields=[
                "source_artifact_count",
                "present_source_artifact_count",
                "missing_source_artifact_count",
                "load_error_count",
                "roadmap_mismatch_count",
                "release_unsafe_count",
                "publishable_count",
                "not_release_evidence_false_count",
                "does_not_run_training_false_count",
                "does_not_run_cuda_false_count",
                "unique_report_count",
                "unsafe_source_artifact_count",
            ],
            list_fields=[
                "missing_source_artifact_ids",
                "load_error_source_artifact_ids",
                "roadmap_mismatch_source_artifact_ids",
                "release_unsafe_source_artifact_ids",
                "unsafe_source_artifact_ids",
            ],
            nested_fields=["status_counts", "report_counts"],
        ),
        "evidence_summary_inventory_summary": _compact_summary_mirror(
            evidence_summary_inventory,
            string_fields=["status", "execution_policy"],
            int_fields=[
                "evidence_key_count",
                "mapping_entry_count",
                "scalar_entry_count",
                "list_entry_count",
                "other_entry_count",
                "roadmap_mismatch_entry_count",
                "release_unsafe_entry_count",
                "publishable_entry_count",
                "release_gain_wording_allowed_count",
                "not_release_evidence_false_entry_count",
                "safe_to_auto_start_true_entry_count",
                "release_claim_allowed_true_entry_count",
                "unsafe_evidence_entry_count",
            ],
            list_fields=[
                "roadmap_mismatch_entry_ids",
                "release_unsafe_entry_ids",
                "unsafe_evidence_entry_ids",
            ],
            nested_fields=["entry_type_counts"],
        ),
        "source_and_downstream_artifact_contract_summary": terminal_source_downstream_contract_summary,
        "remaining_release_blocker_matrix_summary": terminal_blocker_matrix_summary,
        "remaining_blocker_resolution_handoff_summary": terminal_blocker_handoff_summary,
        "remaining_action_dependency_graph_summary": terminal_action_dependency_graph_summary,
        "remaining_action_unblock_sequence_summary": terminal_action_unblock_sequence_summary,
        "remaining_blocker_artifact_presence_summary": terminal_blocker_presence_summary,
        "release_claim_exit_criteria_summary": terminal_release_exit_summary,
        "release_gate_input_dependency_summary": terminal_release_gate_input_dependency_summary,
        "release_gate_post_input_refresh_plan_summary": (
            terminal_release_gate_post_input_refresh_plan_summary
        ),
        "release_gate_input_detection_source_summary": (
            terminal_release_gate_input_detection_source_summary
        ),
        "release_gate_input_acceptance_criteria_summary": (
            terminal_release_gate_input_acceptance_criteria_summary
        ),
        "release_gate_input_refresh_readiness_summary": (
            terminal_release_gate_input_refresh_readiness_summary
        ),
        "release_gate_input_refresh_blocker_summary": (
            terminal_release_gate_input_refresh_blocker_summary
        ),
        "release_gate_input_lifecycle_summary": (
            terminal_release_gate_input_lifecycle_summary
        ),
        "external_input_release_gate_alignment_summary": (
            terminal_external_input_release_gate_alignment_summary
        ),
        "release_gate_post_input_refresh_command_surface_summary": (
            terminal_release_gate_post_input_refresh_command_surface_summary
        ),
        "release_gate_post_input_refresh_sequence_integrity_summary": (
            terminal_release_gate_post_input_refresh_sequence_integrity_summary
        ),
        "release_gate_post_input_refresh_terminal_guard_dependency_summary": (
            terminal_release_gate_post_input_refresh_terminal_guard_dependency_summary
        ),
        "release_gate_post_input_refresh_artifact_coverage_summary": (
            terminal_release_gate_post_input_refresh_artifact_coverage_summary
        ),
        "release_gate_post_input_refresh_command_artifact_link_summary": (
            terminal_release_gate_post_input_refresh_command_artifact_link_summary
        ),
        "release_gate_post_input_refresh_guard_consumption_summary": (
            terminal_release_gate_post_input_refresh_guard_consumption_summary
        ),
        "release_gate_post_input_refresh_guard_report_acceptance_summary": (
            terminal_release_gate_post_input_refresh_guard_report_acceptance_summary
        ),
        "external_input_json_refresh_runner_manifest_summary": (
            terminal_external_input_json_refresh_runner_manifest_summary
        ),
        "manual_protected_gpu_command_surface_summary": terminal_command_surface_summary,
        "protected_followup_run_plan_artifact_chain_summary": _compact_summary_mirror(
            protected_run_plan_chain,
            string_fields=["status", "execution_policy"],
            int_fields=[
                "expected_artifact_count",
                "present_artifact_count",
                "missing_artifact_count",
                "total_command_count",
                "manual_start_required_command_count",
                "release_claim_allowed_after_success_command_count",
                "unsafe_command_count",
                "unsafe_scaffold_count",
                "contract_ok_artifact_count",
                "unsafe_artifact_count",
            ],
            list_fields=[
                "present_artifact_ids",
                "missing_artifact_ids",
                "unsafe_artifact_ids",
            ],
            nested_fields=["artifact_status_counts"],
        ),
        "forbidden_claim_wording_audit": {
            "claim_wording_policy": claim_wording_policy,
            "release_gain_claim_wording_allowed": bool(readiness.get("release_gain_claim_wording_allowed")),
            "forbidden_claim_wording_hit_count": forbidden_claim_wording_hit_count,
            "forbidden_claim_wording_hits": [
                dict(_mapping(item)) for item in _list(claim_wording_audit.get("forbidden_claim_wording_hits"))
            ],
            "not_release_evidence": bool(claim_wording_audit.get("not_release_evidence")),
            "safe_to_auto_start": bool(claim_wording_audit.get("safe_to_auto_start")),
            "release_claim_allowed": bool(claim_wording_audit.get("release_claim_allowed")),
        },
        "release_unblocker_summary": terminal_release_unblocker,
        "external_input_transition_table": terminal_external_input_transition_table,
        "missing_external_inputs": missing_inputs[:20],
        "manual_gpu_required_inputs": manual_inputs[:20],
        "current_gpu_heavy_action_count": _safe_int(remaining.get("current_gpu_heavy_action_count")),
        "followup_gpu_required_action_count": _safe_int(remaining.get("followup_gpu_required_action_count")),
        "manual_review_ready_count": _safe_int(
            _mapping(_mapping(readiness.get("evidence_summary")).get("manual_review_queue_summary")).get(
                "manual_review_ready_count"
            )
        ),
        "manual_gpu_execution_summary": {
            "summary_version": _safe_int(manual_gpu_execution.get("summary_version"), 1),
            "gpu_related_action_count": _safe_int(manual_gpu_execution.get("gpu_related_action_count")),
            "gpu_related_action_ids": _strings(
                manual_gpu_execution.get("gpu_related_action_ids")
            )[:20],
            "current_gpu_heavy_action_count": _safe_int(
                manual_gpu_execution.get("current_gpu_heavy_action_count"),
                _safe_int(remaining.get("current_gpu_heavy_action_count")),
            ),
            "current_gpu_heavy_action_ids": _strings(
                manual_gpu_execution.get("current_gpu_heavy_action_ids")
            )[:20],
            "followup_gpu_required_action_count": _safe_int(
                manual_gpu_execution.get("followup_gpu_required_action_count"),
                _safe_int(remaining.get("followup_gpu_required_action_count")),
            ),
            "followup_gpu_required_action_ids": _strings(
                manual_gpu_execution.get("followup_gpu_required_action_ids")
            )[:20],
            "protected_manual_gpu_ready_action_count": _safe_int(
                manual_gpu_execution.get("protected_manual_gpu_ready_action_count")
            ),
            "protected_manual_gpu_ready_action_ids": _strings(
                manual_gpu_execution.get("protected_manual_gpu_ready_action_ids")
            )[:20],
            "blocked_missing_prerequisite_gpu_action_count": _safe_int(
                manual_gpu_execution.get("blocked_missing_prerequisite_gpu_action_count")
            ),
            "blocked_missing_prerequisite_gpu_action_ids": _strings(
                manual_gpu_execution.get("blocked_missing_prerequisite_gpu_action_ids")
            )[:20],
            "waiting_manual_gpu_evidence_action_count": _safe_int(
                manual_gpu_execution.get("waiting_manual_gpu_evidence_action_count")
            ),
            "waiting_manual_gpu_evidence_action_ids": _strings(
                manual_gpu_execution.get("waiting_manual_gpu_evidence_action_ids")
            )[:20],
            "manual_start_required_action_count": _safe_int(
                manual_gpu_execution.get("manual_start_required_action_count")
            ),
            "manual_start_required_action_ids": _strings(
                manual_gpu_execution.get("manual_start_required_action_ids")
            )[:20],
            "auto_startable_gpu_action_count": _safe_int(
                manual_gpu_execution.get("auto_startable_gpu_action_count")
            ),
            "auto_startable_gpu_action_ids": _strings(
                manual_gpu_execution.get("auto_startable_gpu_action_ids")
            )[:20],
            "release_claim_allowed_after_success_action_count": _safe_int(
                manual_gpu_execution.get("release_claim_allowed_after_success_action_count")
            ),
            "release_claim_allowed_after_success_action_ids": _strings(
                manual_gpu_execution.get("release_claim_allowed_after_success_action_ids")
            )[:20],
            "execution_policy": str(manual_gpu_execution.get("execution_policy") or ""),
            "safe_to_auto_start": bool(manual_gpu_execution.get("safe_to_auto_start")),
            "release_claim_allowed": bool(manual_gpu_execution.get("release_claim_allowed")),
            "blocked_actions": _strings(manual_gpu_execution.get("blocked_actions"))[:20],
        },
        "source_cache_axis_pipeline_readiness_summary": {
            "summary_version": _safe_int(
                source_cache_axis_pipeline_readiness_summary.get("summary_version"), 1
            ),
            "roadmap": str(source_cache_axis_pipeline_readiness_summary.get("roadmap") or ""),
            "artifact_role": str(
                source_cache_axis_pipeline_readiness_summary.get("artifact_role") or ""
            ),
            "report": str(source_cache_axis_pipeline_readiness_summary.get("report") or ""),
            "status": str(source_cache_axis_pipeline_readiness_summary.get("status") or ""),
            "axis_readiness_status": str(
                source_cache_axis_pipeline_readiness_summary.get("axis_readiness_status") or ""
            ),
            "pipeline_complete": bool(
                source_cache_axis_pipeline_readiness_summary.get("pipeline_complete")
            ),
            "external_input_required": bool(
                source_cache_axis_pipeline_readiness_summary.get("external_input_required")
            ),
            "preflight_admitted": bool(
                source_cache_axis_pipeline_readiness_summary.get("preflight_admitted")
            ),
            "manual_canary_plan_ready": bool(
                source_cache_axis_pipeline_readiness_summary.get("manual_canary_plan_ready")
            ),
            "waiting_external_input": bool(
                source_cache_axis_pipeline_readiness_summary.get("waiting_external_input")
            ),
            "duplicate_or_stale_axis_blocked": bool(
                source_cache_axis_pipeline_readiness_summary.get("duplicate_or_stale_axis_blocked")
            ),
            "cache_axis_not_ready": bool(
                source_cache_axis_pipeline_readiness_summary.get("cache_axis_not_ready")
            ),
            "stage_count": _safe_int(source_cache_axis_pipeline_readiness_summary.get("stage_count")),
            "stage_ok_count": _safe_int(
                source_cache_axis_pipeline_readiness_summary.get("stage_ok_count")
            ),
            "stage_freshness_checked": bool(
                source_cache_axis_pipeline_readiness_summary.get("stage_freshness_checked")
            ),
            "stage_freshness_ok": bool(
                source_cache_axis_pipeline_readiness_summary.get("stage_freshness_ok")
            ),
            "stale_stage_count": _safe_int(
                source_cache_axis_pipeline_readiness_summary.get("stale_stage_count")
            ),
            "stale_stage_ids": _strings(
                source_cache_axis_pipeline_readiness_summary.get("stale_stage_ids")
            )[:20],
            "stale_ready_stage_count": _safe_int(
                source_cache_axis_pipeline_readiness_summary.get("stale_ready_stage_count")
            ),
            "stale_ready_stage_ids": _strings(
                source_cache_axis_pipeline_readiness_summary.get("stale_ready_stage_ids")
            )[:20],
            "ready_stage_freshness_ok": bool(
                source_cache_axis_pipeline_readiness_summary.get("ready_stage_freshness_ok")
            ),
            "blocker_count": _safe_int(source_cache_axis_pipeline_readiness_summary.get("blocker_count")),
            "next_action_count": _safe_int(
                source_cache_axis_pipeline_readiness_summary.get("next_action_count")
            ),
            "fail_closed": bool(source_cache_axis_pipeline_readiness_summary.get("fail_closed")),
            "not_release_evidence": bool(
                source_cache_axis_pipeline_readiness_summary.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(
                source_cache_axis_pipeline_readiness_summary.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                source_cache_axis_pipeline_readiness_summary.get("release_claim_allowed")
            ),
            "does_not_run_training": bool(
                source_cache_axis_pipeline_readiness_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(
                source_cache_axis_pipeline_readiness_summary.get("does_not_run_cuda")
            ),
        },
        "external_input_admission_summary": {
            "summary_version": _safe_int(external_input_admission_summary.get("summary_version"), 1),
            "roadmap": str(external_input_admission_summary.get("roadmap") or ""),
            "artifact_role": str(external_input_admission_summary.get("artifact_role") or ""),
            "report": str(external_input_admission_summary.get("report") or ""),
            "status": str(external_input_admission_summary.get("status") or ""),
            "external_input_required": bool(
                external_input_admission_summary.get("external_input_required")
            ),
            "sd15_status": str(external_input_admission_summary.get("sd15_status") or ""),
            "sd15_checkpoint_exists": bool(
                external_input_admission_summary.get("sd15_checkpoint_exists")
            ),
            "sd15_evidence_ready": bool(
                external_input_admission_summary.get("sd15_evidence_ready")
            ),
            "source_axis_status": str(
                external_input_admission_summary.get("source_axis_status") or ""
            ),
            "source_axis_external_input_required_count": _safe_int(
                external_input_admission_summary.get("source_axis_external_input_required_count")
            ),
            "source_axis_candidate_available_family_count": _safe_int(
                external_input_admission_summary.get("source_axis_candidate_available_family_count")
            ),
            "fail_closed": bool(external_input_admission_summary.get("fail_closed")),
            "not_release_evidence": bool(
                external_input_admission_summary.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(external_input_admission_summary.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                external_input_admission_summary.get("release_claim_allowed")
            ),
            "does_not_run_training": bool(
                external_input_admission_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(external_input_admission_summary.get("does_not_run_cuda")),
        },
        "external_input_intake_registry_summary": {
            "summary_version": _safe_int(
                external_input_intake_registry_summary.get("summary_version"), 1
            ),
            "roadmap": str(external_input_intake_registry_summary.get("roadmap") or ""),
            "artifact_role": str(external_input_intake_registry_summary.get("artifact_role") or ""),
            "report": str(external_input_intake_registry_summary.get("report") or ""),
            "status": str(external_input_intake_registry_summary.get("status") or ""),
            "external_input_detected": bool(
                external_input_intake_registry_summary.get("external_input_detected")
            ),
            "external_input_required": bool(
                external_input_intake_registry_summary.get("external_input_required")
            ),
            "publishable": bool(external_input_intake_registry_summary.get("publishable")),
            "sd15_checkpoint_exists": bool(
                external_input_intake_registry_summary.get("sd15_checkpoint_exists")
            ),
            "source_new_root_count": _safe_int(
                external_input_intake_registry_summary.get("source_new_root_count")
            ),
            "intake_item_count": _safe_int(
                external_input_intake_registry_summary.get("intake_item_count")
            ),
            "missing_external_input_count": _safe_int(
                external_input_intake_registry_summary.get("missing_external_input_count")
            ),
            "registration_slot_count": _safe_int(
                external_input_intake_registry_summary.get("registration_slot_count")
            ),
            "rescan_request_count": _safe_int(
                external_input_intake_registry_summary.get("rescan_request_count")
            ),
            "fail_closed": bool(external_input_intake_registry_summary.get("fail_closed")),
            "not_release_evidence": bool(
                external_input_intake_registry_summary.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(
                external_input_intake_registry_summary.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                external_input_intake_registry_summary.get("release_claim_allowed")
            ),
            "does_not_run_training": bool(
                external_input_intake_registry_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(
                external_input_intake_registry_summary.get("does_not_run_cuda")
            ),
        },
        "external_input_replay_plan_summary": {
            "summary_version": _safe_int(external_input_replay_plan_summary.get("summary_version"), 1),
            "roadmap": str(external_input_replay_plan_summary.get("roadmap") or ""),
            "artifact_role": str(external_input_replay_plan_summary.get("artifact_role") or ""),
            "report": str(external_input_replay_plan_summary.get("report") or ""),
            "status": str(external_input_replay_plan_summary.get("status") or ""),
            "external_input_detected": bool(
                external_input_replay_plan_summary.get("external_input_detected")
            ),
            "sd15_checkpoint_exists": bool(
                external_input_replay_plan_summary.get("sd15_checkpoint_exists")
            ),
            "new_source_root_count": _safe_int(
                external_input_replay_plan_summary.get("new_source_root_count")
            ),
            "command_count": _safe_int(external_input_replay_plan_summary.get("command_count")),
            "ready_command_count": _safe_int(
                external_input_replay_plan_summary.get("ready_command_count")
            ),
            "template_command_count": _safe_int(
                external_input_replay_plan_summary.get("template_command_count")
            ),
            "publishable": bool(external_input_replay_plan_summary.get("publishable")),
            "fail_closed": bool(external_input_replay_plan_summary.get("fail_closed")),
            "not_release_evidence": bool(
                external_input_replay_plan_summary.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(
                external_input_replay_plan_summary.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                external_input_replay_plan_summary.get("release_claim_allowed")
            ),
            "does_not_run_training": bool(
                external_input_replay_plan_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(external_input_replay_plan_summary.get("does_not_run_cuda")),
        },
        "external_input_handoff_packet_summary": {
            "summary_version": _safe_int(external_input_handoff_packet_summary.get("summary_version"), 1),
            "roadmap": str(external_input_handoff_packet_summary.get("roadmap") or ""),
            "artifact_role": str(external_input_handoff_packet_summary.get("artifact_role") or ""),
            "report": str(external_input_handoff_packet_summary.get("report") or ""),
            "status": str(external_input_handoff_packet_summary.get("status") or ""),
            "external_input_detected": bool(
                external_input_handoff_packet_summary.get("external_input_detected")
            ),
            "external_input_required": bool(
                external_input_handoff_packet_summary.get("external_input_required")
            ),
            "missing_external_input_count": _safe_int(
                external_input_handoff_packet_summary.get("missing_external_input_count")
            ),
            "missing_external_inputs": _strings(
                external_input_handoff_packet_summary.get("missing_external_inputs")
            )[:20],
            "input_lifecycle_status": str(
                external_input_handoff_packet_summary.get("input_lifecycle_status") or ""
            ),
            "detected_input_count": _safe_int(
                external_input_handoff_packet_summary.get("detected_input_count")
            ),
            "accepted_input_count": _safe_int(
                external_input_handoff_packet_summary.get("accepted_input_count")
            ),
            "detected_unaccepted_input_count": _safe_int(
                external_input_handoff_packet_summary.get("detected_unaccepted_input_count")
            ),
            "pending_input_count": _safe_int(
                external_input_handoff_packet_summary.get("pending_input_count")
            ),
            "detected_input_ids": _strings(
                external_input_handoff_packet_summary.get("detected_input_ids")
            )[:20],
            "accepted_input_ids": _strings(
                external_input_handoff_packet_summary.get("accepted_input_ids")
            )[:20],
            "detected_unaccepted_input_ids": _strings(
                external_input_handoff_packet_summary.get("detected_unaccepted_input_ids")
            )[:20],
            "pending_input_ids": _strings(
                external_input_handoff_packet_summary.get("pending_input_ids")
            )[:20],
            "handoff_step_count": _safe_int(
                external_input_handoff_packet_summary.get("handoff_step_count")
            ),
            "registration_slot_count": _safe_int(
                external_input_handoff_packet_summary.get("registration_slot_count")
            ),
            "command_count": _safe_int(external_input_handoff_packet_summary.get("command_count")),
            "ready_command_count": _safe_int(
                external_input_handoff_packet_summary.get("ready_command_count")
            ),
            "blocked_command_count": _safe_int(
                external_input_handoff_packet_summary.get("blocked_command_count")
            ),
            "unsafe_command_count": _safe_int(
                external_input_handoff_packet_summary.get("unsafe_command_count")
            ),
            "replay_status": str(external_input_handoff_packet_summary.get("replay_status") or ""),
            "replay_command_count": _safe_int(
                external_input_handoff_packet_summary.get("replay_command_count")
            ),
            "replay_ready_command_count": _safe_int(
                external_input_handoff_packet_summary.get("replay_ready_command_count")
            ),
            "sd15_checkpoint_required": bool(
                external_input_handoff_packet_summary.get("sd15_checkpoint_required")
            ),
            "source_or_cache_axis_required": bool(
                external_input_handoff_packet_summary.get("source_or_cache_axis_required")
            ),
            "warm_cache_or_caption_repair_required": bool(
                external_input_handoff_packet_summary.get("warm_cache_or_caption_repair_required")
            ),
            "anima_source_or_cache_axis_required": bool(
                external_input_handoff_packet_summary.get("anima_source_or_cache_axis_required")
            ),
            "json_replay_ready": bool(
                external_input_handoff_packet_summary.get("json_replay_ready")
            ),
            "next_manual_gpu_gate": str(
                external_input_handoff_packet_summary.get("next_manual_gpu_gate") or ""
            ),
            "release_gate_blockers": _strings(
                external_input_handoff_packet_summary.get("release_gate_blockers")
            )[:20],
            "publishable": bool(external_input_handoff_packet_summary.get("publishable")),
            "fail_closed": bool(external_input_handoff_packet_summary.get("fail_closed")),
            "not_release_evidence": bool(
                external_input_handoff_packet_summary.get("not_release_evidence")
            ),
            "safe_to_auto_start": bool(
                external_input_handoff_packet_summary.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                external_input_handoff_packet_summary.get("release_claim_allowed")
            ),
            "does_not_run_training": bool(
                external_input_handoff_packet_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(
                external_input_handoff_packet_summary.get("does_not_run_cuda")
            ),
        },
        "newbie_warm_cache_inventory_summary": _compact_summary_mirror(
            newbie_warm_cache_inventory_summary,
            string_fields=[
                "report",
                "status",
                "selected_axis_kind",
                "selected_axis_caption_coverage",
            ],
            bool_fields=[
                "selected_axis_cache_ready",
                "evidence_pack_indexed",
                "claimable",
                "selected_axis_supersedes_cache_missing_blockers",
            ],
            int_fields=[
                "axis_count",
                "ready_axis_count",
                "completed_canary_axis_count",
                "selected_axis_completed_canary_command_count",
                "selected_axis_sample_count",
                "selected_axis_manifest_sample_count",
                "selected_axis_metadata_sample_count",
                "historical_cache_readiness_blocker_count",
            ],
        ),
        "source_cache_axis_admission_preflight_summary": _compact_summary_mirror(
            source_cache_axis_admission_preflight_summary,
            string_fields=[
                "report",
                "status",
                "candidate_source",
                "candidate_family",
                "candidate_root",
                "candidate_source_manifest_sha1",
                "matched_axis_source_kind",
                "matched_axis_state",
                "next_action",
            ],
            bool_fields=[
                "admission_allows_protected_manual_gpu_plan",
                "matched_axis_found",
                "matched_axis_cache_ready",
                "matched_axis_quality_ok",
                "matched_axis_attempted_or_completed",
                "new_axis_required",
                "duplicate_or_stale_axis_blocked",
                "current_axis_do_not_rerun_without_new_axis",
            ],
            int_fields=["blocker_count", "current_axis_completed_canary_command_count"],
            number_fields=[
                "matched_axis_candidate_rank_score",
                "matched_axis_caption_sample_coverage",
            ],
            list_fields=[
                "blockers",
                "acceptance_gates",
                "new_axis_reason_ids",
                "new_axis_required_identity_change_fields",
                "new_axis_same_root_identity_change_fields",
                "new_axis_acceptance_requirements",
            ],
        ),
        "source_axis_unblock_recommendation_summary": _compact_summary_mirror(
            source_axis_unblock_recommendation_summary,
            string_fields=[
                "status",
                "gate_id",
                "source_axis_requirement_status",
                "source_cache_preflight_status",
                "source_cache_preflight_next_action",
                "followup_execution_surface_status",
                "recommended_next_action",
                "recommended_release_policy",
            ],
            bool_fields=[
                "natural_load_gate_blocked",
                "source_axis_external_input_required",
                "new_axis_required",
                "duplicate_or_stale_axis_blocked",
                "current_axis_do_not_rerun_without_new_axis",
                "no_active_release_relevant_gpu_work",
                "no_diagnostic_manual_gpu_work",
                "old_axes_exhausted_without_new_axis",
            ],
            int_fields=[
                "natural_load_blocked_family_count",
                "source_axis_external_input_required_count",
                "source_axis_no_ready_family_count",
                "current_axis_completed_canary_command_count",
                "active_release_relevant_command_count",
                "diagnostic_manual_ready_command_count",
                "completed_existing_command_count",
                "rerun_blocked_without_new_axis_count",
            ],
            list_fields=[
                "natural_load_blocked_family_ids",
                "next_family_focus_ids",
                "new_axis_reason_ids",
                "required_identity_change_fields",
                "same_root_identity_change_fields",
                "new_axis_acceptance_requirements",
                "active_release_relevant_command_ids",
                "diagnostic_manual_ready_command_ids",
                "completed_existing_command_ids",
                "rerun_blocked_without_new_axis_command_ids",
                "blocked_actions",
            ],
        ),
        "source_cache_axis_manual_canary_plan_summary": _compact_summary_mirror(
            source_cache_axis_manual_canary_plan_summary,
            string_fields=["report", "status", "preflight_status"],
            bool_fields=["preflight_admitted", "requires_gpu_if_executed"],
            int_fields=["command_count", "blocked_command_count", "blocker_count"],
        ),
        "post_manual_evidence_rebuild_plan_summary": _compact_summary_mirror(
            post_manual_evidence_rebuild_plan_summary,
            string_fields=[
                "report",
                "status",
                "sd15_status",
                "next_rebuild_stage_id",
                "release_readiness",
                "natural_load_status",
            ],
            bool_fields=[
                "manual_canary_plan_ready",
                "manual_gpu_evidence_ready",
                "manual_gpu_evidence_required",
                "source_cache_axis_manual_canary_plan_required",
                "sd15_checkpoint_required",
                "natural_load_canary_pending",
                "release_claims_rebuild_required",
            ],
            int_fields=[
                "manual_canary_command_count",
                "command_count",
                "ready_command_count",
                "stage_count",
                "ready_stage_count",
                "blocked_stage_count",
                "expected_output_count",
                "existing_expected_output_count",
                "missing_expected_output_count",
                "blocked_expected_output_count",
                "pending_expected_output_count",
                "evidence_gap_count",
                "natural_load_ready_family_count",
                "natural_load_family_count",
                "blocker_count",
            ],
            list_fields=[
                "release_gate_blockers",
                "next_required_inputs",
                "first_blocked_stage_ids",
            ],
            nested_fields=["manual_evidence_blocking_summary"],
        ),
        "source_cache_axis_pipeline": {
            "status": str(pipeline.get("status") or unblocker.get("source_cache_axis_pipeline_status") or ""),
            "axis_readiness_status": str(
                pipeline.get("axis_readiness_status")
                or unblocker.get("source_cache_axis_readiness_status")
                or ""
            ),
            "pipeline_complete": pipeline_complete,
            "stage_ok_count": stage_ok,
            "stage_count": stage_count,
            "stage_roadmap_lineage": stage_lineage,
        },
        "external_input_handoff": {
            "status": str(handoff.get("status") or unblocker.get("external_input_handoff_status") or ""),
            "missing_external_input_count": _safe_int(handoff.get("missing_external_input_count"), len(missing_inputs)),
            "missing_external_inputs": missing_inputs[:20],
        },
        "external_input_filesystem_audit": filesystem_audit,
        "source_cache_negative_evidence_summary": source_cache_negative,
        "source_cache_axis_identity_registry": _source_cache_axis_identity_registry_summary(identity_registry),
        "source_cache_axis_identity_registry_summary": _source_cache_axis_identity_registry_summary(identity_registry),
        "release_readiness_guard_report_summary": guard_lineage,
        "roadmap_lineage_audit": roadmap_lineage,
        "artifact_freshness_audit": {
            "summary_version": _safe_int(freshness.get("summary_version"), 1),
            "roadmap": str(freshness.get("roadmap") or ROADMAP),
            "artifact_role": str(
                freshness.get("artifact_role") or "gpu_bubble_artifact_freshness_audit"
            ),
            "observed_at_epoch": freshness.get("observed_at_epoch", 0.0),
            "readiness": dict(_mapping(freshness.get("readiness"))),
            "upstream_artifacts": [dict(_mapping(item)) for item in _list(freshness.get("upstream_artifacts"))],
            "required_artifact_missing_count": _safe_int(freshness.get("required_artifact_missing_count")),
            "required_artifact_missing_ids": _strings(freshness.get("required_artifact_missing_ids"))[:20],
            "upstream_newer_than_readiness_count": _safe_int(
                freshness.get("upstream_newer_than_readiness_count")
            ),
            "upstream_newer_than_readiness_ids": _strings(
                freshness.get("upstream_newer_than_readiness_ids")
            )[:20],
            "readiness_not_older_than_upstream": bool(freshness.get("readiness_not_older_than_upstream")),
            "terminal_observation_not_older_than_readiness": bool(
                freshness.get("terminal_observation_not_older_than_readiness")
            ),
            "freshness_ok": bool(freshness.get("freshness_ok")),
            "drift_reason_ids": _strings(freshness.get("drift_reason_ids"))[:20],
            "fail_closed": bool(freshness.get("fail_closed", freshness.get("freshness_ok"))),
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "publishable": False,
            "does_not_run_training": True,
            "does_not_run_cuda": True,
        },
        "input_resolution_summary": terminal_input_resolution_summary,
        "post_manual_rebuild": {
            "status": str(post_manual.get("status") or unblocker.get("post_manual_rebuild_status") or ""),
            "ready_command_count": _safe_int(
                post_manual.get("ready_command_count"),
                _safe_int(unblocker.get("post_manual_ready_command_count")),
            ),
            "next_rebuild_stage_id": str(
                _mapping(post_manual.get("next_rebuild_stage")).get("stage_id")
                or unblocker.get("post_manual_next_rebuild_stage_id")
                or ""
            ),
            "manual_gpu_evidence_required": bool(manual_evidence_blocking.get("manual_gpu_evidence_required")),
            "release_gate_blockers": _strings(manual_evidence_blocking.get("release_gate_blockers"))
            or _strings(unblocker.get("post_manual_release_gate_blockers")),
        },
        "manual_evidence_blocking_summary": terminal_manual_evidence_blocking_summary,
        "sd15_manual_ab_contract_summary": terminal_sd15_manual_ab_contract_summary,
        "sd15_release_gap": {
            "status": str(sd15.get("status") or unblocker.get("sd15_release_gap_status") or ""),
            "blockers": _strings(sd15.get("blockers")) or _strings(unblocker.get("sd15_release_gap_blockers")),
        },
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "blocked_actions": [
            "do_not_mark_gpu_bubble_roadmap_complete_from_terminal_self_check",
            "do_not_publish_gpu_bubble_release_claim_from_terminal_self_check",
            "do_not_auto_start_gpu_heavy_from_terminal_self_check",
        ],
        "notes": [
            "This self-check is JSON-only and proves terminal blocker state, not release evidence.",
            "Further substantive evidence requires external inputs and protected manual GPU execution.",
        ],
        "source_reports": {
            "readiness_next_actions": str(readiness.get("report") or ""),
            "external_input_handoff_packet": str(handoff.get("report") or ""),
            "external_input_intake_registry": str(intake.get("report") or ""),
            "live_external_input_intake_registry": str(live_intake.get("report") or ""),
            "source_axis_requirement": str(source_requirement.get("report") or ""),
            "source_axis_freshness_dedupe_audit": str(source_freshness.get("report") or ""),
            "source_cache_axis_identity_registry": str(identity_registry.get("report") or ""),
            "newbie_warm_cache_inventory": str(warm_cache.get("report") or ""),
            "source_cache_axis_pipeline_readiness": str(pipeline.get("report") or ""),
            "post_manual_evidence_rebuild_plan": str(post_manual.get("report") or ""),
            "sd15_readiness": str(sd15.get("report") or ""),
            "newbie_tail8_seed2027_rerun_preflight": str(
                _mapping(readiness.get("newbie_tail8_seed2027_rerun_preflight_summary")).get("report")
                or _mapping(_mapping(readiness.get("evidence_summary")).get(
                    "newbie_tail8_seed2027_rerun_preflight_summary"
                )).get("report")
                or ""
            ),
            "newbie_tail8_forward_anomaly_review": str(
                _mapping(readiness.get("newbie_tail8_forward_anomaly_review_summary")).get("report")
                or _mapping(_mapping(readiness.get("evidence_summary")).get(
                    "newbie_tail8_forward_anomaly_review_summary"
                )).get("report")
                or ""
            ),
            "release_readiness_guard_report": str(guard_report.get("report") or ""),
        },
    }


__all__ = [
    "READINESS_REPORT",
    "REPORT",
    "ROADMAP",
    "build_gpu_bubble_readiness_terminal_self_check",
]
