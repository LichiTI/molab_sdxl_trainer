"""JSON-only freshness/dedupe audit for GPU-bubble source/cache axes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_AXIS_FRESHNESS_DEDUPE_AUDIT_REPORT = "bubble_source_axis_freshness_dedupe_audit_v0"
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _norm_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _current_roots(requirement: Mapping[str, Any], intake: Mapping[str, Any]) -> list[str]:
    roots: set[str] = set()
    for raw in _list(requirement.get("families")):
        roots.update(_strings(_mapping(raw).get("current_source_roots")))
    source_axis = _mapping(intake.get("source_axis"))
    roots.update(_strings(source_axis.get("current_source_roots")))
    roots.update(_strings(source_axis.get("known_source_roots")))
    for raw in _list(source_axis.get("roots")):
        item = _mapping(raw)
        if str(item.get("intake_status") or "") == "current_axis_duplicate":
            roots.add(str(item.get("root") or ""))
    return sorted(root for root in roots if root)


def _new_roots(intake: Mapping[str, Any]) -> list[str]:
    rows = []
    for raw in _list(_mapping(intake.get("source_axis")).get("roots")):
        item = _mapping(raw)
        if str(item.get("intake_status") or "") == "new_root_available":
            root = str(item.get("root") or "")
            if root:
                rows.append(root)
    return sorted(rows)


def _family_requirements(requirement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in _list(requirement.get("families")):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if family:
            rows[family] = item
    return rows


def _ranked_axes(source_axis_scout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]


def _axis_key(axis: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        _family_key(axis.get("family")),
        _norm_path(axis.get("source_data")),
        _safe_int(axis.get("sample_offset")),
        str(axis.get("source_manifest_sha1") or "").strip(),
    )


def _scout_axis_by_family_root_offset(source_axis_scout: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for axis in _ranked_axes(source_axis_scout):
        key = (
            _family_key(axis.get("family")),
            _norm_path(axis.get("source_data")),
            _safe_int(axis.get("sample_offset")),
        )
        if key[0] and key[1] and key not in rows:
            rows[key] = axis
    return rows


def _warm_cache_inventory_axes(
    newbie_warm_cache_inventory: Mapping[str, Any],
    *,
    source_axis_scout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scout_by_axis = _scout_axis_by_family_root_offset(source_axis_scout)
    rows: list[dict[str, Any]] = []
    for raw in _list(newbie_warm_cache_inventory.get("axes")):
        axis = _mapping(raw)
        source_data = str(axis.get("source_data") or "")
        sample_offset = _safe_int(axis.get("sample_offset"))
        scout_axis = scout_by_axis.get(("newbie", _norm_path(source_data), sample_offset), {})
        manifest = str(axis.get("source_manifest_sha1") or "").strip() or str(
            scout_axis.get("source_manifest_sha1") or ""
        ).strip()
        if not source_data or sample_offset <= 0:
            continue
        completed_count = _safe_int(axis.get("completed_canary_command_count"))
        do_not_rerun = bool(axis.get("do_not_rerun_without_new_axis"))
        completed_or_stale = completed_count > 0 or do_not_rerun
        blocked_reasons = set(_strings(axis.get("blocked_reasons")))
        if str(newbie_warm_cache_inventory.get("status") or ""):
            blocked_reasons.add(str(newbie_warm_cache_inventory.get("status") or ""))
        if completed_or_stale:
            blocked_reasons.add("candidate_axis_already_attempted_or_completed")
        if do_not_rerun:
            blocked_reasons.add("do_not_rerun_without_new_axis")
        caption_coverage = _safe_float(
            scout_axis.get("caption_sample_coverage"),
            _safe_float(_mapping(axis.get("manifest")).get("caption_coverage")),
        )
        candidate_rank_score = _safe_float(scout_axis.get("candidate_rank_score"))
        caption_ok = caption_coverage >= 0.875
        rank_score_ok = bool(scout_axis) and candidate_rank_score >= 4.0
        if not scout_axis:
            blocked_reasons.add("candidate_rank_score_missing_from_scout")
        if not caption_ok:
            blocked_reasons.add("caption_coverage_below_scout_threshold")
        if not rank_score_ok:
            blocked_reasons.add("candidate_rank_score_below_scout_threshold")
        cache_ready = bool(axis.get("cache_ready"))
        rows.append(
            {
                "family": "newbie",
                "source_data": source_data,
                "sample_offset": sample_offset,
                "source_manifest_sha1": manifest,
                "state": str(axis.get("status") or axis.get("axis_kind") or ""),
                "axis_kind": str(axis.get("axis_kind") or ""),
                "cache_ready": cache_ready,
                "quality_ok": bool(cache_ready and caption_ok and rank_score_ok),
                "claimable": bool(axis.get("claimable")),
                "do_not_rerun_without_new_axis": do_not_rerun,
                "completed_canary_command_count": completed_count,
                "attempted_or_completed": completed_or_stale,
                "completed_existing_evidence": completed_count > 0,
                "planned_followup_attempt": False,
                "attempted_in_followup_plan": False,
                "blocked_reasons": sorted(blocked_reasons),
            }
        )
    return rows


def _identity_digest(parts: Sequence[Any]) -> str:
    text = "|".join(str(part or "") for part in parts)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_axis_identity_registry(
    *,
    source_axis_scout: Mapping[str, Any],
    newbie_warm_cache_inventory: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    current_roots: Sequence[str],
    new_roots: Sequence[str],
) -> dict[str, Any]:
    current_root_keys = {_norm_path(item) for item in current_roots}
    new_root_keys = {_norm_path(item) for item in new_roots}
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}

    def add_row(
        *,
        source_kind: str,
        identity_scope: str,
        family: str = "",
        source_root: str = "",
        sample_offset: Any = None,
        source_manifest_sha1: str = "",
        state: str = "",
        cache_ready: bool = False,
        quality_ok: bool = False,
        attempted_or_completed: bool = False,
        completed_existing_evidence: bool = False,
        planned_followup_attempt: bool = False,
        attempted_in_followup_plan: bool = False,
        candidate_fresh: bool = False,
        candidate_duplicate_or_stale: bool = False,
        axis_kind: str = "",
        claimable: bool = False,
        do_not_rerun_without_new_axis: bool = False,
        completed_canary_command_count: int = 0,
        blocked_reasons: Sequence[str] = (),
    ) -> None:
        normalized_root = _norm_path(source_root)
        offset_text = "" if sample_offset is None else str(_safe_int(sample_offset))
        manifest = str(source_manifest_sha1 or "").strip()
        normalized_family = _family_key(family)
        full_axis_identity = bool(normalized_family and normalized_root and offset_text and manifest)
        identity_parts = (
            identity_scope,
            normalized_family,
            normalized_root,
            offset_text,
            manifest,
        )
        identity_key = "|".join(str(part or "") for part in identity_parts)
        digest = _identity_digest(identity_parts)
        duplicate_or_stale = bool(
            attempted_or_completed
            or completed_existing_evidence
            or planned_followup_attempt
            or attempted_in_followup_plan
            or candidate_duplicate_or_stale
        )
        existing = seen.get(identity_key)
        if existing is not None:
            existing["source_kind"] = str(existing.get("source_kind") or source_kind)
            existing["source_kinds"] = sorted(
                set(_strings(existing.get("source_kinds")) + [source_kind])
            )
            existing["state"] = state or str(existing.get("state") or "")
            existing["axis_kind"] = axis_kind or str(existing.get("axis_kind") or "")
            existing["cache_ready"] = bool(existing.get("cache_ready")) or cache_ready
            existing["quality_ok"] = bool(existing.get("quality_ok")) or quality_ok
            existing["claimable"] = bool(existing.get("claimable")) or claimable
            existing["do_not_rerun_without_new_axis"] = (
                bool(existing.get("do_not_rerun_without_new_axis")) or do_not_rerun_without_new_axis
            )
            existing["completed_canary_command_count"] = max(
                _safe_int(existing.get("completed_canary_command_count")),
                _safe_int(completed_canary_command_count),
            )
            existing["blocked_reasons"] = sorted(
                set(_strings(existing.get("blocked_reasons")) + _strings(blocked_reasons))
            )
            existing["attempted_or_completed"] = bool(existing.get("attempted_or_completed")) or attempted_or_completed
            existing["completed_existing_evidence"] = (
                bool(existing.get("completed_existing_evidence")) or completed_existing_evidence
            )
            existing["planned_followup_attempt"] = (
                bool(existing.get("planned_followup_attempt")) or planned_followup_attempt
            )
            existing["attempted_in_followup_plan"] = (
                bool(existing.get("attempted_in_followup_plan")) or attempted_in_followup_plan
            )
            existing["duplicate_or_stale_axis"] = bool(existing.get("duplicate_or_stale_axis")) or duplicate_or_stale
            existing["fresh_axis_candidate"] = bool(
                bool(existing.get("fresh_axis_candidate")) or (candidate_fresh and not duplicate_or_stale)
            ) and not bool(existing.get("duplicate_or_stale_axis"))
            existing["full_axis_identity_present"] = bool(existing.get("full_axis_identity_present")) or full_axis_identity
            return
        row = {
            "row_id": f"{source_kind}:{digest[7:23]}",
            "source_kind": source_kind,
            "source_kinds": [source_kind],
            "identity_scope": identity_scope,
            "identity_key": identity_key,
            "identity_digest": digest,
            "family": normalized_family,
            "source_root": source_root,
            "normalized_source_root": normalized_root,
            "sample_offset": _safe_int(sample_offset) if sample_offset is not None else None,
            "source_manifest_sha1": manifest,
            "state": state,
            "axis_kind": axis_kind,
            "cache_ready": cache_ready,
            "quality_ok": quality_ok,
            "claimable": claimable,
            "do_not_rerun_without_new_axis": do_not_rerun_without_new_axis,
            "completed_canary_command_count": _safe_int(completed_canary_command_count),
            "blocked_reasons": _strings(blocked_reasons),
            "attempted_or_completed": attempted_or_completed,
            "completed_existing_evidence": completed_existing_evidence,
            "planned_followup_attempt": planned_followup_attempt,
            "attempted_in_followup_plan": attempted_in_followup_plan,
            "current_source_root": bool(normalized_root and normalized_root in current_root_keys),
            "new_source_root": bool(normalized_root and normalized_root in new_root_keys),
            "duplicate_or_stale_axis": duplicate_or_stale,
            "fresh_axis_candidate": bool(candidate_fresh and not duplicate_or_stale),
            "full_axis_identity_present": full_axis_identity,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
        }
        seen[identity_key] = row
        rows.append(row)

    for root in current_roots:
        add_row(source_kind="current_source_root", identity_scope="root", source_root=str(root))
    for root in new_roots:
        add_row(source_kind="new_source_root", identity_scope="root", source_root=str(root))
    for axis in _ranked_axes(source_axis_scout):
        add_row(
            source_kind="ranked_axis",
            identity_scope="axis",
            family=str(axis.get("family") or ""),
            source_root=str(axis.get("source_data") or ""),
            sample_offset=axis.get("sample_offset"),
            source_manifest_sha1=str(axis.get("source_manifest_sha1") or ""),
            state=str(axis.get("state") or ""),
            cache_ready=bool(axis.get("cache_ready")),
            quality_ok=bool(axis.get("quality_ok")),
            attempted_or_completed=bool(axis.get("attempted_or_completed")),
            completed_existing_evidence=bool(axis.get("completed_existing_evidence")),
            planned_followup_attempt=bool(axis.get("planned_followup_attempt")),
            attempted_in_followup_plan=bool(axis.get("attempted_in_followup_plan")),
            blocked_reasons=_strings(axis.get("blocked_reasons")),
        )
    for axis in _warm_cache_inventory_axes(
        newbie_warm_cache_inventory,
        source_axis_scout=source_axis_scout,
    ):
        add_row(
            source_kind="warm_cache_inventory_axis",
            identity_scope="axis",
            family=str(axis.get("family") or ""),
            source_root=str(axis.get("source_data") or ""),
            sample_offset=axis.get("sample_offset"),
            source_manifest_sha1=str(axis.get("source_manifest_sha1") or ""),
            state=str(axis.get("state") or ""),
            cache_ready=bool(axis.get("cache_ready")),
            quality_ok=bool(axis.get("quality_ok")),
            attempted_or_completed=bool(axis.get("attempted_or_completed")),
            completed_existing_evidence=bool(axis.get("completed_existing_evidence")),
            planned_followup_attempt=bool(axis.get("planned_followup_attempt")),
            attempted_in_followup_plan=bool(axis.get("attempted_in_followup_plan")),
            axis_kind=str(axis.get("axis_kind") or ""),
            claimable=bool(axis.get("claimable")),
            do_not_rerun_without_new_axis=bool(axis.get("do_not_rerun_without_new_axis")),
            completed_canary_command_count=_safe_int(axis.get("completed_canary_command_count")),
            blocked_reasons=_strings(axis.get("blocked_reasons")),
        )
    if str(candidate.get("family") or "") or str(candidate.get("root") or ""):
        add_row(
            source_kind="registered_candidate",
            identity_scope="axis",
            family=str(candidate.get("family") or ""),
            source_root=str(candidate.get("root") or ""),
            sample_offset=candidate.get("sample_offset"),
            source_manifest_sha1=str(candidate.get("source_manifest_sha1") or ""),
            state=str(candidate_audit.get("status") or ""),
            candidate_fresh=bool(candidate_audit.get("fresh")),
            candidate_duplicate_or_stale=bool(candidate_audit.get("duplicate_or_stale")),
        )

    unsafe_row_ids = [
        str(row.get("row_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed"))
        or not bool(row.get("not_release_evidence"))
    ]
    full_axis_rows = [row for row in rows if bool(row.get("full_axis_identity_present"))]
    duplicate_rows = [row for row in rows if bool(row.get("duplicate_or_stale_axis"))]
    fresh_rows = [row for row in rows if bool(row.get("fresh_axis_candidate"))]
    warm_cache_rows = [
        row for row in rows if "warm_cache_inventory_axis" in _strings(row.get("source_kinds"))
    ]
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_source_cache_axis_identity_registry",
        "identity_schema_version": 1,
        "row_count": len(rows),
        "root_identity_row_count": sum(1 for row in rows if row["identity_scope"] == "root"),
        "full_axis_identity_row_count": len(full_axis_rows),
        "current_source_root_count": len(current_roots),
        "new_source_root_count": len(new_roots),
        "duplicate_or_stale_axis_count": len(duplicate_rows),
        "fresh_axis_candidate_count": len(fresh_rows),
        "warm_cache_inventory_axis_count": len(warm_cache_rows),
        "warm_cache_inventory_duplicate_or_stale_axis_count": sum(
            1 for row in warm_cache_rows if bool(row.get("duplicate_or_stale_axis"))
        ),
        "unsafe_row_count": len(unsafe_row_ids),
        "unsafe_row_ids": unsafe_row_ids[:20],
        "rows": rows,
        "fail_closed": bool(rows) and not unsafe_row_ids,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "blocked_actions": [
            "do_not_auto_start_gpu_heavy_from_axis_identity_registry",
            "do_not_publish_release_claim_from_axis_identity_registry",
            "do_not_treat_root_only_identity_as_full_axis_admission",
        ],
    }


def _candidate(preflight: Mapping[str, Any]) -> dict[str, Any]:
    item = _mapping(preflight.get("candidate"))
    return {
        "family": _family_key(item.get("family")),
        "root": str(item.get("root") or ""),
        "sample_offset": item.get("sample_offset"),
        "source_manifest_sha1": str(item.get("source_manifest_sha1") or "").strip(),
        "preflight_status": str(preflight.get("status") or ""),
        "preflight_admitted": bool(preflight.get("admission_allows_protected_manual_gpu_plan")),
    }


def _matching_axes(
    *,
    source_axis_scout: Mapping[str, Any],
    family: str,
    root: str,
    sample_offset: Any,
    source_manifest_sha1: str,
) -> list[dict[str, Any]]:
    root_key = _norm_path(root)
    manifest = str(source_manifest_sha1 or "").strip()
    offset = _safe_int(sample_offset) if sample_offset is not None else None
    matches: list[dict[str, Any]] = []
    for axis in _ranked_axes(source_axis_scout):
        if family and _family_key(axis.get("family")) != family:
            continue
        if root_key and _norm_path(axis.get("source_data")) != root_key:
            continue
        if offset is not None and _safe_int(axis.get("sample_offset")) != offset:
            continue
        if manifest and str(axis.get("source_manifest_sha1") or "").strip() != manifest:
            continue
        matches.append(
            {
                "family": _family_key(axis.get("family")),
                "source_data": str(axis.get("source_data") or ""),
                "sample_offset": _safe_int(axis.get("sample_offset")),
                "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
                "state": str(axis.get("state") or ""),
                "cache_ready": bool(axis.get("cache_ready")),
                "quality_ok": bool(axis.get("quality_ok")),
                "attempted_or_completed": bool(axis.get("attempted_or_completed")),
                "completed_existing_evidence": bool(axis.get("completed_existing_evidence")),
                "planned_followup_attempt": bool(axis.get("planned_followup_attempt")),
                "attempted_in_followup_plan": bool(axis.get("attempted_in_followup_plan")),
                "blocked_reasons": _strings(axis.get("blocked_reasons")),
            }
        )
    return matches


def _completed_axis_count(source_axis_scout: Mapping[str, Any]) -> int:
    keys = set()
    for axis in _ranked_axes(source_axis_scout):
        if bool(axis.get("completed_existing_evidence")) or bool(axis.get("attempted_or_completed")):
            keys.add(_axis_key(axis))
    return len(keys)


def _completed_out_dirs(requirement: Mapping[str, Any]) -> list[str]:
    out_dirs: set[str] = set()
    for raw in _list(requirement.get("families")):
        run = _mapping(_mapping(raw).get("run_readiness"))
        out_dirs.update(_strings(run.get("completed_out_dirs")))
    return sorted(item for item in out_dirs if item)


def _family_audits(requirement: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, item in _family_requirements(requirement).items():
        run = _mapping(item.get("run_readiness"))
        rows.append(
            {
                "family": family,
                "status": str(item.get("status") or ""),
                "requirement": str(item.get("requirement") or ""),
                "source_axis_state": str(item.get("source_axis_state") or ""),
                "requires_external_input": bool(item.get("requires_external_input")),
                "do_not_rerun_current_axis": bool(item.get("do_not_rerun_current_axis")),
                "current_source_roots": _strings(item.get("current_source_roots")),
                "completed_command_ids": _strings(run.get("completed_command_ids")),
                "completed_out_dirs": _strings(run.get("completed_out_dirs")),
                "blocked_actions": _strings(item.get("blocked_actions")),
            }
        )
    return rows


def _candidate_audit(
    *,
    candidate: Mapping[str, Any],
    source_axis_scout: Mapping[str, Any],
    current_roots: Sequence[str],
) -> dict[str, Any]:
    family = str(candidate.get("family") or "")
    root = str(candidate.get("root") or "")
    manifest = str(candidate.get("source_manifest_sha1") or "")
    offset = candidate.get("sample_offset")
    if not family or not root:
        return {
            "status": "not_registered",
            "fresh": False,
            "duplicate_or_stale": False,
            "blockers": ["candidate_root_or_family_required"],
            "matching_axis_count": 0,
            "matching_axes": [],
        }
    matches = _matching_axes(
        source_axis_scout=source_axis_scout,
        family=family,
        root=root,
        sample_offset=offset,
        source_manifest_sha1=manifest,
    )
    blockers: set[str] = set()
    current_keys = {_norm_path(item) for item in current_roots}
    if _norm_path(root) in current_keys:
        blockers.add("root_matches_current_source_axis")
    for match in matches:
        if bool(match.get("completed_existing_evidence")) or bool(match.get("attempted_or_completed")):
            blockers.add("candidate_axis_already_attempted_or_completed")
        if bool(match.get("planned_followup_attempt")) or bool(match.get("attempted_in_followup_plan")):
            blockers.add("candidate_axis_already_in_followup_run_plan")
        blockers.update(_strings(match.get("blocked_reasons")))
    duplicate = any(
        reason
        in {
            "candidate_axis_already_attempted_or_completed",
            "candidate_axis_already_in_followup_run_plan",
            "axis_already_has_completed_evidence",
            "axis_already_in_followup_run_plan",
        }
        for reason in blockers
    )
    missing = not matches
    if missing:
        blockers.add("candidate_axis_not_found_in_scout")
    if duplicate:
        status = "duplicate_or_stale"
    elif missing:
        status = "candidate_not_scanned"
    elif "root_matches_current_source_axis" in blockers:
        status = "current_root_manual_review_required"
    else:
        status = "fresh_candidate_review_ready"
    return {
        "status": status,
        "fresh": status == "fresh_candidate_review_ready",
        "duplicate_or_stale": duplicate,
        "blockers": sorted(blockers),
        "matching_axis_count": len(matches),
        "matching_axes": matches[:8],
    }


def _axis_state(status: str) -> str:
    if status == "waiting_for_external_input":
        return "axis_waiting_external_input"
    if status == "duplicate_or_stale_candidate_blocked":
        return "duplicate_or_stale_axis_blocked"
    if status == "fresh_candidate_preflight_admitted":
        return "fresh_axis_preflight_admitted"
    return "axis_review_required"


def build_source_axis_freshness_dedupe_audit(
    *,
    external_input_intake_registry: Mapping[str, Any] | None = None,
    source_axis_scout: Mapping[str, Any] | None = None,
    source_axis_requirement: Mapping[str, Any] | None = None,
    source_cache_axis_admission_preflight: Mapping[str, Any] | None = None,
    source_cache_axis_manual_canary_plan: Mapping[str, Any] | None = None,
    external_input_replay_plan: Mapping[str, Any] | None = None,
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-GPU audit that separates fresh axes from duplicates."""

    intake = _mapping(external_input_intake_registry)
    scout = _mapping(source_axis_scout)
    requirement = _mapping(source_axis_requirement)
    preflight = _mapping(source_cache_axis_admission_preflight)
    manual_plan = _mapping(source_cache_axis_manual_canary_plan)
    replay_plan = _mapping(external_input_replay_plan)
    warm_cache = _mapping(newbie_warm_cache_inventory)
    roots = _current_roots(requirement, intake)
    new_roots = _new_roots(intake)
    cand = _candidate(preflight)
    candidate_audit = _candidate_audit(candidate=cand, source_axis_scout=scout, current_roots=roots)
    identity_registry = _source_axis_identity_registry(
        source_axis_scout=scout,
        newbie_warm_cache_inventory=warm_cache,
        candidate=cand,
        candidate_audit=candidate_audit,
        current_roots=roots,
        new_roots=new_roots,
    )
    preflight_admitted = bool(cand.get("preflight_admitted"))
    manual_plan_ready = str(manual_plan.get("status") or "") == "protected_manual_canary_plan_ready"
    external_detected = bool(new_roots)
    if candidate_audit["duplicate_or_stale"]:
        status = "duplicate_or_stale_candidate_blocked"
    elif candidate_audit["status"] == "fresh_candidate_review_ready" and preflight_admitted:
        status = "fresh_candidate_preflight_admitted"
    elif external_detected or new_roots:
        status = "new_source_review_required"
    else:
        status = "waiting_for_external_input"
    axis_state = _axis_state(status)
    blocker_set = set(candidate_audit["blockers"])
    if not new_roots:
        blocker_set.add("new_source_root_required")
    if not preflight_admitted:
        blocker_set.add("source_cache_axis_preflight_not_admitted")
    return {
        "schema_version": 1,
        "report": SOURCE_AXIS_FRESHNESS_DEDUPE_AUDIT_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "axis_state": axis_state,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "external_input_detected": external_detected,
        "new_source_root_count": len(new_roots),
        "current_source_roots": roots,
        "new_source_roots": new_roots,
        "completed_axis_count": _completed_axis_count(scout),
        "completed_out_dir_count": len(_completed_out_dirs(requirement)),
        "completed_out_dirs": _completed_out_dirs(requirement),
        "families": _family_audits(requirement),
        "candidate": cand,
        "candidate_audit": candidate_audit,
        "source_cache_axis_identity_registry": identity_registry,
        "preflight_admitted": preflight_admitted,
        "manual_canary_plan_ready": manual_plan_ready,
        "replay_plan": {
            "status": str(replay_plan.get("status") or ""),
            "ready_command_count": _safe_int(replay_plan.get("ready_command_count")),
            "template_command_count": _safe_int(replay_plan.get("template_command_count")),
        },
        "blockers": sorted(blocker_set),
        "blocked_actions": [
            "auto_start_gpu_heavy_from_freshness_audit",
            "promote_freshness_audit_as_release_evidence",
            "rerun_completed_followup_out_dirs_without_distinct_axis",
            "treat_same_root_as_fresh_without_offset_or_manifest_review",
        ],
        "acceptance_gates": [
            "candidate_axis_not_attempted_or_completed",
            "candidate_axis_not_in_followup_run_plan",
            "same_root_requires_distinct_unattempted_offset_or_manifest",
            "freshness_audit_is_not_release_evidence",
            "admitted_preflight_required_before_manual_canary_plan",
        ],
        "notes": [
            "This audit is JSON-only and does not start GPU work.",
            "Same root is not automatically duplicate, but the same family/root/offset/manifest already attempted or completed is blocked.",
            "A fresh audit only allows preflight/manual-plan review; release claims still require downstream canary and claims rebuild.",
        ],
    }


__all__ = [
    "SOURCE_AXIS_FRESHNESS_DEDUPE_AUDIT_REPORT",
    "ROADMAP",
    "build_source_axis_freshness_dedupe_audit",
]
