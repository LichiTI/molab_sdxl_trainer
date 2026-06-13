"""Natural-load canary gates for release candidate promotion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .bubble_natural_data_wait_ab_evidence import NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT
from .bubble_natural_data_wait_evidence import NATURAL_DATA_WAIT_EVIDENCE_REPORT
from .bubble_natural_release_guard import (
    natural_ab_release_reasons,
    natural_data_wait_release_reasons,
)


NATURAL_LOAD_CANARY_REPORT = "bubble_natural_load_canary_v0"
DEFAULT_CANARY_FAMILIES = ("sdxl", "anima", "newbie")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"newbie_dit", "dit"}:
        return "newbie"
    return text


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _candidate_kind(report: Mapping[str, Any]) -> str:
    kind = str(report.get("report") or "")
    if kind == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        return "natural_closed_loop"
    if kind == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        return "natural_ab"
    return ""


def _axes(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("report") == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        return _mapping(_mapping(report.get("analysis")).get("matrix_axes"))
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        before = _mapping(report.get("before"))
        after = _mapping(report.get("after"))
        return _mapping(after.get("matrix_axes")) or _mapping(before.get("matrix_axes"))
    return {}


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("report") == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        return _mapping(report.get("metrics"))
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        return _mapping(_mapping(report.get("after")).get("metrics"))
    return {}


def _comparison(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        return _mapping(report.get("comparison"))
    return {}


def _compact_axes(axes: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "family",
        "native_cache_mode",
        "resolution",
        "train_batch_size",
        "steps",
        "samples",
        "dataloader_workers",
        "dataloader_prefetch_factor",
        "pin_memory",
        "source_fixture",
        "fixture_samples",
        "source_file_count",
        "source_manifest_sha1",
        "cache_state",
        "cache_has_family_cache",
        "material_source_label",
    )
    return {key: axes.get(key) for key in keys if key in axes}


def _cache_gate(report: Mapping[str, Any], axes: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    native_cache_mode = str(axes.get("native_cache_mode") or "").strip().lower()
    cache_state = str(axes.get("cache_state") or "").strip().lower()
    source_fixture = str(axes.get("source_fixture") or "").strip()
    if bool(_mapping(report.get("decision")).get("cache_probe_only")):
        reasons.append("cache_probe_only")
    if cache_state == "missing_at_start":
        reasons.append("cache_state_missing_at_start")
    if native_cache_mode in {"online_cache", "rebuild_cache"}:
        reasons.append(f"native_cache_mode_{native_cache_mode}")
    if source_fixture == "real_material_canary_v0":
        if cache_state != "warm_cache":
            reasons.append(f"real_material_cache_state_{cache_state or 'missing'}")
        if axes.get("cache_has_family_cache") is not True:
            reasons.append("real_material_family_cache_missing")
        if _safe_int(axes.get("fixture_samples")) <= 0:
            reasons.append("real_material_fixture_samples_missing")
        if _safe_int(axes.get("source_file_count")) <= 0:
            reasons.append("real_material_source_files_missing")
        if not str(axes.get("source_manifest_sha1") or ""):
            reasons.append("real_material_source_manifest_sha1_missing")
    has_inventory = any(
        key in axes
        for key in (
            "cache_state",
            "cache_present_before",
            "native_cache_mode",
            "source_fixture",
            "fixture_samples",
        )
    )
    if not has_inventory:
        reasons.append("cache_inventory_missing")
    return ("blocked" if reasons else "passed"), reasons


def _loss_gate(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    loss_status = str(_mapping(report.get("loss_stability")).get("status") or "")
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        if loss_status == "stable":
            return "passed", []
        return "blocked", [f"loss_stability_{loss_status or 'missing'}"]
    if loss_status in {"observed", "stable"}:
        return "passed", []
    return "blocked", [f"loss_stability_{loss_status or 'missing'}"]


def _vram_gate(report: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[str, list[str]]:
    peak = _safe_float(metrics.get("peak_vram_mb"))
    if peak <= 0.0:
        return "blocked", ["peak_vram_missing"]
    max_ratio = _safe_float(report.get("max_vram_ratio"), 0.0)
    memory_ratio = _safe_float(metrics.get("memory_ratio"), 0.0)
    if max_ratio > 0.0 and memory_ratio > max_ratio:
        return "blocked", ["peak_vram_ratio_exceeded"]
    return "passed", []


def _action_boundary_gate(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        action = _mapping(report.get("action"))
        if str(action.get("action_kind") or "") != "next_run_dataloader_workers_prefetch":
            return "blocked", ["natural_ab_action_kind_missing"]
        after = _mapping(action.get("after"))
        if _safe_float(after.get("dataloader_workers"), -1.0) < 0.0:
            return "blocked", ["natural_ab_after_workers_missing"]
        return "passed", []
    actions = report.get("action_chain")
    chain = actions if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) else []
    for item in chain:
        action = _mapping(item)
        if str(action.get("action_kind") or "") != "set_dataloader_workers":
            continue
        if str(action.get("adapter_id") or "") != "dataloader_rebuild_runtime_contract_v0":
            continue
        if str(action.get("apply_boundary") or "") in {"epoch_start", "epoch_boundary"}:
            return "passed", []
    return "blocked", ["dataloader_rebuild_epoch_boundary_action_missing"]


def _schema_gate(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    if "schema_version" not in report:
        return "blocked", ["schema_version_missing"]
    if _safe_int(report.get("schema_version")) != 1:
        return "blocked", ["schema_version_invalid"]
    return "passed", []


def _wording_gate(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    release_claim = _mapping(report.get("release_claim"))
    scope = str(release_claim.get("scope") or "")
    reasons: list[str] = []
    if not _safe_bool(release_claim.get("eligible"), False):
        reasons.append("release_claim_not_eligible")
    if not scope.startswith("case_specific"):
        reasons.append("case_specific_scope_missing")
    if report.get("report") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        reasons.extend(natural_ab_release_reasons(report))
    elif report.get("report") == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        reasons.extend(natural_data_wait_release_reasons(report))
    if not str(report.get("case_id") or ""):
        reasons.append("case_id_missing")
    if not str(report.get("family") or ""):
        reasons.append("family_missing")
    deduped = sorted(dict.fromkeys(reasons))
    return ("blocked" if deduped else "passed"), deduped


def _reason_category(reason: str) -> str:
    if reason == "natural_load_canary_evidence_missing":
        return "missing_evidence"
    if reason in {
        "cache_probe_only",
        "cache_inventory_missing",
        "cache_state_missing_at_start",
        "real_material_family_cache_missing",
        "real_material_fixture_samples_missing",
        "real_material_source_files_missing",
        "real_material_source_manifest_sha1_missing",
    } or reason.startswith(("native_cache_mode_", "real_material_cache_state_")):
        return "cache_readiness"
    if reason.startswith("loss_stability_"):
        return "loss_guardrail"
    if reason in {
        "before_data_wait_below_threshold",
        "after_data_wait_not_below_threshold",
        "data_wait_not_reduced",
        "natural_data_wait_below_threshold",
        "natural_data_wait_status_not_observed",
        "natural_ab_status_not_keep_recommended",
    }:
        return "data_wait_gate"
    if reason in {"throughput_gain_below_threshold"}:
        return "throughput_gate"
    if reason.startswith("peak_vram"):
        return "vram_guardrail"
    if reason in {
        "dataloader_rebuild_epoch_boundary_action_missing",
        "natural_ab_action_kind_missing",
        "natural_ab_after_workers_missing",
    }:
        return "action_boundary"
    if reason in {
        "release_claim_not_eligible",
        "case_specific_scope_missing",
        "benchmark_injection_blockers_present",
    }:
        return "release_claim_gate"
    if reason == "diagnostic_only_evidence":
        return "diagnostic_only"
    if reason in {"case_id_missing", "family_missing", "schema_version_missing", "schema_version_invalid"}:
        return "evidence_schema"
    return "other"


def _next_actions(status: str, reasons: Sequence[str]) -> list[str]:
    if status == "ready":
        return ["keep_as_release_candidate_input"]
    if status == "missing":
        return ["run_or_collect_natural_load_canary_evidence"]

    categories = {_reason_category(reason) for reason in reasons}
    actions: list[str] = []
    if "cache_readiness" in categories:
        actions.append("prepare_or_verify_warm_family_cache_inventory")
    if "loss_guardrail" in categories:
        actions.append("rerun_or_review_loss_stability_before_release")
    if "data_wait_gate" in categories:
        actions.append("collect_stronger_natural_data_wait_evidence")
    if "throughput_gate" in categories:
        actions.append("rerun_with_throughput_gain_or_mark_nonrelease")
    if "vram_guardrail" in categories:
        actions.append("reduce_vram_pressure_or_mark_candidate_nonrelease")
    if "action_boundary" in categories:
        actions.append("capture_epoch_boundary_dataloader_rebuild_action")
    if "release_claim_gate" in categories:
        actions.append("rebuild_formal_ab_with_case_specific_release_claim_after_gates_pass")
    if "diagnostic_only" in categories:
        actions.append("keep_diagnostic_evidence_out_of_release_claims")
    if "evidence_schema" in categories:
        actions.append("repair_canary_evidence_schema_fields")
    if "missing_evidence" in categories:
        actions.append("run_or_collect_natural_load_canary_evidence")
    return actions or ["inspect_blocked_natural_load_canary_evidence"]


def _rerun_intent(status: str, reasons: Sequence[str]) -> str:
    categories = {_reason_category(reason) for reason in reasons}
    if status == "missing" or "missing_evidence" in categories:
        return "collect_first_family_natural_load_canary"
    if "cache_readiness" in categories:
        return "prepare_warm_cache_then_collect_family_canary"
    if "action_boundary" in categories:
        return "capture_epoch_boundary_dataloader_rebuild_action"
    if "data_wait_gate" in categories:
        return "collect_stronger_baseline_data_wait_axis"
    if "loss_guardrail" in categories:
        return "rerun_with_loss_stability_review"
    if "throughput_gate" in categories:
        return "rerun_with_throughput_gain_review"
    if "vram_guardrail" in categories:
        return "rerun_with_vram_pressure_review"
    if "release_claim_gate" in categories:
        return "rebuild_case_specific_release_claim_after_gates_pass"
    return "inspect_blocked_family_canary_evidence"


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[int, str]:
    reasons = set(_string_list(candidate.get("blocked_reasons")))
    kind = str(candidate.get("kind") or "")
    cache_reasons = {
        reason
        for reason in reasons
        if _reason_category(reason) == "cache_readiness"
    }
    score = 100
    if "dataloader_rebuild_epoch_boundary_action_missing" in reasons and not cache_reasons:
        score = 10
    elif _reason_category("natural_data_wait_below_threshold") not in {
        _reason_category(reason) for reason in reasons
    } and not cache_reasons:
        score = 20
    elif kind == "natural_ab" and not cache_reasons:
        score = 30
    elif not cache_reasons:
        score = 40
    return score, str(candidate.get("case_id") or "")


def _preferred_blocked_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    blocked = [item for item in candidates if not _safe_bool(item.get("eligible"), False)]
    if not blocked:
        return {}
    return sorted(blocked, key=_candidate_rank)[0]


def _family_gpu_rerun_plan(
    *,
    family: str,
    status: str,
    reasons: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    next_actions: Sequence[str],
) -> dict[str, Any]:
    if status == "ready":
        return {
            "status": "not_needed",
            "family": family,
            "requires_gpu_if_executed": False,
            "manual_start_required": False,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_success": False,
        }

    preferred = _preferred_blocked_candidate(candidates)
    axes = _mapping(preferred.get("matrix_axes"))
    command_profile = "family_real_material_canary"
    if str(preferred.get("kind") or "") == "natural_closed_loop":
        command_profile = "natural_closed_loop_epoch_boundary_canary"
    elif str(preferred.get("kind") or "") == "natural_ab":
        command_profile = "natural_data_wait_ab_conservative_recheck"
    return {
        "status": "manual_gpu_rerun_required" if status != "missing" else "manual_gpu_evidence_required",
        "family": family,
        "intent": _rerun_intent(status, reasons),
        "recommended_command_profile": command_profile,
        "source_candidate_case_id": str(preferred.get("case_id") or ""),
        "source_candidate_kind": str(preferred.get("kind") or ""),
        "blocked_reason_ids": list(reasons)[:16],
        "next_actions": list(next_actions)[:12],
        "candidate_axes": _compact_axes(axes),
        "requires_gpu_if_executed": True,
        "manual_start_required": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "post_run_review_required": True,
        "not_release_evidence_until_rebuilt": True,
        "rebuild_required_after_success": [
            "current_combined/evidence_pack.json",
            "current_combined/natural_load_canary.json",
            "current_combined/release_claims.json",
        ],
    }


def _blocker_summary(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for family in families:
        for reason in _string_list(family.get("blocked_reasons")):
            category = _reason_category(reason)
            entry = categories.setdefault(category, {"count": 0, "reasons": []})
            entry["count"] = int(entry["count"]) + 1
            if reason not in entry["reasons"]:
                entry["reasons"].append(reason)
    return {
        category: {
            "count": entry["count"],
            "reasons": sorted(entry["reasons"]),
        }
        for category, entry in sorted(categories.items())
    }


def _candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    axes = _axes(report)
    metrics = _metrics(report)
    comparison = _comparison(report)
    gates: dict[str, dict[str, Any]] = {}
    blocked: list[str] = []
    for name, (status, reasons) in {
        "cache_inventory": _cache_gate(report, axes),
        "loss_stability": _loss_gate(report),
        "vram": _vram_gate(report, metrics),
        "action_boundary": _action_boundary_gate(report),
        "schema": _schema_gate(report),
        "case_specific_wording": _wording_gate(report),
    }.items():
        gates[name] = {"status": status, "reasons": reasons}
        blocked.extend(reasons)
    return {
        "case_id": str(report.get("case_id") or ""),
        "family": _family(report.get("family")),
        "kind": _candidate_kind(report),
        "status": str(report.get("status") or ""),
        "eligible": not blocked,
        "blocked_reasons": blocked,
        "gates": gates,
        "metrics": {
            "steady_samples_per_second": metrics.get("steady_samples_per_second"),
            "data_wait_share": metrics.get("data_wait_share"),
            "peak_vram_mb": metrics.get("peak_vram_mb"),
        },
        "comparison": {
            "data_wait_share_before": comparison.get("data_wait_share_before"),
            "data_wait_share_after": comparison.get("data_wait_share_after"),
            "steady_samples_per_second_gain_pct": comparison.get("steady_samples_per_second_gain_pct"),
            "final_loss_delta": comparison.get("final_loss_delta"),
        },
        "matrix_axes": _compact_axes(axes),
        "action_boundary": {
            "release_scope": _mapping(report.get("release_claim")).get("scope"),
            "native_cache_mode": axes.get("native_cache_mode"),
            "cache_state": axes.get("cache_state"),
            "source_fixture": axes.get("source_fixture"),
        },
    }


def build_bubble_natural_load_canary_report(
    reports: Iterable[Mapping[str, Any]],
    *,
    required_families: Sequence[str] = DEFAULT_CANARY_FAMILIES,
) -> dict[str, Any]:
    """Validate natural-load evidence before it is promoted as release-ready."""

    candidates = [
        _candidate(report)
        for report in reports
        if _candidate_kind(_mapping(report))
    ]
    required = [_family(item) for item in required_families]
    families: list[dict[str, Any]] = []
    for family in required:
        family_candidates = [item for item in candidates if item.get("family") == family]
        accepted = [item for item in family_candidates if item.get("eligible")]
        if accepted:
            status = "ready"
            reasons: list[str] = []
        elif family_candidates:
            status = "blocked"
            reasons = sorted({reason for item in family_candidates for reason in item.get("blocked_reasons", [])})
        else:
            status = "missing"
            reasons = ["natural_load_canary_evidence_missing"]
        next_actions = _next_actions(status, reasons)
        family_record = {
            "family": family,
            "status": status,
            "candidate_count": len(family_candidates),
            "accepted_candidate_count": len(accepted),
            "blocked_reasons": reasons,
            "blocking_categories": sorted({_reason_category(reason) for reason in reasons}),
            "next_actions": next_actions,
            "gpu_rerun_plan": _family_gpu_rerun_plan(
                family=family,
                status=status,
                reasons=reasons,
                candidates=family_candidates,
                next_actions=next_actions,
            ),
            "accepted_candidates": accepted[:5],
            "candidates": family_candidates[:10],
        }
        families.append(family_record)
    ready = all(item.get("status") == "ready" for item in families)
    missing_families = [item["family"] for item in families if item.get("status") == "missing"]
    blocked_families = [
        item["family"] for item in families if item.get("status") in {"blocked", "missing"}
    ]
    return {
        "schema_version": 1,
        "report": NATURAL_LOAD_CANARY_REPORT,
        "status": "ready" if ready else "blocked_pending_canary",
        "required_families": required,
        "release_candidate_allowed": ready,
        "family_count": len(families),
        "ready_family_count": sum(1 for item in families if item.get("status") == "ready"),
        "missing_families": missing_families,
        "blocked_families": blocked_families,
        "blocked_family_count": len(blocked_families),
        "blocked_family_statuses": {
            item["family"]: str(item.get("status") or "")
            for item in families
            if item.get("status") in {"blocked", "missing"}
        },
        "families": families,
        "blocker_summary": _blocker_summary(families),
        "gpu_rerun_plan": {
            "status": "not_needed" if ready else "manual_gpu_rerun_required",
            "manual_ready_family_count": sum(
                1
                for item in families
                if str(_mapping(item.get("gpu_rerun_plan")).get("status") or "").startswith("manual_gpu")
            ),
            "safe_to_auto_start": False,
            "release_claim_allowed_after_success": False,
            "families": [
                dict(_mapping(item.get("gpu_rerun_plan")))
                for item in families
                if str(_mapping(item.get("gpu_rerun_plan")).get("status") or "") != "not_needed"
            ],
        },
        "candidate_count": len(candidates),
        "accepted_candidate_count": sum(1 for item in candidates if item.get("eligible")),
    }


__all__ = [
    "DEFAULT_CANARY_FAMILIES",
    "NATURAL_LOAD_CANARY_REPORT",
    "build_bubble_natural_load_canary_report",
]
