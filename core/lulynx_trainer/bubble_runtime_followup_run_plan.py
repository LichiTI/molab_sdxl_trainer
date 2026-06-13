"""Command planning for bubble runtime follow-up evidence runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_runtime_followup_scout_plans import build_source_axis_scout_guidance, build_source_axis_scout_plans


FOLLOWUP_RUN_PLAN_REPORT = "bubble_runtime_followup_run_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
PROFILE_RELEASE_RELEVANT_CONSERVATIVE = "release_relevant_conservative"
PROFILE_DIAGNOSTIC_ONLY_PROBE = "diagnostic_only_probe"
PROFILE_AGGRESSIVE_SCAFFOLD_BLOCKED = "aggressive_scaffold_blocked"

_HOLD_CATEGORY_REASONS = {
    "cache_readiness": "cache_or_probe_blocker",
    "data_wait_gate": "data_wait_gate",
    "diagnostic_only": "diagnostic_only_evidence",
    "loss_guardrail": "loss_guardrail",
    "release_claim_gate": "release_claim_gate",
    "throughput_gate": "throughput_gate",
    "vram_guardrail": "vram_guardrail",
}

_HOLD_REASON_KEYWORDS = {
    "after_data_wait_increased": (
        "after_data_wait",
        "data_wait_not_reduced",
        "data_wait_worsened",
        "data_wait_delta_positive",
    ),
    "baseline_data_wait_insufficient": (
        "before_data_wait_below_threshold",
        "insufficient_baseline_data_wait",
        "baseline_data_wait",
        "natural_data_wait_below_threshold",
    ),
    "cache_or_probe_blocker": ("cache_probe_only", "family_cache", "blocked_invalid_family_cache", "cache_not_ready"),
    "diagnostic_only_evidence": ("diagnostic_only",),
    "loss_guardrail": ("loss_guardrail", "loss_regressed", "loss_stability"),
    "throughput_regressed": (
        "negative_throughput",
        "speedup_negative",
        "throughput_gain_below_threshold",
        "throughput_regressed",
    ),
    "vram_guardrail": ("vram_guardrail", "peak_vram", "vram"),
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _families_requiring_runs(followup_plan: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    for item in followup_plan.get("items", []):
        mapped = _mapping(item)
        family = str(mapped.get("family") or "").strip().lower().replace("-", "_")
        categories = set(_string_list([mapped.get("category")]))
        reasons = set(_string_list(mapped.get("reasons")))
        if family and (
            "action_boundary" in categories
            or "cache_readiness" in categories
            or "release_claim_gate" in categories
            or "dataloader_rebuild_epoch_boundary_action_missing" in reasons
            or any("family_cache" in reason or "cache_inventory" in reason for reason in reasons)
            or "release_claim_not_eligible" in reasons
        ):
            families.add("newbie" if family == "dit" else family)
    return families


def _scout_candidate_families(source_axis_scout: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    for raw in source_axis_scout.get("ranked_axes", []):
        axis = _mapping(raw)
        family = _family_key(axis.get("family"))
        if family not in {"anima", "sdxl"}:
            continue
        if str(axis.get("state") or "") != "candidate" or not bool(axis.get("cache_ready")):
            continue
        recommendation = str(axis.get("recommendation") or "")
        if recommendation in {
            "run_conservative_recheck_only_aggressive_held",
            "run_release_relevant_conservative_recheck",
        }:
            families.add(family)
    return families


def _scout_guidance_families(source_axis_scout: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    for raw in source_axis_scout.get("family_summaries", []):
        summary = _mapping(raw)
        family = _family_key(summary.get("family"))
        if not family:
            continue
        if int(summary.get("candidate_count") or 0) > 0:
            continue
        source_axis_state = str(summary.get("source_axis_state") or "")
        if source_axis_state in {
            "exhausted_current_source_axis",
            "no_ready_source_axis",
            "no_source_axes_found",
        }:
            families.add(family)
    return families


def _family_key(raw: Any) -> str:
    family = str(raw or "").strip().lower().replace("-", "_")
    return "newbie" if family == "dit" else family


def _family_items(followup_plan: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for raw in followup_plan.get("items", []):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if family:
            by_family.setdefault(family, []).append(item)
    return by_family


def _hold_reason_codes(items: Sequence[Mapping[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for item in items:
        category = str(item.get("category") or "").strip().lower()
        if category in _HOLD_CATEGORY_REASONS:
            codes.add(_HOLD_CATEGORY_REASONS[category])
        for reason in _string_list(item.get("reasons")):
            lowered = reason.strip().lower()
            for code, keywords in _HOLD_REASON_KEYWORDS.items():
                if any(keyword in lowered for keyword in keywords):
                    codes.add(code)
    return sorted(codes)


def _family_policy(followup_plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for family, items in _family_items(followup_plan).items():
        hold_reasons = _hold_reason_codes(items)
        hold = bool(hold_reasons)
        policies[family] = {
            "family": family,
            "recent_failure_hold": hold,
            "hold_scope": ["aggressive_policy_scaffold"] if hold else [],
            "hold_reason_codes": hold_reasons,
            "allowed_profiles": [
                PROFILE_RELEASE_RELEVANT_CONSERVATIVE,
                PROFILE_DIAGNOSTIC_ONLY_PROBE,
            ],
            "blocked_profiles": [PROFILE_AGGRESSIVE_SCAFFOLD_BLOCKED] if hold else [],
            "source_item_count": len(items),
        }
    return policies


def _ensure_default_policies(policies: dict[str, dict[str, Any]], families: set[str]) -> dict[str, dict[str, Any]]:
    result = {family: dict(policy) for family, policy in policies.items()}
    for family in sorted(families):
        result.setdefault(
            family,
            {
                "family": family,
                "recent_failure_hold": False,
                "hold_scope": [],
                "hold_reason_codes": [],
                "allowed_profiles": [
                    PROFILE_RELEASE_RELEVANT_CONSERVATIVE,
                    PROFILE_DIAGNOSTIC_ONLY_PROBE,
                ],
                "blocked_profiles": [],
                "source_item_count": 0,
            },
        )
    return result


def _command_profile(plan: Mapping[str, Any]) -> str:
    if bool(plan.get("diagnostic_only")):
        return PROFILE_DIAGNOSTIC_ONLY_PROBE
    return PROFILE_RELEASE_RELEVANT_CONSERVATIVE


def _annotate_command(plan: dict[str, Any], family_policies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    item = dict(plan)
    profile = _command_profile(item)
    family_policy = dict(_mapping(family_policies.get(str(item.get("family") or ""))))
    hold = bool(family_policy.get("recent_failure_hold"))
    item["profile"] = profile
    item["recent_failure_hold"] = {
        "active": hold,
        "scope": list(family_policy.get("hold_scope") or []),
        "reason_codes": list(family_policy.get("hold_reason_codes") or []),
    }
    if profile == PROFILE_DIAGNOSTIC_ONLY_PROBE:
        item["policy_state"] = "diagnostic_only_not_release_claim"
    elif hold:
        item["policy_state"] = "conservative_recheck_allowed_aggressive_held"
    else:
        item["policy_state"] = "release_relevant_conservative"
    return item


def _aggressive_scaffolds(families: set[str], family_policies: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    scaffolds: list[dict[str, Any]] = []
    for family in sorted(families):
        policy = dict(_mapping(family_policies.get(family)))
        hold = bool(policy.get("recent_failure_hold"))
        blocked_by = ["default_disabled", "requires_manual_review", "missing_release_gate_evidence"]
        if hold:
            blocked_by.insert(0, "recent_failure_hold")
        scaffolds.append(
            {
                "id": f"{family}_workers4_prefetch8_persistent_aggressive_scaffold",
                "family": family,
                "profile": PROFILE_AGGRESSIVE_SCAFFOLD_BLOCKED,
                "enabled": False,
                "release_relevant": False,
                "diagnostic_only": False,
                "requires_gpu_if_enabled": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
                "not_release_evidence": True,
                "manual_start_required": True,
                "blocked_by": blocked_by,
                "recent_failure_hold": {
                    "active": hold,
                    "scope": list(policy.get("hold_scope") or []),
                    "reason_codes": list(policy.get("hold_reason_codes") or []),
                },
                "candidate_overlay": {
                    "cached_dataloader_workers": 4,
                    "cached_dataloader_prefetch_factor": 8,
                    "cached_dataloader_persistent_workers": True,
                },
                "rationale": (
                    "Aggressive data-supply expansion is listed for review only; it must not run until "
                    "conservative release gates and family-specific failure holds are cleared."
                ),
            }
        )
    return scaffolds


def _profile_summary(
    commands: Sequence[Mapping[str, Any]],
    scaffolds: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {
        PROFILE_RELEASE_RELEVANT_CONSERVATIVE: 0,
        PROFILE_DIAGNOSTIC_ONLY_PROBE: 0,
        PROFILE_AGGRESSIVE_SCAFFOLD_BLOCKED: len(scaffolds),
    }
    for command in commands:
        profile = str(command.get("profile") or "")
        if profile in counts:
            counts[profile] += 1
    return {
        "profiles": counts,
        "recent_failure_hold_family_count": sum(1 for item in policies.values() if item.get("recent_failure_hold")),
        "aggressive_scaffold_count": len(scaffolds),
    }


def _execution_surface_summary(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed_ids: list[str] = []
    rerun_blocked_ids: list[str] = []
    active_release_ids: list[str] = []
    diagnostic_ids: list[str] = []
    blocked_ids: list[str] = []
    for command in commands:
        command_id = str(command.get("id") or "")
        completed = bool(_mapping(command.get("existing_evidence")).get("completed"))
        diagnostic = bool(command.get("diagnostic_only"))
        release_relevant = bool(command.get("release_relevant"))
        if completed:
            completed_ids.append(command_id)
        if bool(command.get("do_not_rerun_without_new_axis")):
            rerun_blocked_ids.append(command_id)
        if completed:
            continue
        if diagnostic:
            diagnostic_ids.append(command_id)
        elif release_relevant:
            active_release_ids.append(command_id)
        else:
            blocked_ids.append(command_id)
    if active_release_ids:
        status = "release_relevant_manual_ready"
    elif diagnostic_ids or completed_ids or blocked_ids:
        status = "blocked_or_diagnostic_only"
    else:
        status = "no_followup_runs_needed"
    return {
        "execution_surface_status": status,
        "active_release_relevant_command_count": len(active_release_ids),
        "active_release_relevant_command_ids": active_release_ids,
        "diagnostic_manual_ready_command_count": len(diagnostic_ids),
        "diagnostic_manual_ready_command_ids": diagnostic_ids,
        "completed_existing_command_ids": completed_ids,
        "rerun_blocked_without_new_axis_command_ids": rerun_blocked_ids,
        "blocked_nonrelease_command_count": len(blocked_ids),
        "blocked_nonrelease_command_ids": blocked_ids,
    }


def _path_text(path: Path) -> str:
    return str(path)


def _existing_evidence(out_dir: Any) -> dict[str, Any]:
    text = str(out_dir or "").strip()
    if not text:
        return {"completed": False, "evidence_paths": [], "missing_paths": ["out_dir_missing"]}
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
    }


def _default_python_executable(repo_root: Path) -> Path:
    flashattention = repo_root / "backend" / "env" / "python-flashattention" / "python.exe"
    if flashattention.is_file():
        return flashattention
    return repo_root / "backend" / "env" / "python_launcher" / "python.exe"


def _real_material_command(
    repo_root: Path,
    *,
    python_executable: Path | None = None,
    family: str,
    source_data: Path,
    sample_offset: int,
    out_dir: Path,
    flags: Sequence[str],
) -> list[str]:
    return [
        _path_text(python_executable or _default_python_executable(repo_root)),
        _path_text(repo_root / "devtools" / "run_bubble_real_material_canary.py"),
        "--family",
        family,
        "--source-data",
        _path_text(source_data),
        "--sample-offset",
        str(max(int(sample_offset), 0)),
        *list(flags),
        "--out-dir",
        _path_text(out_dir),
    ]


def _sdxl_plans(repo_root: Path, source_data: Path, out_root: Path, python_executable: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": "sdxl_offset36_long_window_1024",
            "family": "sdxl",
            "priority": 10,
            "release_relevant": True,
            "diagnostic_only": False,
            "sample_offset": 36,
            "source_data": _path_text(source_data),
            "out_dir": _path_text(out_root / "real_material_canary_sdxl_offset36_long_window_p60_followup"),
            "rationale": (
                "Re-run the closest SDXL window with a longer steady window and the "
                "release 1024 axis before changing thresholds."
            ),
            "command": _real_material_command(
                repo_root,
                python_executable=python_executable,
                family="sdxl",
                source_data=source_data,
                sample_offset=36,
                out_dir=out_root / "real_material_canary_sdxl_offset36_long_window_p60_followup",
                flags=[
                    "--sdxl-resolution",
                    "1024",
                    "--sdxl-samples",
                    "8",
                    "--sdxl-steps",
                    "16",
                    "--sdxl-warmup",
                    "4",
                    "--sdxl-tune-interval",
                    "4",
                    "--min-throughput-gain",
                    "0.03",
                ],
            ),
        },
        {
            "id": "sdxl_offset32_samples12_1024",
            "family": "sdxl",
            "priority": 20,
            "release_relevant": True,
            "diagnostic_only": False,
            "sample_offset": 32,
            "source_data": _path_text(source_data),
            "out_dir": _path_text(out_root / "real_material_canary_sdxl_offset32_samples12_p60_followup"),
            "rationale": "Expand the offset32 overlap window to test whether sample pressure crosses the baseline data-wait gate.",
            "command": _real_material_command(
                repo_root,
                python_executable=python_executable,
                family="sdxl",
                source_data=source_data,
                sample_offset=32,
                out_dir=out_root / "real_material_canary_sdxl_offset32_samples12_p60_followup",
                flags=[
                    "--sdxl-resolution",
                    "1024",
                    "--sdxl-samples",
                    "12",
                    "--sdxl-steps",
                    "16",
                    "--sdxl-warmup",
                    "4",
                    "--sdxl-tune-interval",
                    "4",
                    "--min-throughput-gain",
                    "0.03",
                ],
            ),
        },
        {
            "id": "sdxl_offset36_res512_diagnostic",
            "family": "sdxl",
            "priority": 30,
            "release_relevant": False,
            "diagnostic_only": True,
            "sample_offset": 36,
            "source_data": _path_text(source_data),
            "out_dir": _path_text(out_root / "real_material_canary_sdxl_offset36_res512_diag_p60_followup"),
            "rationale": "Diagnostic bridge for checking whether 1024 compute work hides SDXL data wait.",
            "command": _real_material_command(
                repo_root,
                python_executable=python_executable,
                family="sdxl",
                source_data=source_data,
                sample_offset=36,
                out_dir=out_root / "real_material_canary_sdxl_offset36_res512_diag_p60_followup",
                flags=[
                    "--diagnostic-only",
                    "--sdxl-resolution",
                    "512",
                    "--sdxl-samples",
                    "8",
                    "--sdxl-steps",
                    "16",
                    "--sdxl-warmup",
                    "4",
                    "--sdxl-tune-interval",
                    "4",
                    "--min-throughput-gain",
                    "0.03",
                ],
            ),
        },
    ]


def _newbie_axis_label(source_data: Path, warm_cache_inventory: Mapping[str, Any] | None = None) -> str:
    inventory = _mapping(warm_cache_inventory)
    offset = inventory.get("selected_axis_sample_offset")
    try:
        offset_value = int(offset)
    except (TypeError, ValueError, OverflowError):
        offset_value = None
    if offset_value is not None and offset_value >= 0:
        return f"offset{offset_value}_warm_cache"
    name = source_data.parent.name.lower() if source_data.name.lower() == "source_data" else source_data.name.lower()
    if "offset32" in name:
        return "offset32_no_skip_clip"
    return "warm_cache"


def _selected_newbie_source_from_inventory(warm_cache_inventory: Mapping[str, Any] | None) -> Path | None:
    inventory = _mapping(warm_cache_inventory)
    if not bool(inventory.get("selected_axis_cache_ready")):
        return None
    source = str(inventory.get("selected_axis_root") or inventory.get("prepared_source_data") or "").strip()
    if not source:
        return None
    return Path(source)


def _newbie_plans(
    repo_root: Path,
    source_data: Path,
    out_root: Path,
    python_executable: Path,
    *,
    warm_cache_inventory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    axis_label = _newbie_axis_label(source_data, warm_cache_inventory)
    source_axis = {
        "source": "newbie_warm_cache_inventory" if _mapping(warm_cache_inventory) else "planner_default",
        "selected_axis_kind": str(_mapping(warm_cache_inventory).get("selected_axis_kind") or ""),
        "selected_axis_sample_offset": _mapping(warm_cache_inventory).get("selected_axis_sample_offset"),
        "selected_axis_repair_produced": bool(_mapping(warm_cache_inventory).get("selected_axis_repair_produced")),
        "source_data_original": str(_mapping(warm_cache_inventory).get("source_data_original") or ""),
    }
    return [
        {
            "id": f"newbie_{axis_label}_long_window",
            "family": "newbie",
            "priority": 10,
            "release_relevant": True,
            "diagnostic_only": False,
            "sample_offset": 0,
            "source_data": _path_text(source_data),
            "source_axis": source_axis,
            "out_dir": _path_text(out_root / f"real_material_canary_newbie_{axis_label}_long_window_p60_followup"),
            "rationale": "Re-run Newbie on the selected warm-cache source with a longer steady window.",
            "command": _real_material_command(
                repo_root,
                python_executable=python_executable,
                family="newbie",
                source_data=source_data,
                sample_offset=0,
                out_dir=out_root / f"real_material_canary_newbie_{axis_label}_long_window_p60_followup",
                flags=[
                    "--newbie-resolution",
                    "64",
                    "--newbie-samples",
                    "8",
                    "--newbie-steps",
                    "16",
                    "--newbie-warmup",
                    "4",
                    "--newbie-tune-interval",
                    "4",
                    "--min-throughput-gain",
                    "0.03",
                ],
            ),
        },
        {
            "id": f"newbie_{axis_label}_batch1_diagnostic",
            "family": "newbie",
            "priority": 20,
            "release_relevant": False,
            "diagnostic_only": True,
            "sample_offset": 0,
            "source_data": _path_text(source_data),
            "source_axis": source_axis,
            "out_dir": _path_text(out_root / f"real_material_canary_newbie_{axis_label}_batch1_diag_p60_followup"),
            "rationale": "Diagnostic underfilled-workload probe to see whether Newbie data wait is masked by compute.",
            "command": _real_material_command(
                repo_root,
                python_executable=python_executable,
                family="newbie",
                source_data=source_data,
                sample_offset=0,
                out_dir=out_root / f"real_material_canary_newbie_{axis_label}_batch1_diag_p60_followup",
                flags=[
                    "--diagnostic-only",
                    "--newbie-resolution",
                    "64",
                    "--newbie-samples",
                    "8",
                    "--newbie-steps",
                    "16",
                    "--newbie-warmup",
                    "4",
                    "--newbie-tune-interval",
                    "4",
                    "--newbie-batch",
                    "1",
                    "--min-throughput-gain",
                    "0.03",
                ],
            ),
        },
    ]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _natural_load_gpu_rerun_rows(followup_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gpu_plan = _mapping(followup_plan.get("gpu_rerun_plan"))
    rows = gpu_plan.get("families")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [_mapping(item) for item in rows if _mapping(item)]
    return []


def _natural_load_axis_flags(family: str, axes: Mapping[str, Any]) -> list[str]:
    resolution = _safe_int(axes.get("resolution"), 1024 if family == "sdxl" else 64)
    samples = max(_safe_int(axes.get("samples"), 8), 8)
    candidate_steps = _safe_int(axes.get("steps"), 16)
    steps = max(candidate_steps * 2, 32)
    warmup = max(min(steps // 4, 8), 4)
    tune_interval = max(min(steps // 4, 8), 4)
    prefix = "newbie" if family == "newbie" else family
    flags = [
        f"--{prefix}-resolution",
        str(resolution),
        f"--{prefix}-samples",
        str(samples),
        f"--{prefix}-steps",
        str(steps),
        f"--{prefix}-warmup",
        str(warmup),
        f"--{prefix}-tune-interval",
        str(tune_interval),
        "--min-throughput-gain",
        "0.03",
    ]
    batch = _safe_int(axes.get("train_batch_size"), 0)
    if batch > 0:
        flags.extend([f"--{prefix}-batch", str(batch)])
    return flags


def _natural_load_gpu_rerun_plans(
    repo_root: Path,
    followup_plan: Mapping[str, Any],
    *,
    source_data: Path,
    newbie_source_data: Path,
    out_root: Path,
    python_executable: Path,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for row in _natural_load_gpu_rerun_rows(followup_plan):
        family = _family_key(row.get("family"))
        if family not in {"sdxl", "newbie"}:
            continue
        if str(row.get("status") or "") not in {"manual_gpu_rerun_required", "manual_gpu_evidence_required"}:
            continue
        axes = _mapping(row.get("candidate_axes"))
        resolution = _safe_int(axes.get("resolution"), 1024 if family == "sdxl" else 64)
        steps = max(_safe_int(axes.get("steps"), 16) * 2, 32)
        source = newbie_source_data if family == "newbie" else source_data
        out_dir = out_root / f"real_material_canary_{family}_natural_load_epoch_boundary_res{resolution}_steps{steps}_p60_followup"
        plans.append(
            {
                "id": f"{family}_natural_load_epoch_boundary_res{resolution}_steps{steps}",
                "family": family,
                "priority": 1,
                "release_relevant": True,
                "diagnostic_only": False,
                "sample_offset": 0,
                "source_data": _path_text(source),
                "out_dir": _path_text(out_dir),
                "rationale": (
                    "Natural-load canary requested an epoch-boundary DataLoader rebuild follow-up; "
                    "this command uses a longer window and a fresh out_dir so completed low-baseline axes are not rerun."
                ),
                "natural_load_gpu_rerun_plan": {
                    "intent": str(row.get("intent") or ""),
                    "recommended_command_profile": str(row.get("recommended_command_profile") or ""),
                    "source_candidate_case_id": str(row.get("source_candidate_case_id") or ""),
                    "source_candidate_kind": str(row.get("source_candidate_kind") or ""),
                    "blocked_reason_ids": _string_list(row.get("blocked_reason_ids"))[:16],
                    "candidate_axes": dict(axes),
                    "rebuild_required_after_success": _string_list(row.get("rebuild_required_after_success")),
                    "safe_to_auto_start": False,
                    "release_claim_allowed_after_success": False,
                },
                "command": _real_material_command(
                    repo_root,
                    python_executable=python_executable,
                    family=family,
                    source_data=source,
                    sample_offset=0,
                    out_dir=out_dir,
                    flags=_natural_load_axis_flags(family, axes),
                ),
            }
        )
    return plans


def build_bubble_runtime_followup_run_plan(
    followup_plan: Mapping[str, Any],
    *,
    repo_root: Path,
    source_data: Path | None = None,
    newbie_source_data: Path | None = None,
    out_root: Path | None = None,
    source_axis_scout: Mapping[str, Any] | None = None,
    warm_cache_inventory: Mapping[str, Any] | None = None,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Build concrete follow-up commands from a machine-readable follow-up plan."""

    repo = Path(repo_root)
    out_base = out_root or repo / "devtools" / "benchmark_evidence" / "bubble_runtime"
    python = python_executable or _default_python_executable(repo)
    default_source = source_data or repo / "sucai" / "6_lulu"
    inventory_newbie_source = _selected_newbie_source_from_inventory(warm_cache_inventory)
    default_newbie_source = (
        newbie_source_data
        or inventory_newbie_source
        or repo
        / "devtools"
        / "benchmark_evidence"
        / "bubble_runtime"
        / "newbie_real_material_cache_ready_offset32_no_skip_clip"
        / "source_data"
    )
    families = _families_requiring_runs(followup_plan)
    families.update(_scout_candidate_families(_mapping(source_axis_scout)))
    families.update(_scout_guidance_families(_mapping(source_axis_scout)))
    family_policies = _ensure_default_policies(_family_policy(followup_plan), families)
    commands: list[dict[str, Any]] = []
    natural_load_gpu_rerun_plans = _natural_load_gpu_rerun_plans(
        repo,
        followup_plan,
        source_data=default_source,
        newbie_source_data=default_newbie_source,
        out_root=out_base,
        python_executable=python,
    )
    commands.extend(natural_load_gpu_rerun_plans)
    if "sdxl" in families:
        commands.extend(_sdxl_plans(repo, default_source, out_base, python))
    if "newbie" in families:
        commands.extend(
            _newbie_plans(
                repo,
                default_newbie_source,
                out_base,
                python,
                warm_cache_inventory=warm_cache_inventory if inventory_newbie_source is not None else None,
            )
        )
    scout_plans = build_source_axis_scout_plans(
        repo,
        _mapping(source_axis_scout),
        out_base,
        families=families,
        existing_commands=commands,
        real_material_command=lambda repo_root, **kwargs: _real_material_command(
            repo_root,
            python_executable=python,
            **kwargs,
        ),
    )
    scout_guidance = build_source_axis_scout_guidance(_mapping(source_axis_scout), families=families)
    commands.extend(scout_plans)
    commands = [_annotate_command(item, family_policies) for item in commands]
    commands.sort(key=lambda item: (str(item["family"]), int(item["priority"]), str(item["id"])))
    for index, item in enumerate(commands, start=1):
        item["run_order"] = index
        item["python_executable"] = _path_text(python)
        item["requires_gpu_if_executed"] = True
        item["manual_start_required"] = True
        item["safe_to_auto_start"] = False
        item["release_claim_allowed_after_success"] = False
        item["not_release_evidence"] = True
        item["post_run_review_required"] = True
        item["dry_run_command"] = [*item["command"], "--dry-run"]
        existing = _existing_evidence(item.get("out_dir"))
        item["existing_evidence"] = existing
        item["do_not_rerun_without_new_axis"] = bool(existing["completed"])
    scaffolds = _aggressive_scaffolds(families, family_policies)
    completed_existing_command_count = sum(1 for item in commands if item["existing_evidence"]["completed"])
    execution_surface = _execution_surface_summary(commands)
    unsafe_command_ids = [
        str(item.get("id") or "")
        for item in commands
        if bool(item.get("safe_to_auto_start"))
        or bool(item.get("release_claim_allowed_after_success"))
        or not bool(item.get("manual_start_required"))
        or not bool(item.get("not_release_evidence"))
    ]
    unsafe_scaffold_ids = [
        str(item.get("id") or "")
        for item in scaffolds
        if bool(item.get("safe_to_auto_start"))
        or bool(item.get("release_claim_allowed_after_success"))
        or not bool(item.get("manual_start_required"))
        or not bool(item.get("not_release_evidence"))
    ]

    return {
        "schema_version": 1,
        "report": FOLLOWUP_RUN_PLAN_REPORT,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_protected_followup_manual_gpu_run_plan",
        "status": "commands_planned" if commands else "no_followup_runs_needed",
        "not_release_evidence": True,
        "publishable": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "manual_start_required": bool(commands),
        "release_claim_allowed": False,
        "source_followup_report": str(followup_plan.get("report") or ""),
        "source_followup_status": str(followup_plan.get("status") or ""),
        "families": sorted(families),
        "python_executable": _path_text(python),
        "command_count": len(commands),
        "completed_existing_command_count": completed_existing_command_count,
        "rerun_blocked_without_new_axis_count": sum(
            1 for item in commands if item.get("do_not_rerun_without_new_axis")
        ),
        **execution_surface,
        "requires_gpu": bool(commands),
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success_command_count": sum(
            1 for item in commands if bool(item.get("release_claim_allowed_after_success"))
        ),
        "manual_start_required_command_count": sum(
            1 for item in commands if bool(item.get("manual_start_required"))
        ),
        "unsafe_command_count": len(unsafe_command_ids),
        "unsafe_command_ids": unsafe_command_ids,
        "unsafe_scaffold_count": len(unsafe_scaffold_ids),
        "unsafe_scaffold_ids": unsafe_scaffold_ids,
        "contract_ok": not unsafe_command_ids and not unsafe_scaffold_ids,
        "natural_load_gpu_rerun_command_count": len(natural_load_gpu_rerun_plans),
        "source_axis_scout_command_count": len(scout_plans),
        "source_axis_scout_guidance_count": len(scout_guidance),
        "source_axis_scout_report": str(_mapping(source_axis_scout).get("report") or ""),
        "source_axis_scout_guidance": scout_guidance,
        "profile_summary": _profile_summary(commands, scaffolds, family_policies),
        "family_policy": [family_policies[key] for key in sorted(family_policies)],
        "aggressive_scaffolds": scaffolds,
        "aggressive_scaffold_count": len(scaffolds),
        "commands": commands,
    }


__all__ = [
    "FOLLOWUP_RUN_PLAN_REPORT",
    "PROFILE_AGGRESSIVE_SCAFFOLD_BLOCKED",
    "PROFILE_DIAGNOSTIC_ONLY_PROBE",
    "PROFILE_RELEASE_RELEVANT_CONSERVATIVE",
    "build_bubble_runtime_followup_run_plan",
]
