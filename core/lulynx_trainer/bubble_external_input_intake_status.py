"""External-input intake registry for GPU-bubble blocked evidence routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_post_input_refresh_contract import POST_INPUT_REFRESH_SEQUENCE
from .bubble_sd15_lora512_release_gap_readiness import sd15_checkpoint_candidates


EXTERNAL_INPUT_INTAKE_STATUS_REPORT = "bubble_external_input_intake_registry_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _norm_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _source_current_roots(source_axis_requirement: Mapping[str, Any]) -> list[str]:
    roots: set[str] = set()
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        roots.update(root for root in _strings(item.get("current_source_roots")) if root)
    return sorted(roots)


def _known_identity_roots(source_cache_axis_identity_registry: Mapping[str, Any]) -> list[str]:
    roots: set[str] = set()
    for raw in _list(source_cache_axis_identity_registry.get("rows")):
        item = _mapping(raw)
        source_kinds = set(_strings(item.get("source_kinds")))
        source_kind = str(item.get("source_kind") or "")
        known_root = bool(
            item.get("current_source_root")
            or item.get("duplicate_or_stale_axis")
            or item.get("attempted_or_completed")
            or item.get("completed_existing_evidence")
            or source_kind == "current_source_root"
            or "current_source_root" in source_kinds
            or "warm_cache_inventory_axis" in source_kinds
        )
        if bool(item.get("new_source_root")) and not known_root:
            continue
        if source_kind == "new_source_root" and not known_root:
            continue
        root = str(item.get("source_root") or item.get("normalized_source_root") or "").strip()
        if root:
            roots.add(root)
            continue
        identity_key = str(item.get("identity_key") or "")
        parts = identity_key.split("|")
        if len(parts) >= 3 and parts[0] in {"root", "axis"} and parts[2]:
            roots.add(parts[2])
    return sorted(roots)


def _source_dirs(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        return []
    dirs = [item for item in source_root.iterdir() if item.is_dir()]
    if _image_count(source_root):
        dirs.append(source_root)
    return sorted(dirs, key=lambda item: str(item).lower())


def _sd15_status(model_root: Path) -> dict[str, Any]:
    candidates = sd15_checkpoint_candidates(model_root)
    existing = [path for path in candidates if path.is_file()]
    checkpoint = existing[0] if existing else None
    model_dir = checkpoint.parent if checkpoint else model_root / "sd15"
    return {
        "status": "checkpoint_available" if checkpoint else "checkpoint_missing",
        "model_dir": str(model_dir),
        "model_dir_exists": model_dir.is_dir(),
        "checkpoint_exists": checkpoint is not None,
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "checkpoint_count": len(existing),
        "checkpoint_candidates": [str(path) for path in candidates],
        "requires_external_input": checkpoint is None,
        "next_action": "refresh_sd15_readiness" if checkpoint else "provide_sd15_checkpoint",
    }


def _source_status(
    source_root: Path,
    source_axis_requirement: Mapping[str, Any],
    source_cache_axis_identity_registry: Mapping[str, Any],
) -> dict[str, Any]:
    current_roots = _source_current_roots(source_axis_requirement)
    known_roots = sorted({*current_roots, *_known_identity_roots(source_cache_axis_identity_registry)})
    current_keys = {_norm_path(root) for root in known_roots}
    dirs = _source_dirs(source_root)
    rows: list[dict[str, Any]] = []
    for item in dirs:
        count = _image_count(item)
        if count <= 0:
            continue
        duplicate = _norm_path(item) in current_keys
        rows.append(
            {
                "root": str(item),
                "image_count": count,
                "matches_current_source_axis": duplicate,
                "intake_status": "current_axis_duplicate" if duplicate else "new_root_available",
            }
        )
    new_roots = [item for item in rows if not bool(item["matches_current_source_axis"])]
    explicit_new_root_required = any(
        str(_mapping(item).get("requirement") or "") == "new_source_axis_required"
        or str(_mapping(item).get("source_axis_state") or "") == "exhausted_current_source_axis"
        for item in _list(source_axis_requirement.get("families"))
    )
    new_source_root_required = not bool(new_roots) and bool(explicit_new_root_required)
    return {
        "status": (
            "new_source_root_available"
            if new_roots
            else "new_source_root_missing"
            if new_source_root_required
            else "source_or_cache_axis_review_required"
        ),
        "source_root": str(source_root),
        "source_root_exists": source_root.is_dir(),
        "current_source_roots": current_roots,
        "known_source_roots": known_roots,
        "source_root_count": len(rows),
        "new_source_root_count": len(new_roots),
        "new_source_root_required": new_source_root_required,
        "roots": rows,
        "requires_external_input": new_source_root_required,
        "next_action": (
            "scan_new_source_root"
            if new_roots
            else "provide_new_source_or_cache_axis"
            if new_source_root_required
            else "prepare_warm_cache_or_caption_repair_axis"
        ),
    }


def _dedupe_baseline(source_axis_requirement: Mapping[str, Any]) -> dict[str, Any]:
    current_roots: set[str] = set()
    completed_command_ids: list[str] = []
    completed_out_dirs: list[str] = []
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        current_roots.update(_strings(item.get("current_source_roots")))
        run_readiness = _mapping(item.get("run_readiness"))
        completed_command_ids.extend(_strings(run_readiness.get("completed_command_ids")))
        completed_out_dirs.extend(_strings(run_readiness.get("completed_out_dirs")))
    return {
        "current_source_roots": sorted(root for root in current_roots if root),
        "completed_command_ids": sorted(set(completed_command_ids)),
        "completed_out_dirs": sorted(set(completed_out_dirs)),
        "do_not_rerun_completed_out_dirs_without_new_axis": True,
    }


def _intake_items(sd15: Mapping[str, Any], source_axis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "sd15_checkpoint",
            "status": "available" if bool(sd15.get("checkpoint_exists")) else "missing",
            "required_for": "sd15_lora_512_release_gap",
            "path": str(sd15.get("checkpoint_path") or ""),
            "next_action": str(sd15.get("next_action") or ""),
        },
        {
            "id": "new_source_root",
            "status": (
                "available"
                if _safe_int(source_axis.get("new_source_root_count"))
                else "missing"
                if bool(source_axis.get("new_source_root_required"))
                else "not_required_current_or_cache_axis_review"
            ),
            "required_for": "natural_load_canary_source_axis",
            "path": "",
            "next_action": str(source_axis.get("next_action") or ""),
        },
        {
            "id": "warm_cache_axis",
            "status": "pending_external_input",
            "required_for": "family_cache_ready_preflight",
            "path": "",
            "next_action": "prepare_or_register_family_cache_axis",
        },
        {
            "id": "caption_repair_axis",
            "status": "pending_external_input",
            "required_for": "caption_sample_coverage_gate",
            "path": "",
            "next_action": "repair_or_register_high_caption_coverage_axis",
        },
        {
            "id": "anima_source_or_cache_axis",
            "status": "pending_external_input",
            "required_for": "anima_saturation_boundary",
            "path": "",
            "next_action": "provide_anima_source_or_cache_axis",
        },
    ]


def _input_resolution_summary(
    *,
    sd15: Mapping[str, Any],
    source_axis: Mapping[str, Any],
    intake_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sd15_checkpoint_exists = bool(sd15.get("checkpoint_exists"))
    new_source_root_count = _safe_int(source_axis.get("new_source_root_count"))
    missing_inputs = [
        str(item.get("id") or "")
        for item in intake_items
        if item.get("status") in {"missing", "pending_external_input"} and item.get("id")
    ]
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "external_input_detected": sd15_checkpoint_exists or bool(new_source_root_count),
        "external_input_required": bool(missing_inputs),
        "missing_external_inputs": missing_inputs,
        "sd15_checkpoint_exists": sd15_checkpoint_exists,
        "sd15_checkpoint_path": str(sd15.get("checkpoint_path") or ""),
        "sd15_checkpoint_required": not sd15_checkpoint_exists,
        "new_source_root_count": new_source_root_count,
        "new_source_root_required": bool(source_axis.get("new_source_root_required")),
        "warm_cache_or_caption_repair_required": any(
            item in set(missing_inputs) for item in ("warm_cache_axis", "caption_repair_axis")
        ),
        "anima_source_or_cache_axis_required": "anima_source_or_cache_axis" in set(missing_inputs),
        "next_json_refresh_sequence": list(POST_INPUT_REFRESH_SEQUENCE),
        "next_manual_gpu_gate": "sd15_manual_ab_after_checkpoint_review",
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }


def _registration_slots(source_axis_requirement: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        family = str(item.get("family") or "")
        if not family:
            continue
        slots.append(
            {
                "family": family,
                "root": "",
                "sample_offset": None,
                "source_manifest_sha1": "",
                "status": "pending_external_input" if bool(item.get("requires_external_input")) else "manual_review",
                "requirement": str(item.get("requirement") or ""),
                "source_axis_state": str(item.get("source_axis_state") or ""),
                "blocked_actions": _strings(item.get("blocked_actions")),
            }
        )
    return slots


def _rescan_requests(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(item.get("family") or "") for item in slots if item.get("family")})
    return [
        {
            "id": "rescan_registered_source_root",
            "status": "waiting_for_registered_root",
            "families": families,
            "root_placeholder": "<new_source_root>",
            "samples": 8,
            "scan_windows": True,
            "thresholds": {
                "caption_sample_coverage_min": 0.875,
                "candidate_rank_score_min": 4.0,
            },
            "safe_to_auto_start": False,
        }
    ]


def _preflight_templates(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for slot in slots:
        family = str(slot.get("family") or "")
        if not family:
            continue
        templates.append(
            {
                "family": family,
                "candidate_root": "<new_source_root>",
                "sample_offset": "<sample_offset_from_rescan>",
                "source_manifest_sha1": "<source_manifest_sha1_from_rescan>",
                "builder": "devtools/build_bubble_source_cache_axis_admission_preflight.py",
                "safe_to_auto_start": False,
            }
        )
    return templates


def _pipeline_summary(pipeline_readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report": str(pipeline_readiness.get("report") or ""),
        "status": str(pipeline_readiness.get("status") or ""),
        "pipeline_complete": bool(pipeline_readiness.get("pipeline_complete")),
        "external_input_required": bool(pipeline_readiness.get("external_input_required")),
        "preflight_admitted": bool(pipeline_readiness.get("preflight_admitted")),
        "manual_canary_plan_ready": bool(pipeline_readiness.get("manual_canary_plan_ready")),
        "stage_count": _safe_int(pipeline_readiness.get("stage_count")),
        "stage_ok_count": _safe_int(pipeline_readiness.get("stage_ok_count")),
    }


def build_external_input_intake_status(
    *,
    repo_root: Path,
    source_axis_requirement: Mapping[str, Any] | None = None,
    pipeline_readiness: Mapping[str, Any] | None = None,
    source_cache_axis_identity_registry: Mapping[str, Any] | None = None,
    model_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Check whether external inputs have appeared without starting GPU work."""

    repo = Path(repo_root)
    models = model_root or repo / "models"
    sources = source_root or repo / "sucai"
    requirement = _mapping(source_axis_requirement)
    pipeline = _mapping(pipeline_readiness)
    identity_registry = _mapping(source_cache_axis_identity_registry)
    sd15 = _sd15_status(models)
    source_axis = _source_status(sources, requirement, identity_registry)
    registration_slots = _registration_slots(requirement)
    intake_items = _intake_items(sd15, source_axis)
    input_resolution_summary = _input_resolution_summary(
        sd15=sd15,
        source_axis=source_axis,
        intake_items=intake_items,
    )
    missing_external_inputs = list(input_resolution_summary["missing_external_inputs"])
    any_available = bool(sd15["checkpoint_exists"]) or bool(source_axis["new_source_root_count"])
    all_missing = bool(sd15["requires_external_input"]) and bool(source_axis["requires_external_input"])
    if any_available:
        status = "external_input_detected_review_required"
    elif all_missing:
        status = "external_input_missing"
    else:
        status = "manual_review_required"
    next_actions = []
    if sd15["checkpoint_exists"]:
        next_actions.append(
            {
                "id": "refresh_sd15_readiness_after_checkpoint_intake",
                "roadmap": ROADMAP,
                "status": "json_refresh_ready",
                "requires_gpu_if_executed": False,
                "not_release_evidence": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
            }
        )
    else:
        next_actions.append(
            {
                "id": "provide_sd15_checkpoint",
                "roadmap": ROADMAP,
                "status": "external_input_required",
                "requires_gpu_if_executed": False,
                "not_release_evidence": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
            }
        )
    if source_axis["new_source_root_count"]:
        next_actions.append(
            {
                "id": "scan_new_source_cache_axis",
                "roadmap": ROADMAP,
                "status": "json_scan_ready",
                "requires_gpu_if_executed": False,
                "not_release_evidence": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
            }
        )
    else:
        next_actions.append(
            {
                "id": "provide_new_source_or_cache_axis",
                "roadmap": ROADMAP,
                "status": "external_input_required",
                "requires_gpu_if_executed": False,
                "not_release_evidence": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
            }
        )
    if "anima_source_or_cache_axis" in missing_external_inputs:
        next_actions.append(
            {
                "id": "provide_anima_source_or_cache_axis",
                "roadmap": ROADMAP,
                "status": "external_input_required",
                "requires_gpu_if_executed": False,
                "not_release_evidence": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
            }
        )
    return {
        "schema_version": 1,
        "report": EXTERNAL_INPUT_INTAKE_STATUS_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "external_input_detected": any_available,
        "external_input_required": bool(missing_external_inputs),
        "sd15": sd15,
        "source_axis": source_axis,
        "pipeline_readiness": _pipeline_summary(pipeline),
        "intake_items": intake_items,
        "missing_external_input_count": len(missing_external_inputs),
        "missing_external_inputs": missing_external_inputs,
        "input_resolution_summary": input_resolution_summary,
        "candidate_registration_slots": registration_slots,
        "dedupe_baseline": _dedupe_baseline(requirement),
        "rescan_requests": _rescan_requests(registration_slots),
        "preflight_templates": _preflight_templates(registration_slots),
        "publishable": False,
        "next_actions": next_actions,
        "blocked_actions": [
            "auto_start_gpu_heavy_after_intake",
            "promote_intake_as_release_evidence",
            "skip_source_scan_scout_preflight_after_new_source",
            "skip_sd15_ab_evidence_after_checkpoint_intake",
            "skip_preflight_after_intake",
        ],
        "acceptance_gates": [
            "sd15_checkpoint_requires_sd15_readiness_refresh_before_manual_ab",
            "new_source_root_requires_source_scan_scout_requirement_preflight_chain",
            "intake_registry_is_not_release_evidence",
            "manual_gpu_run_requires_downstream_preflight_and_plan",
            "release_claim_requires_natural_load_and_release_claims_rebuild",
        ],
        "notes": [
            "This intake report is JSON-only and does not start GPU work.",
            "Detected inputs only unlock JSON refresh or protected manual planning; release evidence still requires the downstream gates.",
        ],
    }


__all__ = [
    "EXTERNAL_INPUT_INTAKE_STATUS_REPORT",
    "ROADMAP",
    "build_external_input_intake_status",
]
