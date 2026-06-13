"""Developer handoff packet for GPU-bubble external input blockers.

This JSON-only packet consolidates the existing intake registry and replay
plan into one handoff surface for SD15 checkpoint and source/cache-axis inputs.
It never starts training or GPU work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_post_input_refresh_contract import (
    POST_INPUT_REFRESH_SEQUENCE as REQUIRED_REFRESH_SEQUENCE,
)


REPORT = "bubble_external_input_handoff_packet_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
INTAKE_REPORT = "bubble_external_input_intake_registry_v0"
REPLAY_REPORT = "bubble_external_input_replay_plan_v0"
WARM_CACHE_DETECTED_STATUSES = {
    "warm_cache_axis_ready",
    "warm_cache_axis_completed_but_not_release_ready",
}


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


def _warm_cache_axis_detected(inventory: Mapping[str, Any]) -> bool:
    status = str(inventory.get("status") or "")
    return (
        bool(inventory.get("selected_axis_cache_ready"))
        or _safe_int(inventory.get("ready_axis_count")) > 0
        or _safe_int(inventory.get("completed_canary_axis_count")) > 0
        or status in WARM_CACHE_DETECTED_STATUSES
    )


def _input_items(
    intake: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    newbie_warm_cache_inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_ids = set(_strings(intake.get("missing_external_inputs")))
    source_cache_ready = bool(pipeline.get("preflight_admitted")) and bool(
        pipeline.get("manual_canary_plan_ready")
    )
    warm_cache_detected = _warm_cache_axis_detected(newbie_warm_cache_inventory)
    for raw in _list(intake.get("intake_items")):
        item = _mapping(raw)
        item_id = str(item.get("id") or "")
        status = str(item.get("status") or "")
        detected = status == "available" or (item_id == "warm_cache_axis" and warm_cache_detected)
        if item_id == "new_source_root" and _safe_int(intake.get("new_source_root_count")) > 0:
            detected = True
        if item_id in {"new_source_root", "warm_cache_axis"} and detected and status != "available":
            status = "detected_refresh_required"
        if item_id == "new_source_root" and detected:
            status = "detected_refresh_required"
        required = item_id in missing_ids or status in {"missing", "pending_external_input"}
        if item_id == "new_source_root":
            required = False if detected else item_id in missing_ids
        rows.append(
            {
                "id": item_id,
                "status": status,
                "required_for": str(item.get("required_for") or ""),
                "path": str(item.get("path") or ""),
                "next_action": str(item.get("next_action") or ""),
                "provided": status == "available",
                "detected": detected,
                "required": required,
            }
        )
    return rows


def _registration_slots(intake: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for raw in _list(intake.get("candidate_registration_slots")):
        item = _mapping(raw)
        slots.append(
            {
                "family": str(item.get("family") or ""),
                "root": str(item.get("root") or ""),
                "sample_offset": item.get("sample_offset"),
                "source_manifest_sha1": str(item.get("source_manifest_sha1") or ""),
                "status": str(item.get("status") or ""),
                "requirement": str(item.get("requirement") or ""),
                "source_axis_state": str(item.get("source_axis_state") or ""),
                "blocked_actions": _strings(item.get("blocked_actions")),
            }
        )
    return slots


def _replay_command_summary(replay: Mapping[str, Any]) -> dict[str, Any]:
    commands = [_mapping(item) for item in _list(replay.get("commands"))]
    status_counts: dict[str, int] = {}
    for command in commands:
        status = str(command.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "report": str(replay.get("report") or ""),
        "status": str(replay.get("status") or ""),
        "external_input_detected": bool(replay.get("external_input_detected")),
        "command_count": _safe_int(replay.get("command_count"), len(commands)),
        "ready_command_count": _safe_int(replay.get("ready_command_count")),
        "template_command_count": _safe_int(replay.get("template_command_count")),
        "status_counts": dict(sorted(status_counts.items())),
        "first_command_ids": [str(command.get("id") or "") for command in commands[:8]],
    }


def _refresh_commands(repo_root: Path, python_exe: str) -> list[dict[str, Any]]:
    py = _repo_path(repo_root, python_exe)
    runtime = "devtools/benchmark_evidence/bubble_runtime"
    rows = [
        {
            "id": "refresh_external_input_intake_registry",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_external_input_intake_status.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/external_input_intake_registry.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_external_input_replay_plan",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_external_input_replay_plan.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/external_input_replay_plan.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_sd15_lora512_release_gap_readiness",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_sd15_lora512_release_gap_readiness.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/sd15_lora512_release_gap_readiness.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_axis_scout",
            "command": [
                py,
                _repo_path(repo_root, "devtools/build_bubble_p60_source_axis_scout_from_intake.py"),
            ],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/p60_source_axis_scout.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_axis_requirement",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_p60_source_axis_requirement.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/p60_source_axis_requirement.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_warm_cache_inventory",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_warm_cache_inventory.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_warm_cache_inventory.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_external_input_admission",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_external_input_admission.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/external_input_admission.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_cache_axis_admission_preflight",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_admission_preflight.py")],
            "expected_outputs": [
                _repo_path(repo_root, f"{runtime}/source_cache_axis_admission_preflight.json")
            ],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_cache_axis_repair_plan",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_repair_plan.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/source_cache_axis_repair_plan.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_cache_axis_manual_canary_plan",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_manual_canary_plan.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/source_cache_axis_manual_canary_plan.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_post_manual_evidence_rebuild_plan",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_post_manual_evidence_rebuild_plan.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/post_manual_evidence_rebuild_plan.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_axis_freshness_dedupe_audit",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_axis_freshness_dedupe_audit.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/source_axis_freshness_dedupe_audit.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_cache_axis_identity_registry",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_identity_registry.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/source_cache_axis_identity_registry.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_source_cache_axis_pipeline_readiness",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_pipeline_readiness.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/source_cache_axis_pipeline_readiness.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_external_input_handoff_packet",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_external_input_handoff_packet.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/external_input_handoff_packet.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_blockskip_quality_followup_manifest",
            "command": [py, _repo_path(repo_root, "devtools/run_bubble_newbie_blockskip_quality_followup.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_blockskip_quality_followup_manifest.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_blockskip_quality_stability_review",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_blockskip_quality_stability_review.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_blockskip_quality_stability_review.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_blockskip_loss_curve_ab_evidence",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_blockskip_loss_curve_ab_evidence.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_blockskip_loss_curve_ab_evidence.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_blockskip_quality_semantic_evidence",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_blockskip_quality_semantic_evidence.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_blockskip_quality_semantic_evidence.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_internal_phase_diagnosis",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_internal_phase_diagnosis.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_internal_phase_diagnosis.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_natural_load_gate_semantics_review",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_natural_load_gate_semantics_review.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_natural_load_gate_semantics_review.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_compute_bound_gate_exit_policy",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_compute_bound_gate_exit_policy.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_compute_bound_gate_exit_policy.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_blockskip_quality_drift_review",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_blockskip_quality_drift_review.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_blockskip_quality_drift_review.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_tail8_attention_compute_review",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_tail8_attention_compute_review.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_tail8_attention_compute_review.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_tail8_forward_anomaly_review",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_tail8_forward_anomaly_review.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_tail8_forward_anomaly_review.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_newbie_tail8_seed2027_rerun_preflight",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_newbie_tail8_seed2027_rerun_preflight.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/newbie_tail8_seed2027_rerun_preflight.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_gpu_bubble_readiness_next_actions",
            "command": [py, _repo_path(repo_root, "devtools/build_gpu_bubble_experiment_readiness_next_actions.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/gpu_bubble_experiment_readiness_next_actions.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "refresh_gpu_bubble_terminal_self_check",
            "command": [py, _repo_path(repo_root, "devtools/build_bubble_gpu_bubble_readiness_terminal_self_check.py")],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/gpu_bubble_readiness_terminal_self_check.json")],
            "requires_gpu_if_executed": False,
        },
        {
            "id": "run_gpu_bubble_release_readiness_guard",
            "command": [
                py,
                _repo_path(repo_root, "devtools/guard_gpu_bubble_release_readiness.py"),
                "--out",
                _repo_path(repo_root, f"{runtime}/gpu_bubble_release_readiness_guard_report.json"),
            ],
            "expected_outputs": [_repo_path(repo_root, f"{runtime}/gpu_bubble_release_readiness_guard_report.json")],
            "requires_gpu_if_executed": False,
        },
    ]
    for row in rows:
        row["command_id"] = str(row["id"])
        row["status"] = "manual_json_refresh_required"
        row["ready"] = False
        row["safe_to_auto_start"] = False
        row["release_claim_allowed_after_success"] = False
        row["not_release_evidence"] = True
        row["roadmap"] = ROADMAP
    return rows


def _refresh_sequence_contract(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    command_ids = [str(_mapping(item).get("command_id") or _mapping(item).get("id") or "") for item in commands]
    positions = {command_id: index for index, command_id in enumerate(command_ids) if command_id}
    missing = [command_id for command_id in REQUIRED_REFRESH_SEQUENCE if command_id not in positions]
    out_of_order: list[str] = []
    for left, right in zip(REQUIRED_REFRESH_SEQUENCE, REQUIRED_REFRESH_SEQUENCE[1:]):
        if left in positions and right in positions and positions[left] >= positions[right]:
            out_of_order.extend([left, right])
    unsafe = [
        command_id
        for command_id, raw in zip(command_ids, commands)
        if bool(_mapping(raw).get("safe_to_auto_start"))
        or bool(_mapping(raw).get("release_claim_allowed_after_success"))
        or not bool(_mapping(raw).get("not_release_evidence"))
    ]
    roadmap_mismatch = [
        command_id
        for command_id, raw in zip(command_ids, commands)
        if str(_mapping(raw).get("roadmap") or "") != ROADMAP
    ]
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_external_input_refresh_sequence_contract",
        "required_command_ids": list(REQUIRED_REFRESH_SEQUENCE),
        "observed_command_ids": command_ids,
        "missing_command_count": len(missing),
        "missing_command_ids": missing,
        "out_of_order_command_count": len(set(out_of_order)),
        "out_of_order_command_ids": sorted(set(out_of_order)),
        "unsafe_command_count": len(unsafe),
        "unsafe_command_ids": unsafe,
        "roadmap_mismatch_count": len(roadmap_mismatch),
        "roadmap_mismatch_command_ids": roadmap_mismatch,
        "sequence_ok": not missing and not out_of_order and not unsafe and not roadmap_mismatch,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }


def _handoff_steps(items: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    missing_ids = {str(item.get("id") or "") for item in items if bool(item.get("required"))}
    if "sd15_checkpoint" in missing_ids:
        steps.append(
            {
                "id": "provide_sd15_checkpoint",
                "status": "external_input_required",
                "target": "models/sd15/<checkpoint>.safetensors",
                "after_input": "refresh_sd15_readiness_and_top_level_readiness",
            }
        )
    if "new_source_root" in missing_ids or slots:
        steps.append(
            {
                "id": "register_new_source_cache_axis",
                "status": "external_input_required",
                "target": "sucai/<new_source_or_cache_axis>",
                "families": sorted({str(slot.get("family") or "") for slot in slots if slot.get("family")}),
                "after_input": "refresh_intake_replay_scan_scout_preflight_chain",
            }
        )
    if "warm_cache_axis" in missing_ids:
        steps.append(
            {
                "id": "prepare_warm_cache_axis",
                "status": "external_input_required",
                "target": "family-specific ready cache/source axis",
                "after_input": "run_json_preflight_before_manual_canary_plan",
            }
        )
    if "caption_repair_axis" in missing_ids:
        steps.append(
            {
                "id": "repair_caption_coverage_axis",
                "status": "external_input_required",
                "target": "caption coverage >= preflight threshold",
                "after_input": "rescan_and_preflight_before_manual_canary_plan",
            }
        )
    if "anima_source_or_cache_axis" in missing_ids:
        steps.append(
            {
                "id": "provide_anima_source_or_cache_axis",
                "status": "external_input_required",
                "target": "anima-specific source or cache axis for saturation boundary",
                "after_input": "refresh_anima_saturation_boundary_and_release_gate_json",
            }
        )
    return steps


def _input_lifecycle_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in items:
        item = _mapping(raw)
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        detected = (
            bool(item.get("detected"))
            or bool(item.get("provided"))
            or str(item.get("status") or "") == "available"
        )
        required = bool(item.get("required"))
        accepted = detected and not required and str(item.get("status") or "") == "available"
        rows.append(
            {
                "input_id": item_id,
                "status": str(item.get("status") or ""),
                "detected": detected,
                "accepted": accepted,
                "pending": not accepted,
                "requires_downstream_refresh": detected,
                "release_claim_allowed_after_detection": False,
                "safe_to_auto_start": False,
                "not_release_evidence": True,
            }
        )
    detected_ids = [row["input_id"] for row in rows if row["detected"]]
    accepted_ids = [row["input_id"] for row in rows if row["accepted"]]
    detected_unaccepted_ids = [
        row["input_id"] for row in rows if row["detected"] and not row["accepted"]
    ]
    pending_ids = [row["input_id"] for row in rows if row["pending"]]
    unsafe_ids = [
        row["input_id"]
        for row in rows
        if row["release_claim_allowed_after_detection"]
        or row["safe_to_auto_start"]
        or not row["not_release_evidence"]
    ]
    if detected_unaccepted_ids:
        status = "detected_input_waiting_acceptance"
    elif detected_ids:
        status = "detected_inputs_accepted_pending_refresh"
    else:
        status = "waiting_for_external_input"
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_external_input_handoff_lifecycle_summary",
        "input_lifecycle_status": status,
        "input_count": len(rows),
        "detected_input_count": len(detected_ids),
        "accepted_input_count": len(accepted_ids),
        "detected_unaccepted_input_count": len(detected_unaccepted_ids),
        "pending_input_count": len(pending_ids),
        "unsafe_input_count": len(unsafe_ids),
        "detected_input_ids": detected_ids,
        "accepted_input_ids": accepted_ids,
        "detected_unaccepted_input_ids": detected_unaccepted_ids,
        "pending_input_ids": pending_ids,
        "unsafe_input_ids": unsafe_ids,
        "rows": rows,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
        "fail_closed": not unsafe_ids,
    }


def _resolution_summary(
    *,
    intake: Mapping[str, Any],
    missing_ids: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    sd15: Mapping[str, Any],
    external_input_detected: bool,
) -> dict[str, Any]:
    missing = set(missing_ids)
    sd15_required = "sd15_checkpoint" in missing or "sd15_base_checkpoint_missing" in _strings(sd15.get("blockers"))
    source_axis_required = bool(
        missing.intersection(
            {
                "new_source_root",
                "warm_cache_axis",
                "caption_repair_axis",
                "anima_source_or_cache_axis",
            }
        )
    )
    replay_ready = _safe_int(replay.get("ready_command_count")) > 0
    preflight_admitted = bool(pipeline.get("preflight_admitted"))
    manual_plan_ready = bool(pipeline.get("manual_canary_plan_ready"))
    intake_resolution = _mapping(intake.get("input_resolution_summary"))
    replay_resolution = _mapping(replay.get("input_resolution_summary"))
    sd15_checkpoint_exists = bool(
        intake_resolution.get(
            "sd15_checkpoint_exists",
            replay_resolution.get("sd15_checkpoint_exists", not sd15_required),
        )
    )
    new_source_root_count = _safe_int(
        intake_resolution.get(
            "new_source_root_count",
            replay_resolution.get("new_source_root_count", replay.get("new_source_root_count")),
        )
    )
    source_cache_ready = bool(pipeline.get("preflight_admitted")) and bool(
        pipeline.get("manual_canary_plan_ready")
    )
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "external_input_required": bool(missing_ids),
        "missing_external_inputs": list(missing_ids),
        "sd15_checkpoint_exists": sd15_checkpoint_exists,
        "sd15_checkpoint_required": sd15_required,
        "sd15_checkpoint_path": str(intake_resolution.get("sd15_checkpoint_path") or ""),
        "new_source_root_count": new_source_root_count,
        "new_source_root_required": "new_source_root" in missing,
        "source_or_cache_axis_required": source_axis_required or not source_cache_ready,
        "warm_cache_or_caption_repair_required": bool(
            missing.intersection({"warm_cache_axis", "caption_repair_axis"})
        ),
        "anima_source_or_cache_axis_required": "anima_source_or_cache_axis" in missing,
        "external_input_detected": external_input_detected,
        "json_replay_ready": replay_ready,
        "preflight_admitted": preflight_admitted,
        "manual_canary_plan_ready": manual_plan_ready,
        "handoff_step_ids": [str(step.get("id") or "") for step in steps],
        "next_json_refresh_sequence": list(REQUIRED_REFRESH_SEQUENCE),
        "next_manual_gpu_gate": (
            "sd15_manual_ab_after_checkpoint_review"
            if sd15_required
            else "source_cache_manual_canary_after_preflight_review"
            if source_axis_required
            else ""
        ),
        "release_gate_blockers": [
            *([] if not sd15_required else ["sd15_lora_512"]),
            *([] if not source_axis_required else ["natural_load_canary_pending"]),
        ],
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }


def build_external_input_handoff_packet(
    *,
    repo_root: Path,
    external_input_intake_registry: Mapping[str, Any] | None = None,
    external_input_replay_plan: Mapping[str, Any] | None = None,
    source_cache_axis_pipeline_readiness: Mapping[str, Any] | None = None,
    sd15_readiness: Mapping[str, Any] | None = None,
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
    python_exe: str = "backend/env/python-flashattention/python.exe",
) -> dict[str, Any]:
    """Build a fail-closed external-input handoff packet."""

    repo = Path(repo_root)
    intake = _mapping(external_input_intake_registry)
    replay = _mapping(external_input_replay_plan)
    pipeline = _mapping(source_cache_axis_pipeline_readiness)
    sd15 = _mapping(sd15_readiness)
    warm_cache = _mapping(newbie_warm_cache_inventory)
    items = _input_items(intake, pipeline, warm_cache)
    slots = _registration_slots(intake)
    missing = [item for item in items if bool(item.get("required"))]
    missing_ids = [str(item.get("id") or "") for item in missing if item.get("id")]
    steps = _handoff_steps(items, slots)
    input_lifecycle = _input_lifecycle_summary(items)
    external_detected = (
        bool(intake.get("external_input_detected"))
        or bool(replay.get("external_input_detected"))
        or _safe_int(input_lifecycle.get("detected_input_count")) > 0
    )
    status = "external_input_detected_review_required" if external_detected else "waiting_for_external_input"
    refresh_commands = _refresh_commands(repo, python_exe)
    refresh_sequence_contract = _refresh_sequence_contract(refresh_commands)
    unsafe_command_count = sum(
        1
        for command in refresh_commands
        if bool(command.get("safe_to_auto_start"))
        or bool(command.get("release_claim_allowed_after_success"))
        or not bool(command.get("not_release_evidence"))
    )
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "ok": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "not_release_evidence": True,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "external_input_detected": external_detected,
        "external_input_required": bool(missing),
        "missing_external_input_count": len(missing),
        "missing_external_inputs": missing_ids,
        "input_lifecycle_status": input_lifecycle["input_lifecycle_status"],
        "detected_input_count": input_lifecycle["detected_input_count"],
        "accepted_input_count": input_lifecycle["accepted_input_count"],
        "detected_unaccepted_input_count": input_lifecycle["detected_unaccepted_input_count"],
        "pending_input_count": input_lifecycle["pending_input_count"],
        "unsafe_input_count": input_lifecycle["unsafe_input_count"],
        "detected_input_ids": input_lifecycle["detected_input_ids"],
        "accepted_input_ids": input_lifecycle["accepted_input_ids"],
        "detected_unaccepted_input_ids": input_lifecycle["detected_unaccepted_input_ids"],
        "pending_input_ids": input_lifecycle["pending_input_ids"],
        "handoff_step_count": len(steps),
        "registration_slot_count": len(slots),
        "command_count": len(refresh_commands),
        "ready_command_count": sum(1 for command in refresh_commands if bool(command.get("ready"))),
        "blocked_command_count": sum(1 for command in refresh_commands if not bool(command.get("ready"))),
        "unsafe_command_count": unsafe_command_count,
        "input_items": items,
        "input_lifecycle_summary": input_lifecycle,
        "handoff_steps": steps,
        "candidate_registration_slots": slots,
        "input_resolution_summary": _resolution_summary(
            intake=intake,
            missing_ids=missing_ids,
            steps=steps,
            replay=replay,
            pipeline=pipeline,
            sd15=sd15,
            external_input_detected=external_detected,
        ),
        "replay_command_summary": _replay_command_summary(replay),
        "pipeline_summary": {
            "report": str(pipeline.get("report") or ""),
            "status": str(pipeline.get("status") or ""),
            "axis_readiness_status": str(pipeline.get("axis_readiness_status") or ""),
            "preflight_admitted": bool(pipeline.get("preflight_admitted")),
            "manual_canary_plan_ready": bool(pipeline.get("manual_canary_plan_ready")),
            "blockers": _strings(pipeline.get("blockers")),
        },
        "sd15_summary": {
            "report": str(sd15.get("report") or ""),
            "status": str(sd15.get("status") or ""),
            "blockers": _strings(sd15.get("blockers")),
        },
        "refresh_sequence_contract": refresh_sequence_contract,
        "refresh_commands": refresh_commands,
        "blocked_actions": [
            "auto_start_gpu_heavy_after_external_input_handoff",
            "promote_handoff_packet_as_release_evidence",
            "skip_sd15_ab_matrix_after_checkpoint",
            "skip_source_scan_scout_preflight_chain",
            "skip_natural_load_canary_or_release_claim_rebuild",
        ],
        "acceptance_gates": [
            "handoff_packet_is_json_only",
            "sd15_checkpoint_refresh_required_before_sd15_manual_ab",
            "new_source_axis_requires_scan_scout_preflight_chain",
            "manual_gpu_canary_requires_separate_protected_plan",
            "release_claim_requires_rebuilt_natural_load_and_release_claims",
        ],
        "recommended_next_action": (
            "review_detected_external_inputs_then_run_json_replay"
            if external_detected
            else "provide_sd15_checkpoint_and_new_source_cache_axis"
        ),
        "notes": [
            "This handoff packet is for developers/operators and is not release evidence.",
            "It consolidates missing external inputs and JSON refresh commands without starting GPU work.",
        ],
    }


__all__ = ["REPORT", "ROADMAP", "build_external_input_handoff_packet"]
