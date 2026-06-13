"""Command-chain audit for TurboCore optimizer v2 approval execution.

The audit is read-only.  It checks that the generated execution plan points to
existing entrypoints and keeps record-writing steps behind the signed-bundle
intake/extraction/preflight chain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_approval_execution_plan import (  # noqa: E402
    APPROVAL_PREFLIGHT_PLACEHOLDER,
    ARTIFACT as APPROVAL_EXECUTION_PLAN_ARTIFACT,
    EXTRACTION_OUTPUT_DIR,
    OWNER_DIRECTION_PACKET_PLACEHOLDER,
    OWNER_DIRECTION_PLACEHOLDER,
    OWNER_REVIEW_PLACEHOLDER,
    PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER,
    PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER,
    SIGNATURE_BUNDLE_PLACEHOLDER,
    SIGNED_BUNDLE_PLACEHOLDER,
    TRAINING_LAUNCH_CONTRACT_PLACEHOLDER,
    build_optimizer_v2_approval_execution_plan,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_approval_command_audit.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
EXPECTED_STEP_IDS = [
    "generate_unsigned_reviewer_template",
    "collect_real_owner_and_product_reviews",
    "validate_phase1_signed_bundle",
    "extract_phase1_signed_bundle_records",
    "preflight_phase1_record_inputs",
    "record_owner_release_review",
    "record_product_exposure_decision",
    "rebuild_release_review_archive",
    "rebuild_owner_direction_packet",
    "rebuild_phase2_signature_bundle",
    "regenerate_phase2_reviewer_template",
    "collect_real_owner_direction_signature",
    "validate_phase2_signed_bundle",
    "extract_phase2_signed_bundle_records",
    "preflight_phase2_record_inputs",
    "record_owner_release_direction",
]
PHASE1_HANDOFF_POST_RETURN_STEP_IDS = (
    "validate_phase1_signed_bundle",
    "extract_phase1_signed_bundle_records",
    "preflight_phase1_record_inputs",
)
REQUIRED_COMMAND_MARKERS = {
    "validate_phase1_signed_bundle": [
        "turbocore_optimizer_v2_signed_bundle_intake_record.py",
        "--signature-bundle",
        "--signed-bundle",
        "--no-artifact",
    ],
    "extract_phase1_signed_bundle_records": [
        "turbocore_optimizer_v2_signed_bundle_extractor.py",
        "--signature-bundle",
        "--signed-bundle",
        "--write-extracted-artifacts",
    ],
    "preflight_phase1_record_inputs": [
        "turbocore_optimizer_v2_approval_execution_preflight.py",
        "--signature-bundle",
        "--signed-bundle",
        "--owner-review",
        "--product-exposure-review",
        "--owner-direction",
        "--training-launch-contract",
        "--product-exposure-evidence",
        "--owner-direction-packet",
    ],
    "validate_phase2_signed_bundle": [
        "turbocore_optimizer_v2_signed_bundle_intake_record.py",
        "--signature-bundle",
        "--signed-bundle",
        "--no-artifact",
    ],
    "extract_phase2_signed_bundle_records": [
        "turbocore_optimizer_v2_signed_bundle_extractor.py",
        "--signature-bundle",
        "--signed-bundle",
        "--write-extracted-artifacts",
    ],
    "preflight_phase2_record_inputs": [
        "turbocore_optimizer_v2_approval_execution_preflight.py",
        "--signature-bundle",
        "--signed-bundle",
        "--owner-review",
        "--product-exposure-review",
        "--owner-direction",
        "--training-launch-contract",
        "--product-exposure-evidence",
        "--owner-direction-packet",
    ],
    "record_owner_release_review": [
        "turbocore_native_update_owner_release_review_record.py",
        "--signed-review",
        "--approval-preflight",
    ],
    "record_product_exposure_decision": [
        "turbocore_native_update_product_exposure_decision.py",
        "--product-exposure-review",
        "--approval-preflight",
    ],
    "rebuild_phase2_signature_bundle": ["turbocore_optimizer_v2_signature_bundle_packet.py"],
    "regenerate_phase2_reviewer_template": ["turbocore_optimizer_v2_reviewer_handoff_packet.py"],
    "record_owner_release_direction": [
        "turbocore_native_update_owner_release_direction_record.py",
        "--signed-direction",
        "--approval-preflight",
    ],
}
EXPECTED_ARG_BINDINGS = {
    "validate_phase1_signed_bundle": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
    },
    "extract_phase1_signed_bundle_records": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
        "--output-dir": EXTRACTION_OUTPUT_DIR,
    },
    "preflight_phase1_record_inputs": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
        "--owner-review": OWNER_REVIEW_PLACEHOLDER,
        "--product-exposure-review": PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER,
        "--owner-direction": OWNER_DIRECTION_PLACEHOLDER,
        "--training-launch-contract": TRAINING_LAUNCH_CONTRACT_PLACEHOLDER,
        "--product-exposure-evidence": PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER,
        "--owner-direction-packet": OWNER_DIRECTION_PACKET_PLACEHOLDER,
    },
    "record_owner_release_review": {
        "--signed-review": OWNER_REVIEW_PLACEHOLDER,
        "--approval-preflight": APPROVAL_PREFLIGHT_PLACEHOLDER,
    },
    "record_product_exposure_decision": {
        "--training-launch-contract": TRAINING_LAUNCH_CONTRACT_PLACEHOLDER,
        "--product-exposure-evidence": PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER,
        "--product-exposure-review": PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER,
        "--approval-preflight": APPROVAL_PREFLIGHT_PLACEHOLDER,
        "--out": "temp\\turbocore_optimizer\\native_update_product_exposure_decision.json",
    },
    "validate_phase2_signed_bundle": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
    },
    "extract_phase2_signed_bundle_records": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
        "--output-dir": EXTRACTION_OUTPUT_DIR,
    },
    "preflight_phase2_record_inputs": {
        "--signature-bundle": SIGNATURE_BUNDLE_PLACEHOLDER,
        "--signed-bundle": SIGNED_BUNDLE_PLACEHOLDER,
        "--owner-review": OWNER_REVIEW_PLACEHOLDER,
        "--product-exposure-review": PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER,
        "--owner-direction": OWNER_DIRECTION_PLACEHOLDER,
        "--training-launch-contract": TRAINING_LAUNCH_CONTRACT_PLACEHOLDER,
        "--product-exposure-evidence": PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER,
        "--owner-direction-packet": OWNER_DIRECTION_PACKET_PLACEHOLDER,
    },
    "record_owner_release_direction": {
        "--signed-direction": OWNER_DIRECTION_PLACEHOLDER,
        "--approval-preflight": APPROVAL_PREFLIGHT_PLACEHOLDER,
    },
}
RECORD_PREFLIGHT_STEP_IDS = {
    "preflight_phase1_record_inputs",
    "preflight_phase2_record_inputs",
}
RECORD_SIGNATURE_DEPENDENCIES = {
    "record_owner_release_review": ["collect_real_owner_and_product_reviews"],
    "record_product_exposure_decision": ["collect_real_owner_and_product_reviews"],
    "record_owner_release_direction": ["collect_real_owner_direction_signature"],
}
RECORD_PREFLIGHT_DEPENDENCIES = {
    "record_owner_release_review": "preflight_phase1_record_inputs",
    "record_product_exposure_decision": "preflight_phase1_record_inputs",
    "record_owner_release_direction": "preflight_phase2_record_inputs",
}
EXPECTED_STEP_PHASES = {
    "generate_unsigned_reviewer_template": "shared",
    "collect_real_owner_and_product_reviews": "phase1",
    "validate_phase1_signed_bundle": "phase1",
    "extract_phase1_signed_bundle_records": "phase1",
    "preflight_phase1_record_inputs": "phase1",
    "record_owner_release_review": "phase1",
    "record_product_exposure_decision": "phase1",
    "rebuild_release_review_archive": "phase2",
    "rebuild_owner_direction_packet": "phase2",
    "rebuild_phase2_signature_bundle": "phase2",
    "regenerate_phase2_reviewer_template": "phase2",
    "collect_real_owner_direction_signature": "phase2",
    "validate_phase2_signed_bundle": "phase2",
    "extract_phase2_signed_bundle_records": "phase2",
    "preflight_phase2_record_inputs": "phase2",
    "record_owner_release_direction": "phase2",
}


def build_optimizer_v2_approval_command_audit(
    *,
    approval_execution_plan: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    plan = _as_dict(approval_execution_plan) or _read_json(APPROVAL_EXECUTION_PLAN_ARTIFACT)
    if not plan:
        plan = build_optimizer_v2_approval_execution_plan(write_artifact=True)
    steps = [_as_dict(step) for step in _as_list(plan.get("steps"))]
    command_checks = [_command_check(step) for step in steps]
    sequence_blockers = _sequence_blockers(steps)
    phase2_sequence_blockers = _phase2_sequence_blockers(steps)
    signature_sequence_blockers = _signature_sequence_blockers(steps)
    phase_marker_blockers = _phase_marker_blockers(steps)
    preflight_artifact_blockers = _preflight_artifact_blockers(steps)
    marker_blockers = _marker_blockers(steps)
    arg_binding_checks = _expected_arg_binding_checks(steps)
    arg_binding_blockers = [reason for check in arg_binding_checks for reason in _strings(check.get("blocked_reasons"))]
    handoff_alignment_check = _phase1_handoff_post_return_alignment_check(plan, steps)
    handoff_alignment_blockers = _strings(handoff_alignment_check.get("blocked_reasons"))
    unsafe = _unsafe_claims(plan, steps)
    blockers = _dedupe(
        _strings(plan.get("blocked_reasons"))
        + sequence_blockers
        + phase2_sequence_blockers
        + signature_sequence_blockers
        + phase_marker_blockers
        + preflight_artifact_blockers
        + marker_blockers
        + arg_binding_blockers
        + handoff_alignment_blockers
        + [reason for check in command_checks for reason in _strings(check.get("blocked_reasons"))]
        + unsafe
    )
    command_steps = [check for check in command_checks if check["command_present"]]
    existing_entrypoints = [check for check in command_steps if check["entrypoint_exists"]]
    missing_entrypoints = [check for check in command_steps if not check["entrypoint_exists"]]
    preflight_checks = _record_preflight_checks(steps)
    record_after_preflight = [check for check in preflight_checks if check["after_required_preflight"]]
    record_before_preflight = [check for check in preflight_checks if not check["after_required_preflight"]]
    record_signature_checks = _record_signature_checks(steps)
    record_after_real_signature = [check for check in record_signature_checks if check["after_real_signature"]]
    record_before_real_signature = [check for check in record_signature_checks if not check["after_real_signature"]]
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_command_audit_v0",
        "gate": "optimizer_v2_approval_command_audit",
        "roadmap": ROADMAP,
        "ok": not blockers,
        "approval_command_audit_ready": not blockers,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_approval_execution_plan": str(APPROVAL_EXECUTION_PLAN_ARTIFACT),
        "command_checks": command_checks,
        "expected_arg_binding_checks": arg_binding_checks,
        "phase1_handoff_post_return_alignment_check": handoff_alignment_check,
        "summary": {
            "v2_approval_command_audit_step_count": len(steps),
            "v2_approval_command_audit_command_step_count": len(command_steps),
            "v2_approval_command_audit_entrypoint_exists_count": len(existing_entrypoints),
            "v2_approval_command_audit_missing_entrypoint_count": len(missing_entrypoints),
            "v2_approval_command_audit_order_valid_count": 1 if not sequence_blockers else 0,
            "v2_approval_command_audit_record_after_preflight_count": len(record_after_preflight),
            "v2_approval_command_audit_record_before_preflight_count": len(record_before_preflight),
            "v2_approval_command_audit_record_after_real_signature_count": len(record_after_real_signature),
            "v2_approval_command_audit_record_before_real_signature_count": len(record_before_real_signature),
            "v2_approval_command_audit_phase2_order_valid_count": 1 if not phase2_sequence_blockers else 0,
            "v2_approval_command_audit_phase2_blocker_count": len(phase2_sequence_blockers),
            "v2_approval_command_audit_signature_order_valid_count": 1 if not signature_sequence_blockers else 0,
            "v2_approval_command_audit_signature_blocker_count": len(signature_sequence_blockers),
            "v2_approval_command_audit_phase_marker_valid_count": 1 if not phase_marker_blockers else 0,
            "v2_approval_command_audit_phase_marker_blocker_count": len(phase_marker_blockers),
            "v2_approval_command_audit_required_marker_missing_count": len(marker_blockers),
            "v2_approval_command_audit_preflight_artifact_write_count": sum(
                1
                for step in steps
                if str(step.get("id") or "") in RECORD_PREFLIGHT_STEP_IDS
                and "--no-artifact" not in str(step.get("command") or "")
            ),
            "v2_approval_command_audit_preflight_no_artifact_blocker_count": len(preflight_artifact_blockers),
            "v2_approval_command_audit_record_command_preflight_arg_count": sum(
                1
                for step in steps
                if step.get("writes_approval_record") is True
                and "--approval-preflight" in str(step.get("command") or "")
            ),
            "v2_approval_command_audit_unsigned_template_allowed_count": sum(
                1 for step in steps if "--allow-unsigned-template" in str(step.get("command") or "")
            ),
            "v2_approval_command_audit_expected_arg_binding_count": len(arg_binding_checks),
            "v2_approval_command_audit_expected_arg_mismatch_count": len(arg_binding_blockers),
            "v2_approval_command_audit_expected_path_binding_ready_count": (
                1 if arg_binding_checks and not arg_binding_blockers else 0
            ),
            "v2_approval_command_audit_phase1_handoff_post_return_command_count": handoff_alignment_check[
                "command_count"
            ],
            "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count": (
                handoff_alignment_check["pre_record_command_count"]
            ),
            "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count": (
                handoff_alignment_check["approval_record_command_count"]
            ),
            "v2_approval_command_audit_phase1_handoff_post_return_command_match_count": handoff_alignment_check[
                "command_match_count"
            ],
            "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count": handoff_alignment_check[
                "command_mismatch_count"
            ],
            "v2_approval_command_audit_phase1_handoff_post_return_ready_count": (
                1 if handoff_alignment_check["ready"] else 0
            ),
            "v2_approval_command_audit_ready_count": 1 if not blockers else 0,
            "v2_approval_command_audit_approval_recorded_count": 0,
            "v2_approval_command_audit_runtime_dispatch_ready_count": 0,
            "v2_approval_command_audit_native_dispatch_allowed_count": 0,
            "v2_approval_command_audit_training_path_enabled_count": 0,
            "v2_approval_command_audit_product_native_ready_count": 0,
            "v2_approval_command_audit_default_behavior_changed_count": 0,
            "v2_approval_command_audit_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": blockers,
        "promotion_blockers": _dedupe(
            [
                "v2_approval_command_audit_is_not_approval_record",
                "real_owner_release_review_signature_missing",
                "real_product_exposure_review_signature_missing",
                "real_owner_release_direction_signature_missing",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + blockers
        ),
        "recommended_next_step": (
            "send the unsigned signed-bundle template to real reviewers; run this audit again before executing record commands"
        ),
        "notes": [
            "This audit only validates execution-plan command shape and ordering.",
            "Owner-direction signature collection must stay after owner review, product exposure decision, release archive, owner-direction packet, phase2 signature-bundle rebuild, and reviewer-handoff regeneration steps.",
            "Each approval-record command must stay after its matching real signature collection step and after preflight.",
            "Approval preflight steps must write the preflight artifact consumed by the following record validators.",
            "Record-writing commands remain manual approval steps and are not executed here.",
            "Native optimizer dispatch remains default-off.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _command_check(step: Mapping[str, Any]) -> dict[str, Any]:
    command = str(step.get("command") or "")
    script = _script_path(command)
    exists = bool(script and (REPO_ROOT / script).exists())
    blockers: list[str] = []
    if command and not script:
        blockers.append(f"{step.get('id')}_command_script_missing")
    if script and not exists:
        blockers.append(f"{step.get('id')}_entrypoint_missing:{script}")
    return {
        "schema_version": 1,
        "step_id": str(step.get("id") or ""),
        "command_present": bool(command),
        "script": script,
        "entrypoint_exists": exists,
        "blocked_reasons": blockers,
    }


def _script_path(command: str) -> str:
    match = re.search(r"(backend[\\/]+core[\\/]+[A-Za-z0-9_./\\-]+\.py)", command)
    return match.group(1).replace("/", "\\") if match else ""


def _sequence_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    ids = [str(step.get("id") or "") for step in sorted(steps, key=lambda item: int(item.get("order", 0) or 0))]
    blockers: list[str] = []
    if ids != EXPECTED_STEP_IDS:
        blockers.append("approval_execution_step_order_mismatch")
    orders = [int(step.get("order", 0) or 0) for step in steps]
    if orders != list(range(1, len(steps) + 1)):
        blockers.append("approval_execution_step_order_not_contiguous")
    if sum(1 for step in steps if step.get("preflights_record_inputs") is True) < 2:
        blockers.append("approval_execution_phase_preflight_steps_missing")
    for step in steps:
        if step.get("ready_to_execute") is True and step.get("writes_approval_record") is True:
            blockers.append(f"{step.get('id')}_record_step_marked_ready")
    return _dedupe(blockers)


def _phase2_sequence_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    by_id = {str(step.get("id") or ""): step for step in steps}
    required = [
        "record_owner_release_review",
        "record_product_exposure_decision",
        "rebuild_release_review_archive",
        "rebuild_owner_direction_packet",
        "rebuild_phase2_signature_bundle",
        "regenerate_phase2_reviewer_template",
        "collect_real_owner_direction_signature",
        "validate_phase2_signed_bundle",
        "extract_phase2_signed_bundle_records",
        "preflight_phase2_record_inputs",
        "record_owner_release_direction",
    ]
    blockers: list[str] = []
    missing = [step_id for step_id in required if step_id not in by_id]
    blockers.extend(f"phase2_step_missing:{step_id}" for step_id in missing)
    if missing:
        return _dedupe(blockers)

    orders = {step_id: int(by_id[step_id].get("order", 0) or 0) for step_id in required}
    for before, after in zip(required, required[1:]):
        if orders[before] >= orders[after]:
            blockers.append(f"phase2_order_invalid:{before}_before_{after}")

    direction_collect = by_id["collect_real_owner_direction_signature"]
    direction_record = by_id["record_owner_release_direction"]
    if direction_collect.get("manual_signature_required") is not True:
        blockers.append("owner_direction_collect_not_manual_signature")
    if str(direction_collect.get("command") or ""):
        blockers.append("owner_direction_collect_has_command")
    if direction_collect.get("writes_approval_record") is True:
        blockers.append("owner_direction_collect_writes_record")
    if direction_record.get("writes_approval_record") is not True:
        blockers.append("owner_direction_record_not_record_step")
    if direction_record.get("ready_to_execute") is True:
        blockers.append("owner_direction_record_marked_ready")
    return _dedupe(blockers)


def _phase_marker_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for step in steps:
        step_id = str(step.get("id") or "")
        expected_phase = EXPECTED_STEP_PHASES.get(step_id, "")
        phase = str(step.get("phase") or "")
        if not phase:
            blockers.append(f"{step_id}_phase_missing")
        elif expected_phase and phase != expected_phase:
            blockers.append(f"{step_id}_phase_mismatch:{phase}_expected_{expected_phase}")
        if step.get("writes_approval_record") is True and phase not in {"phase1", "phase2"}:
            blockers.append(f"{step_id}_record_phase_invalid:{phase}")
        if step.get("manual_signature_required") is True and phase not in {"phase1", "phase2"}:
            blockers.append(f"{step_id}_manual_signature_phase_invalid:{phase}")
    return _dedupe(blockers)


def _signature_sequence_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    by_id = {str(step.get("id") or ""): step for step in steps}
    for record_step_id, dependency_ids in RECORD_SIGNATURE_DEPENDENCIES.items():
        record_step = by_id.get(record_step_id)
        if not record_step:
            blockers.append(f"signature_record_step_missing:{record_step_id}")
            continue
        record_order = int(record_step.get("order", 0) or 0)
        if record_step.get("writes_approval_record") is not True:
            blockers.append(f"{record_step_id}_not_record_step")
        for dependency_id in dependency_ids:
            dependency_step = by_id.get(dependency_id)
            if not dependency_step:
                blockers.append(f"signature_dependency_missing:{record_step_id}:{dependency_id}")
                continue
            dependency_order = int(dependency_step.get("order", 0) or 0)
            if dependency_step.get("manual_signature_required") is not True:
                blockers.append(f"{dependency_id}_not_manual_signature_step")
            if record_order <= dependency_order:
                blockers.append(f"{record_step_id}_record_before_real_signature:{dependency_id}")
    return _dedupe(blockers)


def _preflight_artifact_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for step in steps:
        step_id = str(step.get("id") or "")
        if step_id not in RECORD_PREFLIGHT_STEP_IDS:
            continue
        command = str(step.get("command") or "")
        if "--no-artifact" in command:
            blockers.append(f"{step_id}_preflight_must_write_record_artifact")
    return _dedupe(blockers)


def _record_signature_checks(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(step.get("id") or ""): step for step in steps}
    checks: list[dict[str, Any]] = []
    for record_step_id, dependency_ids in RECORD_SIGNATURE_DEPENDENCIES.items():
        record_step = by_id.get(record_step_id)
        record_order = int(record_step.get("order", 0) or 0) if record_step else 0
        dependency_orders = [
            int(by_id[dependency_id].get("order", 0) or 0)
            for dependency_id in dependency_ids
            if dependency_id in by_id
        ]
        after_real_signature = bool(
            record_step
            and record_step.get("writes_approval_record") is True
            and len(dependency_orders) == len(dependency_ids)
            and all(record_order > dependency_order for dependency_order in dependency_orders)
        )
        checks.append(
            {
                "schema_version": 1,
                "record_step_id": record_step_id,
                "real_signature_step_ids": dependency_ids,
                "after_real_signature": after_real_signature,
            }
        )
    return checks


def _record_preflight_checks(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(step.get("id") or ""): step for step in steps}
    checks: list[dict[str, Any]] = []
    for record_step_id, preflight_step_id in RECORD_PREFLIGHT_DEPENDENCIES.items():
        record_step = by_id.get(record_step_id)
        preflight_step = by_id.get(preflight_step_id)
        record_order = int(record_step.get("order", 0) or 0) if record_step else 0
        preflight_order = int(preflight_step.get("order", 0) or 0) if preflight_step else 0
        after_required_preflight = bool(
            record_step
            and preflight_step
            and record_step.get("writes_approval_record") is True
            and preflight_step.get("preflights_record_inputs") is True
            and record_order > preflight_order
        )
        checks.append(
            {
                "schema_version": 1,
                "record_step_id": record_step_id,
                "required_preflight_step_id": preflight_step_id,
                "after_required_preflight": after_required_preflight,
            }
        )
    return checks


def _phase1_handoff_post_return_alignment_check(
    plan: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
) -> dict[str, Any]:
    alignment = _as_dict(plan.get("phase1_handoff_post_return_alignment"))
    by_id = {str(step.get("id") or ""): step for step in steps}
    plan_commands = [
        str(_as_dict(by_id.get(step_id)).get("command") or "")
        for step_id in PHASE1_HANDOFF_POST_RETURN_STEP_IDS
    ]
    handoff_commands = _strings(alignment.get("handoff_commands"))
    expected_count = len(PHASE1_HANDOFF_POST_RETURN_STEP_IDS)
    command_count = len(handoff_commands)
    command_match_count = sum(
        1
        for plan_command, handoff_command in zip(plan_commands, handoff_commands)
        if plan_command and plan_command == handoff_command
    )
    pre_record_command_count = int(alignment.get("pre_record_command_count", 0) or 0)
    approval_record_command_count = int(alignment.get("approval_record_command_count", 0) or 0)
    command_mismatch_count = expected_count - command_match_count
    blockers: list[str] = []
    if not alignment:
        blockers.append("phase1_handoff_post_return_alignment_missing")
    if any(not command for command in plan_commands):
        blockers.append("phase1_handoff_post_return_plan_command_missing")
    if command_count != expected_count:
        blockers.append("phase1_handoff_post_return_command_count_mismatch")
    if command_match_count != expected_count:
        blockers.append("phase1_handoff_post_return_command_mismatch")
    if pre_record_command_count != expected_count:
        blockers.append("phase1_handoff_post_return_pre_record_command_count_mismatch")
    if approval_record_command_count != 0:
        blockers.append("phase1_handoff_post_return_approval_record_command_present")
    if alignment.get("ready") is not True:
        blockers.append("phase1_handoff_post_return_alignment_not_ready")
    return {
        "schema_version": 1,
        "ready": not blockers,
        "expected_step_ids": list(PHASE1_HANDOFF_POST_RETURN_STEP_IDS),
        "command_count": command_count,
        "pre_record_command_count": pre_record_command_count,
        "approval_record_command_count": approval_record_command_count,
        "command_match_count": command_match_count,
        "command_mismatch_count": command_mismatch_count,
        "blocked_reasons": _dedupe(blockers),
    }


def _marker_blockers(steps: list[Mapping[str, Any]]) -> list[str]:
    by_id = {str(step.get("id") or ""): str(step.get("command") or "") for step in steps}
    blockers: list[str] = []
    for step_id, markers in REQUIRED_COMMAND_MARKERS.items():
        command = by_id.get(step_id, "")
        for marker in markers:
            if marker not in command:
                blockers.append(f"{step_id}_command_missing:{marker}")
    return blockers


def _expected_arg_binding_checks(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(step.get("id") or ""): str(step.get("command") or "") for step in steps}
    checks: list[dict[str, Any]] = []
    for step_id, expected_args in EXPECTED_ARG_BINDINGS.items():
        parsed = _parsed_args(by_id.get(step_id, ""))
        for flag, expected in expected_args.items():
            actual = parsed.get(flag, "")
            checks.append(
                {
                    "schema_version": 1,
                    "step_id": step_id,
                    "arg": flag,
                    "expected": expected,
                    "actual": actual,
                    "matches_expected": actual == expected,
                    "blocked_reasons": []
                    if actual == expected
                    else [f"{step_id}_arg_binding_mismatch:{flag}:{actual or '<missing>'}"],
                }
            )
    return checks


def _parsed_args(command: str) -> dict[str, str]:
    parts = command.split()
    parsed: dict[str, str] = {}
    for index, part in enumerate(parts):
        if part.startswith("--"):
            parsed[part] = parts[index + 1] if index + 1 < len(parts) and not parts[index + 1].startswith("--") else ""
    return parsed


def _unsafe_claims(plan: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for field in (
        "approval_recorded",
        "approval_artifact_written",
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "product_native_ready",
    ):
        if plan.get(field) is True:
            claims.append(f"approval_execution_plan_unsafe:{field}")
    for step in steps:
        for field in (
            "default_behavior_changed",
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "training_path_enabled",
            "product_native_ready",
        ):
            if step.get(field) is True:
                claims.append(f"{step.get('id')}_unsafe:{field}")
    return _dedupe(claims)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-execution-plan", default="", help="Optional v2 approval execution plan JSON.")
    parser.add_argument("--no-artifact", action="store_true", help="Print audit without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_approval_command_audit(
        approval_execution_plan=_read_json(Path(args.approval_execution_plan)) if args.approval_execution_plan else None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_approval_command_audit"]


if __name__ == "__main__":
    raise SystemExit(main())
