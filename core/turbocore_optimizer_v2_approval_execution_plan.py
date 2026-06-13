"""Execution plan for applying real TurboCore optimizer v2 approvals.

The plan is a machine-readable checklist.  It does not sign anything, does not
write approval records, and never enables native optimizer dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    ARTIFACT as REVIEWER_HANDOFF_ARTIFACT,
    SIGNED_BUNDLE_TEMPLATE_ARTIFACT,
    build_optimizer_v2_reviewer_handoff_packet,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_plan.json"
SIGNATURE_BUNDLE_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signature_bundle_packet.json"
SIGNED_BUNDLE_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signed_bundle.reviewed.json"
OWNER_REVIEW_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_owner_release_review.json"
PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_product_exposure_review.json"
OWNER_DIRECTION_PLACEHOLDER = "temp\\turbocore_optimizer\\signed_owner_release_direction.json"
TRAINING_LAUNCH_CONTRACT_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_training_launch_contract.json"
PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_product_exposure_evidence.json"
OWNER_DIRECTION_PACKET_PLACEHOLDER = "temp\\turbocore_optimizer\\native_update_owner_release_direction_packet.json"
APPROVAL_PREFLIGHT_PLACEHOLDER = "temp\\turbocore_optimizer\\turbocore_optimizer_v2_approval_execution_preflight.json"
EXTRACTION_OUTPUT_DIR = "temp\\turbocore_optimizer"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_optimizer_v2_approval_execution_plan(
    *,
    reviewer_handoff: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    handoff = _as_dict(reviewer_handoff) or _read_json(REVIEWER_HANDOFF_ARTIFACT)
    if not handoff:
        handoff = build_optimizer_v2_reviewer_handoff_packet(write_artifact=True)
    summary = _as_dict(handoff.get("summary"))
    ready_templates = int(summary.get("v2_reviewer_handoff_signed_template_entry_count", 0) or 0)
    blocked_entries = int(summary.get("v2_reviewer_handoff_blocked_entry_count", 0) or 0)
    unsafe = _unsafe_claims(handoff)
    steps = _execution_steps()
    phase1_handoff_alignment = _phase1_handoff_post_return_alignment(
        steps,
        _as_list(handoff.get("phase1_post_return_operator_commands")),
    )
    plan_ready = (
        handoff.get("reviewer_handoff_ready") is True
        and ready_templates >= 2
        and phase1_handoff_alignment["ready"]
        and not unsafe
    )
    payload = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_execution_plan_v0",
        "gate": "optimizer_v2_approval_execution_plan",
        "roadmap": ROADMAP,
        "ok": plan_ready,
        "execution_plan_ready": plan_ready,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_reviewer_handoff": str(REVIEWER_HANDOFF_ARTIFACT),
        "signed_bundle_template_artifact": str(SIGNED_BUNDLE_TEMPLATE_ARTIFACT),
        "expected_signature_bundle_path": SIGNATURE_BUNDLE_PLACEHOLDER,
        "expected_signed_bundle_path": SIGNED_BUNDLE_PLACEHOLDER,
        "phase1_handoff_post_return_alignment": phase1_handoff_alignment,
        "steps": steps,
        "summary": {
            "v2_approval_execution_step_count": len(steps),
            "v2_approval_execution_phase1_step_count": _phase_count(steps, "phase1"),
            "v2_approval_execution_phase2_step_count": _phase_count(steps, "phase2"),
            "v2_approval_execution_shared_step_count": _phase_count(steps, "shared"),
            "v2_approval_execution_ready_step_count": sum(1 for step in steps if step["ready_to_execute"]),
            "v2_approval_execution_manual_signature_step_count": sum(
                1 for step in steps if step["manual_signature_required"]
            ),
            "v2_approval_execution_record_step_count": sum(1 for step in steps if step["writes_approval_record"]),
            "v2_approval_execution_extraction_step_count": sum(
                1 for step in steps if step["extracts_signed_inputs"]
            ),
            "v2_approval_execution_preflight_step_count": sum(1 for step in steps if step["preflights_record_inputs"]),
            "v2_approval_execution_phase1_record_chain_step_count": _record_chain_count(steps, "phase1"),
            "v2_approval_execution_phase2_record_chain_step_count": _record_chain_count(steps, "phase2"),
            "v2_approval_execution_reviewer_template_entry_count": ready_templates,
            "v2_approval_execution_blocked_signature_entry_count": blocked_entries,
            "v2_approval_execution_phase1_handoff_post_return_command_count": phase1_handoff_alignment[
                "command_count"
            ],
            "v2_approval_execution_phase1_handoff_post_return_pre_record_command_count": phase1_handoff_alignment[
                "pre_record_command_count"
            ],
            "v2_approval_execution_phase1_handoff_post_return_approval_record_command_count": (
                phase1_handoff_alignment["approval_record_command_count"]
            ),
            "v2_approval_execution_phase1_handoff_post_return_command_match_count": phase1_handoff_alignment[
                "command_match_count"
            ],
            "v2_approval_execution_phase1_handoff_post_return_command_mismatch_count": phase1_handoff_alignment[
                "command_mismatch_count"
            ],
            "v2_approval_execution_phase1_handoff_post_return_ready_count": 1
            if phase1_handoff_alignment["ready"]
            else 0,
            "v2_approval_execution_plan_ready_count": 1 if plan_ready else 0,
            "v2_approval_execution_approval_recorded_count": 0,
            "v2_approval_execution_runtime_dispatch_ready_count": 0,
            "v2_approval_execution_native_dispatch_allowed_count": 0,
            "v2_approval_execution_training_path_enabled_count": 0,
            "v2_approval_execution_product_native_ready_count": 0,
            "v2_approval_execution_default_behavior_changed_count": 0,
            "v2_approval_execution_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            ([] if ready_templates >= 2 else ["reviewer_signed_bundle_template_not_ready"])
            + ([] if blocked_entries >= 1 else ["owner_direction_sequence_not_visible"])
            + ([] if phase1_handoff_alignment["ready"] else ["phase1_handoff_post_return_commands_mismatch"])
            + unsafe
        ),
        "promotion_blockers": [
            "real_owner_release_review_signature_missing",
            "real_product_exposure_review_signature_missing",
            "real_owner_release_direction_signature_missing",
            "product_dispatch_still_requires_explicit_route_binding",
        ],
        "recommended_next_step": (
            "send the unsigned signed-bundle template to the real reviewer, then validate the returned JSON with the intake CLI"
        ),
        "notes": [
            "The plan is ordered because owner direction depends on recorded owner review, product exposure decision, and release archive.",
            "Phase 2 rebuilds the signature bundle and reviewer handoff after the owner-direction packet is rebuilt so returned signatures bind to the current digest.",
            "Step commands are intentionally explicit and default-off; no product route binding is performed here.",
            "Signed-bundle extraction writes validator inputs only; it is not an approval record.",
            "Approval execution preflight checks validator inputs only; it is not an approval record.",
            "Reviewer handoff post-return commands must match the phase1 intake/extraction/preflight plan before records are written.",
            "Synthetic smoke signatures are not real approval evidence.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _execution_steps() -> list[dict[str, Any]]:
    return [
        _step(1, "generate_unsigned_reviewer_template", "Generate reviewer handoff and unsigned signed-bundle template.", "backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_reviewer_handoff_packet.py", False, False, phase="shared"),
        _step(2, "collect_real_owner_and_product_reviews", "Reviewer fills reviewer/reviewed_at/approve/ack fields for owner_release_review and product_exposure_review.", "", True, False, phase="phase1"),
        _step(3, "validate_phase1_signed_bundle", "Validate the returned phase1 signed bundle without writing approval artifacts.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_signed_bundle_intake_record.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --no-artifact", False, False, phase="phase1", validates_signed_bundle=True),
        _step(4, "extract_phase1_signed_bundle_records", "Extract phase1 signed review JSON inputs for the existing record validators.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_signed_bundle_extractor.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --output-dir {EXTRACTION_OUTPUT_DIR} --write-extracted-artifacts", False, False, phase="phase1", extracts_signed_inputs=True),
        _step(5, "preflight_phase1_record_inputs", "Check phase1 extracted signed reviews and support artifacts before owner/product record validators.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_approval_execution_preflight.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --owner-review {OWNER_REVIEW_PLACEHOLDER} --product-exposure-review {PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER} --owner-direction {OWNER_DIRECTION_PLACEHOLDER} --training-launch-contract {TRAINING_LAUNCH_CONTRACT_PLACEHOLDER} --product-exposure-evidence {PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER} --owner-direction-packet {OWNER_DIRECTION_PACKET_PLACEHOLDER}", False, False, phase="phase1", preflights_record_inputs=True),
        _step(6, "record_owner_release_review", "Write owner release review record from extracted owner_release_review JSON after phase1 preflight.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_native_update_owner_release_review_record.py --signed-review {OWNER_REVIEW_PLACEHOLDER} --approval-preflight {APPROVAL_PREFLIGHT_PLACEHOLDER}", False, True, phase="phase1"),
        _step(7, "record_product_exposure_decision", "Write default-off product exposure decision from extracted product_exposure_review JSON after phase1 preflight.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_native_update_product_exposure_decision.py --training-launch-contract temp\\turbocore_optimizer\\native_update_training_launch_contract.json --product-exposure-evidence temp\\turbocore_optimizer\\native_update_product_exposure_evidence.json --product-exposure-review {PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER} --approval-preflight {APPROVAL_PREFLIGHT_PLACEHOLDER} --out temp\\turbocore_optimizer\\native_update_product_exposure_decision.json", False, True, phase="phase1"),
        _step(8, "rebuild_release_review_archive", "Rebuild release review archive after owner/product records exist.", "backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_native_update_release_review_archive.py", False, False, phase="phase2"),
        _step(9, "rebuild_owner_direction_packet", "Rebuild owner release-direction packet after archive and product exposure decision are recorded.", "backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_native_update_owner_release_direction_packet.py", False, False, phase="phase2"),
        _step(10, "rebuild_phase2_signature_bundle", "Rebuild the v2 signature bundle so owner-direction uses the current direction packet digest.", "backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_signature_bundle_packet.py", False, False, phase="phase2"),
        _step(11, "regenerate_phase2_reviewer_template", "Regenerate reviewer handoff and signed-bundle template after owner-direction becomes signable.", "backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_reviewer_handoff_packet.py", False, False, phase="phase2"),
        _step(12, "collect_real_owner_direction_signature", "After the refreshed direction template is ready, reviewer fills reviewer/reviewed_at/approve/ack fields for owner_release_direction.", "", True, False, phase="phase2"),
        _step(13, "validate_phase2_signed_bundle", "Validate the returned phase2 signed bundle without writing approval artifacts.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_signed_bundle_intake_record.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --no-artifact", False, False, phase="phase2", validates_signed_bundle=True),
        _step(14, "extract_phase2_signed_bundle_records", "Extract phase2 signed owner-direction JSON input for the existing record validator.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_signed_bundle_extractor.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --output-dir {EXTRACTION_OUTPUT_DIR} --write-extracted-artifacts", False, False, phase="phase2", extracts_signed_inputs=True),
        _step(15, "preflight_phase2_record_inputs", "Check phase2 extracted owner-direction signature and support artifacts before direction record validator.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_optimizer_v2_approval_execution_preflight.py --signature-bundle {SIGNATURE_BUNDLE_PLACEHOLDER} --signed-bundle {SIGNED_BUNDLE_PLACEHOLDER} --owner-review {OWNER_REVIEW_PLACEHOLDER} --product-exposure-review {PRODUCT_EXPOSURE_REVIEW_PLACEHOLDER} --owner-direction {OWNER_DIRECTION_PLACEHOLDER} --training-launch-contract {TRAINING_LAUNCH_CONTRACT_PLACEHOLDER} --product-exposure-evidence {PRODUCT_EXPOSURE_EVIDENCE_PLACEHOLDER} --owner-direction-packet {OWNER_DIRECTION_PACKET_PLACEHOLDER}", False, False, phase="phase2", preflights_record_inputs=True),
        _step(16, "record_owner_release_direction", "Record signed owner direction from extracted owner_release_direction JSON after phase2 preflight.", f"backend\\env\\python-flashattention\\python.exe backend\\core\\turbocore_native_update_owner_release_direction_record.py --signed-direction {OWNER_DIRECTION_PLACEHOLDER} --approval-preflight {APPROVAL_PREFLIGHT_PLACEHOLDER}", False, True, phase="phase2"),
    ]


def _step(
    order: int,
    step_id: str,
    description: str,
    command: str,
    manual_signature_required: bool,
    writes_approval_record: bool,
    extracts_signed_inputs: bool = False,
    preflights_record_inputs: bool = False,
    validates_signed_bundle: bool = False,
    phase: str = "shared",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "order": order,
        "id": step_id,
        "description": description,
        "command": command,
        "ready_to_execute": order <= 3,
        "manual_signature_required": manual_signature_required,
        "writes_approval_record": writes_approval_record,
        "phase": phase,
        "validates_signed_bundle": validates_signed_bundle,
        "extracts_signed_inputs": extracts_signed_inputs,
        "preflights_record_inputs": preflights_record_inputs,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
    }


def _phase_count(steps: list[Mapping[str, Any]], phase: str) -> int:
    return sum(1 for step in steps if step.get("phase") == phase)


def _record_chain_count(steps: list[Mapping[str, Any]], phase: str) -> int:
    return sum(
        1
        for step in steps
        if step.get("phase") == phase
        and (
            step.get("validates_signed_bundle") is True
            or step.get("extracts_signed_inputs") is True
            or step.get("preflights_record_inputs") is True
            or step.get("writes_approval_record") is True
        )
    )


def _phase1_handoff_post_return_alignment(
    steps: list[Mapping[str, Any]],
    handoff_commands: list[Any],
) -> dict[str, Any]:
    expected_step_ids = (
        "validate_phase1_signed_bundle",
        "extract_phase1_signed_bundle_records",
        "preflight_phase1_record_inputs",
    )
    plan_commands = [
        str(step.get("command") or "")
        for step in steps
        if str(step.get("id") or "") in expected_step_ids
    ]
    handoff_entries = [_as_dict(item) for item in handoff_commands]
    handoff_command_strings = [str(item.get("command") or "") for item in handoff_entries]
    command_match_count = sum(
        1
        for expected_command, handoff_command in zip(plan_commands, handoff_command_strings)
        if expected_command and expected_command == handoff_command
    )
    expected_count = len(expected_step_ids)
    command_count = len(handoff_entries)
    pre_record_command_count = sum(1 for item in handoff_entries if item.get("writes_approval_record") is False)
    approval_record_command_count = sum(1 for item in handoff_entries if item.get("writes_approval_record") is True)
    command_mismatch_count = expected_count - command_match_count
    ready = (
        command_count == expected_count
        and command_match_count == expected_count
        and pre_record_command_count == expected_count
        and approval_record_command_count == 0
    )
    return {
        "ready": ready,
        "command_count": command_count,
        "pre_record_command_count": pre_record_command_count,
        "approval_record_command_count": approval_record_command_count,
        "command_match_count": command_match_count,
        "command_mismatch_count": command_mismatch_count,
        "expected_step_ids": list(expected_step_ids),
        "plan_commands": plan_commands,
        "handoff_command_ids": [str(item.get("id") or "") for item in handoff_entries],
        "handoff_commands": handoff_command_strings,
    }


def _unsafe_claims(report: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for field in (
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "product_native_ready",
        "approval_recorded",
        "approval_artifact_written",
    ):
        if report.get(field) is True:
            claims.append(f"reviewer_handoff_unsafe:{field}")
    return claims


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-handoff", default="", help="Optional v2 reviewer handoff packet JSON path.")
    parser.add_argument("--no-artifact", action="store_true", help="Print validation without writing the plan artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_approval_execution_plan(
        reviewer_handoff=_read_json(Path(args.reviewer_handoff)) if args.reviewer_handoff else None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


__all__ = ["build_optimizer_v2_approval_execution_plan"]


if __name__ == "__main__":
    raise SystemExit(main())
