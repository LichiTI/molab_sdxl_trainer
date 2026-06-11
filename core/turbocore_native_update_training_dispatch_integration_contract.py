"""Default-off native-update training dispatch integration contract.

This contract sits after the native-update rollout review package.  It records
that the recovery, owner, stream, executor, and training-path boundaries are
present and still default-off.  It does not authorize product dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    event_list as _event_list,
    history_summary as _history_summary,
    string_list as _string_list,
)


SCOPE = "native_update_training_dispatch_integration_contract"
READY_DECISION = "native_update_training_dispatch_integration_contract_recorded_default_off"
HOLD_DECISION = "native_update_training_dispatch_integration_contract_hold_for_review_default_off"
BLOCKED_DECISION = "native_update_training_dispatch_integration_contract_blocked_default_off"
REJECTED_DECISION = "native_update_training_dispatch_integration_contract_rejected_default_off"
REQUIRED_REVIEW_ACKS = (
    "acknowledge_rollout_review_package_ready",
    "acknowledge_recovery_boundary_default_off",
    "acknowledge_owner_gradient_sync_default_off",
    "acknowledge_flat_owner_default_off",
    "acknowledge_training_kernel_default_off",
    "acknowledge_stream_lifetime_ownership_default_off",
    "acknowledge_runtime_executor_default_off",
    "acknowledge_training_path_request_default_off",
    "acknowledge_no_product_training_dispatch",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_later_activation_contract_required",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "training_dispatch",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_mutation_allowed",
    "training_parameter_mutation_allowed",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_integration_request_fields",
    "post_native_update_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
    "launch_request",
    "training_request",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)
COMPONENTS = (
    "recovery",
    "owner_gradient_sync",
    "training_flat_owner",
    "training_dispatch_kernel",
    "stream_lifetime_ownership",
    "training_executor",
    "training_path_request",
)


def build_native_update_training_dispatch_integration_contract(
    *,
    rollout_review_package: Mapping[str, Any] | None = None,
    dispatch_contract: Mapping[str, Any] | None = None,
    integration_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record integration-boundary evidence without enabling native dispatch."""

    rollout = _rollout_summary(_as_dict(rollout_review_package))
    dispatch = _dispatch_summary(_as_dict(dispatch_contract))
    review = _review_summary(_as_dict(integration_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        rollout=rollout,
        dispatch=dispatch,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review)
    if evidence_blockers:
        decision = BLOCKED_DECISION
    elif not review.get("present"):
        decision = HOLD_DECISION
    elif review_blockers:
        decision = BLOCKED_DECISION
    elif review.get("approve_native_update_training_dispatch_integration_contract") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_training_dispatch_integration_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_training_dispatch_integration_contract_v0",
        "gate": "native_update_training_dispatch_integration_contract",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_integration_review": evidence_ready,
        "integration_contract_recorded": decision == READY_DECISION,
        "native_update_training_dispatch_integration_contract_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "integration_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_integration_request_fields": {},
        "rollout_review_summary": rollout,
        "dispatch_contract_summary": dispatch,
        "integration_review": review,
        "integration_review_template": _review_template(rollout, dispatch),
        "progress_gates": {
            "rollout_review_package_ready": bool(rollout.get("ready")),
            "dispatch_contract_present": bool(dispatch.get("present")),
            "default_off_boundary_confirmed": bool(dispatch.get("default_off")),
            "component_boundary_count": dispatch.get("component_boundary_count", 0),
            "component_default_off_count": dispatch.get("component_default_off_count", 0),
            "request_ui_schema_exposure_blocked": bool(
                rollout.get("request_ui_schema_exposure_blocked")
                and dispatch.get("request_ui_schema_exposure_blocked")
            ),
            "signed_integration_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This contract records default-off native-update integration boundaries only.",
            "It does not enable product native dispatch, emit request fields, expose UI/schema, or launch training.",
            "A later activation contract is required before any training dispatch can be wired into product requests.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
        payload.setdefault("_source_digest", _digest_payload(payload))
        return payload
    return {}


def _rollout_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(package, "rollout_review_package")
    return {
        "present": bool(package),
        "source": str(package.get("_source_path") or ""),
        "digest": str(package.get("_source_digest") or _digest_payload(package) if package else ""),
        "ok": package.get("ok") is True,
        "evidence_package_ready": package.get("evidence_package_ready") is True,
        "ready_for_owner_review": package.get("ready_for_owner_review") is True,
        "rollout_review_recorded": package.get("native_update_rollout_review_recorded") is True,
        "ready": bool(
            package
            and package.get("ok") is True
            and package.get("evidence_package_ready") is True
            and package.get("ready_for_owner_review") is True
            and package.get("post_native_update_request_fields", {}) == {}
            and not unsafe
        ),
        "request_ui_schema_exposure_blocked": bool(
            package.get("request_adapter_mapping_allowed") is False
            and package.get("request_fields_emitted") is False
            and package.get("schema_exposure_allowed") is False
            and package.get("ready_for_ui") is False
        ),
        "decision": str(package.get("decision") or ""),
        "blocked_reasons": unsafe,
    }


def _dispatch_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    components = _component_summaries(report)
    unsafe = _unsafe_claims(report, "dispatch_contract")
    component_blockers = _component_blockers(components)
    default_off = bool(
        report
        and report.get("training_dispatch") is False
        and report.get("training_path_enabled") is False
        and report.get("would_allow_native_dispatch") is False
        and report.get("pytorch_optimizer_authoritative") is True
        and not unsafe
    )
    ready = bool(report and default_off and not component_blockers)
    return {
        "present": bool(report),
        "source": str(report.get("_source_path") or ""),
        "digest": str(report.get("_source_digest") or _digest_payload(report) if report else ""),
        "contract": str(report.get("contract") or ""),
        "ready": ready,
        "default_off": default_off,
        "pytorch_optimizer_authoritative": report.get("pytorch_optimizer_authoritative") is True,
        "component_boundary_count": sum(1 for item in components.values() if item.get("boundary_ready")),
        "component_default_off_count": sum(1 for item in components.values() if item.get("default_off")),
        "components": components,
        "request_ui_schema_exposure_blocked": not bool(
            report.get("request_adapter_mapping_allowed") is True
            or report.get("request_fields_emitted") is True
            or report.get("schema_exposure_allowed") is True
            or report.get("ready_for_ui") is True
        ),
        "blocked_reasons": _dedupe(unsafe + component_blockers),
        "source_blocked_reasons": _string_list(report.get("blocked_reasons")),
    }


def _component_summaries(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    recovery = _as_dict(report.get("recovery"))
    owner_sync = _as_dict(report.get("owner_gradient_sync"))
    flat_owner = _as_dict(report.get("training_flat_owner"))
    kernel = _as_dict(report.get("training_dispatch_kernel"))
    stream = _as_dict(report.get("stream_lifetime_ownership"))
    executor = _as_dict(report.get("training_executor"))
    return {
        "recovery": {
            "boundary_ready": bool(
                recovery.get("recovery_observation_bridge_ready")
                and recovery.get("default_off_recovery_bridge_ready")
            ),
            "default_off": bool(
                recovery.get("training_dispatch_recovery_ready") is False
                or "training_dispatch_recovery_default_off" in _string_list(recovery.get("blocked_reasons"))
                or "native_runtime_recovery_training_dispatch_disabled" in _string_list(recovery.get("blocked_reasons"))
            ),
            "blocked_reasons": _string_list(recovery.get("blocked_reasons")),
        },
        "owner_gradient_sync": _generic_component(owner_sync, "sync_boundary_ready"),
        "training_flat_owner": _generic_component(flat_owner, "owner_boundary_ready"),
        "training_dispatch_kernel": _generic_component(kernel, "kernel_boundary_ready", evidence_key="kernel_present_evidence"),
        "stream_lifetime_ownership": _generic_component(stream, "ownership_boundary_ready", evidence_key="ordering_verified"),
        "training_executor": _generic_component(executor, "executor_boundary_ready"),
        "training_path_request": {
            "boundary_ready": True,
            "default_off": bool(report.get("training_path_enabled") is False and report.get("training_dispatch") is False),
            "blocked_reasons": [
                item
                for item in _string_list(report.get("blocked_reasons"))
                if item in {"native_dispatch_training_path_default_off", "native_dispatch_training_path_disabled"}
            ],
        },
    }


def _generic_component(
    value: Mapping[str, Any],
    boundary_key: str,
    *,
    evidence_key: str | None = None,
) -> dict[str, Any]:
    evidence_ok = bool(value.get(evidence_key)) if evidence_key else True
    return {
        "boundary_ready": bool(value.get(boundary_key) and evidence_ok),
        "default_off": value.get("default_off") is True and value.get("bound_to_training_path") is False,
        "blocked_reasons": _string_list(value.get("blocked_reasons")),
    }


def _component_blockers(components: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    for name in COMPONENTS:
        component = _as_dict(components.get(name))
        if not component.get("boundary_ready"):
            blocked.append(f"native_update_integration_component_boundary_not_ready:{name}")
        if not component.get("default_off"):
            blocked.append(f"native_update_integration_component_not_default_off:{name}")
    return _dedupe(blocked)


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_training_dispatch_integration_contract": review.get(
            "approve_native_update_training_dispatch_integration_contract"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _evidence_blockers(
    *,
    rollout: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not rollout.get("present"):
        blocked.append("native_update_integration_rollout_review_package_missing")
    elif not rollout.get("ready"):
        blocked.append("native_update_integration_rollout_review_package_not_ready")
        blocked.extend(_string_list(rollout.get("blocked_reasons")))
    if not dispatch.get("present"):
        blocked.append("native_update_integration_dispatch_contract_missing")
    elif not dispatch.get("ready"):
        blocked.append("native_update_integration_dispatch_contract_not_ready")
        blocked.extend(_string_list(dispatch.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_integration_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_integration_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_integration_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_integration_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_integration_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_integration_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_integration_review_ack_missing:{field}")
    return _dedupe(blocked)


def _review_template(rollout: Mapping[str, Any], dispatch: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_training_dispatch_integration_contract": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "rollout_digest": rollout.get("digest"),
        "dispatch_contract_digest": dispatch.get("digest"),
        "component_default_off_count": dispatch.get("component_default_off_count"),
    }
    return template


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"native_update_integration_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_integration_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_training_dispatch_integration_contract", "prepare_later_default_off_activation_contract"]
    if evidence_ready:
        return ["collect_signed_training_dispatch_integration_review"]
    return ["repair_rollout_or_dispatch_contract_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive default-off integration contract and prepare the later activation contract"
    if evidence_ready:
        return "record integration review while keeping native update dispatch, request fields, and UI exposure disabled"
    return "repair rollout-review or dispatch-contract evidence before integration review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off integration contract.")
    parser.add_argument("--rollout-review-package", required=True)
    parser.add_argument("--dispatch-contract", required=True)
    parser.add_argument("--integration-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=load_json(args.rollout_review_package),
        dispatch_contract=load_json(args.dispatch_contract),
        integration_review=load_json(args.integration_review) if args.integration_review else None,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
