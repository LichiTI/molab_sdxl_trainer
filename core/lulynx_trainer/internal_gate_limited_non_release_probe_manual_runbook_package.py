"""Report-only manual runbook package for a limited non-release probe.

This package follows the signed execution contract and records only the manual
runbook, stop-condition inventory, and before/after evidence template for a
future batch1-only probe. It never enables the internal gate or starts work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_RUNBOOK_PACKAGE = (
    "lulynx_internal_gate_limited_non_release_probe_manual_runbook_package_v0"
)
READY_CONTRACT_STATUS = "ready_for_limited_non_release_probe_execution_contract"
APPROVED_CONTRACT_DECISION = (
    "internal_gate_limited_non_release_probe_execution_contract_recorded_default_off"
)


def build_lulynx_internal_gate_limited_non_release_probe_manual_runbook_package(
    *,
    internal_gate_limited_non_release_probe_execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed manual runbook package for a later batch1 probe."""

    contract = dict(_mapping(internal_gate_limited_non_release_probe_execution_contract))
    contract_summary = _contract_summary(contract)
    checks = _checks(contract=contract)
    blockers = _blockers(contract=contract, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_RUNBOOK_PACKAGE,
        "status": "ready_for_manual_batch1_non_release_probe_runbook" if ready else "blocked",
        "passed": ready,
        "manual_review_required": True,
        "safe_to_auto_start": False,
        "internal_gate_enablement_allowed": False,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "checks": checks,
        "blockers": blockers,
        "execution_contract_summary": contract_summary,
        "manual_runbook": _manual_runbook(),
        "stop_conditions_inventory": _stop_conditions_inventory(),
        "before_after_evidence_template": _before_after_evidence_template(),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(*, contract: Mapping[str, Any]) -> dict[str, bool]:
    execution_contract = _mapping(contract.get("execution_contract"))
    blocked_surface = set(_string_list(execution_contract.get("blocked_execution_surface")))
    guardrails = set(_string_list(execution_contract.get("required_guardrails")))
    stop_conditions = set(_string_list(execution_contract.get("required_stop_conditions")))
    return {
        "execution_contract_present": bool(contract),
        "execution_contract_ready": bool(contract.get("passed"))
        and str(contract.get("status") or "") == READY_CONTRACT_STATUS
        and str(contract.get("decision") or "") == APPROVED_CONTRACT_DECISION,
        "internal_gate_enablement_not_allowed": not bool(
            contract.get("internal_gate_enablement_allowed")
        ),
        "release_claim_closed": not bool(contract.get("release_claim_allowed")),
        "batch1_only_contract": str(execution_contract.get("probe_batch_contract") or "")
        == "real_gpu_batch1_only",
        "non_release_only_contract": str(execution_contract.get("probe_release_policy") or "")
        == "non_release_only",
        "blocks_gate_enablement": "turn_internal_gate_on_now" in blocked_surface,
        "blocks_training_start": "start_training_now" in blocked_surface,
        "blocks_new_training_entrypoint": "new_training_entrypoint" in blocked_surface,
        "blocks_batch2_4_8_release_probe": "batch2_4_8_release_probe" in blocked_surface,
        "blocks_release_claim": "release_claim" in blocked_surface,
        "manual_start_only_guardrail_present": "manual_start_only" in guardrails,
        "default_off_guardrail_present": "internal_gate_default_off" in guardrails,
        "before_after_guardrail_present": "before_after_evidence_required" in guardrails,
        "stop_conditions_complete": {
            "loss_regression",
            "throughput_regression",
            "vram_regression",
            "unexpected_runtime_path_diff",
            "missing_manifest_evidence",
        }.issubset(stop_conditions),
    }


def _blockers(*, contract: Mapping[str, Any], checks: Mapping[str, bool]) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not contract:
        blockers.append("internal_gate_limited_non_release_probe_execution_contract_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_execution_contract:{item}"
        for item in _string_list(contract.get("blockers"))
    )
    return _dedupe(blockers)


def _contract_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    execution_contract = _mapping(report.get("execution_contract"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "decision": str(report.get("decision") or ""),
        "internal_gate_enablement_allowed": bool(report.get("internal_gate_enablement_allowed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "probe_scope": str(execution_contract.get("probe_scope") or ""),
        "probe_batch_contract": str(execution_contract.get("probe_batch_contract") or ""),
        "probe_release_policy": str(execution_contract.get("probe_release_policy") or ""),
    }


def _manual_runbook() -> dict[str, Any]:
    return {
        "runbook_kind": "manual_batch1_non_release_probe_runbook_v0",
        "scope": "behavior_equivalent_internal_gate_batch1_non_release_probe",
        "preflight_checks": [
            "confirm_internal_gate_stays_disabled_by_default",
            "confirm_batch1_only_probe_scope",
            "confirm_release_claims_stay_closed",
            "confirm_before_after_evidence_template_prepared",
            "confirm_stop_conditions_are_visible_to_manual_operator",
        ],
        "operator_steps": [
            "review_signed_execution_contract_and current blockers",
            "prepare before baseline evidence from existing batch1 manifest",
            "prepare probe-only after evidence destination",
            "record manual stop / rollback conditions before any future probe",
            "keep batch2_4_8, release claim, and default enablement blocked",
        ],
        "forbidden_actions": [
            "turn_internal_gate_on_now",
            "start_training_now",
            "emit_new_training_entrypoint",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        ],
    }


def _stop_conditions_inventory() -> dict[str, Any]:
    return {
        "inventory_kind": "manual_probe_stop_conditions_inventory_v0",
        "required_stop_conditions": [
            {
                "id": "loss_regression",
                "trigger": "after probe loss deviates beyond approved tolerance",
                "manual_action": "stop probe planning progression and record blocker",
            },
            {
                "id": "throughput_regression",
                "trigger": "after probe throughput drops versus baseline evidence",
                "manual_action": "hold follow-up execution planning and refresh evidence",
            },
            {
                "id": "vram_regression",
                "trigger": "after probe memory footprint exceeds guardrail",
                "manual_action": "mark probe unsafe and keep default-off boundary",
            },
            {
                "id": "unexpected_runtime_path_diff",
                "trigger": "runtime path diverges from behavior-equivalent batch1 contract",
                "manual_action": "invalidate comparison and require contract refresh",
            },
            {
                "id": "missing_manifest_evidence",
                "trigger": "before/after manifest evidence is incomplete or absent",
                "manual_action": "block probe review closure until evidence is refreshed",
            },
        ],
    }


def _before_after_evidence_template() -> dict[str, Any]:
    return {
        "template_kind": "manual_probe_before_after_evidence_template_v0",
        "required_before_fields": [
            "run_manifest_path",
            "batch_contract",
            "runtime_feature_snapshot",
            "steady_samples_per_second",
            "active_window_gpu20_or_gpu50",
            "peak_vram_mb",
            "final_loss_or_loss_window",
        ],
        "required_after_fields": [
            "probe_manifest_path",
            "batch_contract",
            "runtime_feature_snapshot",
            "steady_samples_per_second",
            "active_window_gpu20_or_gpu50",
            "peak_vram_mb",
            "final_loss_or_loss_window",
        ],
        "required_comparisons": [
            "throughput_delta",
            "active_gpu_window_delta",
            "vram_delta",
            "loss_delta",
            "runtime_path_diff_summary",
        ],
        "release_policy": "comparison_is_for_non_release_probe_review_only",
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_probe_review_packet_without_enabling_internal_gate",
            "refresh_before_after_baseline_inputs_for_batch1_only_probe",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_manual_runbook_package_prerequisites"]
    if any("execution_contract" in item for item in blockers):
        actions.append("refresh_limited_non_release_probe_execution_contract")
    if any("release_claim" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_manual_runbook_package")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_RUNBOOK_PACKAGE",
    "build_lulynx_internal_gate_limited_non_release_probe_manual_runbook_package",
]
