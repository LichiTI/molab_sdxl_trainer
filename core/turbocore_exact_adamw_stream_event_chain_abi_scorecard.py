"""Report-only stream/event-chain ABI scorecard for exact AdamW."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_dispatch_contract import build_native_update_dispatch_contract
from core.turbocore_v5_stream_sync_policy import build_v5_stream_sync_policy


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "turbocore_exact_adamw_stream_event_chain_abi_scorecard_v0"
GATE = "exact_adamw_stream_event_chain_ownership_abi"


def build_exact_adamw_stream_event_chain_abi_scorecard(
    *,
    dispatch_contract_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Summarize stream ownership ABI evidence without enabling dispatch."""

    dispatch_contract = _as_dict(dispatch_contract_report) or _default_dispatch_contract()
    stream_contract = _as_dict(dispatch_contract.get("stream_lifetime_ownership"))
    evidence = _as_dict(dispatch_contract.get("evidence"))
    sync_policy = build_v5_stream_sync_policy(
        timing_triage={
            "timing_triage_ready": True,
            "primary_bottleneck": "stream_event_chain_sync_fast_path",
            "metrics": {
                "runtime_synchronization": "context_synchronize",
                "runtime_stream_binding": "report_only_unbound",
            },
        },
        stream_guard={
            "event_chain_verified": bool(evidence.get("event_chain_verified", False)),
            "pre_launch_ordering_verified": True,
            "post_launch_ordering_verified": True,
            "stream_wait_event_verified": True,
            "stream_handle_reported": True,
            "stream_handle_nonzero": True,
            "stream_handle_kind": "external_cuda_stream_handle",
            "stream_lifetime_bound": bool(evidence.get("stream_lifetime_ownership_bound", False)),
        },
        native_runtime={
            "adamw_launch_on_borrowed_stream_supported": True,
            "ctx_synchronize_free_training_step_supported": True,
            "event_chain_synchronization_supported": True,
        },
        requested_mode="off",
    )
    abi_ready = _abi_ready(dispatch_contract, stream_contract, evidence)
    unsafe = _unsafe_claims(dispatch_contract)
    ok = bool(abi_ready and not unsafe)
    report = {
        "schema_version": 1,
        "scorecard": CONTRACT,
        "gate": GATE,
        "ok": ok,
        "evidence_ready": ok,
        "promotion_ready": False,
        "ready_for_optimizer_family_coverage_review": ok,
        "manual_review_required": True,
        "optimizer_type": "AdamW",
        "native_route": "rust_cuda_adamw_v0",
        "exact_semantics_only": True,
        "stream_event_chain_ownership_abi_ready": abi_ready,
        "stream_lifetime_ownership_boundary_ready": bool(
            stream_contract.get("ownership_boundary_ready", False)
        ),
        "stream_lifetime_ownership_bound_evidence": bool(
            stream_contract.get("ownership_bound_evidence", False)
        ),
        "stream_ordering_verified": bool(stream_contract.get("ordering_verified", False)),
        "event_chain_verified": bool(evidence.get("event_chain_verified", False)),
        "dispatch_contract_default_off": bool(stream_contract.get("default_off", False)),
        "training_dispatch": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "sync_fast_path_allowed": False,
        "sync_policy": _compact_sync_policy(sync_policy),
        "dispatch_contract": _compact_dispatch_contract(dispatch_contract),
        "summary": {
            "optimizer_count": 1,
            "stream_event_chain_ownership_abi_ready_count": 1 if abi_ready else 0,
            "stream_lifetime_ownership_boundary_ready_count": 1
            if stream_contract.get("ownership_boundary_ready") is True
            else 0,
            "stream_lifetime_ownership_bound_evidence_count": 1
            if stream_contract.get("ownership_bound_evidence") is True
            else 0,
            "stream_ordering_verified_count": 1 if stream_contract.get("ordering_verified") is True else 0,
            "event_chain_verified_count": 1 if evidence.get("event_chain_verified") is True else 0,
            "sync_fast_path_allowed_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
            "default_behavior_changed_count": 0,
        },
        "blocked_reasons": _dedupe(["unsafe_default_off_claim"] if unsafe else []),
        "promotion_blockers": [
            "exact_adamw_owner_release_approval_missing",
            "exact_adamw_request_schema_ui_non_exposure_hold_missing",
        ],
        "recommended_next_step": (
            "record explicit owner/release approval for exact AdamW native dispatch"
        ),
        "notes": [
            "This scorecard only proves the stream/event-chain ownership ABI can be read by the native-update contract.",
            "It does not bind training tensors, launch training, remove ctx synchronization, or allow native dispatch.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _default_dispatch_contract() -> dict[str, Any]:
    shadow = {
        "native_binding_probe": {
            "stream_lifetime_bound": True,
            "event_chain_verified": True,
            "pre_launch_ordering_verified": True,
            "post_launch_ordering_verified": True,
            "stream_wait_event_verified": True,
            "stream_guard_present": True,
            "stream_guard_ready": True,
            "stream_identity_ready": True,
            "stream_handle_reported": True,
            "stream_handle_nonzero": True,
            "stream_handle_kind": "external_cuda_stream_handle",
        }
    }
    preflight = {
        "dispatch_preflight_passed": False,
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "stream_lifetime_ownership_bound": True,
        "stream_ordering_verified": True,
        "event_chain_verified": True,
        "performance_test_ready": False,
        "evidence": {
            "performance": {
                "representative_performance_gate_ready": False,
                "blocked_reasons": ["representative_performance_gate_missing"],
            }
        },
        "blocked_reasons": ["representative_performance_gate_missing"],
    }
    readiness = {
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "stream_lifetime_ownership_bound": True,
        "stream_ordering_verified": True,
        "event_chain_verified": True,
        "performance_test_ready": False,
        "native_checks": {
            "flat_owner_contract_ready": True,
            "reference_flat_owner_ready": True,
            "training_flat_owner_promoted": False,
            "training_dispatch_kernel_contract_ready": True,
            "training_dispatch_kernel_present": True,
        },
        "owner_checks": {
            "direct_gradient_write_boundary_ready": True,
            "direct_gradient_write_native_supported": True,
            "direct_gradient_write_training_integrated": True,
            "owner_gradient_sync_boundary_ready": True,
            "owner_gradient_sync_supported": True,
            "owner_gradient_sync_training_integrated": True,
        },
    }
    return build_native_update_dispatch_contract(
        mode="native_experimental",
        requested=True,
        readiness_report=readiness,
        shadow_report=shadow,
        dispatch_preflight=preflight,
        fallback_policy={"fallback_to_pytorch_enabled": True},
        runtime_context={},
    )


def _abi_ready(
    dispatch_contract: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    return bool(
        dispatch_contract.get("training_dispatch") is False
        and dispatch_contract.get("training_path_enabled") is False
        and stream_contract.get("ownership_boundary_ready") is True
        and stream_contract.get("ownership_bound_evidence") is True
        and stream_contract.get("ordering_verified") is True
        and stream_contract.get("default_off") is True
        and evidence.get("event_chain_verified") is True
        and evidence.get("stream_lifetime_ownership_bound") is True
        and "stream_lifetime_ownership_default_off"
        in _strings(stream_contract.get("blocked_reasons"))
    )


def _compact_dispatch_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    stream_contract = _as_dict(report.get("stream_lifetime_ownership"))
    return {
        "contract": str(report.get("contract") or ""),
        "training_dispatch": bool(report.get("training_dispatch", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "would_allow_native_dispatch": bool(report.get("would_allow_native_dispatch", False)),
        "native_mutation_allowed": bool(report.get("native_mutation_allowed", False)),
        "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
        "stream_lifetime_ownership_bound": bool(report.get("stream_lifetime_ownership_bound", False)),
        "stream_ordering_verified": bool(report.get("stream_ordering_verified", False)),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "stream_lifetime_ownership": {
            "contract": str(stream_contract.get("contract") or ""),
            "ownership_boundary_ready": bool(stream_contract.get("ownership_boundary_ready", False)),
            "ownership_bound_evidence": bool(stream_contract.get("ownership_bound_evidence", False)),
            "ordering_verified": bool(stream_contract.get("ordering_verified", False)),
            "default_off": bool(stream_contract.get("default_off", False)),
            "blocked_reasons": _strings(stream_contract.get("blocked_reasons")),
        },
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_sync_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": str(report.get("policy") or ""),
        "requested_mode": str(report.get("requested_mode") or ""),
        "sync_fast_path_allowed": bool(report.get("sync_fast_path_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "requires_explicit_opt_in": bool(report.get("requires_explicit_opt_in", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _unsafe_claims(report: Mapping[str, Any]) -> bool:
    return any(
        report.get(field) is True
        for field in (
            "training_dispatch",
            "training_path_enabled",
            "would_allow_native_dispatch",
            "native_mutation_allowed",
            "training_parameter_mutation_allowed",
        )
    )


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_exact_adamw_stream_event_chain_abi_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["GATE", "build_exact_adamw_stream_event_chain_abi_scorecard"]
