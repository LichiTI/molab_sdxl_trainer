"""Request-adapter contract for the lossless guarded variant.

The contract is deliberately default-off. It lets request/config adapter code
describe a future guarded-variant knob while rejecting activation until manual
heavy validation proves product safety.
"""

from __future__ import annotations

from typing import Any, Mapping


ACTIVATION_KEYS = (
    "enable_lossless_guarded_variant",
    "runtime_activation_allowed",
    "request_adapter_activation_allowed",
    "training_path_enabled",
    "resource_center_allowed",
    "resource_center_candidate",
    "candidate",
    "default_enabled",
    "product_ready",
    "safe_to_auto_execute",
)

UNSAFE_RUNTIME_CLAIM_KEYS = (
    "training_path_enabled",
    "resource_center_allowed",
    "resource_center_candidate",
    "candidate",
    "default_enabled",
    "product_ready",
    "safe_to_auto_execute",
    "runtime_activation_allowed",
    "request_adapter_activation_allowed",
    "runtime_ab_ready",
    "execute_allowed_by_default",
    "execute_requested",
    "validation_passed",
    "product_unlock_ready",
    "declares_validation_passed",
)

UNSAFE_REQUEST_CLAIM_KEYS = (
    "runtime_ab_ready",
    "execute_allowed_by_default",
    "execute_requested",
    "validation_passed",
    "product_unlock_ready",
    "declares_validation_passed",
)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _open_claims(source: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if bool(source.get(key))]


def build_lossless_guarded_variant_request_adapter_contract(
    runtime_contract_report: Mapping[str, Any],
    request_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project guarded-variant runtime contract into request-adapter policy."""

    report = dict(runtime_contract_report)
    request = dict(request_payload or {})
    summary = dict(report.get("summary") or {})
    runtime_contract = dict(report.get("runtime_contract") or report)
    requested_unit_ids = (
        _string_list(request.get("requested_unit_ids"))
        or _string_list(runtime_contract.get("requested_unit_ids"))
    )
    requested_activation_keys = [
        key for key in ACTIVATION_KEYS if bool(request.get(key))
    ]
    unsafe_request_claims = _open_claims(request, UNSAFE_REQUEST_CLAIM_KEYS)
    blocked_reasons: list[str] = []

    if report.get("probe") != "lossless_guarded_variant_runtime_contract_probe_v1":
        blocked_reasons.append("unexpected_runtime_contract_probe")
    if not bool(report.get("ok")) or not bool(
        summary.get("guarded_variant_runtime_contract_ready")
    ):
        blocked_reasons.append("runtime_contract_not_ready")
    if bool(runtime_contract.get("runtime_activation_allowed")):
        blocked_reasons.append("runtime_contract_activation_open")
    if bool(runtime_contract.get("request_adapter_activation_allowed")):
        blocked_reasons.append("runtime_contract_request_adapter_open")
    unsafe_runtime_claims = list(
        dict.fromkeys(
            [
                *_open_claims(report, UNSAFE_RUNTIME_CLAIM_KEYS),
                *_open_claims(summary, UNSAFE_RUNTIME_CLAIM_KEYS),
                *_open_claims(runtime_contract, UNSAFE_RUNTIME_CLAIM_KEYS),
            ]
        )
    )
    if unsafe_runtime_claims:
        blocked_reasons.append("runtime_contract_unsafe_claim_open")
    if requested_activation_keys:
        blocked_reasons.append("requested_activation_denied")
    if unsafe_request_claims:
        blocked_reasons.append("request_payload_unsafe_claim_open")
    if not requested_unit_ids:
        blocked_reasons.append("requested_unit_ids_missing")

    ready = not [
        reason
        for reason in blocked_reasons
        if reason
        not in {
            "requested_activation_denied",
        }
    ]

    return {
        "schema_version": 1,
        "contract": "lossless_guarded_variant_request_adapter_contract_v1",
        "ok": ready,
        "request_adapter_contract_ready": ready,
        "json_only": True,
        "manifest_only": True,
        "does_not_submit_request": True,
        "does_not_mutate_runtime": True,
        "training_path_enabled": False,
        "resource_center_allowed": False,
        "resource_center_candidate": False,
        "candidate": False,
        "default_enabled": False,
        "product_ready": False,
        "safe_to_auto_execute": False,
        "runtime_activation_allowed": False,
        "request_adapter_activation_allowed": False,
        "manual_heavy_validation_required": True,
        "requested_unit_ids": requested_unit_ids,
        "requested_activation_keys": requested_activation_keys,
        "unsafe_request_claim_keys": unsafe_request_claims,
        "unsafe_runtime_claim_keys": unsafe_runtime_claims,
        "request_activation_denied": bool(
            requested_activation_keys or unsafe_request_claims
        ),
        "activation_blockers": [
            "manual_heavy_validation_required",
            *(
                ["requested_activation_denied"]
                if requested_activation_keys
                else []
            ),
            *(
                ["request_payload_unsafe_claim_open"]
                if unsafe_request_claims
                else []
            ),
        ],
        "blocked_reasons": blocked_reasons,
        "adapter_policy": {
            "policy_id": "lossless_guarded_variant_default_off_request_adapter",
            "exposes_request_schema": True,
            "accepts_requested_unit_ids": True,
            "accepts_activation_request_for_audit": True,
            "activation_request_effect": "deny_and_record",
            "runtime_activation_allowed": False,
            "request_adapter_activation_allowed": False,
            "requires_manual_heavy_validation": True,
        },
        "recommended_next_step": (
            "keep request adapter default-off; use this contract only to audit denied activation requests"
            if ready
            else "complete guarded-variant runtime contract before request-adapter wiring"
        ),
    }


__all__ = ["build_lossless_guarded_variant_request_adapter_contract"]
