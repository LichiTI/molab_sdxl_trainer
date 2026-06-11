"""Default-off runtime contract for lossless guarded-variant mitigation."""

from __future__ import annotations

from typing import Any, Mapping


ALLOWED_UNIT_KINDS = {
    "replacement_phase_guard_contract",
    "mixed_regression_split_contract",
    "raw_control_jitter_baseline_contract",
}
REQUEST_ACTIVATION_KEYS = (
    "runtime_activation_allowed",
    "runtime_default_change_allowed",
    "request_adapter_activation_allowed",
)


def _items_by_id(items: object) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        unit_id = str(item.get("unit_id") or "")
        if unit_id:
            rows[unit_id] = dict(item)
    return rows


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def build_lossless_guarded_variant_runtime_contract(
    blueprint: Mapping[str, Any],
    request_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a runtime-facing contract without enabling execution.

    The result is intentionally contract-only. It lets request/runtime code
    reason about the replacement-phase guard while keeping raw/control jitter
    groups as blockers and keeping all product gates closed.
    """

    source = dict(blueprint)
    request = dict(request_contract or {})
    summary = dict(source.get("summary") or {})
    units = _items_by_id(source.get("implementation_units"))
    requested_unit_ids = _string_list(request.get("requested_unit_ids")) or sorted(units)
    selected_units = [units[unit_id] for unit_id in requested_unit_ids if unit_id in units]
    missing_requested_unit_ids = [
        unit_id for unit_id in requested_unit_ids if unit_id not in units
    ]
    blocked_reasons: list[str] = []

    if source.get("probe") != (
        "lossless_compute_tail_raw_order_guarded_variant_mitigation_blueprint_v1"
    ):
        blocked_reasons.append("unexpected_blueprint_probe")
    if not bool(source.get("ok")) or not bool(
        summary.get("guarded_variant_mitigation_blueprint_ready")
    ):
        blocked_reasons.append("blueprint_not_ready")
    if missing_requested_unit_ids:
        blocked_reasons.append("requested_unit_missing")
    if any(
        bool(source.get(key) or summary.get(key) or request.get(key))
        for key in (
            "training_path_enabled",
            "resource_center_allowed",
            "resource_center_candidate",
            "default_enabled",
            "product_ready",
            "safe_to_auto_execute",
        )
    ):
        blocked_reasons.append("unsafe_gate_open")
    requested_activation_keys = [
        key for key in REQUEST_ACTIVATION_KEYS if bool(request.get(key))
    ]
    if requested_activation_keys:
        blocked_reasons.append("requested_runtime_activation_not_allowed")
    if not selected_units:
        blocked_reasons.append("implementation_unit_missing")

    unknown_kinds = sorted(
        {
            str(unit.get("unit_kind") or "")
            for unit in selected_units
            if str(unit.get("unit_kind") or "") not in ALLOWED_UNIT_KINDS
        }
    )
    if unknown_kinds:
        blocked_reasons.append("unknown_unit_kind")

    replacement_units = [
        unit
        for unit in selected_units
        if unit.get("unit_kind") == "replacement_phase_guard_contract"
    ]
    raw_control_units = [
        unit
        for unit in selected_units
        if unit.get("unit_kind") == "raw_control_jitter_baseline_contract"
    ]
    mixed_units = [
        unit
        for unit in selected_units
        if unit.get("unit_kind") == "mixed_regression_split_contract"
    ]
    if not replacement_units:
        blocked_reasons.append("replacement_phase_guard_unit_missing")
    if raw_control_units or mixed_units:
        blocked_reasons.append("raw_control_or_mixed_regression_units_block_activation")

    ready = not blocked_reasons or blocked_reasons == [
        "raw_control_or_mixed_regression_units_block_activation"
    ]
    activation_allowed = False
    return {
        "schema_version": 1,
        "contract": "lossless_guarded_variant_runtime_contract_v1",
        "ok": ready,
        "runtime_contract_ready": ready,
        "runtime_activation_allowed": activation_allowed,
        "runtime_default_change_allowed": False,
        "request_adapter_activation_allowed": False,
        "training_path_enabled": False,
        "resource_center_allowed": False,
        "resource_center_candidate": False,
        "default_enabled": False,
        "product_ready": False,
        "safe_to_auto_execute": False,
        "manual_heavy_validation_required": True,
        "requested_unit_ids": requested_unit_ids,
        "requested_activation_keys": requested_activation_keys,
        "request_activation_denied": bool(requested_activation_keys),
        "selected_unit_count": len(selected_units),
        "replacement_phase_guard_unit_count": len(replacement_units),
        "mixed_regression_split_unit_count": len(mixed_units),
        "raw_control_jitter_baseline_unit_count": len(raw_control_units),
        "missing_requested_unit_ids": missing_requested_unit_ids,
        "unknown_unit_kinds": unknown_kinds,
        "activation_blockers": [
            "manual_heavy_validation_required",
            *(
                ["requested_runtime_activation_not_allowed"]
                if requested_activation_keys
                else []
            ),
            *(
                ["raw_control_or_mixed_regression_units_block_activation"]
                if (raw_control_units or mixed_units)
                else []
            ),
        ],
        "blocked_reasons": blocked_reasons,
        "selected_units": selected_units,
        "recommended_next_step": (
            "wire replacement-phase guard as default-off request/runtime contract and validate manually"
            if ready
            else "complete guarded-variant mitigation blueprint before runtime contract wiring"
        ),
    }


__all__ = ["build_lossless_guarded_variant_runtime_contract"]
