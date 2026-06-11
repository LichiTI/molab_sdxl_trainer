"""Service adapter for semi-automatic bubble advisor next-run patches."""

from __future__ import annotations

from typing import Any, Mapping

from backend.core.lulynx_trainer.bubble_runtime_patch_apply import prepare_bubble_advisor_next_request


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _select_base_request(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("base_request", "request", "config"):
        value = _mapping(payload.get(key))
        if value:
            return value
    return {}


def _select_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("report", "bubble_controller", "action_plan", "plan"):
        value = _mapping(payload.get(key))
        if value:
            return value
    return {}


def build_bubble_advisor_apply_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a route-friendly response for applying a bubble advisor patch.

    The returned payload intentionally keeps both ``patched_request`` and
    ``next_request_overlay`` so UI callers can either replace a next-run request
    draft or merge only the changed fields into an existing config draft.
    """

    data = _mapping(payload)
    result = prepare_bubble_advisor_next_request(
        _select_base_request(data),
        _select_report(data),
        action_id=str(data.get("action_id") or "").strip() or None,
        allow_current_mismatch=_boolish(data.get("allow_current_mismatch"), False),
        embed_ledger=_boolish(data.get("embed_ledger"), True),
        enable_next_run_observation=_boolish(data.get("keep_bubble_controller_enabled"), True),
    )
    status = str(result.get("status") or "")
    overlay = dict(_mapping(result.get("next_request_overlay")))
    return {
        "schema_version": 1,
        "adapter": "bubble_advisor_apply_payload_v0",
        "ok": status in {"applied", "no_change"},
        "status": status,
        "reason": str(result.get("reason") or ""),
        "can_apply_to_next_request": bool(result.get("can_apply_to_next_request")),
        "can_apply_during_current_run": False,
        "patched_request": dict(_mapping(result.get("patched_request"))),
        "prepared_request": dict(_mapping(result.get("prepared_request"))),
        "next_request_overlay": overlay,
        "config_patch": overlay,
        "applied_overlay": dict(_mapping(result.get("applied_overlay"))),
        "blocked_reasons": list(result.get("blocked_reasons") or []),
        "skipped": list(result.get("skipped") or []),
        "action_ledger": dict(_mapping(result.get("action_ledger"))),
        "raw_result": result,
    }


__all__ = ["build_bubble_advisor_apply_payload"]
