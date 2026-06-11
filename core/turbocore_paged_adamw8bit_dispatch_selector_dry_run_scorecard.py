"""Report-only dispatch selector dry-run for PagedAdamW8bit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_paged_adamw8bit_training_tensor_binding_canary_scorecard import (
    build_paged_adamw8bit_training_tensor_binding_canary_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
SELECTOR_KIND = "paged_adamw8bit_native_dispatch_selector_dry_run_v0"
LAUNCH_PLAN = "paged_adamw8bit_training_tensor_binding_launch_plan_v0"


def build_paged_adamw8bit_dispatch_selector_dry_run_scorecard(
    *,
    training_tensor_binding_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    request_shape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a request-shaped selector decision without dispatching."""

    tensor_binding = dict(
        training_tensor_binding_report
        or build_paged_adamw8bit_training_tensor_binding_canary_scorecard(run_live_probe=False)
    )
    mode = _normalize_mode(native_training_mode)
    request = _request_shape(request_shape)
    route = _route_decision(tensor_binding, mode, request)
    selector_ready = (
        bool(tensor_binding.get("training_tensor_binding_canary_ready", False))
        and route["decision"]
        in {
            "off",
            "would_select_native_but_dispatch_disabled",
            "blocked_before_real_training_matrix",
        }
        and not bool(route.get("native_dispatch_allowed", True))
    )
    validations = _validations(tensor_binding, route, request, selector_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_dispatch_selector_dry_run_scorecard_v0",
        "gate": "paged_adamw8bit_dispatch_selector_dry_run",
        "ok": not failed,
        "promotion_ready": False,
        "dispatch_selector_dry_run_ready": selector_ready,
        "runtime_dispatch_ready": False,
        "real_training_matrix_ready": False,
        "native_training_mode": mode,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "selector_kind": SELECTOR_KIND,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "request_shape": request,
        "route_decision": route,
        "training_tensor_binding_summary": dict(tensor_binding.get("summary") or {}),
        "validations": validations,
        "summary": {
            "dispatch_selector_dry_run_ready": selector_ready,
            "route_decision": route.get("decision"),
            "route_reason": route.get("reason"),
            "training_tensor_binding_canary_ready": bool(
                tensor_binding.get("training_tensor_binding_canary_ready", False)
            ),
            "real_training_matrix_ready": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_real_training_matrix_missing",
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run PagedAdamW8bit real-training matrix gate before enabling canary dispatch"
            if selector_ready
            else "fix PagedAdamW8bit dispatch selector dry-run blockers"
        ),
        "notes": [
            "This selector is request-shaped but never dispatches a native optimizer update.",
            "It proves route decisions after P8L without treating synthetic canary evidence as production readiness.",
            "Real training matrix evidence is still required before any canary dispatch promotion.",
        ],
    }


def _request_shape(request_shape: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(request_shape or {})
    return {
        "schema_version": 1,
        "optimizer_kind": str(raw.get("optimizer_kind") or OPTIMIZER_KIND),
        "optimizer_family": str(raw.get("optimizer_family") or OPTIMIZER_FAMILY),
        "param_dtype": str(raw.get("param_dtype") or "float32"),
        "grad_dtype": str(raw.get("grad_dtype") or "float32"),
        "state_dtype": str(raw.get("state_dtype") or "uint8_blockwise"),
        "device_type": str(raw.get("device_type") or "cuda"),
        "contiguous": bool(raw.get("contiguous", True)),
        "checkpoint_adapter_runtime": bool(raw.get("checkpoint_adapter_runtime", True)),
        "training_tensor_binding_canary": bool(raw.get("training_tensor_binding_canary", True)),
    }


def _route_decision(
    tensor_binding: Mapping[str, Any],
    mode: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    request_ok, request_blockers = _request_supported(request)
    tensor_ready = bool(tensor_binding.get("training_tensor_binding_canary_ready", False))
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not request_ok:
        decision = "fallback"
        reason = "request_shape_not_supported"
    elif not tensor_ready:
        decision = "fallback"
        reason = "training_tensor_binding_canary_missing"
    elif mode == "observe":
        decision = "would_select_native_but_dispatch_disabled"
        reason = "observe_mode_and_real_training_matrix_missing"
    elif mode in {"canary", "auto"}:
        decision = "blocked_before_real_training_matrix"
        reason = "real_training_matrix_missing"
    else:
        decision = "fallback"
        reason = "unknown_native_training_mode"
    return {
        "schema_version": 1,
        "selector_kind": SELECTOR_KIND,
        "feature": "paged_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": request_ok,
        "request_blockers": request_blockers,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "launch_plan": LAUNCH_PLAN,
        "missing_before_dispatch": [
            "real_training_matrix",
            "promotion_review",
        ],
    }


def _request_supported(request: Mapping[str, Any]) -> tuple[bool, list[str]]:
    blockers = []
    if request.get("optimizer_kind") != OPTIMIZER_KIND:
        blockers.append("optimizer_kind_not_paged_adamw8bit")
    if request.get("optimizer_family") != OPTIMIZER_FAMILY:
        blockers.append("optimizer_family_not_adamw_quantized_paged")
    if request.get("param_dtype") != "float32" or request.get("grad_dtype") != "float32":
        blockers.append("dtype_not_float32_probe_matrix")
    if request.get("state_dtype") != "uint8_blockwise":
        blockers.append("state_dtype_not_uint8_blockwise")
    if request.get("device_type") != "cuda":
        blockers.append("device_not_cuda")
    if not bool(request.get("contiguous", False)):
        blockers.append("tensor_not_contiguous")
    if not bool(request.get("checkpoint_adapter_runtime", False)):
        blockers.append("checkpoint_adapter_runtime_missing")
    if not bool(request.get("training_tensor_binding_canary", False)):
        blockers.append("training_tensor_binding_canary_missing")
    return not blockers, blockers


def _validations(
    tensor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    selector_ready: bool,
) -> list[dict[str, Any]]:
    request_ok, _ = _request_supported(request)
    return [
        _validation(
            "p8l_training_tensor_binding_canary_ready",
            bool(tensor_binding.get("training_tensor_binding_canary_ready", False)),
            "paged_adamw8bit_training_tensor_binding_canary_missing",
        ),
        _validation(
            "request_shape_supported",
            request_ok,
            "paged_adamw8bit_dispatch_selector_request_not_supported",
        ),
        _validation(
            "selector_dry_run_manifest",
            selector_ready,
            "paged_adamw8bit_dispatch_selector_dry_run_missing",
        ),
        _validation(
            "selector_blocks_dispatch",
            route.get("decision") in {"would_select_native_but_dispatch_disabled", "blocked_before_real_training_matrix"}
            and not bool(route.get("native_dispatch_allowed", True)),
            "paged_adamw8bit_dispatch_selector_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "canary").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "canary"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_paged_adamw8bit_dispatch_selector_dry_run_scorecard"]
