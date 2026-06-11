"""Runtime optimization intent resolver.

The service is deliberately lightweight: it normalizes request/config intent into
plain compatibility fields without importing torch or model libraries.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from core.contracts.runtime import RuntimeOptimizationRequest, RuntimeOptimizationResolution
from core.lulynx_trainer.model_acceleration_matrix import compile_policy_for


_COMPILE_FIELDS = {
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "torch_compile_dynamic",
    "torch_compile_fullgraph",
    "torch_compile_scope",
    "anima_compile_scope",
    "compile_cache_enabled",
    "compile_contract_strict",
    "compile_static_shape_drop_last",
    "compile_require_cache_first",
    "native_token_bucket_compile",
}

_DIT_ROUTES = {"anima", "newbie", "flux"}
_SDXL_ROUTES = {"sdxl", "xl"}


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _normalize_route(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).strip().lower().replace("\\", "/")
    compact = text.replace("_", "-")
    if "newbie" in compact:
        return "newbie"
    if "anima" in compact or "qwen-image" in compact or "qwen_image" in text:
        return "anima"
    if "sdxl" in compact or "xl-lora" in compact or "xl-" in compact:
        return "sdxl"
    if "sd15" in compact or "sd-1" in compact or "sd-lora" in compact:
        return "sd15"
    return compact.split("/", 1)[0].split("-", 1)[0] if compact else ""


def _explicit_fields(payload: Mapping[str, Any], explicit_fields: Iterable[str] | None = None) -> set[str]:
    fields = set(explicit_fields or ())
    fields.update(str(key) for key in payload.keys())
    return fields


def build_runtime_optimization_request(
    payload: Mapping[str, Any],
    *,
    route: str = "",
    explicit_fields: Iterable[str] | None = None,
) -> RuntimeOptimizationRequest:
    """Create a request-native optimization intent from a normalized payload."""

    data = {key: payload.get(key) for key in payload.keys() if key in _COMPILE_FIELDS or key.startswith("compile_")}
    data["route"] = route or str(payload.get("route") or payload.get("schema_id") or payload.get("model_type") or "")
    data["explicit_fields"] = sorted(_explicit_fields(payload, explicit_fields))
    return RuntimeOptimizationRequest.model_validate(data)


def _explicit_compile_runtime_from_fields(request: RuntimeOptimizationRequest, explicit: set[str]) -> str:
    if "torch_compile" in explicit and request.torch_compile is False:
        return "off"
    if request.compile_cache_enabled is True:
        if str(request.torch_compile_backend or "").strip().lower().replace("-", "_") in {"cudagraph", "cudagraphs"}:
            return "compile_cudagraph"
        return "compile_cache"
    if str(request.torch_compile_backend or "").strip().lower().replace("-", "_") in {"cudagraph", "cudagraphs"}:
        return "cudagraph"
    if (
        _boolish(request.torch_compile, default=False)
        or _has_text(request.torch_compile_scope)
        or _has_text(request.anima_compile_scope)
    ):
        return "compile"
    return "off"


def _resolve_auto_compile_runtime(
    request: RuntimeOptimizationRequest,
    explicit: set[str],
) -> tuple[str, str]:
    explicit_compile_fields = explicit.intersection(_COMPILE_FIELDS - {"compile_cache_enabled"})
    if explicit_compile_fields or request.compile_cache_enabled is not None:
        runtime = _explicit_compile_runtime_from_fields(request, explicit)
        return runtime, f"compile_runtime=auto kept explicit compile fields as {runtime}"

    route = _normalize_route(request.route, request.schema_id)
    if route in _DIT_ROUTES:
        return "compile_cache", f"compile_runtime=auto resolved to compile_cache for {route}"
    if route in _SDXL_ROUTES or route == "sd15":
        return "off", f"compile_runtime=auto resolved to off for conservative {route or 'sdxl'} route"
    return "off", "compile_runtime=auto resolved to off for unknown route"


def _apply_route_compile_defaults(
    fields: Dict[str, Any],
    request: RuntimeOptimizationRequest,
    *,
    compile_runtime: str,
    explicit: set[str],
    reasons: list[str],
) -> None:
    if compile_runtime not in {"compile", "compile_cache", "compile_cudagraph"}:
        return
    route = _normalize_route(request.route, request.schema_id)
    spec = compile_policy_for(route, "aggressive")
    if spec is None:
        return

    if str(fields.get("compile_shape_strategy") or "auto").strip().lower() == "auto":
        fields["compile_shape_strategy"] = spec.shape_strategy
        reasons.append(f"compile shape defaulted to {spec.shape_strategy} for {route or 'unknown'}")
    if str(fields.get("compile_target_strategy") or "auto").strip().lower() == "auto":
        fields["compile_target_strategy"] = spec.target_strategy
        reasons.append(f"compile target defaulted to {spec.target_strategy} for {route or 'unknown'}")

    for key, value in spec.extra_patch.items():
        if key in explicit:
            continue
        fields.setdefault(key, value)


def resolve_runtime_optimization(request: RuntimeOptimizationRequest) -> RuntimeOptimizationResolution:
    """Resolve compile/runtime intent into plain config adapter fields."""

    explicit = set(request.explicit_fields or [])
    compile_runtime = request.compile_runtime
    if compile_runtime == "auto":
        compile_runtime, auto_reason = _resolve_auto_compile_runtime(request, explicit)
    else:
        auto_reason = ""

    fields: Dict[str, Any] = {
        "compile_runtime": compile_runtime,
        "compile_shape_strategy": request.compile_shape_strategy,
        "compile_target_strategy": request.compile_target_strategy,
    }
    reasons: list[str] = []
    if auto_reason:
        reasons.append(auto_reason)
    compile_active = _boolish(request.torch_compile, default=False) or _has_text(request.torch_compile_scope) or _has_text(request.anima_compile_scope)

    for key in (
        "torch_compile_backend",
        "torch_compile_mode",
        "torch_compile_scope",
        "anima_compile_scope",
    ):
        value = getattr(request, key)
        if _has_text(value):
            fields[key] = value
    for key in (
        "torch_compile",
        "torch_compile_dynamic",
        "torch_compile_fullgraph",
        "compile_cache_enabled",
        "compile_contract_strict",
        "compile_static_shape_drop_last",
        "compile_require_cache_first",
        "native_token_bucket_compile",
    ):
        value = getattr(request, key)
        if value is not None:
            fields[key] = value

    if compile_runtime == "off":
        if compile_active:
            reasons.append("compile_runtime=off did not override explicit torch compile fields")
        else:
            fields["torch_compile"] = False
            fields.setdefault("torch_compile_scope", "")
        return RuntimeOptimizationResolution(
            route=request.route,
            fields=fields,
            requested=request.model_dump(mode="json"),
            reasons=reasons,
        )

    _apply_route_compile_defaults(
        fields,
        request,
        compile_runtime=compile_runtime,
        explicit=explicit,
        reasons=reasons,
    )

    if "torch_compile" not in explicit or request.torch_compile is None:
        fields["torch_compile"] = True
        reasons.append("compile_runtime enabled torch_compile")
    elif request.torch_compile is False:
        reasons.append("explicit torch_compile=false kept over compile_runtime")

    if compile_runtime == "compile_cache" and "torch_compile_scope" not in explicit and not _has_text(request.torch_compile_scope):
        fields["torch_compile_scope"] = "per_block"
        reasons.append("compile_runtime=compile_cache defaulted torch_compile_scope=per_block")

    if compile_runtime in {"compile_cache", "compile_cudagraph"}:
        if "compile_cache_enabled" not in explicit or request.compile_cache_enabled is None:
            fields["compile_cache_enabled"] = True
    elif compile_runtime == "compile":
        if "compile_cache_enabled" not in explicit or request.compile_cache_enabled is None:
            fields["compile_cache_enabled"] = False

    if compile_runtime in {"cudagraph", "compile_cudagraph"}:
        if "torch_compile_backend" not in explicit or not _has_text(request.torch_compile_backend):
            fields["torch_compile_backend"] = "cudagraphs"

    return RuntimeOptimizationResolution(
        route=request.route,
        fields=fields,
        requested=request.model_dump(mode="json"),
        reasons=reasons,
    )


def resolve_runtime_optimization_payload(payload: Mapping[str, Any], *, route: str = "") -> RuntimeOptimizationResolution:
    request = build_runtime_optimization_request(payload, route=route)
    return resolve_runtime_optimization(request)
