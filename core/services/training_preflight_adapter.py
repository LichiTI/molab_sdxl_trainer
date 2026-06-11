"""Training preflight helper checks shared by compatibility routes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.core.contracts import PlatformIssue, RunResult, RunStatus
from backend.core.services.dataset_analysis_adapter import analyze_dataset_payload
from backend.core.lulynx_trainer.module_offload_contract import (
    get_module_offload_conflict,
    is_swap_requested,
    resolve_module_offload_config,
)
from backend.core.lulynx_trainer.model_acceleration_application import model_acceleration_preflight_payload
from backend.core.services.compat_route_utils import serialize_resolved_result
from backend.core.services.descriptive_model_preflight import descriptive_model_preflight_payload
from backend.core.services.training_request_adapter import (
    build_training_config_from_schema_route_payload,
    derive_attention_backend,
    derive_schema_id,
    effective_execution_profile_id,
    normalize_compat_training_request,
    normalize_compat_training_route_data,
    placeholder_schema_error,
)

logger = logging.getLogger(__name__)


_ISSUE_KEYS = {"severity", "code", "message", "field", "hint", "details"}
def _placeholder_schema_preflight_payload(schema_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any] | None:
    normalized = str(schema_id or "").strip().lower().replace("_", "-")
    message = placeholder_schema_error(normalized)
    if message is None:
        return None
    return {
        "can_start": False,
        "errors": [
            {
                "severity": "error",
                "code": "schema_placeholder_not_runnable",
                "message": message,
            }
        ],
        "warnings": [],
        "config_resolution": {
            "schema_id": normalized,
            "model_type": str(raw_data.get("model_type") or ""),
            "training_type": str(raw_data.get("training_type") or "lora"),
            "native_route_status": "placeholder",
        },
    }

def default_execution_resolver_dependencies() -> tuple[Any, type[Exception]]:
    """Resolve the execution resolver lazily for route use."""

    from backend.core.execution_resolver import ResolutionError, get_execution_resolver

    return get_execution_resolver(), ResolutionError

def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "enable"}


def empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def dataset_preflight_probe(
    raw_data: Dict[str, Any],
    *,
    analyzer: Callable[[dict[str, Any]], dict[str, Any]] = analyze_dataset_payload,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    dataset_summary: Optional[Dict[str, Any]] = None
    warnings: List[Dict[str, str]] = []
    train_data_dir = str(raw_data.get("train_data_dir") or "").strip()
    if not train_data_dir:
        return dataset_summary, warnings
    try:
        dataset_data = analyzer(
            {
                "path": train_data_dir,
                "caption_extension": str(raw_data.get("caption_extension") or ".txt"),
            }
        )
        dataset_summary = dict(dataset_data.get("summary") or {})
        dataset_summary["path"] = train_data_dir
        repeat_warning = str((dataset_data.get("repeat_prefix") or {}).get("warning") or "").strip()
        if repeat_warning:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "dataset_repeat_prefix",
                    "message": repeat_warning,
                }
            )
    except Exception as exc:
        warnings.append(
            {
                "severity": "warning",
                "code": "dataset_probe_failed",
                "message": str(exc),
            }
        )
    return dataset_summary, warnings


def training_advisor_preflight_payload(
    advisor_config: Dict[str, Any],
    *,
    advisor_builder: Callable[[Dict[str, Any]], Any],
) -> Dict[str, Any]:
    try:
        report = advisor_builder(advisor_config).to_dict()
        return {
            "available": True,
            "summary": report.get("summary", {}),
            "vram": report.get("vram", {}),
            "dataset": report.get("dataset", {}),
            "compile_token": report.get("compile_token", {}),
            "a_tier": report.get("a_tier", {}),
            "b_tier": report.get("b_tier", {}),
            "findings": report.get("findings", []),
        }
    except Exception as exc:
        logger.warning("training advisor preflight report failed: %s", exc)
        return {"available": False, "error": str(exc)}


def _legacy_preflight_issue(item: Any, *, default_severity: str) -> PlatformIssue:
    if isinstance(item, PlatformIssue):
        return item
    if isinstance(item, dict):
        severity = str(item.get("severity") or default_severity or "warning").strip().lower() or "warning"
        code = str(item.get("code") or f"training_preflight_{severity}").strip()
        message = str(item.get("message") or item.get("detail") or code).strip()
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        extras = {key: value for key, value in item.items() if key not in _ISSUE_KEYS}
        if extras:
            details = {**details, "legacy": extras}
        return PlatformIssue(
            code=code,
            message=message,
            severity=severity,
            field=str(item.get("field") or ""),
            hint=str(item.get("hint") or ""),
            details=details,
        )
    severity = str(default_severity or "warning").strip().lower() or "warning"
    message = str(item or f"Training preflight {severity}").strip()
    return PlatformIssue(code=f"training_preflight_{severity}", message=message, severity=severity)


def training_preflight_run_result_from_payload(payload: Dict[str, Any], *, request_id: str = "") -> RunResult:
    """Wrap the legacy preflight payload in the common RunResult envelope."""

    data = dict(payload or {})
    errors = data.get("errors") if isinstance(data.get("errors"), list) else []
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    issues = [
        *[_legacy_preflight_issue(item, default_severity="error") for item in errors],
        *[_legacy_preflight_issue(item, default_severity="warning") for item in warnings],
    ]
    can_start = bool(data.get("can_start")) and not any(issue.severity == "error" for issue in issues)
    message = "Training preflight passed." if can_start else "Training preflight blocked."
    first_error = next((issue for issue in issues if issue.severity == "error"), None)
    if first_error is not None:
        message = f"Training preflight blocked: {first_error.message}"
    return RunResult(
        request_id=request_id,
        status=RunStatus.SUCCEEDED if can_start else RunStatus.FAILED,
        message=message,
        issues=issues,
        data=data,
    )


def training_preflight_payload_from_run_result(result: RunResult) -> Dict[str, Any]:
    """Return the legacy `/api/train/preflight` payload from a RunResult."""

    if isinstance(result.data, dict) and "can_start" in result.data:
        return dict(result.data)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for issue in result.issues:
        item = issue.model_dump(mode="json", exclude_none=True)
        if issue.severity == "error":
            errors.append(item)
        else:
            warnings.append(item)
    return {
        "can_start": result.ok and not errors,
        "errors": errors,
        "warnings": warnings,
        "config_resolution": {},
    }


def build_training_preflight_payload(
    *,
    raw_data: Dict[str, Any],
    training_request: Any,
    request: Any,
    effective_execution_profile_id: Callable[[Any], str],
    derive_schema_id: Callable[[Dict[str, Any]], str],
    derive_attention_backend: Callable[[Dict[str, Any], str], str],
    build_training_config_from_schema: Callable[[str, Dict[str, Any], Any], tuple[dict, str, str, dict]],
    resolver: Any,
    resolution_error_type: type[Exception],
    serialize_resolved_result: Callable[[Any], Dict[str, Any]],
    advisor_builder: Callable[[Dict[str, Any]], Any],
) -> Dict[str, Any]:
    execution_profile_id = effective_execution_profile_id(
        getattr(training_request, "execution_profile_id", "") or getattr(request, "execution_profile_id", "")
    )
    raw_data["execution_profile_id"] = execution_profile_id
    schema_id = getattr(training_request, "schema_id", "") or derive_schema_id(raw_data)
    model_type = getattr(training_request, "model_type", "") or getattr(request, "model_type", "sdxl")
    training_type = getattr(training_request, "training_type", "") or getattr(request, "training_type", "lora")
    config_resolution: Dict[str, Any] = {}
    advisor_config: Dict[str, Any] = raw_data
    placeholder_payload = _placeholder_schema_preflight_payload(schema_id, raw_data)
    if placeholder_payload is not None:
        return placeholder_payload
    is_webui_experimental_schema = schema_id in {
        "lab-distiller",
        "sdxl-turbo-lora",
        "anima-few-step-lora",
        "newbie-few-step-lora",
    }
    if schema_id and not is_webui_experimental_schema:
        try:
            trainer_config, model_type, training_type, config_resolution = build_training_config_from_schema(
                schema_id,
                raw_data,
                getattr(request, "extra_config_layers", []),
            )
            advisor_config = dict(trainer_config)
            advisor_config.setdefault("schema_id", schema_id)
            advisor_config.setdefault("model_type", model_type)
            advisor_config.setdefault("training_type", training_type)
        except ValueError as exc:
            descriptive_payload = descriptive_model_preflight_payload(
                raw_data=raw_data,
                schema_id=schema_id,
                model_type=model_type,
                training_type=training_type,
                error_message=str(exc),
            )
            if descriptive_payload is not None:
                return descriptive_payload
            return {
                "can_start": False,
                "errors": [{"severity": "error", "code": "schema_error", "message": str(exc)}],
                "warnings": [],
                "config_resolution": {},
            }
    elif is_webui_experimental_schema:
        model_type = (
            "anima" if schema_id == "anima-few-step-lora"
            else "newbie" if schema_id == "newbie-few-step-lora"
            else "sdxl"
        )
        training_type = "lab_distiller" if schema_id == "lab-distiller" else "lora"
        advisor_config = dict(raw_data)
        advisor_config.setdefault("schema_id", schema_id)
        advisor_config.setdefault("model_type", model_type)
        advisor_config.setdefault("training_type", training_type)
        config_resolution = {
            "schema_id": schema_id,
            "webui_experimental": True,
            "message": "This WebUI-only experimental trainer is launched through /api/lulynx-lab.",
        }

    dataset_summary, dataset_probe_warnings = dataset_preflight_probe(raw_data)
    memory_errors, memory_warnings = memory_swap_preflight(raw_data)
    module_offload_errors, module_offload_warnings = module_offload_preflight(
        raw_data,
        schema_id=schema_id,
        training_type=training_type,
    )
    precision_swap_profile = precision_swap_preflight_profile(advisor_config, schema_id=schema_id)
    native_unet_profile = native_unet_preflight_profile(advisor_config, schema_id=schema_id)
    model_acceleration = model_acceleration_preflight_payload(
        advisor_config,
        schema_id=schema_id,
        training_type=training_type,
    )
    requested_attention = derive_attention_backend(
        raw_data,
        getattr(training_request, "attention_backend", "") or getattr(request, "attention_backend", "auto"),
    )

    try:
        resolved = resolver.resolve(
            execution_profile_id=execution_profile_id,
            requested_attention=requested_attention,
            schema_id=schema_id,
            allow_attention_fallback=getattr(request, "allow_attention_fallback", True),
            model_type=model_type,
            training_type=training_type,
        )
    except resolution_error_type as exc:
        return {
            "can_start": False,
            "errors": [{"severity": "error", "code": getattr(exc, "code", "resolution_error"), "message": str(exc)}],
            "warnings": [],
        }

    warnings = list(getattr(resolved, "warnings", []) or [])
    warnings.extend(memory_warnings)
    warnings.extend(module_offload_warnings)
    warnings.extend(dataset_probe_warnings)
    errors = list(memory_errors)
    errors.extend(module_offload_errors)
    training_advisor = training_advisor_preflight_payload(
        advisor_config,
        advisor_builder=advisor_builder,
    )
    recommended_config_patch = dict(model_acceleration.get("recommended_config_patch") or {})
    return {
        "can_start": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": dataset_summary,
        "resolved": serialize_resolved_result(resolved),
        "execution_profile_id": getattr(resolved, "execution_profile_id", execution_profile_id),
        "requested_attention_backend": getattr(resolved, "requested_attention", requested_attention),
        "resolved_attention_backend": getattr(
            resolved,
            "resolved_attention",
            getattr(resolved, "requested_attention", requested_attention),
        ),
        "config_resolution": config_resolution,
        "training_advisor": training_advisor,
        "model_acceleration": model_acceleration,
        "recommended_config_patch": recommended_config_patch,
        "precision_swap_profile": precision_swap_profile,
        "native_unet_profile": native_unet_profile,
    }


def invalid_training_request_preflight_payload(message: str) -> Dict[str, Any]:
    return {
        "can_start": False,
        "errors": [{"severity": "error", "code": "invalid_training_request", "message": message}],
        "warnings": [],
        "config_resolution": {},
    }


def build_training_preflight_run_result(
    *,
    raw_data: Dict[str, Any],
    training_request: Any,
    request: Any,
    effective_execution_profile_id: Callable[[Any], str],
    derive_schema_id: Callable[[Dict[str, Any]], str],
    derive_attention_backend: Callable[[Dict[str, Any], str], str],
    build_training_config_from_schema: Callable[[str, Dict[str, Any], Any], tuple[dict, str, str, dict]],
    resolver: Any,
    resolution_error_type: type[Exception],
    serialize_resolved_result: Callable[[Any], Dict[str, Any]],
    advisor_builder: Callable[[Dict[str, Any]], Any],
) -> RunResult:
    """Build a request-native RunResult while preserving legacy preflight data."""

    payload = build_training_preflight_payload(
        raw_data=raw_data,
        training_request=training_request,
        request=request,
        effective_execution_profile_id=effective_execution_profile_id,
        derive_schema_id=derive_schema_id,
        derive_attention_backend=derive_attention_backend,
        build_training_config_from_schema=build_training_config_from_schema,
        resolver=resolver,
        resolution_error_type=resolution_error_type,
        serialize_resolved_result=serialize_resolved_result,
        advisor_builder=advisor_builder,
    )
    return training_preflight_run_result_from_payload(
        payload,
        request_id=str(getattr(training_request, "request_id", "") or ""),
    )


def build_training_preflight_route_payload(
    *,
    raw_data: Dict[str, Any],
    request: Any,
    backend_root: Path | None = None,
    build_training_config_from_schema: Callable[[str, Dict[str, Any], Any], tuple[dict, str, str, dict]] | None = None,
    advisor_builder: Callable[[Dict[str, Any]], Any],
    resolver_dependencies: Callable[[], tuple[Any, type[Exception]]] = default_execution_resolver_dependencies,
) -> Dict[str, Any]:
    """Route-facing `/api/train/preflight` adapter with default dependency wiring."""

    if build_training_config_from_schema is None:
        if backend_root is None:
            raise ValueError("backend_root is required")

        def build_training_config_from_schema(schema_id: str, data: Dict[str, Any], extra_config_layers: Any = None):
            return build_training_config_from_schema_route_payload(
                schema_id,
                data,
                extra_config_layers,
                backend_root=backend_root,
            )

    raw_data = normalize_compat_training_route_data(raw_data, request)
    try:
        training_request = normalize_compat_training_request(raw_data)
    except ValueError as exc:
        return training_preflight_payload_from_run_result(
            training_preflight_run_result_from_payload(invalid_training_request_preflight_payload(str(exc)))
        )

    resolver, resolution_error_type = resolver_dependencies()
    return training_preflight_payload_from_run_result(build_training_preflight_run_result(
        raw_data=raw_data,
        training_request=training_request,
        request=request,
        effective_execution_profile_id=effective_execution_profile_id,
        derive_schema_id=derive_schema_id,
        derive_attention_backend=derive_attention_backend,
        build_training_config_from_schema=build_training_config_from_schema,
        resolver=resolver,
        resolution_error_type=resolution_error_type,
        serialize_resolved_result=serialize_resolved_result,
        advisor_builder=advisor_builder,
    ))


def memory_swap_preflight(raw_data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    granularity = str(raw_data.get("swap_granularity", "off") or "off").strip().lower().replace("-", "_")
    valid = {"off", "auto", "block", "merged_block", "layer"}
    if granularity not in valid:
        errors.append({"severity": "error", "code": "invalid_swap_granularity", "message": f"Invalid swap_granularity: {granularity}"})
        granularity = "off"
    try:
        swap_ratio = float(raw_data.get("swap_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        errors.append({"severity": "error", "code": "invalid_swap_ratio", "message": "swap_ratio must be a number between 0 and 1"})
        swap_ratio = 0.0
    if swap_ratio < 0.0 or swap_ratio > 1.0:
        errors.append({"severity": "error", "code": "invalid_swap_ratio", "message": "swap_ratio must be between 0 and 1"})
    try:
        swap_count = int(raw_data.get("swap_count", 0) or 0)
    except (TypeError, ValueError):
        errors.append({"severity": "error", "code": "invalid_swap_count", "message": "swap_count must be a non-negative integer"})
        swap_count = 0
    try:
        blocks_to_swap = int(raw_data.get("blocks_to_swap", 0) or 0)
    except (TypeError, ValueError):
        blocks_to_swap = 0

    swap_enabled = (granularity != "off" and (swap_ratio > 0.0 or swap_count > 0 or granularity == "auto")) or blocks_to_swap > 0
    if not swap_enabled:
        return errors, warnings

    if truthy(raw_data.get("torch_compile")):
        errors.append({"severity": "error", "code": "swap_torch_compile_conflict", "message": "Memory swap is incompatible with torch_compile. Disable torch_compile or turn swap off."})
    if truthy(raw_data.get("vram_swap_to_ram")):
        errors.append({"severity": "error", "code": "swap_vram_swap_conflict", "message": "Memory swap is incompatible with vram_swap_to_ram. Use only one VRAM offload strategy."})
    if truthy(raw_data.get("safe_fallback")) or truthy(raw_data.get("newbie_safe_fallback")):
        errors.append({"severity": "error", "code": "swap_safe_fallback_conflict", "message": "Memory swap is incompatible with safe_fallback. Disable one of them."})
    if granularity == "layer" and truthy(raw_data.get("gradient_checkpointing")):
        errors.append({"severity": "error", "code": "layer_swap_gradient_checkpointing_conflict", "message": "Layer swap is incompatible with gradient_checkpointing. Use block/merged_block swap or disable gradient checkpointing."})
    return errors, warnings


def module_offload_preflight(
    raw_data: Dict[str, Any],
    *,
    schema_id: str = "",
    training_type: str = "",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    view = resolve_module_offload_config(raw_data)
    if not view.requested:
        return errors, warnings

    route = str(schema_id or training_type or raw_data.get("training_type", "") or "").strip().lower().replace("_", "-")
    try:
        num_processes = int(raw_data.get("num_processes", 1) or 1)
    except (TypeError, ValueError):
        num_processes = 1
    try:
        num_machines = int(raw_data.get("num_machines", 1) or 1)
    except (TypeError, ValueError):
        num_machines = 1
    distributed_enabled = (
        truthy(raw_data.get("multi_gpu"))
        or truthy(raw_data.get("enable_distributed"))
        or truthy(raw_data.get("enable_distributed_training"))
        or num_processes > 1
        or num_machines > 1
    )
    pipeline_enabled = (
        any(token in route for token in ("controlnet", "ip-adapter", "lllite"))
        or truthy(raw_data.get("ip_adapter_enabled"))
        or bool(str(raw_data.get("controlnet_model", "") or "").strip())
    )
    conflict_codes: List[str] = []
    if is_swap_requested(raw_data):
        conflict_codes.append("swap")
    if truthy(raw_data.get("vram_swap_to_ram")):
        conflict_codes.append("vram_swap_to_ram")
    if truthy(raw_data.get("safe_fallback")) or truthy(raw_data.get("newbie_safe_fallback")):
        conflict_codes.append("safe_fallback")
    if truthy(raw_data.get("torch_compile")):
        conflict_codes.append("torch_compile")
    if distributed_enabled:
        conflict_codes.append("distributed")
    if truthy(raw_data.get("deepspeed")):
        conflict_codes.append("deepspeed")
    if pipeline_enabled:
        conflict_codes.append("pipeline")
    if truthy(raw_data.get("gradient_checkpointing")):
        conflict_codes.append("gradient_checkpointing")
    if truthy(raw_data.get("cpu_offload_checkpointing")):
        conflict_codes.append("cpu_offload_checkpointing")

    for code in conflict_codes:
        error_code, message = get_module_offload_conflict(code)
        errors.append({"severity": "error", "code": error_code, "message": message})
    return errors, warnings


def parse_resolution_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return max(int(value[0] or 1024), 1), max(int(value[1] or 1024), 1)
        except (TypeError, ValueError):
            return (1024, 1024)
    text = str(value or "").strip().lower().replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        if len(parts) >= 2:
            return max(int(parts[0]), 1), max(int(parts[1]), 1)
        if len(parts) == 1:
            size = max(int(parts[0]), 1)
            return size, size
    except ValueError:
        pass
    return (1024, 1024)


def precision_swap_preflight_profile(raw_data: Dict[str, Any], *, schema_id: str = "") -> Optional[Dict[str, Any]]:
    if not truthy(raw_data.get("lulynx_precision_swap_enabled")):
        return None
    route = str(schema_id or raw_data.get("model_train_type") or raw_data.get("model_type") or "").strip().lower()
    if "sdxl" not in route and str(raw_data.get("model_arch") or "").strip().lower() != "sdxl":
        return None
    strategy = str(raw_data.get("lulynx_precision_swap_strategy") or "balanced").strip().lower()
    if strategy not in {"balanced", "aggressive", "off"}:
        strategy = "balanced"
    resolution = parse_resolution_pair(raw_data.get("resolution", raw_data.get("max_resolution", "1024,1024")))
    names = ["down.0", "down.1", "down.2", "mid.0", "up.0", "up.1", "up.2"]
    stages = ["down", "down", "down", "mid", "up", "up", "up"]
    if strategy == "off":
        count = 0
    elif strategy == "aggressive":
        count = 4
    else:
        count = 2
    pixels = max(resolution[0], 1) * max(resolution[1], 1)
    scale = pixels / float(1024 * 1024)

    def activation_hint(stage: str, order: int) -> float:
        stage_weight = {"down": 0.65, "mid": 0.9, "up": 1.0}.get(stage, 0.5)
        order_weight = 1.0 + max(order, 0) * 0.03
        return 128.0 * scale * stage_weight * order_weight

    def selection_score(stage: str, order: int) -> float:
        stage_bonus = {"up": 1.18, "mid": 1.08, "down": 0.94}.get(stage, 1.0)
        return activation_hint(stage, order) * stage_bonus

    ranked_indices = sorted(range(len(names)), key=lambda idx: selection_score(stages[idx], idx), reverse=True)
    selected_indices = sorted(ranked_indices[:count]) if count else []
    selected_names = [names[i] for i in selected_indices]

    units = []
    for idx, name in enumerate(names):
        hint = activation_hint(stages[idx], idx)
        units.append({
            "name": name,
            "stage": stages[idx],
            "order": idx,
            "parameter_mb": 0.0,
            "activation_hint_mb": round(hint, 3),
            "recompute_safe": True,
            "selected": idx in selected_indices,
        })
    selected_hint = sum(float(units[i]["activation_hint_mb"]) for i in selected_indices)
    return {
        "family": "sdxl",
        "backend": "selected_block_swap",
        "strategy": strategy,
        "profile_source": "preflight_static",
        "resolution": [resolution[0], resolution[1]],
        "units_total": len(units),
        "selected_count": len(selected_indices),
        "selected_names": selected_names,
        "selected_indices": selected_indices,
        "compatible_blocks_to_swap": len(selected_indices),
        "total_parameter_mb": 0.0,
        "selected_parameter_mb": 0.0,
        "selected_activation_hint_mb": round(selected_hint, 3),
        "units": units,
        "runtime_observations": {},
    }


def native_unet_preflight_profile(raw_data: Dict[str, Any], *, schema_id: str = "") -> Optional[Dict[str, Any]]:
    route = str(schema_id or raw_data.get("model_train_type") or raw_data.get("model_type") or "").strip().lower()
    if "sdxl" not in route and str(raw_data.get("model_arch") or "").strip().lower() != "sdxl":
        return None
    backend = str(raw_data.get("sdxl_unet_backend") or "diffusers").strip().lower().replace("-", "_")
    if not backend or backend == "diffusers":
        return None
    aliases = {
        "native": "lulynx_native",
        "shadow": "native_shadow",
        "proxy": "native_proxy",
        "skeleton": "native_skeleton",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"native_shadow", "native_proxy", "native_skeleton", "lulynx_native"}:
        return None
    weight_residency = str(raw_data.get("lulynx_weight_residency") or "resident").strip().lower().replace("-", "_")
    if weight_residency not in {"resident", "linear_cpu_pinned", "linear_conv_cpu_pinned"}:
        weight_residency = "resident"
    try:
        weight_residency_min_params = max(int(float(raw_data.get("lulynx_weight_residency_min_params") or 0)), 0)
    except Exception:
        weight_residency_min_params = 0
    try:
        from backend.core.lulynx_trainer.native_unet import build_sdxl_native_unet_preflight_profile

        profile = build_sdxl_native_unet_preflight_profile(backend=backend)
        if profile:
            profile["weight_residency"] = {"mode": weight_residency, "min_parameter_count": weight_residency_min_params}
            return profile
    except Exception:
        # Some launcher-only runtimes do not import torch.  Keep preflight
        # useful there, while flashattention training runtimes report the real
        # synthetic forward probe above.
        pass
    mode = "native_full" if backend == "lulynx_native" else ("reference_proxy" if backend == "native_proxy" else ("skeleton_metadata" if backend == "native_skeleton" else "shadow"))
    return {
        "family": "sdxl",
        "backend": backend,
        "available": True,
        "mode": mode,
        "active": backend in {"native_proxy", "lulynx_native"},
        "blocks_total": 7,
        "native_forward_integrated": False,
        "native_forward_probe_ok": None,
        "native_forward_probe": {
            "ok": None,
            "mode": "deferred_to_training_runtime",
            "reason": "preflight runtime could not import native torch probe; training runtime will report the real probe",
        },
        "native_coverage": {
            "status": "available",
            "skeleton_ready": True,
            "native_forward_integrated": False,
            "native_forward_probe_ok": None,
            "implemented_top_blocks": 7,
            "down_blocks": 3,
            "up_blocks": 3,
            "mid_blocks": 1,
            "cross_attn_down_blocks": 2,
            "cross_attn_up_blocks": 2,
            "attention_backend": "sdpa|flash2",
            "profile_source": "preflight_static",
        },
        "weight_residency": {"mode": weight_residency},
    }
