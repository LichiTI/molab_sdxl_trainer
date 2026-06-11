"""TurboCore capability report helpers.

The retired Rust launcher/API bridge used to live under ``backend/native``.
This module reports TurboCore training capabilities separately so that
developer-only TurboCore flags cannot be mistaken for an implemented training
runtime.
"""

from __future__ import annotations

import importlib.util
import importlib
import os
import platform
import sys
from typing import Any, Dict, Iterable, List, Optional

try:
    from core.turbocore_evidence import build_turbocore_evidence, build_turbocore_validation_status
except Exception:  # pragma: no cover - import layout differs in direct smoke runs
    build_turbocore_evidence = None  # type: ignore[assignment]
    build_turbocore_validation_status = None  # type: ignore[assignment]
try:
    from core.turbocore_native_abi import validate_workspace_pipeline_native_capabilities
except Exception:  # pragma: no cover - import layout differs in direct smoke runs
    validate_workspace_pipeline_native_capabilities = None  # type: ignore[assignment]
try:
    from core.turbocore_workspace_pipeline import build_turbocore_native_training_capability_stub
except Exception:  # pragma: no cover - import layout differs in direct smoke runs
    build_turbocore_native_training_capability_stub = None  # type: ignore[assignment]
try:
    from core.services.native_module_loader import ensure_lulynx_native_artifact_path
except Exception:  # pragma: no cover - import layout differs in direct smoke runs
    ensure_lulynx_native_artifact_path = None  # type: ignore[assignment]


KNOWN_TURBOCORE_FEATURES = (
    "lora_fused",
    "native_optimizer",
    "static_route_step",
    "data_pipeline",
    "workspace_pool",
    "dataset_staging",
    "experimental_fp8",
)

_INACTIVE_TRAINING_PATH_FEATURE_REASONS = {
    "lora_fused": "native_lora_fused_kernel_not_implemented",
}

_TRAINING_ROUTE_CANDIDATES = {
    ("anima", "lora"),
    ("newbie", "lora"),
    ("sdxl", "lora"),
}


def _canonical_native_stub() -> Dict[str, Any]:
    if build_turbocore_native_training_capability_stub is None:
        return {
            "schema_version": 1,
            "training_path_enabled": False,
            "training_bridge": {
                "available": False,
                "status": "unavailable",
                "reason": "python_native_stub_builder_unavailable",
            },
            "features": {},
        }
    try:
        return dict(build_turbocore_native_training_capability_stub())
    except Exception as exc:  # pragma: no cover - defensive fallback only
        return {
            "schema_version": 1,
            "training_path_enabled": False,
            "training_bridge": {
                "available": False,
                "status": "unavailable",
                "reason": "python_native_stub_builder_failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "features": {},
        }


def _inject_native_artifact_dir_from_env() -> None:
    if ensure_lulynx_native_artifact_path is not None:
        ensure_lulynx_native_artifact_path()
        return
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    if raw not in sys.path and os.path.isdir(raw):
        sys.path.insert(0, raw)


def _listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    normalized = str(value).strip().lower()
    return [normalized] if normalized else []


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _report_reason_for_native_feature(
    feature: str,
    native_feature: Dict[str, Any],
    native_training_bridge: Dict[str, Any],
) -> str:
    reason = str(native_feature.get("reason") or "native_feature_unavailable")
    if not bool(native_training_bridge.get("training_path_enabled", False)):
        return _INACTIVE_TRAINING_PATH_FEATURE_REASONS.get(feature, reason)
    return reason


def probe_native_launcher_bridge() -> Dict[str, Any]:
    """Report the retired PyO3 launcher bridge status."""
    return {
        "module": "lulynx_native",
        "importable": False,
        "origin": None,
        "native_root": "",
        "artifacts_found": [],
        "status": "retired",
        "reason": "legacy_launcher_bridge_removed",
    }


def probe_native_training_bridge() -> Dict[str, Any]:
    """Probe the Rust-side TurboCore training capability bridge.

    The bridge may exist before any runnable kernels exist.  A successful probe
    therefore only means Python can query native capability metadata; individual
    features still decide whether they are active.
    """
    _inject_native_artifact_dir_from_env()

    diagnostic: Dict[str, Any] = {
        "module": "lulynx_native",
        "importable": False,
        "origin": None,
        "provider": "python_stub",
        "status": "capability_stub",
        "reason": "native_training_capability_stub",
    }
    stub = _canonical_native_stub()
    bridge = dict(stub.get("training_bridge") or {})
    bridge.setdefault("available", bool(bridge.get("available", False)))
    bridge.setdefault("status", "capability_stub" if bridge["available"] else "unavailable")
    bridge.setdefault("reason", "native_training_capability_stub")
    bridge["features"] = stub.get("features") if isinstance(stub.get("features"), dict) else {}
    bridge["schema_version"] = stub.get("schema_version", 1)
    bridge["training_path_enabled"] = bool(stub.get("training_path_enabled", False))
    bridge["diagnostic"] = diagnostic

    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        bridge["diagnostic"] = {
            **diagnostic,
            "status": "unavailable",
            "reason": "lulynx_native_not_importable",
        }
        return bridge

    origin = getattr(spec, "origin", None)
    bridge["diagnostic"] = {
        **diagnostic,
        "importable": True,
        "origin": str(origin) if origin else None,
    }
    try:
        native = importlib.import_module("lulynx_native")
    except Exception as exc:  # pragma: no cover - depends on local native build
        bridge["diagnostic"] = {
            **bridge["diagnostic"],
            "status": "unavailable",
            "reason": "lulynx_native_import_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return bridge

    getter = getattr(native, "get_turbocore_training_capabilities", None)
    if getter is None:
        bridge["diagnostic"] = {
            **bridge["diagnostic"],
            "status": "unavailable",
            "reason": "turbocore_training_capability_function_missing",
        }
        return bridge

    try:
        report = getter()
    except Exception as exc:  # pragma: no cover - depends on local native build
        bridge["diagnostic"] = {
            **bridge["diagnostic"],
            "status": "unavailable",
            "reason": "turbocore_training_capability_probe_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return bridge

    if not isinstance(report, dict):
        bridge["diagnostic"] = {
            **bridge["diagnostic"],
            "status": "unavailable",
            "reason": "turbocore_training_capability_report_invalid",
            "raw_type": type(report).__name__,
        }
        return bridge

    native_bridge = dict(report.get("training_bridge") or {})
    native_features = report.get("features") if isinstance(report.get("features"), dict) else {}
    schema_version = report.get("schema_version", 1)
    if validate_workspace_pipeline_native_capabilities is not None:
        try:
            validation = validate_workspace_pipeline_native_capabilities(report)
        except Exception as exc:  # pragma: no cover - defensive reporting only
            bridge["diagnostic"] = {
                **bridge["diagnostic"],
                "status": "unavailable",
                "reason": "turbocore_training_capability_validation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return bridge
        if not bool(validation.get("ok", False)):
            bridge["diagnostic"] = {
                **bridge["diagnostic"],
                "status": "unavailable",
                "reason": "turbocore_training_capability_schema_incomplete",
                "validation": validation,
            }
            return bridge
    native_bridge.setdefault("available", bool(native_bridge.get("available", False)))
    native_bridge.setdefault("status", "capability_stub" if native_bridge["available"] else "unavailable")
    native_bridge.setdefault("reason", "native_training_capability_report")
    native_bridge["features"] = native_features
    native_bridge["schema_version"] = schema_version
    native_bridge["training_path_enabled"] = bool(report.get("training_path_enabled", False))
    native_bridge["diagnostic"] = {
        **bridge["diagnostic"],
        "provider": "native_module",
        "status": "capability_stub",
        "reason": "native_training_capability_report",
    }
    return native_bridge


def build_turbocore_capability_report(
    config: Dict[str, Any],
    *,
    resolution: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    source: str = "turbocore_capabilities",
) -> Dict[str, Any]:
    model_type = str(config.get("model_type", "unknown") or "unknown").strip().lower()
    training_type = str(config.get("training_type", "lora") or "lora").strip().lower()
    requested_core = str(config.get("execution_core", "standard") or "standard").strip().lower()
    requested_features = _listish(config.get("turbocore_features", []))
    disabled_by_request = set(_listish(config.get("turbocore_disable", [])))
    strict = _boolish(config.get("turbocore_strict", False))
    allow_fallback = _boolish(config.get("turbocore_allow_fallback", True), default=True)
    experimental_fp8 = _boolish(config.get("turbocore_experimental_fp8", False))

    resolution = dict(resolution or {})
    route_key = (model_type, training_type)
    route_candidate = route_key in _TRAINING_ROUTE_CANDIDATES

    feature_status: Dict[str, Dict[str, Any]] = {}
    native_training_bridge = probe_native_training_bridge()
    native_features = native_training_bridge.get("features", {})
    if not isinstance(native_features, dict):
        native_features = {}

    for feature in KNOWN_TURBOCORE_FEATURES:
        requested = feature in requested_features or (feature == "experimental_fp8" and experimental_fp8)
        disabled = feature in disabled_by_request
        reason = "turbocore_training_bridge_not_implemented"
        status = "unavailable"
        if disabled:
            reason = "disabled_by_request"
            status = "disabled"
        elif not requested:
            status = "not_requested"
            reason = "not_requested"
        elif feature in native_features:
            native_feature = native_features.get(feature) or {}
            if bool(native_feature.get("available", False)):
                status = str(native_feature.get("status") or "available")
                reason = str(native_feature.get("reason") or "native_feature_available")
            else:
                status = str(native_feature.get("status") or "unavailable")
                reason = _report_reason_for_native_feature(feature, native_feature, native_training_bridge)
        feature_status[feature] = {
            "requested": requested,
            "status": status,
            "reason": reason,
        }

    payload = {
        "source": source,
        "schema_id": config.get("schema_id", ""),
        "model_type": model_type,
        "training_type": training_type,
        "route": {
            "key": f"{model_type}/{training_type}",
            "candidate_for_future_turbocore": route_candidate,
            "status": "candidate" if route_candidate else "unsupported",
            "reason": "native_training_bridge_not_implemented",
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "native_bridge": {
            "launcher_bridge": probe_native_launcher_bridge(),
            "training_bridge": native_training_bridge,
        },
        "requested_execution_core": resolution.get("requested_execution_core", requested_core),
        "effective_execution_core": resolution.get("effective_execution_core", "standard"),
        "strict": bool(resolution.get("turbocore_strict", strict)),
        "allow_fallback": bool(resolution.get("turbocore_allow_fallback", allow_fallback)),
        "requested_features": resolution.get("turbocore_features_requested", requested_features),
        "active_features": resolution.get("turbocore_features_active", []),
        "disabled_features": resolution.get("turbocore_features_disabled", []),
        "feature_status": feature_status,
        "fallback_reason": resolution.get("turbocore_fallback_reason", ""),
        "warnings": resolution.get("turbocore_warnings", []),
        "error": error,
    }

    if build_turbocore_evidence is not None:
        try:
            payload["evidence"] = build_turbocore_evidence(config)
        except Exception as exc:  # pragma: no cover - defensive reporting only
            payload["evidence_error"] = f"{type(exc).__name__}: {exc}"
    if build_turbocore_validation_status is not None:
        try:
            payload["validation_status"] = build_turbocore_validation_status(
                config,
                capability_report=payload,
            )
        except Exception as exc:  # pragma: no cover - defensive reporting only
            payload["validation_status_error"] = f"{type(exc).__name__}: {exc}"
    if validate_workspace_pipeline_native_capabilities is not None:
        try:
            payload["native_abi_validation"] = validate_workspace_pipeline_native_capabilities(native_training_bridge)
        except Exception as exc:  # pragma: no cover - defensive reporting only
            payload["native_abi_validation_error"] = f"{type(exc).__name__}: {exc}"
    return payload
