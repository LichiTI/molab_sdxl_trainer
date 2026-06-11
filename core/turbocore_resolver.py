"""TurboCore execution resolver.

Resolves requested execution core and requested TurboCore features into the
currently effective execution mode.  This is intentionally conservative:
until native TurboCore subsystems are implemented, all requests downgrade to
StandardCore unless the caller explicitly disallows fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_VALID_EXECUTION_CORES = {"standard", "turbo", "auto"}
_KNOWN_TURBOCORE_FEATURES = {
    "lora_fused",
    "native_optimizer",
    "static_route_step",
    "data_pipeline",
    "workspace_pool",
    "experimental_fp8",
}


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


def _normalize_execution_core(value: Any) -> str:
    normalized = str(value or "standard").strip().lower()
    return normalized or "standard"


def _normalize_feature_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.split(",")]
        return list(dict.fromkeys(item for item in items if item))
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = [
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        ]
        return list(dict.fromkeys(normalized))
    normalized = str(value).strip().lower()
    return [normalized] if normalized else []


@dataclass
class ResolvedTurboCoreExecution:
    requested_execution_core: str
    effective_execution_core: str
    requested_features: List[str] = field(default_factory=list)
    disabled_by_request: List[str] = field(default_factory=list)
    active_features: List[str] = field(default_factory=list)
    disabled_features: List[Dict[str, str]] = field(default_factory=list)
    allow_fallback: bool = True
    strict: bool = False
    experimental_fp8_requested: bool = False
    fallback_reason: str = ""
    warnings: List[str] = field(default_factory=list)


class TurboCoreResolutionError(Exception):
    """Raised when TurboCore resolution fails before launch."""

    def __init__(self, message: str, code: str = "turbocore_resolution_failed"):
        super().__init__(message)
        self.code = code


class TurboCoreExecutionResolver:
    """Resolve requested execution core into the currently effective mode."""

    def resolve(
        self,
        *,
        requested_execution_core: Any,
        requested_features: Any,
        disabled_by_request: Any,
        allow_fallback: Any,
        strict: Any,
        experimental_fp8_requested: Any,
        model_type: str,
        training_type: str,
    ) -> ResolvedTurboCoreExecution:
        requested_core = _normalize_execution_core(requested_execution_core)
        if requested_core not in _VALID_EXECUTION_CORES:
            raise TurboCoreResolutionError(
                f"Unknown execution_core: '{requested_core}'. "
                f"Supported values: {', '.join(sorted(_VALID_EXECUTION_CORES))}",
                code="unknown_execution_core",
            )

        requested_feature_list = _normalize_feature_list(requested_features)
        disabled_feature_list = _normalize_feature_list(disabled_by_request)
        allow_fallback_flag = _boolish(allow_fallback, default=True)
        strict_flag = _boolish(strict, default=False)
        fp8_requested = _boolish(experimental_fp8_requested, default=False)

        unknown_requested = [
            feature for feature in requested_feature_list
            if feature not in _KNOWN_TURBOCORE_FEATURES
        ]
        if unknown_requested:
            raise TurboCoreResolutionError(
                f"Unknown TurboCore feature(s): {', '.join(unknown_requested)}",
                code="unknown_turbocore_feature",
            )

        unknown_disabled = [
            feature for feature in disabled_feature_list
            if feature not in _KNOWN_TURBOCORE_FEATURES
        ]
        warnings: List[str] = []
        if unknown_disabled:
            warnings.append(
                "Ignoring unknown TurboCore disabled feature(s): "
                + ", ".join(unknown_disabled)
            )
            disabled_feature_list = [
                feature for feature in disabled_feature_list
                if feature in _KNOWN_TURBOCORE_FEATURES
            ]

        disabled_features: List[Dict[str, str]] = []
        active_features: List[str] = []
        fallback_reason = ""

        # Skeleton phase: TurboCore features are not implemented yet, so any
        # non-standard request downgrades to StandardCore unless fallback is
        # explicitly forbidden.
        if requested_core == "standard":
            effective_core = "standard"
            if requested_feature_list or fp8_requested:
                warnings.append(
                    "TurboCore options were requested while execution_core=standard; "
                    "keeping StandardCore and ignoring TurboCore requests."
                )
                for feature in requested_feature_list:
                    disabled_features.append(
                        {"feature": feature, "reason": "execution_core_standard"}
                    )
                if fp8_requested:
                    disabled_features.append(
                        {"feature": "experimental_fp8", "reason": "execution_core_standard"}
                    )
        else:
            fallback_reason = (
                "TurboCore was requested for "
                f"{model_type}/{training_type}, but no TurboCore runtime features "
                "are implemented yet; falling back to StandardCore."
            )
            if strict_flag or not allow_fallback_flag:
                raise TurboCoreResolutionError(
                    fallback_reason,
                    code="turbocore_unavailable",
                )
            effective_core = "standard"
            warnings.append(fallback_reason)
            for feature in requested_feature_list:
                if feature in disabled_feature_list:
                    disabled_features.append(
                        {"feature": feature, "reason": "disabled_by_request"}
                    )
                else:
                    disabled_features.append(
                        {"feature": feature, "reason": "turbocore_not_implemented"}
                    )
            if fp8_requested:
                disabled_features.append(
                    {"feature": "experimental_fp8", "reason": "turbocore_not_implemented"}
                )

        # Explicit user disables always win, even in the future when features
        # start to activate.
        for feature in disabled_feature_list:
            if feature in active_features:
                active_features.remove(feature)
            if all(item.get("feature") != feature for item in disabled_features):
                disabled_features.append(
                    {"feature": feature, "reason": "disabled_by_request"}
                )

        return ResolvedTurboCoreExecution(
            requested_execution_core=requested_core,
            effective_execution_core=effective_core,
            requested_features=requested_feature_list,
            disabled_by_request=disabled_feature_list,
            active_features=active_features,
            disabled_features=disabled_features,
            allow_fallback=allow_fallback_flag,
            strict=strict_flag,
            experimental_fp8_requested=fp8_requested,
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    def resolve_from_config(
        self,
        config: Dict[str, Any],
        *,
        model_type: str,
        training_type: str,
    ) -> ResolvedTurboCoreExecution:
        return self.resolve(
            requested_execution_core=config.get("execution_core", "standard"),
            requested_features=config.get("turbocore_features", []),
            disabled_by_request=config.get("turbocore_disable", []),
            allow_fallback=config.get("turbocore_allow_fallback", True),
            strict=config.get("turbocore_strict", False),
            experimental_fp8_requested=config.get("turbocore_experimental_fp8", False),
            model_type=model_type,
            training_type=training_type,
        )


_resolver: Optional[TurboCoreExecutionResolver] = None


def get_turbocore_resolver() -> TurboCoreExecutionResolver:
    global _resolver
    if _resolver is None:
        _resolver = TurboCoreExecutionResolver()
    return _resolver


def reset_turbocore_resolver() -> None:
    global _resolver
    _resolver = None
