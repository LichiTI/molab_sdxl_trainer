"""Sandbox policy resolution for plugin SDK runner execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from backend.core.contracts import PluginRunnerRegistration
from backend.core.services.plugin_execution_guard import elevated_runner_permissions


_MODE_ALIASES = {
    "inline": "in_process",
    "in-process": "in_process",
    "in_process": "in_process",
    "process": "subprocess",
    "subprocess": "subprocess",
    "isolated": "subprocess",
}

_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_TIMEOUT_SECONDS = 3600
_SENSITIVE_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "API_KEY",
    "ACCESS_KEY",
    "AUTH_TOKEN",
    "SESSION_KEY",
    "CREDENTIAL",
)


@dataclass(frozen=True)
class PluginSdkSandboxPolicy:
    """Resolved execution policy for one plugin SDK runner invocation."""

    execution_mode: str = "in_process"
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    env_allowlist: tuple[str, ...] = ()
    blocked_env_allowlist: tuple[str, ...] = ()
    requested_by: str = "default"
    elevated_permissions: tuple[str, ...] = ()
    isolation_warning: str = ""
    enforce_subprocess_for_elevated: bool = False
    default_subprocess_for_untrusted: bool = False
    force_subprocess_for_untrusted: bool = False
    trust_state: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "plugin-sdk-sandbox-policy-v1",
            "execution_mode": self.execution_mode,
            "timeout_seconds": self.timeout_seconds,
            "env_allowlist": list(self.env_allowlist),
            "blocked_env_allowlist": list(self.blocked_env_allowlist),
            "requested_by": self.requested_by,
            "elevated_permissions": list(self.elevated_permissions),
            "isolation_warning": self.isolation_warning,
            "enforce_subprocess_for_elevated": self.enforce_subprocess_for_elevated,
            "default_subprocess_for_untrusted": self.default_subprocess_for_untrusted,
            "force_subprocess_for_untrusted": self.force_subprocess_for_untrusted,
            "trust_state": self.trust_state,
        }


def build_plugin_sdk_sandbox_policy(
    registration: PluginRunnerRegistration,
    context_metadata: Mapping[str, Any] | None = None,
    approval_snapshot: Mapping[str, Any] | None = None,
) -> PluginSdkSandboxPolicy:
    """Resolve runner sandbox policy without executing plugin code.

    Existing runners remain in-process by default. A plugin manifest or host
    context can explicitly request subprocess isolation; future policy can move
    selected elevated permissions to subprocess by default without changing the
    executor surface.
    """

    metadata = dict(context_metadata or {})
    extra = dict(getattr(registration, "model_extra", None) or {})
    mode_source = "default"
    raw_mode = _first_non_empty(metadata.get("plugin_sdk_execution_mode"), metadata.get("plugin_execution_mode"))
    if raw_mode:
        mode_source = "context"
    if raw_mode is None:
        raw_mode = _first_non_empty(extra.get("execution_mode"), getattr(registration, "execution_mode", ""))
        if raw_mode:
            mode_source = "runner"
    if bool(metadata.get("force_plugin_subprocess")):
        raw_mode = "subprocess"
        mode_source = "context.force_plugin_subprocess"
    explicit_mode = raw_mode is not None
    mode = _normalize_mode(raw_mode) or "in_process"

    timeout_seconds = _coerce_timeout(
        _first_non_empty(
            metadata.get("plugin_sdk_timeout_seconds"),
            metadata.get("plugin_timeout_seconds"),
            extra.get("timeout_seconds"),
            getattr(registration, "timeout_seconds", None),
        )
    )
    env_allowlist, blocked_env_allowlist = _normalize_env_allowlist(
        extra.get("env_allowlist"),
        getattr(registration, "env_allowlist", None),
        metadata.get("plugin_sdk_env_allowlist"),
    )
    if mode == "subprocess" and mode_source == "default":
        mode_source = "runner"
    if approval_snapshot and approval_snapshot.get("permission_source") == "approval-store" and mode == "subprocess":
        mode_source = mode_source or "approval-store"
    elevated_permissions = _elevated_permissions(registration, approval_snapshot)
    enforce_subprocess_for_elevated = bool(
        metadata.get("plugin_sdk_enforce_subprocess_for_elevated")
        or metadata.get("enforce_plugin_subprocess_for_elevated")
    )
    default_subprocess_for_untrusted = bool(
        metadata.get("plugin_sdk_default_subprocess_for_untrusted")
        or metadata.get("default_plugin_subprocess_for_untrusted")
    )
    force_subprocess_for_untrusted = bool(
        metadata.get("plugin_sdk_force_subprocess_for_untrusted")
        or metadata.get("force_plugin_subprocess_for_untrusted")
    )
    trust_state = _trust_state(approval_snapshot)
    if force_subprocess_for_untrusted and trust_state != "trusted":
        mode = "subprocess"
        mode_source = "host.force_untrusted_subprocess"
    elif default_subprocess_for_untrusted and not explicit_mode and trust_state != "trusted":
        mode = "subprocess"
        mode_source = "context.default_untrusted_subprocess"
    isolation_warning = ""
    if elevated_permissions and mode != "subprocess":
        isolation_warning = "elevated_permissions_in_process"
    return PluginSdkSandboxPolicy(
        execution_mode=mode,
        timeout_seconds=timeout_seconds,
        env_allowlist=env_allowlist,
        blocked_env_allowlist=blocked_env_allowlist,
        requested_by=mode_source or "default",
        elevated_permissions=tuple(elevated_permissions),
        isolation_warning=isolation_warning,
        enforce_subprocess_for_elevated=enforce_subprocess_for_elevated,
        default_subprocess_for_untrusted=default_subprocess_for_untrusted,
        force_subprocess_for_untrusted=force_subprocess_for_untrusted,
        trust_state=trust_state,
    )


def _elevated_permissions(
    registration: PluginRunnerRegistration,
    approval_snapshot: Mapping[str, Any] | None,
) -> list[str]:
    if approval_snapshot:
        values = approval_snapshot.get("elevated_permissions") or []
        if values:
            return sorted(str(item) for item in values if str(item).strip())
    return elevated_runner_permissions(list(registration.permissions or []))


def _trust_state(approval_snapshot: Mapping[str, Any] | None) -> str:
    if not approval_snapshot:
        return "unknown"
    source = str(approval_snapshot.get("permission_source") or "").strip()
    approved = bool(approval_snapshot.get("approved"))
    signer = str(approval_snapshot.get("signer") or "").strip()
    if source == "approval-store" and approved and signer:
        return "trusted"
    if source == "approval-store" and approved:
        return "approved_unsigned"
    if source in {"no-runner-permissions", "explicit-context"}:
        return source
    return source or "unknown"


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return _MODE_ALIASES.get(text, "")


def _coerce_timeout(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_TIMEOUT_SECONDS
    if parsed < 1:
        return 1
    if parsed > _MAX_TIMEOUT_SECONDS:
        return _MAX_TIMEOUT_SECONDS
    return parsed


def _normalize_env_allowlist(*values: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    result: list[str] = []
    blocked: list[str] = []
    for value in values:
        items: list[Any]
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            if is_sensitive_env_key(text):
                if text not in blocked:
                    blocked.append(text)
                continue
            if text not in result:
                result.append(text)
    return tuple(result), tuple(blocked)


def is_sensitive_env_key(key: object) -> bool:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(key or "").strip().upper())
    if not normalized:
        return False
    return any(marker in normalized for marker in _SENSITIVE_ENV_MARKERS)
