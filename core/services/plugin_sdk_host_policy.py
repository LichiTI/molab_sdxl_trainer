"""Host-level safety defaults for Plugin SDK runner execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PluginSdkHostPolicy:
    """Runtime-owned policy injected before plugin SDK runner execution.

    Plugin manifests are plugin-controlled input, so public/release defaults
    must come from the host. Developer mode keeps compatibility for local SDK
    authoring, while normal mode isolates unsigned/untrusted runners by default.
    """

    developer_mode: bool = False
    default_subprocess_for_untrusted: bool = True
    force_subprocess_for_untrusted: bool = True
    enforce_subprocess_for_elevated: bool = True
    source: str = "host.normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "plugin-sdk-host-policy-v1",
            "developer_mode": self.developer_mode,
            "default_subprocess_for_untrusted": self.default_subprocess_for_untrusted,
            "force_subprocess_for_untrusted": self.force_subprocess_for_untrusted,
            "enforce_subprocess_for_elevated": self.enforce_subprocess_for_elevated,
            "source": self.source,
        }

    def context_metadata(self) -> dict[str, Any]:
        return {
            "plugin_sdk_host_policy": self.to_dict(),
            "plugin_sdk_default_subprocess_for_untrusted": self.default_subprocess_for_untrusted,
            "plugin_sdk_force_subprocess_for_untrusted": self.force_subprocess_for_untrusted,
            "plugin_sdk_enforce_subprocess_for_elevated": self.enforce_subprocess_for_elevated,
        }


def build_plugin_sdk_host_policy(*, developer_mode: bool = False) -> PluginSdkHostPolicy:
    """Return the default host policy for the current runtime mode."""

    if developer_mode:
        return PluginSdkHostPolicy(
            developer_mode=True,
            default_subprocess_for_untrusted=_env_bool("LULYNX_PLUGIN_SDK_DEV_DEFAULT_SUBPROCESS", False),
            force_subprocess_for_untrusted=_env_bool("LULYNX_PLUGIN_SDK_DEV_FORCE_SUBPROCESS", False),
            enforce_subprocess_for_elevated=_env_bool("LULYNX_PLUGIN_SDK_DEV_ENFORCE_ELEVATED", False),
            source="host.developer",
        )
    return PluginSdkHostPolicy(
        developer_mode=False,
        default_subprocess_for_untrusted=_env_bool("LULYNX_PLUGIN_SDK_DEFAULT_SUBPROCESS", True),
        force_subprocess_for_untrusted=_env_bool("LULYNX_PLUGIN_SDK_FORCE_UNTRUSTED_SUBPROCESS", True),
        enforce_subprocess_for_elevated=_env_bool("LULYNX_PLUGIN_SDK_ENFORCE_ELEVATED_SUBPROCESS", True),
        source="host.normal",
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default
