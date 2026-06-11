"""Apply model-aware acceleration decisions to normalized config dicts.

This module is intentionally torch-free.  The policy resolver decides which
fields are safe to recommend; this layer only applies those recommendations
when the user explicitly opted into an acceleration profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_acceleration_policy import (
    PROFILE_OFF,
    AccelerationPolicyDecision,
    resolve_model_acceleration_policy,
)
from .model_acceleration_snapshot import build_model_acceleration_cache_safety


@dataclass
class AccelerationPolicyApplication:
    decision: AccelerationPolicyDecision
    config: dict[str, Any]
    applied_config_patch: dict[str, Any]

    @property
    def applied(self) -> bool:
        return bool(self.applied_config_patch)

    def as_dict(self) -> dict[str, Any]:
        payload = self.decision.as_dict()
        payload["cache_safety"] = build_model_acceleration_cache_safety(
            self.config,
            family=self.decision.model_family,
            profile=self.decision.effective_profile,
        )
        payload.update(
            {
                "available": self.decision.effective_profile != PROFILE_OFF,
                "applied": self.applied,
                "applied_config_patch": dict(self.applied_config_patch),
                "applied_config_patch_keys": sorted(self.applied_config_patch),
                "source": "acceleration_profile",
            }
        )
        return payload


def apply_model_acceleration_policy_to_config(
    config: Mapping[str, Any],
    *,
    schema_id: str = "",
    training_type: str = "",
    data_dir: str | Path | None = None,
) -> AccelerationPolicyApplication:
    """Return a copy of ``config`` with opt-in acceleration fields applied."""

    normalized = dict(config or {})
    decision = resolve_model_acceleration_policy(
        normalized,
        schema_id=schema_id,
        training_type=training_type,
        data_dir=data_dir,
    )
    applied_patch: dict[str, Any] = {}
    if decision.effective_profile != PROFILE_OFF:
        for key, value in decision.recommended_config_patch.items():
            normalized[key] = value
            applied_patch[key] = value
        normalized["acceleration_profile"] = decision.effective_profile
        normalized["speed_profile"] = decision.effective_profile

    return AccelerationPolicyApplication(
        decision=decision,
        config=normalized,
        applied_config_patch=applied_patch,
    )


def model_acceleration_preflight_payload(
    config: Mapping[str, Any],
    *,
    schema_id: str = "",
    training_type: str = "",
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the report-only acceleration block used by preflight payloads."""

    return apply_model_acceleration_policy_to_config(
        config,
        schema_id=schema_id,
        training_type=training_type,
        data_dir=data_dir,
    ).as_dict()


__all__ = [
    "AccelerationPolicyApplication",
    "apply_model_acceleration_policy_to_config",
    "model_acceleration_preflight_payload",
]
