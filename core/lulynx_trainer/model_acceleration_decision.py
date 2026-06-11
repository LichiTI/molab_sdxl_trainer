"""Decision payload for model-aware acceleration policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model_acceleration_matrix import acceleration_matrix_summary_for


@dataclass
class AccelerationPolicyDecision:
    requested_profile: str
    effective_profile: str
    model_family: str
    schema_id: str = ""
    training_type: str = ""
    recommended_config_patch: dict[str, Any] = field(default_factory=dict)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def add_patch(self, track: str, key: str, value: Any, message: str = "") -> None:
        self.recommended_config_patch[key] = value
        self.tracks.append({"name": track, "status": "recommended", "message": message, "patch": {key: value}})

    def add_skip(self, track: str, key: str, current: Any, reason: str) -> None:
        self.skipped.append({"track": track, "key": key, "current": current, "reason": reason})
        self.tracks.append({"name": track, "status": "preserved", "message": f"{key} preserved: {reason}", "patch": {}})

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "model_family": self.model_family,
            "schema_id": self.schema_id,
            "training_type": self.training_type,
            "recommended_config_patch": dict(self.recommended_config_patch),
            "tracks": [dict(track) for track in self.tracks],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "skipped": list(self.skipped),
            "matrix_summary": acceleration_matrix_summary_for(self.model_family),
        }


__all__ = ["AccelerationPolicyDecision"]
