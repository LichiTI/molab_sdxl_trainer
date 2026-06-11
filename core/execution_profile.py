"""Training execution profile — runtime representation.

Data classes for profiles as resolved on disk (with actual Python path,
installed status, and available attention backends).

POLYFORM NONCOMMERCIAL -- see LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingExecutionProfile:
    """A resolved execution profile — manifest entry + disk state."""

    id: str
    label_zh: str
    label_en: str
    python_executable: str                     # Absolute path
    env_root: str                              # Absolute path to env dir
    installed: bool                            # Python exists + deps marker
    supported_attention_backends: List[str]    # Theoretical (from manifest)
    available_attention_backends: List[str]    # Actually importable right now
    default_attention_backend: str
    env_vars: dict[str, str]
    experimental: bool = False
    hardware_constraint: str = ""


@dataclass
class ResolvedExecution:
    """Full resolution record for a single training run."""

    profile: TrainingExecutionProfile
    python_executable: str
    execution_profile_id: str
    schema_id: str
    model_type: str
    training_type: str
    # 3-layer attention state
    requested_attention: str       # What user chose
    resolved_attention: str        # After auto/default + availability check
    applied_attention: str         # Filled by trainer at runtime (starts as resolved)
    allow_attention_fallback: bool
    fallback_reason: str           # "" if no fallback
    warnings: List[str] = field(default_factory=list)


class ResolutionError(Exception):
    """Raised when execution profile resolution fails."""

    def __init__(self, message: str, code: str = "resolution_failed"):
        super().__init__(message)
        self.code = code
