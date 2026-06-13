"""Trainer definition registry.

Provides an immutable ``TrainerSpec`` dataclass and a ``TrainerRegistry``
container that maps training-type strings to their definitions.  The
registry is designed to be populated at application startup and queried
at launch time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TrainerSpec:
    """Immutable descriptor for a single trainer variant.

    Attributes
    ----------
    training_type:
        Canonical key (e.g. ``"anima-lora"``).
    script_path:
        Relative path to the training script.
    direct_python:
        If True, the script is launched directly without an accelerate wrapper.
    direct_cli_args:
        Extra CLI arguments appended when ``direct_python`` is True.
    route_kind:
        Structural route kind for contract resolution.
    route_label:
        Human-readable label for display.
    skip_model_validation:
        If True, the preflight stage skips model path validation.
    allow_dataset_config_without_data_dir:
        If True, a dataset config file can substitute for train_data_dir.
    allow_dataset_class_without_data_dir:
        If True, a dataset_class key can substitute for train_data_dir.
    """
    training_type: str
    script_path: str
    direct_python: bool = False
    direct_cli_args: tuple[str, ...] = ()
    route_kind: str | None = None
    route_label: str | None = None
    skip_model_validation: bool = False
    allow_dataset_config_without_data_dir: bool = False
    allow_dataset_class_without_data_dir: bool = False


class TrainerRegistry:
    """In-memory registry mapping training types to ``TrainerSpec`` objects.

    Populate with ``register()`` at startup, then query with ``get()``
    or ``resolve_script()`` at launch time.
    """

    def __init__(self) -> None:
        self._specs: dict[str, TrainerSpec] = {}

    def register(self, spec: TrainerSpec) -> None:
        """Register a trainer spec.  Overwrites any existing entry for the same key."""
        self._specs[spec.training_type] = spec

    def register_many(self, specs: Sequence[TrainerSpec]) -> None:
        """Register multiple specs in bulk."""
        for spec in specs:
            self.register(spec)

    def get(self, training_type: str) -> TrainerSpec | None:
        """Look up a trainer spec by training type (case-insensitive)."""
        return self._specs.get(str(training_type or "").strip().lower())

    def resolve_script(self, training_type: str, fallback: str | None = None) -> str | None:
        """Resolve the script path for a training type, with optional fallback."""
        spec = self.get(training_type)
        return spec.script_path if spec else fallback

    def resolve_script_with_backend(
        self,
        training_type: str,
        backend_overrides: dict[str, str] | None = None,
    ) -> str | None:
        """Resolve script path with optional hardware-backend overrides.

        ``backend_overrides`` maps ``(training_type, backend_name)`` compound
        keys (e.g. ``"anima-lora:rocm-amd"``) to alternative script paths.
        """
        spec = self.get(training_type)
        if spec is None:
            return None
        overrides = backend_overrides or {}
        for key, alt_path in overrides.items():
            parts = key.split(":", 1)
            if len(parts) == 2 and parts[0] == training_type:
                return alt_path
        return spec.script_path

    def is_direct_python(self, training_type: str) -> bool:
        """Return True if the training type uses direct Python launch."""
        spec = self.get(training_type)
        return bool(spec and spec.direct_python)

    def keys(self) -> list[str]:
        """Return all registered training type keys."""
        return list(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, training_type: str) -> bool:
        return str(training_type or "").strip().lower() in self._specs
