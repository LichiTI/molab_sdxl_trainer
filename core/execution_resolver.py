"""Training execution resolver.

Validates the full chain: profile → Python → attention backend → capability matrix.
Replaces get_heavy_python() for the training path.

POLYFORM NONCOMMERCIAL -- see LICENSE.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .execution_manifest import (
    AttentionBackendSpec,
    ProfileManifestEntry,
    get_attention_backend_spec,
    get_manifest,
    get_profile_entry,
    is_profile_installed,
    normalize_attention_backend,
    normalize_profile_id,
    resolve_env_dir,
    resolve_python_executable,
)
from .execution_profile import (
    ResolvedExecution,
    ResolutionError,
    TrainingExecutionProfile,
)

logger = logging.getLogger(__name__)

# Map attention backend id → Python module to import-check
_ATTENTION_MODULE_MAP: Dict[str, str] = {
    "flash2": "flash_attn",
    "flash": "flash_attn",
    "sageattn": "sageattention",
    "flexattn": "torch.nn.attention.flex_attention",
    "spargeattn2": "spas_sage_attn",
    "xformers": "xformers",
    "sdpa": "torch.nn.functional",
    "torch": "torch",
}

_DEFAULT_PROFILE_ID = "standard"
_AUTO_PROFILE_IDS = {"", "default", _DEFAULT_PROFILE_ID}
_ATTENTION_PROFILE_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "flash2": ("flash2",),
    "sageattn": ("sageattention2", "sageattention", "sageattention-blackwell"),
    "spargeattn2": ("spargeattn2",),
    "flexattn": ("flexattention",),
}


class TrainingExecutionResolver:
    """Resolves execution profile + attention backend for a training run."""

    def __init__(self, backend_root: Path) -> None:
        self._backend_root = backend_root

    # ------------------------------------------------------------------
    # Profile listing
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[TrainingExecutionProfile]:
        """Return all profiles with disk-state filled in."""
        profiles = []
        for entry in get_manifest():
            profiles.append(self._build_profile(entry))
        return profiles

    def get_profile(self, profile_id: str) -> Optional[TrainingExecutionProfile]:
        entry = get_profile_entry(profile_id)
        if entry is None:
            return None
        return self._build_profile(entry)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        execution_profile_id: str,
        requested_attention: str,
        schema_id: str,
        allow_attention_fallback: bool,
        model_type: str,
        training_type: str,
    ) -> ResolvedExecution:
        """Resolve and validate the full execution chain.

        Raises ``ResolutionError`` on hard failures.
        """
        # 1. Look up profile
        entry = self._resolve_profile_entry(
            execution_profile_id=execution_profile_id,
            requested_attention=requested_attention,
        )
        if entry is None:
            raise ResolutionError(
                f"Unknown execution profile: '{execution_profile_id}'",
                code="profile_not_found",
            )

        requested_probe = self._attention_backend_for_probe(entry, requested_attention)

        # 2. Build profile with disk state.  Resolve only needs the backend this
        # run will actually use; probing every supported backend can take tens
        # of seconds in CUDA runtimes.
        profile = self._build_profile(entry, probe_backends={requested_probe})

        # 3. Check Python + deps installed
        if not profile.installed:
            python_exe = resolve_python_executable(entry, self._backend_root)
            if not python_exe.is_file():
                raise ResolutionError(
                    f"Python not found for profile '{entry.id}': {python_exe}",
                    code="python_missing",
                )
            env_dir = resolve_env_dir(entry, self._backend_root)
            deps_marker = env_dir / ".deps_installed" if env_dir else None
            detail_parts = [
                f"env={env_dir}" if env_dir else "env=not found",
                f"python={python_exe}",
                f"marker={deps_marker}" if deps_marker else "marker=not found",
            ]
            raise ResolutionError(
                f"Dependencies not installed for profile '{entry.id}'. "
                f"Install deps first. ({'; '.join(detail_parts)})",
                code="deps_missing",
            )

        # 4. Resolve attention backend
        resolved_attention, fallback_reason = self._resolve_attention(
            entry=entry,
            requested=requested_attention,
            allow_fallback=allow_attention_fallback,
            model_type=model_type,
            training_type=training_type,
            available=profile.available_attention_backends,
        )

        warnings: List[str] = []
        if fallback_reason:
            warnings.append(fallback_reason)

        return ResolvedExecution(
            profile=profile,
            python_executable=profile.python_executable,
            execution_profile_id=entry.id,
            schema_id=schema_id,
            model_type=model_type,
            training_type=training_type,
            requested_attention=requested_attention,
            resolved_attention=resolved_attention,
            applied_attention=resolved_attention,  # Will be updated by trainer
            allow_attention_fallback=allow_attention_fallback,
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Capability query
    # ------------------------------------------------------------------

    def get_attention_backend_spec(
        self, profile_id: str, attention_backend: str
    ) -> Optional[AttentionBackendSpec]:
        return get_attention_backend_spec(profile_id, attention_backend)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        entry: ProfileManifestEntry,
        probe_backends: Optional[Iterable[str]] = None,
    ) -> TrainingExecutionProfile:
        python_exe = resolve_python_executable(entry, self._backend_root)
        env_dir = resolve_env_dir(entry, self._backend_root)
        installed = is_profile_installed(entry, self._backend_root)
        probe_set = None
        if probe_backends is not None:
            probe_set = {
                normalize_attention_backend(backend)
                for backend in probe_backends
                if normalize_attention_backend(backend)
            }

        # Check which backends are actually importable
        available: List[str] = []
        if installed:
            for backend_id in entry.supported_attention_backends:
                if probe_set is not None and backend_id not in probe_set:
                    continue
                mod = _ATTENTION_MODULE_MAP.get(backend_id, "")
                if not mod:
                    continue
                if self._check_importable(python_exe, mod):
                    available.append(backend_id)

        return TrainingExecutionProfile(
            id=entry.id,
            label_zh=entry.label_zh,
            label_en=entry.label_en,
            python_executable=str(python_exe),
            env_root=str(env_dir),
            installed=installed,
            supported_attention_backends=list(entry.supported_attention_backends),
            available_attention_backends=available,
            default_attention_backend=entry.default_attention_backend,
            env_vars=dict(entry.env_vars),
            experimental=entry.experimental,
            hardware_constraint=entry.hardware_constraint,
        )

    @staticmethod
    def _attention_backend_for_probe(
        entry: ProfileManifestEntry,
        requested_attention: str,
    ) -> str:
        requested = normalize_attention_backend(requested_attention)
        if requested == "auto" or not requested:
            return entry.default_attention_backend
        return requested

    def _resolve_profile_entry(
        self,
        execution_profile_id: str,
        requested_attention: str,
    ) -> Optional[ProfileManifestEntry]:
        """Resolve profile id, allowing legacy standard+dedicated-attn requests.

        Older WebUI payloads may keep ``execution_profile_id=standard`` while
        setting an attention backend that lives in its own runtime profile.
        Treat the default profile as auto-selectable in that case, while
        keeping explicit non-default profiles strict.
        """
        original_profile_id = str(execution_profile_id or "").strip()
        profile_id = normalize_profile_id(original_profile_id)
        if profile_id in {"", "default"}:
            profile_id = _DEFAULT_PROFILE_ID

        requested = normalize_attention_backend(requested_attention)
        if requested in {"", "auto"} or profile_id not in _AUTO_PROFILE_IDS:
            return get_profile_entry(profile_id)

        candidates = [
            entry
            for candidate_id in _ATTENTION_PROFILE_CANDIDATES.get(requested, ())
            if (entry := get_profile_entry(candidate_id)) is not None
            and requested in entry.supported_attention_backends
        ]
        if not candidates:
            return get_profile_entry(profile_id)

        chosen = next(
            (entry for entry in candidates if is_profile_installed(entry, self._backend_root)),
            None,
        )
        if chosen is None:
            return get_profile_entry(profile_id)
        if chosen.id != profile_id:
            display_from = original_profile_id or _DEFAULT_PROFILE_ID
            logger.info(
                "Execution profile '%s' was switched to '%s' for attention backend '%s'.",
                display_from,
                chosen.id,
                requested,
            )
        return chosen

    def _resolve_attention(
        self,
        entry: ProfileManifestEntry,
        requested: str,
        allow_fallback: bool,
        model_type: str,
        training_type: str,
        available: List[str],
    ) -> tuple[str, str]:
        """Resolve the attention backend. Returns (resolved, fallback_reason)."""
        requested = normalize_attention_backend(requested)

        # Normalize "auto" → profile default
        if requested == "auto" or not requested:
            requested = entry.default_attention_backend

        # Check theoretical support
        if requested not in entry.supported_attention_backends:
            if not allow_fallback:
                raise ResolutionError(
                    f"Attention backend '{requested}' is not supported by profile "
                    f"'{entry.id}'. Supported: {', '.join(entry.supported_attention_backends)}",
                    code="attention_not_supported",
                )
            return "sdpa", (
                f"{requested} is not supported by profile '{entry.id}', "
                "falling back to sdpa"
            )

        # Check capability matrix: backend × model_family × training_type.
        # Empty sets mean "not wired" for that dimension, not "all allowed".
        spec = get_attention_backend_spec(entry.id, requested)
        if spec is not None:
            if model_type not in spec.supported_model_families:
                if not allow_fallback:
                    raise ResolutionError(
                        f"Attention backend '{requested}' is not wired for model "
                        f"family '{model_type}' yet. Supported families: "
                        f"{', '.join(spec.supported_model_families) or '(none — not implemented)'}",
                        code="attention_not_wired_for_model",
                    )
                # Fallback to sdpa
                return "sdpa", (
                    f"{requested} not wired for {model_type}, falling back to sdpa"
                )
            if training_type not in spec.supported_trainer_paths:
                if not allow_fallback:
                    raise ResolutionError(
                        f"Attention backend '{requested}' does not support "
                        f"training type '{training_type}' yet.",
                        code="attention_not_wired_for_trainer",
                    )
                return "sdpa", (
                    f"{requested} not wired for {training_type}, falling back to sdpa"
                )

        # Check actual availability (package importable)
        if requested not in available:
            mod = _ATTENTION_MODULE_MAP.get(requested, "")
            if not allow_fallback:
                raise ResolutionError(
                    f"Attention backend '{requested}' requires package '{mod}' "
                    f"which is not installed in profile '{entry.id}'.",
                    code="attention_package_missing",
                )
            return "sdpa", (
                f"{requested} requires '{mod}' which is not installed, "
                f"falling back to sdpa"
            )

        return requested, ""

    @staticmethod
    def _check_importable(python_exe: Path, module_name: str) -> bool:
        """Check if a module is importable in the given Python interpreter."""
        if module_name == "torch.nn.attention.flex_attention":
            script = "from torch.nn.attention.flex_attention import flex_attention; assert callable(flex_attention)"
        elif module_name == "flash_attn":
            script = (
                "import torch\n"
                "assert torch.cuda.is_available()\n"
                "assert not bool(getattr(torch.version, 'hip', None))\n"
                "cap = torch.cuda.get_device_capability(torch.cuda.current_device())\n"
                "assert cap >= (8, 0), cap\n"
                "import flash_attn\n"
                "from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func\n"
                "assert callable(flash_attn_func)\n"
                "assert callable(flash_attn_varlen_func)\n"
            )
        else:
            script = f"import {module_name}"
        try:
            result = subprocess.run(
                [str(python_exe), "-c", script],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_resolver: Optional[TrainingExecutionResolver] = None


def get_execution_resolver() -> TrainingExecutionResolver:
    """Return the global resolver instance."""
    global _resolver
    if _resolver is None:
        backend_root = Path(__file__).resolve().parent.parent
        _resolver = TrainingExecutionResolver(backend_root)
    return _resolver


def reset_execution_resolver() -> None:
    """Reset the singleton (for testing)."""
    global _resolver
    _resolver = None
