"""Static execution profile manifest — single source of truth.

Defines all known training execution profiles with their environment paths,
supported attention backends, and per-backend capability matrix.

Both the training resolver and the launcher install logic read from this
module.  The training API does NOT depend on the launcher process.

POLYFORM NONCOMMERCIAL -- see LICENSE.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Capability sets
# ---------------------------------------------------------------------------

# All model families that the native trainer can route through execution profiles.
ALL_MODEL_FAMILIES = frozenset({"sdxl", "sd15", "anima", "newbie", "flux"})

# All trainer paths (training_type values) that share the same runtime deps.
ALL_TRAINER_PATHS = frozenset({
    "lora",
    "full_finetune",
    "dreambooth",
    "textual_inversion",
    "ip-adapter",
    "controlnet",
    "lllite",
})

# Runtime attention backends are wired through either:
#   - diffusers AttentionProcessor for SDXL/SD1.5 U-Net routes
#   - native DiT attention patcher for Anima/Newbie routes
RUNTIME_ATTENTION_MODEL_FAMILIES = frozenset({"sdxl", "sd15", "anima", "newbie"})
RUNTIME_ATTENTION_TRAINER_PATHS = frozenset({
    "lora",
    "full_finetune",
    "dreambooth",
    "controlnet",
    "lllite",
})

# DiT routes are currently trained through cached LoRA/full-finetune paths.
DIT_ATTENTION_MODEL_FAMILIES = frozenset({"anima", "newbie"})
DIT_ATTENTION_TRAINER_PATHS = frozenset({"lora", "full_finetune"})

# Sparge2 has stricter runtime shape limits, so keep preflight narrower until
# per-layer validation exists for every DiT/U-Net route.
SPARGE2_MODEL_FAMILIES = frozenset({"sdxl", "sd15", "anima", "newbie"})
SPARGE2_TRAINER_PATHS = frozenset({"lora", "full_finetune"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionBackendSpec:
    """Capability spec for a single attention backend within a profile.

    ``supported_model_families`` and ``supported_trainer_paths`` define
    where this backend is actually wired.  Empty frozenset means the
    backend is NOT wired for that dimension — preflight will block it.
    """

    id: str                                    # "flash2", "sageattn", "sdpa", ...
    module_name: str                           # Python module to import-check
    supported_model_families: frozenset[str]   # e.g. {"sdxl", "sd15"}
    supported_trainer_paths: frozenset[str]    # e.g. {"lora", "controlnet"}


@dataclass(frozen=True)
class ProfileManifestEntry:
    """Immutable blueprint for one execution profile."""

    id: str
    label_zh: str
    label_en: str
    env_dir_name: str                          # Directory name under backend/env/
    default_attention_backend: str
    supported_attention_backends: tuple[str, ...]  # Theoretical support
    attention_backends: tuple[AttentionBackendSpec, ...]  # Capability matrix
    env_vars: Dict[str, str] = field(default_factory=dict)
    experimental: bool = False
    hardware_constraint: str = "nvidia"        # "nvidia", "nvidia_frontier", "intel", "amd", "apple"
    python_rel_path: str = "python.exe"
    deps_marker: str = ".deps_installed"


# ---------------------------------------------------------------------------
# Manifest — all known profiles
# ---------------------------------------------------------------------------

_MANIFEST_ENTRIES: List[ProfileManifestEntry] = [
    # -- NVIDIA standard --
    ProfileManifestEntry(
        id="standard",
        label_zh="标准版",
        label_en="Standard",
        env_dir_name="python",
        default_attention_backend="sdpa",
        supported_attention_backends=("sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
    ),
    ProfileManifestEntry(
        id="sageattention",
        label_zh="SageAttention",
        label_en="SageAttention",
        env_dir_name="python-sageattention",
        default_attention_backend="sageattn",
        supported_attention_backends=("sageattn", "sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("sageattn", "sageattention", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        env_vars={"LULYNX_SAGEATTENTION_STARTUP": "1"},
    ),
    ProfileManifestEntry(
        id="sageattention2",
        label_zh="SageAttention 2",
        label_en="SageAttention 2",
        env_dir_name="python-sageattention2",
        default_attention_backend="sageattn",
        supported_attention_backends=("sageattn", "sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("sageattn", "sageattention", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
    ),
    ProfileManifestEntry(
        id="flash2",
        label_zh="FlashAttention 2",
        label_en="FlashAttention 2",
        env_dir_name="python-flashattention",
        default_attention_backend="flash2",
        supported_attention_backends=("flash2", "sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("flash2", "flash_attn", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        env_vars={"LULYNX_FLASH2_STARTUP": "1", "LULYNX_FLASHATTENTION_STARTUP": "1"},
    ),
    ProfileManifestEntry(
        id="spargeattn2",
        label_zh="Sparse GEMM Attention 2 (实验性)",
        label_en="Sparse GEMM Attention 2 (Experimental)",
        env_dir_name="python-spargeattn2",
        default_attention_backend="spargeattn2",
        supported_attention_backends=("spargeattn2", "sdpa", "torch"),
        attention_backends=(
            AttentionBackendSpec("spargeattn2", "spas_sage_attn", SPARGE2_MODEL_FAMILIES, SPARGE2_TRAINER_PATHS),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        experimental=True,
        hardware_constraint="nvidia_frontier",
    ),
    # -- NVIDIA Blackwell --
    ProfileManifestEntry(
        id="blackwell",
        label_zh="Blackwell",
        label_en="Blackwell",
        env_dir_name="python_blackwell",
        default_attention_backend="sdpa",
        supported_attention_backends=("sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        env_vars={"LULYNX_BLACKWELL_STARTUP": "1"},
        hardware_constraint="nvidia_frontier",
    ),
    ProfileManifestEntry(
        id="flexattention",
        label_zh="FlexAttention (Blackwell 实验性)",
        label_en="FlexAttention (Blackwell Experimental)",
        env_dir_name="python-flexattention-blackwell",
        default_attention_backend="sdpa",
        supported_attention_backends=("flexattn", "sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec(
                "flexattn",
                "torch.nn.attention.flex_attention",
                frozenset({"anima", "newbie"}),
                frozenset({"lora", "full_finetune"}),
            ),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        env_vars={
            "LULYNX_BLACKWELL_STARTUP": "1",
            "LULYNX_FLEXATTENTION_STARTUP": "1",
        },
        experimental=True,
        hardware_constraint="nvidia_frontier",
    ),
    ProfileManifestEntry(
        id="sageattention-blackwell",
        label_zh="SageAttention (Blackwell)",
        label_en="SageAttention (Blackwell)",
        env_dir_name="python-sageattention-blackwell",
        default_attention_backend="sageattn",
        supported_attention_backends=("sageattn", "sdpa", "xformers", "torch"),
        attention_backends=(
            AttentionBackendSpec("sageattn", "sageattention", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("xformers", "xformers", RUNTIME_ATTENTION_MODEL_FAMILIES, RUNTIME_ATTENTION_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
    ),
    # -- Intel --
    ProfileManifestEntry(
        id="intel-xpu",
        label_zh="Intel XPU (实验性)",
        label_en="Intel XPU (Experimental)",
        env_dir_name="python_xpu_intel",
        default_attention_backend="sdpa",
        supported_attention_backends=("sdpa", "torch"),
        attention_backends=(
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        experimental=True,
        hardware_constraint="intel",
    ),
    # -- AMD --
    ProfileManifestEntry(
        id="rocm-amd",
        label_zh="ROCm AMD (实验性)",
        label_en="ROCm AMD (Experimental)",
        env_dir_name="python_rocm_amd",
        default_attention_backend="sdpa",
        supported_attention_backends=("sdpa", "torch"),
        attention_backends=(
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        experimental=True,
        hardware_constraint="amd",
    ),
    # -- Apple Silicon --
    ProfileManifestEntry(
        id="apple-mps",
        label_zh="Apple M 芯片 MPS (实验性)",
        label_en="Apple M-series MPS (Experimental)",
        env_dir_name="python_apple_mps",
        default_attention_backend="sdpa",
        supported_attention_backends=("sdpa", "torch"),
        attention_backends=(
            AttentionBackendSpec("sdpa", "torch.nn.functional", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
            AttentionBackendSpec("torch", "torch", ALL_MODEL_FAMILIES, ALL_TRAINER_PATHS),
        ),
        env_vars={
            "LULYNX_APPLE_MPS_STARTUP": "1",
            "LULYNX_MPS_EXPERIMENTAL": "1",
            "LULYNX_STARTUP_ATTENTION_POLICY": "runtime_guarded",
        },
        experimental=True,
        hardware_constraint="apple",
        python_rel_path="bin/python",
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_MANIFEST_BY_ID: Dict[str, ProfileManifestEntry] = {
    e.id: e for e in _MANIFEST_ENTRIES
}

_PROFILE_ID_ALIASES: Dict[str, str] = {
    "flashattention": "flash2",
}

_ATTENTION_BACKEND_ALIASES: Dict[str, str] = {
    "flash": "flash2",
    "flashattn": "flash2",
    "flashattention": "flash2",
    "flashattention2": "flash2",
    "fa2": "flash2",
    "sage": "sageattn",
    "sageattention": "sageattn",
    "sageattention2": "sageattn",
    "sparge": "spargeattn2",
    "spargeattn": "spargeattn2",
    "flex": "flexattn",
    "flexattention": "flexattn",
    "flexattn": "flexattn",
}


def normalize_profile_id(profile_id: str) -> str:
    """Return the canonical execution profile id."""
    profile_id = str(profile_id or "").strip().lower()
    return _PROFILE_ID_ALIASES.get(profile_id, profile_id)


def normalize_attention_backend(attention_backend: str) -> str:
    """Return the canonical attention backend id."""
    attention_backend = str(attention_backend or "").strip().lower()
    return _ATTENTION_BACKEND_ALIASES.get(attention_backend, attention_backend)


def get_manifest() -> List[ProfileManifestEntry]:
    """Return all profile manifest entries."""
    return list(_MANIFEST_ENTRIES)


def get_profile_entry(profile_id: str) -> ProfileManifestEntry | None:
    """Look up a single profile by id.  Returns None if not found."""
    return _MANIFEST_BY_ID.get(normalize_profile_id(profile_id))


def get_attention_backend_spec(
    profile_id: str, attention_backend: str
) -> AttentionBackendSpec | None:
    """Return the capability spec for a specific backend within a profile."""
    entry = _MANIFEST_BY_ID.get(normalize_profile_id(profile_id))
    if entry is None:
        return None
    attention_backend = normalize_attention_backend(attention_backend)
    for spec in entry.attention_backends:
        if spec.id == attention_backend:
            return spec
    return None


def resolve_env_dir(entry: ProfileManifestEntry, backend_root: Path) -> Path:
    """Return the absolute path to the profile's environment directory."""
    return backend_root / "env" / entry.env_dir_name


def resolve_python_executable(entry: ProfileManifestEntry, backend_root: Path) -> Path:
    """Return the absolute path to the profile's Python executable."""
    env_dir = resolve_env_dir(entry, backend_root)
    return env_dir / entry.python_rel_path


def is_profile_installed(entry: ProfileManifestEntry, backend_root: Path) -> bool:
    """Check if the profile's Python environment is installed on disk."""
    python_exe = resolve_python_executable(entry, backend_root)
    if not python_exe.is_file():
        return False
    env_dir = resolve_env_dir(entry, backend_root)
    marker = env_dir / entry.deps_marker
    return marker.is_file()
