"""Deterministic, GPU-aware runtime recommendation engine.

Pure logic — no I/O, no randomness.  Given a GPU profile and a runtime
registry, produces a sorted list of recommendations with bilingual
explanations.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts.gpu import GpuInfo
from .contracts.runtime import CompatibilityRule, CompatibilityStatus, RuntimeCategory, RuntimeDef
from .registry import RuntimeRegistry

# ── VRAM thresholds (MB) ──────────────────────────────────────
_VRAM_LOW = 4096   # <4 GB  →  downgrade one level
_VRAM_HIGH = 8192  # ≥8 GB  →  no penalty

# ── result container ───────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeRecommendation:
    """A single scored recommendation."""

    runtime: RuntimeDef
    compatibility: CompatibilityRule


# ── category → vendor mapping ─────────────────────────────────

_CAT_VENDOR: dict[RuntimeCategory, str] = {
    RuntimeCategory.NVIDIA: "nvidia",
    RuntimeCategory.NVIDIA_FRONTIER: "nvidia",
    RuntimeCategory.INTEL: "intel",
    RuntimeCategory.AMD: "amd",
    RuntimeCategory.CPU: "cpu",
}

# ── status severity ordering (for downgrade) ──────────────────

_STATUS_ORDER: list[CompatibilityStatus] = [
    CompatibilityStatus.RECOMMENDED,
    CompatibilityStatus.SUPPORTED,
    CompatibilityStatus.CAUTION,
    CompatibilityStatus.NOT_RECOMMENDED,
]


def _downgrade(status: CompatibilityStatus) -> CompatibilityStatus:
    idx = _STATUS_ORDER.index(status)
    return _STATUS_ORDER[min(idx + 1, len(_STATUS_ORDER) - 1)]


# ── bilingual reason templates ────────────────────────────────

_MATCH_REASONS: dict[RuntimeCategory, tuple[str, str]] = {
    RuntimeCategory.NVIDIA: (
        "Matches detected NVIDIA GPU.",
        "匹配检测到的 NVIDIA GPU。",
    ),
    RuntimeCategory.NVIDIA_FRONTIER: (
        "Matches detected NVIDIA GPU (frontier features).",
        "匹配检测到的 NVIDIA GPU（前沿特性）。",
    ),
    RuntimeCategory.INTEL: (
        "Matches detected Intel GPU.",
        "匹配检测到的 Intel GPU。",
    ),
    RuntimeCategory.AMD: (
        "Matches detected AMD GPU.",
        "匹配检测到的 AMD GPU。",
    ),
    RuntimeCategory.CPU: (
        "CPU-only fallback — no compatible GPU acceleration.",
        "纯 CPU 回退 — 无兼容 GPU 加速。",
    ),
}

_MISMATCH_REASON: tuple[str, str] = (
    "Runtime category does not match detected GPU vendor.",
    "运行时类别与检测到的 GPU 厂商不匹配。",
)

_LOW_VRAM_REASON: tuple[str, str] = (
    "VRAM below 4 GB; performance may be limited.",
    "显存低于 4 GB；性能可能受限。",
)

_EXPERIMENTAL_REASON: tuple[str, str] = (
    "Runtime is experimental.",
    "该运行时为实验性。",
)


# ── core logic ────────────────────────────────────────────────


def _classify(
    rd: RuntimeDef,
    gpu_vendor: str,
    vram_mb: int,
) -> CompatibilityRule:
    """Return a :class:`CompatibilityRule` for one runtime against one GPU."""
    cat_vendor = _CAT_VENDOR[rd.category]

    # base verdict
    if gpu_vendor == "unknown" and rd.category == RuntimeCategory.CPU:
        base = CompatibilityStatus.RECOMMENDED
    elif cat_vendor == gpu_vendor or rd.category == RuntimeCategory.CPU:
        base = CompatibilityStatus.SUPPORTED
    else:
        base = CompatibilityStatus.NOT_RECOMMENDED

    # upgrade NVIDIA to RECOMMENDED when vendor matches
    if cat_vendor == gpu_vendor and rd.category in (
        RuntimeCategory.NVIDIA,
        RuntimeCategory.NVIDIA_FRONTIER,
        RuntimeCategory.INTEL,
        RuntimeCategory.AMD,
    ):
        base = CompatibilityStatus.RECOMMENDED

    # pick reason
    if cat_vendor == gpu_vendor or (gpu_vendor == "unknown" and rd.category == RuntimeCategory.CPU):
        reason_en, reason_zh = _MATCH_REASONS[rd.category]
    else:
        reason_en, reason_zh = _MISMATCH_REASON

    # VRAM penalty
    if vram_mb > 0 and vram_mb < _VRAM_LOW and base != CompatibilityStatus.NOT_RECOMMENDED:
        base = _downgrade(base)
        reason_en += " " + _LOW_VRAM_REASON[0]
        reason_zh += " " + _LOW_VRAM_REASON[1]

    # experimental penalty
    if rd.experimental and base != CompatibilityStatus.NOT_RECOMMENDED:
        base = _downgrade(base)
        reason_en += " " + _EXPERIMENTAL_REASON[0]
        reason_zh += " " + _EXPERIMENTAL_REASON[1]

    return CompatibilityRule(status=base, reason_en=reason_en, reason_zh=reason_zh)


# ── public API ────────────────────────────────────────────────


def recommend(
    gpu: GpuInfo | None,
    registry: RuntimeRegistry,
) -> list[RuntimeRecommendation]:
    """Return all runtimes sorted best-first by compatibility.

    *gpu* may be ``None`` when no GPU was detected; in that case only
    CPU-category runtimes are recommended.
    """
    gpu_vendor = gpu.vendor.lower() if gpu else "unknown"
    vram = gpu.vram_mb if gpu else 0

    recs: list[RuntimeRecommendation] = []
    for rd in registry.all():
        rule = _classify(rd, gpu_vendor, vram)
        recs.append(RuntimeRecommendation(runtime=rd, compatibility=rule))

    recs.sort(key=_sort_key)
    return recs


def best(gpu: GpuInfo | None, registry: RuntimeRegistry) -> RuntimeRecommendation | None:
    """Return the single best recommendation, or ``None`` if registry empty."""
    recs = recommend(gpu, registry)
    return recs[0] if recs else None


# ── sorting helpers ───────────────────────────────────────────

def _sort_key(rec: RuntimeRecommendation) -> tuple[int, int]:
    """Lower tuple = better recommendation."""
    status_rank = _STATUS_ORDER.index(rec.compatibility.status)
    experimental_penalty = 1 if rec.runtime.experimental else 0
    return (status_rank, experimental_penalty)
