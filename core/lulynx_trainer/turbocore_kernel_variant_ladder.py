# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Deterministic variant ladder for turbocore CUDA kernel autotuning.

Borrowed *idea* from ref/autokernel-main (MIT): a fixed evaluation harness
consumes candidate kernels from an experiment source, keeps improvements and
reverts regressions. AutoKernel's source is an LLM coding agent; phase 1 here
is a deterministic ladder of hand-written cleanroom variants x launch-config
sweep. The :class:`ExperimentSource` protocol is the reserved seam where a
local-LLM agent source can plug in later (launcher resource center can host
the model download).

HARD CONSTRAINT for every variant: it must export the same ``extern "C"``
kernel name and parameter signature as the in-tree v0 source —
``cuModuleGetFunction`` resolves the function by that exact name, and the
Rust launch path passes a fixed 11-argument list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Protocol

ADAMW_KERNEL_NAME = "adamw_flat_fp32_cuda_v0"
ADAMW_SOURCE_FILENAME = "adamw_flat_fp32_cuda_v0.cu"

BLOCK_SIZE_SWEEP = (64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class Candidate:
    """One experiment: a kernel source + launch configuration."""

    label: str
    tier: str
    source_text: str
    block_size: int
    elements_per_thread: int = 1


@dataclass
class ExperimentResult:
    label: str
    tier: str
    block_size: int
    elements_per_thread: int
    status: str  # kept | reverted | crashed
    gbps: float = 0.0
    ms_per_step: float = 0.0
    parity_ok: bool = False
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "tier": self.tier,
            "block_size": self.block_size,
            "elements_per_thread": self.elements_per_thread,
            "status": self.status,
            "gbps": self.gbps,
            "ms_per_step": self.ms_per_step,
            "parity_ok": self.parity_ok,
            "error": self.error,
        }


class ExperimentSource(Protocol):
    """Where candidates come from.

    Phase 1: :class:`VariantLadderSource` (deterministic enumeration).
    Phase 2 (reserved): a local-LLM agent source implementing the same
    protocol — ``propose`` reads the history and emits a rewritten kernel,
    ``notify`` feeds the bench verdict back so the agent can iterate.
    """

    def propose(self) -> Optional[Candidate]:
        """Return the next candidate, or None when exhausted."""
        ...

    def notify(self, result: ExperimentResult) -> None:
        """Feed back the outcome of the last proposed candidate."""
        ...


# ---------------------------------------------------------------------------
# AdamW flat fp32 variants (cleanroom, signature-identical to v0)
# ---------------------------------------------------------------------------

_ADAMW_SIGNATURE = """extern "C" __global__ void adamw_flat_fp32_cuda_v0(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    int numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step_index
)"""

# Per-element update body shared by all variants. Bit-identical to v0 math:
# same operations in the same order, fp32 throughout, std powf/sqrtf.
_ADAMW_ELEMENT_BODY = """
        float g = grad[i];
        float m = exp_avg[i] * beta1 + g * (1.0f - beta1);
        float v = exp_avg_sq[i] * beta2 + g * g * (1.0f - beta2);
        exp_avg[i] = m;
        exp_avg_sq[i] = v;
        float p = param[i];
        if (weight_decay != 0.0f) {
            p *= 1.0f - lr * weight_decay;
        }
        float denom = sqrtf(v) / sqrtf(bias_correction2) + eps;
        param[i] = p - step_size * m / denom;
"""

_ADAMW_PROLOGUE = """    int step = step_index + 1;
    float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(step));
    float step_size = lr / bias_correction1;
"""


def _adamw_restrict_source() -> str:
    """Tier 1: __restrict__ + hoisted uniform prologue, one element/thread."""
    return f"""// tuned variant: restrict (autotune ladder tier 1)
// constraint: same extern "C" name/signature as in-tree v0
{_ADAMW_SIGNATURE} {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {{
        return;
    }}
{_ADAMW_PROLOGUE}
    {{
{_ADAMW_ELEMENT_BODY}
    }}
}}
"""


def _adamw_strided_source(elements_per_thread: int) -> str:
    """Tier 2/3: each thread handles N consecutive elements.

    Consecutive-per-thread layout lets the compiler coalesce the four
    streams into wider transactions (LDG.128 when N>=4 and aligned).
    Grid sizing must use elements_per_thread={elements_per_thread} so the
    launch covers numel exactly (Rust launch config knob).
    """
    ept = int(elements_per_thread)
    return f"""// tuned variant: strided x{ept} (autotune ladder)
// constraint: same extern "C" name/signature as in-tree v0
// launch contract: grid = ceil(numel / (block_size * {ept}))
{_ADAMW_SIGNATURE} {{
    long long base = (long long)(blockIdx.x * blockDim.x + threadIdx.x) * {ept};
    if (base >= numel) {{
        return;
    }}
{_ADAMW_PROLOGUE}
    long long limit = base + {ept} < numel ? base + {ept} : (long long)numel;
    #pragma unroll
    for (long long i = base; i < limit; ++i) {{
{_ADAMW_ELEMENT_BODY}
    }}
}}
"""


def adamw_variant_sources() -> List[tuple]:
    """(label, tier, source_text, elements_per_thread) ladder, cheap→deep."""
    return [
        ("restrict", "restrict", _adamw_restrict_source(), 1),
        ("strided_x2", "strided", _adamw_strided_source(2), 2),
        ("strided_x4", "strided", _adamw_strided_source(4), 4),
        ("strided_x8", "strided", _adamw_strided_source(8), 8),
    ]


class VariantLadderSource:
    """Phase-1 deterministic ExperimentSource: variant ladder x block sweep.

    Ordering: for each variant, sweep all block sizes. ``notify`` is a no-op
    (the ladder is exhaustive, not adaptive) — an LLM agent source would use
    it to steer the next rewrite.
    """

    def __init__(
        self,
        *,
        block_sizes: tuple = BLOCK_SIZE_SWEEP,
        variants: Optional[List[tuple]] = None,
    ) -> None:
        self._results: List[ExperimentResult] = []
        self._iter = self._enumerate(
            variants if variants is not None else adamw_variant_sources(),
            block_sizes,
        )

    @staticmethod
    def _enumerate(variants: List[tuple], block_sizes: tuple) -> Iterator[Candidate]:
        for label, tier, source_text, ept in variants:
            for block in block_sizes:
                yield Candidate(
                    label=f"{label}@b{block}",
                    tier=tier,
                    source_text=source_text,
                    block_size=int(block),
                    elements_per_thread=int(ept),
                )

    def propose(self) -> Optional[Candidate]:
        return next(self._iter, None)

    def notify(self, result: ExperimentResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> List[ExperimentResult]:
        return list(self._results)


__all__ = [
    "ADAMW_KERNEL_NAME",
    "ADAMW_SOURCE_FILENAME",
    "BLOCK_SIZE_SWEEP",
    "Candidate",
    "ExperimentResult",
    "ExperimentSource",
    "VariantLadderSource",
    "adamw_variant_sources",
]
