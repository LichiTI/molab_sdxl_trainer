"""Decision gate for experimental compile paths.

The first implementation is deliberately measurement-oriented and side-effect
free: callers can feed eager/compiled timings and VRAM peaks into the gate and
get a clear keep/disable decision.  Route code can later attach real warmup
probes without changing the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CompileProbeResult:
    route: str
    target: str
    decision: str
    reason: str
    eager_seconds: float = 0.0
    compiled_seconds: float = 0.0
    eager_peak_vram: float = 0.0
    compiled_peak_vram: float = 0.0
    speedup_ratio: float = 0.0
    vram_increase_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def keep(self) -> bool:
        return self.decision == "keep"

    def log_lines(self) -> Iterable[str]:
        yield (
            "[compile-probe] "
            f"route={self.route} target={self.target} decision={self.decision} "
            f"reason=\"{self.reason}\" eager={self.eager_seconds:.4f}s "
            f"compiled={self.compiled_seconds:.4f}s speedup={self.speedup_ratio * 100.0:.2f}% "
            f"eager_vram={self.eager_peak_vram / (1024.0 * 1024.0):.1f}MB "
            f"compiled_vram={self.compiled_peak_vram / (1024.0 * 1024.0):.1f}MB "
            f"vram_delta={self.vram_increase_ratio * 100.0:+.2f}%"
        )
        for warning in self.warnings:
            yield f"[compile-probe][warn] {warning}"


def evaluate_compile_probe(
    *,
    route: str,
    target: str,
    eager_seconds: float,
    compiled_seconds: float,
    eager_peak_vram: float = 0.0,
    compiled_peak_vram: float = 0.0,
    min_speedup_ratio: float = 0.03,
    max_vram_increase_ratio: float = 0.15,
    failed_reason: str = "",
) -> CompileProbeResult:
    route_name = str(route or "unknown").strip().lower()
    target_name = str(target or "unknown")
    warnings: list[str] = []

    if failed_reason:
        return CompileProbeResult(
            route=route_name,
            target=target_name,
            decision="disable",
            reason=failed_reason,
            eager_seconds=max(float(eager_seconds or 0.0), 0.0),
            compiled_seconds=max(float(compiled_seconds or 0.0), 0.0),
            warnings=warnings,
        )

    eager = max(float(eager_seconds or 0.0), 0.0)
    compiled = max(float(compiled_seconds or 0.0), 0.0)
    if eager <= 0.0 or compiled <= 0.0:
        return CompileProbeResult(
            route=route_name,
            target=target_name,
            decision="disable",
            reason="missing valid eager/compiled timing",
            eager_seconds=eager,
            compiled_seconds=compiled,
            warnings=warnings,
        )

    speedup_ratio = (eager - compiled) / eager
    eager_vram = max(float(eager_peak_vram or 0.0), 0.0)
    compiled_vram = max(float(compiled_peak_vram or 0.0), 0.0)
    vram_ratio = 0.0
    if eager_vram > 0.0 and compiled_vram > eager_vram:
        vram_ratio = (compiled_vram - eager_vram) / eager_vram

    if vram_ratio > max(float(max_vram_increase_ratio or 0.0), 0.0):
        return CompileProbeResult(
            route=route_name,
            target=target_name,
            decision="disable",
            reason="compiled path increased peak VRAM beyond limit",
            eager_seconds=eager,
            compiled_seconds=compiled,
            eager_peak_vram=eager_vram,
            compiled_peak_vram=compiled_vram,
            speedup_ratio=speedup_ratio,
            vram_increase_ratio=vram_ratio,
            warnings=warnings,
        )

    if speedup_ratio < float(min_speedup_ratio or 0.0):
        return CompileProbeResult(
            route=route_name,
            target=target_name,
            decision="disable",
            reason="compiled path did not meet minimum speedup",
            eager_seconds=eager,
            compiled_seconds=compiled,
            eager_peak_vram=eager_vram,
            compiled_peak_vram=compiled_vram,
            speedup_ratio=speedup_ratio,
            vram_increase_ratio=vram_ratio,
            warnings=warnings,
        )

    return CompileProbeResult(
        route=route_name,
        target=target_name,
        decision="keep",
        reason="compiled path passed probe thresholds",
        eager_seconds=eager,
        compiled_seconds=compiled,
        eager_peak_vram=eager_vram,
        compiled_peak_vram=compiled_vram,
        speedup_ratio=speedup_ratio,
        vram_increase_ratio=vram_ratio,
        warnings=warnings,
    )
