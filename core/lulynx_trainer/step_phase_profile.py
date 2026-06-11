"""Optional CUDA-synchronized training step phase profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch


@dataclass
class _PhaseStart:
    cpu_started: float
    cuda_event: Any = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return min(max(float(value or 0.0), 0.0), 1.0)


def _round(value: float, digits: int = 6) -> float:
    return round(float(value or 0.0), digits)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if value is not None]
    return sum(items) / len(items) if items else 0.0


def _phase_dict(profile: Mapping[str, Any]) -> Dict[str, float]:
    phases = profile.get("phases_ms")
    if not isinstance(phases, Mapping):
        return {}
    return {str(key): max(_safe_float(value), 0.0) for key, value in phases.items()}


def _transfer_share(transfer_profile: Optional[Mapping[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    if not isinstance(transfer_profile, Mapping) or not transfer_profile:
        return 0.0, {}
    share = _safe_float(transfer_profile.get("step_share"), -1.0)
    if share < 0.0:
        transfer_seconds = _safe_float(transfer_profile.get("transfer_seconds"))
        step_seconds = _safe_float(transfer_profile.get("step_seconds"))
        share = transfer_seconds / max(step_seconds, 1e-9) if step_seconds > 0.0 else 0.0
    evidence = {
        "step_share": _round(_clamp01(share)),
        "transfer_seconds": _round(_safe_float(transfer_profile.get("transfer_seconds"))),
        "step_seconds": _round(_safe_float(transfer_profile.get("step_seconds"))),
        "ops": _safe_int(transfer_profile.get("ops")),
        "mib": _round(_safe_float(transfer_profile.get("mib")), 4),
        "bandwidth_mib_s": _round(_safe_float(transfer_profile.get("bandwidth_mib_s")), 4),
        "recommendation": str(transfer_profile.get("recommendation", "") or ""),
    }
    return _clamp01(share), evidence


def _nested_transfer_share(profiles: Iterable[Mapping[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    shares: List[float] = []
    latest: Dict[str, Any] = {}
    for profile in profiles:
        bubble = profile.get("gpu_bubble_profile") if isinstance(profile, Mapping) else None
        evidence = bubble.get("evidence") if isinstance(bubble, Mapping) else None
        if not isinstance(evidence, Mapping):
            continue
        share = _safe_float(evidence.get("h2d_transfer_share"), -1.0)
        if share >= 0.0:
            shares.append(_clamp01(share))
        transfer = evidence.get("transfer")
        if isinstance(transfer, Mapping) and transfer:
            latest = dict(transfer)
    if not shares:
        return 0.0, latest
    return _clamp01(_mean(shares)), latest


def _sum_named_share(phase_share: Mapping[str, float], names: Iterable[str]) -> float:
    name_set = {str(name) for name in names}
    return _clamp01(sum(float(phase_share.get(name, 0.0) or 0.0) for name in name_set))


def _sum_keyword_share(phase_share: Mapping[str, float], keywords: Iterable[str]) -> float:
    keys = tuple(str(keyword).lower() for keyword in keywords)
    total = 0.0
    for label, share in phase_share.items():
        lowered = str(label).lower()
        if any(keyword in lowered for keyword in keys):
            total += float(share or 0.0)
    return _clamp01(total)


def _recommendations_for(bottlenecks: List[str], bubble_ratio: float) -> List[str]:
    advice: List[str] = []
    if "data_bound" in bottlenecks:
        advice.append("profile DataLoader workers/persistent_workers/prefetch_factor and prefer cache-first datasets for repeated runs")
    if "transfer_bound" in bottlenecks:
        advice.append("A/B pinned memory, non_blocking H2D, batch prefetch, and reduced CPU-offload churn")
    if "optimizer_bound" in bottlenecks:
        advice.append("A/B foreach/fused/8-bit/native optimizer paths and check whether larger batch or accumulation improves occupancy")
    if "logging_checkpoint_bound" in bottlenecks:
        advice.append("move logging/checkpoint work off the hot step or increase save/log intervals during throughput probes")
    if "host_scheduling_bound" in bottlenecks:
        advice.append("inspect Python callbacks, SafeGuard scans, profiler syncs, and other host-side scheduling gaps")
    if "compute_bound" in bottlenecks:
        advice.append("GPU compute dominates; compare attention backend, compile/cache strategy, and model-family kernel choices")
    if not advice and bubble_ratio >= 0.05:
        advice.append("collect a longer phase-profile window with CUDA utilization telemetry to separate host wait from true compute")
    if not advice:
        advice.append("no dominant bubble detected in the current profiling window")
    return advice


def build_step_phase_bubble_profile(
    phase_profiles: Iterable[Mapping[str, Any]],
    *,
    transfer_profile: Optional[Mapping[str, Any]] = None,
    steady_warmup: int = 0,
) -> Dict[str, Any]:
    """Aggregate step phase samples and estimate the dominant GPU bubble source."""

    profiles = [dict(profile) for profile in phase_profiles if isinstance(profile, Mapping)]
    if steady_warmup > 0 and len(profiles) > steady_warmup:
        profiles = profiles[steady_warmup:]
    if not profiles:
        return {
            "schema_version": 1,
            "profile": "step_phase_bubble_profile_v0",
            "step_count": 0,
            "mean_step_ms": 0.0,
            "phase_share": {},
            "bubble_ratio_estimate": 0.0,
            "dominant_bottleneck": "unknown",
            "bottlenecks": [],
            "recommendations": ["enable step_phase_profile for a short probe window to collect phase evidence"],
            "evidence": {},
        }

    step_walls = [max(_safe_float(profile.get("step_wall_ms")), 0.0) for profile in profiles]
    mean_step_ms = max(_mean(step_walls), 0.0)
    phase_totals: Dict[str, float] = {}
    for profile in profiles:
        for label, value in _phase_dict(profile).items():
            phase_totals[label] = phase_totals.get(label, 0.0) + value

    step_count = len(profiles)
    phase_mean_ms = {
        label: value / max(float(step_count), 1.0)
        for label, value in sorted(phase_totals.items())
    }
    data_wait_ms = float(phase_mean_ms.get("data_wait", 0.0) or 0.0)
    denom = max(mean_step_ms + data_wait_ms, mean_step_ms, 1e-9)
    phase_share = {
        label: _round(value / denom)
        for label, value in phase_mean_ms.items()
    }

    train_step_share = _clamp01(float(phase_share.get("train_step_total", 0.0) or 0.0))
    optimizer_update_share = _clamp01(float(phase_share.get("optimizer_update_total", 0.0) or 0.0))
    optimizer_plus_zero_share = _sum_named_share(phase_share, ("optimizer_step", "zero_grad"))
    optimizer_share = max(optimizer_update_share, optimizer_plus_zero_share)
    data_wait_share = float(phase_share.get("data_wait", 0.0) or 0.0)
    logging_checkpoint_share = _sum_keyword_share(
        phase_share,
        ("log", "callback", "checkpoint", "save", "validation", "safeguard"),
    )
    h2d_transfer_share, transfer_evidence = _transfer_share(transfer_profile)
    if not transfer_evidence:
        h2d_transfer_share, transfer_evidence = _nested_transfer_share(profiles)
    elif mean_step_ms > 0.0 and denom > mean_step_ms:
        h2d_transfer_share = _clamp01(h2d_transfer_share * (mean_step_ms / denom))

    non_transfer_accounted = _clamp01(
        train_step_share + optimizer_share + data_wait_share + logging_checkpoint_share
    )
    host_gap_share = _clamp01(1.0 - non_transfer_accounted)
    wait_like_share = _clamp01(data_wait_share + h2d_transfer_share + logging_checkpoint_share)
    bubble_ratio = _clamp01(wait_like_share + host_gap_share)

    bottlenecks: List[str] = []
    if data_wait_share >= 0.08:
        bottlenecks.append("data_bound")
    if h2d_transfer_share >= 0.05:
        bottlenecks.append("transfer_bound")
    if optimizer_share >= 0.15:
        bottlenecks.append("optimizer_bound")
    if logging_checkpoint_share >= 0.05:
        bottlenecks.append("logging_checkpoint_bound")
    if host_gap_share >= 0.12:
        bottlenecks.append("host_scheduling_bound")
    if not bottlenecks and train_step_share >= 0.65:
        bottlenecks.append("compute_bound")

    problem_scores = {
        "data_bound": data_wait_share,
        "transfer_bound": h2d_transfer_share,
        "optimizer_bound": optimizer_share,
        "logging_checkpoint_bound": logging_checkpoint_share,
        "host_scheduling_bound": host_gap_share,
    }
    active_scores = {key: value for key, value in problem_scores.items() if key in bottlenecks}
    if active_scores:
        dominant = max(active_scores.items(), key=lambda item: item[1])[0]
    elif "compute_bound" in bottlenecks:
        dominant = "compute_bound"
    else:
        dominant = "balanced"

    top_phases = sorted(
        (
            {"label": label, "mean_ms": _round(ms, 4), "share": _round(phase_share.get(label, 0.0))}
            for label, ms in phase_mean_ms.items()
        ),
        key=lambda item: item["mean_ms"],
        reverse=True,
    )[:8]

    return {
        "schema_version": 1,
        "profile": "step_phase_bubble_profile_v0",
        "step_count": step_count,
        "mean_step_ms": _round(mean_step_ms, 4),
        "phase_mean_ms": {label: _round(value, 4) for label, value in phase_mean_ms.items()},
        "phase_share": phase_share,
        "bubble_ratio_estimate": _round(bubble_ratio),
        "dominant_bottleneck": dominant,
        "bottlenecks": bottlenecks,
        "recommendations": _recommendations_for(bottlenecks, bubble_ratio),
        "evidence": {
            "train_step_share": _round(train_step_share),
            "optimizer_share": _round(optimizer_share),
            "data_wait_share": _round(data_wait_share),
            "h2d_transfer_share": _round(h2d_transfer_share),
            "logging_checkpoint_share": _round(logging_checkpoint_share),
            "host_gap_share": _round(host_gap_share),
            "top_phases": top_phases,
            "transfer": transfer_evidence,
        },
    }


class StepPhaseProfiler:
    """Collect coarse phase timings for optimizer-step profiling windows.

    The profiler is intentionally opt-in because CUDA synchronization changes
    training timing. It is meant for short benchmark probes, not normal runs.
    """

    def __init__(self, *, enabled: bool = False, sync_cuda: bool = True, history_size: int = 50) -> None:
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self.history_size = max(int(history_size or 50), 1)
        self._phases_ms: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._cuda_event_ms: Dict[str, float] = {}
        self._cuda_event_counts: Dict[str, int] = {}
        self._cuda_event_error_count = 0
        self._history: List[Dict[str, Any]] = []
        self._last_snapshot: Dict[str, Any] = {}
        self._last_bubble_profile: Dict[str, Any] = {}

    def reset_group(self) -> None:
        self._phases_ms = {}
        self._counts = {}
        self._cuda_event_ms = {}
        self._cuda_event_counts = {}
        self._cuda_event_error_count = 0

    def start(self) -> Any:
        if not self.enabled:
            return time.perf_counter()
        if self.enabled and self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return _PhaseStart(cpu_started=time.perf_counter(), cuda_event=event)
            except Exception:
                self._cuda_event_error_count += 1
        return _PhaseStart(cpu_started=time.perf_counter())

    def start_cpu(self) -> Any:
        if not self.enabled:
            return time.perf_counter()
        return _PhaseStart(cpu_started=time.perf_counter())

    def record(self, label: str, started: Any) -> float:
        return self._record_elapsed(label, started, sync_cuda=True)

    def record_cpu(self, label: str, started: Any) -> float:
        return self._record_elapsed(label, started, sync_cuda=False)

    def record_optimizer_update_substage(self, label: str, started: Any) -> float:
        key = f"optimizer_update_substage.{str(label or 'unknown')}"
        return self._record_elapsed(key, started, sync_cuda=False)

    def record_optimizer_step_micro_substage(self, label: str, started: Any) -> float:
        key = f"optimizer_step_micro.{str(label or 'unknown')}"
        return self._record_elapsed(key, started, sync_cuda=False)

    def latest_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_snapshot)

    def latest_bubble_profile(self) -> Dict[str, Any]:
        return dict(self._last_bubble_profile)

    def _record_elapsed(self, label: str, started: Any, *, sync_cuda: bool) -> float:
        if not self.enabled:
            return time.perf_counter()
        key = str(label or "unknown")
        cpu_started = float(started.cpu_started if isinstance(started, _PhaseStart) else started or 0.0)
        cuda_elapsed_ms: Optional[float] = None
        start_event = started.cuda_event if isinstance(started, _PhaseStart) else None
        if sync_cuda and self.sync_cuda and torch.cuda.is_available():
            if start_event is not None:
                try:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    end_event.synchronize()
                    cuda_elapsed_ms = max(float(start_event.elapsed_time(end_event) or 0.0), 0.0)
                except Exception:
                    self._cuda_event_error_count += 1
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed_ms = max((now - float(cpu_started or now)) * 1000.0, 0.0)
        self._phases_ms[key] = self._phases_ms.get(key, 0.0) + elapsed_ms
        self._counts[key] = self._counts.get(key, 0) + 1
        if cuda_elapsed_ms is not None:
            self._cuda_event_ms[key] = self._cuda_event_ms.get(key, 0.0) + cuda_elapsed_ms
            self._cuda_event_counts[key] = self._cuda_event_counts.get(key, 0) + 1
        return now

    def snapshot(
        self,
        *,
        step_wall_seconds: float,
        accumulation_steps: int,
        transfer_profile: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        step_wall_ms = max(float(step_wall_seconds or 0.0) * 1000.0, 0.0)
        phases = {key: round(float(value), 4) for key, value in sorted(self._phases_ms.items())}
        counts = {key: int(value) for key, value in sorted(self._counts.items())}
        cuda_events = {key: round(float(value), 4) for key, value in sorted(self._cuda_event_ms.items())}
        cuda_event_counts = {key: int(value) for key, value in sorted(self._cuda_event_counts.items())}
        optimizer_step_ms = float(self._phases_ms.get("optimizer_step", 0.0))
        scheduler_step_ms = float(self._phases_ms.get("scheduler_step", 0.0))
        zero_grad_ms = float(self._phases_ms.get("zero_grad", 0.0))
        update_total_ms = float(self._phases_ms.get("optimizer_update_total", 0.0))
        optimizer_plus_zero_ms = optimizer_step_ms + zero_grad_ms
        optimizer_update_subphase_ms = optimizer_step_ms + scheduler_step_ms + zero_grad_ms
        optimizer_update_unaccounted_ms = max(update_total_ms - optimizer_update_subphase_ms, 0.0)
        optimizer_update_accounted_exceeds_total_ms = max(optimizer_update_subphase_ms - update_total_ms, 0.0)
        optimizer_update_outer_substage_profile = self._optimizer_update_outer_substage_profile(
            update_total_ms=update_total_ms,
            existing_accounted_subphase_ms=optimizer_update_subphase_ms,
        )
        optimizer_step_micro_profile = self._optimizer_step_micro_profile(
            optimizer_step_ms=optimizer_step_ms,
        )
        unaccounted_share_of_step = round(optimizer_update_unaccounted_ms / max(step_wall_ms, 1e-9), 6)
        unaccounted_share_of_update = (
            round(optimizer_update_unaccounted_ms / max(update_total_ms, 1e-9), 6)
            if update_total_ms > 0.0
            else 0.0
        )
        optimizer_update_breakdown = {
            "schema_version": 1,
            "profile": "optimizer_update_breakdown_v0",
            "source": "step_phase_profile.phases_ms",
            "outer_phase_label": "optimizer_update_total",
            "subphase_labels": ["optimizer_step", "scheduler_step", "zero_grad"],
            "has_outer_phase": "optimizer_update_total" in self._phases_ms,
            "has_subphase_profile": any(
                label in self._phases_ms
                for label in ("optimizer_step", "scheduler_step", "zero_grad")
            ),
            "optimizer_update_total_ms": round(update_total_ms, 4),
            "optimizer_step_ms": round(optimizer_step_ms, 4),
            "scheduler_step_ms": round(scheduler_step_ms, 4),
            "zero_grad_ms": round(zero_grad_ms, 4),
            "subphase_accounted_ms": round(optimizer_update_subphase_ms, 4),
            "accounted_subphase_ms": round(optimizer_update_subphase_ms, 4),
            "unaccounted_optimizer_update_ms": round(optimizer_update_unaccounted_ms, 4),
            "accounted_exceeds_total_ms": round(optimizer_update_accounted_exceeds_total_ms, 4),
            "subphase_accounted_share_of_update": round(
                optimizer_update_subphase_ms / max(update_total_ms, 1e-9),
                6,
            )
            if update_total_ms > 0.0
            else 0.0,
            "unaccounted_share_of_update": unaccounted_share_of_update,
            "unaccounted_optimizer_update_total_share": unaccounted_share_of_update,
            "unaccounted_share_of_step": unaccounted_share_of_step,
            "unaccounted_optimizer_update_share": unaccounted_share_of_step,
            "outer_substage_profile": optimizer_update_outer_substage_profile,
            "outer_substage_profile_available": bool(
                optimizer_update_outer_substage_profile.get("profiled_substage_count")
            ),
            "optimizer_step_micro_profile": optimizer_step_micro_profile,
            "optimizer_step_micro_profile_available": bool(
                optimizer_step_micro_profile.get("profiled_substage_count")
            ),
            "optimizer_step_micro_profiled_ms": optimizer_step_micro_profile.get(
                "profiled_substage_total_ms", 0.0
            ),
            "optimizer_step_micro_profile_labels": optimizer_step_micro_profile.get(
                "profiled_substage_labels", []
            ),
            "planned_outer_substage_accounted_ms": optimizer_update_outer_substage_profile.get(
                "profiled_substage_total_ms", 0.0
            ),
            "residual_after_planned_substages_ms": optimizer_update_outer_substage_profile.get(
                "residual_after_planned_substages_ms", 0.0
            ),
            "outer_substage_labels": optimizer_update_outer_substage_profile.get(
                "profiled_substage_labels", []
            ),
            "runtime_default_change": False,
        }
        denom = max(step_wall_ms, 1e-9)
        snapshot = {
            "enabled": True,
            "sync_cuda": bool(self.sync_cuda),
            "step_wall_ms": round(step_wall_ms, 4),
            "accumulation_steps": max(int(accumulation_steps or 1), 1),
            "phases_ms": phases,
            "counts": counts,
            "cuda_event_profile_available": bool(cuda_events),
            "cuda_event_ms": cuda_events,
            "cuda_event_counts": cuda_event_counts,
            "cuda_event_error_count": int(self._cuda_event_error_count),
            "optimizer_step_share": round(optimizer_step_ms / denom, 6),
            "optimizer_plus_zero_grad_share": round(optimizer_plus_zero_ms / denom, 6),
            "optimizer_update_share": round(update_total_ms / denom, 6),
            "optimizer_update_breakdown": optimizer_update_breakdown,
        }
        self._history.append(dict(snapshot))
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]
        bubble_profile = build_step_phase_bubble_profile(
            self._history,
            transfer_profile=transfer_profile,
        )
        snapshot["gpu_bubble_profile"] = bubble_profile
        self._last_bubble_profile = dict(bubble_profile)
        self._last_snapshot = dict(snapshot)
        return snapshot

    def _optimizer_update_outer_substage_profile(
        self,
        *,
        update_total_ms: float,
        existing_accounted_subphase_ms: float,
    ) -> Dict[str, Any]:
        prefix = "optimizer_update_substage."
        substage_ms = {
            label[len(prefix) :]: float(value or 0.0)
            for label, value in sorted(self._phases_ms.items())
            if str(label).startswith(prefix)
        }
        substage_counts = {
            label[len(prefix) :]: int(value or 0)
            for label, value in sorted(self._counts.items())
            if str(label).startswith(prefix)
        }
        total_ms = sum(substage_ms.values())
        accounted_ms = float(existing_accounted_subphase_ms or 0.0) + total_ms
        residual_ms = max(float(update_total_ms or 0.0) - accounted_ms, 0.0)
        accounted_exceeds_total_ms = max(accounted_ms - float(update_total_ms or 0.0), 0.0)
        return {
            "schema_version": 1,
            "profile": "optimizer_update_outer_substage_profile_v0",
            "source": "step_phase_profile.phases_ms",
            "outer_phase_label": "optimizer_update_total",
            "label_prefix": prefix,
            "profiled_substage_labels": list(substage_ms),
            "profiled_substage_count": len(substage_ms),
            "profiled_substage_ms": {
                label: round(value, 4) for label, value in substage_ms.items()
            },
            "profiled_substage_counts": substage_counts,
            "profiled_substage_total_ms": round(total_ms, 4),
            "existing_accounted_subphase_ms": round(
                float(existing_accounted_subphase_ms or 0.0), 4
            ),
            "planned_outer_accounted_ms": round(accounted_ms, 4),
            "residual_after_planned_substages_ms": round(residual_ms, 4),
            "accounted_exceeds_total_ms": round(accounted_exceeds_total_ms, 4),
            "residual_bucket_labels": [
                "optimizer_update_boundary_sync_gap",
                "outer_phase_record_overhead",
            ],
            "runtime_default_change": False,
        }

    def _optimizer_step_micro_profile(
        self,
        *,
        optimizer_step_ms: float,
    ) -> Dict[str, Any]:
        prefix = "optimizer_step_micro."
        substage_ms = {
            label[len(prefix) :]: float(value or 0.0)
            for label, value in sorted(self._phases_ms.items())
            if str(label).startswith(prefix)
        }
        substage_counts = {
            label[len(prefix) :]: int(value or 0)
            for label, value in sorted(self._counts.items())
            if str(label).startswith(prefix)
        }
        total_ms = sum(substage_ms.values())
        residual_ms = max(float(optimizer_step_ms or 0.0) - total_ms, 0.0)
        accounted_exceeds_total_ms = max(total_ms - float(optimizer_step_ms or 0.0), 0.0)
        return {
            "schema_version": 1,
            "profile": "optimizer_step_micro_profile_v0",
            "source": "step_phase_profile.phases_ms",
            "outer_phase_label": "optimizer_step",
            "label_prefix": prefix,
            "profiled_substage_labels": list(substage_ms),
            "profiled_substage_count": len(substage_ms),
            "profiled_substage_ms": {
                label: round(value, 4) for label, value in substage_ms.items()
            },
            "profiled_substage_counts": substage_counts,
            "profiled_substage_total_ms": round(total_ms, 4),
            "optimizer_step_ms": round(float(optimizer_step_ms or 0.0), 4),
            "residual_after_profiled_substages_ms": round(residual_ms, 4),
            "accounted_exceeds_optimizer_step_ms": round(accounted_exceeds_total_ms, 4),
            "residual_bucket_labels": [
                "optimizer_step_internal_kernel_or_python_gap",
                "optimizer_step_record_overhead",
            ],
            "runtime_default_change": False,
        }


__all__ = ["StepPhaseProfiler", "build_step_phase_bubble_profile"]
