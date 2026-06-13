# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Runtime memory / transfer / timing / cudagraph cluster extracted verbatim from
``training_loop.py`` as a mixin.

These methods (cuda-cache release strategy, data-transfer profile sampling/flush,
profiled module moves + device/dtype + offload stats, cuda-memory snapshot, peak
VRAM diagnostics, cuda-cache release, precision/block-swap observations, VRAM
smart-sensing, phase module-state verification, cpu-resident component ensuring,
step-timing windows + transfer-profile step/recommendation, and cudagraph
eligibility/init/replay) run as bound methods of the ``TrainingLoop`` instance —
identical ``self`` semantics, identical call sites. Behaviour is unchanged; this
split consolidates the GPU memory/transfer/timing runtime machinery so the core
training loop reads cleanly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch

from .device_state import capture_module_state

logger = logging.getLogger(__name__)


class TrainingLoopRuntimeMemoryMixin:
    @staticmethod
    def _normalize_cuda_cache_release_strategy(strategy: Optional[str]) -> str:
        normalized = str(strategy or "oom_only").strip().lower()
        aliases = {
            "none": "off",
            "false": "off",
            "0": "off",
            "disabled": "off",
            "oom": "oom_only",
            "on_oom": "oom_only",
            "safe": "oom_only",
            "phase": "phase_boundary",
            "boundary": "phase_boundary",
            "component_offload": "phase_boundary",
            "after_step": "aggressive",
            "every_step": "aggressive",
        }
        normalized = aliases.get(normalized, normalized)
        return (
            normalized
            if normalized in {"off", "oom_only", "phase_boundary", "after_optimizer", "aggressive"}
            else "oom_only"
        )

    def _record_transfer_profile_sample(self, label: str, bytes_moved: int, elapsed_seconds: float) -> None:
        self._transfer_profile_seconds += max(float(elapsed_seconds or 0.0), 0.0)
        self._transfer_profile_bytes += int(bytes_moved)
        self._transfer_profile_ops += 1
        bucket = self._transfer_profile_by_label.setdefault(
            label,
            {"seconds": 0.0, "bytes": 0.0, "ops": 0.0},
        )
        bucket["seconds"] += max(float(elapsed_seconds or 0.0), 0.0)
        bucket["bytes"] += float(bytes_moved)
        bucket["ops"] += 1.0

    def _flush_transfer_profile_events(self) -> None:
        if not self._transfer_profile_pending_events:
            return
        pending = self._transfer_profile_pending_events
        self._transfer_profile_pending_events = []
        for event in pending:
            end_event = event.get("end")
            start_event = event.get("start")
            try:
                if end_event is not None:
                    end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event)) if start_event is not None and end_event is not None else 0.0
            except Exception as exc:
                logger.debug("data transfer CUDA event profiling sample skipped: %s", exc)
                continue
            self._record_transfer_profile_sample(
                str(event.get("label", "unknown")),
                int(event.get("bytes", 0) or 0),
                elapsed_ms / 1000.0,
            )

    def _profiled_to(
        self,
        tensor: torch.Tensor,
        *,
        label: str,
        device: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_device = self.device if device is None else device
        non_blocking = bool(getattr(self, "data_transfer_non_blocking", True))
        profile_enabled = bool(getattr(self, "data_transfer_profile_enabled", False))
        profile_mode = self._normalize_data_transfer_profile_mode(
            getattr(self, "data_transfer_profile_mode", "event")
        )
        profile_enabled = profile_enabled and profile_mode != "off"
        bytes_moved = int(tensor.numel() * tensor.element_size())
        cuda_profile = profile_enabled and torch.cuda.is_available() and str(target_device).startswith("cuda")

        if cuda_profile and profile_mode == "sync":
            torch.cuda.synchronize()
        start_event = end_event = None
        if cuda_profile and profile_mode == "event":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter() if profile_enabled and (not cuda_profile or profile_mode == "sync") else 0.0
        moved = tensor.to(device=target_device, dtype=dtype, non_blocking=non_blocking)
        if profile_enabled:
            if cuda_profile and profile_mode == "event":
                end_event.record()
                self._transfer_profile_pending_events.append(
                    {
                        "label": label,
                        "bytes": bytes_moved,
                        "start": start_event,
                        "end": end_event,
                    }
                )
            elif cuda_profile and profile_mode == "sync":
                torch.cuda.synchronize()
                self._record_transfer_profile_sample(label, bytes_moved, time.perf_counter() - start)
            else:
                self._record_transfer_profile_sample(label, bytes_moved, time.perf_counter() - start)
        return moved

    def _module_device_dtype(self, module: Optional[torch.nn.Module]) -> tuple[torch.device, torch.dtype]:
        if module is None:
            return self._runtime_device, self.dtype
        try:
            param = next(module.parameters())
        except StopIteration:
            return self._runtime_device, self.dtype
        return param.device, param.dtype

    def _move_module_for_runtime(
        self,
        module: Optional[torch.nn.Module],
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[bool, torch.device, torch.dtype]:
        original_device, original_dtype = self._module_device_dtype(module)
        if module is None:
            return False, original_device, original_dtype
        target_dtype = original_dtype if dtype is None else dtype
        moved = original_device != self._runtime_device or original_dtype != target_dtype
        if moved:
            module.to(device=self._runtime_device, dtype=target_dtype)
        return moved, original_device, original_dtype

    def _restore_module_after_runtime(
        self,
        module: Optional[torch.nn.Module],
        moved: bool,
        original_device: torch.device,
        original_dtype: torch.dtype,
    ) -> None:
        if module is None or not moved:
            return
        module.to(device=original_device, dtype=original_dtype)

    def _refresh_module_offload_stats(self) -> None:
        if self._module_offload_manager is not None:
            self.memory_optimization_state["runtime_stats"] = self._module_offload_manager.stats_dict()

    def _cuda_memory_snapshot(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        allocated = float(torch.cuda.memory_allocated()) / (1024 * 1024)
        reserved = float(torch.cuda.memory_reserved()) / (1024 * 1024)
        peak_allocated = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
        peak_reserved = float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "peak_allocated_mb": round(peak_allocated, 1),
            "peak_reserved_mb": round(peak_reserved, 1),
            "reserved_gap_mb": round(max(reserved - allocated, 0.0), 1),
            "peak_reserved_gap_mb": round(max(peak_reserved - peak_allocated, 0.0), 1),
        }

    def _build_peak_vram_diagnostics(self, stages: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        normalized = {
            str(name): dict(snapshot)
            for name, snapshot in stages.items()
            if isinstance(snapshot, dict) and snapshot
        }
        if not normalized:
            return {}
        max_reserved_stage = max(
            normalized,
            key=lambda name: float(normalized[name].get("peak_reserved_mb", 0.0) or 0.0),
        )
        max_allocated_stage = max(
            normalized,
            key=lambda name: float(normalized[name].get("peak_allocated_mb", 0.0) or 0.0),
        )
        return {
            "stages": normalized,
            "max_reserved_stage": max_reserved_stage,
            "max_reserved_mb": round(float(normalized[max_reserved_stage].get("peak_reserved_mb", 0.0) or 0.0), 1),
            "max_allocated_stage": max_allocated_stage,
            "max_allocated_mb": round(float(normalized[max_allocated_stage].get("peak_allocated_mb", 0.0) or 0.0), 1),
            "allocator_cache_gap_mb": round(
                max(
                    float(normalized[max_reserved_stage].get("peak_reserved_mb", 0.0) or 0.0)
                    - float(normalized[max_reserved_stage].get("peak_allocated_mb", 0.0) or 0.0),
                    0.0,
                ),
                1,
            ),
        }

    def _maybe_release_cuda_cache(self, phase: str, step: int, *, force: bool = False) -> Dict[str, Any]:
        strategy = self._cuda_cache_release_strategy
        if not force and strategy == "off":
            return {}
        if not torch.cuda.is_available():
            return {}
        phase = str(phase or "")
        if not force:
            allowed = False
            phase_key = phase
            if strategy == "oom_only":
                allowed = False
            elif strategy == "after_optimizer":
                allowed = phase == "after_optimizer"
            elif strategy == "phase_boundary":
                allowed = phase == "phase_boundary"
            elif strategy == "aggressive":
                allowed = phase in {"after_optimizer", "phase_boundary", "swap_prepare"}
            if not allowed:
                return {}
            interval = max(int(self._cuda_cache_release_interval or 1), 1)
            if int(step or 0) % interval != 0:
                return {}
            seen_step = self._cuda_cache_release_seen_steps.get(phase_key)
            if seen_step == int(step or 0):
                return {}
        else:
            phase_key = phase
        before = self._cuda_memory_snapshot()
        started = time.perf_counter()
        try:
            torch.cuda.empty_cache()
            after = self._cuda_memory_snapshot()
        except Exception as exc:
            report = {
                "strategy": strategy,
                "phase": phase,
                "step": int(step or 0),
                "forced": bool(force),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._last_cuda_cache_release = report
            return report
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        before_reserved = float(before.get("reserved_mb", 0.0) or 0.0)
        after_reserved = float(after.get("reserved_mb", 0.0) or 0.0)
        before_allocated = float(before.get("allocated_mb", 0.0) or 0.0)
        after_allocated = float(after.get("allocated_mb", 0.0) or 0.0)
        report = {
            "strategy": strategy,
            "phase": phase,
            "step": int(step or 0),
            "forced": bool(force),
            "ok": True,
            "elapsed_ms": round(elapsed_ms, 2),
            "before": before,
            "after": after,
            "released_reserved_mb": round(max(before_reserved - after_reserved, 0.0), 1),
            "released_allocated_mb": round(max(before_allocated - after_allocated, 0.0), 1),
        }
        self._cuda_cache_release_seen_steps[phase_key] = int(step or 0)
        self._last_cuda_cache_release = report
        return report

    def _update_precision_swap_observations(
        self,
        step_wall_seconds: float,
        step_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        profile = self.memory_optimization_state.get("precision_swap_profile")
        if not isinstance(profile, dict):
            return
        self._runtime_observation_steps += 1
        self._runtime_observation_total_step_seconds += max(float(step_wall_seconds or 0.0), 0.0)
        stats = getattr(getattr(self, "_block_offloader", None), "stats", None)
        observations: Dict[str, Any] = dict(profile.get("runtime_observations") or {})
        observations.update(
            {
                "steps_observed": int(self._runtime_observation_steps),
                "last_step_wall_seconds": round(float(step_wall_seconds or 0.0), 4),
                "avg_step_wall_seconds": round(
                    self._runtime_observation_total_step_seconds / max(self._runtime_observation_steps, 1),
                    4,
                ),
            }
        )
        if stats is not None:
            observations.update(
                {
                    "swap_count": int(getattr(stats, "swap_count", 0) or 0),
                    "wait_count": int(getattr(stats, "wait_count", 0) or 0),
                    "total_swap_ms": round(float(getattr(stats, "total_swap_ms", 0.0) or 0.0), 2),
                    "prepare_count": int(getattr(stats, "prepare_count", 0) or 0),
                    "total_prepare_ms": round(float(getattr(stats, "total_prepare_ms", 0.0) or 0.0), 2),
                }
            )
        if isinstance(step_info, dict) and isinstance(step_info.get("peak_vram_stages"), dict):
            observations["peak_vram_stages"] = dict(step_info["peak_vram_stages"])
        if isinstance(step_info, dict) and isinstance(step_info.get("peak_vram_diagnostics"), dict):
            observations["peak_vram_diagnostics"] = dict(step_info["peak_vram_diagnostics"])
        if isinstance(step_info, dict) and isinstance(step_info.get("cuda_cache_release"), dict):
            observations["cuda_cache_release"] = dict(step_info["cuda_cache_release"])
        if isinstance(step_info, dict) and isinstance(step_info.get("precision_swap_offload"), dict):
            observations["precision_swap_offload"] = dict(step_info["precision_swap_offload"])
        profile["runtime_observations"] = observations
        self.memory_optimization_state["precision_swap_profile"] = profile
        self.memory_optimization_state["runtime_observations"] = observations

    def _update_block_swap_profile(self) -> None:
        offloader = getattr(self, "_block_offloader", None)
        profile_fn = getattr(offloader, "profile_state", None)
        if not callable(profile_fn):
            return
        try:
            self.memory_optimization_state["block_swap_profile"] = profile_fn()
        except Exception as exc:
            self.memory_optimization_state["block_swap_profile_error"] = str(exc)

    def _update_vram_smart_sensing_runtime(
        self,
        step_wall_seconds: float,
        step_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runtime-only slowdown sensing. It never mutates training strategy."""

        if not self.vram_smart_sensing_enabled:
            return {}
        step_seconds = max(float(step_wall_seconds or 0.0), 0.0)
        if step_seconds <= 0.0:
            return {}
        self._smart_sensing_observed_steps += 1
        observed = int(self._smart_sensing_observed_steps)
        baseline_steps = int(self.vram_smart_sensing_baseline_steps)
        if observed <= baseline_steps:
            self._smart_sensing_baseline_total_seconds += step_seconds
            if observed < baseline_steps:
                return {}
        baseline_avg = self._smart_sensing_baseline_total_seconds / max(min(observed, baseline_steps), 1)
        if observed == baseline_steps:
            report = {
                "enabled": True,
                "phase": "baseline_ready",
                "observed_steps": observed,
                "baseline_steps": baseline_steps,
                "baseline_avg_step_seconds": round(float(baseline_avg), 4),
                "slowdown_ratio_threshold": round(float(self.vram_smart_sensing_slowdown_ratio), 3),
                "action": "observe",
                "recommendations": [],
            }
            self._last_vram_smart_sensing_report = report
            return report

        self._smart_sensing_recent_seconds.append(step_seconds)
        if len(self._smart_sensing_recent_seconds) > self.vram_smart_sensing_window_steps:
            self._smart_sensing_recent_seconds.pop(0)
        window_avg = sum(self._smart_sensing_recent_seconds) / max(len(self._smart_sensing_recent_seconds), 1)
        ratio = window_avg / max(baseline_avg, 1e-6)
        if ratio < self.vram_smart_sensing_slowdown_ratio:
            return {}

        cuda = self._cuda_memory_snapshot()
        free_mb = 0.0
        total_mb = 0.0
        used_fraction = 0.0
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_mb = float(free_bytes) / (1024.0 * 1024.0)
                total_mb = float(total_bytes) / (1024.0 * 1024.0)
                used_fraction = 1.0 - (free_mb / max(total_mb, 1.0))
            except Exception:
                free_mb = 0.0
                total_mb = 0.0
        reserved_mb = float(cuda.get("reserved_mb", 0.0) or 0.0)
        reserved_fraction = reserved_mb / max(total_mb, 1.0) if total_mb > 0.0 else 0.0
        vram_pressure = bool(
            (total_mb > 0.0 and used_fraction >= 0.92)
            or (total_mb > 0.0 and reserved_fraction >= 0.90)
            or (free_mb > 0.0 and free_mb <= 768.0)
        )
        recommendations = ["check_shared_vram_or_pageable_memory"]
        if vram_pressure:
            recommendations.extend([
                "enable_streaming_offload",
                "enable_streaming_prefetch",
                "enable_sparse_swap",
                "enable_delta_cache_observe",
            ])
        else:
            recommendations.append("inspect_data_or_cpu_pipeline")
        report = {
            "enabled": True,
            "phase": "runtime_slowdown",
            "observed_steps": observed,
            "baseline_steps": baseline_steps,
            "baseline_avg_step_seconds": round(float(baseline_avg), 4),
            "window_steps": len(self._smart_sensing_recent_seconds),
            "window_avg_step_seconds": round(float(window_avg), 4),
            "last_step_wall_seconds": round(float(step_seconds), 4),
            "slowdown_ratio": round(float(ratio), 3),
            "slowdown_ratio_threshold": round(float(self.vram_smart_sensing_slowdown_ratio), 3),
            "vram_pressure": vram_pressure,
            "shared_vram_suspected": bool(vram_pressure),
            "cuda": {
                **cuda,
                "free_mb": round(float(free_mb), 1),
                "total_mb": round(float(total_mb), 1),
                "used_fraction": round(float(used_fraction), 4),
                "reserved_fraction": round(float(reserved_fraction), 4),
            },
            "action": "recommend_only",
            "recommendations": recommendations,
            "notes": [
                "runtime sensing is advisory only; it does not change residency, cache, or transfer format mid-run",
                "shared_vram_suspected is inferred from slowdown plus CUDA memory pressure, not a direct OS shared-memory counter",
            ],
        }
        if isinstance(step_info, dict) and isinstance(step_info.get("data_transfer_profile"), dict):
            report["data_transfer_profile"] = dict(step_info["data_transfer_profile"])
        self._last_vram_smart_sensing_report = report
        return report

    def _verify_phase_module_states(self, phase: str) -> None:
        if not self._module_offload_verify_state:
            return
        expected = {
            "unet": (self.unet, self._runtime_device, True, None),
            "vae": (self.vae, torch.device("cpu") if self._vae_cpu_residency else self._runtime_device, False, False),
            "text_encoder_1": (
                self.text_encoder_1,
                torch.device("cpu") if self._text_encoder_cpu_residency else self._runtime_device,
                self._train_text_encoder_1,
                None if self._train_text_encoder_1 else False,
            ),
            "text_encoder_2": (
                self.text_encoder_2,
                torch.device("cpu") if self._text_encoder_cpu_residency else self._runtime_device,
                self._train_text_encoder_2,
                None if self._train_text_encoder_2 else False,
            ),
        }
        for name, (module, expected_device, expected_training, expected_requires_grad) in expected.items():
            state = capture_module_state(module)
            if state is None:
                continue
            expected_device = torch.device(expected_device)
            if name == "unet" and self._module_offload_manager is not None:
                pass
            elif state.device.type != expected_device.type:
                if name == "unet" and module is not None:
                    active_param = next((param for param in module.parameters() if param.requires_grad), None)
                    if active_param is not None and active_param.device.type == expected_device.type:
                        continue
                message = f"[module-offload-state] {phase}: {name} is on {state.device}, expected {expected_device}"
                if name == "unet":
                    raise RuntimeError(message)
                logger.warning(message)
            if state.training != bool(expected_training):
                logger.warning(
                    "[module-offload-state] %s: %s training=%s expected=%s",
                    phase,
                    name,
                    state.training,
                    expected_training,
                )
            if expected_requires_grad is not None and state.requires_grad != bool(expected_requires_grad):
                logger.warning(
                    "[module-offload-state] %s: %s requires_grad=%s expected=%s",
                    phase,
                    name,
                    state.requires_grad,
                    expected_requires_grad,
                )

    def _ensure_cpu_resident_components(self, phase: str) -> None:
        """Keep frozen SDXL helper modules out of VRAM between encode phases."""
        moved_from_cuda = False

        def _move_to_cpu(module: Optional[torch.nn.Module], name: str) -> None:
            nonlocal moved_from_cuda
            state = capture_module_state(module)
            if module is None or state is None:
                return
            needs_grad_fix = state.requires_grad is not False
            needs_mode_fix = state.training is not False
            if state.device.type == "cpu" and state.dtype == torch.float32 and not needs_grad_fix and not needs_mode_fix:
                return
            if state.device.type == "cuda":
                moved_from_cuda = True
            module.eval()
            module.requires_grad_(False)
            module.to(device="cpu", dtype=torch.float32)
            logger.debug("[component-residency] %s: moved %s to cpu/float32", phase, name)

        if self._vae_cpu_residency:
            _move_to_cpu(self.vae, "vae")
        if self._text_encoder_cpu_residency:
            _move_to_cpu(self.text_encoder_1, "text_encoder_1")
            _move_to_cpu(self.text_encoder_2, "text_encoder_2")

        if moved_from_cuda and torch.cuda.is_available():
            self._maybe_release_cuda_cache("phase_boundary", self.global_step)
    def _record_transfer_profile_step(self, step_wall_seconds: float) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "data_transfer_profile_enabled", False)):
            return None
        self._flush_transfer_profile_events()
        self._transfer_profile_steps += 1
        self._transfer_profile_step_seconds += max(float(step_wall_seconds or 0.0), 0.0)
        window = max(int(getattr(self, "data_transfer_profile_window", 50) or 50), 1)

        total_step = max(self._transfer_profile_step_seconds, 1e-9)
        transfer = self._transfer_profile_seconds
        mib = self._transfer_profile_bytes / (1024.0 * 1024.0)
        bandwidth = mib / max(transfer, 1e-9) if transfer > 0 else 0.0
        top = sorted(
            self._transfer_profile_by_label.items(),
            key=lambda item: item[1].get("seconds", 0.0),
            reverse=True,
        )[:5]
        step_share = transfer / total_step
        recommendation = self._transfer_profile_recommendation(step_share)
        snapshot: Dict[str, Any] = {
            "steps": self._transfer_profile_steps,
            "window": window,
            "window_complete": self._transfer_profile_steps >= window,
            "step_seconds": self._transfer_profile_step_seconds,
            "transfer_seconds": transfer,
            "step_share": step_share,
            "ops": self._transfer_profile_ops,
            "bytes": self._transfer_profile_bytes,
            "mib": mib,
            "bandwidth_mib_s": bandwidth,
            "recommendation": recommendation,
            "top": [
                {
                    "label": name,
                    "seconds": float(stats.get("seconds", 0.0)),
                    "ops": int(stats.get("ops", 0.0)),
                    "bytes": int(stats.get("bytes", 0.0)),
                }
                for name, stats in top
            ],
        }
        self._last_transfer_profile_snapshot = snapshot

        if self._transfer_profile_steps < window:
            return snapshot

        top_summary = ", ".join(
            f"{name}:{stats['seconds'] * 1000.0:.1f}ms/{stats['ops']:.0f}ops"
            for name, stats in top
        ) or "none"
        logger.info(
            '[data-transfer-profile] steps=%d transfer=%.2fms step_share=%.2f%% ops=%d bytes=%.1fMiB bandwidth=%.1fMiB/s top=%s advice="%s"',
            self._transfer_profile_steps,
            transfer * 1000.0,
            step_share * 100.0,
            self._transfer_profile_ops,
            mib,
            bandwidth,
            top_summary,
            recommendation,
        )

        self._transfer_profile_steps = 0
        self._transfer_profile_step_seconds = 0.0
        self._transfer_profile_seconds = 0.0
        self._transfer_profile_bytes = 0
        self._transfer_profile_ops = 0
        self._transfer_profile_by_label.clear()
        return snapshot

    def _infer_step_timing_samples(self, accumulation_steps: int) -> tuple[int, str]:
        """Best-effort physical sample count without touching CUDA state."""

        trace = getattr(self, "_pipeline_trace", None)
        metadata = getattr(trace, "_metadata", {}) if trace is not None else {}
        batch_contract = metadata.get("batch_contract") if isinstance(metadata, dict) else None
        if isinstance(batch_contract, dict):
            inferred = int(batch_contract.get("inferred_physical_batch_size") or 0)
            if inferred > 0:
                return inferred, "pipeline_batch_contract"
            expected = int(batch_contract.get("expected_physical_batch_size") or 0)
            if expected > 0:
                return expected, "pipeline_batch_contract_expected"
        fallback = max(int(accumulation_steps or 1), 1)
        return fallback, "accumulation_steps_fallback"

    def _record_step_timing_window(
        self,
        step_wall_seconds: float,
        *,
        global_step: int,
        accumulation_steps: int,
        samples_seen: Optional[int] = None,
        samples_source: str = "",
    ) -> Dict[str, Any]:
        step_seconds = max(float(step_wall_seconds or 0.0), 0.0)
        if samples_seen is None:
            samples_seen, samples_source = self._infer_step_timing_samples(accumulation_steps)
        sample_count = max(int(samples_seen or 0), 0)
        source = str(samples_source or "provided")
        item = {
            "step": int(global_step or 0),
            "step_wall_seconds": step_seconds,
            "samples_seen": sample_count,
            "samples_source": source,
        }
        history = getattr(self, "_step_timing_history", None)
        if not isinstance(history, list):
            history = []
            self._step_timing_history = history
        history.append(item)
        max_history = max(int(getattr(self, "_step_timing_max_history", 128) or 128), 1)
        if len(history) > max_history:
            del history[: len(history) - max_history]

        first = history[0]
        warmup = max(int(getattr(self, "_step_timing_steady_warmup_steps", 1) or 1), 0)
        steady = history[warmup:] if len(history) > warmup else []
        steady_seconds = [float(row.get("step_wall_seconds", 0.0) or 0.0) for row in steady]
        steady_samples = sum(int(row.get("samples_seen", 0) or 0) for row in steady)
        steady_total = sum(steady_seconds)
        sorted_seconds = sorted(steady_seconds)
        median = 0.0
        if sorted_seconds:
            mid = len(sorted_seconds) // 2
            if len(sorted_seconds) % 2:
                median = sorted_seconds[mid]
            else:
                median = (sorted_seconds[mid - 1] + sorted_seconds[mid]) / 2.0
        summary = {
            "profile": "lulynx_step_timing_window_v0",
            "observed_steps": len(history),
            "window": max_history,
            "steady_warmup_steps": warmup,
            "first_step_ms": round(float(first.get("step_wall_seconds", 0.0) or 0.0) * 1000.0, 4),
            "last_step_ms": round(step_seconds * 1000.0, 4),
            "steady_steps": len(steady_seconds),
            "steady_mean_step_ms": round((steady_total / len(steady_seconds)) * 1000.0, 4)
            if steady_seconds
            else 0.0,
            "steady_median_step_ms": round(median * 1000.0, 4) if steady_seconds else 0.0,
            "steady_total_seconds": round(steady_total, 6),
            "samples_seen": steady_samples,
            "samples_per_second": round(steady_samples / steady_total, 4) if steady_total > 0.0 else 0.0,
            "samples_source": source,
            "sync_cuda": False,
        }
        self._last_step_timing_window = summary
        return summary

    @staticmethod
    def _transfer_profile_recommendation(step_share: float) -> str:
        if step_share < 0.01:
            return "H2D transfer below 1%; async prefetch is unlikely to help"
        if step_share < 0.05:
            return "H2D transfer is visible; tune cached DataLoader and non_blocking before async prefetch"
        return "H2D transfer exceeds 5%; async prefetch / batch prepare is worth testing"
    def _cudagraph_eligible(self) -> bool:
        """Check if CUDA graph capture is possible for this training loop."""
        if self._model_arch not in {"anima", "newbie"}:
            return False
        if self._block_offloader is not None:
            return False
        if self._module_offload_manager is not None:
            return False
        if self.cpu_offload_checkpointing:
            return False
        if self.safe_fallback:
            return False
        if self._torch_compile_active:
            return False
        if not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"):
            return False
        return True

    def _try_init_cudagraph(self, unet_kwargs: Dict[str, Any]) -> bool:
        """Attempt to warmup and capture a CUDA graph for the UNet forward pass.

        Returns True if capture succeeded, False otherwise.  On success,
        ``self._cudagraph_active`` is set to True and subsequent steps use
        ``replay()`` instead of a full forward pass.

        Only called when ``anima_compile_scope == "full_cudagraph"`` and
        fixed token counts are configured (``anima_fixed_text_tokens`` /
        ``anima_fixed_visual_tokens``).
        """
        if not self._cudagraph_eligible():
            logger.info("[CUDAGraph] Not eligible — skipping capture")
            return False

        from .cudagraph_capture import CUDAGraphCapture, cudagraph_available

        if not cudagraph_available():
            logger.info("[CUDAGraph] CUDA graphs not available on this system")
            return False

        try:
            # Build sample inputs matching what unet_kwargs will look like
            # during training.  Shapes must be static for the graph.
            sample_inputs = {}
            for k, v in unet_kwargs.items():
                if isinstance(v, torch.Tensor):
                    sample_inputs[k] = torch.zeros_like(v)
                elif isinstance(v, dict):
                    sample_inputs[k] = {
                        dk: torch.zeros_like(dv) if isinstance(dv, torch.Tensor) else dv
                        for dk, dv in v.items()
                    }
                else:
                    sample_inputs[k] = v

            capture = CUDAGraphCapture(self.unet, sample_inputs, device=self.device)
            capture.warmup(num_steps=3)
            capture.capture()

            self._cudagraph_capture = capture
            self._cudagraph_active = True
            logger.info("[CUDAGraph] Capture successful — forward pass will use graph replay")
            return True

        except Exception as exc:
            logger.warning("[CUDAGraph] Capture failed: %s — falling back to eager", exc)
            self._cudagraph_capture = None
            self._cudagraph_active = False
            return False

    def _cudagraph_replay(self, unet_kwargs: Dict[str, Any]):
        """Replay the captured CUDA graph with new inputs.

        Returns the model output (same as self.unet(**unet_kwargs)).sample.
        """
        if self._cudagraph_capture is None or not self._cudagraph_active:
            return None
        output = self._cudagraph_capture.replay(unet_kwargs)
        # Output shape matches what unet() returns — typically a
        # UNet2DOutput or similar with a .sample attribute.
        return output


__all__ = ["TrainingLoopRuntimeMemoryMixin"]
