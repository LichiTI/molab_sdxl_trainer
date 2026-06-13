# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Logging-runtime cluster extracted verbatim from ``trainer.py`` as a mixin.

These methods (run-dir resolution, logging-runtime init/finalize, step/epoch log
writers, step-logging gate + overhead accounting) run as bound methods of the
trainer instance — identical ``self`` semantics, identical call sites. Behaviour
is unchanged; this split only keeps ``trainer.py`` navigable.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TrainerLoggingRuntimeMixin:
    def _resolve_logging_run_dir(self) -> Optional[Path]:
        raw_dir = str(getattr(self.config, "logging_dir", "") or "").strip()
        if not raw_dir:
            return None
        base = Path(raw_dir)
        prefix = str(getattr(self.config, "log_prefix", "") or "").strip()
        if prefix:
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            return base / f"{prefix}{timestamp}"
        return base

    def _initialize_logging_runtime(self) -> None:
        self._tb_writer = None
        self._tb_log_dir = None
        self._wandb_enabled = False

        log_with = str(getattr(self.config, "log_with", "") or "").strip().lower()
        if not log_with:
            return

        run_dir = self._resolve_logging_run_dir()
        if log_with == "tensorboard":
            if run_dir is None:
                self._log("Logging requested with tensorboard, but logging_dir is empty; skipping tracker init.")
                return
            try:
                from torch.utils.tensorboard import SummaryWriter

                run_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(run_dir))
                self._tb_log_dir = run_dir
                self._tb_flush_interval_steps = max(
                    1,
                    int(getattr(self.config, "tensorboard_flush_interval_steps", 10) or 10),
                )
                self._log(f"TensorBoard logging initialized: {run_dir}")
            except Exception as exc:
                self._tb_writer = None
                self._tb_log_dir = None
                self._log(f"TensorBoard logging unavailable: {exc}")
            return

        if log_with == "wandb":
            api_key = str(getattr(self.config, "wandb_api_key", "") or "").strip()
            run_name = str(getattr(self.config, "wandb_run_name", "") or "").strip()
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
            if run_name:
                os.environ["WANDB_NAME"] = run_name
            self._wandb_enabled = True
            summary = "WandB logging requested"
            if api_key:
                summary += " (api key set)"
            if run_name:
                summary += f", run_name={run_name}"
            self._log(summary)
            return

        self._log(f"Unknown log_with={log_with}; skipping logging runtime init.")

    def _write_step_log(self, step: int, loss: float, lr: float, info: Optional[Dict] = None) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/train", float(loss), int(step))
            self._tb_writer.add_scalar("lr", float(lr), int(step))
            if info is not None:
                stages = info.get("peak_vram_stages")
                if stages:
                    self._tb_writer.add_scalar("vram/peak_forward_mb", stages["forward_mb"], int(step))
                    self._tb_writer.add_scalar("vram/peak_backward_mb", stages["backward_mb"], int(step))
                    self._tb_writer.add_scalar("vram/peak_optimizer_mb", stages["optimizer_mb"], int(step))
                diagnostics = info.get("peak_vram_diagnostics")
                if diagnostics:
                    if diagnostics.get("max_reserved_mb") is not None:
                        self._tb_writer.add_scalar("vram/max_reserved_mb", diagnostics["max_reserved_mb"], int(step))
                    if diagnostics.get("max_allocated_mb") is not None:
                        self._tb_writer.add_scalar("vram/max_allocated_mb", diagnostics["max_allocated_mb"], int(step))
                    if diagnostics.get("allocator_cache_gap_mb") is not None:
                        self._tb_writer.add_scalar("vram/allocator_cache_gap_mb", diagnostics["allocator_cache_gap_mb"], int(step))
                icu = info.get("icu_score")
                if icu is not None:
                    self._tb_writer.add_scalar("health/icu_score", int(icu), int(step))
                attn_ent = info.get("attn_entropy")
                if attn_ent is not None:
                    self._tb_writer.add_scalar("health/attn_entropy", float(attn_ent), int(step))
                act_drift = info.get("act_drift")
                if act_drift:
                    for layer_name, drifts in act_drift.items():
                        short = layer_name.split(".")[-1] if "." in layer_name else layer_name
                        self._tb_writer.add_scalar(f"drift/{short}/mean", drifts.get("mean_drift", 0), int(step))
                        self._tb_writer.add_scalar(f"drift/{short}/std", drifts.get("std_drift", 0), int(step))
                loss_mods = info.get("loss_modifiers")
                if loss_mods:
                    self._tb_writer.add_scalar("loss/active_modifiers", len(loss_mods), int(step))
                grad_stats = info.get("grad_stats")
                if grad_stats:
                    self._tb_writer.add_scalar("grad/norm", grad_stats["norm"], int(step))
                    self._tb_writer.add_scalar("grad/norm_ema", grad_stats["norm_ema"], int(step))
                    self._tb_writer.add_scalar("grad/norm_var", grad_stats["norm_var"], int(step))
                    self._tb_writer.add_scalar("grad/fisher_diag", grad_stats["fisher_diag"], int(step))
                    if grad_stats.get("cosine_sim") is not None:
                        self._tb_writer.add_scalar("grad/cosine_sim", grad_stats["cosine_sim"], int(step))
                hess = info.get("hessian_trace")
                if hess is not None:
                    self._tb_writer.add_scalar("health/hessian_trace", float(hess), int(step))
                forgetting = info.get("forgetting")
                if forgetting:
                    self._tb_writer.add_scalar("health/forgetting_score", forgetting.get("score", 100), int(step))
                    self._tb_writer.add_scalar("health/forgetting_ratio", forgetting.get("ratio", 1.0), int(step))
            flush_interval = max(1, int(getattr(self, "_tb_flush_interval_steps", 10) or 10))
            if int(step) % flush_interval == 0:
                self._tb_writer.flush()

    def _write_epoch_log(self, epoch: int, avg_loss: float) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/epoch", float(avg_loss), int(epoch) + 1)
            self._tb_writer.flush()

    def _should_emit_step_logging(self, step: int) -> bool:
        interval = max(1, int(getattr(self, "_step_logging_interval", 1) or 1))
        return step <= 1 or step % interval == 0

    def _record_step_logging_overhead(self, step: int, overhead_seconds: float, step_wall_seconds: float) -> None:
        if not getattr(self, "_adaptive_step_logging_enabled", True):
            return
        if step_wall_seconds <= 0.0 or overhead_seconds < 0.0:
            return

        self._step_logging_profile_steps += 1
        self._step_logging_profile_total += float(step_wall_seconds)
        self._step_logging_profile_overhead += float(overhead_seconds)

        window = max(1, int(getattr(self, "_step_logging_window", 50) or 50))
        if self._step_logging_profile_steps < window:
            return

        total = max(self._step_logging_profile_total, 1e-9)
        ratio = self._step_logging_profile_overhead / total
        threshold = max(0.0, float(getattr(self, "_step_logging_threshold", 0.01) or 0.01))
        interval = max(1, int(getattr(self, "_step_logging_interval", 1) or 1))
        max_interval = max(1, int(getattr(self, "_step_logging_max_interval", 64) or 64))

        if ratio > threshold and interval < max_interval:
            new_interval = min(interval * 2, max_interval)
            self._step_logging_interval = new_interval
            self._tb_flush_interval_steps = max(int(getattr(self, "_tb_flush_interval_steps", 10) or 10), new_interval)
            self._log(
                "[adaptive-logging] step logging overhead %.2f%% exceeded %.2f%% over %d logged step(s); "
                "reducing step log frequency to every %d optimizer step(s)."
                % (ratio * 100.0, threshold * 100.0, self._step_logging_profile_steps, new_interval)
            )

        self._step_logging_profile_steps = 0
        self._step_logging_profile_total = 0.0
        self._step_logging_profile_overhead = 0.0

    def _finalize_logging_runtime(self) -> None:
        writer = self._tb_writer
        self._tb_writer = None
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception as exc:
                logger.debug("TensorBoard writer close failed: %s", exc)
        self._tb_log_dir = None


__all__ = ["TrainerLoggingRuntimeMixin"]
