# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Bubble closed-loop runtime cluster extracted verbatim from ``trainer.py`` as a mixin.

These methods (bubble closed-loop step-sample recording, window profiling,
runtime-mutation + config-overlay application, the closed-loop driver, and the
epoch-boundary dataloader-rebuild machinery it triggers) run as bound methods of
the trainer instance — identical ``self`` semantics, identical call sites.
Behaviour is unchanged; this split only keeps ``trainer.py`` navigable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .bubble_runtime_closed_loop_executor import (
    mark_closed_loop_action_applied,
    mark_closed_loop_action_closed,
)
from .bubble_runtime_controller import build_bubble_controller_report
from .dataloader_rebuild_runtime import (
    build_dataloader_rebuild_readiness_profile,
    rebuild_dataloader_from_plan,
)
from .runtime_feature_snapshot import build_lulynx_trainer_runtime_features
from .training_loop import TrainingLoop

logger = logging.getLogger(__name__)


class TrainerBubbleRuntimeMixin:
    def _record_bubble_closed_loop_step_sample(self, step: int, loss: float, info: Dict[str, Any]) -> None:
        if not bool(getattr(self.config, "bubble_controller_enabled", False)):
            return
        wall_seconds = float(info.get("step_wall_seconds", 0.0) or 0.0)
        if wall_seconds <= 0.0:
            return
        batch = max(int(getattr(self.config, "train_batch_size", 1) or 1), 1)
        sample = {
            "step": int(step),
            "step_wall_seconds": wall_seconds,
            "samples_per_second": batch / max(wall_seconds, 1e-9),
            "loss": float(loss),
        }
        self._bubble_closed_loop_step_window.append(sample)
        tune_interval = max(int(getattr(self.config, "bubble_controller_tune_interval_steps", 32) or 32), 1)
        max_window = max(tune_interval * 4, 64)
        if len(self._bubble_closed_loop_step_window) > max_window:
            self._bubble_closed_loop_step_window = self._bubble_closed_loop_step_window[-max_window:]

    def _bubble_closed_loop_window_profile(self) -> Dict[str, Any]:
        samples = list(getattr(self, "_bubble_closed_loop_step_window", []) or [])
        if not samples:
            return {}
        tune_interval = max(int(getattr(self.config, "bubble_controller_tune_interval_steps", 32) or 32), 1)
        window = samples[-tune_interval:]
        mean_wall = sum(float(item.get("step_wall_seconds", 0.0) or 0.0) for item in window) / max(len(window), 1)
        mean_sps = sum(float(item.get("samples_per_second", 0.0) or 0.0) for item in window) / max(len(window), 1)
        return {
            "schema_version": 1,
            "profile": "bubble_closed_loop_step_window_v0",
            "step_count": len(window),
            "first_step": int(window[0].get("step", 0) or 0),
            "last_step": int(window[-1].get("step", 0) or 0),
            "mean_step_ms": round(mean_wall * 1000.0, 4),
            "steady_samples_per_second": round(mean_sps, 6),
            "throughput_estimated": False,
            "final_loss": round(float(window[-1].get("loss", 0.0) or 0.0), 6),
        }

    def _apply_bubble_runtime_mutations(self, mutations: List[Dict[str, Any]], *, reason: str) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        loop = getattr(self, "training_loop", None)
        for mutation in mutations:
            path = str(mutation.get("path") or "")
            value = mutation.get("recommended")
            if not path:
                continue
            setattr(self.config, path, value)
            if path == "tensorboard_flush_interval_steps":
                self._tb_flush_interval_steps = max(int(value or 1), 1)
            elif path == "adaptive_step_logging_enabled":
                self._adaptive_step_logging_enabled = bool(value)
            elif path == "layer_monitor_interval" and loop is not None:
                loop._layer_monitor_interval = max(int(value or 1), 1)
            elif path == "eval_every_n_steps" and loop is not None:
                loop.eval_every_n_steps = max(int(value or 0), 0)
            elif path == "data_transfer_non_blocking" and loop is not None:
                loop.data_transfer_non_blocking = bool(value)
            elif path == "data_transfer_profile_mode" and loop is not None:
                loop.data_transfer_profile_mode = TrainingLoop._normalize_data_transfer_profile_mode(str(value or "event"))
            elif path == "step_phase_profile_enabled" and loop is not None:
                profiler = getattr(loop, "_step_phase_profiler", None)
                if profiler is not None:
                    profiler.enabled = bool(value)
            applied[path] = value
        if applied:
            self._log(f"[BubbleController] {reason}: applied runtime overlay {applied}")
        return applied

    def _maybe_run_bubble_closed_loop(self, step: int, loss: float, info: Dict[str, Any]) -> None:
        if not bool(getattr(self.config, "bubble_controller_enabled", False)):
            return
        if str(getattr(self.config, "bubble_controller_mode", "report_only") or "report_only").strip().lower() != "auto_apply":
            return
        tune_interval = max(int(getattr(self.config, "bubble_controller_tune_interval_steps", 32) or 32), 1)
        warmup = max(int(getattr(self.config, "bubble_controller_warmup_steps", 8) or 8), 0)
        state = getattr(self, "_bubble_closed_loop_state", {}) or {}
        active = state.get("active_action") if isinstance(state, dict) else None
        if not active and (int(step) < warmup or int(step) % tune_interval != 0):
            return
        if active:
            cooldown_until = int(active.get("cooldown_until_step", int(step)) or int(step))
            if int(step) < cooldown_until and int(step) % tune_interval != 0:
                return
        try:
            features = build_lulynx_trainer_runtime_features(self)
            report = build_bubble_controller_report(
                self.config,
                runtime_features=features,
                closed_loop_state=state,
                current_step=int(step),
            )
            self._bubble_closed_loop_last_report = dict(report)
            closed_loop = report.get("closed_loop", {}) if isinstance(report, dict) else {}
            executor = closed_loop.get("executor", {}) if isinstance(closed_loop, dict) else {}
            status = str(executor.get("status") or "")
            if status == "ready_to_apply":
                runtime_apply = executor.get("runtime_apply", {}) if isinstance(executor, dict) else {}
                mutations = [
                    dict(item)
                    for item in runtime_apply.get("mutations", [])
                    if isinstance(item, dict)
                ]
                applied = self._apply_bubble_runtime_mutations(mutations, reason="auto-apply low-risk action")
                self._bubble_closed_loop_state = mark_closed_loop_action_applied(
                    executor,
                    current_step=int(step),
                    applied_overlay=applied,
                )
            elif status in {
                "dataloader_rebuild_epoch_boundary_ready",
                "dataloader_rebuild_rollback_epoch_boundary_ready",
            }:
                self._bubble_dataloader_epoch_pending = dict(executor)
                self._log(
                    "[BubbleController] DataLoader rebuild queued for epoch boundary: "
                    f"{status}"
                )
            elif status == "rollback_recommended":
                rollback = executor.get("rollback", {}) if isinstance(executor, dict) else {}
                mutations = [
                    dict(item)
                    for item in rollback.get("mutations", [])
                    if isinstance(item, dict)
                ]
                applied = self._apply_bubble_runtime_mutations(mutations, reason="rollback low-risk action")
                self._bubble_closed_loop_state = mark_closed_loop_action_closed(
                    executor,
                    status="rolled_back" if applied else "rollback_failed",
                    current_step=int(step),
                    applied_overlay=applied,
                )
            elif status in {"keep_recommended", "keep_observed", "needs_more_evidence"}:
                closed_status = "kept" if status != "needs_more_evidence" else "needs_more_evidence"
                self._bubble_closed_loop_state = mark_closed_loop_action_closed(
                    executor,
                    status=closed_status,
                    current_step=int(step),
                )
        except Exception as exc:
            logger.debug("Bubble closed-loop step skipped: %s", exc)

    def _maybe_apply_bubble_epoch_boundary_dataloader_rebuild(self, dataloader: Any, *, epoch: int) -> Any:
        if not bool(getattr(self.config, "bubble_controller_enabled", False)):
            return dataloader
        mode = str(getattr(self.config, "bubble_controller_mode", "report_only") or "report_only").strip().lower()
        if mode.replace("-", "_") != "auto_apply":
            return dataloader
        if getattr(self, "training_loop", None) is None:
            return dataloader
        step = int(getattr(self.training_loop, "global_step", 0) or 0)
        try:
            self._refresh_dataloader_rebuild_readiness(dataloader, epoch=epoch, boundary="epoch_start")
            executor = dict(getattr(self, "_bubble_dataloader_epoch_pending", {}) or {})
            if not executor:
                state = getattr(self, "_bubble_closed_loop_state", {}) or {}
                active = state.get("active_action") if isinstance(state, dict) else None
                tune_interval = max(int(getattr(self.config, "bubble_controller_tune_interval_steps", 32) or 32), 1)
                warmup = max(int(getattr(self.config, "bubble_controller_warmup_steps", 8) or 8), 0)
                if not active and (step < warmup or step % tune_interval != 0):
                    return dataloader
                features = build_lulynx_trainer_runtime_features(self)
                report = build_bubble_controller_report(
                    self.config,
                    runtime_features=features,
                    closed_loop_state=state,
                    current_step=step,
                )
                self._bubble_closed_loop_last_report = dict(report)
                closed_loop = report.get("closed_loop", {}) if isinstance(report, dict) else {}
                executor = dict(closed_loop.get("executor", {}) if isinstance(closed_loop, dict) else {})
            status = str(executor.get("status") or "")
            if status == "dataloader_rebuild_epoch_boundary_ready":
                return self._apply_bubble_dataloader_rebuild_action(
                    dataloader,
                    executor=executor,
                    epoch=epoch,
                    step=step,
                    target="next",
                )
            if status == "dataloader_rebuild_rollback_epoch_boundary_ready":
                return self._apply_bubble_dataloader_rebuild_action(
                    dataloader,
                    executor=executor,
                    epoch=epoch,
                    step=step,
                    target="rollback",
                )
        except Exception as exc:
            logger.debug("Bubble DataLoader epoch-boundary action skipped: %s", exc)
        return dataloader

    def _apply_bubble_dataloader_rebuild_action(
        self,
        dataloader: Any,
        *,
        executor: Dict[str, Any],
        epoch: int,
        step: int,
        target: str,
    ) -> Any:
        is_rollback = target == "rollback"
        runtime_apply = executor.get("runtime_apply", {}) if isinstance(executor, dict) else {}
        rollback = executor.get("rollback", {}) if isinstance(executor, dict) else {}
        rebuild_info = rollback.get("dataloader_rebuild") if is_rollback else runtime_apply.get("dataloader_rebuild")
        if not isinstance(rebuild_info, dict) or not rebuild_info:
            rebuild_info = executor.get("dataloader_rebuild", {}) if isinstance(executor, dict) else {}
        plan = rebuild_info.get("runtime_rebuild_plan") if isinstance(rebuild_info, dict) else None
        if not isinstance(plan, dict) or not plan:
            return dataloader

        result = rebuild_dataloader_from_plan(dataloader, plan, target=target)
        if not result.get("ok"):
            self._mark_bubble_dataloader_rebuild_failed(executor, step=step, result=result, target=target)
            self._bubble_dataloader_epoch_pending = {}
            self._log(
                "[BubbleController] DataLoader "
                f"{'rollback' if is_rollback else 'rebuild'} failed at epoch {epoch + 1}: "
                f"{result.get('reason')}"
            )
            return dataloader

        rebuilt = self._replace_runtime_dataloader(result.get("dataloader"), epoch=epoch)
        overlay = rollback.get("restore") if is_rollback else runtime_apply.get("applied_overlay")
        applied = self._apply_bubble_config_overlay(
            overlay if isinstance(overlay, dict) else {},
            reason="rollback DataLoader rebuild" if is_rollback else "auto-apply DataLoader rebuild",
        )
        self._refresh_dataloader_rebuild_readiness(rebuilt, epoch=epoch, boundary="epoch_start")
        if is_rollback:
            self._bubble_closed_loop_state = mark_closed_loop_action_closed(
                executor,
                status="rolled_back",
                current_step=step,
                applied_overlay=applied,
            )
        else:
            self._bubble_closed_loop_state = mark_closed_loop_action_applied(
                executor,
                current_step=step,
                applied_overlay=applied,
            )
        self._log(
            "[BubbleController] DataLoader "
            f"{'rollback' if is_rollback else 'rebuild'} applied at epoch {epoch + 1}: "
            f"{result.get('rebuilt_descriptor', {})}"
        )
        self._bubble_dataloader_epoch_pending = {}
        return rebuilt

    def _apply_bubble_config_overlay(self, overlay: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        for path, value in overlay.items():
            path_text = str(path or "")
            if not path_text:
                continue
            setattr(self.config, path_text, value)
            applied[path_text] = value
        if applied:
            self._log(f"[BubbleController] {reason}: applied config overlay {applied}")
        return applied

    def _replace_runtime_dataloader(self, dataloader: Any, *, epoch: int) -> Any:
        if dataloader is None:
            return dataloader
        wrapper = getattr(self, "_ddp_wrapper", None)
        if wrapper is not None and getattr(self, "_dataset", None) is not None:
            try:
                from .distributed import wrap_dataloader_for_ddp

                dataloader = wrap_dataloader_for_ddp(
                    dataloader,
                    self._dataset,
                    shuffle=True,
                    seed=int(getattr(self.config, "seed", 42) or 42),
                )
            except Exception as exc:
                self._log(f"[BubbleController] DDP dataloader rewrap skipped: {exc}")
        if wrapper is not None:
            try:
                wrapper._dataloader = dataloader
                wrapper._ddp_sampler = getattr(dataloader, "sampler", None)
                wrapper.set_epoch(epoch)
            except Exception as exc:
                self._log(f"[BubbleController] DDP dataloader pointer refresh skipped: {exc}")
        self._dataloader = dataloader
        return dataloader

    def _mark_bubble_dataloader_rebuild_failed(
        self,
        executor: Dict[str, Any],
        *,
        step: int,
        result: Dict[str, Any],
        target: str,
    ) -> None:
        if target == "rollback":
            self._bubble_closed_loop_state = mark_closed_loop_action_closed(
                executor,
                status="rollback_failed",
                current_step=step,
                applied_overlay={"dataloader_rebuild_error": result.get("reason")},
            )
            return
        pending = mark_closed_loop_action_applied(executor, current_step=step, applied_overlay={})
        failed_executor = {
            **executor,
            "active_action": pending.get("active_action", {}),
            "action_history": pending.get("action_history", []),
            "evaluation": {
                "apply_error": {
                    "target": target,
                    "reason": result.get("reason"),
                    "error": result.get("error"),
                }
            },
        }
        self._bubble_closed_loop_state = mark_closed_loop_action_closed(
            failed_executor,
            status="apply_failed",
            current_step=step,
            applied_overlay={"dataloader_rebuild_error": result.get("reason")},
        )

    def _refresh_dataloader_rebuild_readiness(self, dataloader: Any, *, epoch: int, boundary: str) -> None:
        try:
            self._dataloader_rebuild_readiness_profile = build_dataloader_rebuild_readiness_profile(
                self,
                dataloader=dataloader,
                safe_boundary=boundary,
                current_epoch=epoch,
            )
        except Exception as exc:
            self._dataloader_rebuild_readiness_profile = {
                "profile": "dataloader_rebuild_readiness_v0",
                "current_run_rebuild_ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


__all__ = ["TrainerBubbleRuntimeMixin"]
