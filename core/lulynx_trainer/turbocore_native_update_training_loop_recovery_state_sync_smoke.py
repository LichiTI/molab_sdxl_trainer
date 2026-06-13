"""Smoke check for TrainingLoop native-update recovery fallback state sync."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.lulynx_trainer.training_loop import TrainingLoop  # noqa: E402
from core.lulynx_trainer.trainer import LulynxTrainer  # noqa: E402
from turbocore_native_update_training_loop_dispatch_smoke import (  # noqa: E402
    _checkpoint_resume_summary,
    _make_loop,
)


def _fake_direct_grad_train_step(
    self: TrainingLoop,
    _batch: dict,
    accumulation_steps: int = 1,
    return_loss_tensor: bool = False,
):
    params = self._get_trainable_params()
    for param in params:
        if param.grad is not None:
            param.grad = None
    self._turbocore_native_update_direct_grad_lifecycle_report = (
        self._prepare_turbocore_native_update_direct_grad_executor_before_backward(params)
    )
    if self._turbocore_update_shadow.enabled and bool(self._turbocore_update_shadow.config.direct_grad):
        self._turbocore_direct_grad_lifecycle_report = self._turbocore_update_shadow.prepare_before_backward(
            params,
            optimizer=self.optimizer,
            max_grad_norm=self.max_grad_norm,
            step=self.global_step,
            reset_owner_grad=bool(getattr(self, "_current_accumulation_group_start", True)),
        )
    loss = sum((param * param).sum() for param in params) / max(int(accumulation_steps or 1), 1)
    loss.backward()
    return loss.detach() if return_loss_tensor else float(loss.detach().item())


def _run_recovery_loop() -> tuple[Any | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        return None, [], []
    loop = _make_loop(direct_grad=True)
    loop.total_steps = 3
    captured: list[dict[str, Any]] = []
    sync_reports: list[dict[str, Any]] = []
    fault_installed = {"value": False}
    original_sync = loop._sync_turbocore_native_update_training_executor_to_pytorch

    def _record_sync(reason: str) -> dict[str, Any]:
        report = original_sync(reason)
        sync_reports.append(dict(report))
        return report

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict,
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ):
        params = self._get_trainable_params()
        for param in params:
            if param.grad is not None:
                param.grad = None
        self._turbocore_native_update_direct_grad_lifecycle_report = (
            self._prepare_turbocore_native_update_direct_grad_executor_before_backward(params)
        )
        if self._turbocore_update_shadow.enabled and bool(self._turbocore_update_shadow.config.direct_grad):
            self._turbocore_direct_grad_lifecycle_report = self._turbocore_update_shadow.prepare_before_backward(
                params,
                optimizer=self.optimizer,
                max_grad_norm=self.max_grad_norm,
                step=self.global_step,
                reset_owner_grad=bool(getattr(self, "_current_accumulation_group_start", True)),
            )
        if int(self.global_step) >= 1 and not fault_installed["value"]:
            _install_native_failure(self)
            fault_installed["value"] = True
        loss = sum((param * param).sum() for param in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop._sync_turbocore_native_update_training_executor_to_pytorch = _record_sync  # type: ignore[method-assign]
    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    from unittest.mock import patch

    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}, {}, {}], 0)
    assert result["steps"] == 3, result
    assert len(captured) == 3, captured
    for after in loop._get_trainable_params():
        assert torch.isfinite(after.detach()).all(), after
    return loop, captured, sync_reports


def _run_deferred_sync_recovery_loop() -> tuple[Any | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        return None, [], []
    loop = _make_loop(direct_grad=True)
    loop._turbocore_native_update_defer_state_sync = True
    loop.total_steps = 4
    captured: list[dict[str, Any]] = []
    sync_reports: list[dict[str, Any]] = []
    fault_installed = {"value": False}
    original_sync = loop._sync_turbocore_native_update_training_executor_to_pytorch

    def _record_sync(reason: str) -> dict[str, Any]:
        report = original_sync(reason)
        sync_reports.append(dict(report))
        return report

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict,
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ):
        params = self._get_trainable_params()
        for param in params:
            if param.grad is not None:
                param.grad = None
        self._turbocore_native_update_direct_grad_lifecycle_report = (
            self._prepare_turbocore_native_update_direct_grad_executor_before_backward(params)
        )
        if self._turbocore_update_shadow.enabled and bool(self._turbocore_update_shadow.config.direct_grad):
            self._turbocore_direct_grad_lifecycle_report = self._turbocore_update_shadow.prepare_before_backward(
                params,
                optimizer=self.optimizer,
                max_grad_norm=self.max_grad_norm,
                step=self.global_step,
                reset_owner_grad=bool(getattr(self, "_current_accumulation_group_start", True)),
            )
        if int(self.global_step) >= 2 and not fault_installed["value"]:
            _install_native_failure(self)
            fault_installed["value"] = True
        loss = sum((param * param).sum() for param in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop._sync_turbocore_native_update_training_executor_to_pytorch = _record_sync  # type: ignore[method-assign]
    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    from unittest.mock import patch

    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}, {}, {}, {}], 0)
    assert result["steps"] == 4, result
    assert len(captured) == 4, captured
    return loop, captured, sync_reports


def _run_deferred_sync_epoch_flush_loop() -> tuple[Any | None, Any | None, list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        return None, None, []
    loop = _make_loop(direct_grad=True)
    loop._turbocore_native_update_defer_state_sync = True
    loop.total_steps = 2
    captured: list[dict[str, Any]] = []

    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    from unittest.mock import patch

    with patch.object(TrainingLoop, "train_step", new=_fake_direct_grad_train_step):
        result = loop.train_epoch([{}, {}], 0)
    assert result["steps"] == 2, result
    assert len(captured) == 2, captured
    executor = getattr(loop, "_turbocore_native_update_training_executor", None)
    assert executor is not None, "deferred native executor should remain open after epoch finalization"
    return loop, executor, captured


def _run_deferred_sync_epoch_failure_loop() -> tuple[Any | None, Any | None, list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        return None, None, []
    loop = _make_loop(direct_grad=True)
    loop._turbocore_native_update_defer_state_sync = True
    loop.total_steps = 2
    captured: list[dict[str, Any]] = []
    sync_failure_installed = {"value": False}

    def _capture_and_install_sync_failure(_step: int, _loss: float, info: dict[str, Any]) -> None:
        captured.append(info)
        runtime = info.get("turbocore_native_update_dispatch_runtime", {})
        if not bool(runtime.get("native_step_executed", False)) or sync_failure_installed["value"]:
            return
        executor = getattr(loop, "_turbocore_native_update_training_executor", None)
        assert executor is not None, "native executor should exist after deferred native step"

        def _fail_sync(*, reason: str = "manual_sync") -> Any:
            raise RuntimeError(f"smoke_injected_optimizer_state_sync_failure:{reason}")

        executor.sync_optimizer_state_to_pytorch = _fail_sync
        sync_failure_installed["value"] = True

    loop.on_step_end = _capture_and_install_sync_failure
    from unittest.mock import patch

    with patch.object(TrainingLoop, "train_step", new=_fake_direct_grad_train_step):
        result = loop.train_epoch([{}, {}], 0)
    assert result["steps"] == 2, result
    assert len(captured) == 2, captured
    executor = getattr(loop, "_turbocore_native_update_training_executor", None)
    assert executor is not None, "failed sync should leave executor open for defensive close/recovery"
    return loop, executor, captured


def _run_deferred_sync_state_save_loop() -> tuple[Any | None, dict[str, Any], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        return None, {}, []
    loop = _make_loop(direct_grad=True)
    loop._turbocore_native_update_defer_state_sync = True
    loop.total_steps = 2
    captured: list[dict[str, Any]] = []
    sync_reports: list[dict[str, Any]] = []
    saved: dict[str, Any] = {}
    original_sync = loop._sync_turbocore_native_update_training_executor_to_pytorch

    def _record_sync(reason: str) -> dict[str, Any]:
        report = original_sync(reason)
        sync_reports.append(dict(report))
        return report

    with tempfile.TemporaryDirectory(prefix="lulynx-turbocore-state-save-") as tmp_dir:
        trainer = LulynxTrainer(
            SimpleNamespace(
                output_dir=tmp_dir,
                output_name="deferred-native",
                model_arch="sd15",
                model_type="sd15",
            )
        )
        trainer.training_loop = loop

        def _capture_and_save(_step: int, _loss: float, info: dict[str, Any]) -> None:
            captured.append(info)
            runtime = info.get("turbocore_native_update_dispatch_runtime", {})
            if not bool(runtime.get("native_step_executed", False)) or saved:
                return
            executor = getattr(loop, "_turbocore_native_update_training_executor", None)
            assert executor is not None, "native executor should exist before state-save sync"
            assert getattr(executor, "_pytorch_optimizer_state_dirty", False) is True, executor
            trainer._save_state(epoch=1, step=int(getattr(loop, "global_step", 0) or 0))
            paths = sorted(Path(tmp_dir).glob("*-state.pt"))
            assert len(paths) == 1, paths
            state = torch.load(paths[0], map_location="cpu", weights_only=False)
            saved.update(
                {
                    "path": str(paths[0]),
                    "state": state,
                    "optimizer_step": _optimizer_state_dict_step_index(state["optimizer_state_dict"]),
                    "turbocore_update_state": state.get("turbocore_update_state") or {},
                    "executor_dirty_after_save": bool(getattr(executor, "_pytorch_optimizer_state_dirty", True)),
                }
            )

        loop._sync_turbocore_native_update_training_executor_to_pytorch = _record_sync  # type: ignore[method-assign]
        loop.on_step_end = _capture_and_save
        from unittest.mock import patch

        with patch.object(TrainingLoop, "train_step", new=_fake_direct_grad_train_step):
            result = loop.train_epoch([{}, {}], 0)
        assert result["steps"] == 2, result
        assert len(captured) == 2, captured
        return loop, dict(saved), sync_reports


def _install_native_failure(loop: TrainingLoop) -> None:
    executor = getattr(loop, "_turbocore_native_update_training_executor", None)
    assert executor is not None, "native update training executor must exist before injected failure"

    def _fail_once() -> Any:
        raise RuntimeError("smoke_injected_native_update_failure")

    executor.executor.step = _fail_once


def test_training_loop_native_failure_latches_pytorch_fallback() -> dict[str, int]:
    loop, captured, sync_reports = _run_recovery_loop()
    if not captured:
        return {"skipped_count": 1}
    warmup_runtime = captured[0]["turbocore_native_update_dispatch_runtime"]
    failure_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    post_latch_runtime = captured[2]["turbocore_native_update_dispatch_runtime"]
    failure_executor = failure_runtime["training_executor"]
    post_latch_reasons = set(post_latch_runtime["blocked_reasons"])
    failure_reasons = set(failure_runtime["blocked_reasons"])

    assert warmup_runtime["native_step_executed"] is False, warmup_runtime
    assert failure_executor["called"] is True, failure_runtime
    assert failure_executor["ok"] is False, failure_runtime
    assert failure_executor["reason"] == "native_update_training_executor_error", failure_runtime
    assert "smoke_injected_native_update_failure" in failure_executor["result"]["error"], failure_runtime
    assert failure_runtime["native_step_executed"] is False, failure_runtime
    assert failure_runtime["should_call_pytorch_optimizer_step"] is True, failure_runtime
    assert failure_runtime["fallback_to_pytorch_required"] is True, failure_runtime
    assert failure_runtime["state"]["disabled_for_run"] is True, failure_runtime
    assert failure_runtime["state"]["disable_reason"] == "native_update_training_executor_error", failure_runtime
    assert "native_update_training_executor_error" in failure_reasons, failure_runtime
    assert "native_dispatch_disabled_for_run" in failure_reasons, failure_runtime
    assert post_latch_runtime["native_step_executed"] is False, post_latch_runtime
    assert post_latch_runtime["should_call_pytorch_optimizer_step"] is True, post_latch_runtime
    assert post_latch_runtime["fallback_to_pytorch_required"] is True, post_latch_runtime
    assert post_latch_runtime["state"]["disabled_for_run"] is True, post_latch_runtime
    assert "native_dispatch_disabled_for_run" in post_latch_reasons, post_latch_runtime
    assert post_latch_runtime["training_executor"] == {}, post_latch_runtime
    assert len(sync_reports) >= 2, sync_reports
    assert all(report["requested_reason"] == "before_pytorch_optimizer_fallback" for report in sync_reports), sync_reports
    assert _optimizer_step_index(loop) == 3, loop.optimizer.state_dict()
    checkpoint = _checkpoint_resume_summary(loop)
    return {
        "skipped_count": 0,
        "failure_latched_count": 1,
        "failure_fallback_pytorch_step_count": 1,
        "post_latch_pytorch_fallback_count": 1,
        "fallback_state_sync_call_count": len(sync_reports),
        "optimizer_step_after_recovery_count": 1,
        **checkpoint,
        "training_path_enabled_count": 0,
    }


def test_training_loop_deferred_state_sync_flushes_before_recovery_fallback() -> dict[str, int]:
    loop, captured, sync_reports = _run_deferred_sync_recovery_loop()
    if not captured:
        return {"skipped_count": 1}
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    failure_runtime = captured[2]["turbocore_native_update_dispatch_runtime"]
    post_latch_runtime = captured[3]["turbocore_native_update_dispatch_runtime"]
    native_result = native_runtime["training_executor"]["result"]
    deferred_sync = native_result["optimizer_state_sync"]
    sync_flushes = [report for report in sync_reports if report.get("synced") is True]

    assert native_runtime["native_step_executed"] is True, native_runtime
    assert native_result["pytorch_optimizer_state_synced"] is False, native_result
    assert deferred_sync["reason"] == "pytorch_optimizer_state_sync_deferred", native_result
    assert failure_runtime["state"]["disabled_for_run"] is True, failure_runtime
    assert failure_runtime["should_call_pytorch_optimizer_step"] is True, failure_runtime
    assert post_latch_runtime["training_executor"] == {}, post_latch_runtime
    assert post_latch_runtime["state"]["disabled_for_run"] is True, post_latch_runtime
    assert sync_flushes, sync_reports
    assert sync_flushes[0]["requested_reason"] == "before_pytorch_optimizer_fallback", sync_reports
    assert int(sync_flushes[0]["step_index"]) >= 1, sync_reports
    assert _optimizer_step_index(loop) == 4, loop.optimizer.state_dict()
    checkpoint = _checkpoint_resume_summary(loop)
    return {
        "skipped_count": 0,
        "deferred_native_success_count": 1,
        "deferred_optimizer_state_sync_deferred_count": 1,
        "deferred_fallback_state_sync_flush_count": len(sync_flushes),
        "deferred_failure_latched_count": 1,
        "deferred_post_latch_pytorch_fallback_count": 1,
        **checkpoint,
        "training_path_enabled_count": 0,
    }


def test_training_loop_deferred_state_sync_flushes_on_epoch_finalization() -> dict[str, int]:
    loop, executor, captured = _run_deferred_sync_epoch_flush_loop()
    if not captured:
        return {"skipped_count": 1}
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    native_result = native_runtime["training_executor"]["result"]
    deferred_sync = native_result["optimizer_state_sync"]
    epoch_sync = getattr(executor, "_last_optimizer_state_sync", {})

    assert native_runtime["native_step_executed"] is True, native_runtime
    assert native_result["pytorch_optimizer_state_synced"] is False, native_result
    assert deferred_sync["reason"] == "pytorch_optimizer_state_sync_deferred", native_result
    assert getattr(executor, "_pytorch_optimizer_state_dirty", True) is False, executor
    assert epoch_sync["synced"] is True, epoch_sync
    assert epoch_sync["requested_reason"] == "epoch_finalization_deferred_state_sync", epoch_sync
    assert int(epoch_sync["step_index"]) == 2, epoch_sync
    assert _optimizer_step_index(loop) == 2, loop.optimizer.state_dict()
    loop._close_turbocore_native_update_training_executor()
    close_sync = getattr(executor, "_last_optimizer_state_sync", {})
    assert getattr(loop, "_turbocore_native_update_training_executor", None) is None, loop
    assert getattr(executor, "_pytorch_optimizer_state_dirty", True) is False, executor
    assert close_sync["requested_reason"] == "epoch_finalization_deferred_state_sync", close_sync
    assert int(close_sync["step_index"]) == 2, close_sync
    assert _optimizer_step_index(loop) == 2, loop.optimizer.state_dict()
    checkpoint = _checkpoint_resume_summary(loop)
    return {
        "skipped_count": 0,
        "deferred_epoch_state_sync_flush_count": 1,
        "deferred_epoch_state_sync_step_count": 1,
        "deferred_epoch_close_already_clean_count": 1,
        "deferred_epoch_optimizer_step_synced_count": 1,
        **checkpoint,
        "training_path_enabled_count": 0,
    }


def test_training_loop_deferred_state_sync_failure_latches_runtime() -> dict[str, int]:
    loop, executor, captured = _run_deferred_sync_epoch_failure_loop()
    if not captured:
        return {"skipped_count": 1}
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    native_result = native_runtime["training_executor"]["result"]
    runtime_state = loop._turbocore_native_update_dispatch_runtime.snapshot()

    assert native_runtime["native_step_executed"] is True, native_runtime
    assert native_result["pytorch_optimizer_state_synced"] is False, native_result
    assert native_result["optimizer_state_sync"]["reason"] == "pytorch_optimizer_state_sync_deferred", native_result
    assert getattr(executor, "_pytorch_optimizer_state_dirty", False) is True, executor
    assert runtime_state["disabled_for_run"] is True, runtime_state
    assert runtime_state["disable_reason"] == "native_update_optimizer_state_sync_error", runtime_state
    assert _optimizer_step_index(loop) == 1, loop.optimizer.state_dict()
    close_failed = False
    try:
        loop._close_turbocore_native_update_training_executor()
    except RuntimeError as exc:
        close_failed = "smoke_injected_optimizer_state_sync_failure" in str(exc)
        loop._turbocore_native_update_training_executor = None
    assert close_failed is True, "dirty executor close must not silently mark failed sync clean"
    return {
        "skipped_count": 0,
        "deferred_epoch_sync_failure_latched_count": 1,
        "deferred_epoch_sync_failure_dirty_state_retained_count": 1,
        "deferred_epoch_sync_failure_optimizer_step_stale_count": 1,
        "training_path_enabled_count": 0,
    }


def test_trainer_state_save_flushes_deferred_native_optimizer_state() -> dict[str, int]:
    loop, saved, sync_reports = _run_deferred_sync_state_save_loop()
    if not saved:
        return {"skipped_count": 1}
    save_sync = [report for report in sync_reports if report.get("requested_reason") == "before_state_save"]
    checkpoint_state = saved["turbocore_update_state"]
    checkpoint_contract = checkpoint_state.get("checkpoint_contract") or {}

    assert loop is not None
    assert save_sync, sync_reports
    assert save_sync[0]["synced"] is True, sync_reports
    assert int(save_sync[0]["step_index"]) == 2, sync_reports
    assert saved["optimizer_step"] == 2, saved
    assert saved["executor_dirty_after_save"] is False, saved
    assert checkpoint_state["checkpoint_metadata_integrated"] is True, checkpoint_state
    assert checkpoint_state["owner_state_included"] is True, checkpoint_state
    assert checkpoint_state["training_path_enabled"] is False, checkpoint_state
    assert checkpoint_contract["trainer_checkpoint_integration"] is True, checkpoint_contract
    assert checkpoint_contract["trainer_state_save_sync_verified"] is True, checkpoint_contract
    assert checkpoint_contract["resume_owner_state_guard_verified"] is True, checkpoint_contract
    assert _optimizer_step_index(loop) == 2, loop.optimizer.state_dict()
    return {
        "skipped_count": 0,
        "deferred_state_save_flush_count": 1,
        "deferred_state_save_optimizer_step_synced_count": 1,
        "deferred_state_save_owner_state_included_count": 1,
        "deferred_state_save_checkpoint_contract_integrated_count": 1,
        "training_path_enabled_count": 0,
    }


def _optimizer_step_index(loop: TrainingLoop | None) -> int:
    assert loop is not None
    param = loop._get_trainable_params()[0]
    step = loop.optimizer.state[param].get("step")
    if torch.is_tensor(step):
        return int(step.detach().cpu().item())
    return int(step or 0)


def _optimizer_state_dict_step_index(optimizer_state_dict: dict[str, Any]) -> int:
    values: list[int] = []
    for item in (optimizer_state_dict.get("state") or {}).values():
        if not isinstance(item, dict) or "step" not in item:
            continue
        step = item["step"]
        if torch.is_tensor(step):
            values.append(int(step.detach().cpu().item()))
        else:
            values.append(int(step or 0))
    return max(values, default=0)


def run_smoke() -> dict[str, Any]:
    result = test_training_loop_native_failure_latches_pytorch_fallback()
    deferred = test_training_loop_deferred_state_sync_flushes_before_recovery_fallback()
    epoch_flush = test_training_loop_deferred_state_sync_flushes_on_epoch_finalization()
    epoch_sync_failure = test_training_loop_deferred_state_sync_failure_latches_runtime()
    state_save = test_trainer_state_save_flushes_deferred_native_optimizer_state()
    skipped = (
        int(result.get("skipped_count", 0))
        + int(deferred.get("skipped_count", 0))
        + int(epoch_flush.get("skipped_count", 0))
        + int(epoch_sync_failure.get("skipped_count", 0))
        + int(state_save.get("skipped_count", 0))
    )
    summary = {
        "exact_adamw_recovery_canary_case_count": 5,
        "exact_adamw_recovery_canary_skipped_count": skipped,
        "exact_adamw_recovery_failure_latched_count": int(result.get("failure_latched_count", 0)),
        "exact_adamw_recovery_failure_fallback_pytorch_step_count": int(
            result.get("failure_fallback_pytorch_step_count", 0)
        ),
        "exact_adamw_recovery_post_latch_pytorch_fallback_count": int(
            result.get("post_latch_pytorch_fallback_count", 0)
        ),
        "exact_adamw_recovery_fallback_state_sync_call_count": int(result.get("fallback_state_sync_call_count", 0)),
        "exact_adamw_recovery_optimizer_step_after_recovery_count": int(
            result.get("optimizer_step_after_recovery_count", 0)
        ),
        "exact_adamw_recovery_checkpoint_owner_state_included_count": int(
            result.get("checkpoint_owner_state_included_count", 0)
        )
        + int(deferred.get("checkpoint_owner_state_included_count", 0)),
        "exact_adamw_recovery_checkpoint_epoch_owner_state_included_count": int(
            epoch_flush.get("checkpoint_owner_state_included_count", 0)
        ),
        "exact_adamw_recovery_checkpoint_resume_mismatch_rejected_count": int(
            result.get("checkpoint_resume_mismatch_rejected_count", 0)
        )
        + int(
            deferred.get("checkpoint_resume_mismatch_rejected_count", 0)
        ),
        "exact_adamw_recovery_deferred_native_success_count": int(
            deferred.get("deferred_native_success_count", 0)
        ),
        "exact_adamw_recovery_deferred_optimizer_state_sync_deferred_count": int(
            deferred.get("deferred_optimizer_state_sync_deferred_count", 0)
        ),
        "exact_adamw_recovery_deferred_fallback_state_sync_flush_count": int(
            deferred.get("deferred_fallback_state_sync_flush_count", 0)
        ),
        "exact_adamw_recovery_deferred_failure_latched_count": int(
            deferred.get("deferred_failure_latched_count", 0)
        ),
        "exact_adamw_recovery_deferred_post_latch_pytorch_fallback_count": int(
            deferred.get("deferred_post_latch_pytorch_fallback_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_state_sync_flush_count": int(
            epoch_flush.get("deferred_epoch_state_sync_flush_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_state_sync_step_count": int(
            epoch_flush.get("deferred_epoch_state_sync_step_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_close_already_clean_count": int(
            epoch_flush.get("deferred_epoch_close_already_clean_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_optimizer_step_synced_count": int(
            epoch_flush.get("deferred_epoch_optimizer_step_synced_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_sync_failure_latched_count": int(
            epoch_sync_failure.get("deferred_epoch_sync_failure_latched_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_sync_failure_dirty_state_retained_count": int(
            epoch_sync_failure.get("deferred_epoch_sync_failure_dirty_state_retained_count", 0)
        ),
        "exact_adamw_recovery_deferred_epoch_sync_failure_optimizer_step_stale_count": int(
            epoch_sync_failure.get("deferred_epoch_sync_failure_optimizer_step_stale_count", 0)
        ),
        "exact_adamw_recovery_deferred_state_save_flush_count": int(
            state_save.get("deferred_state_save_flush_count", 0)
        ),
        "exact_adamw_recovery_deferred_state_save_optimizer_step_synced_count": int(
            state_save.get("deferred_state_save_optimizer_step_synced_count", 0)
        ),
        "exact_adamw_recovery_deferred_state_save_owner_state_included_count": int(
            state_save.get("deferred_state_save_owner_state_included_count", 0)
        ),
        "exact_adamw_recovery_deferred_state_save_checkpoint_contract_integrated_count": int(
            state_save.get("deferred_state_save_checkpoint_contract_integrated_count", 0)
        ),
        "exact_adamw_recovery_training_path_enabled_count": int(result.get("training_path_enabled_count", 0)),
        "training_path_enabled_count": 0,
    }
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_loop_recovery_state_sync_smoke",
        "ok": True,
        "skipped": skipped == 1,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "summary": summary,
    }


def main() -> int:
    run_smoke()
    print("turbocore_native_update_training_loop_recovery_state_sync_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
