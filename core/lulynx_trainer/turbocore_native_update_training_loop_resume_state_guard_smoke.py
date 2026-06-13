"""Smoke check for TrainingLoop native-update resume owner-state guards."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.lulynx_trainer.training_loop import TrainingLoop  # noqa: E402
from core.lulynx_trainer.trainer import LulynxTrainer  # noqa: E402
from turbocore_native_update_training_loop_dispatch_smoke import _make_loop  # noqa: E402


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


def _run_loop(loop: TrainingLoop, steps: int) -> list[dict[str, Any]]:
    loop.total_steps = int(steps)
    captured: list[dict[str, Any]] = []
    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    with patch.object(TrainingLoop, "train_step", new=_fake_direct_grad_train_step):
        result = loop.train_epoch([{} for _ in range(steps)], 0)
    assert result["steps"] == steps, result
    assert len(captured) == steps, captured
    return captured


def _owner_step_from_checkpoint(checkpoint: dict[str, Any]) -> int:
    owner_state = checkpoint.get("owner_state_dict") if isinstance(checkpoint.get("owner_state_dict"), dict) else {}
    return int(owner_state.get("step_index", 0) or 0)


def _pending_owner_state(loop: TrainingLoop) -> dict[str, Any] | None:
    shadow = getattr(loop, "_turbocore_update_shadow", None)
    return getattr(shadow, "_pending_owner_state", None)


def _prepare_shadow_owner_before_backward(loop: TrainingLoop, *, step: int = 0) -> dict[str, Any]:
    params = loop._get_trainable_params()
    shadow = getattr(loop, "_turbocore_update_shadow", None)
    assert shadow is not None, "turbocore update shadow must exist"
    return shadow.prepare_before_backward(
        params,
        optimizer=loop.optimizer,
        max_grad_norm=loop.max_grad_norm,
        step=step,
        reset_owner_grad=True,
    )


def _executor_owner_step(loop: TrainingLoop) -> int:
    executor = getattr(getattr(loop, "_turbocore_update_shadow", None), "executor", None)
    owner = getattr(executor, "owner", None)
    return int(getattr(owner, "step_index", 0) or 0)


def _mismatched_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(checkpoint)
    owner_state = payload.get("owner_state_dict") if isinstance(payload.get("owner_state_dict"), dict) else {}
    layout = owner_state.get("layout") if isinstance(owner_state.get("layout"), dict) else {}
    layout["total_numel"] = int(layout.get("total_numel", 0) or 0) + 1
    owner_state["layout"] = layout
    payload["owner_state_dict"] = owner_state
    return payload


def test_compatible_resume_owner_state_is_consumed_before_native_step() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"skipped_count": 1}
    source_loop = _make_loop(direct_grad=True)
    _run_loop(source_loop, 2)
    checkpoint = source_loop.get_turbocore_update_checkpoint_state()
    source_owner_step = _owner_step_from_checkpoint(checkpoint)
    checkpoint_contract = checkpoint.get("checkpoint_contract") or {}
    assert source_owner_step > 0, checkpoint
    assert checkpoint_contract["trainer_checkpoint_integration"] is True, checkpoint_contract

    resumed_loop = _make_loop(direct_grad=True)
    load_report = resumed_loop.load_turbocore_update_checkpoint_state(checkpoint)
    assert load_report["loaded"] is True, load_report
    assert load_report["compatible"] is True, load_report
    assert load_report["owner_state_pending"] is True, load_report
    assert _pending_owner_state(resumed_loop) is not None, load_report
    lifecycle = _prepare_shadow_owner_before_backward(resumed_loop)
    assert lifecycle.get("skipped") is not True, lifecycle
    assert _pending_owner_state(resumed_loop) is None, lifecycle
    assert _executor_owner_step(resumed_loop) == source_owner_step, lifecycle

    captured = _run_loop(resumed_loop, 2)
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    assert native_runtime["native_step_executed"] is True, native_runtime
    assert resumed_loop.load_turbocore_update_checkpoint_state(_mismatched_checkpoint(checkpoint))["compatible"] is False
    assert checkpoint["training_path_enabled"] is False, checkpoint
    return {
        "skipped_count": 0,
        "resume_state_guard_compatible_loaded_count": 1,
        "resume_state_guard_owner_state_pending_count": 1,
        "resume_state_guard_pending_consumed_count": 1,
        "resume_state_guard_native_step_after_resume_count": 1,
        "resume_state_guard_owner_step_restored_count": 1,
        "resume_state_guard_checkpoint_contract_integrated_count": 1,
        "resume_state_guard_training_path_enabled_count": 0,
    }


def test_mismatched_resume_owner_state_is_rejected() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"skipped_count": 1}
    source_loop = _make_loop(direct_grad=True)
    _run_loop(source_loop, 2)
    checkpoint = source_loop.get_turbocore_update_checkpoint_state()

    resumed_loop = _make_loop(direct_grad=True)
    load_report = resumed_loop.load_turbocore_update_checkpoint_state(_mismatched_checkpoint(checkpoint))
    assert load_report["loaded"] is True, load_report
    assert load_report["compatible"] is False, load_report
    assert load_report["owner_state_pending"] is False, load_report
    assert _pending_owner_state(resumed_loop) is None, load_report
    lifecycle = _prepare_shadow_owner_before_backward(resumed_loop)
    assert lifecycle.get("skipped") is not True, lifecycle
    assert _pending_owner_state(resumed_loop) is None, lifecycle
    captured = _run_loop(resumed_loop, 2)
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    assert native_runtime["native_step_executed"] is True, native_runtime
    return {
        "skipped_count": 0,
        "resume_state_guard_mismatch_rejected_count": 1,
        "resume_state_guard_mismatch_pending_blocked_count": 1,
        "resume_state_guard_training_path_enabled_count": 0,
    }


def test_trainer_state_file_roundtrip_restores_pending_owner_state() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"skipped_count": 1}
    source_loop = _make_loop(direct_grad=True)
    _run_loop(source_loop, 2)
    checkpoint = source_loop.get_turbocore_update_checkpoint_state()
    source_owner_step = _owner_step_from_checkpoint(checkpoint)
    assert source_owner_step > 0, checkpoint

    with tempfile.TemporaryDirectory() as tmp_dir:
        source_trainer = _make_trainer(source_loop, tmp_dir, "turbocore-resume")
        source_trainer._save_state(epoch=3)
        state_path = Path(tmp_dir) / "turbocore-resume-000003-state.pt"
        assert state_path.exists(), state_path

        resumed_loop = _make_loop(direct_grad=True)
        resumed_trainer = _make_trainer(resumed_loop, tmp_dir, "turbocore-resume")
        state = resumed_trainer._load_state(str(state_path))
        assert isinstance(state, dict), state
        assert state.get("turbocore_update_state"), state
        load_report = resumed_loop.load_turbocore_update_checkpoint_state(state["turbocore_update_state"])

    assert load_report["loaded"] is True, load_report
    assert load_report["compatible"] is True, load_report
    assert load_report["owner_state_pending"] is True, load_report
    assert _pending_owner_state(resumed_loop) is not None, load_report
    lifecycle = _prepare_shadow_owner_before_backward(resumed_loop)
    assert lifecycle.get("skipped") is not True, lifecycle
    assert _pending_owner_state(resumed_loop) is None, lifecycle
    assert _executor_owner_step(resumed_loop) == source_owner_step, lifecycle
    captured = _run_loop(resumed_loop, 2)
    native_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    assert native_runtime["native_step_executed"] is True, native_runtime
    return {
        "skipped_count": 0,
        "resume_state_guard_file_roundtrip_loaded_count": 1,
        "resume_state_guard_file_roundtrip_pending_consumed_count": 1,
        "resume_state_guard_file_roundtrip_native_step_count": 1,
        "resume_state_guard_training_path_enabled_count": 0,
    }


def _make_trainer(loop: TrainingLoop, output_dir: str, output_name: str) -> LulynxTrainer:
    trainer = LulynxTrainer(
        SimpleNamespace(
            output_dir=output_dir,
            output_name=output_name,
            model_arch="sd15",
            model_type="sd15",
        )
    )
    trainer.training_loop = loop
    trainer._log = lambda _message: None
    trainer._write_run_manifest = lambda *_args, **_kwargs: None
    return trainer


def run_smoke() -> dict[str, Any]:
    compatible = test_compatible_resume_owner_state_is_consumed_before_native_step()
    mismatched = test_mismatched_resume_owner_state_is_rejected()
    file_roundtrip = test_trainer_state_file_roundtrip_restores_pending_owner_state()
    skipped = (
        int(compatible.get("skipped_count", 0))
        + int(mismatched.get("skipped_count", 0))
        + int(file_roundtrip.get("skipped_count", 0))
    )
    summary = {
        "exact_adamw_resume_state_guard_canary_case_count": 3,
        "exact_adamw_resume_state_guard_canary_skipped_count": skipped,
        "exact_adamw_resume_state_guard_compatible_loaded_count": int(
            compatible.get("resume_state_guard_compatible_loaded_count", 0)
        ),
        "exact_adamw_resume_state_guard_owner_state_pending_count": int(
            compatible.get("resume_state_guard_owner_state_pending_count", 0)
        ),
        "exact_adamw_resume_state_guard_pending_consumed_count": int(
            compatible.get("resume_state_guard_pending_consumed_count", 0)
        ),
        "exact_adamw_resume_state_guard_native_step_after_resume_count": int(
            compatible.get("resume_state_guard_native_step_after_resume_count", 0)
        ),
        "exact_adamw_resume_state_guard_owner_step_restored_count": int(
            compatible.get("resume_state_guard_owner_step_restored_count", 0)
        ),
        "exact_adamw_resume_state_guard_checkpoint_contract_integrated_count": int(
            compatible.get("resume_state_guard_checkpoint_contract_integrated_count", 0)
        ),
        "exact_adamw_resume_state_guard_mismatch_rejected_count": int(
            mismatched.get("resume_state_guard_mismatch_rejected_count", 0)
        ),
        "exact_adamw_resume_state_guard_mismatch_pending_blocked_count": int(
            mismatched.get("resume_state_guard_mismatch_pending_blocked_count", 0)
        ),
        "exact_adamw_resume_state_guard_file_roundtrip_loaded_count": int(
            file_roundtrip.get("resume_state_guard_file_roundtrip_loaded_count", 0)
        ),
        "exact_adamw_resume_state_guard_file_roundtrip_pending_consumed_count": int(
            file_roundtrip.get("resume_state_guard_file_roundtrip_pending_consumed_count", 0)
        ),
        "exact_adamw_resume_state_guard_file_roundtrip_native_step_count": int(
            file_roundtrip.get("resume_state_guard_file_roundtrip_native_step_count", 0)
        ),
        "exact_adamw_resume_state_guard_training_path_enabled_count": 0,
        "training_path_enabled_count": 0,
    }
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_loop_resume_state_guard_smoke",
        "ok": True,
        "skipped": skipped == 2,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "summary": summary,
    }


def main() -> int:
    run_smoke()
    print("turbocore_native_update_training_loop_resume_state_guard_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
