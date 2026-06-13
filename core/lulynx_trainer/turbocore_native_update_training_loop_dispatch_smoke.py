"""Smoke check for TrainingLoop native update dispatch execution."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.training_loop import TrainingLoop  # noqa: E402


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def _make_loop(*, direct_grad: bool = False) -> TrainingLoop:
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector([param]),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cuda",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow",
        turbocore_update_shadow_compare_interval=1,
        turbocore_update_shadow_direct_grad=bool(direct_grad),
        turbocore_update_shadow_checkpoint_contract=True,
        turbocore_update_shadow_copyback_probe=True,
        turbocore_update_shadow_copyback_dispatch_experimental=True,
        turbocore_update_shadow_native_binding_probe=True,
        turbocore_update_shadow_owner_native_launch_probe=True,
        turbocore_update_shadow_owner_native_launch_max_numel=1024,
        turbocore_update_shadow_owner_native_event_chain_probe=True,
        turbocore_update_shadow_save_owner_state=True,
        turbocore_native_update_mode="native_experimental",
        turbocore_native_update_required_shadow_passes=1,
        turbocore_native_update_allow_missing_kernel=True,
        turbocore_native_update_dispatch_enabled=True,
        turbocore_native_update_training_path_enabled=True,
        turbocore_native_update_require_native_cuda=True,
    )
    loop.total_steps = 2
    return loop


def _run_two_step_loop(*, direct_grad: bool) -> tuple[TrainingLoop | None, list[dict[str, object]]]:
    if not torch.cuda.is_available():
        return None, []
    loop = _make_loop(direct_grad=direct_grad)
    captured: list[dict[str, object]] = []

    def _fake_train_step(self: TrainingLoop, _batch: dict, accumulation_steps: int = 1, return_loss_tensor: bool = False):
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

    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}, {}], 0)

    assert result["steps"] == 2, result
    assert len(captured) == 2, captured
    return loop, captured


def test_training_loop_second_step_executes_native_update() -> dict[str, int]:
    loop, captured = _run_two_step_loop(direct_grad=False)
    if not captured:
        return {"skipped_count": 1}
    first_runtime = captured[0]["turbocore_native_update_dispatch_runtime"]
    second_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    second_profile = captured[1]["turbocore_native_update_runtime_profile"]
    training_executor = second_runtime["training_executor"]
    executor_result = training_executor["result"]
    update_report = executor_result["update_report"]
    owner_step = update_report["owner_step"]
    native_report = owner_step["native_report"]

    assert first_runtime["native_step_executed"] is False, first_runtime
    assert second_runtime["native_step_executed"] is True, second_runtime
    assert second_runtime["native_kernel_launched"] is True, second_runtime
    assert second_runtime["should_call_pytorch_optimizer_step"] is False, second_runtime
    assert second_runtime["fallback_to_pytorch_required"] is False, second_runtime
    assert training_executor["called"] is True, second_runtime
    assert training_executor["native_step_executed"] is True, second_runtime
    assert executor_result["pytorch_optimizer_state_synced"] is True, second_runtime
    assert executor_result["timing"]["executor_step_ms"] >= 0.0, second_runtime
    assert update_report["owner_backend"] == "rust_cuda_adamw_v0", second_runtime
    assert update_report["used_direct_grad"] is False, second_runtime
    assert update_report["timing"]["owner_step_ms"] >= 0.0, second_runtime
    assert native_report["runtime_synchronization"] == "cuCtxSynchronize_after_native_step", second_runtime
    assert second_profile["resolved"] == "native_training_step", second_profile
    assert second_profile["native_step_executed"] is True, second_profile
    assert captured[1].get("turbocore_native_update_direct_grad_lifecycle", {}) == {}, captured[1]
    checkpoint = _checkpoint_resume_summary(loop)
    return {
        "skipped_count": 0,
        "native_dispatch_ready_count": 1,
        "native_kernel_launched_count": 1,
        "pytorch_optimizer_step_skipped_count": 1,
        "optimizer_state_synced_count": 1,
        "direct_grad_ready_count": 0,
        "direct_grad_hook_write_count": 0,
        **checkpoint,
        "canary_training_path_enabled_count": 1,
    }


def test_training_loop_second_step_executes_native_update_with_direct_grad() -> dict[str, int]:
    loop, captured = _run_two_step_loop(direct_grad=True)
    if not captured:
        return {"skipped_count": 1}
    second_runtime = captured[1]["turbocore_native_update_dispatch_runtime"]
    lifecycle = captured[1]["turbocore_native_update_direct_grad_lifecycle"]
    training_executor = second_runtime["training_executor"]
    executor_result = training_executor["result"]
    update_report = executor_result["update_report"]
    direct_snapshot = update_report["direct_grad_snapshot"]

    assert lifecycle["enabled"] is True, lifecycle
    assert lifecycle["direct_grad_to_training_executor_owner"] is True, lifecycle
    assert lifecycle["snapshot"]["hooks_installed"] == 1, lifecycle
    assert second_runtime["native_step_executed"] is True, second_runtime
    assert second_runtime["native_kernel_launched"] is True, second_runtime
    assert second_runtime["should_call_pytorch_optimizer_step"] is False, second_runtime
    assert training_executor["native_step_executed"] is True, second_runtime
    assert executor_result["should_call_pytorch_optimizer_step"] is False, second_runtime
    assert update_report["owner_backend"] == "rust_cuda_adamw_v0", second_runtime
    assert update_report["used_direct_grad"] is True, second_runtime
    assert update_report["timing"]["grad_sync_ms"] == 0.0, second_runtime
    assert direct_snapshot["hooks_installed"] == 1, direct_snapshot
    assert direct_snapshot["writes"] >= 1, direct_snapshot
    assert direct_snapshot["written_numel"] == 3, direct_snapshot
    assert executor_result["pytorch_optimizer_state_synced"] is True, second_runtime
    checkpoint = _checkpoint_resume_summary(loop)
    return {
        "skipped_count": 0,
        "native_dispatch_ready_count": 1,
        "native_kernel_launched_count": 1,
        "pytorch_optimizer_step_skipped_count": 1,
        "optimizer_state_synced_count": 1,
        "direct_grad_ready_count": 1,
        "direct_grad_hook_write_count": int(direct_snapshot["writes"]),
        "direct_grad_written_numel": int(direct_snapshot["written_numel"]),
        **checkpoint,
        "canary_training_path_enabled_count": 1,
    }


def _checkpoint_resume_summary(loop: TrainingLoop | None) -> dict[str, int]:
    if loop is None:
        return {}
    checkpoint = loop.get_turbocore_update_checkpoint_state()
    restore = loop.load_turbocore_update_checkpoint_state(checkpoint)
    mismatch = loop.load_turbocore_update_checkpoint_state(_mismatched_checkpoint(checkpoint))
    contract = checkpoint.get("checkpoint_contract") if isinstance(checkpoint.get("checkpoint_contract"), dict) else {}
    assert checkpoint["checkpoint_metadata_integrated"] is True, checkpoint
    assert checkpoint["owner_state_included"] is True, checkpoint
    assert contract["roundtrip_checked"] is True, checkpoint
    assert contract["roundtrip_ok"] is True, checkpoint
    assert restore["loaded"] is True, restore
    assert restore["compatible"] is True, restore
    assert restore["owner_state_pending"] is True, restore
    assert mismatch["loaded"] is True, mismatch
    assert mismatch["compatible"] is False, mismatch
    assert checkpoint["training_path_enabled"] is False, checkpoint
    assert restore["training_path_enabled"] is False, restore
    assert mismatch["training_path_enabled"] is False, mismatch
    return {
        "checkpoint_owner_state_included_count": 1,
        "checkpoint_roundtrip_ok_count": 1,
        "checkpoint_resume_compatible_count": 1,
        "checkpoint_resume_mismatch_rejected_count": 1,
        "checkpoint_training_path_enabled_count": 0,
    }


def _mismatched_checkpoint(checkpoint: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(checkpoint)
    owner_state = payload.get("owner_state_dict") if isinstance(payload.get("owner_state_dict"), dict) else {}
    layout = owner_state.get("layout") if isinstance(owner_state.get("layout"), dict) else {}
    layout["total_numel"] = int(layout.get("total_numel", 0) or 0) + 1
    owner_state["layout"] = layout
    payload["owner_state_dict"] = owner_state
    return payload


def run_smoke() -> dict[str, object]:
    non_direct = test_training_loop_second_step_executes_native_update()
    direct = test_training_loop_second_step_executes_native_update_with_direct_grad()
    skipped = int(non_direct.get("skipped_count", 0)) + int(direct.get("skipped_count", 0))
    summary = {
        "exact_adamw_training_loop_canary_case_count": 2,
        "exact_adamw_training_loop_canary_skipped_count": skipped,
        "exact_adamw_training_loop_native_dispatch_ready_count": int(
            non_direct.get("native_dispatch_ready_count", 0)
        )
        + int(direct.get("native_dispatch_ready_count", 0)),
        "exact_adamw_training_loop_native_kernel_launched_count": int(
            non_direct.get("native_kernel_launched_count", 0)
        )
        + int(direct.get("native_kernel_launched_count", 0)),
        "exact_adamw_training_loop_pytorch_optimizer_step_skipped_count": int(
            non_direct.get("pytorch_optimizer_step_skipped_count", 0)
        )
        + int(direct.get("pytorch_optimizer_step_skipped_count", 0)),
        "exact_adamw_training_loop_optimizer_state_synced_count": int(
            non_direct.get("optimizer_state_synced_count", 0)
        )
        + int(direct.get("optimizer_state_synced_count", 0)),
        "exact_adamw_training_loop_direct_grad_ready_count": int(direct.get("direct_grad_ready_count", 0)),
        "exact_adamw_training_loop_direct_grad_hook_write_count": int(
            direct.get("direct_grad_hook_write_count", 0)
        ),
        "exact_adamw_training_loop_direct_grad_written_numel": int(direct.get("direct_grad_written_numel", 0)),
        "exact_adamw_training_loop_checkpoint_owner_state_included_count": int(
            non_direct.get("checkpoint_owner_state_included_count", 0)
        )
        + int(direct.get("checkpoint_owner_state_included_count", 0)),
        "exact_adamw_training_loop_checkpoint_roundtrip_ok_count": int(
            non_direct.get("checkpoint_roundtrip_ok_count", 0)
        )
        + int(direct.get("checkpoint_roundtrip_ok_count", 0)),
        "exact_adamw_training_loop_checkpoint_resume_compatible_count": int(
            non_direct.get("checkpoint_resume_compatible_count", 0)
        )
        + int(direct.get("checkpoint_resume_compatible_count", 0)),
        "exact_adamw_training_loop_checkpoint_resume_mismatch_rejected_count": int(
            non_direct.get("checkpoint_resume_mismatch_rejected_count", 0)
        )
        + int(direct.get("checkpoint_resume_mismatch_rejected_count", 0)),
        "exact_adamw_training_loop_checkpoint_training_path_enabled_count": int(
            non_direct.get("checkpoint_training_path_enabled_count", 0)
        )
        + int(direct.get("checkpoint_training_path_enabled_count", 0)),
        "exact_adamw_training_loop_canary_training_path_enabled_count": int(
            non_direct.get("canary_training_path_enabled_count", 0)
        )
        + int(direct.get("canary_training_path_enabled_count", 0)),
        "exact_adamw_training_loop_product_default_changed_count": 0,
    }
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_loop_dispatch_smoke",
        "ok": True,
        "skipped": skipped == 2,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "summary": summary,
    }


def main() -> int:
    run_smoke()
    print("turbocore_native_update_training_loop_dispatch_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
