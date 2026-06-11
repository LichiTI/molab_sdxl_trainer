"""Smoke check for TrainingLoop native update dispatch execution."""

from __future__ import annotations

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


def _make_loop() -> TrainingLoop:
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
        turbocore_update_shadow_direct_grad=False,
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


def test_training_loop_second_step_executes_native_update() -> None:
    if not torch.cuda.is_available():
        return
    loop = _make_loop()
    captured: list[dict[str, object]] = []

    def _fake_train_step(self: TrainingLoop, _batch: dict, accumulation_steps: int = 1, return_loss_tensor: bool = False):
        params = self._get_trainable_params()
        for param in params:
            if param.grad is not None:
                param.grad = None
        loss = sum((param * param).sum() for param in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}, {}], 0)

    assert result["steps"] == 2, result
    assert len(captured) == 2, captured
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


def main() -> int:
    test_training_loop_second_step_executes_native_update()
    print("turbocore_native_update_training_loop_dispatch_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
