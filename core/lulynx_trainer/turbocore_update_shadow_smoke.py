"""Smoke checks for TrainingLoop TurboCore update shadow reporting."""

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
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-2, weight_decay=0.0)
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
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1000.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow",
        turbocore_update_shadow_compare_interval=1,
        turbocore_update_shadow_compare_sample_params=1,
        turbocore_update_shadow_stop_after_consecutive_passes=1,
        turbocore_update_shadow_direct_grad=True,
        turbocore_update_shadow_checkpoint_contract=True,
        turbocore_update_shadow_copyback_probe=True,
        turbocore_update_shadow_copyback_dispatch_experimental=False,
        turbocore_update_shadow_native_binding_probe=True,
        turbocore_update_shadow_owner_native_launch_probe=True,
        turbocore_update_shadow_owner_native_launch_max_numel=1024,
        turbocore_update_shadow_save_owner_state=True,
        turbocore_native_update_mode="profile",
        turbocore_native_update_required_shadow_passes=1,
    )
    loop.total_steps = 1
    return loop


def test_training_loop_emits_update_shadow_report() -> None:
    loop = _make_loop()
    captured: list[dict[str, object]] = []

    def _fake_train_step(self: TrainingLoop, _batch: dict, accumulation_steps: int = 1, return_loss_tensor: bool = False):
        params = self._get_trainable_params()
        loss = sum((param * param).sum() for param in params)
        self._turbocore_direct_grad_lifecycle_report = self._turbocore_update_shadow.prepare_before_backward(
            params,
            optimizer=self.optimizer,
            max_grad_norm=self.max_grad_norm,
            step=self.global_step,
            reset_owner_grad=True,
        )
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    def _on_step_end(_step: int, _loss: float, info: dict[str, object]) -> None:
        captured.append(info)

    loop.on_step_end = _on_step_end
    before = loop._get_trainable_params()[0].detach().clone()
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)

    after = loop._get_trainable_params()[0].detach().clone()
    assert result["steps"] == 1, result
    assert not torch.allclose(before, after), (before, after)
    assert captured, "Expected on_step_end info"
    report = captured[-1].get("turbocore_update_shadow")
    assert isinstance(report, dict), captured[-1]
    assert report["training_path_enabled"] is False, report
    assert report["stage"] == "before_optimizer", report
    assert report["direct_grad_audit"]["enabled"] is True, report
    assert report["direct_grad_audit"]["warmup_no_writes"] is False, report
    assert report["direct_grad_audit"]["parity_ok"] is True, report
    assert report["direct_grad_audit"]["training_path_enabled"] is False, report
    assert report["checkpoint_contract"]["roundtrip_checked"] is True, report
    assert report["copyback_probe"]["parameters_mutated"] is False, report
    assert report["copyback_probe"]["real_parameters_mutated"] is False, report
    assert report["copyback_probe"]["scratch_copyback_validated"] is True, report
    assert report["copyback_probe"]["shape_validated"] is True, report
    assert report["copyback_probe"]["dtype_cast_validated"] is True, report
    assert report["copyback_probe"]["scratch_max_abs_diff"] == 0.0, report
    assert report["copyback_dispatch_probe"] == {}, report
    assert report["native_binding_probe"]["request_shape_ready"] is True, report
    assert report["native_binding_probe"]["handle_count"] == 4, report
    assert report["native_binding_probe"]["stream_contract"]["contract"] == "turbocore_tensor_binding_stream_lifetime_v0", report
    assert report["native_binding_probe"]["stream_lifetime_bound"] is False, report
    assert report["native_binding_probe"]["stream_guard_present"] is True, report
    assert report["native_binding_probe"]["stream_guard_ready"] is False, report
    assert report["native_binding_probe"]["stream_identity_ready"] is True, report
    assert report["native_binding_probe"]["stream_guard_level"] == "identity_verified_sync_blocked", report
    assert report["native_binding_probe"]["stream_handle_kind"] == "non_cuda_stream", report
    assert report["native_binding_probe"]["stream_handle_reported"] is False, report
    assert report["native_binding_probe"]["stream_handle_nonzero"] is False, report
    assert report["native_binding_probe"]["synchronization_guard_ready"] is False, report
    assert report["native_binding_probe"]["synchronization_strategy"] == "non_cuda_no_stream_sync", report
    assert report["native_binding_probe"]["event_chain_contract"] == "turbocore_stream_event_chain_guard_v2", report
    assert report["native_binding_probe"]["event_chain_state"] == "not_attempted", report
    assert report["native_binding_probe"]["event_chain_probe_requested"] is False, report
    assert report["native_binding_probe"]["event_chain_probe_attempted"] is False, report
    assert report["native_binding_probe"]["event_chain_verified"] is False, report
    assert report["native_binding_probe"]["stream_wait_event_verified"] is False, report
    assert report["native_binding_probe"]["native_launch_candidate"] is False, report
    owner_launch = report["owner_native_launch_probe"]
    assert owner_launch["contract"] == "turbocore_owner_buffer_native_launch_v1", report
    assert owner_launch["skipped"] is True, report
    assert owner_launch["reason"] == "torch_cuda_unavailable" or owner_launch["reason"] == "owner_buffers_not_cuda", report
    assert owner_launch["native_launch_attempted"] is False, report
    assert owner_launch["kernel_executed"] is False, report
    assert owner_launch["persistent_owner_mutated"] is False, report
    assert owner_launch["training_dispatch"] is False, report
    assert owner_launch["training_path_enabled"] is False, report
    assert report["after_optimizer"]["compared"] is True, report
    assert report["after_optimizer"]["sampled"] is False, report
    assert report["after_optimizer"]["auto_stopped_after_this_step"] is True, report
    assert "max_abs_param_diff" in report["after_optimizer"], report
    gate = captured[-1].get("turbocore_native_update_gate")
    assert isinstance(gate, dict), captured[-1]
    assert gate["training_path_enabled"] is False, gate
    assert gate["native_kernel_present"] is False, gate
    assert gate["fallback_policy"]["training_path_enabled"] is False, gate
    assert gate["shadow"]["copyback_probe_present"] is True, gate
    assert gate["shadow"]["copyback_scratch_validated"] is True, gate
    assert gate["shadow"]["copyback_real_parameters_mutated"] is False, gate
    assert gate["shadow"]["copyback_dispatch_probe_present"] is False, gate
    assert gate["shadow"]["native_binding_probe_present"] is True, gate
    assert gate["shadow"]["native_binding_request_shape_ready"] is True, gate
    assert gate["shadow"]["native_binding_stream_lifetime_bound"] is False, gate
    assert gate["shadow"]["native_binding_stream_contract_present"] is True, gate
    assert gate["shadow"]["native_binding_stream_kind"] == "default", gate
    assert gate["shadow"]["native_binding_stream_lease_id"] >= 1, gate
    assert gate["shadow"]["native_binding_stream_guard_present"] is True, gate
    assert gate["shadow"]["native_binding_stream_guard_ready"] is False, gate
    assert gate["shadow"]["native_binding_stream_identity_ready"] is True, gate
    assert gate["shadow"]["native_binding_stream_guard_level"] == "identity_verified_sync_blocked", gate
    assert gate["shadow"]["native_binding_stream_handle_kind"] == "non_cuda_stream", gate
    assert gate["shadow"]["native_binding_stream_handle_reported"] is False, gate
    assert gate["shadow"]["native_binding_stream_handle_nonzero"] is False, gate
    assert gate["shadow"]["native_binding_synchronization_guard_ready"] is False, gate
    assert gate["shadow"]["native_binding_synchronization_strategy"] == "non_cuda_no_stream_sync", gate
    assert gate["shadow"]["native_binding_event_chain_contract"] == "turbocore_stream_event_chain_guard_v2", gate
    assert gate["shadow"]["native_binding_event_chain_state"] == "not_attempted", gate
    assert gate["shadow"]["native_binding_event_chain_probe_requested"] is False, gate
    assert gate["shadow"]["native_binding_event_chain_probe_attempted"] is False, gate
    assert gate["shadow"]["native_binding_stream_wait_event_verified"] is False, gate
    assert gate["shadow"]["native_binding_native_launch_candidate"] is False, gate
    assert gate["fallback_policy"]["copyback_scratch_validated"] is True, gate
    assert gate["fallback_policy"]["native_binding_probe_present"] is True, gate
    assert gate["fallback_policy"]["native_binding_stream_contract_present"] is True, gate
    assert "keep_pytorch_optimizer_due_to_unbound_stream_lifetime" in gate["fallback_policy"]["actions"], gate
    assert "keep_pytorch_optimizer_due_to_stream_guard_not_ready" in gate["fallback_policy"]["actions"], gate
    assert "keep_pytorch_optimizer_due_to_event_chain_not_verified" in gate["fallback_policy"]["actions"], gate
    assert "keep_pytorch_optimizer_due_to_stream_identity_not_ready" not in gate["fallback_policy"]["actions"], gate
    assert "native_kernel_promotion_not_enabled" in gate["blocked_reasons"], gate
    assert "parameter_owner_copyback_dispatch_disabled" in gate["blocked_reasons"], gate
    assert "parameter_owner_copyback_not_integrated" not in gate["blocked_reasons"], gate
    lifecycle = captured[-1].get("turbocore_direct_grad_lifecycle")
    assert isinstance(lifecycle, dict), captured[-1]
    assert lifecycle["stage"] == "before_backward", lifecycle
    assert lifecycle["snapshot"]["hooks_installed"] == 1, lifecycle
    ckpt = loop.get_turbocore_update_checkpoint_state()
    assert ckpt["checkpoint_metadata_integrated"] is True, ckpt
    assert ckpt["owner_state_included"] is True, ckpt
    restore_report = loop.load_turbocore_update_checkpoint_state(ckpt)
    assert restore_report["loaded"] is True, restore_report
    assert restore_report["compatible"] is True, restore_report


def test_copyback_dispatch_probe_is_explicit_and_restores_params() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-2, weight_decay=0.0)
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
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1000.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow",
        turbocore_update_shadow_copyback_probe=True,
        turbocore_update_shadow_copyback_dispatch_experimental=True,
    )
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    before = param.detach().clone()
    report = loop._turbocore_update_shadow.prepare_before_optimizer(
        loop._get_trainable_params(),
        optimizer=optimizer,
        max_grad_norm=loop.max_grad_norm,
        step=0,
    )
    after = param.detach().clone()
    dispatch = report["copyback_dispatch_probe"]
    assert dispatch["copyback_dispatch_enabled"] is True, report
    assert dispatch["copyback_dispatch_target"] == "training_parameters", report
    assert dispatch["copyback_dispatch_validated"] is True, report
    assert dispatch["real_parameters_mutated"] is True, report
    assert dispatch["real_parameters_restored"] is True, report
    assert dispatch["training_path_enabled"] is False, report
    assert torch.allclose(before, after), (before, after, dispatch)


def test_training_loop_refreshes_readiness_from_copyback_dispatch_probe() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-2, weight_decay=0.0)
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
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1000.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow",
        turbocore_update_shadow_copyback_probe=True,
        turbocore_update_shadow_copyback_dispatch_experimental=True,
        turbocore_update_shadow_save_owner_state=True,
        turbocore_native_update_mode="profile",
        turbocore_native_update_required_shadow_passes=1,
    )
    loop.total_steps = 1
    captured: list[dict[str, object]] = []

    def _fake_train_step(self: TrainingLoop, _batch: dict, accumulation_steps: int = 1, return_loss_tensor: bool = False):
        loss = sum((item * item).sum() for item in self._get_trainable_params())
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(info)
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)

    assert result["steps"] == 1, result
    assert captured, "Expected on_step_end info"
    shadow = captured[-1].get("turbocore_update_shadow")
    gate = captured[-1].get("turbocore_native_update_gate")
    assert isinstance(shadow, dict), captured[-1]
    assert isinstance(gate, dict), captured[-1]
    dispatch = shadow["copyback_dispatch_probe"]
    assert dispatch["copyback_dispatch_validated"] is True, shadow
    reasons = set(gate["blocked_reasons"])
    readiness_reasons = set(loop._turbocore_native_update_readiness["blocked_reasons"])
    owner = loop._turbocore_native_update_readiness["owner_checks"]
    assert "parameter_owner_copyback_dispatch_not_validated" not in reasons, gate
    assert "parameter_owner_copyback_dispatch_not_validated" not in readiness_reasons, loop._turbocore_native_update_readiness
    assert owner["parameter_owner_copyback_dispatch_validated"] is True, owner
    assert owner["parameter_owner_copyback_integrated"] is True, owner
    assert gate["training_path_enabled"] is False, gate
    assert gate["would_enable_native_update"] is False, gate


def main() -> int:
    test_training_loop_emits_update_shadow_report()
    test_copyback_dispatch_probe_is_explicit_and_restores_params()
    test_training_loop_refreshes_readiness_from_copyback_dispatch_probe()
    print("turbocore_update_shadow_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
