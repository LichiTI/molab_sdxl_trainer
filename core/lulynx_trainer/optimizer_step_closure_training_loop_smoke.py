# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""TrainingLoop smoke for optimizers that require optimizer.step(closure)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer
from core.lulynx_trainer.optimizer_step_contracts import (
    optimizer_requires_step_closure,
    optimizer_step_closure_requires_initial_backward,
)
from core.lulynx_trainer.training_pipeline_trace import LulynxTrainingPipelineTrace
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime


class _NoopProfiler:
    enabled = False

    def reset_group(self) -> None:
        return None

    def start(self) -> float:
        return 0.0

    def start_cpu(self) -> float:
        return 0.0

    def record(self, *_args, **_kwargs) -> None:
        return None

    def record_optimizer_update_substage(self, *_args, **_kwargs) -> float:
        return 0.0

    def snapshot(self, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    def latest_snapshot(self) -> dict[str, Any]:
        return {}

    def latest_bubble_profile(self) -> dict[str, Any]:
        return {}


class _NoopArmer:
    def last_gate_report(self) -> dict[str, Any]:
        return {}


class _NoopGate:
    requested = False


class _TinyInjector:
    def __init__(self, param: torch.nn.Parameter) -> None:
        self.param = param
        self.injected_layers: dict[str, Any] = {}
        self.seen_steps: list[int] = []

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]

    def set_global_step(self, step: int) -> None:
        self.seen_steps.append(int(step))


class _ClosureLoop(TrainingLoop):
    def __init__(self, optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> None:
        self.param = param
        self.unet = torch.nn.Identity()
        self.optimizer = optimizer
        self.lora_injector = _TinyInjector(param)
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        self.device = "cpu"
        self._runtime_device = torch.device("cpu")
        self.dtype = torch.float32
        self.training_type = "optimizer_step_closure_smoke"
        self.gradient_accumulation_steps = 2
        self.gradient_accumulation_mode = "fast"
        self.max_grad_norm = 1.0
        self.global_step = 0
        self.total_steps = 1
        self.completed_by_step_limit = False
        self._should_stop = False
        self.skip_until_initial_step = False
        self.initial_step_target = 0
        self.current_epoch = 0
        self._advanced_monitoring = False
        self._attn_entropy_interval = 100
        self._act_drift_interval = 100
        self._peak_vram_diag_interval = 100
        self._current_micro_batch_index = 1
        self._current_micro_batch_count = 1
        self._current_sync_gradients = True
        self._current_accumulation_group_start = True
        self._optimizer_step_closure_active = False
        self._pcgrad_pending_grads: list[dict[str, torch.Tensor]] = []
        self._pcgrad_param_names: dict[int, str] = {}
        self._pcgrad_last_stats: dict[str, Any] = {}
        self.pcgrad_enabled = False
        self._dynamic_batch_scheduler = None
        self._block_offloader = None
        self._module_offload_manager = None
        self.adapter_cpu_residency = None
        self.safeguard = None
        self.auditor = None
        self.auditor_interval = 100
        self.on_before_train_step = None
        self.on_before_optimizer_step = None
        self.on_step_end = None
        self.on_epoch_end = None
        self.validation_dataloader = None
        self.eval_every_n_steps = 0
        self.te_manager = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self._train_text_encoder_any = False
        self._drift_monitor = None
        self._drift_check_interval = 100
        self._layer_monitor_enabled = False
        self._layer_monitor_interval = 100
        self._layer_monitor_max_layers = 0
        self._layer_monitor_sparsity_epsilon = 0.0
        self._layer_monitor_mode = "off"
        self._layer_monitor_sample_size = 0
        self._grad_tracker = None
        self._gradient_release_manager = None
        self._pipeline_trace = LulynxTrainingPipelineTrace()
        self._last_pipeline_trace = None
        self._turbocore_update_shadow = type("_Shadow", (), {"enabled": False})()
        self._turbocore_native_update_gate = _NoopGate()
        self._turbocore_native_update_dispatch_armer = _NoopArmer()
        self._turbocore_native_update_dispatch_runtime = TurboCoreNativeUpdateDispatchRuntime()
        self._turbocore_native_update_readiness: dict[str, Any] = {}
        self._turbocore_native_update_runtime_profile: dict[str, Any] = {}
        self._turbocore_native_update_diagnostic_executor_replay = False
        self._turbocore_native_update_defer_state_sync = False
        self._turbocore_direct_grad_lifecycle_report: dict[str, Any] = {}
        self._step_phase_profiler = _NoopProfiler()
        self.memory_optimization_state: dict[str, Any] = {}
        self.outer_deferred_backward_calls = 0
        self.closure_backward_calls = 0
        self.bound_step_closure_calls = 0

    def _make_optimizer_step_closure(
        self,
        microbatches: list[dict[str, Any]],
        accumulation_steps: int,
    ):
        self.bound_step_closure_calls += 1
        return super()._make_optimizer_step_closure(microbatches, accumulation_steps)

    def _train_step_impl(
        self,
        batch: dict[str, torch.Tensor],
        accumulation_steps: int | None = None,
        do_backward: bool = True,
        return_loss_tensor: bool = False,
    ) -> float | torch.Tensor:
        accumulation_steps = max(int(accumulation_steps or self.gradient_accumulation_steps), 1)
        x = batch["x"].to(self.param)
        target = batch["target"].to(self.param)
        loss = ((self.param * x) - target).pow(2).mean()
        if do_backward:
            deferred = optimizer_requires_step_closure(self.optimizer) and not bool(
                getattr(self, "_optimizer_step_closure_active", False)
            ) and not optimizer_step_closure_requires_initial_backward(self.optimizer)
            if deferred:
                self.outer_deferred_backward_calls += 1
            else:
                (loss / accumulation_steps).backward()
                self.closure_backward_calls += 1
        if return_loss_tensor:
            return loss.detach()
        return float(loss.detach().item())

    def _maybe_save_safe_state(self) -> None:
        return None

    def _refresh_module_offload_stats(self) -> None:
        return None

    def _record_transfer_profile_step(self, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    def _update_block_swap_profile(self) -> None:
        return None

    def _update_precision_swap_observations(self, *_args, **_kwargs) -> None:
        return None

    def _update_vram_smart_sensing_runtime(self, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    def _maybe_release_cuda_cache(self, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    def _turbocore_native_update_runtime_context(self) -> dict[str, Any]:
        return {"training_path_enabled": False}

    def _sync_turbocore_native_update_training_executor_to_pytorch(self, reason: str) -> dict[str, Any]:
        return {"synced": False, "reason": reason, "training_path_enabled": False}

    def _close_turbocore_native_update_training_executor(self, reason: str = "manual_close") -> dict[str, Any]:
        return {"closed": False, "reason": reason, "training_path_enabled": False}

    def _run_audit(self) -> None:
        return None


def _make_lbfgs_optimizer(param: torch.nn.Parameter) -> torch.optim.Optimizer:
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name="LBFGS",
        lr=0.25,
        weight_decay=0.0,
        optimizer_args={
            "name": "LBFGS",
            "max_iter": 1,
            "history_size": 4,
            "line_search_fn": None,
        },
    )
    assert optimizer_requires_step_closure(optimizer) is True
    return optimizer


def _make_bsam_optimizer(param: torch.nn.Parameter) -> torch.optim.Optimizer:
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name="BSAM",
        lr=0.05,
        weight_decay=0.0,
        optimizer_args={"name": "BSAM", "num_data": 1},
    )
    assert optimizer_requires_step_closure(optimizer) is True
    return optimizer


def test_lbfgs_training_loop_binds_recompute_closure() -> None:
    torch.manual_seed(20260531)
    param = torch.nn.Parameter(torch.tensor([0.8, -0.4], dtype=torch.float32))
    optimizer = _make_lbfgs_optimizer(param)
    loop = _ClosureLoop(optimizer, param)
    initial = param.detach().clone()
    dataloader = [
        {"x": torch.tensor([1.0, 0.5]), "target": torch.tensor([0.1, -0.2])},
        {"x": torch.tensor([0.5, -1.0]), "target": torch.tensor([-0.3, 0.4])},
    ]

    result = loop.train_epoch(dataloader, epoch=0)

    assert result["steps"] == 2, result
    assert loop.global_step == 1
    assert loop.outer_deferred_backward_calls == 2
    assert loop.bound_step_closure_calls == 1
    assert loop.closure_backward_calls == 2
    assert param.grad is None
    assert not torch.allclose(param.detach(), initial)


def test_bsam_training_loop_preserves_initial_backward_contract() -> None:
    torch.manual_seed(20260531)
    param = torch.nn.Parameter(torch.tensor([0.8, -0.4], dtype=torch.float32))
    optimizer = _make_bsam_optimizer(param)
    loop = _ClosureLoop(optimizer, param)
    initial = param.detach().clone()
    dataloader = [
        {"x": torch.tensor([1.0, 0.5]), "target": torch.tensor([0.1, -0.2])},
        {"x": torch.tensor([0.5, -1.0]), "target": torch.tensor([-0.3, 0.4])},
    ]

    result = loop.train_epoch(dataloader, epoch=0)

    assert result["steps"] == 2, result
    assert loop.global_step == 1
    assert loop.outer_deferred_backward_calls == 0
    assert loop.bound_step_closure_calls == 1
    assert loop.closure_backward_calls == 6
    assert param.grad is None
    assert not torch.allclose(param.detach(), initial)


def main() -> int:
    test_lbfgs_training_loop_binds_recompute_closure()
    test_bsam_training_loop_preserves_initial_backward_contract()
    print("optimizer_step_closure_training_loop_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
