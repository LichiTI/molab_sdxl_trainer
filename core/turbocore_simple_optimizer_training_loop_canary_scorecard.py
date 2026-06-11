"""TrainingLoop canary for simple formula native optimizers.

This scorecard exercises the existing TrainingLoop/native-dispatch boundary
with an explicit internal canary.  It does not add request/UI/schema fields and
does not enable native dispatch by default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch

from core.lulynx_trainer.training_loop import TrainingLoop


TARGETS = ("lion", "sgd_nesterov")


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_simple_optimizer_training_loop_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_simple_optimizer_training_loop_canary")
    cases = [_run_case(kind) for kind in TARGETS]
    ok = all(bool(case.get("ok", False)) for case in cases)
    blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_training_loop_canary_scorecard_v0",
        "gate": "simple_formula_training_loop_native_canary",
        "ok": ok,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "native_step_count": sum(1 for case in cases if case.get("native_step_executed") is True),
            "native_kernel_launch_count": sum(1 for case in cases if case.get("native_kernel_launched") is True),
        },
        "promotion_blockers": blockers + ["product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "measure simple optimizer product canary overhead and then decide rollout scope"
            if ok
            else "fix simple optimizer TrainingLoop canary blockers"
        ),
    }


def _run_case(optimizer_kind: str) -> dict[str, Any]:
    loop = _make_loop(optimizer_kind)
    _seed_previous_gate(loop)
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        params = self._get_trainable_params()
        for param in params:
            param.grad = None
        loss = sum(((param * 0.5) ** 2).sum() for param in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    runtime = _runtime_payload(captured[0] if captured else {})
    executor = _executor_payload(runtime)
    native_step = bool(runtime.get("native_step_executed", False))
    native_kernel = bool(runtime.get("native_kernel_launched", False))
    ok = bool(result.get("steps") == 1 and native_step and native_kernel)
    return {
        "schema_version": 1,
        "ok": ok,
        "optimizer_kind": optimizer_kind,
        "probe": "simple_optimizer_training_loop_native_canary_v0",
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(executor.get("called", False)),
        "training_executor_ok": bool(executor.get("native_step_executed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(runtime.get("blocked_reasons", []) + ["training_loop_native_step_missing"]),
    }


def _make_loop(optimizer_kind: str) -> TrainingLoop:
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5, 0.25], dtype=torch.float32, device=device))
    optimizer = torch.optim.SGD([param], lr=1e-3, momentum=0.9, weight_decay=0.01, nesterov=True)
    if optimizer_kind == "lion":
        optimizer.param_groups[0]["betas"] = (0.9, 0.99)
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
        turbocore_native_update_mode="native_experimental",
        turbocore_native_update_required_shadow_passes=1,
        turbocore_native_update_allow_missing_kernel=True,
        turbocore_native_update_dispatch_enabled=True,
        turbocore_native_update_training_path_enabled=True,
        turbocore_native_update_require_native_cuda=True,
        turbocore_native_update_simple_optimizer_kind=optimizer_kind,
    )
    loop.total_steps = 1
    return loop


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_simple_gate()


def _explicit_simple_gate() -> dict[str, Any]:
    request = {
        "requested": True,
        "dispatch_allowed": True,
        "training_path_enabled": True,
        "training_path_request": {
            "request_boundary_ready": True,
            "explicit_training_path_requested": True,
        },
    }
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "rehearsal": {"would_launch_native_kernel": True},
        "recovery": {
            "default_off_recovery_bridge_ready": True,
            "training_dispatch_recovery_ready": True,
        },
        "owner_gradient_sync": _ready_contract("sync_boundary_ready", "owner_gradient_sync_preconditions_ready"),
        "training_flat_owner": _ready_contract("owner_boundary_ready", "training_flat_owner_preconditions_ready"),
        "training_dispatch_kernel": _ready_contract("kernel_boundary_ready", "training_dispatch_kernel_preconditions_ready"),
        "training_executor": {"executor_boundary_ready": True, "training_executor_preconditions_ready": True},
        "stream_lifetime_ownership": {
            "ownership_boundary_ready": True,
            "stream_lifetime_ownership_preconditions_ready": True,
        },
        "evidence": {
            "owner_native_launch_ok": True,
            "copyback_dispatch_validated": True,
            "event_chain_verified": True,
            "stream_ordering_verified": True,
            "representative_performance_gate_ready": True,
        },
        "blocked_reasons": [],
    }
    return {
        "dispatch_request": request,
        "dispatch_contract": contract,
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True},
        },
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {
        boundary: True,
        precondition: True,
        "native_supported": True,
        "training_lifecycle_integrated": True,
    }


def _runtime_payload(step_info: dict[str, Any]) -> dict[str, Any]:
    value = step_info.get("turbocore_native_update_dispatch_runtime", {})
    return dict(value) if isinstance(value, dict) else {}


def _executor_payload(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get("training_executor", {})
    return dict(value) if isinstance(value, dict) else {}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_training_loop_canary_scorecard_v0",
        "gate": "simple_formula_training_loop_native_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "cases": [],
        "summary": {"case_count": 0, "native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run simple optimizer TrainingLoop canary on CUDA",
    }


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_simple_optimizer_training_loop_canary_scorecard"]
