"""TrainingLoop canary for quantized simple variants.

This is an explicit internal canary for the runtime canary manifest.  It runs a
single TrainingLoop step through the native executor while keeping product
dispatch and request/UI/schema exposure disabled at the scorecard boundary.
"""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import patch

import torch

from core.configs import OptimizerType
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_simple_optimizer_quantized_runtime_canary_scorecard import (
    build_simple_optimizer_quantized_runtime_canary_scorecard,
)


TARGET_OPTIMIZERS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)
KIND_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit",
    OptimizerType.PAGED_LION_8BIT: "paged_lion8bit",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit",
}
EXECUTOR_REQUIRED = "turbocore_simple_quantized_optimizer_training_executor_v0"


def build_simple_optimizer_quantized_training_loop_canary_scorecard(
    *,
    runtime_canary_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build TrainingLoop canary preconditions without dispatching training."""

    runtime = dict(runtime_canary_report or build_simple_optimizer_quantized_runtime_canary_scorecard())
    cases = _run_live_cases(runtime) if torch.cuda.is_available() else []
    rows = [_row(optimizer, runtime, _case_for(optimizer, cases)) for optimizer in TARGET_OPTIMIZERS]
    manifest_ready_count = sum(1 for row in rows if row["training_loop_canary_manifest_ready"] is True)
    canary_ready_count = sum(1 for row in rows if row["training_loop_canary_ready"] is True)
    manifest_ready = manifest_ready_count == len(TARGET_OPTIMIZERS)
    canary_ready = canary_ready_count == len(TARGET_OPTIMIZERS)
    blockers = _promotion_blockers(rows, manifest_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_training_loop_canary_scorecard_v0",
        "gate": "simple_formula_quantized_training_loop_canary",
        "ok": bool(runtime.get("ok", False)) and manifest_ready and canary_ready,
        "promotion_ready": False,
        "training_loop_canary_manifest_ready": manifest_ready,
        "training_loop_canary_ready": canary_ready,
        "runtime_canary_manifest_ready": bool(runtime.get("runtime_canary_manifest_ready", False)),
        "runtime_canary_ready": canary_ready,
        "native_training_mode": str(runtime.get("native_training_mode") or "canary"),
        "optimizer_family": "simple_formula_quantized",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "rows": rows,
        "cases": cases,
        "runtime_canary_summary": dict(runtime.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "training_loop_canary_manifest_ready_count": manifest_ready_count,
            "training_loop_canary_ready_count": canary_ready_count,
            "runtime_canary_manifest_ready_count": int(
                dict(runtime.get("summary") or {}).get("runtime_canary_manifest_ready_count", 0) or 0
            ),
            "executor_implementation_ready_count": canary_ready_count,
            "native_kernel_launch_count": sum(1 for case in cases if case.get("native_kernel_launched") is True),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(
            [item for item in blockers if item not in {"e2e_no_regression_missing", "product_rollout_review_missing"}]
        ),
        "recommended_next_step": (
            "run simple quantized e2e no-regression matrix before product rollout review"
            if canary_ready
            else "implement simple quantized TrainingLoop executor and live canary"
            if manifest_ready and torch.cuda.is_available()
            else "complete quantized simple runtime canary manifest before TrainingLoop canary manifest"
        ),
        "notes": [
            "This scorecard exercises an explicit internal TrainingLoop canary and keeps the scorecard boundary default-off.",
            "The quantized simple executor owns canary-only uint8 state; product optimizer-state sync is still a later rollout gate.",
            "The manifest keeps runtime dispatch and product exposure disabled.",
        ],
    }


def _row(optimizer: OptimizerType, runtime: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    runtime_ready = _runtime_manifest_ready_for(optimizer, runtime)
    manifest_ready = runtime_ready
    canary_ready = manifest_ready and case.get("ok") is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
        "optimizer_family": "simple_formula_quantized",
        "runtime_canary_manifest_ready": runtime_ready,
        "training_loop_canary_manifest_ready": manifest_ready,
        "training_loop_canary_ready": canary_ready,
        "training_loop_executor_ready": canary_ready,
        "executor_required": EXECUTOR_REQUIRED,
        "native_training_mode": str(runtime.get("native_training_mode") or "canary"),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "request_fields": {
            "optimizer_type": optimizer.value,
            "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
            "native_training_mode": str(runtime.get("native_training_mode") or "canary"),
            "training_loop_executor": EXECUTOR_REQUIRED,
        },
        "case": dict(case),
        "missing_before_live_canary": [] if canary_ready else _missing_before_live_canary(manifest_ready, case),
        "blocked_reasons": [] if canary_ready else _row_blockers(optimizer, manifest_ready, case),
    }


def _runtime_manifest_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("route_decisions", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("runtime_canary_manifest_ready") is True)


def _promotion_blockers(rows: list[Mapping[str, Any]], manifest_ready: bool) -> list[str]:
    blockers = [reason for row in rows for reason in _strings(row.get("blocked_reasons"))]
    canary_ready = all(row.get("training_loop_canary_ready") is True for row in rows)
    if manifest_ready and not canary_ready:
        blockers.extend(
            [
                "simple_quantized_training_executor_missing",
                "simple_quantized_training_loop_executor_branch_missing",
                "simple_quantized_live_training_tensor_binding_missing",
            ]
        )
    if canary_ready:
        blockers.extend(
            [
                "e2e_no_regression_missing",
                "product_rollout_review_missing",
            ]
        )
    return blockers


class _Injector:
    def __init__(self, param: torch.nn.Parameter) -> None:
        self.param = param

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _run_live_cases(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not bool(runtime.get("runtime_canary_manifest_ready", False)):
        return []
    return [_run_case(optimizer) for optimizer in TARGET_OPTIMIZERS]


def _run_case(optimizer: OptimizerType) -> dict[str, Any]:
    loop = _make_loop(KIND_BY_OPTIMIZER[optimizer])
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        params = self._get_trainable_params()
        for item in params:
            item.grad = None
        loss = sum(((item.float() * 0.37) ** 2).mean() + item.float().mean() * 0.007 for item in params)
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        _seed_previous_gate(self)
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    runtime_payload = _runtime_payload(captured[0] if captured else {})
    executor_payload = _executor_payload(runtime_payload)
    executor_result = _as_dict(executor_payload.get("result"))
    optimizer_sync = _as_dict(executor_result.get("optimizer_state_sync"))
    native_step = runtime_payload.get("native_step_executed") is True
    native_kernel = runtime_payload.get("native_kernel_launched") is True
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and runtime_payload.get("should_call_pytorch_optimizer_step") is False
        and executor_result.get("optimizer_kind") == KIND_BY_OPTIMIZER[optimizer]
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "simple_quantized_optimizer_training_loop_native_canary_v0",
        "optimizer_type": optimizer.value,
        "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime_payload.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime_payload.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(executor_payload.get("called", False)),
        "training_executor_ok": bool(executor_payload.get("ok", False)),
        "executor_result_ok": executor_result.get("ok") is True,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "pytorch_optimizer_state_synced": executor_result.get("pytorch_optimizer_state_synced") is True,
        "optimizer_state_sync_synced": optimizer_sync.get("synced") is True,
        "optimizer_state_sync_state_tensors": int(optimizer_sync.get("state_tensors", 0) or 0),
        "optimizer_state_sync_parameter_tensors": int(optimizer_sync.get("parameter_tensors", 0) or 0),
        "optimizer_state_sync_reason": str(optimizer_sync.get("reason") or ""),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": []
        if ok
        else _dedupe(_strings(runtime_payload.get("blocked_reasons")) + ["simple_quantized_training_loop_native_step_missing"]),
    }


def _make_loop(optimizer_kind: str) -> TrainingLoop:
    param = torch.nn.Parameter(torch.linspace(-0.5, 0.5, steps=512, device="cuda", dtype=torch.float32))
    optimizer = torch.optim.SGD([param], lr=1e-2, momentum=0.9, weight_decay=0.01, nesterov=True)
    if optimizer_kind in {"lion8bit", "paged_lion8bit"}:
        optimizer.param_groups[0]["lr"] = 1e-3
        optimizer.param_groups[0]["betas"] = (0.9, 0.99)
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector(param),
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
        turbocore_native_update_quantized_optimizer_kind=optimizer_kind,
    )
    loop.total_steps = 1
    _seed_previous_gate(loop)
    return loop


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_gate()


def _explicit_gate() -> dict[str, Any]:
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


def _runtime_payload(step_info: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _case_for(optimizer: OptimizerType, cases: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    for case in cases:
        if case.get("optimizer_type") == optimizer.value:
            return case
    return {}


def _missing_before_live_canary(manifest_ready: bool, case: Mapping[str, Any]) -> list[str]:
    if not manifest_ready:
        return ["runtime_canary_manifest"]
    if not torch.cuda.is_available():
        return ["cuda"]
    if case.get("ok") is not True:
        return ["simple_quantized_training_executor", "training_loop_executor_branch", "live_training_tensor_binding"]
    return []


def _row_blockers(optimizer: OptimizerType, manifest_ready: bool, case: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not manifest_ready:
        blockers.append(f"{optimizer.value}_runtime_canary_manifest_missing")
    if not torch.cuda.is_available():
        blockers.append("cuda_required_for_simple_quantized_training_loop_canary")
    blockers.extend(_strings(case.get("blocked_reasons")))
    if manifest_ready and torch.cuda.is_available() and case.get("ok") is not True:
        blockers.append(f"{optimizer.value}_training_loop_canary_missing")
    return _dedupe(blockers)


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_simple_optimizer_quantized_training_loop_canary_scorecard"]
