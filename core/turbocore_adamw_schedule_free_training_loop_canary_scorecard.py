"""Report-only TrainingLoop canary manifest for AdamWScheduleFree."""

from __future__ import annotations

import importlib.util
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import torch

from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_adamw_schedule_free_native_scratch_kernel_scorecard import (
    ENTRYPOINT,
    KERNEL_NAME,
    build_adamw_schedule_free_native_scratch_kernel_scorecard,
)
from core.turbocore_adamw_schedule_free_runtime_canary_scorecard import (
    build_adamw_schedule_free_runtime_canary_scorecard,
)
from core.turbocore_adamw_schedule_free_training_executor import ENTRYPOINT as LIVE_ENTRYPOINT
from core.turbocore_adamw_schedule_free_training_executor import build_adamw_schedule_free_training_executor
from core.turbocore_adamw_schedule_free_training_tensor_binding_canary_scorecard import (
    build_adamw_schedule_free_training_tensor_binding_canary_scorecard,
)
from core.services.native_module_loader import native_with_entrypoints


def build_adamw_schedule_free_training_loop_canary_scorecard() -> dict[str, Any]:
    """Expose AdamWScheduleFree TrainingLoop canary status without dispatch."""

    native_scratch = build_adamw_schedule_free_native_scratch_kernel_scorecard()
    runtime = build_adamw_schedule_free_runtime_canary_scorecard()
    tensor_binding = build_adamw_schedule_free_training_tensor_binding_canary_scorecard()
    live_native = native_with_entrypoints(LIVE_ENTRYPOINT)
    training_executor_ready = callable(build_adamw_schedule_free_training_executor)
    live_entrypoint_ready = live_native is not None and tensor_binding.get("native_live_entrypoint_ready") is True
    tensor_binding_ready = tensor_binding.get("training_tensor_binding_ready") is True
    unsafe = _unsafe_claims(native_scratch, runtime)
    blockers = _dedupe(
        unsafe
        + _strings(native_scratch.get("blocked_reasons"))
        + _strings(runtime.get("blocked_reasons"))
    )
    foundation_ready = (
        native_scratch.get("native_scratch_kernel_parity_ready") is True
        and runtime.get("runtime_canary_manifest_ready") is True
        and not blockers
    )
    case = _run_training_loop_case() if foundation_ready and tensor_binding_ready else _case_blocked(
        "adamw_schedule_free_training_loop_foundation_missing"
    )
    loop_ready = bool(case.get("ok", False))
    loop_blockers = _strings(case.get("blocked_reasons"))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_training_loop_canary_scorecard_v0",
        "gate": "adamw_schedule_free_training_loop_canary_manifest",
        "ok": foundation_ready and loop_ready,
        "promotion_ready": False,
        "training_loop_canary_manifest_ready": foundation_ready,
        "training_loop_canary_ready": loop_ready,
        "training_loop_canary_hit": loop_ready,
        "runtime_canary_manifest_ready": runtime.get("runtime_canary_manifest_ready") is True,
        "runtime_canary_ready": False,
        "runtime_dispatch_ready": False,
        "training_executor_ready": training_executor_ready,
        "training_tensor_binding_ready": tensor_binding_ready,
        "native_live_entrypoint_ready": live_entrypoint_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "adamw_schedule_free",
        "optimizer_family": "adamw_schedule_free",
        "native_route": "rust_cuda_adamw_schedule_free_scratch_v0",
        "entrypoint": ENTRYPOINT,
        "kernel_name": KERNEL_NAME,
        "native_scratch_kernel": _compact_native_scratch(native_scratch),
        "runtime_canary": _compact_runtime(runtime),
        "training_tensor_binding_canary": _compact_tensor_binding(tensor_binding),
        "case": case,
        "summary": {
            "foundation_ready": foundation_ready,
            "formula_canary_ready": _formula_canary_ready(native_scratch),
            "native_scratch_kernel_ready": native_scratch.get("native_scratch_kernel_parity_ready") is True,
            "runtime_canary_manifest_ready": runtime.get("runtime_canary_manifest_ready") is True,
            "training_loop_canary_ready": loop_ready,
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "primed_pytorch_state": bool(case.get("primed_pytorch_state", False)),
            "step_after_native": case.get("step_after_native"),
            "training_executor_ready": training_executor_ready,
            "native_live_entrypoint_ready": live_entrypoint_ready,
            "training_tensor_binding_ready": tensor_binding_ready,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + loop_blockers
            + [
                *([] if live_entrypoint_ready else ["adamw_schedule_free_live_tensor_entrypoint_missing"]),
                *([] if tensor_binding_ready else ["adamw_schedule_free_training_tensor_binding_missing"]),
                "product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": _dedupe(blockers + loop_blockers),
        "recommended_next_step": (
            "add AdamWScheduleFree e2e shadow matrix and rollout policy"
            if foundation_ready and loop_ready
            else "wire AdamWScheduleFree TrainingLoop native dispatch canary with dispatch still default-off"
            if foundation_ready and tensor_binding_ready
            else "fix AdamWScheduleFree formula/native scratch/runtime manifest blockers before TrainingLoop canary"
        ),
        "notes": [
            "This scorecard runs a toy explicit canary only and keeps product dispatch disabled.",
            "AdamWScheduleFree formula, native scratch, and live tensor binding evidence are prerequisites.",
            "Product native rollout remains pending manual review.",
        ],
    }


class _Injector:
    def __init__(self, param: torch.nn.Parameter) -> None:
        self.param = param

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _run_training_loop_case() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _case_blocked("cuda_required_for_adamw_schedule_free_training_loop_canary")
    if importlib.util.find_spec("schedulefree") is None:
        return _case_blocked("schedulefree_required_for_adamw_schedule_free_training_loop_canary")
    loop, param, optimizer = _make_loop()
    _prime_optimizer_state(param, optimizer)
    _seed_previous_gate(loop)
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
        loss = sum(((item.float() * 0.53) ** 2).mean() + item.float().mean() * 0.013 for item in params)
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    if not captured:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adamw_schedule_free_training_loop_native_canary_v0",
            "result": result,
            "captured_step_count": 0,
            "blocked_reasons": ["adamw_schedule_free_training_loop_did_not_emit_step"],
        }
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = _as_dict(training_executor.get("result")) if isinstance(training_executor, dict) else {}
    native_step = bool(runtime.get("native_step_executed", False))
    native_kernel = bool(runtime.get("native_kernel_launched", False))
    k_after = int(optimizer.param_groups[0].get("k", optimizer.state[param].get("step", 0)) or 0)
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and not bool(runtime.get("should_call_pytorch_optimizer_step", True))
        and k_after == 2
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "adamw_schedule_free_training_loop_native_canary_v0",
        "result": result,
        "captured_step_count": len(captured),
        "primed_pytorch_state": True,
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(training_executor.get("called", False)) if isinstance(training_executor, dict) else False,
        "training_executor_ok": bool(training_executor.get("ok", False)) if isinstance(training_executor, dict) else False,
        "executor_result_ok": bool(executor_result.get("ok", False)),
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "step_after_native": k_after,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(
            _strings(runtime.get("blocked_reasons")) + ["adamw_schedule_free_training_loop_native_step_missing"]
        ),
    }


def _make_loop() -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    import schedulefree

    param = torch.nn.Parameter(torch.linspace(-0.25, 0.35, steps=4096, device="cuda", dtype=torch.float32))
    optimizer = schedulefree.AdamWScheduleFree(
        [param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        warmup_steps=4,
        r=0.0,
        weight_lr_power=2.0,
        foreach=False,
    )
    optimizer.train()
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
        turbocore_native_update_quantized_optimizer_kind="adamw_schedule_free",
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _prime_optimizer_state(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    if hasattr(optimizer, "train"):
        optimizer.train()
    param.grad = None
    loss = ((param.float() * 0.31) ** 2).mean() + param.float().mean() * 0.007
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_gate()


def _explicit_gate() -> dict[str, Any]:
    request = {
        "requested": True,
        "dispatch_allowed": True,
        "training_path_enabled": True,
        "training_path_request": {"request_boundary_ready": True, "explicit_training_path_requested": True},
    }
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "rehearsal": {"would_launch_native_kernel": True},
        "recovery": {"default_off_recovery_bridge_ready": True, "training_dispatch_recovery_ready": True},
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
        "kernel_launch_plan": {"launch_allowed": True, "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True}},
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _runtime_payload(step_info: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _compact_native_scratch(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "native_scratch_kernel_parity_ready": report.get("native_scratch_kernel_parity_ready") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "summary": dict(report.get("summary") or {}),
        "scratch_canary_summary": dict(report.get("scratch_canary_summary") or {}),
    }


def _compact_runtime(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "runtime_canary_manifest_ready": report.get("runtime_canary_manifest_ready") is True,
        "runtime_canary_ready": report.get("runtime_canary_ready") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "summary": dict(report.get("summary") or {}),
    }


def _compact_tensor_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "scorecard": str(report.get("scorecard", "")),
        "training_tensor_binding_ready": report.get("training_tensor_binding_ready") is True,
        "native_live_entrypoint_ready": report.get("native_live_entrypoint_ready") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "summary": dict(report.get("summary") or {}),
    }


def _formula_canary_ready(report: Mapping[str, Any]) -> bool:
    summary = report.get("scratch_canary_summary")
    if not isinstance(summary, Mapping):
        return False
    case_count = int(summary.get("case_count", 0) or 0)
    passed_count = int(summary.get("passed_case_count", 0) or 0)
    return case_count > 0 and passed_count == case_count


def _unsafe_claims(*reports: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for report in reports:
        scorecard = str(report.get("scorecard", "unknown_scorecard"))
        for field in ("training_path_enabled", "default_behavior_changed", "runtime_dispatch_ready", "native_dispatch_allowed"):
            if report.get(field) is True:
                out.append(f"unsafe_adamw_schedule_free_training_loop_claim:{scorecard}:{field}")
    return out


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _case_blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "probe": "adamw_schedule_free_training_loop_native_canary_v0",
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [reason],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adamw_schedule_free_training_loop_canary_scorecard"]
