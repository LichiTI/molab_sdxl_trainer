"""TrainingLoop canary for built-in adaptive-LR native dispatch."""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import patch

import torch

from core.lulynx_trainer.auto_prodigy_optimizer import AutoProdigy
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_adaptive_lr_runtime_dispatch_shadow_scorecard import (
    build_adaptive_lr_runtime_dispatch_shadow_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


FAMILY_CASES = (
    ("adaptive_lr_prodigy", "prodigy"),
    ("adaptive_lr_dadapt", "dadapt"),
)


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_adaptive_lr_training_loop_canary_scorecard(
    *,
    runtime_dispatch_shadow_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shadow = dict(runtime_dispatch_shadow_report or build_adaptive_lr_runtime_dispatch_shadow_scorecard())
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_adaptive_lr_training_loop_canary", shadow)
    cases = [_run_case(family, optimizer_kind) for family, optimizer_kind in FAMILY_CASES]
    rows = [_row(case.optimizer.value, cases) for case in TARGET_CASES]
    validations = _validations(shadow, cases, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_training_loop_canary_scorecard_v0",
        "gate": "adaptive_lr_training_loop_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_loop_canary_ready": ready,
        "training_loop_canary_hit": ready,
        "runtime_dispatch_shadow_ready": shadow.get("runtime_dispatch_shadow_ready") is True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "optimizer_family": "built_in_adaptive_lr",
        "family_cases": cases,
        "rows": rows,
        "runtime_dispatch_shadow_summary": dict(shadow.get("summary") or {}),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "family_case_count": len(cases),
            "family_passed_case_count": sum(1 for case in cases if case.get("ok") is True),
            "training_loop_canary_ready_count": sum(1 for row in rows if row["training_loop_canary_ready"] is True),
            "native_step_count": sum(1 for case in cases if case.get("native_step_executed") is True),
            "native_kernel_launch_count": sum(1 for case in cases if case.get("native_kernel_launched") is True),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_end_to_end_shadow_matrix_missing",
                "adaptive_lr_canary_rollout_policy_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add adaptive-LR e2e shadow matrix with dispatch still default-off"
            if ready
            else "fix adaptive-LR TrainingLoop canary blockers"
        ),
        "notes": [
            "This scorecard runs explicit toy TrainingLoop canaries only.",
            "It uses Prodigy-family and DAdapt-family representatives to cover the 11 built-in adaptive-LR rows.",
            "Product native dispatch remains disabled pending shadow matrix, rollout policy, and manual review.",
        ],
    }


def _run_case(family: str, optimizer_kind: str) -> dict[str, Any]:
    loop, param, optimizer, optimizer_backend = _make_loop(optimizer_kind)
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
        loss = sum(((item.float() * 0.47) ** 2).mean() + item.float().mean() * 0.009 for item in params)
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    if not captured:
        return _case_blocked(family, optimizer_kind, "adaptive_lr_training_loop_did_not_emit_step")
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = _as_dict(training_executor.get("result")) if isinstance(training_executor, Mapping) else {}
    first_case = _first_executor_case(executor_result)
    native_step = runtime.get("native_step_executed") is True
    native_kernel = runtime.get("native_kernel_launched") is True
    step_after = int(first_case.get("step_after", 0) or 0)
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and runtime.get("should_call_pytorch_optimizer_step") is False
        and step_after >= 2
        and executor_result.get("adaptive_lr_family") == family
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "adaptive_lr_training_loop_native_canary_v0",
        "family": family,
        "optimizer_kind": optimizer_kind,
        "optimizer_backend": optimizer_backend,
        "result": result,
        "captured_step_count": len(captured),
        "primed_pytorch_state": True,
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": runtime.get("should_call_pytorch_optimizer_step") is True,
        "training_executor_called": training_executor.get("called") is True if isinstance(training_executor, Mapping) else False,
        "training_executor_ok": training_executor.get("ok") is True if isinstance(training_executor, Mapping) else False,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "executor_family": str(executor_result.get("adaptive_lr_family") or ""),
        "step_after_native": step_after,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "executor_case": first_case,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(
            _strings(runtime.get("blocked_reasons")) + ["adaptive_lr_training_loop_native_step_missing"]
        ),
    }


def _make_loop(optimizer_kind: str) -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer, str]:
    param = torch.nn.Parameter(torch.linspace(-0.18, 0.22, steps=128, device="cuda", dtype=torch.float32))
    optimizer, backend = _make_optimizer(optimizer_kind, param)
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
        turbocore_native_update_quantized_optimizer_kind=optimizer_kind,
    )
    loop.total_steps = 1
    return loop, param, optimizer, backend


def _make_optimizer(optimizer_kind: str, param: torch.nn.Parameter) -> tuple[torch.optim.Optimizer, str]:
    if optimizer_kind == "prodigy":
        return AutoProdigy([param], lr=1.0, d0=1e-5, growth_rate=1.01, weight_decay=0.01), "AutoProdigy"
    try:
        from dadaptation import DAdaptAdam

        return DAdaptAdam([param], lr=1.0, weight_decay=0.01), "dadaptation.DAdaptAdam"
    except Exception:
        return torch.optim.AdamW([param], lr=1e-3, weight_decay=0.01), "torch.optim.AdamW_fallback_for_canary"


def _prime_optimizer_state(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    if hasattr(optimizer, "train"):
        try:
            optimizer.train()  # type: ignore[attr-defined]
        except Exception:
            pass
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


def _row(optimizer_type: str, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    family = _family(optimizer_type)
    case = next((item for item in cases if item.get("family") == family), {})
    ready = case.get("ok") is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "training_loop_canary_ready": ready,
        "native_step_executed": case.get("native_step_executed") is True,
        "native_kernel_launched": case.get("native_kernel_launched") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "next_gate": "adaptive_lr_e2e_shadow_matrix",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_training_loop_canary_failed"],
    }


def _validations(
    shadow: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _validation("runtime_dispatch_shadow_ready", shadow.get("runtime_dispatch_shadow_ready") is True, "adaptive_lr_runtime_dispatch_shadow_missing"),
        _validation("family_cases_ready", all(case.get("ok") is True for case in cases), "adaptive_lr_training_loop_family_case_failed"),
        _validation("all_rows_ready", all(row.get("training_loop_canary_ready") is True for row in rows), "adaptive_lr_training_loop_row_not_ready"),
        _validation(
            "runtime_dispatch_still_default_off",
            shadow.get("runtime_dispatch_ready") is False
            and shadow.get("native_dispatch_allowed") is False
            and shadow.get("training_path_enabled") is False
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows)
            and all(row.get("training_path_enabled") is False for row in rows),
            "adaptive_lr_training_loop_enabled_product_dispatch",
        ),
    ]


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _runtime_payload(step_info: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _first_executor_case(executor_result: Mapping[str, Any]) -> dict[str, Any]:
    cases = executor_result.get("cases")
    if isinstance(cases, list) and cases:
        return _as_dict(cases[0])
    return {}


def _family(optimizer_type: str) -> str:
    if optimizer_type in {"AutoProdigy", "prodigy", "prodigyplus.ProdigyPlusScheduleFree"}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _blocked(reason: str, shadow: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_training_loop_canary_scorecard_v0",
        "gate": "adaptive_lr_training_loop_canary",
        "ok": False,
        "promotion_ready": False,
        "training_loop_canary_ready": False,
        "runtime_dispatch_shadow_ready": shadow.get("runtime_dispatch_shadow_ready") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "family_cases": [],
        "rows": [],
        "summary": {"training_loop_canary_ready_count": 0, "native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run adaptive-LR TrainingLoop canary on CUDA",
    }


def _case_blocked(family: str, optimizer_kind: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "family": family,
        "optimizer_kind": optimizer_kind,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "blocked_reasons": [reason],
    }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adaptive_lr_training_loop_canary_scorecard"]
