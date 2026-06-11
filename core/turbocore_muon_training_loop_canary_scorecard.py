"""TrainingLoop canary for built-in Muon native dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import torch

from core.lulynx_trainer.muon_optimizer import Muon
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_muon_training_tensor_binding_canary_scorecard import (
    build_muon_training_tensor_binding_canary_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_muon_training_loop_canary_scorecard(
    *,
    training_tensor_binding_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    binding = _as_dict(training_tensor_binding_report or build_muon_training_tensor_binding_canary_scorecard())
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_muon_training_loop_canary", binding)
    case = _run_case()
    row = _row(case)
    validations = _validations(binding, case, row)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_training_loop_canary_scorecard_v0",
        "gate": "muon_model_shape_aware_training_loop_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_loop_canary_ready": ready,
        "training_loop_canary_hit": ready,
        "training_tensor_binding_canary_ready": binding.get("training_tensor_binding_canary_ready") is True,
        "training_tensor_binding_parity_ready": binding.get("training_tensor_binding_parity_ready") is True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "optimizer_family": "built_in_muon_model_shape_aware",
        "family_cases": [case],
        "rows": [row],
        "training_tensor_binding_summary": _as_dict(binding.get("summary")),
        "validations": validations,
        "summary": {
            "optimizer_count": 1,
            "training_loop_canary_ready_count": 1 if row["training_loop_canary_ready"] is True else 0,
            "training_loop_canary_hit_count": 1 if ready else 0,
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "training_executor_called_count": 1 if case.get("training_executor_called") is True else 0,
            "training_parameters_mutated_count": 1 if case.get("training_parameters_mutated") is True else 0,
            "training_tensor_binding_canary_ready_count": 1
            if binding.get("training_tensor_binding_canary_ready") is True
            else 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "muon_e2e_shadow_matrix_missing",
                "muon_canary_rollout_policy_missing",
                "muon_owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Muon e2e shadow matrix with dispatch still default-off"
            if ready
            else "fix Muon TrainingLoop canary blockers"
        ),
        "notes": [
            "This scorecard runs one explicit toy TrainingLoop canary for built-in Muon.",
            "It requires the Muon live tensor-binding canary before exercising the TrainingLoop executor.",
            "Product native dispatch remains disabled pending shadow matrix, rollout policy, and manual review.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _run_case() -> dict[str, Any]:
    loop, param = _make_loop()
    _seed_previous_gate(loop)
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        for item in self._get_trainable_params():
            item.grad = None
        loss = ((param.float() * 0.41) ** 2).mean() + param.float().mean() * 0.013
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    before = param.detach().clone()
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    after = param.detach().clone()
    if not captured:
        return _case_blocked("muon_training_loop_did_not_emit_step")
    runtime = _as_dict(captured[0].get("turbocore_native_update_dispatch_runtime"))
    training_executor = _as_dict(runtime.get("training_executor"))
    executor_result = _as_dict(training_executor.get("result"))
    first_case = _first_executor_case(executor_result)
    native_step = runtime.get("native_step_executed") is True
    native_kernel = runtime.get("native_kernel_launched") is True
    mutated = _max_abs_diff(before, after) > 0.0
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and mutated
        and runtime.get("should_call_pytorch_optimizer_step") is False
        and executor_result.get("optimizer_kind") == "muon"
        and int(first_case.get("step_after", 0) or 0) >= 1
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "muon_training_loop_native_canary_v0",
        "optimizer_kind": "muon",
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "training_parameters_mutated": mutated,
        "should_call_pytorch_optimizer_step": runtime.get("should_call_pytorch_optimizer_step") is True,
        "training_executor_called": training_executor.get("called") is True,
        "training_executor_ok": training_executor.get("ok") is True,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "step_after_native": int(first_case.get("step_after", 0) or 0),
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "param_shape": [int(dim) for dim in param.shape],
        "state_keys": list(first_case.get("state_keys", [])),
        "executor_case": first_case,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(
            _strings(runtime.get("blocked_reasons")) + ["muon_training_loop_native_step_missing"]
        ),
    }


def _make_loop() -> tuple[TrainingLoop, torch.nn.Parameter]:
    param = torch.nn.Parameter(
        torch.linspace(-0.18, 0.22, steps=16, device="cuda", dtype=torch.float32).view(4, 4).contiguous()
    )
    optimizer = Muon([{"params": [param], "use_muon": True, "lr": 1.0e-2, "weight_decay": 0.0}], lr=1.0e-2)
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
        turbocore_native_update_quantized_optimizer_kind="muon",
    )
    loop.total_steps = 1
    return loop, param


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
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True},
        },
    }


def _row(case: Mapping[str, Any]) -> dict[str, Any]:
    ready = case.get("ok") is True
    return {
        "schema_version": 1,
        "optimizer_type": "Muon",
        "family": "built_in_muon_model_shape_aware",
        "training_loop_canary_ready": ready,
        "native_step_executed": case.get("native_step_executed") is True,
        "native_kernel_launched": case.get("native_kernel_launched") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "next_gate": "muon_e2e_shadow_matrix_default_off",
        "blocked_reasons": [] if ready else ["Muon_training_loop_canary_failed"],
    }


def _validations(
    binding: Mapping[str, Any],
    case: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "training_tensor_binding_canary_ready",
            binding.get("training_tensor_binding_canary_ready") is True,
            "muon_training_tensor_binding_canary_missing",
        ),
        _validation("training_loop_case_ready", case.get("ok") is True, "muon_training_loop_case_failed"),
        _validation("training_loop_row_ready", row.get("training_loop_canary_ready") is True, "muon_training_loop_row_not_ready"),
        _validation(
            "dispatch_still_default_off",
            row.get("runtime_dispatch_ready") is False
            and row.get("native_dispatch_allowed") is False
            and row.get("training_path_enabled") is False,
            "muon_training_loop_enabled_product_dispatch",
        ),
    ]


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _first_executor_case(executor_result: Mapping[str, Any]) -> dict[str, Any]:
    cases = executor_result.get("cases")
    if isinstance(cases, list) and cases:
        return _as_dict(cases[0])
    return {}


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_training_loop_canary_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked(reason: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_muon_training_loop_canary_scorecard_v0",
        "gate": "muon_model_shape_aware_training_loop_canary",
        "ok": False,
        "promotion_ready": False,
        "training_loop_canary_ready": False,
        "training_tensor_binding_canary_ready": binding.get("training_tensor_binding_canary_ready") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "family_cases": [],
        "rows": [],
        "summary": {"training_loop_canary_ready_count": 0, "native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run Muon TrainingLoop canary on CUDA",
    }


def _case_blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "optimizer_kind": "muon",
        "native_step_executed": False,
        "native_kernel_launched": False,
        "blocked_reasons": [reason],
    }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


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


__all__ = ["build_muon_training_loop_canary_scorecard"]
