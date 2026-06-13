"""Per-selected TrainingLoop native canaries for plugin adaptive-LR optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import plugin_resume_case
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_plugin_adaptivelr_family_batch_scorecard import TARGET_PLUGIN_OPTIMIZERS


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_adaptivelr_training_loop_canary_scorecard.json"
NATIVE_FAMILY_BY_SELECTED = {
    "prodigy": "prodigy",
    "dadaptadagrad": "dadapt",
    "dadaptadam": "dadapt",
    "dadaptadan": "dadapt",
    "dadaptlion": "dadapt",
    "dadaptsgd": "dadapt",
}


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_plugin_adaptivelr_training_loop_canary_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        report = _blocked("cuda_required_for_plugin_adaptivelr_training_loop_canary")
    else:
        cases = [_run_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        ready = len(cases) == len(TARGET_PLUGIN_OPTIMIZERS) and all(case.get("ok") is True for case in cases)
        selected_count = len(cases)
        native_step_count = sum(1 for case in cases if case.get("native_step_executed") is True)
        native_kernel_launch_count = sum(1 for case in cases if case.get("native_kernel_launched") is True)
        training_executor_called_count = sum(1 for case in cases if case.get("training_executor_called") is True)
        skip_pytorch_count = sum(1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False)
        native_family_alias_count = len({
            str(case.get("native_family_alias") or "") for case in cases if case.get("ok") is True
        })
        report = {
            "schema_version": 1,
            "scorecard": "turbocore_plugin_adaptivelr_training_loop_canary_scorecard_v0",
            "gate": "plugin_adaptivelr_selected_training_loop_native_canary",
            "roadmap": ROADMAP,
            "ok": ready,
            "promotion_ready": False,
            "selected_native_canary_ready": ready,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
            "selected_optimizer_family": "adaptive_lr_state_machine",
            "cases": cases,
            "summary": {
                "selected_optimizer_count": selected_count,
                "case_count": selected_count,
                "native_step_count": native_step_count,
                "native_kernel_launch_count": native_kernel_launch_count,
                "training_executor_called_count": training_executor_called_count,
                "skip_pytorch_count": skip_pytorch_count,
                "native_family_alias_count": native_family_alias_count,
                "plugin_adaptivelr_training_loop_case_count": selected_count,
                "plugin_adaptivelr_training_loop_native_step_count": native_step_count,
                "plugin_adaptivelr_training_loop_native_kernel_launch_count": native_kernel_launch_count,
                "plugin_adaptivelr_training_loop_training_executor_called_count": training_executor_called_count,
                "plugin_adaptivelr_training_loop_skip_pytorch_count": skip_pytorch_count,
                "plugin_adaptivelr_training_loop_native_family_alias_count": native_family_alias_count,
                "runtime_dispatch_ready_count": 0,
                "native_dispatch_allowed_count": 0,
                "training_path_enabled_count": 0,
                "product_native_ready_count": 0,
            },
            "promotion_blockers": _dedupe(
                blockers
                + [
                    "plugin_adaptivelr_owner_release_review_missing",
                    "plugin_adaptivelr_product_training_route_not_bound",
                ]
            ),
            "blocked_reasons": blockers,
            "recommended_next_step": (
                "add adaptive-LR variant parity/replay gates before product route binding"
                if ready
                else "fix selected plugin adaptive-LR TrainingLoop native canary blockers"
            ),
            "notes": [
                "Each selected adaptive-LR plugin optimizer runs its own TrainingLoop native canary.",
                "DAdapt selected optimizers currently share the dadapt native family kernel alias.",
                "This is CUDA/native canary evidence only; product dispatch remains default-off.",
            ],
        }
    if write_artifact:
        _write_artifact(report)
    return report


def _run_case(selected_optimizer_name: str) -> dict[str, Any]:
    loop, param, optimizer = _make_loop(selected_optimizer_name)
    _prime_optimizer_state(param, optimizer)
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
        loss = sum(((item.float() * 0.43) ** 2).mean() + item.float().mean() * 0.008 for item in self._get_trainable_params())
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    if not captured:
        return _case_blocked(selected_optimizer_name, "plugin_adaptivelr_training_loop_did_not_emit_step")
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = _as_dict(training_executor.get("result")) if isinstance(training_executor, Mapping) else {}
    first_executor_case = _first_executor_case(executor_result)
    native_step = runtime.get("native_step_executed") is True
    native_kernel = runtime.get("native_kernel_launched") is True
    expected_alias = NATIVE_FAMILY_BY_SELECTED[selected_optimizer_name]
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and runtime.get("should_call_pytorch_optimizer_step") is False
        and executor_result.get("optimizer_kind") == expected_alias
        and executor_result.get("adaptive_lr_family") == _family_for_alias(expected_alias)
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "selected_optimizer_name": selected_optimizer_name,
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "optimizer_class": type(optimizer).__name__,
        "native_family_alias": expected_alias,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "executor_family": str(executor_result.get("adaptive_lr_family") or ""),
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": runtime.get("should_call_pytorch_optimizer_step") is True,
        "fallback_to_pytorch_required": runtime.get("fallback_to_pytorch_required") is True,
        "training_executor_called": training_executor.get("called") is True if isinstance(training_executor, Mapping) else False,
        "training_executor_ok": training_executor.get("ok") is True if isinstance(training_executor, Mapping) else False,
        "executor_result_ok": executor_result.get("ok") is True,
        "executor_case": first_executor_case,
        "step_after_native": int(first_executor_case.get("step_after", 0) or 0),
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "source_scorecard": "turbocore_plugin_adaptivelr_training_loop_canary_scorecard_v0",
        "blocked_reasons": [] if ok else _dedupe(
            _strings(runtime.get("blocked_reasons"))
            + [f"plugin_{selected_optimizer_name}_adaptivelr_training_loop_native_step_missing"]
        ),
    }


def _make_loop(selected_optimizer_name: str) -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    param = torch.nn.Parameter(torch.linspace(-0.18, 0.22, steps=128, device="cuda", dtype=torch.float32))
    optimizer = _create_selected_plugin_optimizer(selected_optimizer_name, param)
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
        turbocore_native_update_quantized_optimizer_kind=selected_optimizer_name,
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _create_selected_plugin_optimizer(selected_optimizer_name: str, param: torch.nn.Parameter) -> torch.optim.Optimizer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    case = plugin_resume_case(selected_optimizer_name)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=1.0,
        weight_decay=0.01,
        optimizer_args=case.optimizer_args,
        lr_scheduler=SchedulerType.COSINE,
        warmup_ratio=0.0,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector([param])
    trainer.model = None
    trainer.trainable_params = []
    trainer._block_weight_manager = None
    trainer._easy_control = None
    trainer._ip_adapter = None
    trainer._repa_projector = None
    trainer._advanced_optimizer_strategy_profile = {}
    trainer._optimizer_backend_profile = {}
    trainer._log_messages = []
    trainer._log = lambda msg: trainer._log_messages.append(str(msg))
    return trainer._create_optimizer()


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
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True},
        },
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _runtime_payload(step_info: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _first_executor_case(executor_result: Mapping[str, Any]) -> dict[str, Any]:
    cases = executor_result.get("cases")
    if isinstance(cases, list) and cases and isinstance(cases[0], Mapping):
        return dict(cases[0])
    return {}


def _family_for_alias(alias: str) -> str:
    return "adaptive_lr_prodigy" if alias == "prodigy" else "adaptive_lr_dadapt"


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adaptivelr_training_loop_canary_scorecard_v0",
        "gate": "plugin_adaptivelr_selected_training_loop_native_canary",
        "roadmap": ROADMAP,
        "ok": False,
        "promotion_ready": False,
        "selected_native_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "cases": [],
        "summary": {
            "selected_optimizer_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "case_count": 0,
            "native_step_count": 0,
            "native_kernel_launch_count": 0,
            "training_path_enabled_count": 0,
            "native_dispatch_allowed_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run selected plugin adaptive-LR TrainingLoop native canaries on CUDA",
    }


def _case_blocked(selected_optimizer_name: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "selected_optimizer_name": selected_optimizer_name,
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "native_family_alias": NATIVE_FAMILY_BY_SELECTED.get(selected_optimizer_name, ""),
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [reason],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "ARTIFACT_NAME",
    "NATIVE_FAMILY_BY_SELECTED",
    "ROADMAP",
    "build_plugin_adaptivelr_training_loop_canary_scorecard",
]
