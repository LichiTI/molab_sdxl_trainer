"""Report-only scorecard for AdamW-like optimizer variant state layouts.

V2-P8 studies quantized, paged, Kahan, and schedule-free AdamW variants
before any native kernel work.  It classifies state layout, resume behavior,
and memory/speed risk without enabling optimizer dispatch.
"""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from core.configs import OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


TARGET_OPTIMIZERS = (
    OptimizerType.ADAMW_8BIT,
    OptimizerType.PAGED_ADAMW,
    OptimizerType.PAGED_ADAMW_32BIT,
    OptimizerType.PAGED_ADAMW_8BIT,
    OptimizerType.KAHAN_ADAMW_8BIT,
    OptimizerType.ADAMW_SCHEDULE_FREE,
)


@dataclass(frozen=True)
class _ProbeCase:
    optimizer: OptimizerType
    device: str
    optimizer_args: str = ""


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_adamw_variant_state_scorecard(
    *,
    optimizers: Sequence[OptimizerType] = TARGET_OPTIMIZERS,
    run_cuda_optional: bool = True,
) -> dict[str, Any]:
    """Build a P8 report without changing training behavior."""

    rows = [_layout_row(optimizer) for optimizer in optimizers]
    missing = [row["optimizer_type"] for row in rows if row["state_layout_status"] == "unclassified"]
    resume_cases = _resume_cases(optimizers, run_cuda_optional=run_cuda_optional)
    resume = [_run_resume_probe(case) for case in resume_cases]
    hard_failures = [case for case in resume if case["status"] == "failed"]
    matrix = [_memory_speed_row(row) for row in rows]
    live_resume_count = sum(1 for case in resume if case["status"] == "passed")
    layout_ready = not missing and all(row["state_layout_status"] != "unknown" for row in rows)
    resume_ready = not hard_failures and any(
        case["optimizer_type"] == OptimizerType.KAHAN_ADAMW_8BIT.value and case["status"] == "passed"
        for case in resume
    )
    matrix_ready = len(matrix) == len(rows) and all(row["memory_speed_matrix_ready"] for row in matrix)
    blockers = _blocked_reasons(missing, hard_failures, resume_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_variant_state_scorecard_v0",
        "gate": "adamw_variants_quantized_paged_state",
        "ok": not hard_failures and not missing,
        "promotion_ready": False,
        "state_layout_stage_ready": layout_ready,
        "resume_matrix_stage_ready": resume_ready,
        "memory_speed_matrix_ready": matrix_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in optimizers],
        "rows": rows,
        "resume_probes": resume,
        "memory_speed_matrix": matrix,
        "summary": {
            "target_optimizer_count": len(rows),
            "classified_optimizer_count": len(rows) - len(missing),
            "live_resume_probe_count": live_resume_count,
            "skipped_resume_probe_count": sum(1 for case in resume if case["status"] == "skipped"),
            "fallback_resume_probe_count": sum(1 for case in resume if case["status"] == "fallback_detected"),
            "hard_failure_count": len(hard_failures),
            "estimated_memory_win_count": sum(
                1 for row in matrix if float(row.get("estimated_state_memory_ratio_vs_adamw", 1.0)) < 1.0
            ),
            "paged_variant_count": sum(1 for row in rows if row["state_schema"].get("paged_state") is True),
            "quantized_variant_count": sum(1 for row in rows if row["state_schema"].get("quantized_state") is True),
        },
        "promotion_blockers": blockers + ["native_kernel_missing", "runtime_canary_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(resume_ready, matrix),
        "notes": [
            "This scorecard is report-only and never enables native optimizer dispatch.",
            "Paged and bitsandbytes rows are allowed to skip live resume probes when CUDA or bitsandbytes is unavailable.",
            "KahanAdamW8bit is a local optimizer, so it is the required live resume anchor for this stage.",
        ],
    }


def _layout_row(optimizer: OptimizerType) -> dict[str, Any]:
    specs: dict[OptimizerType, dict[str, Any]] = {
        OptimizerType.ADAMW_8BIT: {
            "family": "adamw_quantized",
            "implementation": "bitsandbytes.optim.AdamW8bit",
            "required_dependency": "bitsandbytes",
            "state_keys": ["step", "exp_avg", "exp_avg_sq", "quantization_map"],
            "state_schema": _schema("uint8_blockwise", quantized=True),
            "resume_risk": "medium",
            "native_kernel_strategy": "separate_dequant_update_requant_kernel",
        },
        OptimizerType.PAGED_ADAMW: {
            "family": "adamw_paged",
            "implementation": "bitsandbytes.optim.PagedAdamW",
            "required_dependency": "bitsandbytes",
            "state_keys": ["step", "exp_avg", "exp_avg_sq", "paged_allocator"],
            "state_schema": _schema("float32_or_param_dtype", paged=True),
            "resume_risk": "medium_high",
            "native_kernel_strategy": "paged_state_residency_contract_first",
        },
        OptimizerType.PAGED_ADAMW_32BIT: {
            "family": "adamw_paged",
            "implementation": "bitsandbytes.optim.PagedAdamW32bit",
            "required_dependency": "bitsandbytes",
            "state_keys": ["step", "exp_avg", "exp_avg_sq", "paged_allocator"],
            "state_schema": _schema("float32", paged=True),
            "resume_risk": "medium_high",
            "native_kernel_strategy": "paged_state_residency_contract_first",
        },
        OptimizerType.PAGED_ADAMW_8BIT: {
            "family": "adamw_quantized_paged",
            "implementation": "bitsandbytes.optim.PagedAdamW8bit",
            "required_dependency": "bitsandbytes",
            "state_keys": ["step", "exp_avg", "exp_avg_sq", "quantization_map", "paged_allocator"],
            "state_schema": _schema("uint8_blockwise", quantized=True, paged=True),
            "resume_risk": "high",
            "native_kernel_strategy": "state_page_residency_plus_dequant_update_requant",
        },
        OptimizerType.KAHAN_ADAMW_8BIT: {
            "family": "adamw_quantized_kahan",
            "implementation": "core.lulynx_trainer.kahan_adamw8bit.KahanAdamW8bit",
            "required_dependency": "",
            "state_keys": ["step", "exp_avg_q", "exp_avg_sq_q", "kahan_comp"],
            "state_schema": _schema("uint8_blockwise", quantized=True, kahan=True),
            "resume_risk": "medium",
            "native_kernel_strategy": "local_layout_can_drive_first_native_quantized_adamw_probe",
        },
        OptimizerType.ADAMW_SCHEDULE_FREE: {
            "family": "adamw_schedule_free",
            "implementation": "schedulefree.AdamWScheduleFree",
            "required_dependency": "schedulefree",
            "state_keys": ["step", "z", "exp_avg_sq", "train_mode", "schedule_weight"],
            "state_schema": _schema("float32_state_machine", scheduler_coupled=True),
            "resume_risk": "high",
            "native_kernel_strategy": "state_machine_reference_before_kernel",
        },
    }
    spec = specs.get(optimizer)
    if spec is None:
        return {
            "optimizer_type": optimizer.value,
            "family": "unknown",
            "state_layout_status": "unclassified",
            "state_schema": {},
            "training_path_enabled": False,
            "default_behavior_changed": False,
        }
    dependency = str(spec.get("required_dependency", ""))
    return {
        "optimizer_type": optimizer.value,
        "family": spec["family"],
        "implementation": spec["implementation"],
        "required_dependency": dependency,
        "dependency_available": _module_available(dependency) if dependency else True,
        "state_keys": list(spec["state_keys"]),
        "state_schema": dict(spec["state_schema"]),
        "state_layout_status": "layout_reference_ready",
        "resume_risk": spec["resume_risk"],
        "native_kernel_strategy": spec["native_kernel_strategy"],
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
    }


def _schema(
    state_dtype: str,
    *,
    quantized: bool = False,
    paged: bool = False,
    kahan: bool = False,
    scheduler_coupled: bool = False,
) -> dict[str, Any]:
    return {
        "math_family": "adamw",
        "state_dtype": state_dtype,
        "param_dtype": "float32|float16|bfloat16",
        "master_weight_required": "dtype_dependent",
        "quantized_state": quantized,
        "paged_state": paged,
        "kahan_compensation": kahan,
        "scheduler_coupled": scheduler_coupled,
        "decoupled_weight_decay": True,
    }


def _resume_cases(optimizers: Sequence[OptimizerType], *, run_cuda_optional: bool) -> list[_ProbeCase]:
    cases: list[_ProbeCase] = []
    if OptimizerType.KAHAN_ADAMW_8BIT in optimizers:
        cases.append(_ProbeCase(OptimizerType.KAHAN_ADAMW_8BIT, "cpu"))
    if OptimizerType.ADAMW_SCHEDULE_FREE in optimizers:
        cases.append(_ProbeCase(OptimizerType.ADAMW_SCHEDULE_FREE, "cpu"))
    if run_cuda_optional:
        for optimizer in (
            OptimizerType.ADAMW_8BIT,
            OptimizerType.PAGED_ADAMW,
            OptimizerType.PAGED_ADAMW_32BIT,
            OptimizerType.PAGED_ADAMW_8BIT,
        ):
            if optimizer in optimizers:
                cases.append(_ProbeCase(optimizer, "cuda"))
    return cases


def _run_resume_probe(case: _ProbeCase) -> dict[str, Any]:
    if case.device == "cuda" and not torch.cuda.is_available():
        return _skipped_probe(case, "cuda_unavailable")
    dependency = _dependency_for(case.optimizer)
    if dependency and not _module_available(dependency):
        return _skipped_probe(case, f"{dependency}_unavailable")
    try:
        value = _initial_tensor(case.device)
        grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), device=value.device).reshape_as(value)
        grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), device=value.device).reshape_as(value)
        trainer = _make_trainer(case, value)
        optimizer = trainer._create_optimizer()
        optimizer_name = type(optimizer).__name__
        requested = case.optimizer.value
        if _fallback_detected(case.optimizer, optimizer_name):
            return _fallback_probe(case, optimizer_name, trainer)
        param = trainer.lora_injector.param
        _step(param, optimizer, grad1)
        state_summary = _state_summary(optimizer.state_dict())
        saved_state = copy.deepcopy(optimizer.state_dict())
        saved_param = param.detach().clone()

        restored = _make_trainer(case, saved_param)
        restored_optimizer = restored._create_optimizer()
        restored_param = restored.lora_injector.param
        restored_optimizer.load_state_dict(saved_state)
        _step(param, optimizer, grad2)
        _step(restored_param, restored_optimizer, grad2)
        diff = _max_abs(param.detach(), restored_param.detach())
        passed = diff <= 1e-5
        return {
            "schema_version": 1,
            "optimizer_type": requested,
            "status": "passed" if passed else "failed",
            "device": case.device,
            "optimizer_class": optimizer_name,
            "max_resume_diff": diff,
            "tolerance": 1e-5,
            "state_summary": state_summary,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": [] if passed else [f"{requested}_resume_parity_failed"],
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "optimizer_type": case.optimizer.value,
            "status": "failed",
            "device": case.device,
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": [f"{case.optimizer.value}_resume_probe_failed:{type(exc).__name__}"],
        }


def _make_trainer(case: _ProbeCase, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = UnifiedTrainingConfig(
        optimizer_type=case.optimizer,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args=case.optimizer_args,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector(value)
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
    trainer._attach_optimizer_profiles_to_training_loop = lambda: None
    return trainer


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _initial_tensor(device: str) -> torch.Tensor:
    return torch.linspace(-0.25, 0.35, steps=16, device=torch.device(device)).reshape(4, 4)


def _state_summary(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    tensor_dtypes: list[str] = []
    non_tensor_types: list[str] = []
    tensor_bytes = 0
    state_keys: set[str] = set()
    for payload in state.values():
        if not isinstance(payload, Mapping):
            non_tensor_types.append(type(payload).__name__)
            continue
        for key, value in payload.items():
            state_keys.add(str(key))
            if torch.is_tensor(value):
                tensor_dtypes.append(str(value.dtype).replace("torch.", ""))
                tensor_bytes += int(value.numel() * value.element_size())
            else:
                non_tensor_types.append(type(value).__name__)
    return {
        "state_entry_count": len(state),
        "state_keys": sorted(state_keys),
        "tensor_dtypes": sorted(set(tensor_dtypes)),
        "non_tensor_types": sorted(set(non_tensor_types)),
        "tensor_bytes": tensor_bytes,
        "contains_custom_objects": any(name not in {"int", "float", "bool", "str", "NoneType"} for name in non_tensor_types),
    }


def _memory_speed_row(layout: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = str(layout.get("optimizer_type", ""))
    schema = dict(layout.get("state_schema", {}) or {})
    if schema.get("quantized_state") and schema.get("kahan_compensation"):
        ratio = 0.754
        speed_risk = "medium"
    elif schema.get("quantized_state"):
        ratio = 0.254
        speed_risk = "medium_high" if schema.get("paged_state") else "medium"
    elif schema.get("paged_state"):
        ratio = 1.0
        speed_risk = "medium_high"
    elif schema.get("scheduler_coupled"):
        ratio = 1.25
        speed_risk = "medium"
    else:
        ratio = 1.0
        speed_risk = "low"
    return {
        "optimizer_type": optimizer,
        "memory_speed_matrix_ready": True,
        "estimated_state_memory_ratio_vs_adamw": ratio,
        "expected_vram_effect": _vram_effect(ratio, bool(schema.get("paged_state"))),
        "expected_speed_risk": speed_risk,
        "kernel_complexity": _kernel_complexity(schema),
        "resume_risk": str(layout.get("resume_risk", "")),
    }


def _vram_effect(ratio: float, paged: bool) -> str:
    if paged and ratio < 1.0:
        return "large_vram_residency_reduction_with_page_migration_risk"
    if paged:
        return "vram_residency_reduction_without_total_state_size_reduction"
    if ratio < 0.5:
        return "large_state_memory_reduction"
    if ratio < 1.0:
        return "moderate_state_memory_reduction"
    if ratio > 1.0:
        return "higher_state_memory_for_state_machine"
    return "no_state_memory_reduction"


def _kernel_complexity(schema: Mapping[str, Any]) -> str:
    parts = []
    if schema.get("paged_state"):
        parts.append("paged_residency")
    if schema.get("quantized_state"):
        parts.append("dequant_requant")
    if schema.get("kahan_compensation"):
        parts.append("kahan_comp")
    if schema.get("scheduler_coupled"):
        parts.append("schedule_free_state_machine")
    return "+".join(parts) if parts else "plain_adamw"


def _fallback_detected(optimizer: OptimizerType, optimizer_name: str) -> bool:
    if optimizer in {
        OptimizerType.ADAMW_8BIT,
        OptimizerType.PAGED_ADAMW,
        OptimizerType.PAGED_ADAMW_32BIT,
        OptimizerType.PAGED_ADAMW_8BIT,
    }:
        return optimizer_name == "AdamW"
    if optimizer == OptimizerType.ADAMW_SCHEDULE_FREE:
        return optimizer_name == "AdamW"
    return False


def _fallback_probe(case: _ProbeCase, optimizer_name: str, trainer: LulynxTrainer) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "optimizer_type": case.optimizer.value,
        "status": "fallback_detected",
        "device": case.device,
        "optimizer_class": optimizer_name,
        "optimizer_backend_profile": dict(getattr(trainer, "_optimizer_backend_profile", {}) or {}),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [f"{case.optimizer.value}_resolved_to_fallback_optimizer"],
    }


def _skipped_probe(case: _ProbeCase, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "optimizer_type": case.optimizer.value,
        "status": "skipped",
        "device": case.device,
        "reason": reason,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [],
    }


def _dependency_for(optimizer: OptimizerType) -> str:
    if optimizer in {
        OptimizerType.ADAMW_8BIT,
        OptimizerType.PAGED_ADAMW,
        OptimizerType.PAGED_ADAMW_32BIT,
        OptimizerType.PAGED_ADAMW_8BIT,
    }:
        return "bitsandbytes"
    if optimizer == OptimizerType.ADAMW_SCHEDULE_FREE:
        return "schedulefree"
    return ""


def _module_available(name: str) -> bool:
    if not name:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _blocked_reasons(missing: Sequence[str], hard_failures: Sequence[Mapping[str, Any]], resume_ready: bool) -> list[str]:
    reasons = [f"unclassified_optimizer:{name}" for name in missing]
    for failure in hard_failures:
        reasons.extend(str(item) for item in failure.get("blocked_reasons", []) or [])
    if not resume_ready:
        reasons.append("required_kahan_adamw8bit_resume_probe_missing")
    return _dedupe(reasons)


def _recommended_next_step(resume_ready: bool, matrix: Sequence[Mapping[str, Any]]) -> str:
    if not resume_ready:
        return "fix KahanAdamW8bit local resume probe before native quantized AdamW work"
    high_risk = [row["optimizer_type"] for row in matrix if row.get("resume_risk") == "high"]
    if high_risk:
        return f"write reference state-machine or paged-state contracts for {', '.join(high_risk)}"
    return "start native quantized AdamW scratch kernel only after explicit product review"


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_OPTIMIZERS", "build_adamw_variant_state_scorecard"]
