"""Report-only residency contract for PagedAdamW8bit.

PagedAdamW8bit combines quantized optimizer state with bitsandbytes-managed
residency.  This module records the state layout and live runtime signature
before any native kernel work.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping

import torch

from core.configs import OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


TARGET_OPTIMIZER = OptimizerType.PAGED_ADAMW_8BIT
REQUIRED_LIVE_KEYS = ("state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2")


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_paged_adamw8bit_residency_scorecard(
    *,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build the P8B residency contract without enabling native dispatch."""

    static_contract = _static_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    live_status = str(live_probe.get("status", "unknown"))
    hard_failure = live_status == "failed"
    ready = bool(static_contract["ok"]) and not hard_failure
    blockers = list(static_contract.get("blocked_reasons", [])) + list(live_probe.get("blocked_reasons", []))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_residency_scorecard_v0",
        "gate": "paged_adamw8bit_residency_contract",
        "ok": ready,
        "promotion_ready": False,
        "residency_contract_ready": ready,
        "live_probe_ready": live_status in {"passed", "skipped"},
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "static_contract": static_contract,
        "live_probe": live_probe,
        "summary": {
            "required_live_key_count": len(REQUIRED_LIVE_KEYS),
            "observed_live_key_count": int(live_probe.get("observed_required_key_count", 0) or 0),
            "live_probe_status": live_status,
            "tensor_metadata_count": len(live_probe.get("tensor_metadata", []) or []),
            "checkpoint_packs_quant_state": bool(live_probe.get("checkpoint_packs_quant_state", False)),
            "resume_probe_passed": bool(live_probe.get("resume_probe_passed", False)),
        },
        "promotion_blockers": _dedupe(blockers + ["native_kernel_missing", "runtime_canary_missing"]),
        "blocked_reasons": _dedupe(blockers),
        "recommended_next_step": (
            "design PagedAdamW8bit native ABI with explicit quantized state and bnb checkpoint adapter"
            if ready
            else "fix PagedAdamW8bit residency contract blockers"
        ),
        "notes": [
            "This scorecard is report-only and never enables native optimizer dispatch.",
            "PagedAdamW8bit state_dict is not a plain tensor layout; bitsandbytes packs quantized state metadata.",
            "A future native route needs a checkpoint adapter and explicit residency/page migration policy.",
        ],
    }


def _static_contract(*, numel: int) -> dict[str, Any]:
    block_size = 256
    absmax_blocks = max(1, (max(int(numel), 1) + block_size - 1) // block_size)
    return {
        "schema_version": 1,
        "ok": True,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_family": "adamw_quantized_paged",
        "math_family": "adamw",
        "decoupled_weight_decay": True,
        "state_roles": [
            {"role": "state1", "meaning": "first moment", "dtype": "uint8", "mutable": True},
            {"role": "state2", "meaning": "second moment", "dtype": "uint8", "mutable": True},
            {"role": "qmap1", "meaning": "first moment quantization map", "dtype": "float32", "mutable": False},
            {"role": "qmap2", "meaning": "second moment quantization map", "dtype": "float32", "mutable": False},
            {"role": "absmax1", "meaning": "first moment block scale", "dtype": "float32", "mutable": True},
            {"role": "absmax2", "meaning": "second moment block scale", "dtype": "float32", "mutable": True},
        ],
        "shape_contract": {
            "param_numel": int(numel),
            "quant_state_numel": int(numel),
            "quant_map_numel": 256,
            "absmax_block_size": block_size,
            "expected_absmax_numel": absmax_blocks,
        },
        "residency_policy": {
            "owner": "bitsandbytes_optimizer",
            "paged_state": True,
            "quantized_state": True,
            "page_migration_risk": "high",
            "state_dict_tensor_layout": "packed_bnb_quant_state_metadata",
            "native_route_requires_checkpoint_adapter": True,
        },
        "native_abi_requirements": [
            "param_flat",
            "grad_flat",
            "state1_uint8",
            "state2_uint8",
            "qmap1_fp32",
            "qmap2_fp32",
            "absmax1_fp32",
            "absmax2_fp32",
            "step",
            "lr",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
        ],
        "blocked_reasons": [],
    }


def _live_probe(*, numel: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    if importlib.util.find_spec("bitsandbytes") is None:
        return _skipped_live_probe("bitsandbytes_unavailable")
    try:
        value = torch.linspace(-1.0, 1.0, steps=max(int(numel), 1), device="cuda")
        trainer = _make_trainer(value)
        optimizer = trainer._create_optimizer()
        optimizer_name = type(optimizer).__name__
        if optimizer_name == "AdamW":
            return {
                "schema_version": 1,
                "status": "failed",
                "reason": "resolved_to_fallback_adamw",
                "optimizer_class": optimizer_name,
                "blocked_reasons": ["paged_adamw8bit_resolved_to_fallback_adamw"],
            }
        param = trainer.lora_injector.param
        grad1 = torch.linspace(-0.1, 0.1, steps=param.numel(), device=param.device)
        grad2 = torch.linspace(0.05, -0.05, steps=param.numel(), device=param.device)
        _step(param, optimizer, grad1)
        live_state = _first_live_state(optimizer)
        tensor_metadata = _tensor_metadata(live_state)
        state_dict = optimizer.state_dict()
        state_dict_summary = _state_dict_summary(state_dict)
        required_ok = _required_live_state_ok(live_state)
        resume_result = _resume_roundtrip(
            saved_state=copy.deepcopy(state_dict),
            saved_param=param.detach().clone(),
            original_param=param,
            original_optimizer=optimizer,
            grad=grad2,
        )
        ok = required_ok and bool(resume_result["ok"])
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "observed_live_keys": sorted(str(key) for key in live_state.keys()),
            "observed_required_key_count": sum(1 for key in REQUIRED_LIVE_KEYS if key in live_state),
            "tensor_metadata": tensor_metadata,
            "state_dict_summary": state_dict_summary,
            "checkpoint_packs_quant_state": "__bnb_optimizer_quant_state__" in state_dict_summary["state_keys"],
            "resume_probe_passed": bool(resume_result["ok"]),
            "max_resume_diff": resume_result["max_resume_diff"],
            "required_live_state_ok": required_ok,
            "blocked_reasons": [] if ok else _live_blockers(required_ok, resume_result),
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"paged_adamw8bit_live_probe_failed:{type(exc).__name__}"],
        }


def _make_trainer(value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = UnifiedTrainingConfig(
        optimizer_type=TARGET_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args="",
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


def _resume_roundtrip(
    *,
    saved_state: Mapping[str, Any],
    saved_param: torch.Tensor,
    original_param: torch.nn.Parameter,
    original_optimizer: torch.optim.Optimizer,
    grad: torch.Tensor,
) -> dict[str, Any]:
    restored = _make_trainer(saved_param)
    restored_optimizer = restored._create_optimizer()
    restored_param = restored.lora_injector.param
    restored_optimizer.load_state_dict(saved_state)
    _step(original_param, original_optimizer, grad)
    _step(restored_param, restored_optimizer, grad)
    diff = _max_abs(original_param.detach(), restored_param.detach())
    return {"ok": diff <= 1e-5, "max_resume_diff": diff}


def _first_live_state(optimizer: torch.optim.Optimizer) -> Mapping[str, Any]:
    for value in optimizer.state.values():
        if isinstance(value, Mapping):
            return value
    return {}


def _tensor_metadata(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(state.items(), key=lambda item: str(item[0])):
        if not torch.is_tensor(value):
            continue
        rows.append(
            {
                "role": str(key),
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "bytes": int(value.numel() * value.element_size()),
                "is_cuda": bool(value.is_cuda),
            }
        )
    return rows


def _state_dict_summary(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    first = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    keys = sorted(str(key) for key in first.keys()) if isinstance(first, Mapping) else []
    types = {str(key): type(value).__name__ for key, value in first.items()} if isinstance(first, Mapping) else {}
    return {
        "state_entry_count": len(state),
        "state_keys": keys,
        "state_value_types": types,
        "plain_tensor_state_keys": [key for key, kind in types.items() if kind == "Tensor"],
    }


def _required_live_state_ok(state: Mapping[str, Any]) -> bool:
    for key in REQUIRED_LIVE_KEYS:
        value = state.get(key)
        if not torch.is_tensor(value):
            return False
    dtype_checks = {
        "state1": torch.uint8,
        "state2": torch.uint8,
        "qmap1": torch.float32,
        "qmap2": torch.float32,
        "absmax1": torch.float32,
        "absmax2": torch.float32,
    }
    return all(state[key].dtype == dtype for key, dtype in dtype_checks.items())


def _live_blockers(required_ok: bool, resume_result: Mapping[str, Any]) -> list[str]:
    blockers = []
    if not required_ok:
        blockers.append("paged_adamw8bit_live_state_signature_mismatch")
    if not bool(resume_result.get("ok", False)):
        blockers.append("paged_adamw8bit_resume_roundtrip_failed")
    return blockers


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "observed_required_key_count": 0,
        "tensor_metadata": [],
        "checkpoint_packs_quant_state": False,
        "resume_probe_passed": False,
        "blocked_reasons": [],
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_OPTIMIZER", "build_paged_adamw8bit_residency_scorecard"]
