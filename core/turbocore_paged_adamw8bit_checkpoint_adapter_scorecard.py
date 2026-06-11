"""Report-only checkpoint adapter proof for PagedAdamW8bit.

This module proves the bnb quant-state pack/unpack boundary that a future
native PagedAdamW8bit route must preserve.  It does not register a runtime
adapter, enable native dispatch, or implement a CUDA kernel.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping, Sequence

import torch

from core.turbocore_paged_adamw8bit_native_abi_scorecard import (
    LAUNCH_PLAN,
    OPTIMIZER_FAMILY,
    OPTIMIZER_KIND,
    TARGET_OPTIMIZER,
    build_paged_adamw8bit_native_abi_scorecard,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
    _max_abs,
    _step,
)


CHECKPOINT_QUANT_STATE_KEY = "__bnb_optimizer_quant_state__"
ADAPTER_KIND = "bnb_quant_state_checkpoint_adapter_v0"
_CHECKPOINT_ENTRY_KEYS = ("step", CHECKPOINT_QUANT_STATE_KEY)


def build_paged_adamw8bit_checkpoint_adapter_scorecard(
    *,
    native_abi_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build a report-only proof for bnb quant-state checkpoint transitions."""

    abi = dict(native_abi_report or build_paged_adamw8bit_native_abi_scorecard())
    contract = _adapter_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(abi, contract, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    ready = not failed
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_checkpoint_adapter_scorecard_v0",
        "gate": "paged_adamw8bit_checkpoint_adapter_proof",
        "ok": ready,
        "promotion_ready": False,
        "checkpoint_adapter_proof_ready": ready,
        "adapter_contract_probe_ready": ready,
        "adapter_implemented": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "launch_plan": LAUNCH_PLAN,
        "adapter_kind": ADAPTER_KIND,
        "checkpoint_quant_state_key": CHECKPOINT_QUANT_STATE_KEY,
        "checkpoint_layout_contract": contract,
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "required_live_key_count": len(REQUIRED_LIVE_KEYS),
            "checkpoint_entry_key_count": len(_CHECKPOINT_ENTRY_KEYS),
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "pack_shadow_roundtrip_passed": bool(live_probe.get("pack_shadow_roundtrip_passed", False)),
            "unpack_restores_live_buffers": bool(live_probe.get("unpack_restores_live_buffers", False)),
            "resume_probe_passed": bool(live_probe.get("resume_probe_passed", False)),
            "max_resume_diff": live_probe.get("max_resume_diff"),
            "adapter_implemented": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "paged_adamw8bit_native_kernel_missing",
                "runtime_canary_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prototype PagedAdamW8bit quantized scratch update behind report-only canary"
            if ready
            else "fix PagedAdamW8bit checkpoint adapter proof blockers"
        ),
        "notes": [
            "This proof clones live bnb buffers into __bnb_optimizer_quant_state__ and reloads them through bitsandbytes.",
            "Passing this gate means the checkpoint boundary is understood, not that native runtime serialization exists.",
            "Native dispatch remains blocked until a runtime adapter, kernel parity, and canary route are implemented.",
        ],
    }


def _adapter_contract(*, numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "checkpoint_entry_required_keys": list(_CHECKPOINT_ENTRY_KEYS),
        "packed_quant_state_required_keys": list(REQUIRED_LIVE_KEYS),
        "live_state_required_keys": list(REQUIRED_LIVE_KEYS),
        "shape_contract": {
            "param_numel": int(numel),
            "quant_state_numel": int(numel),
            "quant_map_numel": 256,
            "absmax_block_size": 256,
        },
        "transitions": [
            {
                "name": "pack_live_buffers_to_bnb_quant_state",
                "source": "optimizer.state[param].state1/state2/qmap1/qmap2/absmax1/absmax2",
                "destination": CHECKPOINT_QUANT_STATE_KEY,
                "lossless_roles": list(REQUIRED_LIVE_KEYS),
            },
            {
                "name": "unpack_bnb_quant_state_to_live_buffers",
                "source": CHECKPOINT_QUANT_STATE_KEY,
                "destination": "optimizer.state[param]",
                "lossless_roles": list(REQUIRED_LIVE_KEYS),
            },
            {
                "name": "resume_step_parity",
                "source": "packed checkpoint state",
                "destination": "restored optimizer next step",
                "tolerance": 1e-5,
            },
        ],
        "runtime_policy": {
            "report_only": True,
            "adapter_implemented": False,
            "native_dispatch_allowed": False,
        },
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
        checkpoint = optimizer.state_dict()
        checkpoint_entry = _first_state_entry(checkpoint)
        quant_state = _quant_state(checkpoint_entry)
        pack_compare = _compare_states(live_state, quant_state, REQUIRED_LIVE_KEYS)

        saved_param = param.detach().clone()
        shadow_checkpoint = _pack_shadow_checkpoint(checkpoint, live_state)
        restored = _make_trainer(saved_param)
        restored_optimizer = restored._create_optimizer()
        restored_optimizer.load_state_dict(shadow_checkpoint)
        restored_state = _first_live_state(restored_optimizer)
        unpack_compare = _compare_states(live_state, restored_state, REQUIRED_LIVE_KEYS)

        _step(param, optimizer, grad2)
        _step(restored.lora_injector.param, restored_optimizer, grad2)
        resume_diff = _max_abs(param.detach(), restored.lora_injector.param.detach())
        resume_ok = resume_diff <= 1e-5
        required_ok = _required_keys_ok(live_state) and _required_keys_ok(quant_state)
        ok = required_ok and bool(pack_compare["ok"]) and bool(unpack_compare["ok"]) and resume_ok
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "checkpoint_entry_keys": sorted(str(key) for key in checkpoint_entry.keys()),
            "packed_quant_state_keys": sorted(str(key) for key in quant_state.keys()),
            "live_state_signature": _state_signature(live_state, REQUIRED_LIVE_KEYS),
            "packed_quant_state_signature": _state_signature(quant_state, REQUIRED_LIVE_KEYS),
            "restored_live_state_signature": _state_signature(restored_state, REQUIRED_LIVE_KEYS),
            "pack_shadow_roundtrip_passed": bool(pack_compare["ok"]),
            "unpack_restores_live_buffers": bool(unpack_compare["ok"]),
            "resume_probe_passed": resume_ok,
            "max_resume_diff": resume_diff,
            "pack_compare": pack_compare,
            "unpack_compare": unpack_compare,
            "blocked_reasons": [] if ok else _live_blockers(required_ok, pack_compare, unpack_compare, resume_ok),
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"paged_adamw8bit_checkpoint_adapter_probe_failed:{type(exc).__name__}"],
        }


def _pack_shadow_checkpoint(checkpoint: Mapping[str, Any], live_state: Mapping[str, Any]) -> dict[str, Any]:
    shadow = copy.deepcopy(checkpoint)
    entry = _first_state_entry(shadow)
    entry[CHECKPOINT_QUANT_STATE_KEY] = {
        key: live_state[key].detach().clone()
        for key in REQUIRED_LIVE_KEYS
        if torch.is_tensor(live_state.get(key))
    }
    return shadow


def _first_state_entry(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    if not isinstance(state, Mapping) or not state:
        return {}
    first = next(iter(state.values()))
    return first if isinstance(first, dict) else {}


def _quant_state(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get(CHECKPOINT_QUANT_STATE_KEY) if isinstance(entry, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _required_keys_ok(state: Mapping[str, Any]) -> bool:
    return all(torch.is_tensor(state.get(key)) for key in REQUIRED_LIVE_KEYS)


def _state_signature(state: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for key in keys:
        value = state.get(key)
        if not torch.is_tensor(value):
            rows.append({"role": key, "present": False})
            continue
        rows.append(
            {
                "role": key,
                "present": True,
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "bytes": int(value.numel() * value.element_size()),
            }
        )
    return rows


def _compare_states(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any]:
    mismatches: list[str] = []
    max_diffs: dict[str, float | int] = {}
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if not torch.is_tensor(left_value) or not torch.is_tensor(right_value):
            mismatches.append(f"{key}:missing")
            continue
        if left_value.shape != right_value.shape:
            mismatches.append(f"{key}:shape")
            continue
        if left_value.dtype != right_value.dtype:
            mismatches.append(f"{key}:dtype")
            continue
        if left_value.dtype == torch.uint8:
            diff = int((left_value.detach() != right_value.detach()).sum().cpu())
            max_diffs[key] = diff
            if diff != 0:
                mismatches.append(f"{key}:uint8_value")
        else:
            diff = _max_abs(left_value.detach(), right_value.detach())
            max_diffs[key] = diff
            if diff > 1e-8:
                mismatches.append(f"{key}:float_value")
    return {
        "schema_version": 1,
        "ok": not mismatches,
        "mismatches": mismatches,
        "max_diffs": max_diffs,
    }


def _validations(
    abi: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8c_native_abi_sketch_ready",
            bool(abi.get("abi_sketch_ready", False)),
            "paged_adamw8bit_native_abi_sketch_missing",
        ),
        _validation(
            "checkpoint_quant_state_layout_named",
            CHECKPOINT_QUANT_STATE_KEY in contract.get("checkpoint_entry_required_keys", [])
            and _same_roles(contract.get("packed_quant_state_required_keys", []), REQUIRED_LIVE_KEYS),
            "paged_adamw8bit_checkpoint_layout_contract_missing",
        ),
        _validation(
            "adapter_runtime_disabled",
            not bool(contract.get("runtime_policy", {}).get("adapter_implemented", True))
            and not bool(contract.get("runtime_policy", {}).get("native_dispatch_allowed", True)),
            "paged_adamw8bit_checkpoint_adapter_runtime_enabled_too_early",
        ),
        _validation(
            "pack_unpack_probe_or_skip",
            str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
            "paged_adamw8bit_checkpoint_adapter_live_probe_failed",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _live_blockers(
    required_ok: bool,
    pack_compare: Mapping[str, Any],
    unpack_compare: Mapping[str, Any],
    resume_ok: bool,
) -> list[str]:
    blockers = []
    if not required_ok:
        blockers.append("paged_adamw8bit_checkpoint_quant_state_signature_mismatch")
    if not bool(pack_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_live_to_checkpoint_pack_mismatch")
    if not bool(unpack_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_checkpoint_to_live_unpack_mismatch")
    if not resume_ok:
        blockers.append("paged_adamw8bit_checkpoint_adapter_resume_parity_failed")
    return blockers


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "pack_shadow_roundtrip_passed": False,
        "unpack_restores_live_buffers": False,
        "resume_probe_passed": False,
        "max_resume_diff": None,
        "blocked_reasons": [],
    }


def _same_roles(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return {str(item) for item in left} == {str(item) for item in right}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ADAPTER_KIND",
    "CHECKPOINT_QUANT_STATE_KEY",
    "build_paged_adamw8bit_checkpoint_adapter_scorecard",
]
