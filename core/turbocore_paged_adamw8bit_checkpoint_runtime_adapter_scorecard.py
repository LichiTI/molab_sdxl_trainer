"""Report-only runtime checkpoint adapter proof for PagedAdamW8bit."""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping, Sequence

import torch

from core.turbocore_paged_adamw8bit_checkpoint_adapter_scorecard import (
    CHECKPOINT_QUANT_STATE_KEY,
    build_paged_adamw8bit_checkpoint_adapter_scorecard,
    _compare_states,
    _pack_shadow_checkpoint,
    _state_signature,
)
from core.turbocore_paged_adamw8bit_live_canary_shadow_scorecard import (
    build_paged_adamw8bit_live_canary_shadow_scorecard,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
    _max_abs,
    _step,
)


ADAPTER_RUNTIME_KIND = "bnb_quant_state_checkpoint_runtime_adapter_v0"
ENVELOPE_SCHEMA = "paged_adamw8bit_checkpoint_runtime_envelope_v0"
OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"


def build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard(
    *,
    checkpoint_adapter_report: Mapping[str, Any] | None = None,
    canary_shadow_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build a runtime-shaped checkpoint adapter proof without dispatch."""

    adapter = dict(
        checkpoint_adapter_report
        or build_paged_adamw8bit_checkpoint_adapter_scorecard(run_live_probe=False, numel=numel)
    )
    shadow = dict(canary_shadow_report or build_paged_adamw8bit_live_canary_shadow_scorecard())
    contract = _runtime_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(adapter, shadow, contract, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_checkpoint_runtime_adapter_scorecard_v0",
        "gate": "paged_adamw8bit_checkpoint_runtime_adapter_proof",
        "ok": ready,
        "promotion_ready": False,
        "checkpoint_adapter_runtime_proof_ready": ready,
        "checkpoint_adapter_runtime_ready": ready,
        "training_checkpoint_integration_ready": False,
        "training_tensor_binding_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "envelope_schema": ENVELOPE_SCHEMA,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "checkpoint_quant_state_key": CHECKPOINT_QUANT_STATE_KEY,
        "runtime_contract": contract,
        "checkpoint_adapter_summary": dict(adapter.get("summary") or {}),
        "canary_shadow_summary": dict(shadow.get("manifest_summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "runtime_envelope_roundtrip_passed": bool(
                live_probe.get("runtime_envelope_roundtrip_passed", False)
            ),
            "runtime_import_restores_live_buffers": bool(
                live_probe.get("runtime_import_restores_live_buffers", False)
            ),
            "resume_probe_passed": bool(live_probe.get("resume_probe_passed", False)),
            "max_resume_diff": live_probe.get("max_resume_diff"),
            "training_checkpoint_integration_ready": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_training_tensor_binding_missing",
                "runtime_canary_e2e_no_regression_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only PagedAdamW8bit training tensor binding canary with e2e no-regression guard"
            if ready
            else "fix PagedAdamW8bit checkpoint runtime adapter proof blockers"
        ),
        "notes": [
            "This adapter builds a runtime-shaped envelope around the bnb optimizer state_dict.",
            "It proves export/import/resume parity but does not hook into trainer checkpoint save/load.",
            "Native optimizer dispatch remains blocked until training tensor binding and e2e no-regression pass.",
        ],
    }


def export_paged_adamw8bit_runtime_checkpoint_envelope(
    *,
    optimizer_state_dict: Mapping[str, Any],
    live_state: Mapping[str, Any],
    run_id: str = "probe",
) -> dict[str, Any]:
    """Create a runtime-shaped checkpoint envelope from live bnb buffers."""

    shadow_state_dict = _pack_shadow_checkpoint(optimizer_state_dict, live_state)
    return {
        "schema_version": 1,
        "envelope_schema": ENVELOPE_SCHEMA,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "checkpoint_quant_state_key": CHECKPOINT_QUANT_STATE_KEY,
        "run_id": str(run_id),
        "optimizer_state_dict": shadow_state_dict,
        "adapter_manifest": {
            "schema_version": 1,
            "required_live_roles": list(REQUIRED_LIVE_KEYS),
            "packed_quant_state_key": CHECKPOINT_QUANT_STATE_KEY,
            "lossless_roles": list(REQUIRED_LIVE_KEYS),
            "report_only": True,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
        },
    }


def import_paged_adamw8bit_runtime_checkpoint_state_dict(
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a runtime envelope and return an optimizer state_dict copy."""

    checks = _envelope_checks(envelope)
    state_dict = envelope.get("optimizer_state_dict") if isinstance(envelope, Mapping) else None
    return copy.deepcopy(state_dict if isinstance(state_dict, Mapping) else {}), {
        "schema_version": 1,
        "ok": all(bool(item.get("ok", False)) for item in checks),
        "checks": checks,
    }


def _runtime_contract(*, numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "envelope_schema": ENVELOPE_SCHEMA,
        "optimizer_kind": OPTIMIZER_KIND,
        "checkpoint_quant_state_key": CHECKPOINT_QUANT_STATE_KEY,
        "required_envelope_fields": [
            "adapter_runtime_kind",
            "optimizer_kind",
            "optimizer_state_dict",
            "adapter_manifest",
        ],
        "required_live_roles": list(REQUIRED_LIVE_KEYS),
        "shape_contract": {
            "param_numel": int(numel),
            "quant_state_numel": int(numel),
            "quant_map_numel": 256,
            "absmax_block_size": 256,
        },
        "runtime_policy": {
            "report_only": True,
            "training_checkpoint_integration_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
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
            return _failed_live_probe(
                "resolved_to_fallback_adamw",
                ["paged_adamw8bit_resolved_to_fallback_adamw"],
                optimizer_class=optimizer_name,
            )

        param = trainer.lora_injector.param
        grad1 = torch.linspace(-0.1, 0.1, steps=param.numel(), device=param.device)
        grad2 = torch.linspace(0.05, -0.05, steps=param.numel(), device=param.device)
        _step(param, optimizer, grad1)
        live_state = _first_live_state(optimizer)
        checkpoint = optimizer.state_dict()
        envelope = export_paged_adamw8bit_runtime_checkpoint_envelope(
            optimizer_state_dict=checkpoint,
            live_state=live_state,
            run_id="p8k_probe",
        )
        imported_state_dict, import_validation = import_paged_adamw8bit_runtime_checkpoint_state_dict(envelope)

        saved_param = param.detach().clone()
        restored = _make_trainer(saved_param)
        restored_optimizer = restored._create_optimizer()
        restored_optimizer.load_state_dict(imported_state_dict)
        restored_state = _first_live_state(restored_optimizer)
        unpack_compare = _compare_states(live_state, restored_state, REQUIRED_LIVE_KEYS)

        reference = _make_trainer(saved_param)
        reference_optimizer = reference._create_optimizer()
        reference_optimizer.load_state_dict(copy.deepcopy(imported_state_dict))
        _step(reference.lora_injector.param, reference_optimizer, grad2)
        _step(restored.lora_injector.param, restored_optimizer, grad2)
        resume_diff = _max_abs(reference.lora_injector.param.detach(), restored.lora_injector.param.detach())
        resume_ok = resume_diff <= 1e-5
        roundtrip_ok = bool(import_validation["ok"]) and bool(unpack_compare["ok"]) and resume_ok
        return {
            "schema_version": 1,
            "status": "passed" if roundtrip_ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "runtime_envelope_roundtrip_passed": roundtrip_ok,
            "runtime_import_restores_live_buffers": bool(unpack_compare["ok"]),
            "resume_probe_passed": resume_ok,
            "max_resume_diff": resume_diff,
            "envelope_summary": _envelope_summary(envelope),
            "import_validation": import_validation,
            "source_live_state_signature": _state_signature(live_state, REQUIRED_LIVE_KEYS),
            "restored_live_state_signature": _state_signature(restored_state, REQUIRED_LIVE_KEYS),
            "unpack_compare": unpack_compare,
            "training_checkpoint_integration_ready": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": [] if roundtrip_ok else _live_blockers(import_validation, unpack_compare, resume_ok),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"paged_adamw8bit_checkpoint_runtime_adapter_probe_failed:{type(exc).__name__}"],
        )


def _envelope_checks(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = envelope.get("adapter_manifest") if isinstance(envelope, Mapping) else None
    state_dict = envelope.get("optimizer_state_dict") if isinstance(envelope, Mapping) else None
    return [
        _check(
            "adapter_runtime_kind",
            envelope.get("adapter_runtime_kind") == ADAPTER_RUNTIME_KIND,
            "adapter_runtime_kind_mismatch",
        ),
        _check(
            "optimizer_kind",
            envelope.get("optimizer_kind") == OPTIMIZER_KIND,
            "optimizer_kind_mismatch",
        ),
        _check(
            "optimizer_state_dict_present",
            isinstance(state_dict, Mapping) and "state" in state_dict,
            "optimizer_state_dict_missing",
        ),
        _check(
            "adapter_manifest_present",
            isinstance(manifest, Mapping)
            and set(str(item) for item in manifest.get("required_live_roles", []) or [])
            == set(REQUIRED_LIVE_KEYS),
            "adapter_manifest_missing_roles",
        ),
        _check(
            "runtime_policy_report_only",
            isinstance(manifest, Mapping)
            and not bool(manifest.get("training_path_enabled", True))
            and not bool(manifest.get("native_dispatch_allowed", True)),
            "adapter_manifest_enabled_training_path",
        ),
    ]


def _envelope_summary(envelope: Mapping[str, Any]) -> dict[str, Any]:
    manifest = envelope.get("adapter_manifest", {}) if isinstance(envelope, Mapping) else {}
    state_dict = envelope.get("optimizer_state_dict", {}) if isinstance(envelope, Mapping) else {}
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    return {
        "schema_version": 1,
        "adapter_runtime_kind": envelope.get("adapter_runtime_kind"),
        "optimizer_kind": envelope.get("optimizer_kind"),
        "has_optimizer_state_dict": isinstance(state_dict, Mapping) and bool(state_dict),
        "state_entry_count": len(state) if isinstance(state, Mapping) else 0,
        "required_live_roles": list(manifest.get("required_live_roles", []) or [])
        if isinstance(manifest, Mapping)
        else [],
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
    }


def _validations(
    adapter: Mapping[str, Any],
    shadow: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = contract.get("runtime_policy", {}) if isinstance(contract.get("runtime_policy"), Mapping) else {}
    return [
        _validation(
            "p8d_checkpoint_adapter_proof_ready",
            bool(adapter.get("checkpoint_adapter_proof_ready", False)),
            "paged_adamw8bit_checkpoint_adapter_proof_missing",
        ),
        _validation(
            "p8j_canary_shadow_ready",
            bool(shadow.get("runtime_canary_shadow_ready", False)),
            "paged_adamw8bit_live_canary_shadow_missing",
        ),
        _validation(
            "runtime_envelope_contract_named",
            contract.get("adapter_runtime_kind") == ADAPTER_RUNTIME_KIND
            and contract.get("envelope_schema") == ENVELOPE_SCHEMA,
            "paged_adamw8bit_checkpoint_runtime_contract_missing",
        ),
        _validation(
            "runtime_adapter_report_only",
            not bool(policy.get("training_checkpoint_integration_ready", True))
            and not bool(policy.get("native_dispatch_allowed", True))
            and not bool(policy.get("training_path_enabled", True)),
            "paged_adamw8bit_checkpoint_runtime_adapter_enabled_training_path",
        ),
        _validation(
            "runtime_adapter_probe_or_skip",
            str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
            "paged_adamw8bit_checkpoint_runtime_adapter_probe_failed",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _check(name: str, ok: bool, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "check": name,
        "ok": bool(ok),
        "reason": None if ok else reason,
    }


def _live_blockers(
    import_validation: Mapping[str, Any],
    unpack_compare: Mapping[str, Any],
    resume_ok: bool,
) -> list[str]:
    blockers = []
    if not bool(import_validation.get("ok", False)):
        blockers.append("paged_adamw8bit_checkpoint_runtime_envelope_invalid")
    if not bool(unpack_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_checkpoint_runtime_import_mismatch")
    if not resume_ok:
        blockers.append("paged_adamw8bit_checkpoint_runtime_resume_parity_failed")
    return blockers


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "runtime_envelope_roundtrip_passed": False,
        "runtime_import_restores_live_buffers": False,
        "resume_probe_passed": False,
        "training_checkpoint_integration_ready": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "runtime_envelope_roundtrip_passed": False,
        "runtime_import_restores_live_buffers": False,
        "resume_probe_passed": False,
        "max_resume_diff": None,
        "training_checkpoint_integration_ready": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ADAPTER_RUNTIME_KIND",
    "ENVELOPE_SCHEMA",
    "export_paged_adamw8bit_runtime_checkpoint_envelope",
    "import_paged_adamw8bit_runtime_checkpoint_state_dict",
    "build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard",
]
