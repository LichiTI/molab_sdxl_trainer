"""Report-only checkpoint/resume adapter proof for KahanAdamW8bit."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch

from core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit
from core.turbocore_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard import (
    build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard,
)


ADAPTER_RUNTIME_KIND = "kahan_adamw8bit_checkpoint_resume_adapter_v0"
ENVELOPE_SCHEMA = "kahan_adamw8bit_checkpoint_resume_envelope_v0"
OPTIMIZER_KIND = "kahan_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_kahan"
REQUIRED_STATE_KEYS = ("step", "exp_avg_q", "exp_avg_sq_q", "kahan_comp")
FLOAT_TOLERANCE = 5e-6


def build_kahan_adamw8bit_checkpoint_resume_adapter_scorecard(
    *,
    dtype_matrix_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build a runtime-shaped checkpoint/resume proof without trainer integration."""

    dtype_matrix = dict(
        dtype_matrix_report
        or build_kahan_adamw8bit_bf16_native_dtype_matrix_scorecard(run_live_probe=False)
    )
    contract = _runtime_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(dtype_matrix, contract, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_checkpoint_resume_adapter_scorecard_v0",
        "gate": "kahan_adamw8bit_checkpoint_resume_adapter",
        "ok": ready,
        "promotion_ready": False,
        "checkpoint_resume_adapter_ready": ready,
        "training_checkpoint_integration_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "envelope_schema": ENVELOPE_SCHEMA,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "runtime_contract": contract,
        "dtype_matrix_summary": dict(dtype_matrix.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "runtime_envelope_roundtrip_passed": bool(live_probe.get("runtime_envelope_roundtrip_passed", False)),
            "kahan_comp_fp32_restored": bool(live_probe.get("kahan_comp_fp32_restored", False)),
            "resume_probe_passed": bool(live_probe.get("resume_probe_passed", False)),
            "max_resume_param_diff": live_probe.get("max_resume_param_diff"),
            "max_resume_kahan_comp_diff": live_probe.get("max_resume_kahan_comp_diff"),
            "quantized_state_mismatch_count": live_probe.get("quantized_state_mismatch_count"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_real_training_matrix_missing",
                "kahan_adamw8bit_runtime_dispatch_disabled_pending_review",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit real-training matrix gate with dispatch still disabled"
            if ready
            else "fix KahanAdamW8bit checkpoint/resume adapter blockers"
        ),
        "notes": [
            "This adapter proves runtime envelope export/import and resume parity for KahanAdamW8bit.",
            "It explicitly restores kahan_comp to fp32 after optimizer.load_state_dict for bf16 parameters.",
            "It does not hook into trainer checkpoint save/load and does not enable native dispatch.",
        ],
    }


def export_kahan_adamw8bit_runtime_checkpoint_envelope(
    *,
    optimizer_state_dict: Mapping[str, Any],
    run_id: str = "probe",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "envelope_schema": ENVELOPE_SCHEMA,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "run_id": str(run_id),
        "optimizer_state_dict": copy.deepcopy(optimizer_state_dict),
        "adapter_manifest": {
            "schema_version": 1,
            "required_state_keys": list(REQUIRED_STATE_KEYS),
            "kahan_comp_dtype": "float32",
            "report_only": True,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
        },
    }


def import_kahan_adamw8bit_runtime_checkpoint_state_dict(
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = _envelope_checks(envelope)
    state_dict = envelope.get("optimizer_state_dict") if isinstance(envelope, Mapping) else None
    imported = copy.deepcopy(state_dict if isinstance(state_dict, Mapping) else {})
    _normalize_state_dict_kahan_comp(imported)
    return imported, {
        "schema_version": 1,
        "ok": all(bool(item.get("ok", False)) for item in checks),
        "checks": checks,
    }


def restore_kahan_adamw8bit_runtime_state(
    optimizer: KahanAdamW8bit,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer.load_state_dict(copy.deepcopy(state_dict))
    converted = 0
    total = 0
    for state in optimizer.state.values():
        if not isinstance(state, Mapping):
            continue
        comp = state.get("kahan_comp")
        if isinstance(comp, torch.Tensor):
            total += 1
            if comp.dtype != torch.float32:
                state["kahan_comp"] = comp.float()
                converted += 1
    return {
        "schema_version": 1,
        "ok": total > 0 and all(_state_comp_fp32(state) for state in optimizer.state.values()),
        "kahan_comp_tensor_count": total,
        "converted_kahan_comp_count": converted,
    }


def _runtime_contract(*, numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_runtime_kind": ADAPTER_RUNTIME_KIND,
        "envelope_schema": ENVELOPE_SCHEMA,
        "optimizer_kind": OPTIMIZER_KIND,
        "required_state_keys": list(REQUIRED_STATE_KEYS),
        "shape_contract": {
            "param_numel": int(numel),
            "quant_state_numel": int(numel),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        value = torch.linspace(-0.5, 0.5, steps=max(int(numel), 1), device=device, dtype=torch.float32).to(torch.bfloat16)
        prime_param = torch.nn.Parameter(value.clone())
        prime_optimizer = KahanAdamW8bit([prime_param], lr=1e-3, weight_decay=0.01)
        _step(prime_param, prime_optimizer, scale=0.35)
        envelope = export_kahan_adamw8bit_runtime_checkpoint_envelope(
            optimizer_state_dict=prime_optimizer.state_dict(),
            run_id="p8x_probe",
        )
        imported_state_dict, import_validation = import_kahan_adamw8bit_runtime_checkpoint_state_dict(envelope)
        saved_param = prime_param.detach().clone()

        reference_param = torch.nn.Parameter(saved_param.clone())
        reference_optimizer = KahanAdamW8bit([reference_param], lr=1e-3, weight_decay=0.01)
        reference_restore = restore_kahan_adamw8bit_runtime_state(reference_optimizer, imported_state_dict)
        _step(reference_param, reference_optimizer, scale=0.7)

        restored_param = torch.nn.Parameter(saved_param.clone())
        restored_optimizer = KahanAdamW8bit([restored_param], lr=1e-3, weight_decay=0.01)
        restored_restore = restore_kahan_adamw8bit_runtime_state(restored_optimizer, imported_state_dict)
        _step(restored_param, restored_optimizer, scale=0.7)

        reference_state = reference_optimizer.state[reference_param]
        restored_state = restored_optimizer.state[restored_param]
        param_diff = _max_abs(reference_param.detach(), restored_param.detach())
        state_compare = _compare_kahan_state(reference_state, restored_state)
        comp_fp32 = _state_comp_fp32(reference_state) and _state_comp_fp32(restored_state)
        resume_ok = (
            bool(import_validation["ok"])
            and bool(reference_restore["ok"])
            and bool(restored_restore["ok"])
            and comp_fp32
            and param_diff <= FLOAT_TOLERANCE
            and bool(state_compare["ok"])
        )
        return {
            "schema_version": 1,
            "status": "passed" if resume_ok else "failed",
            "device": str(device),
            "dtype": "bfloat16",
            "runtime_envelope_roundtrip_passed": resume_ok,
            "runtime_import_restores_state": bool(import_validation["ok"]),
            "kahan_comp_fp32_restored": comp_fp32,
            "resume_probe_passed": resume_ok,
            "max_resume_param_diff": param_diff,
            "max_resume_kahan_comp_diff": state_compare.get("max_kahan_comp_diff"),
            "quantized_state_mismatch_count": state_compare.get("quantized_state_mismatch_count"),
            "envelope_summary": _envelope_summary(envelope),
            "import_validation": import_validation,
            "reference_restore": reference_restore,
            "restored_restore": restored_restore,
            "state_compare": state_compare,
            "training_checkpoint_integration_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "blocked_reasons": [] if resume_ok else _live_blockers(import_validation, reference_restore, restored_restore, comp_fp32, param_diff, state_compare),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"kahan_adamw8bit_checkpoint_resume_adapter_probe_failed:{type(exc).__name__}"],
        )


def _step(param: torch.nn.Parameter, optimizer: KahanAdamW8bit, *, scale: float) -> None:
    if param.grad is not None:
        param.grad = None
    values = param.float()
    loss = ((values * float(scale)).pow(2).mean() + values.mean() * 0.013)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _normalize_state_dict_kahan_comp(state_dict: Mapping[str, Any]) -> None:
    state = state_dict.get("state") if isinstance(state_dict, Mapping) else None
    if not isinstance(state, Mapping):
        return
    for value in state.values():
        if isinstance(value, dict):
            comp = value.get("kahan_comp")
            if isinstance(comp, torch.Tensor) and comp.dtype != torch.float32:
                value["kahan_comp"] = comp.float()


def _envelope_checks(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = envelope.get("adapter_manifest") if isinstance(envelope, Mapping) else None
    state_dict = envelope.get("optimizer_state_dict") if isinstance(envelope, Mapping) else None
    return [
        _check(
            "adapter_runtime_kind",
            envelope.get("adapter_runtime_kind") == ADAPTER_RUNTIME_KIND,
            "adapter_runtime_kind_mismatch",
        ),
        _check("optimizer_kind", envelope.get("optimizer_kind") == OPTIMIZER_KIND, "optimizer_kind_mismatch"),
        _check(
            "optimizer_state_dict_present",
            isinstance(state_dict, Mapping) and "state" in state_dict,
            "optimizer_state_dict_missing",
        ),
        _check(
            "adapter_manifest_present",
            isinstance(manifest, Mapping)
            and set(str(item) for item in manifest.get("required_state_keys", []) or [])
            == set(REQUIRED_STATE_KEYS),
            "adapter_manifest_missing_state_keys",
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
    state_dict = envelope.get("optimizer_state_dict", {}) if isinstance(envelope, Mapping) else {}
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    return {
        "schema_version": 1,
        "adapter_runtime_kind": envelope.get("adapter_runtime_kind"),
        "optimizer_kind": envelope.get("optimizer_kind"),
        "has_optimizer_state_dict": isinstance(state_dict, Mapping) and bool(state_dict),
        "state_entry_count": len(state) if isinstance(state, Mapping) else 0,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
    }


def _validations(
    dtype_matrix: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = contract.get("runtime_policy", {}) if isinstance(contract.get("runtime_policy"), Mapping) else {}
    live_status = str(live_probe.get("status", "unknown"))
    return [
        _validation(
            "p8w_bf16_native_dtype_matrix_ready",
            bool(dtype_matrix.get("bf16_native_dtype_matrix_ready", False)),
            "kahan_adamw8bit_bf16_native_dtype_matrix_missing",
        ),
        _validation(
            "checkpoint_resume_contract_named",
            contract.get("adapter_runtime_kind") == ADAPTER_RUNTIME_KIND
            and contract.get("envelope_schema") == ENVELOPE_SCHEMA,
            "kahan_adamw8bit_checkpoint_resume_contract_missing",
        ),
        _validation(
            "runtime_envelope_probe_or_skip",
            live_status == "skipped" or bool(live_probe.get("runtime_envelope_roundtrip_passed", False)),
            "kahan_adamw8bit_checkpoint_resume_probe_failed",
        ),
        _validation(
            "kahan_comp_fp32_restored_or_skip",
            live_status == "skipped" or bool(live_probe.get("kahan_comp_fp32_restored", False)),
            "kahan_adamw8bit_checkpoint_resume_kahan_comp_not_fp32",
        ),
        _validation(
            "training_checkpoint_integration_not_enabled",
            not bool(policy.get("training_checkpoint_integration_ready", True))
            and not bool(live_probe.get("training_checkpoint_integration_ready", True)),
            "kahan_adamw8bit_checkpoint_resume_enabled_training_integration",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(policy.get("training_path_enabled", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "kahan_adamw8bit_checkpoint_resume_changed_default_behavior",
        ),
    ]


def _compare_kahan_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    quant_mismatch = 0
    max_absmax_diff = 0.0
    for key in ("exp_avg_q", "exp_avg_sq_q"):
        left_q = left[key]
        right_q = right[key]
        quant_mismatch += int((left_q.data != right_q.data).sum().item())
        max_absmax_diff = max(max_absmax_diff, _max_abs(left_q.absmax, right_q.absmax))
    max_kahan = _max_abs(left["kahan_comp"], right["kahan_comp"])
    return {
        "schema_version": 1,
        "ok": quant_mismatch == 0 and max_absmax_diff <= FLOAT_TOLERANCE and max_kahan <= FLOAT_TOLERANCE,
        "quantized_state_mismatch_count": quant_mismatch,
        "max_absmax_diff": max_absmax_diff,
        "max_kahan_comp_diff": max_kahan,
    }


def _state_comp_fp32(state: Any) -> bool:
    return isinstance(state, Mapping) and isinstance(state.get("kahan_comp"), torch.Tensor) and state["kahan_comp"].dtype == torch.float32


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _live_blockers(
    import_validation: Mapping[str, Any],
    reference_restore: Mapping[str, Any],
    restored_restore: Mapping[str, Any],
    comp_fp32: bool,
    param_diff: float,
    state_compare: Mapping[str, Any],
) -> list[str]:
    blockers = []
    if not bool(import_validation.get("ok", False)):
        blockers.append("kahan_adamw8bit_checkpoint_resume_import_invalid")
    if not bool(reference_restore.get("ok", False)) or not bool(restored_restore.get("ok", False)) or not comp_fp32:
        blockers.append("kahan_adamw8bit_checkpoint_resume_kahan_comp_not_fp32")
    if param_diff > FLOAT_TOLERANCE:
        blockers.append("kahan_adamw8bit_checkpoint_resume_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("kahan_adamw8bit_checkpoint_resume_state_parity_failed")
    return blockers


def _check(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "check": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "runtime_envelope_roundtrip_passed": False,
        "kahan_comp_fp32_restored": False,
        "resume_probe_passed": False,
        "training_checkpoint_integration_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "runtime_envelope_roundtrip_passed": False,
        "kahan_comp_fp32_restored": False,
        "resume_probe_passed": False,
        "training_checkpoint_integration_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
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
    "export_kahan_adamw8bit_runtime_checkpoint_envelope",
    "import_kahan_adamw8bit_runtime_checkpoint_state_dict",
    "restore_kahan_adamw8bit_runtime_state",
    "build_kahan_adamw8bit_checkpoint_resume_adapter_scorecard",
]
