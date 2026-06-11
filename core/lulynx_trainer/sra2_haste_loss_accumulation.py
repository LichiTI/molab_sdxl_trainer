"""Default-off loss accumulation bridge for SRA2 + HASTE.

This module gives the trainer a small future call surface without wiring the
feature into the live training path. It validates the preflight contract,
computes graph-safe auxiliary loss in probe mode, and keeps all runtime/default
activation flags disabled.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .sra2_haste_alignment_facade import SRA2HasteAlignmentPolicy, sra2_haste_alignment_loss


def build_sra2_haste_loss_accumulation_contract(
    preflight: Mapping[str, Any],
    *,
    alignment_policy: Mapping[str, Any] | SRA2HasteAlignmentPolicy | None = None,
) -> dict[str, Any]:
    report = dict(preflight)
    layers = [str(item) for item in report.get("hidden_capture_layers", ()) if str(item).strip()]
    contract = dict(report.get("contract") or {})
    quality_gate = dict(contract.get("quality_gate") or {})
    blockers: list[str] = []

    if report.get("scorecard") != "sra2_haste_trainer_preflight_v0":
        blockers.append("unexpected_trainer_preflight")
    if not bool(report.get("preflight_ready", report.get("ok", False))):
        blockers.append("preflight_not_ready")
    if _unsafe_flags(report):
        blockers.append("unsafe_preflight_flag")
    if not layers:
        blockers.append("hidden_capture_layers_missing")
    if not str(report.get("vae_feature_source") or "").strip():
        blockers.append("vae_feature_source_missing")
    if not quality_gate:
        blockers.append("quality_gate_missing")

    policy = _policy_payload(alignment_policy or contract.get("alignment_policy"))
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "sra2_haste_loss_accumulation_contract_v0",
        "ok": ready,
        "loss_accumulation_contract_ready": ready,
        "probe_loss_accumulation_allowed": ready,
        "trainer_loss_accumulation_allowed": False,
        "hidden_capture_layers": layers,
        "vae_feature_source": str(report.get("vae_feature_source") or ""),
        "quality_gate": quality_gate,
        "alignment_policy": policy,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run representative Anima/Newbie loss-quality A/B before trainer wiring"
            if ready
            else "complete SRA2/HASTE preflight and quality-gate metadata"
        ),
    }


def sra2_haste_accumulate_training_loss_probe(
    base_loss: torch.Tensor,
    *,
    hidden_captures: Mapping[str, torch.Tensor],
    vae_features: torch.Tensor,
    accumulation_contract: Mapping[str, Any],
    step: int = 0,
    total_steps: int = 0,
    loss_history: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    contract = dict(accumulation_contract)
    blockers = _contract_blockers(contract)
    layers = [str(item) for item in contract.get("hidden_capture_layers", ()) if str(item).strip()]
    missing_layers = [layer for layer in layers if layer not in hidden_captures]
    blockers.extend(f"hidden_capture_missing:{layer}" for layer in missing_layers)

    if blockers:
        zero = base_loss * 0.0
        return base_loss + zero, _accumulation_report(
            contract=contract,
            active_layer_count=0,
            alignment_loss=float(zero.detach().float().item()),
            total_loss=float((base_loss + zero).detach().float().item()),
            blockers=blockers,
            child_profiles=[],
        )

    losses: list[torch.Tensor] = []
    profiles: list[dict[str, Any]] = []
    policy = SRA2HasteAlignmentPolicy(**dict(contract.get("alignment_policy") or {})).normalized()
    for layer in layers:
        loss, profile = sra2_haste_alignment_loss(
            hidden_captures[layer],
            vae_features,
            policy,
            step=step,
            total_steps=total_steps,
            loss_history=loss_history,
        )
        losses.append(loss)
        profiles.append({"layer": layer, **dict(profile)})

    alignment_loss = torch.stack([loss.float() for loss in losses]).mean() if losses else base_loss * 0.0
    total_loss = base_loss + alignment_loss.to(device=base_loss.device, dtype=base_loss.dtype)
    child_blockers = [f"{item['layer']}:alignment_inactive" for item in profiles if not bool(item.get("active"))]
    return total_loss, _accumulation_report(
        contract=contract,
        active_layer_count=sum(1 for item in profiles if bool(item.get("active"))),
        alignment_loss=float(alignment_loss.detach().float().item()),
        total_loss=float(total_loss.detach().float().item()),
        blockers=child_blockers,
        child_profiles=profiles,
    )


def _contract_blockers(contract: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if contract.get("scorecard") != "sra2_haste_loss_accumulation_contract_v0":
        blockers.append("unexpected_loss_accumulation_contract")
    if not bool(contract.get("loss_accumulation_contract_ready", contract.get("ok", False))):
        blockers.append("loss_accumulation_contract_not_ready")
    if not bool(contract.get("probe_loss_accumulation_allowed", False)):
        blockers.append("probe_loss_accumulation_not_allowed")
    if _unsafe_flags(contract):
        blockers.append("unsafe_loss_accumulation_contract_flag")
    return blockers


def _accumulation_report(
    *,
    contract: Mapping[str, Any],
    active_layer_count: int,
    alignment_loss: float,
    total_loss: float,
    blockers: Sequence[str],
    child_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "sra2_haste_loss_accumulation_probe_v0",
        "ok": ready,
        "loss_accumulation_probe_ready": ready,
        "active_layer_count": int(active_layer_count),
        "alignment_loss": float(alignment_loss),
        "total_loss": float(total_loss),
        "child_profiles": [dict(item) for item in child_profiles],
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": list(blockers),
        "recommended_next_step": (
            "collect representative SRA2/HASTE loss-quality A/B evidence"
            if ready
            else "fix loss accumulation probe blockers before trainer wiring"
        ),
        "source_contract": str(contract.get("scorecard") or ""),
    }


def _policy_payload(policy: Mapping[str, Any] | SRA2HasteAlignmentPolicy | None) -> dict[str, Any]:
    cfg = policy if isinstance(policy, SRA2HasteAlignmentPolicy) else SRA2HasteAlignmentPolicy(**dict(policy or {}))
    normalized = cfg.normalized()
    return {
        "enabled": bool(normalized.enabled),
        "loss_type": normalized.loss_type,
        "normalize_targets": bool(normalized.normalize_targets),
        "stop_grad_target": bool(normalized.stop_grad_target),
        "base_weight": float(normalized.base_weight),
        "start_step": int(normalized.start_step),
        "stop_step": int(normalized.stop_step),
        "decay_start_step": int(normalized.decay_start_step),
        "decay_end_step": int(normalized.decay_end_step),
        "min_weight": float(normalized.min_weight),
        "plateau_patience": int(normalized.plateau_patience),
        "min_relative_improvement": float(normalized.min_relative_improvement),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "request_fields_emitted",
        "request_adapter_registered",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_sra2_haste_loss_accumulation_contract",
    "sra2_haste_accumulate_training_loss_probe",
]
