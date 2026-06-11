"""V4 checkpoint/resume boundary scorecard for exact AdamW canary."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.lulynx_trainer.training_loop import TrainingLoop


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_v4_checkpoint_resume_scorecard(
    *,
    p1_audit: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint/resume state boundaries without enabling rollout."""

    p1_ready = bool(p1_audit.get("milestone_completed", False)) if isinstance(p1_audit, Mapping) else True
    live = _run_live_probe() if run_live_probe else _skipped("live_probe_disabled")
    progress_gates = {
        "p1_result_ingestion_contract_complete": p1_ready,
        "checkpoint_metadata_integrated": bool(live.get("checkpoint_metadata_integrated", False)),
        "owner_state_included": bool(live.get("owner_state_included", False)),
        "compatible_resume_accepted": bool(live.get("restore_loaded", False) and live.get("restore_compatible", False)),
        "owner_state_pending_after_resume": bool(live.get("owner_state_pending", False)),
        "mismatched_resume_rejected": bool(live.get("mismatch_loaded", False) and not live.get("mismatch_compatible", True)),
        "disabled_shadow_checkpoint_default_off": bool(live.get("disabled_checkpoint_default_off", False)),
        "training_path_stays_default_off": bool(live.get("training_path_stays_default_off", False)),
    }
    ready = all(progress_gates.values())
    blockers = [f"v4_p2_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v4_checkpoint_resume_scorecard_v0",
        "gate": "v4_checkpoint_resume_boundary",
        "ok": ready,
        "milestone_completed": ready,
        "checkpoint_resume_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "live_probe": live,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire V4 explicit canary rollout policy while keeping defaults off"
            if ready
            else "fix V4 checkpoint/resume boundary blockers"
        ),
        "notes": [
            "This scorecard uses the existing TrainingLoop checkpoint methods.",
            "The probe validates state compatibility and fallback behavior, not long training quality.",
            "Default and auto rollout stay disabled regardless of checkpoint readiness.",
        ],
    }


def _run_live_probe() -> dict[str, Any]:
    loop = _make_loop(shadow_mode="shadow", save_owner_state=True)
    _prime_shadow_owner(loop)
    checkpoint = loop.get_turbocore_update_checkpoint_state()
    restore = loop.load_turbocore_update_checkpoint_state(checkpoint)
    mismatch = loop.load_turbocore_update_checkpoint_state(_mismatched_checkpoint(checkpoint))
    disabled = _make_loop(shadow_mode="off", save_owner_state=False).get_turbocore_update_checkpoint_state()
    contract = checkpoint.get("checkpoint_contract") if isinstance(checkpoint.get("checkpoint_contract"), dict) else {}
    return {
        "schema_version": 1,
        "probe": "v4_checkpoint_resume_live_probe_v0",
        "ok": True,
        "checkpoint_metadata_integrated": bool(checkpoint.get("checkpoint_metadata_integrated", False)),
        "trainer_state_metadata_integrated": bool(checkpoint.get("trainer_state_metadata_integrated", False)),
        "owner_state_included": bool(checkpoint.get("owner_state_included", False)),
        "parameter_tensors": int(checkpoint.get("parameter_tensors", 0) or 0),
        "parameter_numel": int(checkpoint.get("parameter_numel", 0) or 0),
        "checkpoint_contract_roundtrip_ok": bool(contract.get("roundtrip_ok", False)),
        "checkpoint_contract_roundtrip_checked": bool(contract.get("roundtrip_checked", False)),
        "restore_loaded": bool(restore.get("loaded", False)),
        "restore_compatible": bool(restore.get("compatible", False)),
        "owner_state_pending": bool(restore.get("owner_state_pending", False)),
        "mismatch_loaded": bool(mismatch.get("loaded", False)),
        "mismatch_compatible": bool(mismatch.get("compatible", False)),
        "mismatch_owner_state_pending": bool(mismatch.get("owner_state_pending", False)),
        "disabled_checkpoint_default_off": bool(
            disabled.get("enabled") is False and disabled.get("training_path_enabled") is False
        ),
        "training_path_stays_default_off": bool(
            checkpoint.get("training_path_enabled") is False
            and restore.get("training_path_enabled") is False
            and mismatch.get("training_path_enabled") is False
        ),
        "blocked_reasons": [],
    }


def _make_loop(*, shadow_mode: str, save_owner_state: bool) -> TrainingLoop:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
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
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1000.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode=shadow_mode,
        turbocore_update_shadow_checkpoint_contract=True,
        turbocore_update_shadow_save_owner_state=save_owner_state,
    )
    loop.total_steps = 1
    return loop


def _prime_shadow_owner(loop: TrainingLoop) -> None:
    params = loop._get_trainable_params()
    for param in params:
        param.grad = torch.full_like(param, 0.125)
    loop._turbocore_update_shadow.prepare_before_optimizer(
        params,
        optimizer=loop.optimizer,
        max_grad_norm=loop.max_grad_norm,
        step=0,
    )


def _mismatched_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(checkpoint))
    owner_state = payload.get("owner_state_dict") if isinstance(payload.get("owner_state_dict"), dict) else {}
    layout = owner_state.get("layout") if isinstance(owner_state.get("layout"), dict) else {}
    layout["total_numel"] = int(layout.get("total_numel", 0) or 0) + 1
    owner_state["layout"] = layout
    payload["owner_state_dict"] = owner_state
    return payload


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "v4_checkpoint_resume_live_probe_v0",
        "ok": False,
        "skipped": True,
        "blocked_reasons": [str(reason)],
    }


__all__ = ["build_v4_checkpoint_resume_scorecard"]
