"""V3 exact AdamW real-canary dispatch scorecard.

This gate uses the existing runtime/request training-loop boundary.  It proves
that exact AdamW can execute one explicit native canary step while the default
configuration stays off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_native_update_dispatch_request import build_native_update_dispatch_request


GATE = "v3_exact_adamw_real_canary_dispatch"


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_v3_exact_adamw_real_canary_scorecard(
    *,
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Build a machine-readable real-canary proof for exact AdamW."""

    default_off = _default_off_request()
    explicit = _explicit_training_request()
    live = _run_live_training_probe() if run_live_training else _skipped("live_training_probe_disabled")
    validations = _validations(default_off, explicit, live)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_real_canary_scorecard_v0",
        "gate": GATE,
        "ok": ready,
        "milestone_completed": ready,
        "promotion_ready": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "explicit_canary_training_path_enabled": bool(live.get("native_step_executed", False)),
        "optimizer_kind": "exact_adamw",
        "fallback_backend": "pytorch_adamw",
        "default_off_request": default_off,
        "explicit_training_request": explicit,
        "live_training_probe": live,
        "validations": validations,
        "summary": {
            "default_off": bool(default_off.get("default_off_verified", False)),
            "explicit_opt_in_required": True,
            "explicit_request_allowed": bool(explicit.get("dispatch_allowed", False)),
            "live_training_native_step": bool(live.get("native_step_executed", False)),
            "pytorch_fallback_preserved": bool(live.get("pytorch_fallback_preserved", False)),
            "native_step_index": live.get("native_step_index"),
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire V3 rollout manifest for exact AdamW native canary"
            if ready
            else "fix exact AdamW real-canary blockers"
        ),
        "notes": [
            "This gate only uses the existing TrainingLoop runtime/request boundary.",
            "Default behavior remains off; the native step requires explicit experimental flags.",
            "The first step stays PyTorch-authoritative so the gate can warm up on shadow evidence.",
        ],
    }


def _default_off_request() -> dict[str, Any]:
    request = build_native_update_dispatch_request(mode="off", dispatch_enabled=False)
    return {
        **request,
        "default_off_verified": bool(
            not request.get("requested", True)
            and not request.get("dispatch_allowed", True)
            and not request.get("training_path_enabled", True)
            and request.get("pytorch_optimizer_authoritative", False)
        ),
    }


def _explicit_training_request() -> dict[str, Any]:
    return build_native_update_dispatch_request(
        mode="native_experimental",
        dispatch_enabled=True,
        gate_report={"would_enable_native_update": True, "native_kernel_present": True},
        dispatch_contract={
            "dispatch_rehearsal_ready": True,
            "would_allow_native_dispatch": True,
            "native_kernel_present": True,
            "stream_lifetime_bound": True,
            "performance_test_ready": True,
            "dispatch_sequence": [
                {"step": "launch_native_adamw_kernel", "planned": True, "enabled": False}
            ],
            "blocked_reasons": [],
        },
        runtime_context={
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "native_update_training_mutation_guard_enabled": True,
        },
    )


def _run_live_training_probe() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped("cuda_required_for_v3_exact_adamw_real_canary")
    loop = _make_loop()
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        params = self._get_trainable_params()
        for param in params:
            if param.grad is not None:
                param.grad = None
        loss = sum((param * param).sum() for param in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}, {}], 0)
    if len(captured) < 2:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "v3_exact_adamw_live_training_probe_v0",
            "blocked_reasons": ["training_loop_did_not_emit_two_step_infos"],
            "result": result,
            "captured_step_count": len(captured),
        }
    first = _runtime_payload(captured[0])
    second = _runtime_payload(captured[1])
    profile = captured[1].get("turbocore_native_update_runtime_profile", {})
    training_executor = second.get("training_executor", {})
    executor_result = training_executor.get("result", {}) if isinstance(training_executor, dict) else {}
    update_report = executor_result.get("update_report", {}) if isinstance(executor_result, dict) else {}
    native_step = bool(second.get("native_step_executed", False))
    return {
        "schema_version": 1,
        "ok": bool(result.get("steps") == 2 and native_step),
        "probe": "v3_exact_adamw_live_training_probe_v0",
        "result": result,
        "captured_step_count": len(captured),
        "first_step_native": bool(first.get("native_step_executed", False)),
        "native_step_executed": native_step,
        "native_step_index": 2 if native_step else None,
        "native_kernel_launched": bool(second.get("native_kernel_launched", False)),
        "should_call_pytorch_optimizer_step": bool(second.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(second.get("fallback_to_pytorch_required", True)),
        "pytorch_fallback_preserved": bool(first.get("should_call_pytorch_optimizer_step", False)),
        "runtime_profile_resolved": str(profile.get("resolved", "") or ""),
        "training_executor_called": bool(training_executor.get("called", False)) if isinstance(training_executor, dict) else False,
        "pytorch_optimizer_state_synced": bool(
            executor_result.get("pytorch_optimizer_state_synced", False)
        ) if isinstance(executor_result, dict) else False,
        "owner_backend": str(update_report.get("owner_backend", "") or ""),
        "blocked_reasons": [] if native_step else ["native_step_not_executed"],
    }


def _make_loop() -> TrainingLoop:
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32, device=device))
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
        device="cuda",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow",
        turbocore_update_shadow_compare_interval=1,
        turbocore_update_shadow_checkpoint_contract=True,
        turbocore_update_shadow_copyback_probe=True,
        turbocore_update_shadow_copyback_dispatch_experimental=True,
        turbocore_update_shadow_native_binding_probe=True,
        turbocore_update_shadow_owner_native_launch_probe=True,
        turbocore_update_shadow_owner_native_launch_max_numel=1024,
        turbocore_update_shadow_owner_native_event_chain_probe=True,
        turbocore_update_shadow_save_owner_state=True,
        turbocore_native_update_mode="native_experimental",
        turbocore_native_update_required_shadow_passes=1,
        turbocore_native_update_allow_missing_kernel=True,
        turbocore_native_update_dispatch_enabled=True,
        turbocore_native_update_training_path_enabled=True,
        turbocore_native_update_require_native_cuda=True,
    )
    loop.total_steps = 2
    return loop


def _runtime_payload(step_info: dict[str, Any]) -> dict[str, Any]:
    value = step_info.get("turbocore_native_update_dispatch_runtime", {})
    return dict(value) if isinstance(value, dict) else {}


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "probe": "v3_exact_adamw_live_training_probe_v0",
        "native_step_executed": False,
        "pytorch_fallback_preserved": False,
        "skipped": True,
        "blocked_reasons": [str(reason)],
    }


def _validations(
    default_off: dict[str, Any],
    explicit: dict[str, Any],
    live: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "default_off_request_verified",
            bool(default_off.get("default_off_verified", False)),
            "v3_exact_adamw_default_off_request_failed",
        ),
        _validation(
            "explicit_request_allows_dispatch",
            bool(explicit.get("dispatch_allowed", False))
            and bool(explicit.get("training_path_enabled", False)),
            "v3_exact_adamw_explicit_request_not_allowed",
        ),
        _validation(
            "first_step_preserves_pytorch_fallback",
            bool(live.get("pytorch_fallback_preserved", False)),
            "v3_exact_adamw_first_step_fallback_missing",
        ),
        _validation(
            "second_step_executes_native_update",
            bool(live.get("native_step_executed", False))
            and bool(live.get("native_kernel_launched", False))
            and not bool(live.get("should_call_pytorch_optimizer_step", True))
            and not bool(live.get("fallback_to_pytorch_required", True)),
            "v3_exact_adamw_native_step_not_executed",
        ),
        _validation(
            "state_sync_and_backend_ready",
            bool(live.get("pytorch_optimizer_state_synced", False))
            and live.get("owner_backend") == "rust_cuda_adamw_v0",
            "v3_exact_adamw_state_sync_or_backend_missing",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v3_exact_adamw_real_canary_scorecard"]
