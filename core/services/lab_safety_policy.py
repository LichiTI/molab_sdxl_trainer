"""Safety policy checks for Lulynx LAB experimental runners."""

from __future__ import annotations

from backend.core.contracts import TurboLoraRequest


class LabSafetyPolicyError(ValueError):
    """Raised when a LAB request violates an experimental safety gate."""


def validate_turbo_lora_real_run_safety(request: TurboLoraRequest) -> None:
    """Validate the conservative real-run gate for Turbo/LCM LoRA smoke runs."""

    if request.dry_run:
        return
    if not request.confirm_real_run:
        raise LabSafetyPolicyError("Real Turbo LoRA smoke requires confirm_real_run=true")
    if request.max_train_steps > 4:
        raise LabSafetyPolicyError("Real Turbo LoRA smoke is capped at 4 steps for now")
    if request.batch_size > 1:
        raise LabSafetyPolicyError("Real Turbo LoRA smoke is capped at batch size 1 for now")


__all__ = ["LabSafetyPolicyError", "validate_turbo_lora_real_run_safety"]
