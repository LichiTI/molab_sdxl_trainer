from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .newbie_contract import (
    NEWBIE_CONTRACT,
    ReadinessLevel,
    SecondaryEncoderKind,
    VerificationStatus,
)


@dataclass(frozen=True)
class NewbieReadinessReport:
    """Snapshot of native Newbie readiness after loader resolution."""

    scaffold_mode: bool
    scaffold_reason: Optional[str] = None

    secondary_encoder_kind: SecondaryEncoderKind = SecondaryEncoderKind.UNKNOWN
    secondary_encoder_loadable: bool = False
    secondary_encoder_training_safe: bool = False

    contract_readiness: ReadinessLevel = ReadinessLevel.NOT_READY
    blocking_issues: tuple = ()
    assumption_counts: dict | None = None

    expects_pooled_embeds: bool = False
    expects_time_ids: bool = False
    expects_clip_pooled_features: bool = True
    expects_transport_object: bool = True

    forward_smoke_passed: bool = False
    gradient_smoke_passed: bool = False
    training_loop_warnings: tuple = ()

    @classmethod
    def from_loaded_model(cls, model: object) -> "NewbieReadinessReport":
        contract = NEWBIE_CONTRACT

        scaffold_mode = bool(getattr(model, "newbie_scaffold_mode", False))
        native_conditioning_ready = bool(
            getattr(model, "newbie_native_conditioning_ready", False)
        )
        transport_ready = bool(getattr(model, "newbie_transport_ready", False))
        forward_smoke_passed = bool(
            getattr(model, "newbie_forward_smoke_passed", False)
        )
        gradient_smoke_passed = bool(
            getattr(model, "newbie_gradient_smoke_passed", False)
        )
        smoke_result = getattr(model, "newbie_smoke_result", None)
        smoke_reason = str(getattr(smoke_result, "reason", "") or "")

        warnings = []
        if scaffold_mode:
            warnings.append(
                "Loader is still operating in scaffold mode. Native Newbie "
                "conditioning and NextDiT transport semantics are not active."
            )
        if not native_conditioning_ready:
            warnings.append(
                "Gemma3 + JinaCLIP native conditioning is not wired into the "
                "training loop yet."
            )
        if not transport_ready:
            warnings.append(
                "Transport / flow training path is not wired for native "
                "Newbie execution yet."
            )
        if not forward_smoke_passed:
            warning = "Native Newbie forward smoke has not passed for this loaded model."
            if smoke_reason:
                warning = f"{warning} Reason: {smoke_reason}"
            warnings.append(warning)
        if not gradient_smoke_passed:
            warning = "Native Newbie gradient smoke has not passed for this loaded model."
            if smoke_reason and forward_smoke_passed:
                warning = f"{warning} Reason: {smoke_reason}"
            warnings.append(warning)

        dynamic_verified = (
            not scaffold_mode
            and native_conditioning_ready
            and transport_ready
            and forward_smoke_passed
            and gradient_smoke_passed
        )
        blocking_issues = tuple(
            cls._remaining_blocking_issues(
                contract=contract,
                native_conditioning_ready=native_conditioning_ready,
                transport_ready=transport_ready,
                forward_smoke_passed=forward_smoke_passed,
                gradient_smoke_passed=gradient_smoke_passed,
            )
        )
        contract_readiness = cls._resolve_contract_readiness(
            contract=contract,
            scaffold_mode=scaffold_mode,
            native_conditioning_ready=native_conditioning_ready,
            transport_ready=transport_ready,
            forward_smoke_passed=forward_smoke_passed,
            gradient_smoke_passed=gradient_smoke_passed,
            blocking_issues=blocking_issues,
        )

        return cls(
            scaffold_mode=scaffold_mode,
            scaffold_reason=(
                "Fallback route used instead of a native NextDiT-family path"
                if scaffold_mode
                else None
            ),
            secondary_encoder_kind=contract.secondary_encoder.kind,
            secondary_encoder_loadable=native_conditioning_ready,
            secondary_encoder_training_safe=(
                native_conditioning_ready and transport_ready
            ),
            contract_readiness=contract_readiness,
            blocking_issues=blocking_issues,
            assumption_counts=contract.assumption_count(),
            expects_pooled_embeds=contract.uses_pooled_prompt_embeds,
            expects_time_ids=contract.uses_time_ids,
            expects_clip_pooled_features=contract.uses_clip_pooled_features,
            expects_transport_object=contract.uses_transport_training,
            forward_smoke_passed=forward_smoke_passed,
            gradient_smoke_passed=gradient_smoke_passed,
            training_loop_warnings=tuple(warnings),
        )

    @staticmethod
    def _remaining_blocking_issues(
        *,
        contract: object,
        native_conditioning_ready: bool,
        transport_ready: bool,
        forward_smoke_passed: bool,
        gradient_smoke_passed: bool,
    ) -> tuple[str, ...]:
        issues = []
        for issue in contract.blocking_issues():
            lowered = issue.lower()
            if "native nextdit-family loader" in lowered and native_conditioning_ready:
                continue
            if "transport / flow training loop" in lowered and transport_ready:
                continue
            issues.append(issue)

        if not forward_smoke_passed:
            issues.append("native Newbie forward smoke has not passed for this loaded model")
        if not gradient_smoke_passed:
            issues.append("native Newbie gradient smoke has not passed for this loaded model")
        return tuple(issues)

    @staticmethod
    def _resolve_contract_readiness(
        *,
        contract: object,
        scaffold_mode: bool,
        native_conditioning_ready: bool,
        transport_ready: bool,
        forward_smoke_passed: bool,
        gradient_smoke_passed: bool,
        blocking_issues: tuple[str, ...],
    ) -> ReadinessLevel:
        if scaffold_mode:
            return ReadinessLevel.SCAFFOLD_ONLY
        if blocking_issues:
            return contract.assess_readiness()
        if (
            native_conditioning_ready
            and transport_ready
            and forward_smoke_passed
            and gradient_smoke_passed
        ):
            if any(
                count > 0
                for status, count in contract.assumption_count().items()
                if status in {
                    VerificationStatus.ASSUMED.value,
                    VerificationStatus.TODO.value,
                    VerificationStatus.BLOCKED.value,
                }
            ):
                return ReadinessLevel.CONDITIONAL
            return ReadinessLevel.READY
        return contract.assess_readiness()

    @property
    def can_train(self) -> bool:
        return (
            not self.scaffold_mode
            and not self.blocking_issues
            and self.secondary_encoder_training_safe
            and self.forward_smoke_passed
            and self.gradient_smoke_passed
        )

    @property
    def should_warn(self) -> bool:
        return bool(self.training_loop_warnings)

    def log_warnings(self, logger: object) -> None:
        for warning in self.training_loop_warnings:
            logger.warning("Newbie readiness: %s", warning)

    def summary(self) -> str:
        lines = [
            "Newbie Readiness Report:",
            f"  scaffold_mode: {self.scaffold_mode}",
            f"  auxiliary_conditioning: {self.secondary_encoder_kind.value}",
            f"  conditioning_ready: {self.secondary_encoder_loadable}",
            f"  training_safe: {self.secondary_encoder_training_safe}",
            f"  forward_smoke_passed: {self.forward_smoke_passed}",
            f"  gradient_smoke_passed: {self.gradient_smoke_passed}",
            f"  contract_readiness: {self.contract_readiness.value}",
            f"  can_train: {self.can_train}",
        ]
        if self.blocking_issues:
            lines.append(f"  blocking_issues: {list(self.blocking_issues)}")
        if self.training_loop_warnings:
            lines.append("  warnings:")
            for warning in self.training_loop_warnings:
                lines.append(f"    - {warning}")
        return "\n".join(lines)
