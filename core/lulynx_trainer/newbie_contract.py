from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class VerificationStatus(str, Enum):
    """How strongly a Newbie-family fact has been verified."""

    VERIFIED = "VERIFIED"
    ASSUMED = "ASSUMED"
    TODO = "TODO"
    BLOCKED = "BLOCKED"


class ReadinessLevel(str, Enum):
    """Aggregate backend readiness for native Newbie support."""

    READY = "READY"
    CONDITIONAL = "CONDITIONAL"
    SCAFFOLD_ONLY = "SCAFFOLD_ONLY"
    NOT_READY = "NOT_READY"


class SecondaryEncoderKind(str, Enum):
    """Auxiliary conditioner type used alongside the Gemma text stack."""

    JINA_CLIP_POOLED = "JINA_CLIP_POOLED"
    CLIP_PROJECTION = "CLIP_PROJECTION"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SecondaryEncoderSpec:
    """Structured description of the auxiliary conditioning branch."""

    kind: SecondaryEncoderKind = SecondaryEncoderKind.JINA_CLIP_POOLED
    kind_status: VerificationStatus = VerificationStatus.VERIFIED
    expected_output: str = "pooled_text_features"
    expected_dim: int = 1024
    needs_hidden_states: bool = False
    needs_projection_head: bool = False
    config_path_hint: Optional[str] = "clip_model/config.json"

    def is_loadable_as_clip(self) -> bool:
        return self.kind == SecondaryEncoderKind.CLIP_PROJECTION

    def readiness_note(self) -> str:
        if self.kind == SecondaryEncoderKind.JINA_CLIP_POOLED:
            return (
                "Auxiliary conditioning is JinaCLIP pooled text features, "
                "not an SDXL-style secondary CLIP hidden-state branch."
            )
        if self.kind == SecondaryEncoderKind.CLIP_PROJECTION:
            return "Auxiliary conditioning follows a CLIP projection branch."
        if self.kind == SecondaryEncoderKind.NONE:
            return "No auxiliary conditioning branch is expected."
        return "Auxiliary conditioning branch is still unresolved."


@dataclass(frozen=True)
class NewbieContract:
    """Observed native-family facts for Newbie bundles."""

    expected_directory_layout: str = "native_bundle"
    directory_layout_status: VerificationStatus = VerificationStatus.VERIFIED
    expected_subdirectories: Tuple[str, ...] = (
        "transformer",
        "text_encoder",
        "clip_model",
        "vae",
    )
    expected_subdirectories_status: VerificationStatus = VerificationStatus.VERIFIED

    transformer_family: str = "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP"
    transformer_family_status: VerificationStatus = VerificationStatus.VERIFIED
    transformer_model_type: str = "nextdit_clip"
    transformer_model_type_status: VerificationStatus = VerificationStatus.VERIFIED

    vae_class: str = "AutoencoderKL"
    vae_class_status: VerificationStatus = VerificationStatus.VERIFIED
    latent_channels: int = 16
    latent_channels_status: VerificationStatus = VerificationStatus.VERIFIED
    vae_scaling_factor: float = 0.3611
    vae_scaling_factor_status: VerificationStatus = VerificationStatus.VERIFIED

    primary_text_family: str = "Gemma3Model"
    primary_text_family_status: VerificationStatus = VerificationStatus.VERIFIED
    primary_text_hidden_size: int = 2560
    primary_text_hidden_size_status: VerificationStatus = VerificationStatus.VERIFIED

    clip_family: str = "JinaCLIPModel"
    clip_family_status: VerificationStatus = VerificationStatus.VERIFIED
    clip_projection_dim: int = 1024
    clip_projection_dim_status: VerificationStatus = VerificationStatus.VERIFIED

    secondary_encoder: SecondaryEncoderSpec = field(default_factory=SecondaryEncoderSpec)

    uses_pooled_prompt_embeds: bool = False
    pooled_embeds_status: VerificationStatus = VerificationStatus.VERIFIED
    uses_time_ids: bool = False
    time_ids_status: VerificationStatus = VerificationStatus.VERIFIED
    uses_transport_training: bool = True
    transport_training_status: VerificationStatus = VerificationStatus.VERIFIED
    uses_clip_pooled_features: bool = True
    clip_pooled_features_status: VerificationStatus = VerificationStatus.VERIFIED

    # Verified by real native-bundle smoke on CPU/CUDA plus real-transformer
    # injector/backward proof. This is loader/transport closure only, not a
    # claim of full end-to-end cached/raw training or preview closure.
    native_loader_status: VerificationStatus = VerificationStatus.VERIFIED
    transport_loop_status: VerificationStatus = VerificationStatus.VERIFIED
    target_registry_status: VerificationStatus = VerificationStatus.VERIFIED

    def assess_readiness(self) -> ReadinessLevel:
        critical = (
            self.native_loader_status,
            self.transport_loop_status,
            self.target_registry_status,
        )
        if any(status == VerificationStatus.BLOCKED for status in critical):
            return ReadinessLevel.NOT_READY
        if any(status == VerificationStatus.TODO for status in critical):
            return ReadinessLevel.NOT_READY

        all_statuses = []
        for _, value in self.__dict__.items():
            if isinstance(value, VerificationStatus):
                all_statuses.append(value)
            elif isinstance(value, SecondaryEncoderSpec):
                all_statuses.append(value.kind_status)

        if all(status == VerificationStatus.VERIFIED for status in all_statuses):
            return ReadinessLevel.READY
        return ReadinessLevel.CONDITIONAL

    def blocking_issues(self) -> List[str]:
        issues: List[str] = []
        if self.native_loader_status != VerificationStatus.VERIFIED:
            issues.append("native NextDiT-family loader is not wired yet")
        if self.transport_loop_status != VerificationStatus.VERIFIED:
            issues.append("transport / flow training loop is not wired yet")
        if self.target_registry_status != VerificationStatus.VERIFIED:
            issues.append("LoRA target registry is still provisional")
        return issues

    def assumption_count(self) -> dict:
        counts = {status.value: 0 for status in VerificationStatus}
        for _, value in self.__dict__.items():
            if isinstance(value, VerificationStatus):
                counts[value.value] += 1
            elif isinstance(value, SecondaryEncoderSpec):
                counts[value.kind_status.value] += 1
        return counts

    def summary(self) -> str:
        lines = [
            "Newbie Contract Audit:",
            f"  readiness: {self.assess_readiness().value}",
            f"  layout: {self.expected_directory_layout} {self.expected_subdirectories}",
            f"  transformer: {self.transformer_family} ({self.transformer_model_type})",
            f"  text: {self.primary_text_family} hidden={self.primary_text_hidden_size}",
            f"  clip: {self.clip_family} pooled_dim={self.clip_projection_dim}",
            f"  vae: {self.vae_class} latent_channels={self.latent_channels} scale={self.vae_scaling_factor}",
            f"  auxiliary_conditioning: {self.secondary_encoder.kind.value}",
            f"  transport: {self.uses_transport_training}",
            f"  pooled_prompt_embeds: {self.uses_pooled_prompt_embeds}",
            f"  time_ids: {self.uses_time_ids}",
        ]
        blockers = self.blocking_issues()
        if blockers:
            lines.append("  blockers:")
            for item in blockers:
                lines.append(f"    - {item}")
        return "\n".join(lines)


NEWBIE_CONTRACT = NewbieContract()


def audit_newbie_contract() -> str:
    return NEWBIE_CONTRACT.summary()


def get_readiness() -> ReadinessLevel:
    return NEWBIE_CONTRACT.assess_readiness()


def get_blocking_issues() -> List[str]:
    return NEWBIE_CONTRACT.blocking_issues()
