"""EasyControl v2 sidecar and adapter contract.

This module does not implement the two-stream DiT patch itself.  It defines the
small, testable contract Lulynx needs before wiring a trainable EasyControl v2
adapter into the runtime/request path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


SUPPORTED_TASKS: Tuple[str, ...] = ("generic", "colorize")
SUPPORTED_CONTROL_KINDS: Tuple[str, ...] = (
    "reference_latent",
    "colorization_lineart",
    "canny",
    "depth",
    "pose",
    "scribble",
    "segmentation",
)
SUPPORTED_TARGET_FAMILIES: Tuple[str, ...] = ("anima", "newbie")


@dataclass(frozen=True)
class EasyControlV2TaskSpec:
    """Runtime-neutral description of an EasyControl v2 control task."""

    task_id: str = "generic"
    control_kind: str = "reference_latent"
    target_family: str = "anima"
    cond_cache_dir: str = ""
    text_cache_dir: str = ""
    control_image_dir: str = ""
    control_suffix: str = ""
    drop_p: float = 0.1
    cond_noise_max: float = 0.0
    scale: float = 1.0
    match_target_bucket: bool = False
    mergeable: bool = False

    def normalized(self) -> "EasyControlV2TaskSpec":
        task_id = _normalize_token(self.task_id or "generic")
        control_kind = _normalize_token(self.control_kind or "reference_latent")
        target_family = _normalize_token(self.target_family or "anima")
        if task_id not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported EasyControl v2 task_id: {self.task_id}")
        if control_kind not in SUPPORTED_CONTROL_KINDS:
            raise ValueError(f"unsupported EasyControl v2 control_kind: {self.control_kind}")
        if target_family not in SUPPORTED_TARGET_FAMILIES:
            raise ValueError(f"unsupported EasyControl v2 target_family: {self.target_family}")
        return EasyControlV2TaskSpec(
            task_id=task_id,
            control_kind=control_kind,
            target_family=target_family,
            cond_cache_dir=str(self.cond_cache_dir or "").strip(),
            text_cache_dir=str(self.text_cache_dir or "").strip(),
            control_image_dir=str(self.control_image_dir or "").strip(),
            control_suffix=str(self.control_suffix or "").strip(),
            drop_p=_clamp(float(self.drop_p if self.drop_p is not None else 0.1), 0.0, 1.0),
            cond_noise_max=max(float(self.cond_noise_max if self.cond_noise_max is not None else 0.0), 0.0),
            scale=max(float(self.scale if self.scale is not None else 1.0), 0.0),
            match_target_bucket=bool(self.match_target_bucket),
            mergeable=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class EasyControlV2SidecarPlan:
    """Expected sidecar files for a single target image stem."""

    stem: str
    cond_latent_path: str
    text_cache_path: str
    control_image_path: str
    requires_text_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EasyControlV2SidecarAudit:
    plans: Tuple[EasyControlV2SidecarPlan, ...]
    missing_required: Tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_required": list(self.missing_required),
            "plans": [plan.to_dict() for plan in self.plans],
        }


def build_easycontrol_v2_task_spec(payload: Mapping[str, Any] | EasyControlV2TaskSpec | Any) -> EasyControlV2TaskSpec:
    if isinstance(payload, EasyControlV2TaskSpec):
        return payload.normalized()
    if not isinstance(payload, Mapping):
        return build_easycontrol_v2_task_spec_from_config(payload)
    return EasyControlV2TaskSpec(
        task_id=str(payload.get("task_id") or payload.get("easycontrol_task") or "generic"),
        control_kind=str(payload.get("control_kind") or payload.get("easycontrol_control_kind") or "reference_latent"),
        target_family=str(payload.get("target_family") or payload.get("model_family") or "anima"),
        cond_cache_dir=str(payload.get("cond_cache_dir") or payload.get("easycontrol_cond_cache_dir") or ""),
        text_cache_dir=str(payload.get("text_cache_dir") or payload.get("easycontrol_text_cache_dir") or ""),
        control_image_dir=str(payload.get("control_image_dir") or ""),
        control_suffix=str(payload.get("control_suffix") or ""),
        drop_p=float(payload.get("drop_p", payload.get("easycontrol_drop_p", 0.1)) or 0.0),
        cond_noise_max=float(payload.get("cond_noise_max", payload.get("easycontrol_cond_noise_max", 0.0)) or 0.0),
        scale=float(payload.get("scale", payload.get("easycontrol_scale", 1.0)) or 0.0),
        match_target_bucket=bool(payload.get("match_target_bucket", payload.get("easycontrol_image_match_size", False))),
    ).normalized()


def build_easycontrol_v2_task_spec_from_config(config: Any) -> EasyControlV2TaskSpec:
    target_family = str(getattr(config, "easycontrol_v2_target_family", "") or "").strip()
    if not target_family:
        target_family = str(getattr(config, "model_type", "anima") or "anima")
    return EasyControlV2TaskSpec(
        task_id=str(getattr(config, "easycontrol_v2_task_id", "generic") or "generic"),
        control_kind=str(getattr(config, "easycontrol_v2_control_kind", "reference_latent") or "reference_latent"),
        target_family=target_family,
        cond_cache_dir=str(getattr(config, "easycontrol_v2_cond_cache_dir", "") or ""),
        text_cache_dir=str(getattr(config, "easycontrol_v2_text_cache_dir", "") or ""),
        control_image_dir=str(
            getattr(config, "easycontrol_v2_control_image_dir", "")
            or getattr(config, "control_image_dir", "")
            or ""
        ),
        control_suffix=str(
            getattr(config, "easycontrol_v2_control_suffix", "")
            or getattr(config, "control_suffix", "")
            or ""
        ),
        drop_p=float(getattr(config, "easycontrol_v2_drop_p", 0.1) or 0.0),
        cond_noise_max=float(getattr(config, "easycontrol_v2_cond_noise_max", 0.0) or 0.0),
        scale=float(getattr(config, "easycontrol_v2_scale", getattr(config, "easy_control_scale", 1.0)) or 0.0),
        match_target_bucket=bool(getattr(config, "easycontrol_v2_match_target_bucket", False)),
    ).normalized()


def build_colorization_task_spec(
    *,
    target_family: str = "anima",
    cond_cache_dir: str = "post_image_dataset/colorize_cond",
    text_cache_dir: str = "post_image_dataset/colorize_text",
    control_image_dir: str = "",
    scale: float = 1.0,
) -> EasyControlV2TaskSpec:
    """Return the built-in colorization profile.

    Colorization uses the shared EasyControl v2 network but changes the
    condition contract: condition latents are manga/lineart-like inputs, and
    the optional text cache can be color-only rather than a full caption cache.
    """

    return EasyControlV2TaskSpec(
        task_id="colorize",
        control_kind="colorization_lineart",
        target_family=target_family,
        cond_cache_dir=cond_cache_dir,
        text_cache_dir=text_cache_dir,
        control_image_dir=control_image_dir,
        drop_p=0.0,
        cond_noise_max=0.02,
        scale=scale,
        match_target_bucket=True,
    ).normalized()


def sidecar_plan_for_target(target_path: str | Path, spec: EasyControlV2TaskSpec) -> EasyControlV2SidecarPlan:
    normalized = spec.normalized()
    target = Path(target_path)
    stem = target.stem
    suffix = normalized.control_suffix
    control_name = f"{stem}{suffix}" if suffix else target.name
    cond_name = f"{stem}.latent.safetensors"
    text_name = f"{stem}.text.safetensors"
    return EasyControlV2SidecarPlan(
        stem=stem,
        cond_latent_path=str(Path(normalized.cond_cache_dir) / cond_name) if normalized.cond_cache_dir else "",
        text_cache_path=str(Path(normalized.text_cache_dir) / text_name) if normalized.text_cache_dir else "",
        control_image_path=str(Path(normalized.control_image_dir) / control_name) if normalized.control_image_dir else "",
        requires_text_cache=normalized.task_id == "colorize",
    )


def audit_easycontrol_v2_sidecars(
    target_paths: tuple[str | Path, ...] | list[str | Path],
    spec: EasyControlV2TaskSpec,
    *,
    check_exists: bool = True,
) -> EasyControlV2SidecarAudit:
    normalized = spec.normalized()
    plans = tuple(sidecar_plan_for_target(path, normalized) for path in target_paths)
    missing: list[str] = []
    for plan in plans:
        if not plan.cond_latent_path:
            missing.append(f"{plan.stem}: cond_cache_dir is required")
        elif check_exists and not Path(plan.cond_latent_path).is_file():
            missing.append(f"{plan.stem}: missing cond latent {plan.cond_latent_path}")
        if plan.requires_text_cache:
            if not plan.text_cache_path:
                missing.append(f"{plan.stem}: text_cache_dir is required for colorize")
            elif check_exists and not Path(plan.text_cache_path).is_file():
                missing.append(f"{plan.stem}: missing text cache {plan.text_cache_path}")
        if plan.control_image_path and check_exists and not Path(plan.control_image_path).is_file():
            missing.append(f"{plan.stem}: missing control image {plan.control_image_path}")
    return EasyControlV2SidecarAudit(plans=plans, missing_required=tuple(missing))


def validate_easycontrol_v2_batch(batch: Mapping[str, Any], spec: EasyControlV2TaskSpec) -> tuple[bool, Tuple[str, ...]]:
    """Validate the batch keys needed by the v2 route without importing torch."""

    normalized = spec.normalized()
    blockers: list[str] = []
    control_latents = batch.get("control_latents", batch.get("cond_latents"))
    target_latents = batch.get("target_latents", batch.get("latents"))
    if control_latents is None:
        blockers.append("missing control_latents or cond_latents")
    if target_latents is not None and _shape_tail(target_latents) != _shape_tail(control_latents):
        blockers.append("control_latents spatial shape must match target_latents")
    if normalized.task_id == "colorize" and "color_text_embeds" not in batch and not normalized.text_cache_dir:
        blockers.append("colorize requires color_text_embeds or text_cache_dir")
    return (not blockers, tuple(blockers))


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _shape_tail(value: Any) -> Optional[tuple[int, int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if len(shape) < 2:
        return None
    return int(shape[-2]), int(shape[-1])
