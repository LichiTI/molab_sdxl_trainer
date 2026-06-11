"""Default-off native PiD decode backend contracts and smoke path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


QWEN_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
QWEN_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)

PID_CODE_LICENSE = "Apache-2.0"
PID_WEIGHT_LICENSE = "NVIDIA NSCLv1 non-commercial; user-provided only"
PID_HF_REPO = "nvidia/PiD"
SMOKE_CHECKPOINT_KIND = "lulynx_pid_tiny_smoke_v0"


@dataclass(frozen=True)
class OfficialPidCheckpoint:
    backbone: str
    ckpt_type: str
    relative_path: str
    scale: int
    latent_channels: int
    normalized_latent: bool


OFFICIAL_PID_CHECKPOINTS: dict[tuple[str, str], OfficialPidCheckpoint] = {
    ("flux", "2k"): OfficialPidCheckpoint("flux", "2k", "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("flux", "2kto4k"): OfficialPidCheckpoint("flux", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("flux2", "2k"): OfficialPidCheckpoint("flux2", "2k", "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth", 4, 128, True),
    ("flux2", "2kto4k"): OfficialPidCheckpoint("flux2", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/model_ema_bf16.pth", 4, 128, True),
    ("sd3", "2k"): OfficialPidCheckpoint("sd3", "2k", "checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("sd3", "2kto4k"): OfficialPidCheckpoint("sd3", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("sdxl", "2kto4k"): OfficialPidCheckpoint("sdxl", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth", 4, 4, True),
    ("qwenimage", "2kto4k"): OfficialPidCheckpoint("qwenimage", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("qwenimage-2512", "2kto4k"): OfficialPidCheckpoint("qwenimage-2512", "2kto4k", "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth", 4, 16, True),
    ("dinov2", "2k"): OfficialPidCheckpoint("dinov2", "2k", "checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step/model_ema_bf16.pth", 4, 768, True),
    ("siglip", "2k"): OfficialPidCheckpoint("siglip", "2k", "checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth", 8, 1152, True),
}
OFFICIAL_PID_CHECKPOINTS[("zimage", "2k")] = OFFICIAL_PID_CHECKPOINTS[("flux", "2k")]
OFFICIAL_PID_CHECKPOINTS[("zimage", "2kto4k")] = OFFICIAL_PID_CHECKPOINTS[("flux", "2kto4k")]
OFFICIAL_PID_CHECKPOINTS[("zimage-turbo", "2k")] = OFFICIAL_PID_CHECKPOINTS[("flux", "2k")]
OFFICIAL_PID_CHECKPOINTS[("zimage-turbo", "2kto4k")] = OFFICIAL_PID_CHECKPOINTS[("flux", "2kto4k")]
OFFICIAL_PID_CHECKPOINTS[("flux2-klein-4b", "2k")] = OFFICIAL_PID_CHECKPOINTS[("flux2", "2k")]
OFFICIAL_PID_CHECKPOINTS[("flux2-klein-4b", "2kto4k")] = OFFICIAL_PID_CHECKPOINTS[("flux2", "2kto4k")]
OFFICIAL_PID_CHECKPOINTS[("flux2-klein-9b", "2k")] = OFFICIAL_PID_CHECKPOINTS[("flux2", "2k")]
OFFICIAL_PID_CHECKPOINTS[("flux2-klein-9b", "2kto4k")] = OFFICIAL_PID_CHECKPOINTS[("flux2", "2kto4k")]


@dataclass(frozen=True)
class PidDecodeRequest:
    decode_backend: str = "pid"
    pid_backbone: str = "qwenimage"
    pid_ckpt_type: str = "2kto4k"
    pid_checkpoint: str = ""
    pid_tile_latent: int = 0
    pid_compile: bool = False
    pid_steps: int = 4
    pid_sigma: float = 0.0
    runtime_activation_enabled: bool = False

    def validate(self) -> None:
        if self.decode_backend not in {"vae", "pid"}:
            raise ValueError("decode_backend must be vae or pid")
        if self.pid_ckpt_type not in {"2k", "2kto4k"}:
            raise ValueError("pid_ckpt_type must be 2k or 2kto4k")
        if self.pid_tile_latent < 0:
            raise ValueError("pid_tile_latent must be >= 0")
        if self.pid_steps < 1:
            raise ValueError("pid_steps must be >= 1")
        if self.pid_sigma < 0.0:
            raise ValueError("pid_sigma must be >= 0")
        if self.runtime_activation_enabled:
            raise ValueError("PiD runtime activation is not enabled by this contract")


def get_official_pid_checkpoint(backbone: str, ckpt_type: str) -> OfficialPidCheckpoint:
    key = (str(backbone or "").strip().lower(), str(ckpt_type or "").strip().lower())
    try:
        return OFFICIAL_PID_CHECKPOINTS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(f"{item[0]}+{item[1]}" for item in OFFICIAL_PID_CHECKPOINTS))
        raise KeyError(f"unsupported PiD checkpoint pair {key}; valid: {valid}") from exc


def build_pid_decode_request(payload: PidDecodeRequest | Mapping[str, Any] | None = None) -> PidDecodeRequest:
    if isinstance(payload, PidDecodeRequest):
        request = payload
    else:
        values = dict(payload or {})
        request = PidDecodeRequest(
            decode_backend=str(values.get("decode_backend", PidDecodeRequest.decode_backend)).strip().lower(),
            pid_backbone=str(values.get("pid_backbone", PidDecodeRequest.pid_backbone)).strip().lower(),
            pid_ckpt_type=str(values.get("pid_ckpt_type", PidDecodeRequest.pid_ckpt_type)).strip().lower(),
            pid_checkpoint=str(values.get("pid_checkpoint", PidDecodeRequest.pid_checkpoint)).strip(),
            pid_tile_latent=int(values.get("pid_tile_latent", PidDecodeRequest.pid_tile_latent)),
            pid_compile=_boolish(values.get("pid_compile", PidDecodeRequest.pid_compile)),
            pid_steps=int(values.get("pid_steps", PidDecodeRequest.pid_steps)),
            pid_sigma=float(values.get("pid_sigma", PidDecodeRequest.pid_sigma)),
            runtime_activation_enabled=_boolish(values.get("runtime_activation_enabled", False)),
        )
    request.validate()
    if request.decode_backend == "pid":
        get_official_pid_checkpoint(request.pid_backbone, request.pid_ckpt_type)
    return request


def build_pid_checkpoint_resolution(
    request: PidDecodeRequest | Mapping[str, Any] | None = None,
    *,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    req = build_pid_decode_request(request)
    spec = get_official_pid_checkpoint(req.pid_backbone, req.pid_ckpt_type)
    user_path = Path(req.pid_checkpoint) if req.pid_checkpoint else None
    suggested_path = Path(checkpoint_root, spec.relative_path) if checkpoint_root else Path(spec.relative_path)
    selected = user_path or suggested_path
    return {
        "schema_version": 1,
        "resolution": "pid_checkpoint_resolution_v0",
        "decode_backend": req.decode_backend,
        "backbone": req.pid_backbone,
        "ckpt_type": req.pid_ckpt_type,
        "selected_checkpoint": str(selected),
        "checkpoint_exists": selected.is_file(),
        "checkpoint_user_provided": bool(user_path),
        "official_suggested_path": str(suggested_path),
        "hf_repo": PID_HF_REPO,
        "pid_scale": spec.scale,
        "latent_channels": spec.latent_channels,
        "code_license": PID_CODE_LICENSE,
        "weight_license": PID_WEIGHT_LICENSE,
        "auto_download_allowed": False,
        "bundle_weights_allowed": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
    }


def build_pid_capability_report(
    request: PidDecodeRequest | Mapping[str, Any] | None = None,
    *,
    source_tree: str | Path = "ref/PiD-main",
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    req = build_pid_decode_request(request)
    source = Path(source_tree)
    resolution = build_pid_checkpoint_resolution(req, checkpoint_root=checkpoint_root)
    code_available = (source / "pid" / "_src" / "inference" / "decoder.py").is_file()
    license_available = (source / "LICENSE").is_file()
    checkpoint_available = bool(resolution["checkpoint_exists"])
    return {
        "schema_version": 1,
        "report": "pid_decode_capability_report_v0",
        "decode_backend": req.decode_backend,
        "backbone": req.pid_backbone,
        "ckpt_type": req.pid_ckpt_type,
        "source_tree": str(source),
        "source_code_available": code_available,
        "source_license_available": license_available,
        "checkpoint_available": checkpoint_available,
        "checkpoint_resolution": resolution,
        "source_code_license": PID_CODE_LICENSE if code_available and license_available else "unknown",
        "weight_license": PID_WEIGHT_LICENSE,
        "license_warning_required": True,
        "user_must_supply_weights": True,
        "auto_download_allowed": False,
        "bundle_weights_allowed": False,
        "native_official_loader_ready": False,
        "tiny_smoke_loader_ready": True,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if checkpoint_available else ["pid_checkpoint_missing"],
    }


def normalize_qwen_pid_latent(samples: torch.Tensor) -> torch.Tensor:
    x = samples
    if x.dim() == 5:
        if x.shape[2] != 1:
            raise ValueError("5D PiD latents must have a singleton frame dimension")
        x = x[:, :, 0]
    if x.dim() != 4:
        raise ValueError("PiD latent must be [batch, channels, height, width]")
    if x.shape[1] != 16:
        raise ValueError("qwenimage PiD latent normalization expects 16 channels")
    mean = torch.tensor(QWEN_LATENTS_MEAN, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    std = torch.tensor(QWEN_LATENTS_STD, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    return (x.float() - mean) / std


def load_pid_tiny_smoke_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != SMOKE_CHECKPOINT_KIND:
        raise ValueError("not a Lulynx PiD tiny smoke checkpoint")
    weight = payload.get("weight")
    bias = payload.get("bias")
    scale = int(payload.get("pid_scale", 4))
    vae_down = int(payload.get("vae_down_factor", 8))
    if not isinstance(weight, torch.Tensor) or weight.dim() != 4 or weight.shape[0] != 3:
        raise ValueError("tiny smoke checkpoint weight must be [3, channels, 1, 1]")
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (3,):
        raise ValueError("tiny smoke checkpoint bias must be [3]")
    if scale < 1 or vae_down < 1:
        raise ValueError("tiny smoke checkpoint scale factors must be >= 1")
    return {"weight": weight.float(), "bias": bias.float(), "pid_scale": scale, "vae_down_factor": vae_down}


def run_pid_tiny_decode_smoke(
    latents: torch.Tensor,
    checkpoint_path: str | Path,
    request: PidDecodeRequest | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    req = build_pid_decode_request(request)
    if req.decode_backend != "pid":
        raise ValueError("tiny PiD decode smoke requires decode_backend=pid")
    state = load_pid_tiny_smoke_checkpoint(checkpoint_path)
    x = normalize_qwen_pid_latent(latents) if req.pid_backbone.startswith("qwenimage") else latents.float()
    if x.dim() != 4:
        raise ValueError("latents must be 4D after normalization")
    if x.shape[1] != state["weight"].shape[1]:
        raise ValueError("latent channel count does not match tiny smoke checkpoint")
    y = F.conv2d(x, state["weight"].to(device=x.device), state["bias"].to(device=x.device))
    for step in range(req.pid_steps):
        y = torch.tanh(y + (req.pid_sigma / float(step + 1)))
    up = int(state["pid_scale"]) * int(state["vae_down_factor"])
    pixels = F.interpolate(y, scale_factor=up, mode="bilinear", align_corners=False).clamp(-1.0, 1.0)
    return {
        "schema_version": 1,
        "result": "pid_tiny_decode_smoke_v0",
        "ok": bool(torch.isfinite(pixels).all().item()),
        "pixels": pixels,
        "pixel_shape": list(pixels.shape),
        "pid_steps": req.pid_steps,
        "pid_scale": int(state["pid_scale"]),
        "vae_down_factor": int(state["vae_down_factor"]),
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
    }


def build_pid_request_field_inventory() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "inventory": "pid_decode_request_fields_v0",
        "fields": ["decode_backend", "pid_backbone", "pid_ckpt_type", "pid_checkpoint", "pid_tile_latent", "pid_compile"],
        "sample_payload": {
            "decode_backend": "pid",
            "pid_backbone": "qwenimage",
            "pid_ckpt_type": "2kto4k",
            "pid_checkpoint": "H:/models/pid/model_ema_bf16.pth",
            "pid_tile_latent": 64,
            "pid_compile": False,
        },
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
    }


def build_pid_decode_scorecard(
    *,
    registry_ok: bool = False,
    capability_report_ok: bool = False,
    tiny_decode_smoke_ok: bool = False,
    license_warning_ok: bool = False,
    no_bundle_ok: bool = False,
    request_field_inventory_ok: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    checks = {
        "registry_missing": registry_ok,
        "capability_report_missing": capability_report_ok,
        "tiny_decode_smoke_missing": tiny_decode_smoke_ok,
        "license_warning_missing": license_warning_ok,
        "no_bundle_proof_missing": no_bundle_ok,
        "request_field_inventory_missing": request_field_inventory_ok,
    }
    blockers.extend(reason for reason, passed in checks.items() if not passed)
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "pid_native_decode_backend_contract_v0",
        "ok": ready,
        "contract_ready": ready,
        "native_official_loader_ready": False,
        "tiny_smoke_loader_ready": True,
        "auto_download_allowed": False,
        "bundle_weights_allowed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add official PiD loader preflight with user-provided checkpoint only"
            if ready
            else "complete PiD registry, capability, smoke, license, and request-field proofs"
        ),
    }


def build_pid_official_loader_preflight(
    capability_report: Mapping[str, Any],
    *,
    accepted_weight_license: bool = False,
    intended_use: str = "research",
) -> dict[str, Any]:
    report = dict(capability_report)
    resolution = dict(report.get("checkpoint_resolution") or {})
    blockers: list[str] = []
    if not report.get("source_code_available"):
        blockers.append("pid_source_code_missing")
    if not report.get("checkpoint_available"):
        blockers.append("pid_checkpoint_missing")
    if not resolution.get("checkpoint_user_provided"):
        blockers.append("pid_checkpoint_not_user_provided")
    if not accepted_weight_license:
        blockers.append("pid_weight_license_not_acknowledged")
    if str(intended_use or "").strip().lower() in {"commercial", "production_commercial", "redistribution"}:
        blockers.append("commercial_use_blocked_by_weight_license")
    ready = not blockers
    return {
        "schema_version": 1,
        "preflight": "pid_official_loader_preflight_v0",
        "ok": ready,
        "official_loader_preflight_ready": ready,
        "backbone": report.get("backbone"),
        "ckpt_type": report.get("ckpt_type"),
        "checkpoint": resolution.get("selected_checkpoint", ""),
        "checkpoint_user_provided": bool(resolution.get("checkpoint_user_provided")),
        "accepted_weight_license": bool(accepted_weight_license),
        "intended_use": str(intended_use or "research"),
        "weight_license": PID_WEIGHT_LICENSE,
        "native_official_loader_ready": False,
        "auto_download_allowed": False,
        "bundle_weights_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "default_behavior_changed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add official PiD model adapter smoke with this user-provided checkpoint"
            if ready
            else "provide a local checkpoint, acknowledge the weight license, and keep redistribution disabled"
        ),
    }


def build_pid_decode_runtime_boundary(
    preflight: Mapping[str, Any],
    *,
    target_routes: tuple[str, ...] = ("preview", "generation", "postprocess"),
) -> dict[str, Any]:
    preflight_ready = bool(dict(preflight).get("official_loader_preflight_ready"))
    routes = tuple(str(route or "").strip().lower() for route in target_routes if str(route or "").strip())
    blockers: list[str] = []
    if not preflight_ready:
        blockers.append("official_loader_preflight_missing")
    if not routes:
        blockers.append("target_routes_missing")
    invalid = [route for route in routes if route not in {"preview", "generation", "postprocess"}]
    if invalid:
        blockers.append("unsupported_target_route:" + ",".join(sorted(invalid)))
    boundary_ready = not blockers
    return {
        "schema_version": 1,
        "boundary": "pid_decode_runtime_boundary_v0",
        "ok": boundary_ready,
        "boundary_ready": boundary_ready,
        "target_routes": list(routes),
        "decode_backend": "pid",
        "vae_replacement_allowed": False,
        "preview_route_registered": False,
        "generation_route_registered": False,
        "postprocess_route_registered": False,
        "request_adapter_registered": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire a guarded preview-only PiD decode adapter after official loader smoke"
            if boundary_ready
            else "complete official loader preflight before any VAE replacement route"
        ),
    }


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "OFFICIAL_PID_CHECKPOINTS",
    "PID_CODE_LICENSE",
    "PID_HF_REPO",
    "PID_WEIGHT_LICENSE",
    "PidDecodeRequest",
    "SMOKE_CHECKPOINT_KIND",
    "build_pid_capability_report",
    "build_pid_checkpoint_resolution",
    "build_pid_decode_request",
    "build_pid_decode_scorecard",
    "build_pid_decode_runtime_boundary",
    "build_pid_official_loader_preflight",
    "build_pid_request_field_inventory",
    "get_official_pid_checkpoint",
    "load_pid_tiny_smoke_checkpoint",
    "normalize_qwen_pid_latent",
    "run_pid_tiny_decode_smoke",
]
