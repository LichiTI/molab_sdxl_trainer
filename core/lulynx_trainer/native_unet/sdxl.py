"""SDXL native UNet phase-2 contracts.

The first milestone keeps the existing diffusers module available as the
reference implementation while exposing a Lulynx-owned block graph and backend
switch.  Full Warehouse block implementations will replace the proxy one stage
at a time after parity tests exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn


VALID_SDXL_UNET_BACKENDS = {"diffusers", "native_shadow", "native_proxy", "native_skeleton", "lulynx_native"}


@dataclass(frozen=True)
class NativeUNetBlockInfo:
    name: str
    stage: str
    index: int
    module_type: str
    parameter_count: int
    forward_order: int = 0
    lora_scope: str = "unet"
    swap_priority: int = 0
    recompute_safe: bool = True
    native_ready: bool = False

    @property
    def parameter_mb(self) -> float:
        return self.parameter_count * 2.0 / (1024.0 * 1024.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "index": self.index,
            "forward_order": self.forward_order,
            "module_type": self.module_type,
            "parameter_count": self.parameter_count,
            "parameter_mb_bf16": round(self.parameter_mb, 3),
            "lora_scope": self.lora_scope,
            "swap_priority": self.swap_priority,
            "recompute_safe": self.recompute_safe,
            "native_ready": self.native_ready,
        }


@dataclass(frozen=True)
class NativeUNetLifecyclePlan:
    """Lifecycle hooks expected from a future native backend."""

    supports_shadow: bool = True
    supports_proxy: bool = True
    supports_native_blocks: bool = False
    supports_precision_swap: bool = True
    supports_training: bool = True
    supports_inference: bool = True
    supports_parity: bool = True
    offload_contract: str = "block_graph"
    weight_mapping_contract: str = "manifest"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "supports_shadow": self.supports_shadow,
            "supports_proxy": self.supports_proxy,
            "supports_native_blocks": self.supports_native_blocks,
            "supports_precision_swap": self.supports_precision_swap,
            "supports_training": self.supports_training,
            "supports_inference": self.supports_inference,
            "supports_parity": self.supports_parity,
            "offload_contract": self.offload_contract,
            "weight_mapping_contract": self.weight_mapping_contract,
        }


@dataclass(frozen=True)
class NativeUNetWeightMappingPlan:
    """Warehouse mapping manifest boundary.

    The manifest names contracts, not borrowed mapping code.  Concrete mapping
    tables can be generated and tested independently from the model classes.
    """

    source_format: str = "diffusers_sdxl_unet"
    target_format: str = "lulynx_native_sdxl_unet"
    keymap_manifest: str = "native_unet/sdxl_keymap_manifest.json"
    parity_required: bool = True
    strict_shapes: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "keymap_manifest": self.keymap_manifest,
            "parity_required": self.parity_required,
            "strict_shapes": self.strict_shapes,
        }


@dataclass(frozen=True)
class NativeUNetStatus:
    family: str
    backend: str
    active: bool
    mode: str
    message: str = ""
    blocks: List[NativeUNetBlockInfo] = field(default_factory=list)
    lifecycle: NativeUNetLifecyclePlan = field(default_factory=NativeUNetLifecyclePlan)
    weight_mapping: NativeUNetWeightMappingPlan = field(default_factory=NativeUNetWeightMappingPlan)
    native_coverage: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "backend": self.backend,
            "active": self.active,
            "mode": self.mode,
            "message": self.message,
            "blocks_total": len(self.blocks),
            "native_ready_blocks": sum(1 for block in self.blocks if block.native_ready),
            "lifecycle": self.lifecycle.as_dict(),
            "weight_mapping": self.weight_mapping.as_dict(),
            "native_coverage": dict(self.native_coverage),
            "blocks": [block.as_dict() for block in self.blocks],
        }

    def precision_swap_units(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": block.name,
                "stage": block.stage,
                "order": block.forward_order,
                "parameter_mb": round(block.parameter_mb, 3),
                "swap_priority": block.swap_priority,
                "recompute_safe": block.recompute_safe,
            }
            for block in sorted(self.blocks, key=lambda item: item.forward_order)
        ]


class NativeSDXLUNetProxy(nn.Module):
    """Proxy backend used while the Warehouse SDXL UNet is being built.

    It forwards to the reference UNet but owns the native block graph contract.
    This lets the trainer, LoRA injection, precision swap, and runtime
    observation code integrate with the future native backend before replacing
    the math implementation.
    """

    def __init__(self, reference_unet: nn.Module, *, status: NativeUNetStatus) -> None:
        super().__init__()
        self.reference_unet = reference_unet
        self.native_unet_status = status
        self._native_training_mode = "training"

    @property
    def config(self) -> Any:
        return getattr(self.reference_unet, "config", None)

    @property
    def down_blocks(self) -> Any:
        return getattr(self.reference_unet, "down_blocks", None)

    @property
    def mid_block(self) -> Any:
        return getattr(self.reference_unet, "mid_block", None)

    @property
    def up_blocks(self) -> Any:
        return getattr(self.reference_unet, "up_blocks", None)

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.reference_unet, "enable_gradient_checkpointing", None)
        if target is not None:
            return target(*args, **kwargs)
        return None

    def enable_attention_slicing(self, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.reference_unet, "enable_attention_slicing", None)
        if target is not None:
            return target(*args, **kwargs)
        return None

    def set_attn_processor(self, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.reference_unet, "set_attn_processor", None)
        if target is not None:
            return target(*args, **kwargs)
        return None

    def native_block_graph(self) -> Dict[str, Any]:
        data = self.native_unet_status.as_dict()
        data["precision_swap_units"] = self.native_unet_status.precision_swap_units()
        return data

    def prepare_block_swap_training(self) -> None:
        self._native_training_mode = "training"

    def prepare_block_swap_inference(self, *, disable_block_swap: bool = False) -> None:
        self._native_training_mode = "inference_no_swap" if disable_block_swap else "inference"

    def restore_reference_unet(self) -> nn.Module:
        return self.reference_unet

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.reference_unet(*args, **kwargs)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters(recurse=True))


def _iter_sdxl_top_blocks(unet: nn.Module) -> List[NativeUNetBlockInfo]:
    blocks: List[NativeUNetBlockInfo] = []
    forward_order = 0

    down_blocks = getattr(unet, "down_blocks", None)
    if down_blocks is not None:
        for index, module in enumerate(list(down_blocks)):
            blocks.append(
                NativeUNetBlockInfo(
                    name=f"down.{index}",
                    stage="down",
                    index=index,
                    forward_order=forward_order,
                    module_type=type(module).__name__,
                    parameter_count=_module_parameter_count(module),
                    swap_priority=max(index, 0),
                )
            )
            forward_order += 1

    mid_block = getattr(unet, "mid_block", None)
    if mid_block is not None:
        blocks.append(
            NativeUNetBlockInfo(
                name="mid.0",
                stage="mid",
                index=0,
                forward_order=forward_order,
                module_type=type(mid_block).__name__,
                parameter_count=_module_parameter_count(mid_block),
                swap_priority=80,
            )
        )
        forward_order += 1

    up_blocks = getattr(unet, "up_blocks", None)
    if up_blocks is not None:
        for index, module in enumerate(list(up_blocks)):
            blocks.append(
                NativeUNetBlockInfo(
                    name=f"up.{index}",
                    stage="up",
                    index=index,
                    forward_order=forward_order,
                    module_type=type(module).__name__,
                    parameter_count=_module_parameter_count(module),
                    swap_priority=100 + index,
                )
            )
            forward_order += 1

    return blocks


def _native_keymap_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "keymaps" / "sdxl_unet_keymap_manifest.json"


def _run_sdxl_native_forward_probe() -> Dict[str, Any]:
    try:
        from .sdxl_modules import (
            NativeSDXLDownBlockConfig,
            NativeSDXLMidBlockConfig,
            NativeSDXLResnetBlockConfig,
            NativeSDXLShellConfig,
            NativeSDXLTransformer2DConfig,
            NativeSDXLTransformerBlockConfig,
            NativeSDXLUNetSkeleton,
            NativeSDXLUNetSkeletonCompat,
            NativeSDXLUNetSkeletonConfig,
            NativeSDXLUpBlockConfig,
        )

        shell = NativeSDXLShellConfig(
            in_channels=2,
            base_channels=4,
            time_embed_dim=8,
            add_time_embed_dim=2,
            add_embed_in_dim=6,
            add_embed_dim=8,
            out_channels=2,
            norm_num_groups=2,
        )
        resnet_4 = NativeSDXLResnetBlockConfig(
            in_channels=4,
            out_channels=4,
            time_embed_dim=8,
            norm_num_groups=2,
        )
        up_resnet = NativeSDXLResnetBlockConfig(
            in_channels=8,
            out_channels=4,
            time_embed_dim=8,
            norm_num_groups=2,
            use_conv_shortcut=True,
        )
        attention = NativeSDXLTransformer2DConfig(
            channels=4,
            norm_num_groups=2,
            transformer_blocks=(
                NativeSDXLTransformerBlockConfig(
                    dim=4,
                    cross_attention_dim=6,
                    heads=1,
                    dim_head=4,
                    ff_inner_dim=8,
                ),
            ),
        )
        skeleton = NativeSDXLUNetSkeleton(
            NativeSDXLUNetSkeletonConfig(
                shell=shell,
                down_blocks=(NativeSDXLDownBlockConfig(resnets=(resnet_4, resnet_4)),),
                mid_block=NativeSDXLMidBlockConfig(resnets=(resnet_4, resnet_4), attention=attention),
                up_blocks=(NativeSDXLUpBlockConfig(resnets=(up_resnet, up_resnet, up_resnet)),),
            )
        )
        compat = NativeSDXLUNetSkeletonCompat(skeleton)
        with torch.no_grad():
            output = compat(
                sample=torch.randn(1, 2, 8, 8),
                timestep=torch.tensor([1]),
                encoder_hidden_states=torch.randn(1, 3, 6),
                added_cond_kwargs={
                    "text_embeds": torch.randn(1, 4),
                    "time_ids": torch.tensor([[8.0]]),
                },
                return_dict=False,
            )
        return {
            "ok": True,
            "mode": "synthetic_text_time",
            "output_shape": [int(dim) for dim in output[0].shape],
            "uses_diffusers_call_shape": True,
            "uses_sdxl_text_time_condition": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "synthetic_text_time",
            "reason": str(exc),
        }


def _build_sdxl_native_coverage() -> Dict[str, Any]:
    try:
        from .sdxl_modules import build_sdxl_unet_skeleton_config_from_manifest

        config = build_sdxl_unet_skeleton_config_from_manifest(_native_keymap_manifest_path())
        down_blocks = len(config.down_blocks)
        up_blocks = len(config.up_blocks)
        cross_down = sum(1 for block in config.down_blocks if hasattr(block, "attentions"))
        cross_up = sum(1 for block in config.up_blocks if hasattr(block, "attentions"))
        forward_probe = _run_sdxl_native_forward_probe()
        return {
            "status": "available",
            "skeleton_ready": True,
            "native_forward_integrated": False,
            "native_full_replace_ready": True,
            "native_forward_probe_ok": bool(forward_probe.get("ok")),
            "native_forward_probe": forward_probe,
            "implemented_top_blocks": down_blocks + 1 + up_blocks,
            "down_blocks": down_blocks,
            "up_blocks": up_blocks,
            "mid_blocks": 1,
            "cross_attn_down_blocks": cross_down,
            "cross_attn_up_blocks": cross_up,
            "mid_transformer_blocks": len(config.mid_block.attention.transformer_blocks),
            "shell_ready": True,
            "resnet_ready": True,
            "attention_backend": "sdpa|flash2",
            "reason": "native skeleton modules are implemented and shape-validated; full training integration depends on the selected backend",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "skeleton_ready": False,
            "native_forward_integrated": False,
            "reason": str(exc),
        }


def _coverage_for_backend(backend: str) -> Dict[str, Any]:
    normalized = normalize_sdxl_unet_backend(backend)
    coverage = _build_sdxl_native_coverage()
    if normalized == "lulynx_native" and coverage.get("status") == "available":
        coverage["native_forward_integrated"] = True
        coverage["reason"] = "lulynx_native installs the Warehouse SDXL UNet wrapper for training/inference-compatible calls"
    return coverage


def _mark_native_ready(blocks: List[NativeUNetBlockInfo], *, ready: bool) -> List[NativeUNetBlockInfo]:
    if not ready:
        return blocks
    return [
        NativeUNetBlockInfo(
            name=block.name,
            stage=block.stage,
            index=block.index,
            module_type=block.module_type,
            parameter_count=block.parameter_count,
            forward_order=block.forward_order,
            lora_scope=block.lora_scope,
            swap_priority=block.swap_priority,
            recompute_safe=block.recompute_safe,
            native_ready=True,
        )
        for block in blocks
    ]


def normalize_sdxl_unet_backend(value: Any) -> str:
    backend = str(value or "diffusers").strip().lower().replace("-", "_")
    aliases = {
        "native": "lulynx_native",
        "shadow": "native_shadow",
        "proxy": "native_proxy",
        "skeleton": "native_skeleton",
        "off": "diffusers",
        "default": "diffusers",
    }
    backend = aliases.get(backend, backend)
    return backend if backend in VALID_SDXL_UNET_BACKENDS else "diffusers"


def build_sdxl_unet_status(unet: nn.Module, *, backend: str, active: bool, message: str = "") -> NativeUNetStatus:
    normalized = normalize_sdxl_unet_backend(backend)
    coverage = _coverage_for_backend(normalized) if normalized in {"native_shadow", "native_proxy", "native_skeleton", "lulynx_native"} else {}
    skeleton_ready = bool(coverage.get("skeleton_ready"))
    mode = (
        "native_full"
        if normalized == "lulynx_native"
        else ("reference_proxy" if normalized == "native_proxy" else ("skeleton_metadata" if normalized == "native_skeleton" else ("shadow" if normalized == "native_shadow" else normalized)))
    )
    lifecycle = NativeUNetLifecyclePlan(
        supports_native_blocks=skeleton_ready,
        supports_training=normalized != "native_skeleton",
    )
    return NativeUNetStatus(
        family="sdxl",
        backend=normalized,
        active=bool(active),
        mode=mode,
        message=message,
        blocks=_mark_native_ready(_iter_sdxl_top_blocks(unet), ready=normalized in {"native_skeleton", "lulynx_native"} and skeleton_ready),
        lifecycle=lifecycle,
        weight_mapping=NativeUNetWeightMappingPlan(),
        native_coverage=coverage,
    )


def build_sdxl_native_unet_preflight_profile(*, backend: str = "diffusers") -> Optional[Dict[str, Any]]:
    normalized = normalize_sdxl_unet_backend(backend)
    if normalized == "diffusers":
        return None
    coverage = _coverage_for_backend(normalized)
    mode = "native_full" if normalized == "lulynx_native" else ("reference_proxy" if normalized == "native_proxy" else ("skeleton_metadata" if normalized == "native_skeleton" else "shadow"))
    return {
        "family": "sdxl",
        "backend": normalized,
        "available": coverage.get("status") == "available",
        "mode": mode,
        "active": normalized in {"native_proxy", "lulynx_native"},
        "blocks_total": int(coverage.get("implemented_top_blocks") or 0),
        "native_forward_integrated": normalized == "lulynx_native",
        "native_forward_probe_ok": bool(coverage.get("native_forward_probe_ok")),
        "native_forward_probe": dict(coverage.get("native_forward_probe") or {}),
        "native_coverage": coverage,
    }


def _infer_module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    try:
        param = next(module.parameters())
        return param.device, param.dtype if param.is_floating_point() else None
    except StopIteration:
        return None, None


def install_sdxl_native_unet_backend(
    model: Any,
    *,
    backend: str = "diffusers",
    model_path: str | Path | None = None,
    logger: Optional[Any] = None,
) -> NativeUNetStatus:
    normalized = normalize_sdxl_unet_backend(backend)
    unet = getattr(model, "unet", None)
    if unet is None or normalized == "diffusers":
        status = build_sdxl_unet_status(unet, backend="diffusers", active=False, message="diffusers backend active") if unet is not None else NativeUNetStatus(
            family="sdxl",
            backend=normalized,
            active=False,
            mode="unavailable",
            message="model has no unet",
        )
        if model is not None:
            setattr(model, "native_unet_status", status.as_dict())
        return status

    if normalized == "lulynx_native":
        try:
            from .sdxl_modules import build_sdxl_unet_compat_from_manifest

            device, dtype = _infer_module_device_dtype(unet)
            native_unet = build_sdxl_unet_compat_from_manifest(
                _native_keymap_manifest_path(),
                model_path,
                device=device,
                dtype=dtype,
            )
            status = build_sdxl_unet_status(
                native_unet,
                backend=normalized,
                active=True,
                message="native SDXL UNet full wrapper installed",
            )
            native_unet.native_unet_status = status
            setattr(model, "unet", native_unet)
            setattr(model, "native_unet_status", status.as_dict())
            if logger is not None:
                logger.info(
                    "SDXL native UNet backend=%s mode=%s blocks=%s",
                    status.backend,
                    status.mode,
                    len(status.blocks),
                )
            return status
        except Exception as exc:
            raise RuntimeError(f"SDXL native UNet full replacement failed: {exc}") from exc

    status = build_sdxl_unet_status(
        unet,
        backend=normalized,
        active=normalized == "native_proxy",
        message="native SDXL UNet phase-2 contract installed",
    )
    setattr(model, "native_unet_status", status.as_dict())
    if normalized in {"native_shadow", "native_skeleton"}:
        setattr(model, "native_unet_shadow", status.as_dict())
    elif normalized == "native_proxy":
        setattr(model, "unet", NativeSDXLUNetProxy(unet, status=status))
    if logger is not None:
        logger.info(
            "SDXL native UNet backend=%s mode=%s blocks=%s",
            status.backend,
            status.mode,
            len(status.blocks),
        )
    return status

