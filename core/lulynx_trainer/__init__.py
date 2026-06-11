"""
Lulynx Native LoRA Trainer package facade.

Keep package import lightweight so callers can safely import submodules such as
``module_offload_contract`` or ``config_adapter`` without eagerly importing the
entire training stack.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "LulynxTrainer",
    "LulynxConfig",
    "ConfigAdapter",
    "TrainingLoop",
    "ModelArch",
    "NetworkType",
    "OptimizerType",
    "SchedulerType",
    "ModelLoader",
    "LoadedModel",
    "LoRAInjector",
    "LoRALayer",
    "LoRALinear",
    "DoRALinear",
    "DoRAInjector",
    "LoraPlusConfig",
    "create_lora_plus_optimizer",
    "CaptionDataset",
    "BucketManager",
    "create_dataloader",
    "TextEncoderDiskCache",
    "TrainingSampler",
    "create_sampler_from_trainer",
    "MultiGPUAccelerator",
    "get_accelerator",
    "init_accelerator",
    "DynamicResourceManager",
    "ResourceConfig",
    "oom_safe_execute",
    "CheckpointManager",
    "CheckpointState",
    "BlockSwapOffloader",
    "apply_channels_last",
    "PipelineSlicer",
    "SlicingConfig",
    "probe_attention_backends",
    "select_attention_backend",
    "apply_attention_to_unet",
    "estimate_vram_for_config",
]

_LAZY_EXPORTS = {
    "LulynxTrainer": (".trainer", "LulynxTrainer"),
    "LulynxConfig": (".config", "LulynxConfig"),
    "ModelArch": (".config", "ModelArch"),
    "NetworkType": (".config", "NetworkType"),
    "OptimizerType": (".config", "OptimizerType"),
    "SchedulerType": (".config", "SchedulerType"),
    "ConfigAdapter": (".config_adapter", "ConfigAdapter"),
    "ModelLoader": (".model_loader", "ModelLoader"),
    "LoadedModel": (".model_loader", "LoadedModel"),
    "LoRAInjector": (".lora_injector", "LoRAInjector"),
    "LoRALayer": (".lora_injector", "LoRALayer"),
    "LoRALinear": (".lora_injector", "LoRALinear"),
    "CaptionDataset": (".dataset_loader", "CaptionDataset"),
    "BucketManager": (".dataset_loader", "BucketManager"),
    "create_dataloader": (".dataset_loader", "create_dataloader"),
    "TextEncoderDiskCache": (".dataset_loader", "TextEncoderDiskCache"),
    "TrainingLoop": (".training_loop", "TrainingLoop"),
    "TrainingSampler": (".sampler", "TrainingSampler"),
    "create_sampler_from_trainer": (".sampler", "create_sampler_from_trainer"),
    "MultiGPUAccelerator": (".accelerator", "MultiGPUAccelerator"),
    "get_accelerator": (".accelerator", "get_accelerator"),
    "init_accelerator": (".accelerator", "init_accelerator"),
    "DynamicResourceManager": (".resource_manager", "DynamicResourceManager"),
    "ResourceConfig": (".resource_manager", "ResourceConfig"),
    "oom_safe_execute": (".resource_manager", "oom_safe_execute"),
    "CheckpointManager": (".checkpoint_manager", "CheckpointManager"),
    "CheckpointState": (".checkpoint_manager", "CheckpointState"),
    "DoRALinear": (".lora_variants", "DoRALinear"),
    "DoRAInjector": (".lora_variants", "DoRAInjector"),
    "LoraPlusConfig": (".lora_variants", "LoraPlusConfig"),
    "create_lora_plus_optimizer": (".lora_variants", "create_lora_plus_optimizer"),
    "BlockSwapOffloader": (".memory_optimizations", "BlockSwapOffloader"),
    "apply_channels_last": (".memory_optimizations", "apply_channels_last"),
    "PipelineSlicer": (".memory_optimizations", "PipelineSlicer"),
    "SlicingConfig": (".memory_optimizations", "SlicingConfig"),
    "probe_attention_backends": (".memory_optimizations", "probe_attention_backends"),
    "select_attention_backend": (".memory_optimizations", "select_attention_backend"),
    "apply_attention_to_unet": (".memory_optimizations", "apply_attention_to_unet"),
    "estimate_vram_for_config": (".memory_optimizations", "estimate_vram_for_config"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
