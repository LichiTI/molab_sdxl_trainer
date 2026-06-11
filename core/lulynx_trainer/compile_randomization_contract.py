"""Shared torch.compile randomization compatibility contracts.

These contracts are evidence metadata only. They do not enable compile and do
not change training behavior; runners and planners use them to keep dynamic
training features out of release claims unless they have an eager boundary or a
separate A/B proof.
"""

from __future__ import annotations

from typing import Any


def sdxl_compile_randomization_contract() -> dict[str, Any]:
    """Return the conservative SDXL compile A/B compatibility contract."""

    return {
        "schema_version": 1,
        "policy": "sdxl_compile_ab_randomization_compat_v0",
        "compile_scope": "per_block",
        "dynamic_shape_policy": "static_batch_resolution_required",
        "allowed_graph_outside_randomization": [
            "diffusion_timestep_sampling",
            "base_noise_tensor_sampling",
            "cpu_or_dataloader_augmentation_with_fixed_output_shape",
            "caption_shuffle_or_dropout_before_tokenized_fixed_shape_batch",
            "lora_dropout_inside_static_module_path",
        ],
        "keep_eager_boundary": [
            "pyramid_or_multires_noise_generation",
            "noise_offset_random_strength",
            "perlin_noise_offset_generation",
            "dataset_caption_dropout_and_shuffle",
            "image_flip_or_color_augmentation",
            "bucket_sampler_and_caption_length_bucket_choice",
            "staged_resolution_dataset_switch",
        ],
        "blocked_or_requires_eager_boundary": [
            "pyramid_noise_with_dynamic_levels_or_scales",
            "dynamic_resolution_or_bucket_shape_changes",
            "resolution_aware_batch_size_changes",
            "random_conditioning_paths_that_change_tensor_shapes",
            "random_module_enable_disable_inside_compiled_blocks",
            "safe_fallback_paths_that_change_graph_structure",
            "controlnet_or_ip_adapter_dynamic_paths_without_separate_probe",
            "memory_or_block_swap_paths",
            "attention_backend_fallback_paths",
        ],
        "training_core_observed_features": [
            {
                "feature": "base diffusion timestep/noise",
                "files": ["training_loop.py", "trainer.py", "controlnet_trainer.py", "ip_adapter_trainer.py"],
                "gate": "safe_graph_outside",
                "reason": "random tensors are sampled before the stable UNet/block call",
            },
            {
                "feature": "pyramid/multires/perlin/noise-offset variants",
                "files": ["trainer.py", "training_loop.py"],
                "gate": "keep_eager",
                "reason": "levels, scales or generated tensors may vary per step",
            },
            {
                "feature": "bucket, caption-length bucket and staged resolution",
                "files": ["dataset_loader.py", "visual_token_bucket.py", "trainer.py"],
                "gate": "compile_blocker_without_static_shape_anchor",
                "reason": "batch image/text token shapes can vary across batches or stages",
            },
            {
                "feature": "caption dropout/shuffle and image augmentation",
                "files": ["dataset_loader.py"],
                "gate": "safe_if_fixed_output_shape_else_keep_eager",
                "reason": "data randomness is acceptable only before fixed-shape model inputs",
            },
            {
                "feature": "ControlNet/IP-Adapter routes",
                "files": ["controlnet_trainer.py", "ip_adapter_trainer.py", "entry_train.py"],
                "gate": "needs_separate_ab",
                "reason": "extra conditioning paths can change tensor shapes and module calls",
            },
        ],
        "release_gate": (
            "compile evidence is invalid if a candidate enables dynamic graph-changing randomization "
            "inside compiled blocks or lacks a stable eager anchor"
        ),
    }


__all__ = ["sdxl_compile_randomization_contract"]
