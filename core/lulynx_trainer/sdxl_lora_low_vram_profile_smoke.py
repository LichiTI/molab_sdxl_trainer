"""Smoke tests for SDXL/LoRA low-VRAM runtime profile resolver."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdxl_lora_low_vram_profile import apply_sdxl_lora_low_vram_profile, normalize_low_vram_profile


def _cfg(**overrides):
    defaults = {
        "model_type": "sdxl",
        "training_type": "lora",
        "low_vram_profile": "off",
        "sdxl_low_vram_optimization": False,
        "cache_latents": False,
        "cache_latents_to_disk": False,
        "cache_text_encoder_outputs": False,
        "train_text_encoder": False,
        "network_train_text_encoder_only": False,
        "shuffle_caption": False,
        "shuffle_caption_tags_only": False,
        "caption_dropout_rate": 0.0,
        "tag_dropout_rate": 0.0,
        "caption_tag_dropout_targets": "",
        "gradient_checkpointing": False,
        "checkpoint_policy": "auto",
        "cpu_offload_checkpointing_mode": "standard",
        "vae_slicing": False,
        "attention_slicing": False,
        "pytorch_cuda_expandable_segments": False,
        "model_to_condition_enabled": False,
        "te_vae_offload_strategy": "resident",
        "cuda_cache_release_strategy": "oom_only",
        "enable_mixed_resolution_training": False,
        "resolution": 1024,
        "staged_resolution_ratio_512": 0,
        "staged_resolution_ratio_768": 0,
        "staged_resolution_ratio_1024": 0,
        "staged_resolution_ratio_1536": 0,
        "staged_resolution_ratio_2048": 0,
        "swap_granularity": "off",
        "swap_ratio": 0.0,
        "swap_count": 0,
        "blocks_to_swap": 0,
        "block_merge_size": 2,
        "block_swap_strategy": "auto",
        "train_batch_size": 2,
        "module_offload_enabled": False,
        "weight_compression_enabled": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_normalize_aliases():
    assert normalize_low_vram_profile("16g") == "standard_16g"
    assert normalize_low_vram_profile("12GB") == "low_12g"
    assert normalize_low_vram_profile("very-low") == "very_low_8g"
    assert normalize_low_vram_profile("???") == "off"
    print("PASS: normalize aliases")


def test_standard_16g_stays_conservative():
    cfg = _cfg(low_vram_profile="standard_16g")
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    assert decision.enabled
    assert cfg.cache_latents is True
    assert cfg.gradient_checkpointing is True
    assert cfg.cache_text_encoder_outputs is True
    assert cfg.te_vae_offload_strategy == "phase"
    assert cfg.swap_granularity == "off"
    assert cfg.checkpoint_policy == "auto"
    print("PASS: standard_16g conservative profile")


def test_low_12g_adds_staged_and_merged_swap():
    cfg = _cfg(low_vram_profile="low_12g", train_batch_size=1)
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    assert decision.enabled
    assert cfg.te_vae_offload_strategy == "aggressive"
    assert cfg.cache_latents_to_disk is True
    assert cfg.enable_mixed_resolution_training is True
    assert cfg.staged_resolution_ratio_768 == 35
    assert cfg.staged_resolution_ratio_1024 == 65
    assert cfg.swap_granularity == "merged_block"
    assert cfg.swap_ratio == 0.25
    print("PASS: low_12g staged + merged swap profile")


def test_very_low_8g_uses_offloaded_checkpoint_and_batch_guard():
    cfg = _cfg(low_vram_profile="very_low_8g", train_batch_size=4)
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    assert decision.enabled
    assert cfg.checkpoint_policy == "offloaded"
    assert cfg.cpu_offload_checkpointing_mode == "pinned_async"
    assert cfg.train_batch_size == 1
    assert cfg.swap_ratio == 0.4
    assert any("train_batch_size=1" in warning for warning in decision.warnings)
    print("PASS: very_low_8g offloaded checkpoint profile")


def test_manual_swap_is_preserved():
    cfg = _cfg(low_vram_profile="low_12g", swap_granularity="block", swap_count=1)
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    assert decision.enabled
    assert cfg.swap_granularity == "block"
    assert cfg.swap_count == 1
    assert any(item.get("key") == "swap_granularity" for item in decision.skipped)
    print("PASS: manual swap preserved")


def test_text_cache_guard_respects_live_captioning():
    cfg = _cfg(low_vram_profile="standard_16g", shuffle_caption=True)
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    assert decision.enabled
    assert cfg.cache_text_encoder_outputs is False
    assert any(item.get("key") == "cache_text_encoder_outputs" for item in decision.skipped)
    print("PASS: text cache guard")


if __name__ == "__main__":
    test_normalize_aliases()
    test_standard_16g_stays_conservative()
    test_low_12g_adds_staged_and_merged_swap()
    test_very_low_8g_uses_offloaded_checkpoint_and_batch_guard()
    test_manual_swap_is_preserved()
    test_text_cache_guard_respects_live_captioning()
    print("PASS: sdxl_lora_low_vram_profile smoke")
