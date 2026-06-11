"""Quick matrix smoke for SDXL/LoRA low-VRAM profile advice and resolver.

This is intentionally model-free: it exercises the runtime config resolver and
Advisor patch suggestions without loading torch/model weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

from sdxl_lora_low_vram_profile import apply_sdxl_lora_low_vram_profile

try:
    from core.lulynx_trainer import training_advisor
except Exception:
    import training_advisor  # type: ignore


PROFILES = ("off", "standard_16g", "low_12g", "very_low_8g")
TRACKED_KEYS = (
    "sdxl_low_vram_optimization",
    "cache_latents",
    "cache_latents_to_disk",
    "cache_text_encoder_outputs",
    "gradient_checkpointing",
    "vae_slicing",
    "attention_slicing",
    "pytorch_cuda_expandable_segments",
    "model_to_condition_enabled",
    "te_vae_offload_strategy",
    "cuda_cache_release_strategy",
    "enable_mixed_resolution_training",
    "staged_resolution_ratio_768",
    "staged_resolution_ratio_1024",
    "swap_granularity",
    "swap_ratio",
    "block_merge_size",
    "block_swap_strategy",
    "checkpoint_policy",
    "cpu_offload_checkpointing_mode",
    "train_batch_size",
)


def _base_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "model_arch": "sdxl",
        "model_type": "sdxl",
        "training_type": "lora",
        "low_vram_profile": "off",
        "sdxl_low_vram_optimization": False,
        "resolution": 1024,
        "batch_size": 2,
        "train_batch_size": 2,
        "network_dim": 64,
        "mixed_precision": "bf16",
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
        "staged_resolution_ratio_512": 0,
        "staged_resolution_ratio_768": 0,
        "staged_resolution_ratio_1024": 0,
        "swap_granularity": "off",
        "swap_ratio": 0.0,
        "swap_count": 0,
        "blocks_to_swap": 0,
        "block_merge_size": 2,
        "block_swap_strategy": "auto",
        "module_offload_enabled": False,
        "module_offload_profile_enabled": False,
        "module_offload_profile": "custom",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row_for_profile(profile: str, *, available_vram_gb: float) -> dict[str, Any]:
    cfg = _base_config(low_vram_profile=profile)
    before = copy.deepcopy(vars(cfg))
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    after = vars(cfg)
    changed_keys = [key for key in TRACKED_KEYS if before.get(key) != after.get(key)]
    advisor_cfg = _base_config(low_vram_profile=profile)
    advisor_report = training_advisor.estimate_training_vram(advisor_cfg, available_vram_gb=available_vram_gb)
    patch = dict(advisor_report.get("recommended_config_patch") or {})
    return {
        "profile": profile,
        "enabled": decision.enabled,
        "effective": decision.effective,
        "changed_keys": changed_keys,
        "change_count": len(changed_keys),
        "cache_latents": cfg.cache_latents,
        "cache_text_encoder_outputs": cfg.cache_text_encoder_outputs,
        "te_vae_offload_strategy": cfg.te_vae_offload_strategy,
        "checkpoint_policy": cfg.checkpoint_policy,
        "cpu_offload_checkpointing_mode": cfg.cpu_offload_checkpointing_mode,
        "swap_granularity": cfg.swap_granularity,
        "swap_ratio": cfg.swap_ratio,
        "train_batch_size": cfg.train_batch_size,
        "enable_mixed_resolution_training": cfg.enable_mixed_resolution_training,
        "warnings": list(decision.warnings),
        "skipped": list(decision.skipped),
        "advisor_safety": advisor_report.get("safety"),
        "advisor_patch_low_vram_profile": patch.get("low_vram_profile"),
        "advisor_patch_keys": sorted(patch.keys()),
    }


def _manual_swap_guard_row() -> dict[str, Any]:
    cfg = _base_config(low_vram_profile="low_12g", swap_granularity="block", swap_count=1, swap_ratio=0.15)
    decision = apply_sdxl_lora_low_vram_profile(cfg, model_arch="sdxl")
    return {
        "profile": "low_12g_manual_swap",
        "enabled": decision.enabled,
        "swap_granularity": cfg.swap_granularity,
        "swap_count": cfg.swap_count,
        "swap_ratio": cfg.swap_ratio,
        "skipped": list(decision.skipped),
    }


def build_matrix(*, available_vram_gb: float) -> list[dict[str, Any]]:
    rows = [_row_for_profile(profile, available_vram_gb=available_vram_gb) for profile in PROFILES]
    rows.append(_manual_swap_guard_row())
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "profile",
        "effective",
        "changes",
        "te/vae",
        "checkpoint",
        "swap",
        "batch",
        "staged",
        "advisor",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(item) for item in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    str(row.get("profile", "")),
                    str(row.get("effective", "")),
                    str(row.get("change_count", "")),
                    str(row.get("te_vae_offload_strategy", "")),
                    str(row.get("checkpoint_policy", "")),
                    f"{row.get('swap_granularity', '')}:{row.get('swap_ratio', '')}",
                    str(row.get("train_batch_size", "")),
                    str(row.get("enable_mixed_resolution_training", "")),
                    str(row.get("advisor_patch_low_vram_profile", "")),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--available-vram-gb", type=float, default=8.0)
    args = parser.parse_args()

    rows = build_matrix(available_vram_gb=args.available_vram_gb)
    by_profile = {row["profile"]: row for row in rows}
    assert by_profile["off"]["enabled"] is False
    assert by_profile["standard_16g"]["enabled"] is True
    assert by_profile["low_12g"]["change_count"] > by_profile["standard_16g"]["change_count"]
    assert by_profile["very_low_8g"]["checkpoint_policy"] == "offloaded"
    assert by_profile["very_low_8g"]["train_batch_size"] == 1
    assert by_profile["low_12g_manual_swap"]["swap_granularity"] == "block"
    assert any(item.get("key") == "swap_granularity" for item in by_profile["low_12g_manual_swap"]["skipped"])

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_table(rows)
        print("sdxl_lora_low_vram_matrix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
