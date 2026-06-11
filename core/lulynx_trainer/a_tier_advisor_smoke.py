"""Smoke tests for A-tier fusion advisor reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_advisor_module():
    module_path = Path(__file__).resolve().with_name("training_advisor.py")
    spec = importlib.util.spec_from_file_location("a_tier_advisor_smoke_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    advisor = _load_advisor_module()
    cfg = SimpleNamespace(
        model_arch="anima",
        train_data_dir="",
        network_dim=96,
        smart_rank_enabled=False,
        auto_controller_enabled=True,
        ac_enabled=False,
        auto_freeze_te=False,
        ac_enable_auto_te_freeze=True,
        smart_early_stop=False,
        ac_enable_smart_early_stopping=True,
        smart_lr_decay=False,
        ac_enable_smart_lr_decay=True,
        ema_use_ema=True,
        ema_decay=0.9995,
        ema_update_after_step=50,
        ema_update_every=2,
        masked_loss=True,
        alpha_mask=False,
        strict_masked_loss=True,
        bw_enable=True,
        bw_preset="mid_only",
        module_offload_enabled=True,
        module_offload_profile="custom",
        module_offload_ratio=0,
        module_offload_prefetch_enabled=True,
        module_offload_min_param_mb=1.5,
        module_offload_include_patterns="proj*",
        module_offload_exclude_patterns="te*",
        vram_swap_to_ram=True,
        enable_bucket=True,
        min_bucket_reso=2048,
        max_bucket_reso=512,
        bucket_reso_steps=64,
        bucket_selection_mode="aspect",
        shuffle_caption=True,
        keep_tokens=2,
        caption_variants_enabled=True,
        so_enable_nan_detection=False,
        so_enable_loss_spike_detection=True,
    )

    a_tier = advisor.inspect_a_tier_features(cfg)
    assert a_tier["status"] == "advisory"
    assert a_tier["family"] == "anima"
    vortex = a_tier["modules"]["memory_vortex_fusion"]
    assert vortex["status"] == "covered_by_module_offload"
    assert "overlapping_cpu_offload_strategies" in vortex["risk_flags"]
    assert "experimental_prefetch" in vortex["risk_flags"]
    assert "runtime H2D materialize statistics" in vortex["absorbed_ideas"]
    assert vortex["include_patterns"] == "proj*"
    assert a_tier["modules"]["smart_rank"]["mode"] == "advisor_only"
    assert a_tier["modules"]["smart_rank"]["advice"]["current_rank"] == 96
    assert a_tier["modules"]["smart_rank"]["advice"]["suggested_rank"] < 96
    assert a_tier["recommended_config_patch"]["smart_rank_enabled"] is False
    assert a_tier["modules"]["auto_controller"]["enabled"] is True
    assert a_tier["modules"]["auto_controller"]["auto_te_freeze"] is True
    assert a_tier["modules"]["ema"]["enabled"] is True
    assert a_tier["modules"]["masked_loss"]["strict"] is True
    assert a_tier["modules"]["block_weight"]["enabled"] is True
    assert a_tier["modules"]["smart_caption"]["keep_tokens"] == 2
    assert a_tier["modules"]["dataset_bucket"]["min_reso"] == 2048
    assert "merge_with_existing_systems" in a_tier["migration_policy"]
    assert any("module_offload and vram_swap_to_ram overlap" in note for note in a_tier["notes"])
    assert any("Bucket min resolution" in note for note in a_tier["notes"])

    report = advisor.build_training_advisor_report(cfg, available_vram_gb=24.0).to_dict()
    codes = {item["code"] for item in report["findings"]}
    assert "auto_controller_without_nan_guard" in codes
    assert "masked_loss_needs_masks" in codes
    assert "bucket_resolution_invalid" in codes
    assert report["summary"]["a_tier_modules"]["smart_rank"] is True

    print("a_tier_advisor_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())