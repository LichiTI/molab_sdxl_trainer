"""Smoke tests for the lightweight S-tier training advisor."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _load_advisor_module():
    module_path = Path(__file__).resolve().with_name("training_advisor.py")
    spec = importlib.util.spec_from_file_location("training_advisor_smoke_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_png(width: int = 8, height: int = 8) -> bytes:
    import zlib

    def chunk(name: bytes, payload: bytes) -> bytes:
        import struct

        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00") + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _make_dataset(root: Path) -> None:
    png = _make_png()
    for idx in range(12):
        image = root / f"sample_{idx}.png"
        image.write_bytes(png if idx != 11 else b"not-a-real-image")
        if idx < 10:
            image.with_suffix(".txt").write_text("tag, test", encoding="utf-8")
    (root / "orphan.txt").write_text("orphan tag", encoding="utf-8")


def main() -> int:
    advisor = _load_advisor_module()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "dataset"
        out_dir = Path(tmp) / "out"
        data_dir.mkdir()
        _make_dataset(data_dir)
        cfg = SimpleNamespace(
            model_arch="sdxl",
            train_data_dir=str(data_dir),
            output_dir=str(out_dir),
            resolution=2048,
            batch_size=4,
            network_dim=64,
            mixed_precision="bf16",
            gradient_checkpointing=False,
            module_offload_enabled=False,
            so_enable_nan_detection=True,
            so_enable_loss_spike_detection=True,
            so_enable_lr_deadlock_detection=True,
            so_enable_auto_recovery=True,
            so_enable_bad_sample_culling=False,
            train_text_encoder=True,
            cache_text_encoder_outputs=False,
            semantic_tuner_enabled=False,
            learning_rate=1e-4,
            optimizer="adamw",
            smart_rank_enabled=False,
            auto_controller_enabled=False,
            ac_enabled=False,
            ema_use_ema=False,
            masked_loss=False,
            alpha_mask=False,
            bw_enable=False,
            vram_swap_to_ram=False,
            enable_bucket=True,
            min_bucket_reso=256,
            max_bucket_reso=2048,
            bucket_reso_steps=64,
            bucket_selection_mode="aspect",
            shuffle_caption=True,
            keep_tokens=1,
            caption_variants_enabled=False,
        )

        report = advisor.build_training_advisor_report(cfg, available_vram_gb=8.0).to_dict()
        assert report["summary"]["status"] in {"warning", "error"}
        assert report["vram"]["safety"] in {"tight", "danger"}
        assert report["vram"]["recommended_config_patch"]["gradient_checkpointing"] is True
        assert report["vram"]["recommended_config_patch"]["low_vram_profile"] in {"low_12g", "very_low_8g"}
        assert "module_offload_profile" not in report["vram"]["recommended_config_patch"]
        assert report["vram"]["low_vram_profile_advice"]["should_patch"] is True
        assert any(item["code"] == "sdxl_lora_low_vram_profile_recommended" for item in report["findings"])
        assert report["vram"]["runtime_probe"]["available"] is False
        assert report["dataset"]["purifier_ready"] is True
        assert report["dataset"]["missing_caption_count"] == 2
        assert report["dataset"]["orphan_caption_count"] == 1
        assert report["dataset"]["duplicate_image_count"] >= 1
        assert report["dataset"]["unreadable_image_count"] == 1
        assert any(issue["type"] == "orphan_caption" for issue in report["dataset"]["sample_issues"])
        assert report["lr_finder"]["available"] is True
        assert report["text_encoder"]["expected_active"] is True
        assert report["a_tier"]["status"] == "advisory"
        assert report["b_tier"]["status"] == "mixed"
        assert report["a_tier"]["modules"]["smart_rank"]["mode"] == "advisor_only"
        assert report["summary"]["a_tier_modules"]["memory_vortex_fusion"] is True
        assert report["summary"]["b_tier_modules"]["pcgrad"] is True
        assert report["b_tier"]["modules"]["pcgrad"]["train_chain_wired"] is True
        assert report["b_tier"]["modules"]["pcgrad"]["status"] == "available_manual"
        assert any(item["code"] == "dataset_missing_captions" for item in report["findings"])

        cfg.auto_controller_enabled = True
        cfg.so_enable_nan_detection = False
        cfg.masked_loss = True
        cfg.min_bucket_reso = 2048
        cfg.max_bucket_reso = 512
        risky_report = advisor.build_training_advisor_report(cfg, available_vram_gb=8.0).to_dict()
        risky_codes = {item["code"] for item in risky_report["findings"]}
        assert "auto_controller_without_nan_guard" in risky_codes
        assert "masked_loss_needs_masks" in risky_codes
        assert "bucket_resolution_invalid" in risky_codes

        cfg.so_enable_nan_detection = True
        cfg.auto_controller_enabled = False
        cfg.masked_loss = False
        cfg.min_bucket_reso = 256
        cfg.max_bucket_reso = 2048
        cfg.model_arch = "anima"
        cfg.train_text_encoder = False
        cfg.so_enable_bad_sample_culling = True
        cfg.so_bad_sample_mode = "move"
        cfg.pcgrad_enabled = True
        cfg.lulynx_geometric_lock = True
        cfg.lulynx_ghost_replay = True
        cfg.hutchinson_auto_freeze = True
        anima_report = advisor.build_training_advisor_report(cfg, available_vram_gb=8.0).to_dict()
        assert anima_report["vram"]["runtime_probe"]["available"] is True
        assert "anima_runtime_vram_probe.py" in anima_report["vram"]["runtime_probe"]["command_cuda"]
        assert anima_report["safeguard"]["bad_sample_mode"] == "move"
        assert any(item["code"] == "safeguard_bad_sample_move" for item in anima_report["findings"])
        anima_codes = {item["code"] for item in anima_report["findings"]}
        assert "pcgrad_enabled_experimental" in anima_codes
        assert "ghost_replay_missing_fingerprint" in anima_codes
        assert "manifold_constraint_enabled_experimental" in anima_codes
        assert "hutchinson_enabled_experimental" in anima_codes

        cfg.resolution = 1024
        cfg.batch_size = 1
        cfg.gradient_checkpointing = True
        cfg.anima_block_residency = "resident"
        cfg.anima_block_checkpointing = False
        cfg.anima_fixed_visual_tokens = 0
        pressure_report = advisor.build_training_advisor_report(cfg, available_vram_gb=24.0).to_dict()
        dit_runtime = pressure_report["vram"]["dit_runtime"]
        assert dit_runtime["full_token_resident_pressure"] is True
        assert dit_runtime["recommendation"] == "streaming_offload"
        assert dit_runtime["checkpoint_missing"] is True
        assert pressure_report["vram"]["recommended_config_patch"]["anima_block_residency"] == "streaming_offload"
        assert pressure_report["vram"]["recommended_config_patch"]["anima_block_checkpointing"] is True
        pressure_codes = {item["code"] for item in pressure_report["findings"]}
        assert "dit_streaming_offload_recommended" in pressure_codes
        assert "dit_block_checkpointing_recommended" in pressure_codes

        saved = advisor.write_training_advisor_report(cfg, out_dir)
        loaded = json.loads((out_dir / "training_advisor_report.json").read_text(encoding="utf-8"))
        assert saved["summary"] == loaded["summary"]

    print("training_advisor_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

