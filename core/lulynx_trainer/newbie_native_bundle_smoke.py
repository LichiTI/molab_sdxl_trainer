# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Real native-bundle smoke for Newbie component-path loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, UnifiedTrainingConfig
from core.lulynx_trainer.newbie_loader import load_newbie_from_config
from core.lulynx_trainer.newbie_smoke import run_loaded_newbie_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="H:/lulynx-trainer/models/newbie")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.NEWBIE,
        pretrained_model_name_or_path=str(model_dir),
        newbie_diffusers_path=str(model_dir),
        newbie_transformer_path=str(model_dir / "transformer"),
        newbie_gemma_model_path=str(model_dir / "text_encoder"),
        newbie_clip_model_path=str(model_dir / "clip_model"),
        newbie_vae_path=str(model_dir / "vae"),
        newbie_gemma_max_token_length=512,
        newbie_clip_max_token_length=2048,
        trust_remote_code=True,
        newbie_target_modules="attention.qkv\nattention.out",
    )

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = load_newbie_from_config(cfg, device=args.device, dtype=dtype)
    smoke = run_loaded_newbie_smoke(model)
    if not smoke.passed:
        raise RuntimeError(f"Newbie native bundle smoke failed: {smoke.reason}")

    if not getattr(model, "newbie_native_bundle_loaded", False):
        raise AssertionError("Newbie native bundle flag was not set")
    if model.tokenizer_1 is None or model.text_encoder_1 is None:
        raise AssertionError("Gemma component did not load")
    if model.tokenizer_2 is None or model.text_encoder_2 is None:
        raise AssertionError("CLIP component did not load")
    if model.vae is None:
        raise AssertionError("VAE component did not load")

    print(
        "Newbie native bundle smoke passed: "
        f"targets={list(smoke.gradient_targets)}, "
        f"latent={smoke.latent_shape}, output={smoke.output_shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
