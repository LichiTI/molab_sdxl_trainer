# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Opt-in heavy smoke for real Newbie cache generation.

This script loads only the cache-builder components from ``models/newbie``
(VAE, Gemma, JinaCLIP), copies a small local image/caption subset into
``H:/tmp``, writes schema-v1 ``*_newbie.npz`` caches, and validates them with
production dimensions.
It deliberately does not load the 3B transformer or unlock training guards.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.newbie_cache_builder import build_newbie_cache
from core.lulynx_trainer.newbie_cached_dataset import NewbieCacheSchema, NewbieCachedDataset
from core.lulynx_trainer.newbie_loader import (
    _load_clip_native,
    _load_gemma_native,
    _load_vae_native,
)


def _image_files(data_dir: Path, *, limit: int) -> list[Path]:
    images: list[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        images.extend(sorted(data_dir.glob(suffix)))
    if not images:
        raise FileNotFoundError(f"No image found in {data_dir}")
    return images[: max(int(limit), 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="H:/lulynx-trainer/models/newbie")
    parser.add_argument("--source-data", default="H:/lulynx-trainer/sucai/6_lulu")
    parser.add_argument("--work-dir", default="H:/tmp/lulynx_newbie_real_cache_smoke")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Validate VAE+Gemma cache generation when the local JinaCLIP code snapshot is incomplete.",
    )
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    source_data = Path(args.source_data)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    images = _image_files(source_data, limit=max(int(args.sample_count), 1))
    for image in images:
        target_image = work_dir / image.name
        shutil.copy2(image, target_image)
        caption = image.with_suffix(".txt")
        if caption.is_file():
            shutil.copy2(caption, target_image.with_suffix(".txt"))

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    vae, vae_notes = _load_vae_native(str(model_dir / "vae"), dtype)
    gemma, gemma_tokenizer, gemma_notes = _load_gemma_native(
        str(model_dir / "text_encoder"),
        dtype,
        max_token_length=512,
        trust_remote_code=args.trust_remote_code,
    )
    if args.skip_clip:
        clip, clip_tokenizer, clip_notes = None, None, ["JinaCLIP skipped by --skip-clip"]
    else:
        clip, clip_tokenizer, clip_notes = _load_clip_native(
            str(model_dir / "clip_model"),
            dtype,
            max_token_length=2048,
            trust_remote_code=args.trust_remote_code,
        )
    notes = vae_notes + gemma_notes + clip_notes
    if vae is None or gemma is None or gemma_tokenizer is None or (not args.skip_clip and (clip is None or clip_tokenizer is None)):
        raise RuntimeError("Failed to load Newbie cache components:\n  " + "\n  ".join(notes))

    loaded = SimpleNamespace(
        vae=vae,
        text_encoder_1=gemma,
        tokenizer_1=gemma_tokenizer,
        text_encoder_2=clip,
        tokenizer_2=clip_tokenizer,
    )
    result = build_newbie_cache(
        loaded_model=loaded,
        data_dir=work_dir,
        device=args.device,
        dtype=dtype,
        resolution=(args.resolution, args.resolution),
        force=True,
        log=print,
    )
    expected_written = len(images)
    if result.errors or result.written != expected_written:
        raise RuntimeError(f"Unexpected real Newbie cache build result: {result}")

    dataset = NewbieCachedDataset(
        work_dir,
        schema=NewbieCacheSchema(
            expected_latent_channels=16,
            expected_hidden_size=2560,
            expected_pooled_size=0 if args.skip_clip else 1024,
            require_pooled_prompt_embeds=not args.skip_clip,
        ),
    )
    if len(dataset) != expected_written:
        raise RuntimeError(f"Expected {expected_written} Newbie cache samples, got {len(dataset)}")
    item = dataset[0]
    print(
        "Real Newbie cache builder smoke passed: "
        f"samples={len(dataset)}, "
        f"latents={tuple(item['latents'].shape)}, "
        f"hidden={tuple(item['encoder_hidden_states'].shape)}, "
        f"pooled={None if item.get('pooled_prompt_embeds') is None else tuple(item['pooled_prompt_embeds'].shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
