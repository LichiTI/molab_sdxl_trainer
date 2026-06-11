# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Heavy smoke for real Newbie native bundle cache-first training.

This is the honest trainer-level closure proof for the packaged Newbie route:

1. copy one local image/caption into a temp train directory
2. run the product ``LulynxTrainer.start()`` path
3. build one ``*_newbie.npz`` cache through the production cache builder
4. reload the training transformer only
5. execute one real optimizer step
6. save a real adapter artifact

The smoke stays intentionally conservative: one sample, one step, rank-1 LoRA,
gradient checkpointing enabled, and preview disabled.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import MixedPrecision, ModelArch, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer import newbie_cached_dataset
from core.lulynx_trainer.trainer import LulynxTrainer


def _first_image(data_dir: Path) -> Path:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        found = sorted(data_dir.glob(suffix))
        if found:
            return found[0]
    raise FileNotFoundError(f"No image found in {data_dir}")


def _prepare_one_sample_dir(source_dir: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    image = _first_image(source_dir)
    target_image = target_dir / image.name
    shutil.copy2(image, target_image)
    caption = image.with_suffix(".txt")
    if caption.is_file():
        shutil.copy2(caption, target_image.with_suffix(".txt"))
    return target_image


def _resolve_runtime() -> tuple[str, torch.dtype, MixedPrecision]:
    """Resolve the runtime device/dtype for the actual trainer step.

    The native bundle is still loaded conservatively on CPU/fp32 first so the
    smoke can separate "bundle load" cost from "one real trainer step" cost.
    """
    detected_device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_device = str(
        os.environ.get("LULYNX_NEWBIE_REAL_SMOKE_DEVICE", detected_device) or detected_device
    ).strip().lower()
    if runtime_device == "cuda" and not torch.cuda.is_available():
        runtime_device = "cpu"
    if runtime_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported LULYNX_NEWBIE_REAL_SMOKE_DEVICE={runtime_device!r}")

    if runtime_device == "cpu":
        return "cpu", torch.float32, MixedPrecision.NO

    dtype_name = str(
        os.environ.get(
            "LULYNX_NEWBIE_REAL_SMOKE_DTYPE",
            "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        )
        or ""
    ).strip().lower()
    if dtype_name in {"", "bf16", "bfloat16"}:
        return "cuda", torch.bfloat16, MixedPrecision.BF16
    if dtype_name in {"fp16", "float16", "half"}:
        return "cuda", torch.float16, MixedPrecision.FP16
    if dtype_name in {"fp32", "float32", "no"}:
        return "cuda", torch.float32, MixedPrecision.NO
    raise ValueError(f"Unsupported LULYNX_NEWBIE_REAL_SMOKE_DTYPE={dtype_name!r}")


def _phase(label: str, *, start_time: float, last_time: float) -> float:
    now = time.perf_counter()
    print(
        f"[newbie-real-smoke] phase={label} dt={now - last_time:.2f}s total={now - start_time:.2f}s",
        flush=True,
    )
    return now


def main() -> int:
    start_time = time.perf_counter()
    last_phase = start_time
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = repo_root / "models" / "newbie"
    source_data = repo_root / "sucai" / "6_lulu"
    runtime_device, runtime_dtype, mixed_precision = _resolve_runtime()
    tmp_root = Path("H:/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="lulynx_newbie_real_trainer_smoke_", dir=str(tmp_root)))
    train_dir = work_dir / "train"
    output_dir = work_dir / "output"

    train_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Newbie model dir not found: {model_dir}")
    if not source_data.exists():
        raise FileNotFoundError(f"Newbie source data not found: {source_data}")

    copied_image = _prepare_one_sample_dir(source_data, train_dir)

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.NEWBIE,
        pretrained_model_name_or_path=str(model_dir),
        train_data_dir=str(train_dir),
        output_dir=str(output_dir),
        output_name="newbie_real_trainer_smoke",
        resolution=64,
        mixed_precision=mixed_precision,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=1,
        max_train_steps=1,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-5,
        save_every_n_epochs=1,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        trust_remote_code=True,
        use_cache=True,
        newbie_run_native_smoke=False,
        newbie_safe_fallback=(runtime_device == "cuda"),
        newbie_transformer_path=str(model_dir / "transformer"),
        newbie_gemma_model_path=str(model_dir / "text_encoder"),
        newbie_clip_model_path=str(model_dir / "clip_model"),
        newbie_vae_path=str(model_dir / "vae"),
        newbie_gemma_max_token_length=64,
        newbie_clip_max_token_length=128,
        newbie_target_modules="layers.0.attention.qkv\nlayers.0.attention.out",
    )

    trainer = LulynxTrainer(cfg)
    trainer.device = runtime_device
    trainer.dtype = runtime_dtype
    logs: list[str] = []
    trainer.set_callbacks(on_log=logs.append)

    print(
        "[newbie-real-smoke] "
        f"runtime_device={runtime_device} runtime_dtype={runtime_dtype} "
        f"work_dir={work_dir}",
        flush=True,
    )
    last_phase = _phase("trainer-start", start_time=start_time, last_time=last_phase)

    captured: dict[str, object] = {}
    clip_calls: list[float] = []
    original_create = newbie_cached_dataset.create_newbie_cached_dataloader
    original_clip = torch.nn.utils.clip_grad_norm_

    def _capture_cached_dataloader(dataset, *, batch_size, shuffle, num_workers=0):
        captured["num_workers"] = num_workers
        captured["dataset_len"] = len(dataset)
        return original_create(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _capture_clip(params, max_norm, *args, **kwargs):
        clip_calls.append(float(max_norm))
        return original_clip(params, max_norm, *args, **kwargs)

    with patch.object(
        newbie_cached_dataset,
        "create_newbie_cached_dataloader",
        side_effect=_capture_cached_dataloader,
    ):
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=_capture_clip):
            ok = trainer.start()
            last_phase = _phase("trainer-finished", start_time=start_time, last_time=last_phase)
    if not ok:
        raise RuntimeError("Real Newbie trainer smoke failed")

    cache_files = sorted(train_dir.glob("*_newbie.npz"))
    if len(cache_files) != 1:
        raise AssertionError(f"Expected exactly one Newbie cache file, got {len(cache_files)} in {train_dir}")

    expected = output_dir / "newbie_real_trainer_smoke.safetensors"
    if not expected.exists():
        raise FileNotFoundError(f"Expected adapter was not saved: {expected}")
    if captured.get("num_workers") != 0:
        raise AssertionError(f"Unexpected cached dataloader worker count: {captured}")
    if captured.get("dataset_len") != 1:
        raise AssertionError(f"Expected one cached sample, got {captured}")
    if trainer.training_loop is None or trainer.training_loop.global_step != 1:
        raise AssertionError(
            f"Expected real Newbie trainer smoke to complete 1 step, got {getattr(trainer.training_loop, 'global_step', None)}"
        )
    if clip_calls != [1.0]:
        raise AssertionError(f"Expected one grad clipping call at max_norm=1.0, got {clip_calls}")

    log_text = "\n".join(logs)
    expected_fragments = (
        "Newbie transformer smoke passed",
        "Newbie cache written:",
        "Newbie cache-first dataset: 1 cached samples",
        "Newbie cache-first training: released cache-builder components after cache generation:",
    )
    for fragment in expected_fragments:
        if fragment not in log_text:
            raise AssertionError(f"Expected log fragment missing: {fragment}")

    print(
        "Real Newbie trainer smoke passed: "
        f"runtime_device={runtime_device}, image={copied_image.name}, "
        f"cache={cache_files[0].name}, saved={expected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
