# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie logging runtime fields on the native cached path."""

from __future__ import annotations

import os
import shutil
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie logging smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.newbie_trainer_cached_smoke import (
    _TinyNewbieDenoiser,
    _write_cache,
)
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.model_loader import LoadedModel
from core.lulynx_trainer.trainer import LulynxTrainer


def _build_trainer(*, log_with: str, logging_dir: Path, log_prefix: str, wandb_api_key: str) -> LulynxTrainer:
    cache_dir = Path("H:/tmp/lulynx_newbie_logging_cache")
    output_dir = Path("H:/tmp/lulynx_newbie_logging_output")
    for path in (cache_dir, output_dir):
        if path.exists():
            shutil.rmtree(path)
    _write_cache(cache_dir)

    cfg = UnifiedTrainingConfig(
        model_type="newbie",
        pretrained_model_name_or_path="H:/tmp/newbie-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name="newbie_logging_smoke",
        mixed_precision="no",
        optimizer_type="AdamW",
        lr_scheduler="constant",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=1,
        max_train_steps=1,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        use_cache=True,
        newbie_target_modules="attention.qkv\nattention.out\nfeed_forward.w1\nfeed_forward.w2\nfeed_forward.w3",
        log_with=log_with,
        logging_dir=str(logging_dir),
        log_prefix=log_prefix,
        wandb_api_key=wandb_api_key,
        wandb_run_name="newbie-logging-smoke",
    )

    model = LoadedModel(
        unet=_TinyNewbieDenoiser(),
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        model_arch="newbie",
    )
    model.newbie_scaffold_mode = False
    model.newbie_native_conditioning_ready = True
    model.newbie_transport_ready = True
    model.newbie_forward_smoke_passed = True
    model.newbie_gradient_smoke_passed = True

    injector = LoRAInjector(rank=1, alpha=1, model_arch="newbie")
    injected = injector._inject_model(model.unet, cfg.newbie_target_modules.splitlines(), prefix="unet")
    if not injected:
        raise RuntimeError("Failed to inject LoRA targets for Newbie logging smoke")

    trainer = LulynxTrainer(cfg)
    trainer.device = "cpu"
    trainer.dtype = __import__("torch").float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._adapter_cpu_residency = None
    trainer._ensure_native_family_training_ready()
    return trainer


def main() -> int:
    logging_root = Path("H:/tmp/lulynx_newbie_logging_tb")
    if logging_root.exists():
        shutil.rmtree(logging_root)
    logging_root.mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer(
        log_with="tensorboard",
        logging_dir=logging_root,
        log_prefix="newbie-prefix-",
        wandb_api_key="",
    )
    ok = trainer.start()
    if not ok:
        raise RuntimeError("Newbie logging smoke failed on tensorboard path")

    run_dirs = [path for path in logging_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise AssertionError(f"Expected exactly one TensorBoard run dir, got {run_dirs}")
    run_dir = run_dirs[0]
    if not run_dir.name.startswith("newbie-prefix-"):
        raise AssertionError(f"log_prefix did not affect run dir name: {run_dir.name}")
    if not any(path.name.startswith("events.out.tfevents.") for path in run_dir.iterdir()):
        raise AssertionError(f"No TensorBoard event file created in {run_dir}")

    os.environ.pop("WANDB_API_KEY", None)
    os.environ.pop("WANDB_NAME", None)
    trainer = _build_trainer(
        log_with="wandb",
        logging_dir=logging_root,
        log_prefix="ignored-",
        wandb_api_key="smoke-api-key",
    )
    trainer._initialize_logging_runtime()
    if os.environ.get("WANDB_API_KEY") != "smoke-api-key":
        raise AssertionError("wandb_api_key was not propagated to WANDB_API_KEY")
    if os.environ.get("WANDB_NAME") != "newbie-logging-smoke":
        raise AssertionError("wandb_run_name was not propagated to WANDB_NAME")
    trainer._finalize_logging_runtime()

    print(f"Newbie logging smoke passed: tensorboard_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
