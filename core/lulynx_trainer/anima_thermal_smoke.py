"""Anima-native thermal smoke.

Exercises the trainer thermal path (cooldown_every_n_epochs, cooldown_until_temp,
cooldown_poll_seconds, gpu_power_limit_w) on an actual Anima cache-first training
run via trainer.start(), not just isolated helper calls.  This proves the
thermal fields are wired end-to-end on the Anima route.

The smoke uses synthetic Anima cached .npz files and patches the epoch loop
so it can run safely on CPU without a real Anima checkpoint.  Subprocess calls
(nvidia-smi) are intercepted so no machine power limits are actually changed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

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
        raise RuntimeError("xFormers is unavailable in the Anima thermal smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

from core.configs import ModelArch, MixedPrecision, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.training_loop import TrainingLoop


def _write_anima_cache(cache_dir: Path) -> None:
    """Write a minimal paired Anima latent/text cache sample."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = "0_thermal"
    # Latent cache: {stem}_*_anima.npz with latents_* keys, shape [channels, h, w]
    np.savez(
        cache_dir / f"{stem}_000_anima.npz",
        latents_0=np.random.randn(4, 4, 4).astype("float32"),
    )
    # Text cache: {stem}_anima_te.npz with prompt_embeds, shape [tokens, dim]
    np.savez(
        cache_dir / f"{stem}_anima_te.npz",
        prompt_embeds=np.random.randn(8, 16).astype("float32"),
    )


def main() -> int:
    cache_dir = Path("H:/tmp/lulynx_anima_thermal_smoke_data")
    output_dir = Path("H:/tmp/lulynx_anima_thermal_smoke")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_anima_cache(cache_dir)

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path="H:/tmp/anima-tiny-placeholder",
        anima_model_path="H:/tmp/anima-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name="anima_thermal_smoke",
        mixed_precision=MixedPrecision.NO,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=2,
        max_train_steps=1,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        anima_cached_training=True,
        anima_cached_latent_crop_size=4,
        anima_cached_text_token_limit=16,
        anima_native_block_count=28,
        # Thermal fields: all five parameters wired end-to-end
        cooldown_every_n_epochs=1,
        cooldown_minutes=10,
        cooldown_until_temp=65,
        cooldown_poll_seconds=7,
        gpu_power_limit_w=250,
    )

    # Inject a synthetic Anima model so the trainer does not need a real checkpoint.
    class _TinyAnimaBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            return x + self.ff(self.norm2(x))

    class _TinyAnimaDiT(nn.Module):
        def __init__(self, channels: int = 4, hidden: int = 8):
            super().__init__()
            self.in_channels = channels
            self.proj_in = nn.Linear(channels, hidden)
            self.blocks = nn.ModuleList([_TinyAnimaBlock(hidden)])
            self.proj_out = nn.Linear(hidden, channels)

        def enable_gradient_checkpointing(self) -> None:
            self.gradient_checkpointing = True

        def forward(
            self,
            x: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            **_: object,
        ) -> SimpleNamespace:
            del timestep, context
            bsz, channels, height, width = x.shape
            h = x.reshape(bsz, channels, height * width).transpose(1, 2)
            h = self.proj_in(h)
            for block in self.blocks:
                h = block(h)
            h = self.proj_out(h).transpose(1, 2).reshape(bsz, channels, height, width)
            return SimpleNamespace(sample=h)

    model_obj = _TinyAnimaDiT()
    # Mark as Anima-native ready so the trainer accepts the route.
    model = SimpleNamespace(
        unet=model_obj,
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        anima_native_train_ready=True,
        anima_cached_training_ready=True,
        anima_native_executable_report=SimpleNamespace(strict_success=True),
    )

    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(
        model_obj,
        ["attn", "ff"],
        prefix="unet",
    )
    if not injected:
        raise RuntimeError("Anima thermal smoke failed to inject LoRA targets")

    trainer = LulynxTrainer(cfg)
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._ensure_native_family_training_ready()

    logs: list[str] = []
    trainer.set_callbacks(on_log=logs.append)

    # Capture thermal interactions during the full training run.
    sleep_calls: list[float] = []
    power_calls: list[tuple[str, ...]] = []
    temps = iter([68, 66, 64])  # drops below target of 65 after two polls

    trainer._read_gpu_temperature_c = lambda: next(temps)
    trainer._sleep_for_cooldown = lambda seconds: sleep_calls.append(float(seconds))

    def _fake_run(cmd: object, **kwargs: object) -> SimpleNamespace:
        del kwargs
        power_calls.append(tuple(str(part) for part in cmd))  # type: ignore[arg-type]
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    # Patch train_epoch so we don't need a real forward pass -- the thermal
    # path runs at the trainer level (start -> _apply_gpu_power_limit, epoch
    # boundary -> _maybe_cooldown_after_epoch), not inside the training loop.
    def _fake_train_epoch(self: TrainingLoop, dataloader: object, epoch: int) -> dict[str, object]:
        del dataloader, epoch
        return {"avg_loss": 0.0, "steps": 1}

    with patch.object(subprocess, "run", side_effect=_fake_run):
        with patch.object(TrainingLoop, "train_epoch", new=_fake_train_epoch):
            ok = trainer.start()
    if not ok:
        raise RuntimeError("Anima thermal smoke: trainer.start() returned False")

    # 1) Power limit: _apply_gpu_power_limit_if_requested runs at start() and
    #    issues one best-effort nvidia-smi request through the real trainer path.
    assert trainer._gpu_power_limit_attempted, (
        "gpu_power_limit_w=250 should have triggered _apply_gpu_power_limit_if_requested"
    )
    assert power_calls == [("nvidia-smi", "-pl", "250")], power_calls

    # 2) Temperature-based cooldown: after epoch 0 (of 2), the trainer polls
    #    GPU temp with cooldown_poll_seconds=7 until temp <= cooldown_until_temp=65.
    #    Sequence 68->66->64: two polls at 7s each, then temp 64 <= 65 so done.
    assert len(sleep_calls) == 2, (
        f"Expected 2 cooldown polls at 7s each, got {sleep_calls}"
    )
    assert all(s == 7.0 for s in sleep_calls), (
        f"Expected all sleeps at poll_seconds=7.0, got {sleep_calls}"
    )

    print(
        "Anima thermal smoke passed: power-limit and temperature-based "
        "cooldown are wired end-to-end on the Anima training route"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
