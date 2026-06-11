"""Newbie-native thermal smoke.

Exercises the trainer thermal path (cooldown_every_n_epochs, cooldown_until_temp,
cooldown_poll_seconds, gpu_power_limit_w) on an actual cached-Newbie training
run via trainer.start(), not just isolated helper calls.  This proves the
thermal fields are wired end-to-end on the Newbie route.
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
        raise RuntimeError("xFormers is unavailable in the Newbie thermal smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

from core.configs import ModelArch, MixedPrecision, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.model_loader import LoadedModel
from core.lulynx_trainer.trainer import LulynxTrainer


class _TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.qkv(x)))


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = _TinyAttention(dim)
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(dim, dim)
        self.feed_forward.w2 = nn.Linear(dim, dim)
        self.feed_forward.w3 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.feed_forward.w2(torch.nn.functional.silu(self.feed_forward.w1(x)))
        return x + 0.1 * self.feed_forward.w3(x)


class _TinyNewbieDenoiser(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 8):
        super().__init__()
        self.in_channels = channels
        self.proj_in = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList([_TinyBlock(hidden)])
        self.proj_out = nn.Linear(hidden, channels)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del timestep
        bsz, channels, height, width = sample.shape
        x = sample.reshape(bsz, channels, height * width).transpose(1, 2)
        x = self.proj_in(x)
        x = x + encoder_hidden_states.mean(dim=1, keepdim=True)
        if added_cond_kwargs and added_cond_kwargs.get("text_embeds") is not None:
            x = x + added_cond_kwargs["text_embeds"].unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x).transpose(1, 2).reshape(bsz, channels, height, width)
        return SimpleNamespace(sample=x)


def _write_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "0_newbie.npz",
        newbie_cache_schema_version=np.array(1, dtype="int64"),
        latents=np.random.randn(16, 4, 4).astype("float32"),
        encoder_hidden_states=np.random.randn(4, 8).astype("float32"),
        pooled_prompt_embeds=np.random.randn(8).astype("float32"),
        attention_mask=np.ones((4,), dtype="bool"),
    )


def main() -> int:
    cache_dir = Path("H:/tmp/lulynx_newbie_thermal_smoke_data")
    output_dir = Path("H:/tmp/lulynx_newbie_thermal_smoke")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_cache(cache_dir)

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.NEWBIE,
        pretrained_model_name_or_path="H:/tmp/newbie-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name="newbie_thermal_smoke",
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
        use_cache=True,
        newbie_target_modules="attention.qkv\nattention.out\nfeed_forward.w1\nfeed_forward.w2\nfeed_forward.w3",
        # Thermal fields: all five parameters wired end-to-end
        cooldown_every_n_epochs=1,
        cooldown_minutes=10,
        cooldown_until_temp=65,
        cooldown_poll_seconds=7,
        gpu_power_limit_w=250,
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
        raise RuntimeError("Newbie thermal smoke failed to inject LoRA targets")

    trainer = LulynxTrainer(cfg)
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._adapter_cpu_residency = None
    trainer._ensure_native_family_training_ready()

    # Capture thermal interactions during the full training run.
    sleep_calls: list[float] = []
    power_calls: list[tuple[str, ...]] = []
    temps = iter([68, 66, 64])  # drops below target of 65 after two polls

    trainer._read_gpu_temperature_c = lambda: next(temps)
    trainer._sleep_for_cooldown = lambda seconds: sleep_calls.append(float(seconds))

    def _fake_run(cmd, **kwargs):
        del kwargs
        power_calls.append(tuple(str(part) for part in cmd))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    with patch.object(subprocess, "run", side_effect=_fake_run):
        ok = trainer.start()
    if not ok:
        raise RuntimeError("Newbie thermal smoke: trainer.start() returned False")

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
    assert not trainer.training_loop or trainer.training_loop.global_step == 1, (
        f"Training should have completed 1 step, got "
        f"{getattr(trainer.training_loop, 'global_step', None)}"
    )

    print("Newbie thermal smoke passed: power-limit and temperature-based "
          "cooldown are wired end-to-end on the Newbie training route")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
