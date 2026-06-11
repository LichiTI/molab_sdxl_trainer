"""Newbie mixed-precision probe.

This probe exercises the cached-Newbie trainer route under ``no`` / ``fp16`` /
``bf16`` and records whether the configured dtype survives into the tiny native
path cleanly.

It is intentionally honest: if a mode hits a real dtype mismatch on the current
CPU tiny route, the probe surfaces that instead of silently promoting the
feature to PASS.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        raise RuntimeError("xFormers is unavailable in the Newbie mixed-precision smoke")

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


class _ObservedNewbieDenoiser(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 8):
        super().__init__()
        self.in_channels = channels
        self.proj_in = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList([_TinyBlock(hidden)])
        self.proj_out = nn.Linear(hidden, channels)
        self.observed_sample_dtypes: list[torch.dtype] = []
        self.observed_encoder_dtypes: list[torch.dtype] = []
        self.gradient_checkpointing = False

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
        self.observed_sample_dtypes.append(sample.dtype)
        self.observed_encoder_dtypes.append(encoder_hidden_states.dtype)
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


def _expected_dtype(mode: MixedPrecision) -> torch.dtype:
    if mode == MixedPrecision.NO:
        return torch.float32
    if mode == MixedPrecision.FP16:
        return torch.float16
    if mode == MixedPrecision.BF16:
        return torch.bfloat16
    raise AssertionError(f"Unexpected mixed precision mode: {mode}")


def _build_config(cache_dir: Path, output_dir: Path, mode: MixedPrecision) -> UnifiedTrainingConfig:
    return UnifiedTrainingConfig(
        model_type=ModelArch.NEWBIE,
        pretrained_model_name_or_path="H:/tmp/newbie-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name=f"newbie_mixed_precision_{mode.value}",
        mixed_precision=mode,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
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
    )


def _build_model() -> tuple[LoadedModel, _ObservedNewbieDenoiser]:
    unet = _ObservedNewbieDenoiser()
    model = LoadedModel(
        unet=unet,
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
    return model, unet


def _run_case(mode: MixedPrecision) -> tuple[bool, str]:
    cache_dir = Path(f"H:/tmp/lulynx_newbie_mixed_precision_{mode.value}_data")
    output_dir = Path(f"H:/tmp/lulynx_newbie_mixed_precision_{mode.value}")
    _write_cache(cache_dir)

    cfg = _build_config(cache_dir, output_dir, mode)
    model, unet = _build_model()

    injector = LoRAInjector(rank=1, alpha=1, model_arch="newbie")
    injected = injector._inject_model(model.unet, cfg.newbie_target_modules.splitlines(), prefix="unet")
    if not injected:
        raise RuntimeError(f"Newbie mixed-precision smoke failed to inject LoRA targets for mode={mode.value}")

    trainer = LulynxTrainer(cfg)
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._adapter_cpu_residency = None
    trainer._ensure_native_family_training_ready()
    trainer.configure(cfg)

    # Mirror the real prepare() path: loaded Newbie modules and injected LoRA
    # layers live at the configured trainer dtype before the training loop runs.
    trainer.model.unet.to(device=trainer.device, dtype=trainer.dtype)
    for _, injected_layer in trainer.lora_injector.injected_layers.items():
        target_layer = getattr(injected_layer, "lora", injected_layer)
        target_layer.to(device=trainer.device, dtype=trainer.dtype)

    expected_dtype = _expected_dtype(mode)
    assert trainer.dtype == expected_dtype, (
        f"trainer dtype mismatch for mode={mode.value}: expected={expected_dtype} actual={trainer.dtype}"
    )

    ok = trainer.start()
    if not ok:
        return False, f"trainer.start() failed for mode={mode.value}"

    assert unet.observed_sample_dtypes, f"No UNet forward was observed for mode={mode.value}"
    assert unet.observed_encoder_dtypes, f"No encoder dtype was observed for mode={mode.value}"

    observed_sample_dtype = unet.observed_sample_dtypes[-1]
    observed_encoder_dtype = unet.observed_encoder_dtypes[-1]

    if mode == MixedPrecision.NO:
        assert observed_sample_dtype == torch.float32, observed_sample_dtype
        assert observed_encoder_dtype == torch.float32, observed_encoder_dtype
    else:
        # CPU path does not autocast to fp16/bf16, but the trainer still moves
        # module weights/LoRA layers to the configured dtype.
        first_param = next(model.unet.parameters())
        assert first_param.dtype == expected_dtype, (
            f"UNet parameter dtype mismatch for mode={mode.value}: {first_param.dtype}"
        )
        lora_param = next(iter(injector.get_trainable_params()))
        assert lora_param.dtype == expected_dtype, (
            f"LoRA parameter dtype mismatch for mode={mode.value}: {lora_param.dtype}"
        )
    return True, f"mode={mode.value}"


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    for mode in (MixedPrecision.NO, MixedPrecision.FP16, MixedPrecision.BF16):
        try:
            ok, detail = _run_case(mode)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((mode.value, ok, detail))

    for mode, ok, detail in results:
        print(f"{mode}: {'PASS' if ok else 'FAIL'} - {detail}")

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print(
            "Newbie mixed-precision smoke passed: no/fp16/bf16 modes align trainer dtype "
            "and cached-Newbie module parameter dtype on the native route"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
