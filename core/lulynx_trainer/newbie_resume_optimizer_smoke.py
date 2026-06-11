# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Tiny cached-Newbie smoke for resume state and optimizer basics closure."""

from __future__ import annotations

import shutil
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
    """Keep the smoke focused on trainer closure when xFormers is unavailable."""

    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie resume smoke")

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
from core.lulynx_trainer.training_loop import TrainingLoop


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
    rng = np.random.default_rng(1234)
    np.savez(
        cache_dir / "0_newbie.npz",
        newbie_cache_schema_version=np.array(1, dtype="int64"),
        latents=rng.standard_normal((16, 4, 4), dtype=np.float32),
        encoder_hidden_states=rng.standard_normal((4, 8), dtype=np.float32),
        pooled_prompt_embeds=rng.standard_normal((8,), dtype=np.float32),
        attention_mask=np.ones((4,), dtype="bool"),
    )


def _build_config(
    *,
    cache_dir: Path,
    output_dir: Path,
    output_name: str,
    max_train_epochs: int,
    max_train_steps: int,
    resume_path: str = "",
) -> UnifiedTrainingConfig:
    return UnifiedTrainingConfig(
        model_type=ModelArch.NEWBIE,
        pretrained_model_name_or_path="H:/tmp/newbie-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        resume_path=resume_path,
        mixed_precision=MixedPrecision.NO,
        optimizer_type=OptimizerType.SGD_NESTEROV,
        optimizer_args="momentum=0.85",
        lr_scheduler=SchedulerType.LINEAR,
        lr_scheduler_args="start_factor=1.0,end_factor=0.5,total_iters=2",
        warmup_ratio=0.0,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=max_train_epochs,
        max_train_steps=max_train_steps,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        save_every_n_epochs=1,
        save_state=True,
        save_state_on_train_end=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        use_cache=True,
        dataloader_num_workers=0,
        newbie_target_modules="attention.qkv\nattention.out\nfeed_forward.w1\nfeed_forward.w2\nfeed_forward.w3",
    )


def _build_model() -> LoadedModel:
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
    return model


def _install_tiny_components(
    trainer: LulynxTrainer,
    *,
    target_modules: str,
    resume_adapter_path: str = "",
) -> None:
    model = _build_model()
    injector = LoRAInjector(rank=1, alpha=1, model_arch="newbie")
    injected = injector._inject_model(model.unet, target_modules.splitlines(), prefix="unet")
    if not injected:
        raise RuntimeError("Newbie resume smoke failed to inject tiny LoRA targets")
    if resume_adapter_path:
        injector.load_lora(resume_adapter_path)
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._adapter_cpu_residency = None
    trainer._ensure_native_family_training_ready()


def _assert_state_dict_equal(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(expected) != set(actual):
        raise AssertionError(
            f"{label}: state dict keys differ: expected={sorted(expected)} actual={sorted(actual)}"
        )
    for key in sorted(expected):
        if not torch.equal(expected[key].cpu(), actual[key].cpu()):
            raise AssertionError(f"{label}: tensor mismatch for key={key}")


def _run_trainer(trainer: LulynxTrainer) -> tuple[bool, dict[str, object]]:
    capture: dict[str, object] = {
        "optimizer_step_calls": 0,
        "scheduler_step_calls": 0,
        "train_epoch_entries": [],
        "train_epoch_results": [],
    }
    original_create_optimizer = LulynxTrainer._create_optimizer
    original_create_scheduler = LulynxTrainer._create_scheduler
    original_train_epoch = TrainingLoop.train_epoch
    original_sgd_step = torch.optim.SGD.step
    original_linear_lr_step = torch.optim.lr_scheduler.LinearLR.step

    def _create_optimizer_with_capture(self: LulynxTrainer):
        optimizer = original_create_optimizer(self)
        capture["optimizer_type"] = type(optimizer).__name__
        capture["optimizer_group"] = {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "momentum": float(optimizer.param_groups[0].get("momentum", 0.0)),
            "nesterov": bool(optimizer.param_groups[0].get("nesterov", False)),
        }
        return optimizer

    def _create_scheduler_with_capture(
        self: LulynxTrainer,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ):
        scheduler = original_create_scheduler(self, optimizer, total_steps)
        capture["scheduler_type"] = type(scheduler).__name__
        capture["scheduler_state_before_train"] = dict(scheduler.state_dict())
        return scheduler

    def _capture_sgd_step(optimizer_self: torch.optim.SGD, *args: object, **kwargs: object):
        capture["optimizer_step_calls"] = int(capture["optimizer_step_calls"]) + 1
        return original_sgd_step(optimizer_self, *args, **kwargs)

    def _capture_linear_lr_step(scheduler_self: torch.optim.lr_scheduler.LinearLR, *args: object, **kwargs: object):
        capture["scheduler_step_calls"] = int(capture["scheduler_step_calls"]) + 1
        return original_linear_lr_step(scheduler_self, *args, **kwargs)

    def _train_epoch_with_capture(
        self: TrainingLoop,
        dataloader: object,
        epoch: int,
    ) -> dict[str, object]:
        entries = capture["train_epoch_entries"]
        assert isinstance(entries, list)
        entries.append(
            {
                "epoch": epoch,
                "global_step": int(self.global_step),
                "scheduler_last_epoch": (
                    int(getattr(self.lr_scheduler, "last_epoch"))
                    if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "last_epoch")
                    else None
                ),
                "optimizer_state_count": len(self.optimizer.state) if self.optimizer is not None else 0,
            }
        )
        result = original_train_epoch(self, dataloader, epoch)
        results = capture["train_epoch_results"]
        assert isinstance(results, list)
        results.append(dict(result))
        return result

    with patch.object(LulynxTrainer, "_create_optimizer", new=_create_optimizer_with_capture):
        with patch.object(LulynxTrainer, "_create_scheduler", new=_create_scheduler_with_capture):
            with patch.object(TrainingLoop, "train_epoch", new=_train_epoch_with_capture):
                with patch.object(torch.optim.SGD, "step", new=_capture_sgd_step):
                    with patch.object(torch.optim.lr_scheduler.LinearLR, "step", new=_capture_linear_lr_step):
                        ok = trainer.start()

    capture["dataset_type"] = type(getattr(trainer, "_dataset", None)).__name__
    capture["final_global_step"] = int(trainer.training_loop.global_step) if trainer.training_loop else -1
    capture["final_scheduler_last_epoch"] = (
        int(getattr(trainer.training_loop.lr_scheduler, "last_epoch"))
        if trainer.training_loop and trainer.training_loop.lr_scheduler is not None
        else None
    )
    capture["final_optimizer_state_count"] = (
        len(trainer.training_loop.optimizer.state)
        if trainer.training_loop and trainer.training_loop.optimizer is not None
        else 0
    )
    return ok, capture


def main() -> int:
    torch.manual_seed(7)

    root = Path("H:/tmp/lulynx_newbie_resume_optimizer_smoke")
    cache_dir = root / "cache"
    output_dir = root / "output"
    if root.exists():
        shutil.rmtree(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_cache(cache_dir)

    first_cfg = _build_config(
        cache_dir=cache_dir,
        output_dir=output_dir,
        output_name="newbie_resume_optimizer_initial",
        max_train_epochs=1,
        max_train_steps=1,
    )
    first_trainer = LulynxTrainer(first_cfg)
    _install_tiny_components(first_trainer, target_modules=first_cfg.newbie_target_modules)
    ok, first_capture = _run_trainer(first_trainer)
    if not ok:
        raise RuntimeError("Initial Newbie cached smoke run failed")

    first_adapter = output_dir / "newbie_resume_optimizer_initial-000001.safetensors"
    first_state = output_dir / "newbie_resume_optimizer_initial-000001-state.pt"
    if not first_adapter.exists():
        raise FileNotFoundError(f"Expected initial adapter save: {first_adapter}")
    if not first_state.exists():
        raise FileNotFoundError(f"Expected companion resume state: {first_state}")
    if first_capture["dataset_type"] != "NewbieCachedDataset":
        raise AssertionError(f"Expected Newbie cached dataset path, got {first_capture['dataset_type']}")
    if first_capture["optimizer_type"] != "SGD":
        raise AssertionError(f"Expected SGD optimizer, got {first_capture['optimizer_type']}")
    if first_capture["scheduler_type"] != "LinearLR":
        raise AssertionError(f"Expected LinearLR scheduler, got {first_capture['scheduler_type']}")
    optimizer_group = first_capture["optimizer_group"]
    assert isinstance(optimizer_group, dict)
    if optimizer_group["momentum"] != 0.85 or optimizer_group["nesterov"] is not True:
        raise AssertionError(f"Unexpected optimizer group configuration: {optimizer_group}")
    if first_capture["optimizer_step_calls"] != 1 or int(first_capture["scheduler_step_calls"]) < 1:
        raise AssertionError(
            "Expected one optimizer step and at least one scheduler step on the cached Newbie run, "
            f"got optimizer={first_capture['optimizer_step_calls']} scheduler={first_capture['scheduler_step_calls']}"
        )
    first_results = first_capture["train_epoch_results"]
    assert isinstance(first_results, list)
    if len(first_results) != 1 or first_results[0].get("steps") != 1:
        raise AssertionError(f"Unexpected initial train_epoch results: {first_results}")
    if first_capture["final_global_step"] != 1:
        raise AssertionError(f"Expected global_step=1 after initial run, got {first_capture['final_global_step']}")
    if first_capture["final_scheduler_last_epoch"] != 1:
        raise AssertionError(
            f"Expected scheduler last_epoch=1 after initial run, got {first_capture['final_scheduler_last_epoch']}"
        )
    if int(first_capture["final_optimizer_state_count"]) <= 0:
        raise AssertionError("Expected optimizer state to be populated after the first Newbie step")

    first_adapter_state = first_trainer._load_state_dict_from_path(first_adapter)

    second_cfg = _build_config(
        cache_dir=cache_dir,
        output_dir=output_dir,
        output_name="newbie_resume_optimizer_resumed",
        max_train_epochs=2,
        max_train_steps=1,
        resume_path=str(first_state),
    )
    second_trainer = LulynxTrainer(second_cfg)
    _install_tiny_components(
        second_trainer,
        target_modules=second_cfg.newbie_target_modules,
        resume_adapter_path=str(first_adapter),
    )
    resumed_state = second_trainer._load_state(second_cfg.resume_path)
    if resumed_state is None:
        raise AssertionError(f"Expected resume state to load from {second_cfg.resume_path}")
    if resumed_state.get("epoch") != 1 or resumed_state.get("global_step") != 1:
        raise AssertionError(f"Unexpected resume state payload: {resumed_state}")
    _assert_state_dict_equal(
        first_adapter_state,
        second_trainer.lora_injector.get_lora_state_dict(),
        label="resume weight load",
    )
    ok, second_capture = _run_trainer(second_trainer)
    if not ok:
        raise RuntimeError("Resumed Newbie cached smoke run failed")

    second_entries = second_capture["train_epoch_entries"]
    second_results = second_capture["train_epoch_results"]
    assert isinstance(second_entries, list)
    assert isinstance(second_results, list)
    if len(second_entries) != 1:
        raise AssertionError(f"Expected exactly one resumed epoch entry, got {second_entries}")
    resumed_entry = second_entries[0]
    if resumed_entry["epoch"] != 1:
        raise AssertionError(f"Expected resume to re-enter at epoch index 1, got {resumed_entry}")
    if resumed_entry["global_step"] != 1:
        raise AssertionError(f"Expected resumed global_step=1 at loop entry, got {resumed_entry}")
    if resumed_entry["scheduler_last_epoch"] != first_capture["final_scheduler_last_epoch"]:
        raise AssertionError(
            "Expected scheduler progress to restore on resume, "
            f"got resumed={resumed_entry['scheduler_last_epoch']} initial={first_capture['final_scheduler_last_epoch']}"
        )
    if resumed_entry["optimizer_state_count"] != first_capture["final_optimizer_state_count"]:
        raise AssertionError(
            "Expected optimizer state to restore on resume, "
            f"got resumed={resumed_entry['optimizer_state_count']} initial={first_capture['final_optimizer_state_count']}"
        )
    if second_capture["optimizer_step_calls"] != 0:
        raise AssertionError(
            "Expected no optimizer updates after resuming at the step limit, "
            f"got optimizer={second_capture['optimizer_step_calls']} scheduler={second_capture['scheduler_step_calls']}"
        )
    if len(second_results) != 1 or second_results[0].get("steps") != 0:
        raise AssertionError(f"Unexpected resumed train_epoch results: {second_results}")
    if second_capture["final_global_step"] != 1:
        raise AssertionError(
            f"Expected global_step to remain 1 after no-op resumed run, got {second_capture['final_global_step']}"
        )

    second_adapter = output_dir / "newbie_resume_optimizer_resumed-000002.safetensors"
    if not second_adapter.exists():
        raise FileNotFoundError(f"Expected resumed adapter save: {second_adapter}")
    second_adapter_state = second_trainer._load_state_dict_from_path(second_adapter)
    _assert_state_dict_equal(first_adapter_state, second_adapter_state, label="resume no-op save")

    print(
        "Newbie resume/optimizer smoke passed: cached path used SGD+LinearLR for one real step, "
        "resume restored step/optimizer/scheduler state from the adapter companion state, "
        "and the resumed epoch performed no extra updates at the saved step limit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
