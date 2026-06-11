"""Anima adapter load/resume smoke.

This validates the production Anima trainer path for:

1. ``network_weights_path`` initial adapter loading
2. ``resume_path`` adapter weight loading (without a trainer-state companion)

The smoke uses the real cache-first Anima route but patches the epoch loop so
it can assert the loaded adapter state before any optimizer step mutates it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, MixedPrecision, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset
from core.lulynx_trainer.anima_targets import get_anima_dit_targets
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.training_loop import TrainingLoop


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
        actual_value = actual[key].detach().cpu()
        expected_value = expected[key].detach().cpu().to(dtype=actual_value.dtype)
        if not torch.equal(expected_value, actual_value):
            raise AssertionError(f"{label}: tensor mismatch for key={key}")


def _build_adapter(checkpoint: Path, output_path: Path) -> dict[str, torch.Tensor]:
    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(28)),
        device="cpu",
        dtype=torch.float32,
    )
    if not report.strict_success:
        raise RuntimeError("Anima executable subset failed to load for adapter smoke")
    for param in model.parameters():
        param.requires_grad_(False)

    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(
        model,
        get_anima_dit_targets(include_llm_adapter=False),
        prefix="unet",
    )
    if not injected:
        raise RuntimeError("Anima adapter smoke failed to inject LoRA layers")

    with torch.no_grad():
        for index, param in enumerate(injector.get_trainable_params(), start=1):
            param.fill_(float(index) / 10.0)

    state_dict = {key: value.detach().cpu().clone() for key, value in injector.get_lora_state_dict().items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path), metadata={"smoke": "anima_resume_adapter"})
    return state_dict


def _base_config(
    *,
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    output_name: str,
) -> UnifiedTrainingConfig:
    return UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(checkpoint),
        anima_model_path=str(checkpoint),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        mixed_precision=MixedPrecision.NO,
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
        anima_cached_training=True,
        anima_cached_latent_crop_size=4,
        anima_cached_text_token_limit=16,
        anima_native_block_count=28,
    )


def _run_case(
    cfg: UnifiedTrainingConfig,
    *,
    expected_state: dict[str, torch.Tensor],
    expected_log_fragment: str,
) -> None:
    trainer = LulynxTrainer(cfg)
    logs: list[str] = []
    trainer.set_callbacks(on_log=logs.append)

    original_train_epoch = TrainingLoop.train_epoch

    def _inspect_loaded_state(self: TrainingLoop, dataloader: object, epoch: int) -> dict[str, object]:
        del dataloader, epoch
        if trainer.lora_injector is None:
            raise AssertionError("Trainer did not initialize Anima LoRA injector")
        actual = trainer.lora_injector.get_lora_state_dict()
        _assert_state_dict_equal(expected_state, actual, label=cfg.output_name)
        return {"avg_loss": 0.0, "steps": 0}

    with patch.object(TrainingLoop, "train_epoch", new=_inspect_loaded_state):
        ok = trainer.start()
    if not ok:
        raise RuntimeError(f"{cfg.output_name}: trainer start failed")
    if not any(expected_log_fragment in message for message in logs):
        raise AssertionError(f"{cfg.output_name}: expected log fragment missing: {expected_log_fragment}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    root = Path("H:/tmp/lulynx_anima_resume_adapter_smoke")
    adapter_path = root / "seed_adapter.safetensors"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")
    if root.exists():
        shutil.rmtree(root)

    expected_state = _build_adapter(checkpoint, adapter_path)

    network_cfg = _base_config(
        checkpoint=checkpoint,
        data_dir=data_dir,
        output_dir=root / "network_weights_out",
        output_name="anima_network_weights_smoke",
    )
    network_cfg.network_weights_path = str(adapter_path)
    _run_case(
        network_cfg,
        expected_state=expected_state,
        expected_log_fragment="Loading initial network weights from",
    )

    resume_cfg = _base_config(
        checkpoint=checkpoint,
        data_dir=data_dir,
        output_dir=root / "resume_out",
        output_name="anima_resume_weights_smoke",
    )
    resume_cfg.resume_path = str(adapter_path)
    _run_case(
        resume_cfg,
        expected_state=expected_state,
        expected_log_fragment="Resuming adapter weights from",
    )

    print(
        "Anima adapter resume smoke passed: network_weights_path and resume_path both load "
        "the expected adapter state on the cache-first Anima trainer path"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
