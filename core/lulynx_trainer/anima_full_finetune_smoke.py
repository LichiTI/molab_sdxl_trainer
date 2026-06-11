"""Contract smoke for Anima DiT-only full finetune setup.

This does not claim quality or real-checkpoint throughput.  It verifies the
Phase-1 runtime boundary: Anima full finetune trains native DiT parameters,
saves full weights, and resumes them without using a LoRA adapter path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, UnifiedTrainingConfig
from core.lulynx_trainer.anima_full_finetune import (
    build_anima_grouped_param_groups,
    build_anima_full_finetune_state_dict,
    collect_trainable_param_name_map,
    load_anima_full_finetune_state,
    prepare_anima_dit_only_full_finetune,
)
from core.lulynx_trainer.anima_native_dit import AnimaNativeDiTTinyTrainable


def _install_checkpoint_stub(unet: torch.nn.Module) -> None:
    def set_anima_block_checkpointing(enabled: bool, mode: str = "block") -> dict:
        active = bool(enabled)
        block_count = len(getattr(getattr(unet, "net", None), "blocks", []) or [])
        setattr(unet, "anima_block_checkpointing", active)
        setattr(unet, "anima_block_checkpointing_mode", mode if active else "off")
        return {
            "enabled": active,
            "mode": mode if active else "off",
            "block_count": block_count,
            "checkpointed_blocks": block_count if active else 0,
        }

    setattr(unet, "set_anima_block_checkpointing", set_anima_block_checkpointing)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        unet=AnimaNativeDiTTinyTrainable(
            latent_channels=16,
            hidden_dim=8,
            patch_size=2,
            block_count=2,
            condition_dim=8,
            device="cpu",
            dtype=torch.float32,
        ),
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        noise_scheduler=None,
        anima_native_train_ready=True,
        anima_cached_training_ready=True,
    )


def test_prepare_blocks_text_encoder_and_trains_dit() -> None:
    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
    )
    model = _model()
    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)

    assert setup.total_params > 0
    assert cfg.network_train_unet_only is True
    assert cfg.network_train_text_encoder_only is False
    assert all(param.requires_grad for param in model.unet.parameters())
    assert getattr(model, "anima_full_finetune_ready") is True


def test_online_cache_records_frozen_text_conditioning_mode() -> None:
    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        anima_cached_training=True,
        native_cache_mode="online_cache",
        network_train_unet_only=True,
        network_train_text_encoder_only=False,
    )
    model = _model()

    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)

    assert setup.text_conditioning_mode == "frozen_te_online_cache"
    assert cfg.anima_full_finetune_train_text_encoder_requested is False
    assert cfg.anima_full_finetune_text_encoder_policy == "dit_only"
    assert cfg.network_train_unet_only is True
    assert cfg.network_train_text_encoder_only is False


def test_grouped_lr_uses_full_dit_param_names() -> None:
    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        learning_rate=1e-6,
        weight_decay=0.01,
        anima_self_attn_lr=2e-6,
        anima_cross_attn_lr=3e-6,
        anima_mlp_lr=4e-6,
        anima_mod_lr=5e-6,
    )
    model = _model()
    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)

    groups = build_anima_grouped_param_groups(
        config=cfg,
        trainable_params=list(setup.trainable_params),
        param_to_name=collect_trainable_param_name_map(model.unet),
    )
    assert groups is not None
    lrs = {float(group["lr"]) for group in groups}
    assert 2e-6 in lrs
    assert 3e-6 in lrs
    assert 4e-6 in lrs
    assert 5e-6 in lrs


def test_full_state_save_and_resume() -> None:
    cfg = UnifiedTrainingConfig(model_type=ModelArch.ANIMA, training_type="full_finetune")
    model = _model()
    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)
    assert setup.trainable_params
    state = build_anima_full_finetune_state_dict(unet=model.unet)

    assert state
    assert all(key.startswith("unet.") for key in state)

    target = _model().unet
    first_key = next(iter(target.state_dict()))
    before = target.state_dict()[first_key].detach().clone()
    load_report = load_anima_full_finetune_state(unet=target, state_dict=state)
    after = target.state_dict()[first_key].detach()

    assert load_report["loaded"] > 0
    assert not torch.equal(before, after)


def test_full_finetune_runtime_guardrails_enable_block_checkpointing() -> None:
    from core.lulynx_trainer.anima_dit_runtime_guardrails import apply_anima_dit_runtime_guardrails

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        anima_block_checkpointing=True,
        anima_block_checkpointing_mode="block",
    )
    model = _model()
    _install_checkpoint_stub(model.unet)

    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)
    assert setup.trainable_params
    report = apply_anima_dit_runtime_guardrails(
        config=cfg,
        model=model,
        device="cpu",
        dtype=torch.float32,
    )

    profile = report["checkpoint_profile"]
    assert profile["enabled"] is True
    assert profile["checkpointed_blocks"] == 2
    assert profile["source"] == "anima_block_checkpointing"


def main() -> int:
    test_prepare_blocks_text_encoder_and_trains_dit()
    print("  [PASS] prepare_blocks_text_encoder_and_trains_dit")
    test_online_cache_records_frozen_text_conditioning_mode()
    print("  [PASS] online_cache_records_frozen_text_conditioning_mode")
    test_grouped_lr_uses_full_dit_param_names()
    print("  [PASS] grouped_lr_uses_full_dit_param_names")
    test_full_state_save_and_resume()
    print("  [PASS] full_state_save_and_resume")
    test_full_finetune_runtime_guardrails_enable_block_checkpointing()
    print("  [PASS] full_finetune_runtime_guardrails_enable_block_checkpointing")
    print("Anima full-finetune smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
