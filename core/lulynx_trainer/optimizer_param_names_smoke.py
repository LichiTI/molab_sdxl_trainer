"""Smoke checks for trainer-side optimizer parameter naming.

The native trainer stores loaded model components in a LoadedModel container.
This smoke keeps the PCGrad/optimizer naming path from regressing back to
treating that container as a single torch module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lulynx_trainer.config import LulynxConfig
from core.lulynx_trainer.model_loader import LoadedModel
from core.lulynx_trainer.trainer import LulynxTrainer


def _linear() -> torch.nn.Module:
    return torch.nn.Linear(2, 2)


def _names_for(model: LoadedModel) -> set[str]:
    trainer = LulynxTrainer(LulynxConfig())
    trainer.model = model
    return set(trainer._optimizer_param_names().values())


def _assert_prefix(names: set[str], prefix: str) -> None:
    if not any(name.startswith(prefix) for name in names):
        raise AssertionError(f"missing parameter prefix {prefix!r}: {sorted(names)}")


def test_sdxl_dual_encoder_container() -> None:
    names = _names_for(
        LoadedModel(
            unet=_linear(),
            text_encoder_1=_linear(),
            text_encoder_2=_linear(),
            vae=_linear(),
            tokenizer_1=None,
            tokenizer_2=None,
            noise_scheduler=None,
            model_arch="sdxl",
        )
    )
    _assert_prefix(names, "unet.")
    _assert_prefix(names, "text_encoder_1.")
    _assert_prefix(names, "text_encoder_2.")
    _assert_prefix(names, "vae.")


def test_sd15_single_encoder_container() -> None:
    names = _names_for(
        LoadedModel(
            unet=_linear(),
            text_encoder_1=_linear(),
            text_encoder_2=None,
            vae=_linear(),
            tokenizer_1=None,
            tokenizer_2=None,
            noise_scheduler=None,
            model_arch="sd15",
        )
    )
    _assert_prefix(names, "unet.")
    _assert_prefix(names, "text_encoder_1.")
    _assert_prefix(names, "vae.")


def test_dit_family_extra_components() -> None:
    model = LoadedModel(
        unet=_linear(),
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        model_arch="anima",
    )
    model.anima_qwen3_encoder = _linear()
    model.future_aux_encoder = _linear()

    names = _names_for(model)
    _assert_prefix(names, "unet.")
    _assert_prefix(names, "anima_qwen3_encoder.")
    _assert_prefix(names, "future_aux_encoder.")


def test_future_dit_slot_without_unet() -> None:
    model = LoadedModel(
        unet=None,
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        model_arch="future_dit",
    )
    model.dit = _linear()
    model.transformer = _linear()
    model.denoiser = _linear()

    names = _names_for(model)
    _assert_prefix(names, "dit.")
    _assert_prefix(names, "transformer.")
    _assert_prefix(names, "denoiser.")


def main() -> None:
    test_sdxl_dual_encoder_container()
    test_sd15_single_encoder_container()
    test_dit_family_extra_components()
    test_future_dit_slot_without_unet()
    print("optimizer_param_names_smoke: ok")


if __name__ == "__main__":
    main()
