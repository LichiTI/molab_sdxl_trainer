# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima preview compatibility boundary (Phase 8).

Validates that:
- PreviewState correctly identifies preview capability
- Cache-first mode (None components) blocks sampler creation
- Anima sampler returns None instead of crashing with None components
- Newbie sampler returns None instead of crashing with None components
- Adapter metadata inspection works even without TE/VAE
"""

from __future__ import annotations

import sys
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

# sampler.py uses relative imports — create a minimal package shim
# so we can import it directly in smoke tests.
if "core" not in sys.modules:
    _stub_pkg = SimpleNamespace(__path__=[], __name__="core", __file__="")
    sys.modules["core"] = _stub_pkg
    sys.modules["core.lulynx_trainer"] = SimpleNamespace(
        __path__=["."], __name__="core.lulynx_trainer", __file__=""
    )

# Patch relative imports for sampler.py
import model_family as _mf
sys.modules["core.lulynx_trainer.model_family"] = _mf

import anima_sampler as _as
sys.modules["core.lulynx_trainer.anima_sampler"] = _as

import newbie_sampler as _ns
sys.modules["core.lulynx_trainer.newbie_sampler"] = _ns

# Now we can load sampler
_sampler_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.sampler",
    "sampler.py",
    submodule_search_locations=[],
)
sampler = importlib.util.module_from_spec(_sampler_spec)
sampler.__package__ = "core.lulynx_trainer"
sys.modules["core.lulynx_trainer.sampler"] = sampler
_sampler_spec.loader.exec_module(sampler)

get_preview_state = sampler.get_preview_state
PreviewState = sampler.PreviewState
create_sampler_from_trainer = sampler.create_sampler_from_trainer
get_adapter_metadata = sampler.get_adapter_metadata


def test_preview_state_real():
    """PreviewState.REAL_PREVIEW when all components are present."""
    model = SimpleNamespace(
        model_arch="anima",
        unet=MagicMock(),
        dit=None,
        vae=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=None,
        tokenizer_1=MagicMock(),
        tokenizer_2=None,
        noise_scheduler=None,
    )
    trainer = SimpleNamespace(model=model, config=SimpleNamespace(model_arch="anima"))
    state = get_preview_state(trainer)
    assert state == PreviewState.REAL_PREVIEW, f"Expected REAL_PREVIEW, got {state}"
    print("[PASS] PreviewState.REAL_PREVIEW when all components present")


def test_preview_state_adapter_inspect():
    """PreviewState.ADAPTER_INSPECT when only DiT is available."""
    model = SimpleNamespace(
        model_arch="anima",
        unet=MagicMock(),
        dit=None,
        vae=None,
        text_encoder_1=None,
        text_encoder_2=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
    )
    trainer = SimpleNamespace(model=model, config=SimpleNamespace(model_arch="anima"))
    state = get_preview_state(trainer)
    assert state == PreviewState.ADAPTER_INSPECT, f"Expected ADAPTER_INSPECT, got {state}"
    print("[PASS] PreviewState.ADAPTER_INSPECT when only DiT available")


def test_preview_state_unsupported():
    """PreviewState.UNSUPPORTED when no model or no pipeline."""
    # No model at all
    trainer_no_model = SimpleNamespace(model=None, config=SimpleNamespace())
    assert get_preview_state(trainer_no_model) == PreviewState.UNSUPPORTED

    # Model but no unet
    model_no_unet = SimpleNamespace(
        model_arch="anima",
        unet=None,
        dit=None,
        vae=None,
        text_encoder_1=None,
        text_encoder_2=None,
        tokenizer_1=None,
        tokenizer_2=None,
    )
    trainer_no_unet = SimpleNamespace(model=model_no_unet, config=SimpleNamespace(model_arch="anima"))
    assert get_preview_state(trainer_no_unet) == PreviewState.UNSUPPORTED

    print("[PASS] PreviewState.UNSUPPORTED for missing model/pipeline")


def test_sampler_factory_blocks_none_components():
    """create_sampler_from_trainer returns None when TE/VAE are None."""
    model = SimpleNamespace(
        model_arch="anima",
        unet=MagicMock(),
        dit=None,
        vae=None,
        text_encoder_1=None,
        text_encoder_2=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
    )
    trainer = SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            model_arch="anima",
            sample_sampler="euler",
            sample_width=0,
            sample_height=0,
            sample_seed=0,
        ),
        lora_injector=None,
        device="cpu",
        dtype=None,
    )
    sampler = create_sampler_from_trainer(trainer)
    assert sampler is None, "Sampler should be None when TE/VAE are missing"
    print("[PASS] create_sampler_from_trainer blocks None components")


def test_anima_sampler_none_guard():
    """sample_anima returns None instead of crashing with None components."""
    from anima_sampler import sample_anima

    result = sample_anima(
        dit_model=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        prompt="test",
    )
    assert result is None, "Should return None with None dit_model"

    result2 = sample_anima(
        dit_model=MagicMock(),
        vae=None,
        text_encoder=MagicMock(),
        tokenizer=MagicMock(),
        prompt="test",
    )
    assert result2 is None, "Should return None with None VAE"

    result3 = sample_anima(
        dit_model=MagicMock(),
        vae=MagicMock(),
        text_encoder=None,
        tokenizer=None,
        prompt="test",
    )
    assert result3 is None, "Should return None with None text encoder"

    print("[PASS] sample_anima returns None for all None component cases")


def test_newbie_sampler_none_guard():
    """sample_newbie returns None instead of crashing with None components."""
    from newbie_sampler import sample_newbie

    result = sample_newbie(
        dit_model=None,
        vae=None,
        text_encoder_1=None,
        text_encoder_2=None,
        tokenizer_1=None,
        tokenizer_2=None,
        prompt="test",
    )
    assert result is None, "Should return None with None dit_model"

    result2 = sample_newbie(
        dit_model=MagicMock(),
        vae=MagicMock(),
        text_encoder_1=MagicMock(),
        text_encoder_2=None,
        tokenizer_1=MagicMock(),
        tokenizer_2=None,
        prompt="test",
    )
    assert result2 is None, "Should return None with None text_encoder_2"

    print("[PASS] sample_newbie returns None for all None component cases")


def test_adapter_metadata():
    """Adapter metadata inspection works without TE/VAE."""
    # Mock injector with metadata
    injector = SimpleNamespace(
        network_type="lora",
        rank=16,
        alpha=16.0,
        target_modules=["q_proj", "v_proj"],
    )
    trainer = SimpleNamespace(lora_injector=injector)
    meta = get_adapter_metadata(trainer)
    assert meta is not None, "Metadata should not be None"
    assert meta["adapter_type"] == "SimpleNamespace"
    assert meta["network_type"] == "lora"
    assert meta["rank"] == 16
    assert meta["alpha"] == 16.0
    assert meta["target_modules"] == ["q_proj", "v_proj"]

    # No injector
    trainer_no_inj = SimpleNamespace(lora_injector=None)
    assert get_adapter_metadata(trainer_no_inj) is None

    print("[PASS] Adapter metadata inspection works without TE/VAE")


def test_newbie_preview_state():
    """PreviewState works for Newbie models too."""
    model = SimpleNamespace(
        model_arch="newbie",
        unet=MagicMock(),
        dit=None,
        vae=None,
        text_encoder_1=None,
        text_encoder_2=None,
        tokenizer_1=None,
        tokenizer_2=None,
    )
    trainer = SimpleNamespace(model=model, config=SimpleNamespace(model_arch="newbie"))
    assert get_preview_state(trainer) == PreviewState.ADAPTER_INSPECT
    print("[PASS] Newbie PreviewState works correctly")


if __name__ == "__main__":
    test_preview_state_real()
    test_preview_state_adapter_inspect()
    test_preview_state_unsupported()
    test_sampler_factory_blocks_none_components()
    test_anima_sampler_none_guard()
    test_newbie_sampler_none_guard()
    test_adapter_metadata()
    test_newbie_preview_state()
    print("\n[PASS] All Phase 8 preview boundary smoke tests passed")
