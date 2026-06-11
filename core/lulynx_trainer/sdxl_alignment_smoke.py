# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for native SDXL TE-cache alignment and component residency."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"
WAREHOUSE_ROOT = CORE_ROOT / "warehouse"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_ensure_namespace("core.warehouse", WAREHOUSE_ROOT)
_ensure_namespace("core.warehouse.training_features", WAREHOUSE_ROOT / "training_features")

training_loop_mod = _load_module("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")
device_state_mod = _load_module("core.lulynx_trainer.device_state", TRAINER_ROOT / "device_state.py")
dataset_loader_mod = _load_module("core.lulynx_trainer.dataset_loader", TRAINER_ROOT / "dataset_loader.py")
checks_mod = _load_module(
    "core.warehouse.training_features.training_config_checks",
    WAREHOUSE_ROOT / "training_features" / "training_config_checks.py",
)

TrainingLoop = training_loop_mod.TrainingLoop
CaptionDataset = dataset_loader_mod.CaptionDataset
SDXLCacheFirstDataset = dataset_loader_mod.SDXLCacheFirstDataset
sdxl_cache_first_collate = dataset_loader_mod.sdxl_cache_first_collate
apply_loaded_model_training_states = device_state_mod.apply_loaded_model_training_states
build_loaded_model_training_states = device_state_mod.build_loaded_model_training_states
module_runtime_state = device_state_mod.module_runtime_state
check_network_targets = checks_mod.check_network_targets


class _TinyTokenizer:
    model_max_length = 8

    def __call__(self, captions, padding, max_length, truncation, return_tensors):
        batch = len(captions)
        ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
        mask = torch.ones_like(ids)
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class _TinyEncoderOutput:
    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor) -> None:
        self.last_hidden_state = hidden
        self.hidden_states = [hidden * 0.5, hidden]
        self.text_embeds = pooled


class _TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, output_hidden_states=True):
        hidden = self.embedding(input_ids)
        pooled = hidden.mean(dim=1)
        return _TinyEncoderOutput(hidden, pooled)


class _TinyLatentDistribution:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def sample(self) -> torch.Tensor:
        return self._tensor


class _TinyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, images: torch.Tensor):
        return SimpleNamespace(latent_dist=_TinyLatentDistribution(self.proj(images)))


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(4, 4, kernel_size=1, bias=False)

    def forward(self, x, *args, **kwargs):
        return self.proj(x)


class _TinyScheduler:
    pass


def _make_loop(*, train_text_encoder: bool, text_encoder_cpu_residency: bool, vae_cpu_residency: bool) -> TrainingLoop:
    return TrainingLoop(
        unet=_TinyUNet(),
        text_encoder_1=_TinyTextEncoder(hidden_size=6),
        text_encoder_2=_TinyTextEncoder(hidden_size=4),
        vae=_TinyVAE(),
        tokenizer_1=_TinyTokenizer(),
        tokenizer_2=_TinyTokenizer(),
        noise_scheduler=_TinyScheduler(),
        lora_injector=None,
        optimizer=torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=0.1),
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        te_manager=None,
        model_arch="sdxl",
        train_text_encoder=train_text_encoder,
        text_encoder_cpu_residency=text_encoder_cpu_residency,
        vae_cpu_residency=vae_cpu_residency,
    )


def test_network_target_contracts() -> bool:
    report = check_network_targets({"cache_text_encoder_outputs": True})
    assert any("UNet-only" in warning for warning in report.warnings), report.warnings

    conflict = check_network_targets(
        {
            "cache_text_encoder_outputs": True,
            "network_train_text_encoder_only": True,
        }
    )
    assert any("Text encoder-only training" in error for error in conflict.errors), conflict.errors
    print("PASS: test_network_target_contracts")
    return True


def test_device_state_training_plan_and_runtime_restore() -> bool:
    model = SimpleNamespace(
        unet=nn.Linear(2, 2),
        text_encoder_1=nn.Linear(2, 2).double(),
        text_encoder_2=nn.Linear(2, 2).double(),
        vae=nn.Linear(2, 2).double(),
    )
    states = build_loaded_model_training_states(
        device="cpu",
        train_text_encoder=False,
        keep_text_encoders_on_cpu=True,
        keep_vae_on_cpu=True,
    )
    apply_loaded_model_training_states(model, states)
    assert not model.text_encoder_1.training
    assert not next(model.text_encoder_1.parameters()).requires_grad
    assert next(model.text_encoder_1.parameters()).dtype == torch.float32
    assert next(model.vae.parameters()).dtype == torch.float32

    model.text_encoder_1.double()
    with module_runtime_state(model.text_encoder_1, device="cpu", dtype=torch.float32):
        assert next(model.text_encoder_1.parameters()).dtype == torch.float32
    assert next(model.text_encoder_1.parameters()).dtype == torch.float64
    print("PASS: test_device_state_training_plan_and_runtime_restore")
    return True


def test_sdxl_frozen_text_encoding_is_no_grad() -> bool:
    loop = _make_loop(
        train_text_encoder=False,
        text_encoder_cpu_residency=True,
        vae_cpu_residency=True,
    )
    result = loop._encode_prompt_native(["hello world"])
    assert loop._text_encoder_cpu_residency is True
    assert result["encoder_hidden_states"].requires_grad is False
    assert result["pooled_prompt_embeds"].requires_grad is False

    latents = loop._encode_latents_with_vae(torch.randn(1, 3, 4, 4))
    assert latents.shape == (1, 4, 4, 4), latents.shape
    assert torch.isfinite(latents).all()
    print("PASS: test_sdxl_frozen_text_encoding_is_no_grad")
    return True


def test_sdxl_cpu_residency_realigns_helper_dtypes() -> bool:
    loop = _make_loop(
        train_text_encoder=False,
        text_encoder_cpu_residency=True,
        vae_cpu_residency=True,
    )
    loop.vae.double()
    loop.text_encoder_1.double()
    loop.text_encoder_2.double()

    loop._ensure_cpu_resident_components("smoke")

    assert next(loop.vae.parameters()).device.type == "cpu"
    assert next(loop.text_encoder_1.parameters()).device.type == "cpu"
    assert next(loop.text_encoder_2.parameters()).device.type == "cpu"
    assert next(loop.vae.parameters()).dtype == torch.float32
    assert next(loop.text_encoder_1.parameters()).dtype == torch.float32
    assert next(loop.text_encoder_2.parameters()).dtype == torch.float32
    assert not loop.vae.training
    assert not loop.text_encoder_1.training
    assert not loop.text_encoder_2.training
    assert not next(loop.vae.parameters()).requires_grad
    assert not next(loop.text_encoder_1.parameters()).requires_grad
    assert not next(loop.text_encoder_2.parameters()).requires_grad
    print("PASS: test_sdxl_cpu_residency_realigns_helper_dtypes")
    return True


def test_sdxl_cache_first_dataset_emits_cached_batch_contract() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        Image.new("RGB", (4, 4), color=(128, 64, 32)).save(root / "sample.png")
        (root / "sample.txt").write_text("masterpiece, test", encoding="utf-8")

        base = CaptionDataset(str(root), resolution=4, enable_bucket=False, shuffle_caption=False)
        cached = SDXLCacheFirstDataset(
            base,
            vae=_TinyVAE(),
            text_encoder_1=_TinyTextEncoder(hidden_size=6),
            text_encoder_2=_TinyTextEncoder(hidden_size=4),
            tokenizer_1=_TinyTokenizer(),
            tokenizer_2=_TinyTokenizer(),
            device="cpu",
            dtype=torch.float32,
            cache_dir=str(root / ".cache"),
            model_id="tiny",
        )
        item = cached[0]
        batch = sdxl_cache_first_collate([item])

        assert "images" not in batch
        assert tuple(batch["latents"].shape) == (1, 4, 4, 4), batch["latents"].shape
        assert tuple(batch["encoder_hidden_states"].shape) == (1, 8, 10), batch["encoder_hidden_states"].shape
        assert tuple(batch["pooled_prompt_embeds"].shape) == (1, 4), batch["pooled_prompt_embeds"].shape

    print("PASS: test_sdxl_cache_first_dataset_emits_cached_batch_contract")
    return True


def test_sdxl_trainable_text_encoding_keeps_grad_flow() -> bool:
    loop = _make_loop(
        train_text_encoder=True,
        text_encoder_cpu_residency=True,
        vae_cpu_residency=True,
    )
    result = loop._encode_prompt_native(["hello world"])
    assert loop._text_encoder_cpu_residency is False
    assert result["encoder_hidden_states"].requires_grad is True
    assert result["pooled_prompt_embeds"].requires_grad is True
    print("PASS: test_sdxl_trainable_text_encoding_keeps_grad_flow")
    return True


def main() -> int:
    tests = [
        test_network_target_contracts,
        test_device_state_training_plan_and_runtime_restore,
        test_sdxl_frozen_text_encoding_is_no_grad,
        test_sdxl_cpu_residency_realigns_helper_dtypes,
        test_sdxl_cache_first_dataset_emits_cached_batch_contract,
        test_sdxl_trainable_text_encoding_keeps_grad_flow,
    ]
    results = []
    for test_fn in tests:
        try:
            results.append((test_fn.__name__, test_fn()))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {exc}")
            results.append((test_fn.__name__, False))

    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("SDXL Alignment Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
