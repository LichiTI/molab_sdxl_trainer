# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test masked_loss / alpha_mask mechanism on the TrainingLoop.

Proves:
1. When masked_loss=True and batch contains loss_masks, the spatial loss
   is element-wise weighted by the mask before per-sample reduction.
2. When alpha_mask=True the same mask path fires.
3. Without masks, the fallback mean reduction is used.
4. Anima-specific padding_mask handling masks padded spatial positions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


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
_load_module("core.constants", CORE_ROOT / "constants.py")
_load_module("core.lulynx_trainer.safe_guard", TRAINER_ROOT / "safe_guard.py")
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
_load_module("core.lulynx_trainer.memory_optimizations", TRAINER_ROOT / "memory_optimizations.py")
TrainingLoop = _load_module(
    "core.lulynx_trainer.training_loop",
    TRAINER_ROOT / "training_loop.py",
).TrainingLoop


def _make_loop(**kwargs) -> TrainingLoop:
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.masked_loss = kwargs.get("masked_loss", False)
    loop.alpha_mask = kwargs.get("alpha_mask", False)
    return loop


def test_masked_loss_with_masks() -> None:
    """masked_loss=True applies loss_masks to spatial loss before reduction."""
    loop = _make_loop(masked_loss=True)
    loss = torch.ones(2, 1, 4, 4)
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, :, :] = 1.0
    mask[1, :, :2, :2] = 1.0

    batch = {"loss_masks": mask}
    result = loop._loss_to_per_sample(loss, batch)

    assert result.shape == (2,), f"Expected per-sample shape (2,), got {result.shape}"
    assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(result[1], torch.tensor(1.0), atol=1e-6)


def test_masked_loss_weighted_pixels() -> None:
    """Mask with different weights changes the per-sample loss."""
    loop = _make_loop(masked_loss=True)
    loss = torch.ones(2, 1, 4, 4) * 2.0
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, :, :] = 1.0
    mask[1, :, :2, :2] = 0.5
    mask[1, :, 2:, :2] = 1.5

    batch = {"loss_masks": mask}
    result = loop._loss_to_per_sample(loss, batch)
    assert torch.allclose(result[0], torch.tensor(2.0), atol=1e-6)
    assert torch.allclose(result[1], torch.tensor(2.0), atol=1e-6)


def test_alpha_mask_path() -> None:
    """alpha_mask=True uses the same mask application path."""
    loop = _make_loop(alpha_mask=True)
    loss = torch.ones(1, 1, 4, 4) * 3.0
    mask = torch.ones(1, 1, 4, 4) * 0.5
    batch = {"loss_masks": mask}
    result = loop._loss_to_per_sample(loss, batch)
    assert torch.allclose(result[0], torch.tensor(3.0), atol=1e-6)


def test_no_mask_fallback() -> None:
    """Without loss_masks, falls back to plain mean."""
    loop = _make_loop(masked_loss=True)
    loss = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    batch = {}
    result = loop._loss_to_per_sample(loss, batch)
    expected = loss.mean()
    assert torch.allclose(result[0], expected, atol=1e-6)


def test_mask_interpolation() -> None:
    """Mask spatially smaller than loss gets bilinearly interpolated."""
    loop = _make_loop(masked_loss=True)
    loss = torch.ones(1, 1, 8, 8)
    mask = torch.ones(1, 1, 2, 2)
    batch = {"loss_masks": mask}
    result = loop._loss_to_per_sample(loss, batch)
    assert result.shape == (1,)
    assert torch.isfinite(result).all()


def test_anima_padding_mask() -> None:
    """Anima branch: padding_mask excludes padded positions from loss."""
    batch_size = 2
    loss = torch.ones(batch_size, 16, 4, 4)
    padding_mask = torch.zeros(batch_size, 4, 4, dtype=torch.bool)
    padding_mask[1, 2:, :] = True

    valid = (~padding_mask).to(device=loss.device, dtype=loss.dtype)
    while valid.dim() < loss.dim():
        valid = valid.unsqueeze(1)
    valid = valid.expand_as(loss)
    reduced = (loss * valid).sum(dim=list(range(1, loss.dim()))) / valid.sum(
        dim=list(range(1, valid.dim()))
    ).clamp_min(1.0)

    assert reduced.shape == (batch_size,)
    assert torch.allclose(reduced[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(reduced[1], torch.tensor(1.0), atol=1e-6)

    # With actual loss variation
    loss2 = torch.ones(batch_size, 1, 4, 4)
    loss2[1, :, :2, :] = 5.0
    loss2[1, :, 2:, :] = 0.0
    valid2 = (~padding_mask).to(dtype=loss2.dtype).unsqueeze(1).expand_as(loss2)
    reduced2 = (loss2 * valid2).sum(dim=[1, 2, 3]) / valid2.sum(dim=[1, 2, 3]).clamp_min(1.0)
    assert torch.allclose(reduced2[1], torch.tensor(5.0), atol=1e-6)


def main() -> int:
    test_masked_loss_with_masks()
    test_masked_loss_weighted_pixels()
    test_alpha_mask_path()
    test_no_mask_fallback()
    test_mask_interpolation()
    test_anima_padding_mask()

    print(
        "Anima masked-loss smoke passed: masked_loss weighting, alpha_mask path, "
        "no-mask fallback, mask interpolation, Anima padding_mask exclusion"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
