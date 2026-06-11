"""Native Newbie cpu_offload_checkpoint smoke.

This is intentionally narrow: build a tiny Warehouse safetensors checkpoint,
load it through the real native Newbie transformer loader, and prove one
forward/backward can run inside ``cpu_offload_checkpoint()`` without NaNs or
device drift.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _install_xformers_stub() -> None:
    """Keep this smoke focused on Newbie when local xFormers is unusable."""

    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie cpu-offload smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

from core.lulynx_trainer.memory_optimizations import cpu_offload_checkpoint
from core.lulynx_trainer.newbie_loader import _NextDiTWrapper, _load_transformer_native


def _build_tiny_state_dict(hidden_dim: int = 8, ff_dim: int = 16) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "layers.0.norm1.weight": torch.ones(hidden_dim),
        "layers.0.norm1.bias": torch.zeros(hidden_dim),
        "layers.0.norm2.weight": torch.ones(hidden_dim),
        "layers.0.norm2.bias": torch.zeros(hidden_dim),
        "layers.0.adaLN_modulation.1.weight": torch.randn(hidden_dim * 6, hidden_dim) * 0.02,
        "layers.0.adaLN_modulation.1.bias": torch.randn(hidden_dim * 6) * 0.02,
        "layers.0.attention.qkv.weight": torch.randn(hidden_dim * 3, hidden_dim) * 0.02,
        "layers.0.attention.qkv.bias": torch.randn(hidden_dim * 3) * 0.02,
        "layers.0.attention.out.weight": torch.randn(hidden_dim, hidden_dim) * 0.02,
        "layers.0.attention.out.bias": torch.randn(hidden_dim) * 0.02,
        "layers.0.feed_forward.w1.weight": torch.randn(ff_dim, hidden_dim) * 0.02,
        "layers.0.feed_forward.w1.bias": torch.randn(ff_dim) * 0.02,
        "layers.0.feed_forward.w2.weight": torch.randn(hidden_dim, ff_dim) * 0.02,
        "layers.0.feed_forward.w2.bias": torch.randn(hidden_dim) * 0.02,
        "layers.0.feed_forward.w3.weight": torch.randn(ff_dim, hidden_dim) * 0.02,
        "layers.0.feed_forward.w3.bias": torch.randn(ff_dim) * 0.02,
        "final_layer.norm_final.weight": torch.ones(hidden_dim),
        "final_layer.norm_final.bias": torch.zeros(hidden_dim),
        "final_layer.adaLN_modulation.1.weight": torch.randn(hidden_dim * 2, hidden_dim) * 0.02,
        "final_layer.adaLN_modulation.1.bias": torch.randn(hidden_dim * 2) * 0.02,
        "final_layer.linear.weight": torch.randn(16, hidden_dim) * 0.02,
        "final_layer.linear.bias": torch.randn(16) * 0.02,
    }


def _write_native_transformer_fixture(root: Path) -> Path:
    transformer_dir = root / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    save_file(_build_tiny_state_dict(), str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    (transformer_dir / "config.json").write_text(
        '{"patch_size": 1, "in_channels": 16, "model_type": "nextdit"}',
        encoding="utf-8",
    )
    return transformer_dir


def _collect_finite_grad_summaries(module: torch.nn.Module) -> dict[str, float]:
    summaries: dict[str, float] = {}
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        grad = child.weight.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise AssertionError(f"Non-finite gradient on {name}")
        summaries[name] = float(grad.detach().abs().sum().cpu())
    return summaries


def main() -> int:
    fixture_root = Path("H:/tmp/lulynx_newbie_cpu_offload_native")
    transformer_dir = _write_native_transformer_fixture(fixture_root)

    model, notes = _load_transformer_native(str(transformer_dir), torch.float32, "cpu")
    if not isinstance(model, _NextDiTWrapper):
        raise AssertionError(f"Expected _NextDiTWrapper, got {type(model)!r}; notes={notes}")

    model.enable_gradient_checkpointing()
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None

    sample = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    timestep = torch.tensor([125.0], dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 4, 8, dtype=torch.float32) * 0.01
    added_cond_kwargs = {"text_embeds": torch.randn(1, 8, dtype=torch.float32) * 0.01}

    seen: dict[str, object] = {}

    def _forward(**kwargs):
        seen["sample_device"] = kwargs["sample"].device.type
        seen["encoder_device"] = kwargs["encoder_hidden_states"].device.type
        seen["text_embed_device"] = kwargs["added_cond_kwargs"]["text_embeds"].device.type
        return model(**kwargs)

    output = cpu_offload_checkpoint(
        _forward,
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    if output.shape != sample.shape:
        raise AssertionError(f"Output shape mismatch: expected {tuple(sample.shape)}, got {tuple(output.shape)}")
    if output.device.type != "cpu":
        raise AssertionError(f"Expected CPU output, got {output.device}")
    if not torch.isfinite(output).all():
        raise AssertionError("Forward output contains non-finite values")
    if seen != {"sample_device": "cpu", "encoder_device": "cpu", "text_embed_device": "cpu"}:
        raise AssertionError(f"Unexpected device placement through cpu_offload_checkpoint: {seen}")

    loss = output.float().square().mean()
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss}")
    loss.backward()

    grad_summaries = _collect_finite_grad_summaries(model)
    expected_targets = (
        "layers.0.attention.qkv",
        "layers.0.attention.out",
        "layers.0.feed_forward.w1",
        "layers.0.feed_forward.w2",
        "layers.0.feed_forward.w3",
    )
    missing = [name for name in expected_targets if grad_summaries.get(name, 0.0) <= 0.0]
    if missing:
        raise AssertionError(f"Expected non-zero finite gradients on native Newbie targets, missing={missing}")

    print(
        "Newbie native cpu-offload smoke passed: "
        f"loss={float(loss.detach().cpu()):.6f}, "
        f"targets={sorted(name for name in expected_targets if name in grad_summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

