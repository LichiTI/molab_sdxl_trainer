"""Synthetic smoke checks for the Warehouse Anima native DiT scaffold.

Run from the repository root:

    python backend/core/lulynx_trainer/anima_smoke.py

This script uses tiny synthetic shapes only.  It does not load real Anima
weights and it does not validate a training forward.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import torch

if __package__ in (None, ""):
    module_path = Path(__file__).with_name("anima_native_dit.py")
    spec = importlib.util.spec_from_file_location("anima_native_dit_smoke", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke module: {module_path}")
    anima_native_dit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = anima_native_dit
    spec.loader.exec_module(anima_native_dit)
else:
    from . import anima_native_dit

build_anima_native_dit_stub = anima_native_dit.build_anima_native_dit_stub
discover_anima_native_param_groups = anima_native_dit.discover_anima_native_param_groups
introspect_anima_state_shapes = anima_native_dit.introspect_anima_state_shapes
patchify_anima_latents = anima_native_dit.patchify_anima_latents
unpatchify_anima_latents = anima_native_dit.unpatchify_anima_latents
AnimaNativeDiTTinyTrainable = anima_native_dit.AnimaNativeDiTTinyTrainable


ShapeMap = Dict[str, Tuple[int, ...]]


def _synthetic_shapes() -> ShapeMap:
    shapes: ShapeMap = {
        "net.x_embedder.proj.1.weight": (8, 4),
        "net.final_layer.adaln_modulation.1.weight": (4, 8),
        "net.final_layer.adaln_modulation.2.weight": (16, 4),
        "net.final_layer.linear.weight": (64, 8),
        "net.llm_adapter.proj.weight": (8, 8),
    }
    for index in range(2):
        prefix = f"net.blocks.{index}"
        for group in ("self_attn", "cross_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "output_proj"):
                shapes[f"{prefix}.{group}.{proj}.weight"] = (8, 8)
        shapes[f"{prefix}.mlp.layer1.weight"] = (16, 8)
        shapes[f"{prefix}.mlp.layer2.weight"] = (8, 16)
        for mod_name in (
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
        ):
            shapes[f"{prefix}.{mod_name}.1.weight"] = (4, 8)
            shapes[f"{prefix}.{mod_name}.2.weight"] = (24, 4)
    return shapes


def main() -> int:
    shapes = _synthetic_shapes()
    introspection = introspect_anima_state_shapes(
        shapes,
        checkpoint_path="<synthetic>",
    )
    assert introspection.is_native_dit
    assert introspection.block_count == 2
    assert introspection.hidden_dim == 8
    assert introspection.final_output_dim == 64
    assert introspection.latent_channels_hint == 16

    shape_groups = discover_anima_native_param_groups(shapes)
    assert len(shape_groups.self_attn) == 8
    assert len(shape_groups.cross_attn) == 8
    assert len(shape_groups.mlp) == 4
    assert len(shape_groups.mod) == 14
    assert len(shape_groups.llm_adapter) == 1

    introspection_groups = discover_anima_native_param_groups(introspection)
    assert len(introspection_groups.self_attn) == 8
    assert len(introspection_groups.cross_attn) == 8
    assert len(introspection_groups.mlp) == 4
    assert len(introspection_groups.mod) == 12
    assert len(introspection_groups.llm_adapter) == 1

    stub = build_anima_native_dit_stub(introspection, shapes)
    module_groups = discover_anima_native_param_groups(stub)
    assert len(module_groups.self_attn) == 8
    assert len(module_groups.cross_attn) == 8
    assert len(module_groups.mlp) == 4
    assert len(module_groups.mod) == 14
    assert len(module_groups.llm_adapter) == 1
    assert getattr(stub, "is_shape_only_stub") is True

    real_contract_latents = torch.randn(1, 16, 8, 8)
    real_contract_mask = torch.ones(1, 1, 8, 8)
    real_contract_tokens, patch_h, patch_w = patchify_anima_latents(
        real_contract_latents,
        real_contract_mask,
        patch_size=2,
    )
    assert tuple(real_contract_tokens.shape) == (1, 16, 68)
    folded = unpatchify_anima_latents(
        torch.randn(1, 16, 64),
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=2,
    )
    assert tuple(folded.shape) == tuple(real_contract_latents.shape)

    try:
        stub(None)
    except NotImplementedError as exc:
        assert "shape-only" in str(exc)
    else:
        raise AssertionError("AnimaNativeDiTStub.forward must remain blocked")

    tiny = AnimaNativeDiTTinyTrainable(
        latent_channels=16,
        hidden_dim=8,
        patch_size=2,
        block_count=2,
        condition_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    tiny_groups = discover_anima_native_param_groups(tiny)
    assert len(tiny_groups.self_attn) == 8
    assert len(tiny_groups.cross_attn) == 8
    assert len(tiny_groups.mlp) == 4
    assert len(tiny_groups.mod) == 2
    assert len(tiny_groups.llm_adapter) == 1

    # Flow-style Anima contract smoke: interpolate latent->noise and predict velocity.
    latents = torch.randn(1, 16, 8, 8)
    noise = torch.randn_like(latents)
    flow_t = torch.tensor([0.25], dtype=latents.dtype)
    view_t = flow_t.view(1, 1, 1, 1)
    noisy_latents = (1.0 - view_t) * latents + view_t * noise
    target = noise - latents
    encoder_hidden_states = torch.randn(1, 4, 8)
    text_embeds = torch.randn(1, 8)

    output = tiny(
        sample=noisy_latents,
        timestep=flow_t * 1000.0,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": text_embeds},
    ).sample
    assert tuple(output.shape) == tuple(target.shape)

    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    grad_hits = [
        name
        for name, param in tiny.named_parameters()
        if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    ]
    assert any("self_attn.q_proj" in name for name in grad_hits)
    assert any("cross_attn.q_proj" in name for name in grad_hits)
    assert any("final_layer.linear" in name for name in grad_hits)

    # LoRA target injection smoke: native target suffixes must match module names.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.lulynx_trainer.anima_targets import get_anima_dit_targets
    from core.lulynx_trainer.lora_injector import LoRAInjector

    lora_tiny = AnimaNativeDiTTinyTrainable(device="cpu", dtype=torch.float32)
    injector = LoRAInjector(rank=2, alpha=2, model_arch="anima")
    injected = injector._inject_model(lora_tiny, get_anima_dit_targets(), prefix="net")
    assert injected

    lora_output = lora_tiny(
        sample=noisy_latents,
        timestep=flow_t * 1000.0,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": text_embeds},
    ).sample
    lora_loss = torch.nn.functional.mse_loss(lora_output, target)
    lora_loss.backward()
    lora_grad_hits = [
        name
        for name, param in injector.get_lora_state_dict().items()
        if isinstance(param, torch.Tensor)
    ]
    assert lora_grad_hits

    # Minimal save/reload boundary for future Anima adapter exports.
    from safetensors.torch import load_file, save_file

    save_path = Path("H:/tmp/lulynx_anima_tiny_adapter.safetensors")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu() for k, v in injector.get_lora_state_dict().items()},
        str(save_path),
        metadata={"model_family": "anima", "smoke": "tiny_native_dit"},
    )
    reloaded = load_file(str(save_path), device="cpu")
    assert reloaded
    assert set(reloaded) == set(injector.get_lora_state_dict())

    print("Anima native synthetic smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

