"""Lightweight validation for Anima native DiT introspection.

Run from repo root:
    python backend/core/lulynx_trainer/anima_native_dit_validate.py
"""

from __future__ import annotations

if __package__:
    from .anima_native_dit import (
        build_anima_native_dit_stub,
        inspect_anima_safetensors,
        introspect_anima_state_shapes,
        load_anima_native_executable_subset,
        load_anima_native_weight_subset,
        patchify_anima_latents,
        unpatchify_anima_latents,
    )
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anima_native_dit import (
        build_anima_native_dit_stub,
        inspect_anima_safetensors,
        introspect_anima_state_shapes,
        load_anima_native_executable_subset,
        load_anima_native_weight_subset,
        patchify_anima_latents,
        unpatchify_anima_latents,
    )


def main() -> None:
    shapes = {
        "net.x_embedder.proj.1.weight": (2048, 68),
        "net.final_layer.linear.weight": (64, 2048),
        "net.blocks.0.self_attn.q_proj.weight": (2048, 2048),
        "net.blocks.0.self_attn.k_proj.weight": (2048, 2048),
        "net.blocks.0.self_attn.v_proj.weight": (2048, 2048),
        "net.blocks.0.self_attn.output_proj.weight": (2048, 2048),
        "net.blocks.0.cross_attn.q_proj.weight": (2048, 2048),
        "net.blocks.0.cross_attn.k_proj.weight": (2048, 1024),
        "net.blocks.0.cross_attn.v_proj.weight": (2048, 1024),
        "net.blocks.0.cross_attn.output_proj.weight": (2048, 2048),
        "net.blocks.0.mlp.layer1.weight": (8192, 2048),
        "net.blocks.0.mlp.layer2.weight": (2048, 8192),
        "net.blocks.0.adaln_modulation_self_attn.1.weight": (256, 2048),
        "net.blocks.0.adaln_modulation_self_attn.2.weight": (6144, 256),
        "net.blocks.1.self_attn.q_proj.weight": (2048, 2048),
        "net.llm_adapter.proj.weight": (2048, 4096),
    }

    report = introspect_anima_state_shapes(shapes, checkpoint_path="synthetic.safetensors")

    assert report.is_native_dit
    assert report.block_count == 2
    assert report.block_indices == (0, 1)
    assert report.hidden_dim == 2048
    assert report.x_embedder_input_dim == 68
    assert report.final_output_dim == 64
    assert report.latent_channels_hint == 16
    assert report.has_llm_adapter
    assert report.detected_groups["self_attn"]
    assert report.detected_groups["cross_attn"]
    assert report.detected_groups["mlp"]
    assert report.detected_groups["adaln_modulation"]

    try:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        real_checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
        if real_checkpoint.exists():
            real = inspect_anima_safetensors(real_checkpoint)
            assert real.is_native_dit
            assert real.block_count == 28
            assert real.hidden_dim == 2048
            assert real.x_embedder_input_dim == 68
            assert real.final_output_dim == 64
            assert real.latent_channels_hint == 16
            assert real.has_llm_adapter
            assert real.detected_groups["self_attn"]
            assert real.detected_groups["cross_attn"]
            assert real.detected_groups["mlp"]
            assert real.detected_groups["adaln_modulation"]
            from safetensors import safe_open

            with safe_open(str(real_checkpoint), framework="pt", device="cpu") as handle:
                real_shapes = {
                    key: tuple(handle.get_slice(key).get_shape())
                    for key in handle.keys()
                }
            stub = build_anima_native_dit_stub(real, real_shapes)
            stub_keys = set(stub.state_dict().keys())
            weight_keys = {
                key
                for key in real_shapes
                if key.endswith(".weight") or key.endswith(".bias")
            }
            assert stub_keys == weight_keys
            _subset_module, subset_report = load_anima_native_weight_subset(
                real_checkpoint,
                prefixes=(
                    "net.x_embedder.",
                    "net.t_embedder.",
                    "net.t_embedding_norm.",
                    "net.blocks.0.",
                    "net.final_layer.",
                ),
                device="meta",
            )
            assert subset_report.strict_success
            assert subset_report.loaded_key_count > 0
            import torch

            param = next(_subset_module.parameters())
            sample = torch.zeros((1, 16, 8, 8), device=param.device, dtype=param.dtype)
            patches, patch_h, patch_w = patchify_anima_latents(sample, patch_size=2)
            embedded = _subset_module.net.x_embedder.proj(patches)
            assert tuple(embedded.shape) == (1, patch_h * patch_w, 2048)
            projected = _subset_module.net.final_layer.linear(embedded)
            folded = unpatchify_anima_latents(projected, patch_h=patch_h, patch_w=patch_w)
            assert tuple(folded.shape) == tuple(sample.shape)
            executable, executable_report = load_anima_native_executable_subset(
                real_checkpoint,
                block_indices=(0,),
                device="cpu",
                dtype=torch.float32,
            )
            assert executable_report.strict_success
            tiny_latents = torch.randn((1, 16, 4, 4), dtype=torch.float32)
            tiny_timestep = torch.tensor([250.0], dtype=torch.float32)
            tiny_context = torch.randn((1, 4, 1024), dtype=torch.float32)
            tiny_out = executable(
                sample=tiny_latents,
                timestep=tiny_timestep,
                encoder_hidden_states=tiny_context,
            ).sample
            assert tuple(tiny_out.shape) == tuple(tiny_latents.shape)
            assert torch.isfinite(tiny_out).all()
            tiny_target = torch.randn_like(tiny_out)
            loss = torch.nn.functional.mse_loss(tiny_out, tiny_target)
            loss.backward()
            grad_hits = [
                name
                for name, param in executable.named_parameters()
                if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            ]
            assert any("self_attn.q_proj" in name for name in grad_hits)
            assert any("cross_attn.q_proj" in name for name in grad_hits)
            assert any("mlp.layer1" in name for name in grad_hits)
            assert any("t_embedder.1.linear_1" in name for name in grad_hits)
            assert any("final_layer.linear" in name for name in grad_hits)
            print(f"Real Anima checkpoint metadata validation passed: {real_checkpoint}")
    except Exception:
        raise

    print("Anima native DiT introspection validation passed.")


if __name__ == "__main__":
    main()
