"""Smoke tests for the SDXL native UNet phase-2 contract."""

from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.native_unet import (
    NativeSDXLAddEmbedding,
    NativeSDXLAttention,
    NativeSDXLAttentionConfig,
    NativeSDXLCrossAttnUpBlock2D,
    NativeSDXLCrossAttnUpBlockConfig,
    NativeSDXLDownBlock2D,
    NativeSDXLDownBlockConfig,
    NativeSDXLDownsample2D,
    NativeSDXLDownsamplerConfig,
    NativeSDXLMidBlock2D,
    NativeSDXLMidBlockConfig,
    NativeSDXLResnetBlock2D,
    NativeSDXLResnetBlockConfig,
    NativeSDXLShellConfig,
    NativeSDXLShellModules,
    NativeSDXLTimestepEmbedding,
    NativeSDXLTransformer2DConfig,
    NativeSDXLTransformerBlockConfig,
    NativeSDXLUNetSkeleton,
    NativeSDXLUNetSkeletonCompat,
    NativeSDXLUNetSkeletonConfig,
    NativeSDXLUNetProxy,
    NativeSDXLUpBlock2D,
    NativeSDXLUpBlockConfig,
    NativeSDXLUpsample2D,
    NativeSDXLUpsamplerConfig,
    build_sdxl_attention_from_manifest,
    build_sdxl_cross_attn_down_block_from_manifest,
    build_sdxl_cross_attn_up_block_from_manifest,
    build_sdxl_down_block_from_manifest,
    build_sdxl_mid_block_from_manifest,
    build_sdxl_native_module_from_manifest,
    build_sdxl_resnet_from_manifest,
    build_sdxl_shell_from_manifest,
    build_sdxl_up_block_from_manifest,
    build_sdxl_unet_skeleton_config_from_manifest,
    compare_shape_metadata,
    load_sdxl_attention_state_from_manifest,
    load_sdxl_down_block_state_from_manifest,
    load_sdxl_mid_block_state_from_manifest,
    load_sdxl_resnet_state_from_manifest,
    load_sdxl_shell_state_from_manifest,
    load_sdxl_up_block_state_from_manifest,
    build_sdxl_native_unet_preflight_profile,
    build_sdxl_unet_status,
    install_sdxl_native_unet_backend,
    timestep_embedding,
)


class _Out:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.mid_block = nn.Linear(4, 4)
        self.up_blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.config = {"sample_size": 4}

    def forward(self, sample: torch.Tensor, **_: object) -> _Out:
        x = sample
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return _Out(x)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_native_shadow() -> None:
    model = SimpleNamespace(unet=_TinyUNet())
    status = install_sdxl_native_unet_backend(model, backend="native_shadow")
    _assert(status.backend == "native_shadow", "shadow backend normalized")
    _assert(not isinstance(model.unet, NativeSDXLUNetProxy), "shadow does not replace unet")
    _assert(len(status.blocks) == 5, "shadow block graph includes down/mid/up")
    _assert(hasattr(model, "native_unet_shadow"), "shadow metadata attached")


def test_native_proxy_forward_parity() -> None:
    torch.manual_seed(1)
    reference = _TinyUNet()
    model = SimpleNamespace(unet=reference)
    sample = torch.randn(2, 4)
    expected = reference(sample=sample).sample.detach()
    status = install_sdxl_native_unet_backend(model, backend="native_proxy")
    _assert(status.active, "proxy active")
    _assert(isinstance(model.unet, NativeSDXLUNetProxy), "proxy replaces unet")
    actual = model.unet(sample=sample).sample.detach()
    torch.testing.assert_close(actual, expected)
    _assert(len(list(model.unet.down_blocks)) == 2, "proxy exposes down blocks")
    graph = model.unet.native_block_graph()
    _assert(graph["lifecycle"]["offload_contract"] == "block_graph", "proxy exposes lifecycle contract")
    _assert(len(graph["precision_swap_units"]) == 5, "proxy exposes precision swap units")


def test_native_skeleton_metadata_mode() -> None:
    model = SimpleNamespace(unet=_TinyUNet())
    status = install_sdxl_native_unet_backend(model, backend="native_skeleton")
    data = status.as_dict()
    _assert(status.backend == "native_skeleton", "skeleton backend normalized")
    _assert(status.mode == "skeleton_metadata", "skeleton mode")
    _assert(not status.active, "skeleton does not replace training UNet")
    _assert(not isinstance(model.unet, NativeSDXLUNetProxy), "skeleton keeps reference unet")
    _assert(data["native_ready_blocks"] == data["blocks_total"], "skeleton marks top blocks native ready")
    _assert(data["native_coverage"]["skeleton_ready"], "skeleton coverage ready")
    _assert(data["native_coverage"]["native_forward_probe_ok"], "skeleton forward probe ready")
    _assert(
        data["native_coverage"]["native_forward_probe"]["uses_sdxl_text_time_condition"],
        "skeleton forward probe uses SDXL text-time condition",
    )


def test_block_graph_contract() -> None:
    status = build_sdxl_unet_status(_TinyUNet(), backend="native_shadow", active=False)
    data = status.as_dict()
    _assert(data["blocks_total"] == 5, "block count")
    _assert(data["native_ready_blocks"] == 0, "native blocks start disabled")
    _assert(data["weight_mapping"]["target_format"] == "lulynx_native_sdxl_unet", "mapping target")
    orders = [item["order"] for item in status.precision_swap_units()]
    _assert(orders == sorted(orders), "precision units are forward ordered")


def test_lulynx_native_preflight_profile() -> None:
    profile = build_sdxl_native_unet_preflight_profile(backend="lulynx_native")
    _assert(profile is not None, "lulynx_native profile exists")
    _assert(profile["active"], "lulynx_native profile is active")
    _assert(profile["mode"] == "native_full", "lulynx_native profile mode")
    _assert(profile["native_forward_integrated"], "lulynx_native profile marks native integration")
    _assert(profile["native_coverage"]["native_forward_integrated"], "lulynx_native nested coverage marks native integration")
    _assert("reference/proxy path" not in profile["native_coverage"].get("reason", ""), "lulynx_native coverage reason is not stale")


def test_weight_residency_threshold_contract() -> None:
    from core.lulynx_trainer.native_unet.weight_residency import (
        LulynxManagedConv2d,
        LulynxManagedLinear,
        apply_weight_residency,
    )

    module = nn.Sequential(
        LulynxManagedLinear(4, 4, bias=False),
        LulynxManagedLinear(64, 64, bias=False),
        LulynxManagedConv2d(4, 4, kernel_size=1, bias=False),
    )
    module.requires_grad_(False)
    report = apply_weight_residency(module, mode="linear_conv_cpu_pinned", min_parameter_count=100)
    data = report.as_dict()
    _assert(data["managed_linear_count"] == 2, "managed linear count")
    _assert(data["managed_conv2d_count"] == 1, "managed conv count")
    _assert(data["active_linear_count"] == 1, "threshold keeps only large linear active")
    _assert(data["active_conv2d_count"] == 0, "threshold skips small conv")
    _assert(data["skipped_small_count"] == 2, "threshold skip count")
    _assert(data["min_parameter_count"] == 100, "threshold recorded")


def _copy_linear(target: nn.Linear, source: nn.Linear) -> None:
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def test_sdxl_shell_synthetic_forward_parity() -> None:
    torch.manual_seed(7)
    config = NativeSDXLShellConfig(
        in_channels=2,
        base_channels=4,
        time_embed_dim=8,
        add_embed_in_dim=6,
        add_embed_dim=8,
        out_channels=2,
        norm_num_groups=2,
    )
    shell = NativeSDXLShellModules(config)

    ref_conv_in = nn.Conv2d(2, 4, 3, padding=1)
    ref_time = NativeSDXLTimestepEmbedding(4, 8)
    ref_add = NativeSDXLAddEmbedding(6, 8)
    ref_norm = nn.GroupNorm(2, 4, eps=1e-5)
    ref_conv_out = nn.Conv2d(4, 2, 3, padding=1)

    shell.conv_in.load_state_dict(ref_conv_in.state_dict())
    _copy_linear(shell.time_embedding.linear_1, ref_time.linear_1)
    _copy_linear(shell.time_embedding.linear_2, ref_time.linear_2)
    _copy_linear(shell.add_embedding.linear_1, ref_add.linear_1)
    _copy_linear(shell.add_embedding.linear_2, ref_add.linear_2)
    shell.conv_norm_out.load_state_dict(ref_norm.state_dict())
    shell.conv_out.load_state_dict(ref_conv_out.state_dict())

    image = torch.randn(2, 2, 8, 8)
    time = torch.randn(2, 4)
    add = torch.randn(2, 6)
    hidden = torch.randn(2, 4, 8, 8)

    torch.testing.assert_close(shell.forward_input(image), ref_conv_in(image))
    torch.testing.assert_close(shell.forward_time_embedding(time), ref_time(time))
    torch.testing.assert_close(shell.forward_add_embedding(add), ref_add(add))
    expected_out = ref_conv_out(torch.nn.functional.silu(ref_norm(hidden)))
    torch.testing.assert_close(shell.forward_output(hidden), expected_out)


def test_sdxl_shell_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    state = load_sdxl_shell_state_from_manifest(manifest)
    shell = build_sdxl_shell_from_manifest(manifest)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, shell.shape_metadata())
    _assert(not shape_report["missing"], "real shell has no missing keys")
    _assert(not shape_report["unexpected"], "real shell has no unexpected keys")
    _assert(not shape_report["mismatched"], "real shell shapes match mapped checkpoint")
    _assert(shell.conv_in.weight.shape[1] == 4, "real SDXL conv_in consumes latent channels")
    _assert(shell.conv_out.weight.shape[0] == 4, "real SDXL conv_out emits latent channels")


def test_sdxl_real_weight_shell_down0_forward_probe() -> None:
    torch.manual_seed(9)
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    shell = build_sdxl_shell_from_manifest(manifest, dtype=torch.float32)
    down = build_sdxl_down_block_from_manifest(manifest, "down_blocks.0", dtype=torch.float32)
    sample = torch.randn(1, 4, 16, 16)
    timestep_embedding_value = torch.randn(1, shell.config.base_channels)
    add_embedding = torch.randn(1, shell.config.add_embed_in_dim)
    temb = shell.forward_time_embedding(timestep_embedding_value) + shell.forward_add_embedding(add_embedding)
    hidden = shell.forward_input(sample)
    hidden, skips = down.forward_with_skips(hidden, temb)
    _assert(tuple(hidden.shape) == (1, 320, 8, 8), "real shell+down0 forward hidden shape")
    _assert(len(skips) == 3, "real shell+down0 forward skip count")
    _assert(tuple(skips[0].shape) == (1, 320, 16, 16), "real shell+down0 first skip shape")


def test_sdxl_resnet_synthetic_forward_parity() -> None:
    torch.manual_seed(9)
    config = NativeSDXLResnetBlockConfig(
        in_channels=4,
        out_channels=6,
        time_embed_dim=8,
        norm_num_groups=2,
        use_conv_shortcut=True,
    )
    block = NativeSDXLResnetBlock2D(config)
    hidden = torch.randn(2, 4, 8, 8)
    temb = torch.randn(2, 8)

    norm1 = nn.GroupNorm(2, 4, eps=1e-5)
    conv1 = nn.Conv2d(4, 6, 3, padding=1)
    time_proj = nn.Linear(8, 6)
    norm2 = nn.GroupNorm(2, 6, eps=1e-5)
    conv2 = nn.Conv2d(6, 6, 3, padding=1)
    shortcut = nn.Conv2d(4, 6, 1)

    block.norm1.load_state_dict(norm1.state_dict())
    block.conv1.load_state_dict(conv1.state_dict())
    block.time_emb_proj.load_state_dict(time_proj.state_dict())
    block.norm2.load_state_dict(norm2.state_dict())
    block.conv2.load_state_dict(conv2.state_dict())
    _assert(block.conv_shortcut is not None, "synthetic resnet uses shortcut")
    block.conv_shortcut.load_state_dict(shortcut.state_dict())

    expected = conv1(torch.nn.functional.silu(norm1(hidden)))
    expected = expected + time_proj(torch.nn.functional.silu(temb))[:, :, None, None]
    expected = conv2(torch.nn.functional.silu(norm2(expected)))
    expected = expected + shortcut(hidden)
    torch.testing.assert_close(block(hidden, temb), expected)


def test_sdxl_resnet_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "down_blocks.0.resnets.0"
    state = load_sdxl_resnet_state_from_manifest(manifest, target_prefix)
    block = build_sdxl_resnet_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real ResNet has no missing keys")
    _assert(not shape_report["unexpected"], "real ResNet has no unexpected keys")
    _assert(not shape_report["mismatched"], "real ResNet shapes match mapped checkpoint")
    _assert(tuple(block.conv1.weight.shape[:2]) == (320, 320), "first SDXL ResNet conv1 shape")
    _assert(block.time_emb_proj.weight.shape[1] == 1280, "first SDXL ResNet consumes time embedding")


def test_sdxl_down_block_synthetic_forward_parity() -> None:
    torch.manual_seed(11)
    first = NativeSDXLResnetBlockConfig(
        in_channels=4,
        out_channels=4,
        time_embed_dim=8,
        norm_num_groups=2,
    )
    second = NativeSDXLResnetBlockConfig(
        in_channels=4,
        out_channels=6,
        time_embed_dim=8,
        norm_num_groups=2,
        use_conv_shortcut=True,
    )
    downsample = NativeSDXLDownsamplerConfig(channels=6, out_channels=6)
    block = NativeSDXLDownBlock2D(NativeSDXLDownBlockConfig(resnets=(first, second), downsampler=downsample))
    ref_first = NativeSDXLResnetBlock2D(first)
    ref_second = NativeSDXLResnetBlock2D(second)
    ref_downsample = NativeSDXLDownsample2D(downsample)

    block.resnets[0].load_state_dict(ref_first.state_dict())
    block.resnets[1].load_state_dict(ref_second.state_dict())
    block.downsamplers[0].load_state_dict(ref_downsample.state_dict())

    hidden = torch.randn(2, 4, 8, 8)
    temb = torch.randn(2, 8)
    expected = ref_downsample(ref_second(ref_first(hidden, temb), temb))
    torch.testing.assert_close(block(hidden, temb), expected)


def test_sdxl_down_block_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "down_blocks.0"
    state = load_sdxl_down_block_state_from_manifest(manifest, target_prefix)
    block = build_sdxl_down_block_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real down block has no missing keys")
    _assert(not shape_report["unexpected"], "real down block has no unexpected keys")
    _assert(not shape_report["mismatched"], "real down block shapes match mapped checkpoint")
    _assert(len(block.resnets) == 2, "first SDXL down block has two resnets")
    _assert(len(block.downsamplers) == 1, "first SDXL down block has downsampler")
    _assert(tuple(block.downsamplers[0].conv.weight.shape) == (320, 320, 3, 3), "first downsampler shape")


def test_sdxl_attention_synthetic_forward_shape() -> None:
    torch.manual_seed(13)
    attention = NativeSDXLAttention(
        NativeSDXLAttentionConfig(
            query_dim=8,
            cross_attention_dim=12,
            heads=2,
            dim_head=4,
        )
    )
    hidden = torch.randn(2, 5, 8)
    context = torch.randn(2, 7, 12)
    output = attention(hidden, context)
    _assert(tuple(output.shape) == (2, 5, 8), "attention keeps query shape")
    output.sum().backward()
    _assert(attention.to_q.weight.grad is not None, "attention supports backward")


def test_sdxl_attention_flash2_cpu_fallback_shape() -> None:
    torch.manual_seed(14)
    attention = NativeSDXLAttention(
        NativeSDXLAttentionConfig(
            query_dim=8,
            cross_attention_dim=12,
            heads=2,
            dim_head=4,
            attention_backend="flash2",
        )
    )
    hidden = torch.randn(1, 4, 8)
    context = torch.randn(1, 3, 12)
    output = attention(hidden, context)
    _assert(tuple(output.shape) == (1, 4, 8), "flash2 config falls back on CPU")


def test_sdxl_attention_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "down_blocks.1.attentions.0"
    state = load_sdxl_attention_state_from_manifest(manifest, target_prefix)
    attention = build_sdxl_attention_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, attention.shape_metadata())
    _assert(not shape_report["missing"], "real attention has no missing keys")
    _assert(not shape_report["unexpected"], "real attention has no unexpected keys")
    _assert(not shape_report["mismatched"], "real attention shapes match mapped checkpoint")
    _assert(len(attention.transformer_blocks) == 2, "SDXL attention has two transformer blocks")


def test_sdxl_cross_attn_down_block_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "down_blocks.1"
    state = load_sdxl_down_block_state_from_manifest(manifest, target_prefix)
    block = build_sdxl_cross_attn_down_block_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real cross-attn down block has no missing keys")
    _assert(not shape_report["unexpected"], "real cross-attn down block has no unexpected keys")
    _assert(not shape_report["mismatched"], "real cross-attn down block shapes match mapped checkpoint")
    _assert(len(block.resnets) == 2, "cross-attn down block has two resnets")
    _assert(len(block.attentions) == 2, "cross-attn down block has two attentions")
    _assert(len(block.downsamplers) == 1, "cross-attn down block has downsampler")


def test_sdxl_cross_attn_down_block_real_weight_forward_smoke() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    block = build_sdxl_cross_attn_down_block_from_manifest(manifest, "down_blocks.1", dtype=torch.float32)
    block.train()
    hidden = torch.randn(1, 320, 4, 4, requires_grad=True)
    temb = torch.randn(1, 1280, requires_grad=True)
    encoder = torch.randn(1, 3, 2048, requires_grad=True)
    output = block(hidden, temb, encoder)
    _assert(tuple(output.shape) == (1, 640, 2, 2), "real cross-attn down block forward shape")
    output.mean().backward()
    _assert(hidden.grad is not None, "real cross-attn down block backward hidden")
    _assert(temb.grad is not None, "real cross-attn down block backward temb")
    _assert(encoder.grad is not None, "real cross-attn down block backward encoder")


def test_sdxl_mid_block_synthetic_forward_shape() -> None:
    torch.manual_seed(17)
    resnet_config = NativeSDXLResnetBlockConfig(
        in_channels=8,
        out_channels=8,
        time_embed_dim=8,
        norm_num_groups=2,
    )
    transformer_config = NativeSDXLTransformer2DConfig(
        channels=8,
        norm_num_groups=2,
        transformer_blocks=(
            NativeSDXLTransformerBlockConfig(
                dim=8,
                cross_attention_dim=12,
                heads=2,
                dim_head=4,
                ff_inner_dim=16,
            ),
        ),
    )
    block = NativeSDXLMidBlock2D(
        NativeSDXLMidBlockConfig(
            resnets=(resnet_config, resnet_config),
            attention=transformer_config,
        )
    )
    hidden = torch.randn(2, 8, 4, 4, requires_grad=True)
    temb = torch.randn(2, 8, requires_grad=True)
    encoder = torch.randn(2, 5, 12, requires_grad=True)
    output = block(hidden, temb, encoder)
    _assert(tuple(output.shape) == (2, 8, 4, 4), "synthetic mid block keeps spatial shape")
    output.mean().backward()
    _assert(hidden.grad is not None, "synthetic mid block backward hidden")
    _assert(temb.grad is not None, "synthetic mid block backward temb")
    _assert(encoder.grad is not None, "synthetic mid block backward encoder")


def test_sdxl_mid_block_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    state = load_sdxl_mid_block_state_from_manifest(manifest)
    block = build_sdxl_mid_block_from_manifest(manifest)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real mid block has no missing keys")
    _assert(not shape_report["unexpected"], "real mid block has no unexpected keys")
    _assert(not shape_report["mismatched"], "real mid block shapes match mapped checkpoint")
    _assert(len(block.resnets) == 2, "real SDXL mid block has two resnets")
    _assert(len(block.attentions) == 1, "real SDXL mid block has one attention")
    _assert(len(block.attentions[0].transformer_blocks) == 10, "real SDXL mid attention has ten transformer blocks")


def test_sdxl_up_block_synthetic_forward_shape() -> None:
    torch.manual_seed(19)
    resnet_config = NativeSDXLResnetBlockConfig(
        in_channels=12,
        out_channels=6,
        time_embed_dim=8,
        norm_num_groups=2,
        use_conv_shortcut=True,
    )
    upsampler_config = NativeSDXLUpsamplerConfig(channels=6, out_channels=6)
    block = NativeSDXLUpBlock2D(
        NativeSDXLUpBlockConfig(
            resnets=(resnet_config,),
            upsampler=upsampler_config,
        )
    )
    hidden = torch.randn(2, 4, 4, 4, requires_grad=True)
    skip = torch.randn(2, 8, 4, 4, requires_grad=True)
    temb = torch.randn(2, 8, requires_grad=True)
    output = block(hidden, [skip], temb)
    _assert(tuple(output.shape) == (2, 6, 8, 8), "synthetic up block upsamples")
    output.mean().backward()
    _assert(hidden.grad is not None, "synthetic up block backward hidden")
    _assert(skip.grad is not None, "synthetic up block backward skip")
    _assert(temb.grad is not None, "synthetic up block backward temb")


def test_sdxl_up_block_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "up_blocks.2"
    state = load_sdxl_up_block_state_from_manifest(manifest, target_prefix)
    block = build_sdxl_up_block_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real up block has no missing keys")
    _assert(not shape_report["unexpected"], "real up block has no unexpected keys")
    _assert(not shape_report["mismatched"], "real up block shapes match mapped checkpoint")
    _assert(len(block.resnets) == 3, "final SDXL up block has three resnets")
    _assert(len(block.upsamplers) == 0, "final SDXL up block has no upsampler")


def test_sdxl_cross_attn_up_block_real_weight_shape_load() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    target_prefix = "up_blocks.1"
    state = load_sdxl_up_block_state_from_manifest(manifest, target_prefix)
    block = build_sdxl_cross_attn_up_block_from_manifest(manifest, target_prefix)
    expected_shapes = {key: [int(dim) for dim in value.shape] for key, value in state.items()}
    shape_report = compare_shape_metadata(expected_shapes, block.shape_metadata())
    _assert(not shape_report["missing"], "real cross-attn up block has no missing keys")
    _assert(not shape_report["unexpected"], "real cross-attn up block has no unexpected keys")
    _assert(not shape_report["mismatched"], "real cross-attn up block shapes match mapped checkpoint")
    _assert(len(block.resnets) == 3, "cross-attn up block has three resnets")
    _assert(len(block.attentions) == 3, "cross-attn up block has three attentions")
    _assert(len(block.upsamplers) == 1, "cross-attn up block has upsampler")


def test_sdxl_unet_skeleton_synthetic_forward_shape() -> None:
    torch.manual_seed(23)
    shell = NativeSDXLShellConfig(
        in_channels=2,
        base_channels=4,
        time_embed_dim=8,
        add_embed_in_dim=6,
        add_embed_dim=8,
        out_channels=2,
        norm_num_groups=2,
    )
    resnet_4 = NativeSDXLResnetBlockConfig(
        in_channels=4,
        out_channels=4,
        time_embed_dim=8,
        norm_num_groups=2,
    )
    up_resnet = NativeSDXLResnetBlockConfig(
        in_channels=8,
        out_channels=4,
        time_embed_dim=8,
        norm_num_groups=2,
        use_conv_shortcut=True,
    )
    attention = NativeSDXLTransformer2DConfig(
        channels=4,
        norm_num_groups=2,
        transformer_blocks=(
            NativeSDXLTransformerBlockConfig(
                dim=4,
                cross_attention_dim=6,
                heads=1,
                dim_head=4,
                ff_inner_dim=8,
            ),
        ),
    )
    skeleton = NativeSDXLUNetSkeleton(
        NativeSDXLUNetSkeletonConfig(
            shell=shell,
            down_blocks=(NativeSDXLDownBlockConfig(resnets=(resnet_4, resnet_4)),),
            mid_block=NativeSDXLMidBlockConfig(resnets=(resnet_4, resnet_4), attention=attention),
            up_blocks=(NativeSDXLUpBlockConfig(resnets=(up_resnet, up_resnet, up_resnet)),),
        )
    )
    sample = torch.randn(2, 2, 8, 8, requires_grad=True)
    timestep_embedding = torch.randn(2, 4, requires_grad=True)
    add_embedding = torch.randn(2, 6, requires_grad=True)
    encoder = torch.randn(2, 3, 6, requires_grad=True)
    output = skeleton(sample, timestep_embedding, add_embedding, encoder)
    _assert(tuple(output.shape) == (2, 2, 8, 8), "synthetic UNet skeleton output shape")
    output.mean().backward()
    _assert(sample.grad is not None, "synthetic skeleton backward sample")
    _assert(timestep_embedding.grad is not None, "synthetic skeleton backward timestep")
    _assert(add_embedding.grad is not None, "synthetic skeleton backward add embedding")
    _assert(encoder.grad is not None, "synthetic skeleton backward encoder")


def _tiny_skeleton() -> NativeSDXLUNetSkeleton:
    shell = NativeSDXLShellConfig(
        in_channels=2,
        base_channels=4,
        time_embed_dim=8,
        add_time_embed_dim=2,
        add_embed_in_dim=6,
        add_embed_dim=8,
        out_channels=2,
        norm_num_groups=2,
    )
    resnet_4 = NativeSDXLResnetBlockConfig(
        in_channels=4,
        out_channels=4,
        time_embed_dim=8,
        norm_num_groups=2,
    )
    up_resnet = NativeSDXLResnetBlockConfig(
        in_channels=8,
        out_channels=4,
        time_embed_dim=8,
        norm_num_groups=2,
        use_conv_shortcut=True,
    )
    attention = NativeSDXLTransformer2DConfig(
        channels=4,
        norm_num_groups=2,
        transformer_blocks=(
            NativeSDXLTransformerBlockConfig(
                dim=4,
                cross_attention_dim=6,
                heads=1,
                dim_head=4,
                ff_inner_dim=8,
            ),
        ),
    )
    return NativeSDXLUNetSkeleton(
        NativeSDXLUNetSkeletonConfig(
            shell=shell,
            down_blocks=(NativeSDXLDownBlockConfig(resnets=(resnet_4, resnet_4)),),
            mid_block=NativeSDXLMidBlockConfig(resnets=(resnet_4, resnet_4), attention=attention),
            up_blocks=(NativeSDXLUpBlockConfig(resnets=(up_resnet, up_resnet, up_resnet)),),
        )
    )


def test_sdxl_unet_skeleton_compat_forward_shape() -> None:
    torch.manual_seed(29)
    compat = NativeSDXLUNetSkeletonCompat(_tiny_skeleton())
    sample = torch.randn(2, 2, 8, 8, requires_grad=True)
    encoder = torch.randn(2, 3, 6, requires_grad=True)
    output = compat(
        sample=sample,
        timestep=torch.tensor([1, 2]),
        encoder_hidden_states=encoder,
        added_cond_kwargs={"add_embedding": torch.randn(2, 6)},
    )
    _assert(tuple(output.sample.shape) == (2, 2, 8, 8), "compat wrapper output shape")
    output.sample.mean().backward()
    _assert(sample.grad is not None, "compat wrapper backward sample")
    _assert(encoder.grad is not None, "compat wrapper backward encoder")


def test_sdxl_unet_skeleton_compat_gradient_checkpointing_backward() -> None:
    torch.manual_seed(30)
    compat = NativeSDXLUNetSkeletonCompat(_tiny_skeleton())
    compat.train()
    compat.enable_gradient_checkpointing()
    checkpointed_modules = [
        module
        for module in compat.modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    _assert(checkpointed_modules, "compat exposes checkpointable modules")
    _assert(all(getattr(module, "gradient_checkpointing") for module in checkpointed_modules), "checkpoint flags propagate")
    sample = torch.randn(2, 2, 8, 8, requires_grad=True)
    encoder = torch.randn(2, 3, 6, requires_grad=True)
    add_embedding = torch.randn(2, 6, requires_grad=True)
    output = compat(
        sample=sample,
        timestep=torch.tensor([1, 2]),
        encoder_hidden_states=encoder,
        added_cond_kwargs={"add_embedding": add_embedding},
    )
    loss = output.sample.square().mean()
    loss.backward()
    _assert(sample.grad is not None, "checkpoint backward sample")
    _assert(encoder.grad is not None, "checkpoint backward encoder")
    _assert(add_embedding.grad is not None, "checkpoint backward add embedding")
    trainable_grads = [
        parameter.grad
        for parameter in compat.parameters()
        if parameter.requires_grad
    ]
    _assert(any(grad is not None for grad in trainable_grads), "checkpoint backward reaches parameters")
    compat.disable_gradient_checkpointing()
    _assert(
        not any(getattr(module, "gradient_checkpointing") for module in checkpointed_modules),
        "checkpoint flags disable",
    )


def test_sdxl_unet_skeleton_compat_text_time_forward_shape() -> None:
    torch.manual_seed(31)
    compat = NativeSDXLUNetSkeletonCompat(_tiny_skeleton())
    sample = torch.randn(2, 2, 8, 8, requires_grad=True)
    encoder = torch.randn(2, 3, 6, requires_grad=True)
    output = compat(
        sample=sample,
        timestep=torch.tensor([1, 2]),
        encoder_hidden_states=encoder,
        added_cond_kwargs={
            "text_embeds": torch.randn(2, 4),
            "time_ids": torch.tensor([[8.0], [16.0]]),
        },
        return_dict=False,
    )
    _assert(isinstance(output, tuple), "compat wrapper tuple return")
    _assert(tuple(output[0].shape) == (2, 2, 8, 8), "compat wrapper text_time output shape")
    output[0].mean().backward()
    _assert(sample.grad is not None, "compat wrapper text_time backward sample")
    _assert(encoder.grad is not None, "compat wrapper text_time backward encoder")


def test_sdxl_timestep_embedding_diffusers_order() -> None:
    values = torch.tensor([1.0, 2.0])
    ours = timestep_embedding(values, 4, flip_sin_to_cos=True, downscale_freq_shift=0)
    expected = torch.tensor(
        [
            [0.5403023, 0.9999500, 0.8414710, 0.0099998],
            [-0.4161468, 0.9998000, 0.9092974, 0.0199987],
        ]
    )
    torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)


def test_sdxl_unet_skeleton_real_config_metadata() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    config = build_sdxl_unet_skeleton_config_from_manifest(manifest)
    _assert(config.shell.base_channels == 320, "real skeleton shell base channels")
    _assert(len(config.down_blocks) == 3, "real skeleton has three down blocks")
    _assert(len(config.up_blocks) == 3, "real skeleton has three up blocks")
    _assert(len(config.mid_block.resnets) == 2, "real skeleton mid has two resnets")
    _assert(len(config.mid_block.attention.transformer_blocks) == 10, "real skeleton mid has ten transformer blocks")
    _assert(len(config.down_blocks[0].resnets) == 2, "real first down block has two resnets")
    _assert(len(config.up_blocks[2].resnets) == 3, "real final up block has three resnets")


def test_sdxl_native_module_factory_dispatch() -> None:
    manifest = Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"
    down = build_sdxl_native_module_from_manifest(manifest, "down_blocks.0")
    cross_down = build_sdxl_native_module_from_manifest(manifest, "down_blocks.1")
    up = build_sdxl_native_module_from_manifest(manifest, "up_blocks.2")
    _assert(isinstance(down, NativeSDXLDownBlock2D), "factory dispatches plain down block")
    _assert(hasattr(cross_down, "attentions"), "factory dispatches cross-attn down block")
    _assert(isinstance(up, NativeSDXLUpBlock2D), "factory dispatches plain up block")


def main() -> int:
    tests = [
        test_native_shadow,
        test_native_proxy_forward_parity,
        test_native_skeleton_metadata_mode,
        test_block_graph_contract,
        test_lulynx_native_preflight_profile,
        test_sdxl_shell_synthetic_forward_parity,
        test_sdxl_shell_real_weight_shape_load,
        test_sdxl_real_weight_shell_down0_forward_probe,
        test_sdxl_resnet_synthetic_forward_parity,
        test_sdxl_resnet_real_weight_shape_load,
        test_sdxl_down_block_synthetic_forward_parity,
        test_sdxl_down_block_real_weight_shape_load,
        test_sdxl_attention_synthetic_forward_shape,
        test_sdxl_attention_flash2_cpu_fallback_shape,
        test_sdxl_attention_real_weight_shape_load,
        test_sdxl_cross_attn_down_block_real_weight_shape_load,
        test_sdxl_cross_attn_down_block_real_weight_forward_smoke,
        test_sdxl_mid_block_synthetic_forward_shape,
        test_sdxl_mid_block_real_weight_shape_load,
        test_sdxl_up_block_synthetic_forward_shape,
        test_sdxl_up_block_real_weight_shape_load,
        test_sdxl_cross_attn_up_block_real_weight_shape_load,
        test_sdxl_unet_skeleton_synthetic_forward_shape,
        test_sdxl_unet_skeleton_compat_forward_shape,
        test_sdxl_unet_skeleton_compat_gradient_checkpointing_backward,
        test_sdxl_unet_skeleton_compat_text_time_forward_shape,
        test_sdxl_timestep_embedding_diffusers_order,
        test_sdxl_unet_skeleton_real_config_metadata,
        test_sdxl_native_module_factory_dispatch,
        test_weight_residency_threshold_contract,
    ]
    print("Native UNet Smoke Tests")
    print("=" * 40)
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print("=" * 40)
    print("All native UNet smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
