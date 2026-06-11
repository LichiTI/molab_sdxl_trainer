# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test shared runtime features: cpu_offload_checkpointing, vram_swap_to_ram,
experimental_attention_profile, cross_attn_fused_kv, and blockwise_fused_optimizers.

Tests the Warehouse implementations without requiring real models or GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn

# ── Direct file imports bypassing __init__.py ────────────────────────────
_here = Path(__file__).resolve().parent


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


_ensure_namespace("core", _here.parent)
_ensure_namespace("core.lulynx_trainer", _here)


def _import_from_file(module_name: str, file_path: Path):
    """Import a single file without triggering the package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mem_opt = _import_from_file(
    "core.lulynx_trainer.memory_optimizations",
    _here / "memory_optimizations.py",
)
_rt_opt = _import_from_file(
    "core.lulynx_trainer.runtime_optimizations",
    _here / "runtime_optimizations.py",
)

# ConfigAdapter depends on .config which depends on ..configs which depends on ..constants
_constants = _import_from_file(
    "core.constants",
    _here.parent / "constants.py",
)
_configs = _import_from_file(
    "core.configs",
    _here.parent / "configs.py",
)
# lulynx_trainer/config.py is a facade that re-exports from core.configs
_facade_config = _import_from_file(
    "core.lulynx_trainer.config",
    _here / "config.py",
)
_config_adapter = _import_from_file(
    "core.lulynx_trainer.config_adapter",
    _here / "config_adapter.py",
)

AdapterCPUResidency = _mem_opt.AdapterCPUResidency
cpu_offload_checkpoint = _mem_opt.cpu_offload_checkpoint
AttentionProfile = _rt_opt.AttentionProfile
FusedKVProjection = _rt_opt.FusedKVProjection
FusedOptimizerConfig = _rt_opt.FusedOptimizerConfig
RuntimeOptimizationPlan = _rt_opt.RuntimeOptimizationPlan
apply_attention_profile = _rt_opt.apply_attention_profile
apply_blockwise_fused_optimizers = _rt_opt.apply_blockwise_fused_optimizers
apply_cross_attn_fused_kv = _rt_opt.apply_cross_attn_fused_kv
build_runtime_optimization_plan = _rt_opt.build_runtime_optimization_plan
sliding_window_attention = _rt_opt.sliding_window_attention
UnifiedTrainingConfig = _configs.UnifiedTrainingConfig
ConfigAdapter = _config_adapter.ConfigAdapter


# ── Helpers ──────────────────────────────────────────────────────────────

class _SimpleModel(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.attn2 = SimpleNamespace(
            to_k=nn.Linear(dim, dim),
            to_v=nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Tests ────────────────────────────────────────────────────────────────

def test_cpu_offload_checkpointing() -> None:
    model = _SimpleModel(dim=8)
    x = torch.randn(2, 8, requires_grad=True)

    def forward_fn(inp):
        return model(inp)

    result = cpu_offload_checkpoint(forward_fn, x)
    assert result.shape == (2, 8), f"Expected shape (2, 8), got {result.shape}"
    assert result.requires_grad, "Result should require grad"

    loss = result.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient should be computed"
    print("  [PASS] cpu_offload_checkpointing: forward + backward")


def test_cpu_offload_checkpoint_kwargs_only() -> None:
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 4, requires_grad=True)

    def forward_fn(*, sample):
        return layer(sample)

    result = cpu_offload_checkpoint(forward_fn, sample=x)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"
    loss = result.sum()
    loss.backward()
    assert layer.weight.grad is not None, "Layer gradient should be computed"
    print("  [PASS] cpu_offload_checkpointing: kwargs-only closure")


def test_cpu_offload_checkpoint_config_fields() -> None:
    config = UnifiedTrainingConfig()
    assert config.cpu_offload_checkpointing is False
    config2 = UnifiedTrainingConfig(cpu_offload_checkpointing=True)
    assert config2.cpu_offload_checkpointing is True
    print("  [PASS] cpu_offload_checkpointing: config field OK")


def test_adapter_cpu_residency() -> None:
    residency = AdapterCPUResidency(device="cpu")
    p1 = nn.Parameter(torch.randn(10, 10))
    p2 = nn.Parameter(torch.randn(5, 5))
    registered = residency.register_parameters([p1, p2])
    assert registered == 2
    assert residency.managed_param_count == 2

    with residency.step_context():
        assert residency.is_active
    assert not residency.is_active

    savings = residency.estimate_vram_savings_mb()
    assert savings >= 0

    residency.cleanup()
    assert residency.managed_param_count == 0
    print("  [PASS] adapter_cpu_residency: registration + step context + estimation")


def test_vram_swap_to_ram_config_fields() -> None:
    config = UnifiedTrainingConfig()
    assert config.vram_swap_to_ram is False
    config2 = UnifiedTrainingConfig(vram_swap_to_ram=True)
    assert config2.vram_swap_to_ram is True
    print("  [PASS] vram_swap_to_ram: config field OK")


def test_attention_profile() -> None:
    config = SimpleNamespace(
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=64,
        attention_backend="sdpa",
    )
    profile = AttentionProfile.from_config(config)
    assert profile.enabled is True
    assert profile.window_size == 64
    assert profile.launcher_attention_backend == "sdpa"
    assert profile.is_active is True

    config2 = SimpleNamespace(
        experimental_attention_profile_enabled=False,
        experimental_attention_profile_window=0,
    )
    profile2 = AttentionProfile.from_config(config2)
    assert profile2.is_active is False

    batch, heads, seq_len, dim = 1, 2, 32, 8
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)
    out = sliding_window_attention(q, k, v, window_size=8)
    assert out.shape == (batch, heads, seq_len, dim)
    print("  [PASS] experimental_attention_profile: profile + sliding window")


def test_attention_profile_config_fields() -> None:
    config = UnifiedTrainingConfig()
    assert config.experimental_attention_profile_enabled is False
    assert config.experimental_attention_profile_window == 0
    assert config.experimental_attention_profile_backend == "auto"
    assert config.experimental_attention_profile_torch_max_tokens == 2048
    config2 = UnifiedTrainingConfig(
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=128,
        experimental_attention_profile_backend="sdpa_masked",
    )
    assert config2.experimental_attention_profile_enabled is True
    assert config2.experimental_attention_profile_window == 128
    assert config2.experimental_attention_profile_backend == "sdpa_masked"
    print("  [PASS] experimental_attention_profile: config fields OK")


def test_cross_attn_fused_kv() -> None:
    dim = 16
    fused = FusedKVProjection(embed_dim=dim, kv_dim=dim)
    x = torch.randn(2, 4, dim)
    k, v = fused(x)
    assert k.shape == (2, 4, dim)
    assert v.shape == (2, 4, dim)

    model = nn.Module()
    block = _CrossAttnBlock(dim=dim)
    model.blocks = nn.ModuleList([block])
    config = SimpleNamespace(cross_attn_fused_kv=True)
    plan = build_runtime_optimization_plan(SimpleNamespace(attention_backend="torch"))
    apply_cross_attn_fused_kv(config, model, plan)

    assert hasattr(block.attn2, "_fused_kv")
    assert isinstance(block.attn2._fused_kv, FusedKVProjection)

    with torch.no_grad():
        orig_k = block.attn2.to_k.weight.clone()
        orig_v = block.attn2.to_v.weight.clone()
        fused_w = block.attn2._fused_kv.kv_proj.weight
        assert torch.allclose(fused_w[:dim], orig_k)
        assert torch.allclose(fused_w[dim:], orig_v)
    print("  [PASS] cross_attn_fused_kv: FusedKVProjection + apply")


def test_cross_attn_fused_kv_config_fields() -> None:
    config = UnifiedTrainingConfig()
    assert config.cross_attn_fused_kv is False
    config2 = UnifiedTrainingConfig(cross_attn_fused_kv=True)
    assert config2.cross_attn_fused_kv is True
    print("  [PASS] cross_attn_fused_kv: config field OK")


def test_blockwise_fused_optimizers() -> None:
    config = SimpleNamespace(blockwise_fused_optimizers=True)
    fuse_cfg = FusedOptimizerConfig.from_config(config)
    assert fuse_cfg.enabled is True

    model = _SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    plan = build_runtime_optimization_plan(SimpleNamespace(attention_backend="torch"))
    result = apply_blockwise_fused_optimizers(config, optimizer, plan)
    assert result is True
    assert getattr(optimizer, "_blockwise_fused", False) is True

    config2 = SimpleNamespace(blockwise_fused_optimizers=False)
    result2 = apply_blockwise_fused_optimizers(config2, optimizer, plan)
    assert result2 is False
    print("  [PASS] blockwise_fused_optimizers: config + optimizer marking")


def test_blockwise_fused_optimizers_config_fields() -> None:
    config = UnifiedTrainingConfig()
    assert config.blockwise_fused_optimizers is False
    config2 = UnifiedTrainingConfig(blockwise_fused_optimizers=True)
    assert config2.blockwise_fused_optimizers is True
    print("  [PASS] blockwise_fused_optimizers: config field OK")


def test_runtime_plan_integration() -> None:
    config = SimpleNamespace(
        attention_backend="flash2",
        torch_compile=False,
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=32,
        experimental_attention_profile_backend="torch_fallback",
    )
    plan = build_runtime_optimization_plan(config)
    model = _SimpleModel()
    apply_attention_profile(config, model, plan)
    assert hasattr(model, "_attention_profile")
    assert model._attention_profile.window_size == 32
    assert model._attention_profile.backend == "torch_fallback"
    assert model._attention_profile.launcher_attention_backend == plan.attention_backend
    print("  [PASS] runtime_plan_integration: attention profile applied via plan")


def test_runtime_plan_auto_attention_is_route_aware() -> None:
    sdxl = build_runtime_optimization_plan(SimpleNamespace(model_type="sdxl", attention_backend="auto", xformers=False, sdpa=False))
    assert sdxl.attention_backend == "sdpa"
    assert any("U-Net/SDXL" in reason for reason in sdxl.reasons)

    newbie = build_runtime_optimization_plan(SimpleNamespace(model_type="newbie", attention_backend="auto", sdpa=True))
    assert newbie.attention_backend == "sdpa"
    assert any("newbie" in reason for reason in newbie.reasons)
    print("  [PASS] runtime_plan_auto_attention_is_route_aware: SDXL avoids DiT backends, Newbie uses DiT policy")


def test_config_adapter_normalization() -> None:
    parsed = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
        "cpu_offload_checkpointing": "true",
        "vram_swap_to_ram": "1",
        "experimental_attention_profile_enabled": "yes",
        "experimental_attention_profile_window": 64,
        "cross_attn_fused_kv": "on",
        "blockwise_fused_optimizers": "enabled",
    })
    assert parsed.cpu_offload_checkpointing is True
    assert parsed.vram_swap_to_ram is True
    assert parsed.experimental_attention_profile_enabled is True
    assert parsed.experimental_attention_profile_window == 64
    assert parsed.cross_attn_fused_kv is True
    assert parsed.blockwise_fused_optimizers is True

    parsed2 = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
        "cpu_offload_checkpointing": "0",
        "vram_swap_to_ram": "false",
        "cross_attn_fused_kv": "disabled",
    })
    assert parsed2.cpu_offload_checkpointing is False
    assert parsed2.vram_swap_to_ram is False
    assert parsed2.cross_attn_fused_kv is False
    print("  [PASS] config_adapter_normalization: all fields normalized")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_cpu_offload_checkpointing,
        test_cpu_offload_checkpoint_kwargs_only,
        test_cpu_offload_checkpoint_config_fields,
        test_adapter_cpu_residency,
        test_vram_swap_to_ram_config_fields,
        test_attention_profile,
        test_attention_profile_config_fields,
        test_cross_attn_fused_kv,
        test_cross_attn_fused_kv_config_fields,
        test_blockwise_fused_optimizers,
        test_blockwise_fused_optimizers_config_fields,
        test_runtime_plan_integration,
        test_runtime_plan_auto_attention_is_route_aware,
        test_config_adapter_normalization,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\nShared runtime smoke: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

