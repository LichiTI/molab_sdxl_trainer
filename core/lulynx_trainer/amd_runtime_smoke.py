from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.amd_runtime",
    os.path.join(_HERE, "amd_runtime.py"),
)
_AMD = importlib.util.module_from_spec(_SPEC)
sys.modules["core.lulynx_trainer.amd_runtime"] = _AMD
_SPEC.loader.exec_module(_AMD)


def test_amd_guard_forces_safe_overrides():
    cfg = SimpleNamespace(
        execution_profile_id="rocm-amd",
        schema_id="anima-lora",
        mixed_precision="bf16",
        optimizer_type="AdamW8bit",
        attention_backend="xformers",
        anima_attn_mode="xformers",
        torch_compile=True,
        dataloader_num_workers=8,
        persistent_data_loader_workers=True,
        optimizer_args="module=pytorch_optimizer",
    )
    guard = _AMD.build_amd_runtime_guard(cfg)
    assert guard.is_amd is True
    assert guard.forced_overrides["attention_backend"] == "sdpa"
    assert guard.forced_overrides["optimizer_type"] == "AdamW"
    assert guard.forced_overrides["torch_compile"] is False
    _AMD.apply_amd_runtime_guard(cfg, guard)
    assert cfg.attention_backend == "sdpa"
    assert cfg.optimizer_type == "AdamW"
    assert cfg.torch_compile is False


def test_amd_guard_rejects_flux_and_lumina_routes():
    for schema_id in ("flux-lora", "lumina-lora"):
        cfg = SimpleNamespace(execution_profile_id="rocm-amd", schema_id=schema_id)
        guard = _AMD.build_amd_runtime_guard(cfg)
        assert guard.route_supported is False
        assert "does not currently expose" in guard.route_reason


def test_amd_sage2_guard_allows_sageattn_on_gfx12():
    original_probe = _AMD.build_amd_runtime_probe
    original_module_available = _AMD._module_available
    _AMD.build_amd_runtime_probe = lambda _config: _AMD.AmdRuntimeProbe(
        runtime_id="rocm-amd-sage2",
        hip_version="7.2",
        bf16_supported=True,
        selected_gpu_name="AMD Radeon RX 9070 XT",
        selected_gpu_arch="gfx1201",
        selected_gpu_memory_mb=16384,
        gpu_count=1,
        gpu_summary="AMD Radeon RX 9070 XT [gfx1201] (16384 MiB)",
        runtime_experimental=True,
    )
    _AMD._module_available = lambda name: name == "sageattention"
    try:
        cfg = SimpleNamespace(execution_profile_id="rocm-amd-sage2", schema_id="anima-lora", attention_backend="sageattn")
        guard = _AMD.build_amd_runtime_guard(cfg)
    finally:
        _AMD.build_amd_runtime_probe = original_probe
        _AMD._module_available = original_module_available

    assert guard.is_amd is True
    assert guard.is_amd_sage2 is True
    assert guard.forced_overrides["attention_backend"] == "sageattn"
    assert guard.forced_overrides["anima_attn_mode"] == "sageattn"


def test_amd_sage2_guard_falls_back_without_supported_gfx():
    original_probe = _AMD.build_amd_runtime_probe
    original_module_available = _AMD._module_available
    _AMD.build_amd_runtime_probe = lambda _config: _AMD.AmdRuntimeProbe(
        runtime_id="rocm-amd-sage2",
        hip_version="7.2",
        bf16_supported=True,
        selected_gpu_name="AMD Radeon RX 6800 XT",
        selected_gpu_arch="gfx1030",
        selected_gpu_memory_mb=16384,
        gpu_count=1,
        gpu_summary="AMD Radeon RX 6800 XT [gfx1030] (16384 MiB)",
        runtime_experimental=True,
    )
    _AMD._module_available = lambda name: name == "sageattention"
    try:
        cfg = SimpleNamespace(execution_profile_id="rocm-amd-sage2", schema_id="anima-lora", attention_backend="sageattn")
        guard = _AMD.build_amd_runtime_guard(cfg)
    finally:
        _AMD.build_amd_runtime_probe = original_probe
        _AMD._module_available = original_module_available

    assert guard.forced_overrides["attention_backend"] == "sdpa"
    assert "gfx1030" in guard.disabled_features["attention_backend"]


def test_sdpa_slice_estimator_only_triggers_above_threshold():
    q = torch.zeros((1, 16, 2048, 64), dtype=torch.float16)
    chunk_count = _AMD.estimate_sdpa_chunk_count(
        q,
        trigger_gb=0.05,
        target_gb=0.02,
    )
    assert chunk_count >= 2
    assert _AMD.estimate_sdpa_chunk_count(q[:, :, :64], trigger_gb=1.0, target_gb=0.5) == 0


if __name__ == "__main__":
    test_amd_guard_forces_safe_overrides()
    test_amd_guard_rejects_flux_and_lumina_routes()
    test_amd_sage2_guard_allows_sageattn_on_gfx12()
    test_amd_sage2_guard_falls_back_without_supported_gfx()
    test_sdpa_slice_estimator_only_triggers_above_threshold()
    print("All AMD runtime smoke tests passed.")
