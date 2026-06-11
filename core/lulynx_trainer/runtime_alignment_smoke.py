# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for runtime_optimizations alignment (Phase 9.3).

Verifies:
  1. RuntimeOptimizationPlan picks up anima_compile_scope from config
  2. Plan log_lines() surfaces all relevant fields and warnings
  3. Auto attention selection respects xformers / sdpa / use_sdpa flags
  4. Explicit flash2 falls back to sdpa when the package is missing
  5. torch.compile + blocks_to_swap incompatibility is auto-resolved
  6. FlexAttention only stays enabled inside the FlexAttention runtime
"""

from __future__ import annotations

import sys
import os
import importlib.util
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.runtime_optimizations",
    os.path.join(_HERE, "runtime_optimizations.py"),
)
_ro = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.runtime_optimizations"] = _ro
_spec.loader.exec_module(_ro)

from core.lulynx_trainer import diffusers_attention as _da


def test_plan_picks_up_anima_compile_scope():
    cfg = SimpleNamespace(
        attention_backend="sdpa",
        sdpa_backend_policy="cutlass",
        torch_compile=False,
        anima_compile_scope="full_cudagraph",
    )
    plan = _ro.build_runtime_optimization_plan(cfg)
    assert plan.anima_compile_scope == "full_cudagraph"
    assert plan.sdpa_backend_policy == "cutlass"
    log = list(plan.log_lines())
    assert any("anima_compile_scope=full_cudagraph" in line for line in log)
    assert any("sdpa_backend=cutlass" in line for line in log)
    print("PASS: anima_compile_scope propagates to plan + log")


def test_auto_attention_selects_xformers_when_xformers_true():
    cfg = SimpleNamespace(attention_backend="auto", xformers=True)
    plan = _ro.build_runtime_optimization_plan(cfg)
    # If xformers package is installed, attention_backend stays xformers;
    # otherwise it falls back to sdpa with a warning.  Either way the
    # plan must reflect a coherent state.
    assert plan.attention_backend in {"xformers", "sdpa"}
    print(f"PASS: auto attention picked '{plan.attention_backend}' for xformers=True")


def test_auto_attention_selects_sdpa_when_use_sdpa_true():
    cfg = SimpleNamespace(attention_backend="auto", use_sdpa=True)
    plan = _ro.build_runtime_optimization_plan(cfg)
    assert plan.attention_backend == "sdpa"
    assert any("sdpa" in r.lower() for r in plan.reasons)
    print("PASS: auto attention falls back to sdpa when use_sdpa=True")


def test_unknown_attention_falls_back_to_torch():
    cfg = SimpleNamespace(attention_backend="garbage_value")
    plan = _ro.build_runtime_optimization_plan(cfg)
    # Unknown aliases normalise to auto. The default route is SDXL/U-Net, whose
    # auto policy stays conservative unless the user explicitly selects flash2.
    assert plan.attention_backend == "sdpa"
    print("PASS: unknown attention backend resolved to SDXL/U-Net sdpa auto policy")


def test_flash2_falls_back_to_sdpa_when_package_missing():
    previous = _ro._importable
    try:
        _ro._importable = lambda module_name: False if module_name == "flash_attn" else previous(module_name)
        cfg = SimpleNamespace(attention_backend="flash2")
        plan = _ro.build_runtime_optimization_plan(cfg)
    finally:
        _ro._importable = previous
    assert plan.attention_backend == "sdpa"
    assert any("flash2" in w.lower() for w in plan.warnings)
    assert any("falling back to sdpa" in w.lower() for w in plan.warnings)
    print("PASS: explicit flash2 falls back to sdpa when package missing")


def test_sdxl_flash2_installs_generic_diffusers_processor():
    try:
        import torch
        from diffusers.models.attention_processor import Attention
    except Exception as exc:
        print(f"SKIP: diffusers Attention smoke unavailable: {exc}")
        return

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)

    def fake_flash_attn(q, k, v, **_kwargs):
        return torch.zeros_like(q)

    previous_create = _da.Flash2DiffusersAttentionKernel.__dict__["create"]
    try:
        _da.Flash2DiffusersAttentionKernel.create = classmethod(lambda cls: cls(fake_flash_attn))
        model = SimpleNamespace(unet=TinyUnet(), vae=None, model_arch="sdxl")
        plan = _ro.RuntimeOptimizationPlan(attention_backend="flash2", requested_attention_backend="flash2")
        _ro.apply_attention_backend(model, plan)
    finally:
        _da.Flash2DiffusersAttentionKernel.create = previous_create

    assert isinstance(model.unet.attn.processor, _da.GenericDiffusersAttnProcessor)
    assert model.unet.attn.processor.backend_id == "flash2"
    assert plan.attention_backend == "flash2"
    assert any("flash2" in reason.lower() and "generic diffusers" in reason.lower() for reason in plan.reasons)
    print("PASS: SDXL flash2 installs the generic diffusers attention processor")


def test_flexattn_requires_flexattention_runtime():
    old_env = os.environ.pop("LULYNX_FLEXATTENTION_STARTUP", None)
    try:
        cfg = SimpleNamespace(attention_backend="flexattn", runtime_id="standard")
        plan = _ro.build_runtime_optimization_plan(cfg)
    finally:
        if old_env is not None:
            os.environ["LULYNX_FLEXATTENTION_STARTUP"] = old_env
    assert plan.attention_backend == "sdpa"
    assert any("outside the FlexAttention runtime" in w for w in plan.warnings)
    print("PASS: flexattn falls back outside the FlexAttention runtime")


def test_flexattn_allowed_in_flexattention_runtime_when_api_available():
    previous = _ro._flex_attention_available
    try:
        _ro._flex_attention_available = lambda: True
        cfg = SimpleNamespace(attention_backend="flexattention", runtime_id="flexattention")
        plan = _ro.build_runtime_optimization_plan(cfg)
    finally:
        _ro._flex_attention_available = previous
    assert plan.attention_backend == "flexattn"
    print("PASS: flexattn remains active inside the FlexAttention runtime when API is available")


def test_torch_compile_disables_block_swap():
    cfg = SimpleNamespace(
        attention_backend="sdpa",
        torch_compile=True,
        blocks_to_swap=4,
    )
    plan = _ro.build_runtime_optimization_plan(cfg)
    assert plan.torch_compile == True
    assert getattr(cfg, "blocks_to_swap", 0) == 0
    assert any("blocks_to_swap" in w.lower() for w in plan.warnings)
    print("PASS: torch.compile + blocks_to_swap auto-resolves blocks_to_swap=0")


def test_log_lines_includes_warnings_and_reasons():
    cfg = SimpleNamespace(
        attention_backend="auto",
        xformers=True,
        torch_compile=True,
    )
    plan = _ro.build_runtime_optimization_plan(cfg)
    plan.warnings.append("test warning")
    plan.reasons.append("test reason")
    log = list(plan.log_lines())
    assert any("test warning" in line for line in log)
    assert any("test reason" in line for line in log)
    print("PASS: log_lines surfaces all warnings and reasons")


def test_auto_attention_route_policy_when_no_overrides():
    """U-Net auto is conservative; DiT auto keeps the FA2-first policy."""
    sdxl_cfg = SimpleNamespace(attention_backend="auto", model_type="sdxl")
    sdxl_plan = _ro.build_runtime_optimization_plan(sdxl_cfg)
    assert sdxl_plan.attention_backend == "sdpa"
    assert any("u-net" in r.lower() or "sdxl" in r.lower() for r in sdxl_plan.reasons)

    cfg = SimpleNamespace(attention_backend="auto", model_type="anima")
    plan = _ro.build_runtime_optimization_plan(cfg)
    if _ro._importable("flash_attn"):
        assert plan.attention_backend == "flash2"
    elif _ro._importable("sageattention"):
        assert plan.attention_backend == "sageattn"
    else:
        assert plan.attention_backend == "sdpa"
    # Reason line must mention the FA2-first policy or its fallback chain
    assert any("FA2" in r or "flash2" in r or "sageattn" in r or "sdpa" in r for r in plan.reasons)
    print(f"PASS: route-aware auto policy resolved SDXL -> sdpa and Anima -> '{plan.attention_backend}'")


def test_sdpa_backend_policy_context_builds_cleanly():
    cfg = SimpleNamespace(attention_backend="sdpa", sdpa_backend_policy="cutlass")
    plan = _ro.build_runtime_optimization_plan(cfg)
    ctx = _ro.build_sdpa_backend_context(plan)
    assert ctx is not None
    assert plan.sdpa_backend_policy == "cutlass"
    print("PASS: sdpa backend policy context builds cleanly")


if __name__ == "__main__":
    test_plan_picks_up_anima_compile_scope()
    test_auto_attention_selects_xformers_when_xformers_true()
    test_auto_attention_selects_sdpa_when_use_sdpa_true()
    test_unknown_attention_falls_back_to_torch()
    test_flash2_falls_back_to_sdpa_when_package_missing()
    test_sdxl_flash2_installs_generic_diffusers_processor()
    test_flexattn_requires_flexattention_runtime()
    test_flexattn_allowed_in_flexattention_runtime_when_api_available()
    test_torch_compile_disables_block_swap()
    test_log_lines_includes_warnings_and_reasons()
    test_auto_attention_route_policy_when_no_overrides()
    test_sdpa_backend_policy_context_builds_cleanly()
    print("\nAll runtime alignment smoke tests passed!")
