# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test runtime compile and env hint plumbing without loading models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.runtime_optimizations import (
    RuntimeOptimizationPlan,
    apply_torch_compile_if_requested,
    build_runtime_optimization_plan,
)
from core.lulynx_trainer.trainer import LulynxTrainer


class _Module(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def main() -> int:
    parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "dynamo_backend": "eager",
            "torch_compile": True,
            "pytorch_cuda_expandable_segments": True,
        }
    )
    assert parsed.torch_compile is True
    assert parsed.torch_compile_backend == "eager"

    runtime_parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "compile_runtime": "compile_cache",
        }
    )
    assert runtime_parsed.compile_runtime == "compile_cache"
    assert runtime_parsed.torch_compile is True
    assert runtime_parsed.compile_cache_enabled is True
    assert runtime_parsed.compile_shape_strategy == "fixed_pad"
    assert runtime_parsed.compile_target_strategy == "block"
    assert runtime_parsed.compile_contract_strict is True
    assert runtime_parsed.compile_static_shape_drop_last is True

    strategy_parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "pretrained_model_name_or_path": "H:/models/anima",
            "compile_runtime": "cache",
            "compile_shape_strategy": "tokenflatten",
            "compile_target_strategy": "forward_impl",
        }
    )
    assert strategy_parsed.compile_runtime == "compile_cache"
    assert strategy_parsed.compile_shape_strategy == "token_flatten"
    assert strategy_parsed.compile_target_strategy == "inner_forward"

    auto_compile_parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/models/newbie",
            "compile_runtime": "auto",
        }
    )
    assert auto_compile_parsed.compile_runtime == "compile_cache"
    assert auto_compile_parsed.compile_shape_strategy == "token_flatten"
    assert auto_compile_parsed.compile_target_strategy == "inner_forward"
    assert auto_compile_parsed.native_token_bucket_compile is True

    auto_shape_cfg = SimpleNamespace(
        model_type="anima",
        attention_backend="auto",
        sdpa_backend_policy="cutlass",
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        torch_compile_dynamic=False,
        torch_compile_fullgraph=False,
        torch_compile_scope="per_block",
        torch_compile_allow_full_with_per_block=False,
        anima_compile_scope="per_block",
        compile_shape_strategy="auto",
        compile_target_strategy="auto",
        native_token_bucket_compile=True,
        anima_cached_training=True,
        anima_fixed_visual_tokens=0,
        anima_fixed_text_tokens=256,
        device="cpu",
        swap_granularity="off",
        swap_count=0,
        swap_ratio=0.0,
        blocks_to_swap=0,
    )
    auto_plan = build_runtime_optimization_plan(auto_shape_cfg)
    assert auto_plan.compile_shape_strategy == "token_flatten"

    cudagraph_parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "compile_runtime": "compile_cudagraph",
        }
    )
    assert cudagraph_parsed.torch_compile is True
    assert cudagraph_parsed.torch_compile_backend == "cudagraphs"
    assert cudagraph_parsed.compile_cache_enabled is True

    explicit_compile_wins = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "compile_runtime": "off",
            "torch_compile": True,
            "dynamo_backend": "eager",
        }
    )
    assert explicit_compile_wins.torch_compile is True
    assert explicit_compile_wins.torch_compile_backend == "eager"

    token_flatten_cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "pretrained_model_name_or_path": "H:/models/anima",
            "torch_compile": True,
            "compile_shape_strategy": "token_flatten",
        }
    )
    trainer_token_flatten = object.__new__(LulynxTrainer)
    trainer_token_flatten.config = token_flatten_cfg
    trainer_token_flatten.runtime_optimization_plan = SimpleNamespace(compile_shape_strategy="token_flatten")
    assert trainer_token_flatten._should_enable_fixed_token_padding() is False

    trainer = object.__new__(LulynxTrainer)
    trainer.config = parsed
    trainer.runtime_optimization_plan = SimpleNamespace(compile_shape_strategy="auto")
    logs: list[str] = []
    trainer._log = logs.append

    old_value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        LulynxTrainer._apply_runtime_env_hints(trainer)
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True,max_split_size_mb:64"
    finally:
        if old_value is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old_value

    module = _Module()
    plan = RuntimeOptimizationPlan(
        attention_backend="torch",
        requested_attention_backend="torch",
        torch_compile=True,
        torch_compile_backend="eager",
        torch_compile_mode="default",
        compile_shape_strategy="token_flatten",
        compile_target_strategy="inner_forward",
    )
    compiled = object()
    with patch("torch.compile", return_value=compiled) as mock_compile:
        result = apply_torch_compile_if_requested(module, plan, label="UNet")
    assert result is compiled
    mock_compile.assert_called_once()

    print("Runtime compile smoke passed: env hints and torch.compile path are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
