# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test dim_from_weights rank inference and config application."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.dim_from_weights import (
    apply_dim_from_weights,
    infer_rank_from_weights,
    resolve_dim_from_weights_path,
)


def _state(rank: int) -> dict[str, torch.Tensor]:
    return {
        "unet_down_blocks_0_attentions_0_to_q.lora_down.weight": torch.randn(rank, 8),
        "unet_down_blocks_0_attentions_0_to_q.lora_up.weight": torch.randn(8, rank),
        "te1_text_model_encoder_layers_0_self_attn_q_proj.lora_A.weight": torch.randn(rank, 6),
        "te1_text_model_encoder_layers_0_self_attn_q_proj.lora_B.weight": torch.randn(6, rank),
    }


def test_infer_rank_from_safetensors() -> None:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter.safetensors"
        save_file(_state(12), str(path))
        rank, per_layer = infer_rank_from_weights(str(path), default_rank=4)

    assert rank == 12
    assert len(per_layer) == 2


def test_infer_rank_from_torch_state_dict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter.pt"
        torch.save({"state_dict": _state(10)}, path)
        rank, per_layer = infer_rank_from_weights(str(path), default_rank=4)

    assert rank == 10
    assert len(per_layer) == 2


def test_apply_dim_from_weights_updates_sdxl_config_rank() -> None:
    from safetensors.torch import save_file

    logs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter.safetensors"
        save_file(_state(14), str(path))
        cfg = SimpleNamespace(
            dim_from_weights=True,
            network_dim=4,
            network_weights_path=str(path),
            anima_dit_adapter_path="",
        )
        resolution = apply_dim_from_weights(cfg, model_arch="sdxl", log_fn=logs.append)

    assert resolution.enabled is True
    assert resolution.applied is True
    assert resolution.original_rank == 4
    assert resolution.inferred_rank == 14
    assert cfg.network_dim == 14
    assert any("inferred rank=14" in line for line in logs)


def test_apply_dim_from_weights_uses_anima_adapter_fallback() -> None:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "anima_dit_adapter.safetensors"
        save_file(_state(6), str(path))
        cfg = SimpleNamespace(
            dim_from_weights=True,
            network_dim=4,
            network_weights_path="",
            anima_dit_adapter_path=str(path),
        )
        assert resolve_dim_from_weights_path(cfg, "anima") == str(path)
        resolution = apply_dim_from_weights(cfg, model_arch="anima")

    assert resolution.applied is True
    assert cfg.network_dim == 6


def main() -> int:
    test_infer_rank_from_safetensors()
    test_infer_rank_from_torch_state_dict()
    test_apply_dim_from_weights_updates_sdxl_config_rank()
    test_apply_dim_from_weights_uses_anima_adapter_fallback()
    print("dim_from_weights_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
