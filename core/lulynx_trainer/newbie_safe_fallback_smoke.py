# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie safe-fallback cleanup on forward OOM."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.training_loop import TrainingLoop


class _FallbackNewbieUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = 0

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del timestep, encoder_hidden_states, added_cond_kwargs
        self.forward_calls += 1
        if self.forward_calls == 1:
            raise RuntimeError("CUDA out of memory: synthetic Newbie safe_fallback smoke")
        return SimpleNamespace(sample=sample * self.scale)


@contextmanager
def _fake_autocast(*, device_type: str, dtype: torch.dtype):
    del device_type, dtype
    yield


def main() -> int:
    model = _FallbackNewbieUnet()
    optimizer = torch.optim.SGD([model.scale], lr=1e-3)
    empty_cache_calls: list[str] = []

    loop = TrainingLoop(
        unet=model,
        text_encoder_1=nn.Identity(),
        text_encoder_2=nn.Identity(),
        vae=nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        lora_injector=SimpleNamespace(get_trainable_params=lambda: [model.scale]),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        safe_fallback=True,
        model_arch="newbie",
    )

    batch = {
        "latents": torch.randn(1, 16, 4, 4, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(1, 4, 16, dtype=torch.float32),
        "pooled_prompt_embeds": torch.randn(1, 16, dtype=torch.float32),
        "captions": ["newbie safe fallback"],
    }

    loop._cudagraph_active = True
    loop._cudagraph_capture = object()

    with patch("torch.autocast", side_effect=_fake_autocast):
        with patch("torch.cuda.empty_cache", side_effect=lambda: empty_cache_calls.append("called")):
            try:
                loop.train_step(batch)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise AssertionError(f"Expected OOM RuntimeError, got {exc!r}") from exc
            else:
                raise AssertionError("Expected safe_fallback to fail safely on synthetic OOM")

    if model.forward_calls != 1:
        raise AssertionError(f"Expected no CPU retry after synthetic OOM, got {model.forward_calls} forward calls")
    if empty_cache_calls != ["called"]:
        raise AssertionError(f"Expected one cache clear before safe failure, got {empty_cache_calls}")
    if loop._cudagraph_active:
        raise AssertionError("Expected OOM cleanup to disable active CUDAGraph replay")
    if loop._cudagraph_capture is not None:
        raise AssertionError("Expected OOM cleanup to drop pending CUDAGraph capture")
    if model.scale.grad is not None:
        raise AssertionError("Expected no gradient after safe_fallback OOM failure")

    print("Newbie safe_fallback smoke passed: synthetic OOM cleared CUDA state and failed safely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
