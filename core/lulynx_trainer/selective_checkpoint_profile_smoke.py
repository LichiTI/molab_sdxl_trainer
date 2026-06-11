"""Smoke checks for selective checkpoint API profiling.

This is intentionally a helper/profile smoke, not a trainer integration test.
It verifies that PyTorch SAC can be detected and exercised in the current
environment, and that Anima/Newbie native DiT routes expose identifiable
block-level candidate entrypoints.
"""

from __future__ import annotations

import functools
from pathlib import Path
import sys

import torch

try:
    from .checkpoint_policy import (
        build_selective_checkpoint_context_fn,
        profile_selective_checkpoint_route,
        selective_checkpoint_api_profile,
    )
    from .checkpoint_policy import resolve_checkpoint_policy
except ImportError:  # pragma: no cover - direct script usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.core.lulynx_trainer.checkpoint_policy import (
        build_selective_checkpoint_context_fn,
        profile_selective_checkpoint_route,
        resolve_checkpoint_policy,
        selective_checkpoint_api_profile,
    )


class _TinyAnimaTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Module()
        self.net.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self.anima_block_checkpointing_mode = "block"

    def _checkpoint_block(self, block, x, emb, context, adaln_lora=None):
        return block(x)

    def _run_blocks(self, x, emb, context, adaln_lora=None):
        for block in self.net.blocks:
            x = self._checkpoint_block(block, x, emb, context, adaln_lora)
        return x


class _TinyNewbieTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._block_modules = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self._newbie_block_checkpointing_mode = "block"

    def _run_dit_block(self, block, x, t_emb):
        return block(x)


def _exercise_selective_checkpoint_context() -> None:
    import torch.utils.checkpoint as checkpoint_mod

    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)
    context_fn = functools.partial(
        checkpoint_mod.create_selective_checkpoint_contexts,
        [torch.ops.aten.mm.default],
    )

    def fn(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.mm(left, right)).sum()

    out = checkpoint_mod.checkpoint(
        fn,
        x,
        y,
        use_reentrant=False,
        preserve_rng_state=False,
        context_fn=context_fn,
    )
    out.backward()
    assert x.grad is not None
    assert y.grad is not None


def _exercise_lulynx_selective_context() -> None:
    import torch.utils.checkpoint as checkpoint_mod

    context_fn = build_selective_checkpoint_context_fn("balanced")
    assert context_fn is not None

    x = torch.randn(2, 4, requires_grad=True)
    layer = torch.nn.Linear(4, 4)

    def fn(value: torch.Tensor) -> torch.Tensor:
        return torch.relu(layer(value)).sum()

    out = checkpoint_mod.checkpoint(
        fn,
        x,
        use_reentrant=False,
        preserve_rng_state=False,
        context_fn=context_fn,
    )
    out.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None


def main() -> None:
    api = selective_checkpoint_api_profile()
    assert api["available"], api
    assert api["has_checkpoint_context_fn"], api
    assert "MUST_SAVE" in api["checkpoint_policy_values"], api
    _exercise_selective_checkpoint_context()
    _exercise_lulynx_selective_context()

    decision = resolve_checkpoint_policy(
        type("Cfg", (), {"checkpoint_policy": "selective", "gradient_checkpointing": False, "cpu_offload_checkpointing": False})(),
        route="anima",
        cuda_available=True,
    )
    assert decision.effective_policy == "selective", decision.as_dict()
    assert decision.gradient_checkpointing is False, decision.as_dict()

    anima = profile_selective_checkpoint_route("anima", _TinyAnimaTarget())
    assert anima.route_supported is True, anima.as_dict()
    assert anima.block_count == 2, anima.as_dict()
    assert "_checkpoint_block" in anima.candidate_entrypoint, anima.as_dict()
    assert anima.forward_wired is True, anima.as_dict()
    assert anima.wiring_state == "experimental_live", anima.as_dict()
    assert not anima.fallback_reason, anima.as_dict()

    newbie = profile_selective_checkpoint_route("newbie", _TinyNewbieTarget())
    assert newbie.route_supported is True, newbie.as_dict()
    assert newbie.block_count == 1, newbie.as_dict()
    assert "_run_dit_block" in newbie.candidate_entrypoint, newbie.as_dict()
    assert newbie.forward_wired is True, newbie.as_dict()
    assert newbie.wiring_state == "experimental_live", newbie.as_dict()
    assert not newbie.fallback_reason, newbie.as_dict()

    unknown = profile_selective_checkpoint_route("sdxl", torch.nn.Linear(4, 4))
    assert unknown.route_supported is False, unknown.as_dict()
    assert unknown.fallback_reason, unknown.as_dict()

    print("selective_checkpoint_profile_smoke: ok")


if __name__ == "__main__":
    main()

