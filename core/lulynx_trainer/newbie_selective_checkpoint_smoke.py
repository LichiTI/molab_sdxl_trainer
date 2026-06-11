"""Smoke-test Newbie selective checkpoint wiring on a tiny block loop."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.checkpoint_policy import (  # noqa: E402
    profile_selective_checkpoint_route,
    resolve_checkpoint_policy,
)


class _TinyNewbieDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._block_modules = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self._gradient_checkpointing = False
        self._newbie_block_checkpointing_mode = "off"

    def set_newbie_block_checkpointing(self, enabled: bool, mode: str = "block"):
        normalized = str(mode or "block").strip().lower().replace("-", "_")
        active = bool(enabled) and normalized in {"", "block", "selective"}
        self._gradient_checkpointing = active
        self._newbie_block_checkpointing_mode = "selective" if active and normalized == "selective" else "block" if active else "off"
        return self.get_newbie_block_checkpointing_profile()

    def get_newbie_block_checkpointing_profile(self):
        return {
            "enabled": bool(self._gradient_checkpointing),
            "mode": self._newbie_block_checkpointing_mode,
            "block_count": len(self._block_modules),
            "checkpointed_blocks": len(self._block_modules) if self._gradient_checkpointing else 0,
        }

    def _run_dit_block(self, block, x, t_emb):
        return torch.relu(block(x + t_emb.unsqueeze(1) * 0.01))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self._block_modules:
            if self._gradient_checkpointing and self.training:
                checkpoint_kwargs = {"use_reentrant": False, "preserve_rng_state": False}
                if self._newbie_block_checkpointing_mode == "selective":
                    from core.lulynx_trainer.checkpoint_policy import build_selective_checkpoint_context_fn

                    context_fn = build_selective_checkpoint_context_fn("balanced")
                    if context_fn is not None:
                        checkpoint_kwargs["context_fn"] = context_fn
                x = torch.utils.checkpoint.checkpoint(
                    self._run_dit_block,
                    block,
                    x,
                    t_emb,
                    **checkpoint_kwargs,
                )
            else:
                x = self._run_dit_block(block, x, t_emb)
        return x


def main() -> int:
    decision = resolve_checkpoint_policy(
        type("Cfg", (), {"checkpoint_policy": "selective", "gradient_checkpointing": False, "cpu_offload_checkpointing": False})(),
        route="newbie",
        cuda_available=True,
    )
    assert decision.effective_policy == "selective", decision.as_dict()
    assert decision.gradient_checkpointing is False, decision.as_dict()

    model = _TinyNewbieDiT().train()
    profile = profile_selective_checkpoint_route("newbie", model)
    assert profile.forward_wired is True, profile.as_dict()
    assert profile.wiring_state == "experimental_live", profile.as_dict()

    set_profile = model.set_newbie_block_checkpointing(True, "selective")
    assert set_profile["enabled"] is True, set_profile
    assert set_profile["mode"] == "selective", set_profile

    x = torch.randn(2, 3, 4, requires_grad=True)
    t = torch.randn(2, 4, requires_grad=True)
    out = model(x, t).sum()
    out.backward()
    assert x.grad is not None
    assert t.grad is not None
    for block in model._block_modules:
        assert block.weight.grad is not None

    print("newbie_selective_checkpoint_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
