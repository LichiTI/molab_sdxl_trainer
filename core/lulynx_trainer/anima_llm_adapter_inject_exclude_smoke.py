"""CPU smoke for the llm_adapter dead-LoRA fix.

Root cause: the LoRA injector matches target suffixes by substring, so the DiT
targets ``self_attn.q_proj`` / ``cross_attn.{q,k,v}_proj`` also match the frozen
native ``anima_llm_adapter.blocks.N.<...>.q_proj`` subtree, producing dead
(zero-gradient) adapters under faithful training (#183 found 36 such tensors).

Fix: ``LoRAInjector.inject(exclude_name_substrings=[...])`` skips any module whose
qualified name contains an excluded substring. The trainer passes
``["llm_adapter"]`` when the adapter is not trained.

This smoke builds a tiny module that mirrors the collision (a DiT block + an
``anima_llm_adapter`` block, both with ``self_attn.q_proj`` etc.) and asserts:
  - WITHOUT exclusion: both subtrees receive LoRA (reproduces the bug).
  - WITH ``exclude=["llm_adapter"]``: only the DiT block does; llm_adapter gets none.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/anima_llm_adapter_inject_exclude_smoke.py
"""
from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch.nn as nn

from core.lulynx_trainer.lora_injector import LoRAInjector

TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj"]


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.cross_attn = _Attn()


class _AdapterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.cross_attn = _Attn()


class _LLMAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_AdapterBlock() for _ in range(2)])


class _Unet(nn.Module):
    """Mirrors the name collision: DiT blocks + a sibling anima_llm_adapter."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.anima_llm_adapter = _LLMAdapter()


def _count(injected):
    dit = sum(1 for n in injected if "llm_adapter" not in n)
    adapter = sum(1 for n in injected if "llm_adapter" in n)
    return dit, adapter


def main() -> int:
    print("== anima llm_adapter inject-exclude smoke ==")

    # (1) no exclusion -> reproduces the bug: llm_adapter gets LoRA too.
    inj = LoRAInjector(rank=4, target_modules=TARGETS)
    res = inj.inject(_Unet(), TARGETS, prefix="unet")
    dit, adapter = _count(res)
    print(f"  no-exclude: dit={dit} llm_adapter={adapter}")
    assert dit > 0, "expected DiT blocks to be injected"
    assert adapter > 0, "expected the bug (llm_adapter injected) to reproduce without exclusion"

    # (2) exclude=['llm_adapter'] -> only DiT, zero adapter LoRA.
    inj2 = LoRAInjector(rank=4, target_modules=TARGETS)
    res2 = inj2.inject(_Unet(), TARGETS, prefix="unet", exclude_name_substrings=["llm_adapter"])
    dit2, adapter2 = _count(res2)
    print(f"  exclude:    dit={dit2} llm_adapter={adapter2}")
    assert adapter2 == 0, f"llm_adapter must be excluded, got {adapter2} injected"
    assert dit2 == dit, f"DiT injection must be unchanged ({dit} -> {dit2})"

    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
