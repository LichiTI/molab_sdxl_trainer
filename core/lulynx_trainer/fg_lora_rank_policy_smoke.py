# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Local wiring + parity smoke for FG-LoRA per-layer rank policy (frontier #4).

Proves the *orthogonal* rank-redistribution direction of ``fg_lora_rank_policy``
is wired into the REAL ``LoRAInjector`` via a per-FULL-PATH rank map (the 1-line
injector change: full-path key beats module-type key beats uniform rank), that
ALL target layers are kept (orthogonal != pruning), and that the parity defaults
(flat profile) stay identical to legacy injection.

Checks (CPU, seconds, no pytest -- run as a script):
  1. FLAT PARITY:    build_orthogonal_rank_map(profile="flat") -> every layer at
     base_rank; injecting it reproduces legacy injection (same layers, all base).
  2. ALL KEPT:       an ascending profile injects ALL targets (no layer dropped).
  3. PER-BLOCK RANK: the same leaf type gets DIFFERENT rank across blocks (deep
     blocks bigger under "ascending"); same block / different leaf share a rank.
  4. MAP CONSUMED:   each injected layer's real LoRA rank == the rank_map value
     keyed by its full path (the injector actually consumed the per-layer map).
  5. FULLPATH > TYPE: a map mixing a full-path key and a module-type key resolves
     the full path first, the type key as fallback -- the exact 1-line semantics.
  6. CONSERVE:       conserve_budget keeps sum(rank) ~= N * base_rank.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/fg_lora_rank_policy_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.fg_lora_rank_policy import (
    FgLoraRankPolicyConfig,
    build_orthogonal_rank_map,
    select_target_full_paths,
)

TARGETS = ["to_q", "to_k", "to_v", "to_out"]
INJECT_SEED = 777
MODEL_SEED = 0
BASE_RANK = 8
BLOCKS = 4


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)


class _UNet(nn.Module):
    def __init__(self, dim: int = 16, blocks: int = BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList([_Attn(dim) for _ in range(blocks)])


def _build_unet() -> _UNet:
    torch.manual_seed(MODEL_SEED)
    return _UNet()


def _matched_names(unet: nn.Module) -> list:
    linear_names = [n for n, m in unet.named_modules() if isinstance(m, nn.Linear)]
    return select_target_full_paths(linear_names, TARGETS)


def _inject(injector: LoRAInjector, unet: nn.Module, apply_policy: bool = True) -> dict:
    # prefix="" => injected dict key == relative module path == rank_map key.
    torch.manual_seed(INJECT_SEED)
    return injector.inject(unet, TARGETS, prefix="", apply_policy=apply_policy)


def _rank_by_name(injected: dict) -> dict:
    return {name: int(layer.lora.lora_down.weight.shape[0]) for name, layer in injected.items()}


def _orthogonal_injector(rank_map: dict) -> LoRAInjector:
    return LoRAInjector(
        rank=BASE_RANK, target_modules=TARGETS,
        adapter_target_policy="fg_lora_orthogonal", adapter_target_rank_map=rank_map,
    )


def run() -> dict:
    results: list = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- 1. FLAT PARITY -------------------------------------------------------
    legacy = _inject(LoRAInjector(rank=BASE_RANK, target_modules=TARGETS), _build_unet(), apply_policy=False)
    legacy_ranks = _rank_by_name(legacy)

    flat_unet = _build_unet()
    flat_map = build_orthogonal_rank_map(_matched_names(flat_unet), BASE_RANK, FgLoraRankPolicyConfig(profile="flat"))
    flat_ranks = _rank_by_name(_inject(_orthogonal_injector(flat_map), flat_unet))
    check("flat_layer_set_parity", set(flat_ranks) == set(legacy_ranks), f"n={len(flat_ranks)}")
    check("flat_all_base_rank", all(r == BASE_RANK for r in flat_ranks.values()), "flat profile == uniform")
    check("flat_matches_legacy", flat_ranks == legacy_ranks)

    # --- 2 + 3 + 4. ASCENDING: all kept, per-block rank, map consumed ---------
    asc_unet = _build_unet()
    matched = _matched_names(asc_unet)
    asc_map = build_orthogonal_rank_map(
        matched, BASE_RANK,
        FgLoraRankPolicyConfig(profile="ascending", conserve_budget=False, min_rank=2, max_rank=16),
    )
    asc_ranks = _rank_by_name(_inject(_orthogonal_injector(asc_map), asc_unet))
    deep = f"blocks.{BLOCKS - 1}.to_q"
    check("ascending_all_layers_kept", set(asc_ranks) == set(matched), f"kept={len(asc_ranks)}/{len(matched)}")
    check(
        "ascending_per_block_rank_differs",
        asc_ranks["blocks.0.to_q"] < asc_ranks[deep],
        f"b0={asc_ranks['blocks.0.to_q']} bN={asc_ranks[deep]}",
    )
    check(
        "ascending_same_block_same_rank",
        asc_ranks["blocks.0.to_q"] == asc_ranks["blocks.0.to_k"] == asc_ranks["blocks.0.to_v"],
        "per-block granularity",
    )
    check(
        "ascending_injected_rank_matches_map",
        all(asc_ranks[name] == asc_map[name] for name in asc_ranks),
        "injector consumed the per-layer map",
    )

    # --- 5. FULLPATH BEATS TYPE (the exact 1-line semantics) ------------------
    mixed_map = {"blocks.0.to_q": 16, "to_q": 2}  # full-path key + module-type key
    mix_ranks = _rank_by_name(_inject(_orthogonal_injector(mixed_map), _build_unet()))
    check("fullpath_key_wins", mix_ranks.get("blocks.0.to_q") == 16, str(mix_ranks.get("blocks.0.to_q")))
    check("type_key_fallback", mix_ranks.get("blocks.1.to_q") == 2, str(mix_ranks.get("blocks.1.to_q")))
    check("uniform_fallback_unkeyed", mix_ranks.get("blocks.0.to_k") == BASE_RANK, str(mix_ranks.get("blocks.0.to_k")))

    # --- 6. CONSERVE BUDGET ---------------------------------------------------
    cons_unet = _build_unet()
    cons_matched = _matched_names(cons_unet)
    cons_map = build_orthogonal_rank_map(
        cons_matched, BASE_RANK,
        FgLoraRankPolicyConfig(profile="center_peak", conserve_budget=True, min_rank=1, max_rank=64),
    )
    budget = BASE_RANK * len(cons_matched)
    total = sum(cons_map.values())
    check("conserve_budget_near_total", abs(total - budget) <= budget * 0.20, f"total={total} budget={budget}")

    passed = sum(1 for r in results if r["ok"])
    return {
        "smoke": "fg_lora_rank_policy_smoke",
        "passed": passed,
        "total": len(results),
        "ok": passed == len(results),
        "results": results,
    }


def main() -> int:
    report = run()
    for r in report["results"]:
        status = "PASS" if r["ok"] else "FAIL"
        line = f"  [{status}] {r['check']}"
        if r["detail"]:
            line += f"  ({r['detail']})"
        print(line)
    print(f"\n[fg_lora_rank_policy_smoke] {report['passed']}/{report['total']} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
