# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Local wiring + parity smoke for adapter-target-policy LoRA injection.

This proves the FG-LoRA style ``adapter_target_policy`` strategy is wired into the
*real* ``LoRAInjector`` as a default-off, strategy-selectable option, and that the
default keeps injection bitwise-identical to legacy behavior.

Checks (CPU, seconds, no pytest -- run as a script):

  1. PARITY (default):   ``LoRAInjector(rank=R)`` vs ``LoRAInjector(rank=R,
     adapter_target_policy="all")`` -> identical injected layers, ranks AND
     bitwise-identical adapter weights; policy reports inactive.
  2. PARITY (active-but-all): an explicitly active policy that selects *every*
     target at the base rank -> still bitwise-identical to legacy. Proves the
     active code path is a no-op when it selects all.
  3. SELECTION:          policy with ``selected={to_q,to_v}`` + per-leaf ranks ->
     only those leaves get adapters, at the requested ranks.
  4. GATING:             the explicit ``inject()`` / text-encoder path ignores the
     policy (apply_policy=False) -> all targets still injected.
  5. RUNTIME PATH:       the real ``load_policy_consumer_from_config`` +
     ``select_targets`` resolution (what trainer.py calls) selects a subset from a
     profile JSON, and feeding it into the injector reproduces that selection.
  6. NATIVE EXPLICIT:    the explicit ``inject(..., apply_policy=True)`` path used
     for Anima/native unet targets honors *dotted* target selection + per-type
     rank, while default policy="all" stays a parity no-op on that same path.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/adapter_target_policy_injection_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.adapter_target_policy_consumer import (
    AdapterTargetPolicyConsumer,
    load_policy_consumer_from_config,
)

TARGETS = ["to_q", "to_k", "to_v", "to_out"]
INJECT_SEED = 777
MODEL_SEED = 0


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)


class _UNet(nn.Module):
    def __init__(self, dim: int = 16, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_Attn(dim) for _ in range(blocks)])


def _build_unet() -> _UNet:
    torch.manual_seed(MODEL_SEED)
    return _UNet()


class _Proj(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)


class _DottedAttn(nn.Module):
    """Two sub-attentions whose leaf names collide ('q_proj'/'v_proj'), so the
    only distinguishing key is the *dotted* name ('cross_attn.v_proj'). Mirrors
    the Anima/native module layout that exposed the #70 leaf-only matching gap."""

    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = _Proj(dim)
        self.cross_attn = _Proj(dim)


class _DottedUNet(nn.Module):
    def __init__(self, dim: int = 16, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DottedAttn(dim) for _ in range(blocks)])


def _build_dotted_unet() -> _DottedUNet:
    torch.manual_seed(MODEL_SEED)
    return _DottedUNet()


def _inject_unet(injector: LoRAInjector) -> dict:
    """Inject into a fresh, identically-seeded UNet under a fixed injection RNG."""
    unet = _build_unet()
    torch.manual_seed(INJECT_SEED)
    return injector.inject_unet(unet)


def _down_weights(injected: dict) -> dict:
    out = {}
    for name, layer in injected.items():
        lora = getattr(layer, "lora", None)
        down = getattr(getattr(lora, "lora_down", None), "weight", None)
        if down is not None:
            out[name] = down.detach().clone()
    return out


def _rank_by_leaf(injected: dict) -> dict:
    ranks = {}
    for name, layer in injected.items():
        leaf = name.split(".")[-1]
        down = layer.lora.lora_down.weight
        ranks[leaf] = int(down.shape[0])
    return ranks


def _leaf_set(injected: dict) -> set:
    return {name.split(".")[-1] for name in injected}


def _dotted_set(injected: dict) -> set:
    """Last two path segments, e.g. 'cross_attn.v_proj' -- the dotted target name."""
    out = set()
    for name in injected:
        parts = name.split(".")
        out.add(".".join(parts[-2:]) if len(parts) >= 2 else parts[-1])
    return out


def _bitwise_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def run() -> dict:
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- 1. PARITY (default vs policy="all") ---------------------------------
    base = _inject_unet(LoRAInjector(rank=4, target_modules=TARGETS))
    all_policy_inj = LoRAInjector(rank=4, target_modules=TARGETS, adapter_target_policy="all")
    all_policy = _inject_unet(all_policy_inj)
    check(
        "default_policy_inactive",
        all_policy_inj._adapter_target_policy_active is False and all_policy_inj.adapter_target_policy == "all",
        f"active={all_policy_inj._adapter_target_policy_active}",
    )
    check("parity_default_layer_set", set(base) == set(all_policy), f"n={len(base)}")
    check("parity_default_bitwise", _bitwise_equal(_down_weights(base), _down_weights(all_policy)))

    # --- 2. PARITY (active policy that selects ALL targets at base rank) ------
    active_all_inj = LoRAInjector(
        rank=4,
        target_modules=TARGETS,
        adapter_target_policy="profiled",
        adapter_target_selected=set(TARGETS),
        adapter_target_rank_map={t: 4 for t in TARGETS},
    )
    active_all = _inject_unet(active_all_inj)
    check("active_all_policy_active", active_all_inj._adapter_target_policy_active is True)
    check("parity_active_all_layer_set", set(base) == set(active_all))
    check("parity_active_all_bitwise", _bitwise_equal(_down_weights(base), _down_weights(active_all)))

    # --- 3. SELECTION (subset + per-leaf ranks) ------------------------------
    sel_inj = LoRAInjector(
        rank=4,
        target_modules=TARGETS,
        adapter_target_policy="profiled",
        adapter_target_selected={"to_q", "to_v"},
        adapter_target_rank_map={"to_q": 8, "to_v": 4},
    )
    sel = _inject_unet(sel_inj)
    check("selection_leaf_set", _leaf_set(sel) == {"to_q", "to_v"}, str(sorted(_leaf_set(sel))))
    check("selection_count", len(sel) == 4, f"n={len(sel)}")  # 2 leaves * 2 blocks
    ranks = _rank_by_leaf(sel)
    check("selection_rank_to_q", ranks.get("to_q") == 8, str(ranks))
    check("selection_rank_to_v", ranks.get("to_v") == 4, str(ranks))

    # --- 4. GATING (explicit inject()/TE path ignores policy) ----------------
    gating_unet = _build_unet()
    torch.manual_seed(INJECT_SEED)
    gated = sel_inj.inject(gating_unet, TARGETS, prefix="")  # apply_policy defaults False
    check("gating_explicit_inject_all", _leaf_set(gated) == set(TARGETS), str(sorted(_leaf_set(gated))))

    # --- 5. RUNTIME PATH (load_policy_consumer_from_config + select_targets) --
    # gradient_selected with these grad-norms keeps the top-fraction (default 1.0
    # would keep all, so use top_k=2 to force a 2-of-4 subset deterministically).
    profile = {
        "layers": [
            {"name": "to_q", "grad_norm": 1.0},
            {"name": "to_k", "grad_norm": 0.05},
            {"name": "to_v", "grad_norm": 0.8},
            {"name": "to_out", "grad_norm": 0.1},
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        profile_path = Path(td) / "atp_profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")

        cfg = SimpleNamespace(
            adapter_target_policy="gradient_selected",
            adapter_target_policy_profile_path=str(profile_path),
            network_dim=4,
            adapter_target_policy_top_k=2,
        )
        consumer = load_policy_consumer_from_config(cfg)
        check("runtime_consumer_loaded", isinstance(consumer, AdapterTargetPolicyConsumer))
        selected_names, rank_map = consumer.select_targets(TARGETS, base_rank=4)
        check(
            "runtime_select_topk_subset",
            set(selected_names) == {"to_q", "to_v"},
            f"selected={sorted(selected_names)}",
        )

        runtime_inj = LoRAInjector(
            rank=4,
            target_modules=TARGETS,
            adapter_target_policy="gradient_selected",
            adapter_target_selected=set(selected_names),
            adapter_target_rank_map=rank_map,
        )
        runtime = _inject_unet(runtime_inj)
        check("runtime_injection_matches_selection", _leaf_set(runtime) == {"to_q", "to_v"})

    # --- 6. NATIVE EXPLICIT PATH (dotted targets + apply_policy=True) ---------
    # Mirrors trainer.py's Anima/native unet injection: the explicit inject()
    # entrypoint opts into the policy (apply_policy=True) with *dotted* target
    # names. Proves dotted targets match by full name (leaf-only would alias
    # self_attn.v_proj == cross_attn.v_proj) and that the subset filter +
    # per-type rank apply on the explicit path -- the gap real Anima exposed.
    dotted_targets = [
        "self_attn.q_proj", "self_attn.v_proj",
        "cross_attn.q_proj", "cross_attn.v_proj",
    ]
    dotted_inj = LoRAInjector(
        rank=4,
        target_modules=dotted_targets,
        adapter_target_policy="gradient_selected",
        adapter_target_selected={"cross_attn.v_proj"},
        adapter_target_rank_map={"cross_attn.v_proj": 8},
    )
    dotted_unet = _build_dotted_unet()
    torch.manual_seed(INJECT_SEED)
    dotted = dotted_inj.inject(dotted_unet, dotted_targets, prefix="unet", apply_policy=True)
    check("native_dotted_subset", _dotted_set(dotted) == {"cross_attn.v_proj"}, str(sorted(_dotted_set(dotted))))
    check("native_dotted_count", len(dotted) == 2, f"n={len(dotted)}")  # 1 dotted target * 2 blocks
    check(
        "native_dotted_rank",
        all(int(layer.lora.lora_down.weight.shape[0]) == 8 for layer in dotted.values()),
        "per-type rank resolved from dotted matched_target",
    )

    # PARITY on the explicit path: default policy="all" still injects every
    # dotted target even with apply_policy=True (policy inactive -> strict no-op),
    # guarding the trainer.py native-inject opt-in against default regressions.
    parity_inj = LoRAInjector(rank=4, target_modules=dotted_targets, adapter_target_policy="all")
    parity_unet = _build_dotted_unet()
    torch.manual_seed(INJECT_SEED)
    parity_dotted = parity_inj.inject(parity_unet, dotted_targets, prefix="unet", apply_policy=True)
    check(
        "native_dotted_parity_all",
        _dotted_set(parity_dotted) == set(dotted_targets) and len(parity_dotted) == 8,
        f"n={len(parity_dotted)}",
    )

    passed = sum(1 for r in results if r["ok"])
    return {
        "smoke": "adapter_target_policy_injection_smoke",
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
    print(f"\n[adapter_target_policy_injection_smoke] {report['passed']}/{report['total']} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
