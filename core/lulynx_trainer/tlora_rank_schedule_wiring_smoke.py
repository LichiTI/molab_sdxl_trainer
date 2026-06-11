# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Local wiring + parity smoke for the T-LoRA temporal rank schedule strategy.

T-LoRA allocates ``max_rank`` capacity but masks the effective rank to a value
driven by the training step (constant / linear / geometric). The schedule only
advances when ``set_global_step`` is pushed every train step -- which the live
training loop previously never did, so the schedule was silently frozen at
``min_rank``. This smoke proves the now-wired runtime path:

  * the injector builds ``TLoRALinear`` when (and only when) T-LoRA is selected;
  * ``LoRAInjector.set_global_step`` (the call the training loop now makes each
    step) advances the effective rank exactly as the ready
    ``tlora_rank_schedule_consumption`` bridge predicts;
  * when T-LoRA is off (default), the injected layers are plain ``LoRALinear``
    and ``set_global_step`` is a bitwise no-op -> parity with legacy LoRA.

Checks (CPU, seconds, no pytest -- run as a script):

  1. CONFIG defaults:    ``t_lora_enabled=False`` and ``tlora_rank_schedule
     ="constant"`` -> default-off + neutral schedule (parity).
  2. INJECTOR off-parity: ``tlora_enabled=False`` -> all injected layers are
     ``LoRALinear`` (not ``TLoRALinear``); ``set_global_step`` mutates nothing.
  3. INJECTOR active:    ``tlora_enabled=True`` -> injected layers are
     ``TLoRALinear``.
  4. INIT no-op:         a fresh ``TLoRALinear`` forward equals the base layer
     output (lora_up zero-init), so enabling T-LoRA is a no-op at step 0.
  5. CONSTANT schedule:  rank stays at ``min_rank`` across steps.
  6. LINEAR schedule:    injector-driven rank trace == bridge expected trace.
  7. GEOMETRIC schedule: injector-driven rank trace == bridge expected trace.
  8. BRIDGE ready:       with the runtime hooks now available, the consumption
     bridge returns ``ok=True`` (the ``set_global_step_hook_missing`` blocker
     that previously gated T-LoRA is gone).
  9. MASK correctness:   the rank mask has exactly ``current_rank`` active dims.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/tlora_rank_schedule_wiring_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    here = Path(__file__).resolve()
    backend_root = here.parents[2]          # .../backend -> exposes `core`
    repo_root = here.parents[3]             # repo root    -> exposes `backend`
    for path in (str(repo_root), str(backend_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.lora_injector import LoRAInjector, LoRALinear
from core.lulynx_trainer.tlora import TLoRALinear
from core.lulynx_trainer.tlora_rank_schedule_consumption import (
    build_tlora_rank_schedule_consumption_plan,
)

TARGETS = ["to_q", "to_k", "to_v", "to_out"]
MODEL_SEED = 0
INJECT_SEED = 777
DIM = 16


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)


class _UNet(nn.Module):
    def __init__(self, dim: int = DIM, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_Attn(dim) for _ in range(blocks)])


def _build_unet() -> _UNet:
    torch.manual_seed(MODEL_SEED)
    return _UNet()


def _inject(injector: LoRAInjector) -> dict:
    unet = _build_unet()
    torch.manual_seed(INJECT_SEED)
    return injector.inject_unet(unet)


def _config_field_defaults() -> dict:
    fields = getattr(UnifiedTrainingConfig, "model_fields", None)
    if fields is None:  # pydantic v1 fallback
        fields = UnifiedTrainingConfig.__fields__
    return {name: getattr(field, "default", None) for name, field in fields.items()}


def _bridge_expected_trace(schedule: str, *, min_rank: int, max_rank: int, total_steps: int) -> tuple[list, list, bool]:
    """Return (trace_steps, expected_rank_trace, plan_ok) from the live bridge."""
    plan = {
        "plan": "tlora_ab_request_patch_plan_v0",
        "request_fields_emitted": True,
        "dry_run_only": True,
        "ok": True,
        "patches": [
            {
                "arm": "tlora",
                "case_id": f"tlora_{schedule}",
                "family": "anima",
                "request_patch": {
                    "tlora_ab_case_id": f"tlora_{schedule}",
                    "model_family": "anima",
                    "tlora_rank_schedule": schedule,
                    "tlora_min_rank": min_rank,
                    "max_train_steps": total_steps,
                },
            }
        ],
    }
    capability = {
        "set_global_step_available": True,
        "rank_mask_buffer_available": True,
        "total_steps_source": "config.max_train_steps",
        "max_rank": max_rank,
        "supported_schedules": ["constant", "linear", "geometric"],
    }
    result = build_tlora_rank_schedule_consumption_plan(request_patch_plan=plan, module_capability=capability)
    row = result["schedule_rows"][0]
    return row["trace_steps"], row["expected_rank_trace"], bool(result["ok"])


def _injector_rank_trace(schedule: str, *, rank: int, min_rank: int, total_steps: int, steps: list) -> list:
    """Drive the REAL injector.set_global_step (the loop's hook) and read ranks."""
    injector = LoRAInjector(
        rank=rank,
        target_modules=TARGETS,
        tlora_enabled=True,
        tlora_min_rank=min_rank,
        tlora_rank_schedule=schedule,
        tlora_total_steps=total_steps,
    )
    _inject(injector)
    layer = next(iter(injector.injected_layers.values()))
    trace = []
    for step in steps:
        injector.set_global_step(int(step))
        trace.append(int(layer.current_rank))
    return trace


def run() -> dict:
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- 1. CONFIG defaults --------------------------------------------------
    defaults = _config_field_defaults()
    check(
        "config_default_off",
        defaults.get("t_lora_enabled") is False and defaults.get("tlora_rank_schedule") == "constant",
        f"t_lora_enabled={defaults.get('t_lora_enabled')}, schedule={defaults.get('tlora_rank_schedule')}",
    )

    # --- 2. INJECTOR off-parity ----------------------------------------------
    off_inj = LoRAInjector(rank=8, target_modules=TARGETS)  # tlora_enabled defaults False
    _inject(off_inj)
    layers_off = list(off_inj.injected_layers.values())
    all_plain = all(isinstance(l, LoRALinear) and not isinstance(l, TLoRALinear) for l in layers_off)
    check("off_layers_plain_lora", all_plain and len(layers_off) == 8, f"n={len(layers_off)}")
    snap_before = {id(l): {n: p.detach().clone() for n, p in l.named_parameters()} for l in layers_off}
    off_inj.set_global_step(100)  # must be a no-op for plain LoRA
    no_op = all(
        torch.equal(snap_before[id(l)][n], p) for l in layers_off for n, p in l.named_parameters()
    )
    check("off_set_global_step_noop", no_op)

    # --- 3. INJECTOR active --------------------------------------------------
    on_inj = LoRAInjector(
        rank=8, target_modules=TARGETS,
        tlora_enabled=True, tlora_min_rank=2, tlora_rank_schedule="linear", tlora_total_steps=100,
    )
    _inject(on_inj)
    layers_on = list(on_inj.injected_layers.values())
    check("on_layers_are_tlora", all(isinstance(l, TLoRALinear) for l in layers_on) and len(layers_on) == 8)

    # --- 4. INIT no-op (lora_up zero -> forward == base) ---------------------
    torch.manual_seed(INJECT_SEED)
    base = nn.Linear(DIM, DIM, bias=False)
    tl = TLoRALinear(original_layer=base, max_rank=16, min_rank=4, schedule="constant", total_steps=100)
    x = torch.randn(3, DIM)
    check("init_forward_is_base", torch.equal(tl(x), base(x)))

    # --- 5. CONSTANT schedule stays at min_rank ------------------------------
    const_trace = _injector_rank_trace("constant", rank=16, min_rank=4, total_steps=100, steps=[0, 50, 100])
    check("constant_stays_min_rank", const_trace == [4, 4, 4], f"trace={const_trace}")

    # --- 6. LINEAR schedule: live trace == bridge expected -------------------
    lin_steps, lin_expected, lin_ok = _bridge_expected_trace("linear", min_rank=4, max_rank=16, total_steps=100)
    lin_live = _injector_rank_trace("linear", rank=16, min_rank=4, total_steps=100, steps=lin_steps)
    check("linear_trace_matches_bridge", lin_live == lin_expected, f"live={lin_live} expected={lin_expected}")

    # --- 7. GEOMETRIC schedule: live trace == bridge expected ----------------
    geo_steps, geo_expected, geo_ok = _bridge_expected_trace("geometric", min_rank=4, max_rank=16, total_steps=100)
    geo_live = _injector_rank_trace("geometric", rank=16, min_rank=4, total_steps=100, steps=geo_steps)
    check("geometric_trace_matches_bridge", geo_live == geo_expected, f"live={geo_live} expected={geo_expected}")
    check("schedules_differ", lin_expected != geo_expected, f"linear={lin_expected} geometric={geo_expected}")

    # --- 8. BRIDGE readiness (set_global_step hook now available) ------------
    check("bridge_ready_with_runtime_hooks", lin_ok and geo_ok, f"linear_ok={lin_ok} geometric_ok={geo_ok}")

    # --- 9. MASK correctness -------------------------------------------------
    tl.set_global_step(100)  # constant -> rank stays 4
    mask_sum = int(tl._rank_mask.sum().item())
    head_ones = bool(torch.all(tl._rank_mask[: tl.current_rank] == 1.0))
    tail_zeros = bool(torch.all(tl._rank_mask[tl.current_rank:] == 0.0))
    check("rank_mask_matches_current_rank", mask_sum == tl.current_rank and head_ones and tail_zeros,
          f"sum={mask_sum} rank={tl.current_rank}")

    passed = sum(1 for r in results if r["ok"])
    return {
        "smoke": "tlora_rank_schedule_wiring_smoke",
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
    print(f"\n[tlora_rank_schedule_wiring_smoke] {report['passed']}/{report['total']} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
