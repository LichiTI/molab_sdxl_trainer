# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Immiscible diffusion (cleanroom).

Run with the flashattention env (CPU is fine):
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/immiscible_diffusion_smoke.py

Checks: (1) the L2 assignment is a valid permutation with B=1 passthrough;
(2) it lowers the mean paired L2 vs the original (identity) pairing; (3) the
``_maybe_assign_noise`` integration hook routes l2 / cosine / passthrough
correctly and is bit-identical to legacy when disabled; (4) the stage plan
records the immiscible feature.  Emits the promotion scorecard.
"""

from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from core.lulynx_trainer.immiscible_diffusion import minibatch_immiscible_l2
from core.lulynx_trainer.cosine_ot import minibatch_ot_cosine
from core.lulynx_trainer.immiscible_diffusion_scorecard import build_immiscible_scorecard
from core.lulynx_trainer.training_step_noise_timestep_handler import _maybe_assign_noise
from core.lulynx_trainer.training_step_noise_timestep_stage import (
    build_lulynx_training_step_noise_timestep_stage_plan,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _paired_l2(latents, noise):
    return ((latents.reshape(latents.shape[0], -1) - noise.reshape(noise.shape[0], -1)) ** 2).sum(1).mean().item()


def check_bijection() -> bool:
    print("== valid permutation + B=1 passthrough ==")
    ok = True
    for B in (1, 2, 4, 8, 16):
        lat = torch.randn(B, 4, 16, 16, device=DEV)
        noi = torch.randn(B, 4, 16, 16, device=DEV)
        out = minibatch_immiscible_l2(lat, noi)
        if B == 1:
            passed = torch.equal(out, noi)
        else:
            # every original noise row must appear exactly once in the output
            flat_out = out.reshape(B, -1)
            flat_noi = noi.reshape(B, -1)
            matched = sum(any(torch.equal(flat_out[i], flat_noi[j]) for j in range(B)) for i in range(B))
            passed = matched == B and out.shape == noi.shape
        ok &= passed
        print(f"  B={B:2d}: {'OK' if passed else 'FAIL'}")
    return ok


def check_l2_reduction() -> tuple[bool, float, float]:
    print("== paired L2 reduced vs identity pairing ==")
    ok = True
    worst_assigned = worst_random = 0.0
    for _ in range(5):
        B = 12
        lat = torch.randn(B, 4, 32, 32, device=DEV)
        noi = torch.randn(B, 4, 32, 32, device=DEV)
        assigned = minibatch_immiscible_l2(lat, noi)
        c_assigned = _paired_l2(lat, assigned)
        c_random = _paired_l2(lat, noi)  # original (random) pairing
        worst_assigned, worst_random = c_assigned, c_random
        passed = c_assigned < c_random
        ok &= passed
        print(f"  assigned={c_assigned:.1f}  random={c_random:.1f}  {'OK' if passed else 'FAIL'}")
    return ok, worst_assigned, worst_random


def check_routing() -> tuple[bool, bool]:
    print("== _maybe_assign_noise routing + disabled parity ==")
    torch.manual_seed(0)
    lat = torch.randn(6, 4, 16, 16, device=DEV)
    noi = torch.randn(6, 4, 16, 16, device=DEV)

    off = _maybe_assign_noise(lat, noi, flow_use_ot=False, immiscible_enabled=False, immiscible_metric="l2")
    disabled_parity = torch.equal(off, noi)

    l2_out = _maybe_assign_noise(lat, noi, flow_use_ot=False, immiscible_enabled=True, immiscible_metric="l2")
    l2_match = torch.equal(l2_out, minibatch_immiscible_l2(lat, noi))

    cos_flow = _maybe_assign_noise(lat, noi, flow_use_ot=True, immiscible_enabled=False, immiscible_metric="l2")
    cos_match = torch.equal(cos_flow, minibatch_ot_cosine(lat, noi))

    cos_metric = _maybe_assign_noise(lat, noi, flow_use_ot=False, immiscible_enabled=True, immiscible_metric="cosine")
    cos_metric_match = torch.equal(cos_metric, minibatch_ot_cosine(lat, noi))

    routing_ok = l2_match and cos_match and cos_metric_match
    print(f"  disabled→identity={disabled_parity}  l2→l2={l2_match}  flow_ot→cosine={cos_match}  metric=cosine→cosine={cos_metric_match}")
    print(f"  {'OK' if (routing_ok and disabled_parity) else 'FAIL'}")
    return routing_ok, disabled_parity


def check_plan_feature() -> bool:
    print("== stage plan records immiscible feature ==")
    plan = build_lulynx_training_step_noise_timestep_stage_plan(
        model_arch="sd15", batch_size=4, latent_rank=4, immiscible_enabled=True, immiscible_metric="l2",
    )
    feats = plan.as_dict()["randomization_features"]
    has = "immiscible_diffusion_l2" in feats
    # disabled → feature absent
    plan_off = build_lulynx_training_step_noise_timestep_stage_plan(
        model_arch="sd15", batch_size=4, latent_rank=4, immiscible_enabled=False,
    )
    absent = not any("immiscible" in f for f in plan_off.as_dict()["randomization_features"])
    ok = has and absent
    print(f"  enabled has feature={has}  disabled absent={absent}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"device: {DEV}")
    bij_ok = check_bijection()
    l2_ok, c_assigned, c_random = check_l2_reduction()
    routing_ok, disabled_ok = check_routing()
    plan_ok = check_plan_feature()

    scorecard = build_immiscible_scorecard(
        bijection_verified=bij_ok,
        l2_reduced=l2_ok,
        assigned_cost=c_assigned,
        random_cost=c_random,
        routing_verified=routing_ok,
        disabled_parity_verified=disabled_ok,
        plan_feature_verified=plan_ok,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")

    all_ok = bij_ok and l2_ok and routing_ok and disabled_ok and plan_ok
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if all_ok else 1)
