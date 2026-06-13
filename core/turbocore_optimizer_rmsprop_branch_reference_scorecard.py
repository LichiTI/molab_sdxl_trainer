"""Report-only reference matrix for RMSProp follow-up branches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_rmsprop_branch_reference_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_rmsprop_branch_reference_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    cases = [
        _run_case(centered=False, momentum=0.0),
        _run_case(centered=True, momentum=0.0),
        _run_case(centered=False, momentum=0.9),
        _run_case(centered=True, momentum=0.9),
    ]
    ready_branches = sorted(
        {
            case["branch_id"]
            for case in cases
            if case["ok"] is True and case["state_keys_ok"] is True and case["param_mutated"] is True
        }
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_rmsprop_branch_reference_scorecard_v0",
        "gate": "optimizer_rmsprop_branch_reference",
        "ok": len(ready_branches) > 0,
        "promotion_ready": False,
        "roadmap": ROADMAP,
        "artifact_first": False,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "branch_reference_ready_branches": ready_branches,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "branch_reference_ready_count": len(ready_branches),
            "rmsprop_centered_reference_ready_count": 1 if "rmsprop_centered" in ready_branches else 0,
            "rmsprop_momentum_reference_ready_count": 1 if "rmsprop_momentum" in ready_branches else 0,
            "training_path_enabled_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "product_native_ready_count": 0,
        },
        "recommended_next_step": "implement the RMSProp native centered and momentum ABI branches next",
        "notes": [
            "This is reference evidence only and does not enable runtime dispatch.",
            "Centered and momentum follow-up branches remain default-off until native ABI and kernel work lands.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_case(*, centered: bool, momentum: float) -> dict[str, Any]:
    torch.manual_seed(20260609)
    param = torch.nn.Parameter(torch.tensor([0.25, -0.5, 0.125, -0.75], dtype=torch.float32))
    grad = torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float32)
    optimizer = torch.optim.RMSprop(
        [param],
        lr=1e-3,
        alpha=0.99,
        eps=1e-8,
        momentum=momentum,
        centered=centered,
        weight_decay=0.01,
    )
    before = param.detach().clone()
    param.grad = grad.clone()
    optimizer.step()
    state = dict(optimizer.state[param])
    expected_state_keys = ["square_avg", "step"]
    if centered:
        expected_state_keys.append("grad_avg")
    if momentum > 0.0:
        expected_state_keys.append("momentum_buffer")
    expected_state_keys = sorted(expected_state_keys)
    actual_state_keys = sorted(str(key) for key in state.keys())
    return {
        "schema_version": 1,
        "branch_id": "rmsprop_centered" if centered else "rmsprop_momentum",
        "ok": True,
        "centered": centered,
        "momentum": momentum,
        "weight_decay": 0.01,
        "param_mutated": not torch.equal(before, param.detach()),
        "state_keys": actual_state_keys,
        "expected_state_keys": expected_state_keys,
        "state_keys_ok": actual_state_keys == expected_state_keys,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if actual_state_keys == expected_state_keys else ["rmsprop_branch_reference_matrix_failed"],
    }


__all__ = ["build_rmsprop_branch_reference_scorecard"]
