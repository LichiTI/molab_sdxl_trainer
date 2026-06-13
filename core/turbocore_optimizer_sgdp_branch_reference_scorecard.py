"""Report-only reference matrix for SGDP follow-up branches."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_OPTIMIZER_ROOT = REPO_ROOT / "plugin" / "pytorch_optimizer-main"
if str(PYTORCH_OPTIMIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_OPTIMIZER_ROOT))

from pytorch_optimizer import SGDP  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_sgdp_branch_reference_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_sgdp_branch_reference_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    cases = [
        _run_case(
            branch_id="sgdp_projection",
            delta=1.0,
            wd_ratio=0.1,
            weight_decouple=True,
            fixed_decay=False,
            nesterov=True,
        ),
        _run_case(
            branch_id="sgdp_decoupled_decay",
            delta=0.0,
            wd_ratio=0.1,
            weight_decouple=True,
            fixed_decay=False,
            nesterov=False,
        ),
        _run_case(
            branch_id="sgdp_decoupled_decay",
            delta=0.0,
            wd_ratio=0.1,
            weight_decouple=True,
            fixed_decay=True,
            nesterov=False,
        ),
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
        "scorecard": "turbocore_optimizer_sgdp_branch_reference_scorecard_v0",
        "gate": "optimizer_sgdp_branch_reference",
        "ok": {"sgdp_projection", "sgdp_decoupled_decay"}.issubset(set(ready_branches)),
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
            "sgdp_projection_reference_ready_count": 1 if "sgdp_projection" in ready_branches else 0,
            "sgdp_decoupled_decay_reference_ready_count": 1 if "sgdp_decoupled_decay" in ready_branches else 0,
            "training_path_enabled_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "product_native_ready_count": 0,
        },
        "recommended_next_step": "validate SGDP projection/decoupled decay native kernel parity before broader training validation",
        "notes": [
            "This is reference evidence only and does not enable runtime dispatch.",
            "SGDP follow-up branches remain default-off until broader validation and approval.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_case(
    *,
    branch_id: str,
    delta: float,
    wd_ratio: float,
    weight_decouple: bool,
    fixed_decay: bool,
    nesterov: bool,
) -> dict[str, Any]:
    param = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
    grad = torch.tensor([[0.0, 0.25, -0.5, 0.125]], dtype=torch.float32)
    optimizer = SGDP(
        [param],
        lr=1e-3,
        momentum=0.9,
        dampening=0.1,
        weight_decay=0.01,
        weight_decouple=weight_decouple,
        fixed_decay=fixed_decay,
        delta=delta,
        wd_ratio=wd_ratio,
        nesterov=nesterov,
        eps=1e-8,
    )
    before = param.detach().clone()
    param.grad = grad.clone()
    optimizer.step()
    state = dict(optimizer.state[param])
    expected_state_keys = ["momentum"]
    actual_state_keys = sorted(str(key) for key in state.keys())
    return {
        "schema_version": 1,
        "branch_id": branch_id,
        "ok": True,
        "delta": delta,
        "wd_ratio": wd_ratio,
        "weight_decouple": weight_decouple,
        "fixed_decay": fixed_decay,
        "nesterov": nesterov,
        "param_mutated": not torch.equal(before, param.detach()),
        "state_keys": actual_state_keys,
        "expected_state_keys": expected_state_keys,
        "state_keys_ok": actual_state_keys == expected_state_keys,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if actual_state_keys == expected_state_keys else ["sgdp_branch_reference_matrix_failed"],
    }


__all__ = ["build_sgdp_branch_reference_scorecard"]
