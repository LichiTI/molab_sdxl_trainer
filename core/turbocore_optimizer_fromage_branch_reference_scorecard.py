"""Report-only reference matrix for Fromage follow-up branches."""

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

from pytorch_optimizer import Fromage  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_fromage_branch_reference_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_fromage_branch_reference_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    cases = [
        _run_case(branch_id="fromage_per_tensor_norm_matrix", p_bound=None),
        _run_case(branch_id="fromage_p_bound", p_bound=1.0),
    ]
    ready_branches = sorted(
        {
            case["branch_id"]
            for case in cases
            if case["ok"] is True and case["param_mutated"] is True and case["state_keys_ok"] is True
        }
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_fromage_branch_reference_scorecard_v0",
        "gate": "optimizer_fromage_branch_reference",
        "ok": {"fromage_per_tensor_norm_matrix", "fromage_p_bound"}.issubset(set(ready_branches)),
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
            "fromage_per_tensor_norm_reference_ready_count": 1
            if "fromage_per_tensor_norm_matrix" in ready_branches
            else 0,
            "fromage_p_bound_reference_ready_count": 1 if "fromage_p_bound" in ready_branches else 0,
            "training_path_enabled_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "product_native_ready_count": 0,
        },
        "recommended_next_step": "validate Fromage per-tensor norm and p_bound native kernel parity before broader training validation",
        "notes": [
            "This is reference evidence only and does not enable runtime dispatch.",
            "Fromage follow-up branches remain default-off until broader validation and approval.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_case(*, branch_id: str, p_bound: float | None) -> dict[str, Any]:
    params = [
        torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([[0.25, -0.75, 0.5]], dtype=torch.float32)),
    ]
    grads = [
        torch.tensor([0.4, -0.8], dtype=torch.float32),
        torch.tensor([[1.0, 0.25, -0.5]], dtype=torch.float32),
    ]
    optimizer = Fromage(params, lr=0.1, p_bound=p_bound)
    before = [param.detach().clone() for param in params]
    for param, grad in zip(params, grads):
        param.grad = grad.clone()
    optimizer.step()
    state_keys = sorted(
        sorted(str(key) for key in optimizer.state[param].keys())
        for param in params
        if optimizer.state[param]
    )
    expected_state_keys = [["max"], ["max"]] if p_bound is not None else []
    return {
        "schema_version": 1,
        "branch_id": branch_id,
        "ok": True,
        "p_bound": p_bound,
        "param_mutated": any(not torch.equal(old, param.detach()) for old, param in zip(before, params)),
        "state_keys": state_keys,
        "expected_state_keys": expected_state_keys,
        "state_keys_ok": state_keys == expected_state_keys,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if state_keys == expected_state_keys else ["fromage_branch_reference_matrix_failed"],
    }


__all__ = ["build_fromage_branch_reference_scorecard"]
