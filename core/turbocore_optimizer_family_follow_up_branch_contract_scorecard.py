"""Report-only branch contracts for TurboCore optimizer family follow-ups.

The O1 v2 roadmap rows are not product-ready just because their base canaries
run.  This scorecard pins each remaining branch to the native/runtime ABI pieces
that still have to exist before a parity canary can be called complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_optimizer_rmsprop_branch_reference_scorecard import (
    build_rmsprop_branch_reference_scorecard,
)
from core.turbocore_optimizer_fromage_branch_reference_scorecard import (
    build_fromage_branch_reference_scorecard,
)
from core.turbocore_optimizer_sgdp_branch_reference_scorecard import (
    build_sgdp_branch_reference_scorecard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_family_follow_up_branch_contract_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
RUNTIME_SOURCE = "backend/core/turbocore_simple_optimizer_training_executor.py"
NATIVE_SOURCES = {
    "rmsprop": "backend/native/src/cuda/rmsprop_flat_fp32_cuda_v0.cu",
    "pid": "backend/native/src/cuda/pid_flat_fp32_cuda_v0.cu",
    "sgdp": "backend/native/src/cuda/sgdp_flat_fp32_cuda_v0.cu",
    "fromage": "backend/native/src/cuda/fromage_flat_fp32_cuda_v0.cu",
}


BRANCH_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "branch_id": "rmsprop_centered",
        "family": "rmsprop",
        "title": "RMSProp centered branch",
        "required_state_roles": ("square_avg", "grad_avg"),
        "required_hyperparameters": ("centered", "alpha", "eps"),
        "required_native_tokens": ("grad_avg", "centered"),
        "required_runtime_tokens": ("centered",),
        "parity_case": "centered denominator uses sqrt(square_avg - grad_avg^2) + eps",
    },
    {
        "branch_id": "rmsprop_momentum",
        "family": "rmsprop",
        "title": "RMSProp momentum branch",
        "required_state_roles": ("square_avg", "momentum_buffer"),
        "required_hyperparameters": ("momentum", "alpha", "eps"),
        "required_native_tokens": ("momentum_buffer", "momentum"),
        "required_runtime_tokens": ("momentum",),
        "parity_case": "momentum buffer drives the final parameter update",
    },
    {
        "branch_id": "pid_momentum_three_buffer",
        "family": "pid",
        "title": "PID momentum three-buffer branch",
        "required_state_roles": ("integral_buffer", "previous_grad", "momentum_buffer"),
        "required_hyperparameters": ("momentum", "integral", "derivative"),
        "required_native_tokens": ("integral_buffer", "previous_grad", "momentum_buffer"),
        "required_runtime_tokens": ("momentum",),
        "parity_case": "PID keeps proportional, integral, derivative, and momentum state distinct",
    },
    {
        "branch_id": "sgdp_projection",
        "family": "sgdp",
        "title": "SGDP projection branch",
        "required_state_roles": ("momentum",),
        "required_hyperparameters": ("delta", "wd_ratio", "nesterov"),
        "required_native_tokens": ("projection", "delta", "wd_ratio"),
        "required_runtime_tokens": ("delta", "wd_ratio"),
        "parity_case": "SGDP projection path applies cosine projection before update",
    },
    {
        "branch_id": "sgdp_decoupled_decay",
        "family": "sgdp",
        "title": "SGDP decoupled decay branch",
        "required_state_roles": ("momentum",),
        "required_hyperparameters": ("weight_decay", "weight_decouple", "fixed_decay"),
        "required_native_tokens": ("weight_decay", "weight_decouple", "fixed_decay"),
        "required_runtime_tokens": ("weight_decay",),
        "parity_case": "decoupled/fixed decay policy is applied outside the gradient update",
    },
    {
        "branch_id": "fromage_per_tensor_norm_matrix",
        "family": "fromage",
        "title": "Fromage multi-parameter/per-tensor norm parity matrix",
        "required_state_roles": ("param_group_offsets",),
        "required_hyperparameters": ("per_tensor_norm",),
        "required_native_tokens": ("param_group_offsets", "per_tensor_norm"),
        "required_runtime_tokens": ("layout", "param_norm"),
        "parity_case": "each tensor uses its own parameter/gradient norm instead of one flat norm",
    },
    {
        "branch_id": "fromage_p_bound",
        "family": "fromage",
        "title": "Fromage p_bound state contract",
        "required_state_roles": ("p_bound",),
        "required_hyperparameters": ("p_bound",),
        "required_native_tokens": ("p_bound",),
        "required_runtime_tokens": ("p_bound",),
        "parity_case": "p_bound constrains the post-update parameter norm and survives resume",
    },
)


def build_family_follow_up_branch_contract_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    runtime_text = _read_repo_text(RUNTIME_SOURCE)
    native_texts = {family: _read_repo_text(path) for family, path in NATIVE_SOURCES.items()}
    rmsprop_reference = build_rmsprop_branch_reference_scorecard(write_artifact=write_artifact)
    sgdp_reference = build_sgdp_branch_reference_scorecard(write_artifact=write_artifact)
    fromage_reference = build_fromage_branch_reference_scorecard(write_artifact=write_artifact)
    reference_ready_branches = {
        str(branch)
        for report in (rmsprop_reference, sgdp_reference, fromage_reference)
        for branch in report.get("branch_reference_ready_branches", [])
    }
    rows = [
        _row(contract, native_texts.get(str(contract["family"]), ""), runtime_text, reference_ready_branches)
        for contract in BRANCH_CONTRACTS
    ]
    tracked_count = sum(1 for row in rows if row["branch_contract_tracked"])
    reference_ready_count = sum(1 for row in rows if row["branch_reference_ready"])
    implementation_ready_count = sum(1 for row in rows if row["branch_implementation_ready"])
    native_gap_count = len(rows) - implementation_ready_count
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_family_follow_up_branch_contract_scorecard_v0",
        "gate": "optimizer_family_follow_up_branch_contract",
        "ok": tracked_count == len(BRANCH_CONTRACTS),
        "promotion_ready": False,
        "roadmap": ROADMAP,
        "artifact_first": False,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "rows": rows,
        "summary": {
            "branch_contract_total_count": len(rows),
            "branch_contract_tracked_count": tracked_count,
            "branch_reference_ready_count": reference_ready_count,
            "branch_implementation_ready_count": implementation_ready_count,
            "branch_native_gap_count": native_gap_count,
            "family_follow_up_branch_contract_tracked_count": tracked_count,
            "family_follow_up_branch_reference_ready_count": reference_ready_count,
            "family_follow_up_branch_implementation_ready_count": implementation_ready_count,
            "family_follow_up_branch_native_gap_count": native_gap_count,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "branch_reference_scorecards": {
            "rmsprop": rmsprop_reference,
            "sgdp": sgdp_reference,
            "fromage": fromage_reference,
        },
        "promotion_blockers": [f"{row['branch_id']}_kernel_contract_gap" for row in rows if not row["branch_implementation_ready"]],
        "blocked_reasons": [],
        "recommended_next_step": "move O1 branch coverage into broader actual-training validation while keeping default-off",
        "notes": [
            "Branch contracts are tracked; ready branches still remain default-off until broader validation.",
            "This scorecard does not enable runtime dispatch, native dispatch, or product training paths.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _row(
    contract: Mapping[str, Any],
    native_text: str,
    runtime_text: str,
    reference_ready_branches: set[str],
) -> dict[str, Any]:
    native_missing = _missing(contract.get("required_native_tokens"), native_text)
    runtime_missing = _missing(contract.get("required_runtime_tokens"), runtime_text)
    family = str(contract.get("family") or "")
    source_path = NATIVE_SOURCES.get(family, "")
    source_exists = bool(source_path) and (REPO_ROOT / source_path).exists()
    branch_id = str(contract.get("branch_id") or "")
    branch_tracked = source_exists and bool(branch_id) and bool(contract.get("parity_case"))
    implementation_ready = branch_tracked and not native_missing and not runtime_missing
    return {
        "schema_version": 1,
        "branch_id": branch_id,
        "family": family,
        "title": str(contract.get("title") or ""),
        "status": "branch_implementation_ready" if implementation_ready else "branch_contract_gap_tracked",
        "branch_contract_tracked": branch_tracked,
        "branch_reference_ready": branch_id in reference_ready_branches,
        "branch_implementation_ready": implementation_ready,
        "native_source": source_path,
        "runtime_source": RUNTIME_SOURCE,
        "native_source_exists": source_exists,
        "required_state_roles": list(contract.get("required_state_roles") or []),
        "required_hyperparameters": list(contract.get("required_hyperparameters") or []),
        "missing_native_tokens": native_missing,
        "missing_runtime_tokens": runtime_missing,
        "parity_case": str(contract.get("parity_case") or ""),
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if implementation_ready else _blockers(native_missing, runtime_missing, source_exists),
    }


def _read_repo_text(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _missing(tokens: Any, text: str) -> list[str]:
    return [str(token) for token in tokens or () if str(token) not in text]


def _blockers(native_missing: list[str], runtime_missing: list[str], source_exists: bool) -> list[str]:
    blockers: list[str] = []
    if not source_exists:
        blockers.append("native_source_missing")
    blockers.extend(f"native_abi_token_missing:{token}" for token in native_missing)
    blockers.extend(f"runtime_config_token_missing:{token}" for token in runtime_missing)
    return blockers


__all__ = ["build_family_follow_up_branch_contract_scorecard"]
