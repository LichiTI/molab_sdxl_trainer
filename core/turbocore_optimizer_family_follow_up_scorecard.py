"""Report-only follow-up scorecard for selected TurboCore optimizer families.

This report is intentionally default-off. It summarizes the current canary
state for the O1 family-follow-up rows in the v2 roadmap and makes the
remaining branches explicit without claiming product dispatch readiness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_fromage_training_loop_canary_scorecard import (
    build_plugin_fromage_training_loop_canary_scorecard,
)
from core.turbocore_plugin_pid_training_loop_canary_scorecard import (
    build_plugin_pid_training_loop_canary_scorecard,
)
from core.turbocore_plugin_rmsprop_training_loop_canary_scorecard import (
    build_plugin_rmsprop_training_loop_canary_scorecard,
)
from core.turbocore_plugin_sgdp_training_loop_canary_scorecard import (
    build_plugin_sgdp_training_loop_canary_scorecard,
)
from core.turbocore_optimizer_family_follow_up_branch_contract_scorecard import (
    build_family_follow_up_branch_contract_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_family_follow_up_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


FAMILY_ITEMS: dict[str, dict[str, Any]] = {
    "rmsprop": {
        "title": "RMSProp",
        "remaining_items": ["centered branch", "momentum branch"],
        "evidence_anchors": [
            "backend/core/turbocore_plugin_rmsprop_training_loop_canary_scorecard.py",
        ],
        "builder": build_plugin_rmsprop_training_loop_canary_scorecard,
    },
    "pid": {
        "title": "PID",
        "remaining_items": ["momentum three-buffer branch"],
        "evidence_anchors": [
            "backend/core/turbocore_plugin_pid_training_loop_canary_scorecard.py",
        ],
        "builder": build_plugin_pid_training_loop_canary_scorecard,
    },
    "sgdp": {
        "title": "SGDP",
        "remaining_items": ["projection branch", "decoupled decay branch"],
        "evidence_anchors": [
            "backend/core/turbocore_plugin_sgdp_training_loop_canary_scorecard.py",
        ],
        "builder": build_plugin_sgdp_training_loop_canary_scorecard,
    },
    "fromage": {
        "title": "Fromage",
        "remaining_items": [
            "multi-parameter/per-tensor norm parity matrix",
            "p_bound state contract",
        ],
        "evidence_anchors": [
            "backend/core/turbocore_plugin_fromage_training_loop_canary_scorecard.py",
            "backend/core/lulynx_trainer/turbocore_plugin_fromage_training_loop_canary_scorecard_smoke.py",
        ],
        "builder": build_plugin_fromage_training_loop_canary_scorecard,
    },
}


def build_family_follow_up_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    rows = [
        _row(name, config, _as_dict(config["builder"]()))
        for name, config in FAMILY_ITEMS.items()
    ]
    branch_contract = build_family_follow_up_branch_contract_scorecard(write_artifact=write_artifact)
    branch_summary = _as_dict(branch_contract.get("summary"))
    open_by_family = _open_branch_titles_by_family(branch_contract)
    rows = [_with_open_branches(row, open_by_family.get(str(row["optimizer_name"]), [])) for row in rows]
    canary_ready_count = sum(1 for row in rows if row["base_canary_ready"])
    remaining_branch_count = sum(int(row["remaining_branch_count"]) for row in rows)
    native_step_count = sum(int(row["native_step_count"]) for row in rows)
    native_kernel_launch_count = sum(int(row["native_kernel_launch_count"]) for row in rows)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_family_follow_up_scorecard_v0",
        "gate": "optimizer_family_follow_up",
        "ok": canary_ready_count == len(rows),
        "promotion_ready": False,
        "roadmap": ROADMAP,
        "artifact_first": False,
        "cuda_executed": native_kernel_launch_count > 0,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "selected_optimizer_family": "family_follow_up",
        "rows": rows,
        "summary": {
            "family_count": len(rows),
            "family_follow_up_family_count": len(rows),
            "base_canary_ready_count": canary_ready_count,
            "family_follow_up_base_canary_ready_count": canary_ready_count,
            "remaining_branch_count": remaining_branch_count,
            "family_follow_up_remaining_branch_count": remaining_branch_count,
            "family_follow_up_native_step_count": native_step_count,
            "family_follow_up_native_kernel_launch_count": native_kernel_launch_count,
            "family_follow_up_branch_contract_tracked_count": int(
                branch_summary.get("family_follow_up_branch_contract_tracked_count", 0) or 0
            ),
            "family_follow_up_branch_reference_ready_count": int(
                branch_summary.get("family_follow_up_branch_reference_ready_count", 0) or 0
            ),
            "family_follow_up_branch_implementation_ready_count": int(
                branch_summary.get("family_follow_up_branch_implementation_ready_count", 0) or 0
            ),
            "family_follow_up_branch_native_gap_count": int(
                branch_summary.get("family_follow_up_branch_native_gap_count", 0) or 0
            ),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "branch_contract_scorecard": branch_contract,
        "recommended_next_step": (
            "move O1 branch coverage into broader actual-training validation while keeping product dispatch default-off"
        ),
        "open_branches": [branch for row in rows for branch in row["remaining_items"]],
        "promotion_blockers": [f"{row['optimizer_name']}_follow_up_remaining" for row in rows if row["remaining_branch_count"] > 0],
        "blocked_reasons": [],
        "notes": [
            "This report only exposes the remaining follow-up branches for already-running canaries.",
            "It does not enable runtime dispatch, native dispatch, or product training paths.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _row(name: str, config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "schema_version": 1,
        "optimizer_name": name,
        "title": str(config.get("title", name)),
        "evidence_anchors": list(config.get("evidence_anchors", [])),
        "base_canary_ready": report.get("ok") is True and report.get("selected_native_canary_ready") is True,
        "selected_optimizer_name": str(report.get("selected_optimizer_name", "")),
        "optimizer_family": str(report.get("optimizer_family", "")),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "product_native_dispatch_ready": report.get("product_native_dispatch_ready") is True,
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "remaining_items": list(config.get("remaining_items", [])),
        "remaining_branch_count": len(config.get("remaining_items", [])),
        "next_step": (
            f"close {name} follow-up branches while keeping default-off"
            if config.get("remaining_items")
            else "no remaining branch"
        ),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _open_branch_titles_by_family(branch_contract: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in branch_contract.get("rows", []):
        if not isinstance(row, Mapping) or bool(row.get("branch_implementation_ready")):
            continue
        family = str(row.get("family") or "")
        title = str(row.get("title") or row.get("branch_id") or "")
        if family and title:
            result.setdefault(family, []).append(title)
    return result


def _with_open_branches(row: dict[str, Any], open_items: list[str]) -> dict[str, Any]:
    updated = dict(row)
    updated["remaining_items"] = list(open_items)
    updated["remaining_branch_count"] = len(open_items)
    updated["next_step"] = (
        f"close {updated['optimizer_name']} follow-up branches while keeping default-off"
        if open_items
        else "no remaining branch"
    )
    return updated


__all__ = ["build_family_follow_up_scorecard"]
