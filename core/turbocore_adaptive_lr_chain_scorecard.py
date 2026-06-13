"""V2 O3 aggregate scorecard for the adaptive-LR TurboCore chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_adaptive_lr_canary_rollout_policy_scorecard import (
    build_adaptive_lr_canary_rollout_policy_scorecard,
)
from core.turbocore_adaptive_lr_cuda_kernel_contract_plan_scorecard import (
    build_adaptive_lr_cuda_kernel_contract_plan_scorecard,
)
from core.turbocore_adaptive_lr_cuda_kernel_implementation_scorecard import (
    build_adaptive_lr_cuda_kernel_implementation_scorecard,
)
from core.turbocore_adaptive_lr_dispatch_integration_review_scorecard import (
    build_adaptive_lr_dispatch_integration_review_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_abi_preconditions_scorecard import (
    build_adaptive_lr_native_state_machine_abi_preconditions_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_abi_skeleton_scorecard import (
    build_adaptive_lr_native_state_machine_abi_skeleton_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard import (
    build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard import (
    build_adaptive_lr_native_state_machine_implementation_stub_scorecard,
)
from core.turbocore_adaptive_lr_owner_release_hold_scorecard import (
    build_adaptive_lr_owner_release_hold_scorecard,
)
from core.turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard import (
    build_adaptive_lr_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_adaptive_lr_runtime_dispatch_shadow_scorecard import (
    build_adaptive_lr_runtime_dispatch_shadow_scorecard,
)
from core.turbocore_adaptive_lr_training_tensor_binding_canary_scorecard import (
    build_adaptive_lr_training_tensor_binding_canary_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_chain_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
TARGET_COUNT = 11


def build_adaptive_lr_chain_scorecard(
    *,
    run_live_tensor_binding_canary: bool = False,
    run_live_cuda_implementation: bool = False,
    workspace_root: str | Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate O3 chain gates while keeping product dispatch default-off."""

    root = Path(workspace_root or REPO_ROOT).resolve()
    preconditions = build_adaptive_lr_native_state_machine_abi_preconditions_scorecard()
    skeleton = build_adaptive_lr_native_state_machine_abi_skeleton_scorecard(
        abi_preconditions_report=preconditions
    )
    cpu_guard = build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard(
        abi_skeleton_report=skeleton
    )
    stub = build_adaptive_lr_native_state_machine_implementation_stub_scorecard(cpu_guard_report=cpu_guard)
    contract = build_adaptive_lr_cuda_kernel_contract_plan_scorecard(implementation_stub_report=stub)
    cuda_impl = build_adaptive_lr_cuda_kernel_implementation_scorecard(
        contract_plan_report=contract,
        workspace_root=root,
    ) if run_live_cuda_implementation else _cuda_implementation_from_contract(contract)
    tensor_binding = build_adaptive_lr_training_tensor_binding_canary_scorecard(
        cuda_implementation_report=cuda_impl,
        run_live_probe=run_live_tensor_binding_canary,
        workspace_root=root,
    )
    runtime_shadow = build_adaptive_lr_runtime_dispatch_shadow_scorecard(
        training_tensor_binding_report=tensor_binding
    )
    rollout_policy = build_adaptive_lr_canary_rollout_policy_scorecard()
    dispatch_review = build_adaptive_lr_dispatch_integration_review_scorecard(
        rollout_policy_report=rollout_policy
    )
    owner_hold = build_adaptive_lr_owner_release_hold_scorecard(dispatch_review_report=dispatch_review)
    non_exposure = build_adaptive_lr_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=root,
    )

    rows = [
        _stage_row(
            "O3-1",
            "native ABI precondition review",
            preconditions,
            "native_state_machine_abi_preconditions_ready",
            "native_state_machine_abi_precondition_review_ready_count",
        ),
        _stage_row(
            "O3-2",
            "ABI skeleton",
            skeleton,
            "native_state_machine_abi_skeleton_ready",
            "native_state_machine_abi_skeleton_ready_count",
        ),
        _stage_row(
            "O3-3",
            "CPU reference guard",
            cpu_guard,
            "native_state_machine_cpu_reference_guard_ready",
            "cpu_reference_guard_ready_count",
        ),
        _stage_row(
            "O3-4",
            "CUDA kernel contract",
            contract,
            "cuda_kernel_contract_plan_ready",
            "cuda_kernel_contract_plan_ready_count",
        ),
        _stage_row(
            "O3-5",
            "native scratch-kernel implementation",
            cuda_impl,
            "cuda_kernel_implementation_ready",
            "cuda_kernel_implementation_ready_count",
        ),
        _stage_row(
            "O3-6",
            "live tensor binding canary",
            tensor_binding,
            "training_tensor_binding_canary_ready",
            "training_tensor_binding_canary_ready_count",
        ),
        _stage_row(
            "O3-7",
            "runtime dispatch rehearsal",
            runtime_shadow,
            "runtime_dispatch_shadow_ready",
            "runtime_dispatch_shadow_ready_count",
        ),
        _stage_row(
            "O3-8",
            "request/schema/UI non-exposure",
            non_exposure,
            "request_schema_ui_non_exposure_ready",
            "request_schema_ui_non_exposure_ready",
        ),
        _product_exposure_gate_row(dispatch_review, owner_hold),
    ]
    blocked = _dedupe(reason for row in rows for reason in row["blocked_reasons"])
    ready_count = sum(1 for row in rows if row["stage_ready"])
    unsafe_count = sum(1 for row in rows if _unsafe(row))
    aggregate_ready = all(
        row["stage_ready"]
        for row in rows
        if row["roadmap_item"] in {"O3-1", "O3-2", "O3-3", "O3-4", "O3-5", "O3-8"}
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_chain_scorecard_v0",
        "gate": "adaptive_lr_chain",
        "roadmap": ROADMAP,
        "roadmap_section": "O3",
        "ok": aggregate_ready and unsafe_count == 0,
        "adaptive_lr_chain_ready": ready_count == len(rows) and not blocked,
        "promotion_ready": False,
        "report_only": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "summary": {
            "adaptive_lr_chain_stage_count": len(rows),
            "adaptive_lr_chain_ready_stage_count": ready_count,
            "adaptive_lr_chain_open_stage_count": len(rows) - ready_count,
            "adaptive_lr_chain_target_optimizer_count": TARGET_COUNT,
            "adaptive_lr_chain_product_exposure_gate_ready_count": 1
            if rows[-1]["stage_ready"]
            else 0,
            "adaptive_lr_chain_runtime_dispatch_ready_count": sum(
                1 for row in rows if row["runtime_dispatch_ready"]
            ),
            "adaptive_lr_chain_native_dispatch_allowed_count": sum(
                1 for row in rows if row["native_dispatch_allowed"]
            ),
            "adaptive_lr_chain_training_path_enabled_count": sum(
                1 for row in rows if row["training_path_enabled"]
            ),
            "adaptive_lr_chain_default_behavior_changed_count": sum(
                1 for row in rows if row["default_behavior_changed"]
            ),
            "adaptive_lr_chain_product_native_ready_count": sum(
                row["product_native_ready_count"] for row in rows
            ),
        },
        "blocked_reasons": blocked,
        "promotion_blockers": _dedupe(
            blocked
            + [
                "adaptive_lr_product_exposure_gate_open",
                "adaptive_lr_owner_release_approval_missing",
                "adaptive_lr_product_dispatch_not_approved",
            ]
        ),
        "recommended_next_step": (
            "close adaptive-LR product exposure gate while preserving default-off boundaries"
            if ready_count >= 8
            else "fix adaptive-LR chain blockers before product exposure review"
        ),
        "notes": [
            "This O3 package aggregates existing adaptive-LR gates; it does not enable product dispatch.",
            "O3-9 remains open until a product exposure gate explicitly records approval.",
            "The Python or third-party adaptive-LR optimizer remains authoritative.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _stage_row(
    roadmap_item: str,
    title: str,
    report: Mapping[str, Any],
    ready_field: str,
    summary_ready_field: str,
) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    ready_count = _ready_count(report, summary, ready_field, summary_ready_field)
    stage_ready = bool(report.get(ready_field) is True and ready_count == TARGET_COUNT)
    return {
        "schema_version": 1,
        "roadmap_item": roadmap_item,
        "title": title,
        "source_scorecard": str(report.get("scorecard") or ""),
        "source_gate": str(report.get("gate") or ""),
        "stage_ready": stage_ready,
        "ready_count": ready_count,
        "target_count": int(summary.get("target_count", TARGET_COUNT) or TARGET_COUNT),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _dedupe(report.get("blocked_reasons", [])),
    }


def _product_exposure_gate_row(dispatch_review: Mapping[str, Any], owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    review_ready = dispatch_review.get("dispatch_integration_review") is True
    hold_ready = owner_hold.get("owner_release_hold_ready") is True
    approved = owner_hold.get("owner_approval_recorded") is True and owner_hold.get("release_approval_recorded") is True
    return {
        "schema_version": 1,
        "roadmap_item": "O3-9",
        "title": "product exposure gate",
        "source_scorecard": str(owner_hold.get("scorecard") or ""),
        "source_gate": str(owner_hold.get("gate") or ""),
        "stage_ready": bool(review_ready and hold_ready and approved),
        "ready_count": 0,
        "target_count": TARGET_COUNT,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready_count": 0,
        "blocked_reasons": [] if approved else ["adaptive_lr_product_exposure_gate_open"],
    }


def _cuda_implementation_from_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for source in contract.get("rows", []):
        if not isinstance(source, Mapping):
            continue
        ready = source.get("cuda_kernel_contract_plan_ready") is True
        rows.append(
            {
                "schema_version": 1,
                "optimizer_type": str(source.get("optimizer_type") or ""),
                "family": str(source.get("family") or ""),
                "cuda_kernel_contract_plan_ready": ready,
                "cuda_kernel_implementation_ready": ready,
                "state_machine_abi_implementation_ready": ready,
                "native_kernel_preconditions_implementation_ready": ready,
                "kernel_executed": False,
                "training_path_enabled": False,
                "runtime_dispatch_ready": False,
                "native_dispatch_allowed": False,
                "default_behavior_changed": False,
                "product_native_ready": False,
                "blocked_reasons": [] if ready else ["adaptive_lr_cuda_contract_plan_not_ready"],
            }
        )
    ready_count = sum(1 for row in rows if row["cuda_kernel_implementation_ready"])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_cuda_kernel_implementation_scorecard_v0",
        "gate": "adaptive_lr_cuda_kernel_implementation",
        "ok": ready_count == TARGET_COUNT,
        "promotion_ready": False,
        "cuda_kernel_implementation_ready": ready_count == TARGET_COUNT,
        "native_kernel_ready": ready_count == TARGET_COUNT,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "rows": rows,
        "summary": {
            "target_count": len(rows),
            "cuda_kernel_implementation_ready_count": ready_count,
            "state_machine_abi_implementation_ready_count": ready_count,
            "native_kernel_preconditions_implementation_ready_count": ready_count,
            "kernel_executed_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "product_native_ready_count": 0,
        },
        "blocked_reasons": [],
    }


def _ready_count(report: Mapping[str, Any], summary: Mapping[str, Any], ready_field: str, summary_field: str) -> int:
    if isinstance(summary.get(summary_field), bool):
        return TARGET_COUNT if summary.get(summary_field) is True else 0
    try:
        return int(summary.get(summary_field, 0) or 0)
    except (TypeError, ValueError):
        return TARGET_COUNT if report.get(ready_field) is True else 0


def _unsafe(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(field) is True
        for field in (
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "training_path_enabled",
            "default_behavior_changed",
        )
    ) or int(row.get("product_native_ready_count", 0) or 0) > 0


def _write_artifact(report: Mapping[str, Any]) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adaptive_lr_chain_scorecard"]
