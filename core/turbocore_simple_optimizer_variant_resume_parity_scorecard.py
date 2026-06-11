"""Resume-parity implementation evidence for simple optimizer variants.

This scorecard consumes the report-only native canary/state evidence and runs
only internal canary probes.  It keeps request/UI/schema/product dispatch off.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_variant_native_canary_scorecard import (
    build_simple_optimizer_variant_native_canary_scorecard,
)
from core.turbocore_simple_optimizer_variant_state_scorecard import (
    build_simple_optimizer_variant_state_scorecard,
)
from core.turbocore_simple_quantized_optimizer_training_executor import (
    build_simple_quantized_optimizer_training_executor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
QUANTIZED_TARGETS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)
SCHEDULE_FREE_TARGETS = (
    OptimizerType.RADAM_SCHEDULE_FREE,
    OptimizerType.SGD_SCHEDULE_FREE,
)
KIND_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit",
    OptimizerType.PAGED_LION_8BIT: "paged_lion8bit",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit",
}


def build_simple_optimizer_variant_resume_parity_scorecard(
    *,
    variant_canary_report: Mapping[str, Any] | None = None,
    variant_state_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build default-off resume parity evidence for quantized and schedule-free variants."""

    root = Path(workspace_root or REPO_ROOT)
    canary = dict(variant_canary_report or build_simple_optimizer_variant_native_canary_scorecard())
    state = dict(variant_state_report or build_simple_optimizer_variant_state_scorecard())
    quantized_cases = _quantized_resume_cases(root, canary) if torch.cuda.is_available() else []
    schedule_cases = _schedule_free_resume_cases(state)
    rows = [_quantized_row(optimizer, canary, _case_for(optimizer, quantized_cases)) for optimizer in QUANTIZED_TARGETS]
    rows.extend(_schedule_free_row(optimizer, canary, schedule_cases) for optimizer in SCHEDULE_FREE_TARGETS)
    ready_count = sum(1 for row in rows if row["resume_parity_matrix_implementation_ready"] is True)
    blockers = _dedupe(reason for row in rows for reason in _strings(row.get("blocked_reasons")))
    ready = ready_count == len(rows)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_variant_resume_parity_scorecard_v0",
        "gate": "simple_formula_variant_resume_parity_implementation",
        "ok": ready and not blockers,
        "promotion_ready": False,
        "variant_resume_parity_matrix_implementation_ready": ready and not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in QUANTIZED_TARGETS + SCHEDULE_FREE_TARGETS],
        "rows": rows,
        "quantized_resume_cases": quantized_cases,
        "schedule_free_resume_cases": schedule_cases,
        "variant_canary_summary": dict(canary.get("summary") or {}),
        "variant_state_summary": dict(state.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(rows),
            "resume_parity_matrix_implementation_ready_count": ready_count,
            "quantized_resume_parity_ready_count": sum(
                1 for row in rows if row["variant_kind"] == "quantized_state" and row["resume_parity_matrix_implementation_ready"]
            ),
            "schedule_free_resume_parity_ready_count": sum(
                1
                for row in rows
                if row["variant_kind"] == "schedule_free_state_machine"
                and row["resume_parity_matrix_implementation_ready"]
            ),
            "quantized_resume_case_count": len(quantized_cases),
            "schedule_free_resume_case_count": len(schedule_cases),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers + ["simple_variant_product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold simple variant native dispatch for explicit owner/release approval"
            if ready
            else "finish simple variant resume parity blockers before dispatch review"
        ),
        "notes": [
            "Quantized rows prove state_dict -> executor restore and next-step parity against the native canary executor.",
            "Schedule-free rows consume the existing state_dict roundtrip and next-step parity state-machine case.",
            "The scorecard is internal evidence only; request/UI/schema/product dispatch remain disabled.",
        ],
    }


def _quantized_resume_cases(root: Path, canary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_run_quantized_resume_case(optimizer, root) for optimizer in QUANTIZED_TARGETS if _canary_ready(optimizer, canary)]


def _run_quantized_resume_case(optimizer: OptimizerType, root: Path) -> dict[str, Any]:
    kind = KIND_BY_OPTIMIZER[optimizer]
    value = torch.linspace(-0.25, 0.35, steps=512, device="cuda", dtype=torch.float32)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), device="cuda", dtype=torch.float32)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), device="cuda", dtype=torch.float32)
    param = torch.nn.Parameter(value.detach().clone())
    optimizer_a = _make_optimizer(param, kind)
    executor_a = build_simple_quantized_optimizer_training_executor(
        optimizer=optimizer_a,
        params=[param],
        config={"optimizer_kind": kind},
        workspace_root=root,
    )
    first = _step_executor(executor_a, param, grad1)
    saved_state = copy.deepcopy(optimizer_a.state_dict())
    saved_param = param.detach().clone()

    restored_param = torch.nn.Parameter(saved_param.detach().clone())
    optimizer_b = _make_optimizer(restored_param, kind)
    optimizer_b.load_state_dict(saved_state)
    executor_b = build_simple_quantized_optimizer_training_executor(
        optimizer=optimizer_b,
        params=[restored_param],
        config={"optimizer_kind": kind},
        workspace_root=root,
    )
    restore = executor_b.restore_optimizer_state_from_pytorch(reason="simple_variant_resume_parity_probe")
    second_a = _step_executor(executor_a, param, grad2)
    second_b = _step_executor(executor_b, restored_param, grad2)
    state_diff = _state_diff(optimizer_a, optimizer_b)
    param_diff = _max_abs(param.detach(), restored_param.detach())
    ok = (
        first.get("ok") is True
        and second_a.get("ok") is True
        and second_b.get("ok") is True
        and restore.get("restored") is True
        and param_diff <= 1e-6
        and state_diff["state_q_equal"]
        and state_diff["scale_max_diff"] <= 1e-7
        and state_diff["step_equal"]
    )
    return {
        "schema_version": 1,
        "case": "quantized_state_dict_restore_next_step",
        "optimizer_type": optimizer.value,
        "optimizer_kind": kind,
        "ok": ok,
        "resume_parity": ok,
        "state_dict_restore_ready": restore.get("restored") is True,
        "next_step_after_restore_ready": second_a.get("ok") is True and second_b.get("ok") is True,
        "max_param_diff": param_diff,
        "state_diff": state_diff,
        "tolerance": 1e-6,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else ["simple_quantized_resume_parity_failed"],
    }


def _schedule_free_resume_cases(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case in state.get("schedule_free_cases", []):
        if isinstance(case, Mapping) and case.get("case") == "roundtrip_state_machine":
            cases.append(dict(case))
    return cases


def _quantized_row(
    optimizer: OptimizerType,
    canary: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    canary_ready = _canary_ready(optimizer, canary)
    ready = canary_ready and case.get("ok") is True
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
        "optimizer_family": "simple_formula",
        "variant_kind": "quantized_state",
        "variant_status": "resume_parity_implementation_ready" if ready else "resume_parity_implementation_blocked",
        "resume_parity_matrix_implementation_ready": ready,
        "source_native_canary_ready": canary_ready,
        "state_dict_restore_ready": case.get("state_dict_restore_ready") is True,
        "next_step_after_restore_ready": case.get("next_step_after_restore_ready") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else _dedupe(["quantized_resume_case_missing"] + _strings(case.get("blocked_reasons"))),
    }


def _schedule_free_row(
    optimizer: OptimizerType,
    canary: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
) -> dict[str, Any]:
    case = _case_for(optimizer, cases)
    canary_ready = _canary_ready(optimizer, canary)
    ready = canary_ready and case.get("ok") is True and case.get("covers_resume") is True
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": "radam_schedule_free" if optimizer == OptimizerType.RADAM_SCHEDULE_FREE else "sgd_schedule_free",
        "optimizer_family": "simple_formula",
        "variant_kind": "schedule_free_state_machine",
        "variant_status": "resume_parity_implementation_ready" if ready else "resume_parity_implementation_blocked",
        "resume_parity_matrix_implementation_ready": ready,
        "source_native_canary_ready": canary_ready,
        "state_dict_restore_ready": case.get("ok") is True,
        "next_step_after_restore_ready": case.get("ok") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else _dedupe(["schedule_free_resume_case_missing"] + _strings(case.get("blocked_reasons"))),
    }


def _step_executor(
    executor: Any,
    param: torch.nn.Parameter,
    grad: torch.Tensor,
) -> dict[str, Any]:
    param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)
    return dict(executor({"training_dispatch": True, "training_path_enabled": True}))


def _make_optimizer(param: torch.nn.Parameter, kind: str) -> torch.optim.Optimizer:
    if kind == "sgd_nesterov8bit":
        return torch.optim.SGD([param], lr=1e-2, momentum=0.9, weight_decay=0.01, nesterov=True)
    return torch.optim.SGD([param], lr=1e-3, momentum=0.0, weight_decay=0.01)


def _state_diff(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> dict[str, Any]:
    left_state = _first_state(left)
    right_state = _first_state(right)
    left_q = left_state.get("turbocore_quantized_state_q")
    right_q = right_state.get("turbocore_quantized_state_q")
    left_scale = left_state.get("turbocore_quantized_scale")
    right_scale = right_state.get("turbocore_quantized_scale")
    return {
        "state_q_equal": isinstance(left_q, torch.Tensor)
        and isinstance(right_q, torch.Tensor)
        and bool(torch.equal(left_q.detach().cpu(), right_q.detach().cpu())),
        "scale_max_diff": _max_abs(left_scale, right_scale)
        if isinstance(left_scale, torch.Tensor) and isinstance(right_scale, torch.Tensor)
        else float("inf"),
        "step_equal": int(left_state.get("turbocore_quantized_optimizer_step", -1) or -1)
        == int(right_state.get("turbocore_quantized_optimizer_step", -2) or -2),
    }


def _first_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict().get("state", {})
    if not isinstance(state, Mapping) or not state:
        return {}
    first = next(iter(state.values()))
    return dict(first) if isinstance(first, Mapping) else {}


def _case_for(optimizer: OptimizerType, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    for case in cases:
        if case.get("optimizer_type") == optimizer.value:
            return dict(case)
    return {}


def _canary_ready(optimizer: OptimizerType, canary: Mapping[str, Any]) -> bool:
    for row in canary.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value:
            return row.get("native_canary_ready") is True and _row_default_off(row)
    return False


def _row_default_off(row: Mapping[str, Any]) -> bool:
    return (
        row.get("training_path_enabled") is False
        and row.get("default_behavior_changed") is False
        and row.get("native_dispatch_allowed") is False
        and row.get("product_native_dispatch_ready") is False
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_variant_resume_parity_scorecard"]
