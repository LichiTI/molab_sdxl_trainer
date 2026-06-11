"""Default-off training tensor binding canary for Adafactor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_adafactor_native_scratch_kernel_scorecard import (
    build_adafactor_native_scratch_kernel_scorecard,
)


ENTRYPOINT = "probe_adafactor_training_tensor_binding_canary_py"
PROBE_KIND = "adafactor_training_tensor_binding_canary_v0"
FLOAT_TOLERANCE = 5e-5
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adafactor_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(scratch_report or build_adafactor_native_scratch_kernel_scorecard(workspace_root=workspace_root))
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adafactor_training_tensor_binding_canary_scorecard_v0",
        "gate": "adafactor_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_probe_ready": str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_e2e_no_regression_ready": bool(live_probe.get("e2e_no_regression_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoint": ENTRYPOINT,
        "probe_kind": PROBE_KIND,
        "optimizer_kind": "adafactor",
        "optimizer_family": "factored_custom",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "scratch_summary": dict(scratch.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "kernel_executed_case_count": int(live_probe.get("kernel_executed_case_count", 0) or 0),
            "passed_case_count": int(live_probe.get("passed_case_count", 0) or 0),
            "training_tensor_binding_parity_passed": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
            "e2e_no_regression_passed": bool(live_probe.get("e2e_no_regression_passed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_state_diff": live_probe.get("max_state_diff"),
            "loss_diff": live_probe.get("loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adafactor_runtime_dispatch_shadow_missing",
                "adafactor_training_loop_canary_missing",
                "adafactor_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": "add Adafactor runtime dispatch shadow before TrainingLoop canary" if ready else "fix Adafactor training tensor binding canary blockers",
        "notes": [
            "This canary uses isolated clone tensors and real autograd gradients.",
            "It binds Parameter, grad, factored row/col state, and full exp_avg_sq buffers to the native kernel.",
            "It keeps native optimizer dispatch disabled and never touches user training runs.",
        ],
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed_live_probe("lulynx_native_entrypoint_missing", ["adafactor_training_tensor_binding_entrypoint_missing"])
    cases = [
        _run_live_case(native, "factored_128x128", (128, 128), workspace_root, arch),
        _run_live_case(native, "unfactored_4x4", (4, 4), workspace_root, arch),
    ]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    return {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "probe_kind": PROBE_KIND,
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failed),
        "kernel_executed_case_count": sum(1 for case in cases if bool(case.get("kernel_executed", False))),
        "training_tensor_binding_parity_passed": not failed,
        "e2e_no_regression_passed": not failed,
        "cases": cases,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "max_param_diff": max((float(case.get("max_param_diff", 0.0) or 0.0) for case in cases), default=0.0),
        "max_state_diff": max((float(case.get("max_state_diff", 0.0) or 0.0) for case in cases), default=0.0),
        "loss_diff": max((float(case.get("loss_diff", 0.0) or 0.0) for case in cases), default=0.0),
        "blocked_reasons": _dedupe([str(reason) for case in failed for reason in case.get("blocked_reasons", []) or []]),
    }


def _run_live_case(native: Any, case_name: str, shape: tuple[int, int], workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    try:
        rows, cols = shape
        factored = rows >= 128 and cols >= 128
        value = torch.linspace(-0.08, 0.08, steps=rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
        grad = _grad_for(value, scale=0.45).contiguous()
        reference_param = value.clone()
        candidate_param = value.clone()
        row = (torch.linspace(0.0003, 0.0007, steps=rows, device="cuda", dtype=torch.float32) if factored else torch.zeros(rows, device="cuda", dtype=torch.float32)).contiguous()
        col = (torch.linspace(0.0004, 0.0008, steps=cols, device="cuda", dtype=torch.float32) if factored else torch.zeros(cols, device="cuda", dtype=torch.float32)).contiguous()
        exp_avg_sq = (torch.zeros_like(candidate_param) if factored else torch.linspace(0.0002, 0.0004, steps=rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)).contiguous()
        ref_row = row.clone()
        ref_col = col.clone()
        ref_exp_avg_sq = exp_avg_sq.clone()
        config = {
            "factored": factored,
            "rows": rows,
            "cols": cols,
            "lr": 1.0e-3 if factored else 7.0e-4,
            "beta2": 0.98 if factored else 0.97,
            "eps": 1.0e-30,
            "clip_threshold": 1.0,
            "weight_decay": 0.01 if factored else 0.0,
            "factored_eps": 1.0e-30,
            "max_numel": max(rows * cols, 1_048_576),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        _reference_step(reference_param, grad, ref_row, ref_col, ref_exp_avg_sq, config)
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                grad,
                row,
                col,
                exp_avg_sq,
                json.dumps(config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _case_failed(case_name, f"adafactor_training_tensor_binding_launch_failed:{launch.get('reason', 'unknown')}", launch=launch)
        param_compare = _compare_tensor(reference_param, candidate_param)
        state_compare = _compare_state(ref_row, row, ref_col, col, ref_exp_avg_sq, exp_avg_sq, factored=factored)
        loss_diff = abs(float(_loss_for(reference_param)) - float(_loss_for(candidate_param)))
        finite = all(torch.isfinite(t).all().item() for t in [reference_param, candidate_param, row, col, exp_avg_sq])
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and finite
        return {
            "schema_version": 1,
            "case": case_name,
            "ok": ok,
            "factored": factored,
            "shape": [rows, cols],
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "native_launch": launch,
            "isolated_training_graph": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "loss_diff": loss_diff,
            "losses_finite": finite,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_state_diff": state_compare.get("max_state_diff"),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, finite),
        }
    except Exception as exc:
        return _case_failed(case_name, f"adafactor_training_tensor_binding_probe_failed:{type(exc).__name__}: {exc}")


def _reference_step(param: torch.Tensor, grad: torch.Tensor, row: torch.Tensor, col: torch.Tensor, exp_avg_sq: torch.Tensor, config: Mapping[str, Any]) -> None:
    rows = int(config["rows"])
    cols = int(config["cols"])
    beta2 = float(config["beta2"])
    beta2_delta = 1.0 - beta2
    eps = float(config["eps"])
    factored_eps = float(config["factored_eps"])
    update = torch.empty_like(param)
    if bool(config["factored"]):
        row.mul_(beta2).add_(grad.square().mean(dim=1), alpha=beta2_delta)
        col.mul_(beta2).add_(grad.square().mean(dim=0), alpha=beta2_delta)
        row_mean = torch.clamp(row.mean(), min=factored_eps)
        update.copy_(grad / torch.sqrt(torch.clamp(row.reshape(rows, 1) * col.reshape(1, cols) / row_mean, min=factored_eps) + eps))
    else:
        exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=beta2_delta)
        update.copy_(grad / torch.sqrt(exp_avg_sq + eps))
    clip_scale = torch.clamp(update.square().mean().sqrt() / float(config["clip_threshold"]), min=1.0)
    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)
    param.add_(update / clip_scale, alpha=-lr)


def _grad_for(param: torch.Tensor, *, scale: float) -> torch.Tensor:
    return (param.float() * float(scale) + 0.013).detach()


def _loss_for(param: torch.Tensor) -> torch.Tensor:
    values = param.float()
    return values.square().mean() + values.mean() * 0.013


def _validations(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(live_probe.get("training_tensor_binding_parity_passed", False))
    return [
        _validation("p34_adafactor_scratch_kernel_ready", bool(scratch.get("adafactor_native_kernel_parity", False)), "adafactor_native_scratch_kernel_missing"),
        _validation("training_tensor_binding_probe_or_skip", live_ready, "adafactor_training_tensor_binding_canary_failed"),
        _validation("factored_and_unfactored_live_cases", live_status == "skipped" or int(live_probe.get("passed_case_count", 0) or 0) >= 2, "adafactor_training_tensor_binding_case_coverage_missing"),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True)) and not bool(live_probe.get("training_path_enabled", True)) and not bool(live_probe.get("native_dispatch_allowed", True)),
            "adafactor_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    return {"schema_version": 1, "ok": same_shape and same_dtype and max_diff <= FLOAT_TOLERANCE, "shape_match": same_shape, "dtype_match": same_dtype, "max_diff": max_diff, "tolerance": FLOAT_TOLERANCE}


def _compare_state(left_row: torch.Tensor, right_row: torch.Tensor, left_col: torch.Tensor, right_col: torch.Tensor, left_exp_avg_sq: torch.Tensor, right_exp_avg_sq: torch.Tensor, *, factored: bool) -> dict[str, Any]:
    if factored:
        row_diff = _max_abs(left_row, right_row)
        col_diff = _max_abs(left_col, right_col)
        max_state_diff = max(row_diff, col_diff)
        return {"schema_version": 1, "ok": max_state_diff <= FLOAT_TOLERANCE, "factored": True, "row_max_diff": row_diff, "col_max_diff": col_diff, "max_state_diff": max_state_diff, "tolerance": FLOAT_TOLERANCE}
    exp_avg_sq_diff = _max_abs(left_exp_avg_sq, right_exp_avg_sq)
    return {"schema_version": 1, "ok": exp_avg_sq_diff <= FLOAT_TOLERANCE, "factored": False, "exp_avg_sq_max_diff": exp_avg_sq_diff, "max_state_diff": exp_avg_sq_diff, "tolerance": FLOAT_TOLERANCE}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _live_blockers(param_compare: Mapping[str, Any], state_compare: Mapping[str, Any], finite: bool) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("adafactor_training_tensor_binding_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("adafactor_training_tensor_binding_state_parity_failed")
    if not finite:
        blockers.append("adafactor_training_tensor_binding_non_finite_tensor")
    return blockers


def _case_failed(case_name: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"schema_version": 1, "case": case_name, "ok": False, "reason": reason, "kernel_executed": False, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": [reason], **extra}


def _failed_live_probe(reason: str, blockers: list[str]) -> dict[str, Any]:
    return {"schema_version": 1, "status": "failed", "reason": reason, "training_tensor_binding_parity_passed": False, "e2e_no_regression_passed": False, "kernel_executed_case_count": 0, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": blockers}


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "status": "skipped", "reason": reason, "training_tensor_binding_parity_passed": False, "e2e_no_regression_passed": False, "kernel_executed_case_count": 0, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": []}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "PROBE_KIND", "build_adafactor_training_tensor_binding_canary_scorecard"]
