"""Report-only training tensor binding canary for AnimaFactoredAdamW."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.lulynx_trainer.anima_factored_optimizer import AnimaFactoredAdamW
from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_anima_factored_adamw_native_scratch_kernel_scorecard import (
    build_anima_factored_adamw_native_scratch_kernel_scorecard,
)


ENTRYPOINT = "probe_anima_factored_adamw_training_tensor_binding_canary_py"
PROBE_KIND = "anima_factored_adamw_training_tensor_binding_canary_v0"
FLOAT_TOLERANCE = 5e-5
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_anima_factored_adamw_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(scratch_report or build_anima_factored_adamw_native_scratch_kernel_scorecard(workspace_root=workspace_root))
    live_probe = (
        _live_probe(workspace_root=workspace_root, arch=arch)
        if run_live_probe
        else _skipped_live_probe("live_probe_disabled")
    )
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard_v0",
        "gate": "anima_factored_adamw_training_tensor_binding_canary",
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
        "optimizer_kind": "anima_factored_adamw",
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
            "training_tensor_binding_parity_passed": bool(
                live_probe.get("training_tensor_binding_parity_passed", False)
            ),
            "e2e_no_regression_passed": bool(live_probe.get("e2e_no_regression_passed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_state_diff": live_probe.get("max_state_diff"),
            "loss_diff": live_probe.get("loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "anima_factored_adamw_runtime_dispatch_shadow_missing",
                "anima_factored_adamw_training_loop_canary_missing",
                "anima_factored_adamw_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add AnimaFactoredAdamW runtime dispatch shadow before TrainingLoop canary"
            if ready
            else "fix AnimaFactoredAdamW training tensor binding canary blockers"
        ),
        "notes": [
            "This canary uses isolated clone tensors and real autograd gradients.",
            "It binds Parameter, grad, exp_avg, factored row/col state, and full exp_avg_sq buffers to the native kernel.",
            "It keeps native optimizer dispatch disabled and never touches user training runs.",
        ],
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed_live_probe(
            "lulynx_native_entrypoint_missing",
            ["anima_factored_adamw_training_tensor_binding_entrypoint_missing"],
        )
    cases = [
        _run_live_case(native, "factored_256x256", (256, 256), workspace_root, arch),
        _run_live_case(native, "unfactored_4x4", (4, 4), workspace_root, arch),
    ]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    max_param_diff = max((float(case.get("max_param_diff", 0.0) or 0.0) for case in cases), default=0.0)
    max_state_diff = max((float(case.get("max_state_diff", 0.0) or 0.0) for case in cases), default=0.0)
    loss_diff = max((float(case.get("loss_diff", 0.0) or 0.0) for case in cases), default=0.0)
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
        "max_param_diff": max_param_diff,
        "max_state_diff": max_state_diff,
        "loss_diff": loss_diff,
        "blocked_reasons": _dedupe([str(reason) for case in failed for reason in case.get("blocked_reasons", []) or []]),
    }


def _run_live_case(
    native: Any,
    case_name: str,
    shape: tuple[int, int],
    workspace_root: str | Path | None,
    arch: str | None,
) -> dict[str, Any]:
    try:
        rows, cols = shape
        value = torch.linspace(-0.15, 0.15, steps=rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
        prime_param = torch.nn.Parameter(value.clone())
        prime_optimizer = _make_optimizer(prime_param)
        prime_loss = _backward_loss(prime_param, scale=0.35)
        prime_optimizer.step()
        prime_optimizer.zero_grad(set_to_none=True)
        checkpoint = copy.deepcopy(prime_optimizer.state_dict())
        saved_param = prime_param.detach().clone()

        reference_param = torch.nn.Parameter(saved_param.clone())
        reference_optimizer = _make_optimizer(reference_param)
        reference_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        reference_loss = _backward_loss(reference_param, scale=0.7)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        candidate_param = torch.nn.Parameter(saved_param.clone())
        candidate_optimizer = _make_optimizer(candidate_param)
        candidate_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        candidate_loss = _backward_loss(candidate_param, scale=0.7)
        state = candidate_optimizer.state[candidate_param]
        group = candidate_optimizer.param_groups[0]
        factored = bool(state.get("factored", False))
        next_step = int(state["step"].item()) + 1
        row_tensor = state["exp_avg_sq_row"].contiguous() if factored else torch.zeros(rows, device="cuda", dtype=torch.float32)
        col_tensor = state["exp_avg_sq_col"].contiguous() if factored else torch.zeros(cols, device="cuda", dtype=torch.float32)
        exp_avg_sq = (
            torch.zeros_like(candidate_param.detach()).contiguous()
            if factored
            else state["exp_avg_sq"].contiguous()
        )
        launch_config = {
            "factored": factored,
            "rows": rows,
            "cols": cols,
            "step": next_step,
            "lr": float(group["lr"]),
            "beta1": float(group["betas"][0]),
            "beta2": float(group["betas"][1]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "factored_eps": float(group["factored_eps"]),
            "max_numel": max(rows * cols, 1_048_576),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                candidate_param.grad.detach().contiguous(),
                state["exp_avg"].contiguous(),
                row_tensor,
                col_tensor,
                exp_avg_sq,
                json.dumps(launch_config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _case_failed(
                case_name,
                f"anima_factored_adamw_training_tensor_binding_launch_failed:{launch.get('reason', 'unknown')}",
                launch=launch,
            )
        state["step"] += 1
        if factored:
            state["exp_avg_sq_row"] = row_tensor.reshape(rows, 1)
            state["exp_avg_sq_col"] = col_tensor.reshape(1, cols)
        else:
            state["exp_avg_sq"] = exp_avg_sq
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_state = reference_optimizer.state[reference_param]
        param_compare = _compare_tensor(reference_param.detach(), candidate_param.detach())
        state_compare = _compare_state(reference_state, state, factored=factored)
        current_loss_diff = abs(float(reference_loss) - float(candidate_loss))
        finite = all(torch.isfinite(torch.tensor(v)).item() for v in [prime_loss, reference_loss, candidate_loss])
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
            "prime_loss": float(prime_loss),
            "reference_loss": float(reference_loss),
            "candidate_loss": float(candidate_loss),
            "loss_diff": current_loss_diff,
            "losses_finite": finite,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_state_diff": state_compare.get("max_state_diff"),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, finite),
        }
    except Exception as exc:
        return _case_failed(case_name, f"anima_factored_adamw_training_tensor_binding_probe_failed:{type(exc).__name__}: {exc}")


def _make_optimizer(param: torch.nn.Parameter) -> AnimaFactoredAdamW:
    return AnimaFactoredAdamW(
        [param],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        min_dim=128,
        min_numel=65536,
        factored_eps=1e-30,
    )


def _backward_loss(param: torch.nn.Parameter, *, scale: float) -> float:
    if param.grad is not None:
        param.grad = None
    values = param.float()
    loss = ((values * float(scale)).pow(2).mean() + values.mean() * 0.013)
    loss.backward()
    return float(loss.detach().cpu())


def _validations(scratch: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(live_probe.get("training_tensor_binding_parity_passed", False))
    return [
        _validation(
            "p24_anima_factored_adamw_scratch_kernel_ready",
            bool(scratch.get("anima_factored_adamw_native_kernel_parity", False)),
            "anima_factored_adamw_native_scratch_kernel_missing",
        ),
        _validation(
            "training_tensor_binding_probe_or_skip",
            live_ready,
            "anima_factored_adamw_training_tensor_binding_canary_failed",
        ),
        _validation(
            "factored_and_unfactored_live_cases",
            live_status == "skipped" or int(live_probe.get("passed_case_count", 0) or 0) >= 2,
            "anima_factored_adamw_training_tensor_binding_case_coverage_missing",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "anima_factored_adamw_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    return {
        "schema_version": 1,
        "ok": same_shape and same_dtype and max_diff <= FLOAT_TOLERANCE,
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "max_diff": max_diff,
        "tolerance": FLOAT_TOLERANCE,
    }


def _compare_state(left: Mapping[str, Any], right: Mapping[str, Any], *, factored: bool) -> dict[str, Any]:
    exp_avg_diff = _max_abs(left["exp_avg"], right["exp_avg"])
    step_match = int(left["step"].item()) == int(right["step"].item())
    if factored:
        row_diff = _max_abs(left["exp_avg_sq_row"], right["exp_avg_sq_row"])
        col_diff = _max_abs(left["exp_avg_sq_col"], right["exp_avg_sq_col"])
        max_state_diff = max(exp_avg_diff, row_diff, col_diff)
        ok = max_state_diff <= FLOAT_TOLERANCE and step_match
        return {
            "schema_version": 1,
            "ok": ok,
            "factored": True,
            "exp_avg_max_diff": exp_avg_diff,
            "row_max_diff": row_diff,
            "col_max_diff": col_diff,
            "max_state_diff": max_state_diff,
            "step_match": step_match,
            "tolerance": FLOAT_TOLERANCE,
        }
    exp_avg_sq_diff = _max_abs(left["exp_avg_sq"], right["exp_avg_sq"])
    max_state_diff = max(exp_avg_diff, exp_avg_sq_diff)
    ok = max_state_diff <= FLOAT_TOLERANCE and step_match
    return {
        "schema_version": 1,
        "ok": ok,
        "factored": False,
        "exp_avg_max_diff": exp_avg_diff,
        "exp_avg_sq_max_diff": exp_avg_sq_diff,
        "max_state_diff": max_state_diff,
        "step_match": step_match,
        "tolerance": FLOAT_TOLERANCE,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _live_blockers(param_compare: Mapping[str, Any], state_compare: Mapping[str, Any], finite: bool) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("anima_factored_adamw_training_tensor_binding_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("anima_factored_adamw_training_tensor_binding_state_parity_failed")
    if not finite:
        blockers.append("anima_factored_adamw_training_tensor_binding_non_finite_loss")
    return blockers


def _case_failed(case_name: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": case_name,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [reason],
        **extra,
    }


def _failed_live_probe(reason: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed_case_count": 0,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed_case_count": 0,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "PROBE_KIND", "build_anima_factored_adamw_training_tensor_binding_canary_scorecard"]
