"""Default-off live tensor binding canary for built-in adaptive-LR kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_adaptive_lr_cuda_kernel_implementation_scorecard import (
    build_adaptive_lr_cuda_kernel_implementation_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


ENTRYPOINT = "probe_adaptive_lr_training_tensor_binding_canary_py"
PROBE_KIND = "adaptive_lr_training_tensor_binding_canary_v0"
FLOAT_TOLERANCE = 5e-5
KERNEL_KIND_BY_FAMILY = {
    "adaptive_lr_prodigy": "prodigy",
    "adaptive_lr_dadapt": "dadapt",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adaptive_lr_training_tensor_binding_canary_scorecard(
    *,
    cuda_implementation_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run isolated CUDA tensor-binding canaries without enabling dispatch."""

    cuda_impl = _as_dict(cuda_implementation_report or build_adaptive_lr_cuda_kernel_implementation_scorecard())
    live_probe = _live_probe(workspace_root=workspace_root, arch=arch) if run_live_probe else _skipped("live_probe_disabled")
    rows = [_row(case.optimizer.value, cuda_impl, live_probe) for case in TARGET_CASES]
    validations = _validations(cuda_impl, live_probe, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_training_tensor_binding_canary_scorecard_v0",
        "gate": "adaptive_lr_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_probe_ready": str(live_probe.get("status", "unknown")) == "passed",
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "entrypoint": ENTRYPOINT,
        "probe_kind": PROBE_KIND,
        "optimizer_family": "built_in_adaptive_lr",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "live_probe": live_probe,
        "rows": rows,
        "cuda_implementation_summary": _as_dict(cuda_impl.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "training_tensor_binding_canary_ready_count": sum(
                1 for row in rows if row["training_tensor_binding_canary_ready"] is True
            ),
            "training_tensor_binding_parity_ready_count": sum(
                1 for row in rows if row["training_tensor_binding_parity_ready"] is True
            ),
            "kernel_executed_count": sum(1 for row in rows if row["kernel_executed"] is True),
            "family_case_count": int(live_probe.get("case_count", 0) or 0),
            "family_passed_case_count": int(live_probe.get("passed_case_count", 0) or 0),
            "runtime_canary_ready_count": 0,
            "runtime_canary_hit_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_training_loop_canary_missing",
                "adaptive_lr_runtime_dispatch_shadow_missing",
                "adaptive_lr_product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add adaptive-LR TrainingLoop canary with dispatch still default-off"
            if ready
            else "fix adaptive-LR live tensor binding canary blockers"
        ),
        "notes": [
            "This canary uses isolated toy CUDA tensors only.",
            "It binds param, grad, exp_avg, exp_avg_sq, and adaptive state buffers to native kernels.",
            "It does not enable TrainingLoop or product native dispatch.",
        ],
    }


def _live_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _failed("cuda_unavailable", ["adaptive_lr_training_tensor_binding_cuda_unavailable"])
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed("lulynx_native_entrypoint_missing", ["adaptive_lr_training_tensor_binding_entrypoint_missing"])
    cases = [
        _run_case(native, "prodigy_live_tensor_binding", "adaptive_lr_prodigy", workspace_root, arch),
        _run_case(native, "dadapt_live_tensor_binding", "adaptive_lr_dadapt", workspace_root, arch),
    ]
    failed = [case for case in cases if case.get("ok") is not True]
    return {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "probe_kind": PROBE_KIND,
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failed),
        "kernel_executed_case_count": sum(1 for case in cases if case.get("kernel_executed") is True),
        "training_tensor_binding_parity_passed": not failed,
        "cases": cases,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "max_param_diff": max((float(case.get("max_param_diff", 0.0) or 0.0) for case in cases), default=0.0),
        "max_state_diff": max((float(case.get("max_state_diff", 0.0) or 0.0) for case in cases), default=0.0),
        "blocked_reasons": _dedupe(reason for case in failed for reason in case.get("blocked_reasons", []) or []),
    }


def _run_case(
    native: Any,
    case_name: str,
    family: str,
    workspace_root: str | Path | None,
    arch: str | None,
) -> dict[str, Any]:
    try:
        numel = 64
        param = torch.linspace(-0.18, 0.22, steps=numel, device="cuda", dtype=torch.float32).contiguous()
        grad = (param * 0.27 + 0.017).contiguous()
        exp_avg = torch.linspace(-0.002, 0.002, steps=numel, device="cuda", dtype=torch.float32).contiguous()
        exp_avg_sq = torch.linspace(0.0003, 0.0009, steps=numel, device="cuda", dtype=torch.float32).contiguous()
        state = torch.tensor([0.72, 1.08, 0.0, 0.0], device="cuda", dtype=torch.float32).contiguous()
        ref_param = param.clone()
        ref_exp_avg = exp_avg.clone()
        ref_exp_avg_sq = exp_avg_sq.clone()
        ref_state = state.clone()
        config = {
            "numel": numel,
            "lr": 0.003,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.01,
            "global_d": 0.72,
            "dynamic_lr": 1.08,
            "max_numel": 1_048_576,
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        _reference_step(ref_param, grad, ref_exp_avg, ref_exp_avg_sq, ref_state, family, config)
        launch = dict(
            getattr(native, ENTRYPOINT)(
                KERNEL_KIND_BY_FAMILY[family],
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                state,
                json.dumps(config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or _cuda_arch(param.device),
            )
        )
        if launch.get("ok") is not True:
            return _case_failed(case_name, family, f"adaptive_lr_live_tensor_launch_failed:{launch.get('reason', 'unknown')}", launch=launch)
        param_cmp = _compare(ref_param, param)
        exp_avg_cmp = _compare(ref_exp_avg, exp_avg)
        exp_avg_sq_cmp = _compare(ref_exp_avg_sq, exp_avg_sq)
        state_cmp = _compare(ref_state, state)
        max_state_diff = max(float(exp_avg_cmp["max_diff"]), float(exp_avg_sq_cmp["max_diff"]), float(state_cmp["max_diff"]))
        finite = all(torch.isfinite(t).all().item() for t in [param, exp_avg, exp_avg_sq, state])
        ok = bool(param_cmp["ok"] and exp_avg_cmp["ok"] and exp_avg_sq_cmp["ok"] and state_cmp["ok"] and finite)
        return {
            "schema_version": 1,
            "case": case_name,
            "family": family,
            "optimizer_kind": KERNEL_KIND_BY_FAMILY[family],
            "ok": ok,
            "kernel_executed": launch.get("kernel_executed") is True,
            "reduce_kernel_executed": launch.get("reduce_kernel_executed") is True,
            "apply_kernel_executed": launch.get("apply_kernel_executed") is True,
            "native_live_tensor_binding": launch.get("native_live_tensor_binding") is True,
            "isolated_training_graph": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "native_launch": launch,
            "losses_finite": finite,
            "param_compare": param_cmp,
            "exp_avg_compare": exp_avg_cmp,
            "exp_avg_sq_compare": exp_avg_sq_cmp,
            "state_compare": state_cmp,
            "max_param_diff": param_cmp["max_diff"],
            "max_state_diff": max_state_diff,
            "blocked_reasons": [] if ok else _case_blockers(param_cmp, exp_avg_cmp, exp_avg_sq_cmp, state_cmp, finite),
        }
    except Exception as exc:
        return _case_failed(case_name, family, f"adaptive_lr_live_tensor_probe_failed:{type(exc).__name__}: {exc}")


def _reference_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    state: torch.Tensor,
    family: str,
    config: Mapping[str, Any],
) -> None:
    reduction = (param.abs() + grad.abs()).sum() if family == "adaptive_lr_dadapt" else (param * grad).abs().sum()
    state[2] += reduction
    g = grad.clone()
    p = param.clone()
    lr = float(config["lr"])
    wd = float(config["weight_decay"])
    if family == "adaptive_lr_prodigy":
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
    elif wd != 0.0:
        g.add_(p, alpha=wd)
    exp_avg.mul_(float(config["beta1"])).add_(g, alpha=1.0 - float(config["beta1"]))
    exp_avg_sq.mul_(float(config["beta2"])).add_(g.square(), alpha=1.0 - float(config["beta2"]))
    if family == "adaptive_lr_dadapt":
        scale = max(float(config["global_d"]) + float(state[2].item()) * 0.001, 1.0e-8) * max(float(config["dynamic_lr"]), 1.0e-8)
    else:
        scale = max(float(config["global_d"]), 1.0e-8) * max(float(config["dynamic_lr"]), 1.0e-8)
    param.copy_(p - lr * scale * exp_avg / (exp_avg_sq.sqrt() + float(config["eps"])))


def _row(optimizer_type: str, cuda_impl: Mapping[str, Any], live_probe: Mapping[str, Any]) -> dict[str, Any]:
    family = _family(optimizer_type)
    impl_ready = _impl_ready_for(optimizer_type, cuda_impl)
    case = _case_for_family(family, live_probe)
    case_ready = case.get("ok") is True
    ready = impl_ready and case_ready
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "optimizer_kind": KERNEL_KIND_BY_FAMILY[family],
        "state_machine_status": "training_tensor_binding_canary_ready" if ready else "training_tensor_binding_canary_blocked",
        "cuda_kernel_implementation_ready": impl_ready,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_parity_ready": case_ready,
        "kernel_executed": case.get("kernel_executed") is True,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "live_case": case,
        "next_gate": "adaptive_lr_training_loop_canary",
        "blocked_reasons": [] if ready else _row_blockers(optimizer_type, impl_ready, case),
    }


def _validations(
    cuda_impl: Mapping[str, Any],
    live_probe: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _validation("cuda_kernel_implementation_ready", cuda_impl.get("cuda_kernel_implementation_ready") is True, "adaptive_lr_cuda_kernel_implementation_missing"),
        _validation("live_tensor_binding_probe_passed", live_probe.get("status") == "passed", "adaptive_lr_live_tensor_binding_probe_failed"),
        _validation("all_rows_ready", all(row.get("training_tensor_binding_canary_ready") is True for row in rows), "adaptive_lr_training_tensor_binding_row_not_ready"),
        _validation(
            "runtime_dispatch_disabled",
            live_probe.get("training_dispatch") is False
            and live_probe.get("training_path_enabled") is False
            and live_probe.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same_shape = tuple(left.shape) == tuple(right.shape)
    same_dtype = left.dtype == right.dtype
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    return {"schema_version": 1, "ok": same_shape and same_dtype and max_diff <= FLOAT_TOLERANCE, "max_diff": max_diff, "tolerance": FLOAT_TOLERANCE}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _case_blockers(*checks: Any) -> list[str]:
    blockers: list[str] = []
    for index, check in enumerate(checks[:-1]):
        if isinstance(check, Mapping) and check.get("ok") is not True:
            blockers.append(f"adaptive_lr_training_tensor_binding_compare_{index}_failed")
    if checks and checks[-1] is not True:
        blockers.append("adaptive_lr_training_tensor_binding_non_finite_tensor")
    return blockers


def _row_blockers(optimizer_type: str, impl_ready: bool, case: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not impl_ready:
        blockers.append(f"{optimizer_type}_adaptive_lr_cuda_implementation_missing")
    if case.get("ok") is not True:
        blockers.append(f"{optimizer_type}_adaptive_lr_training_tensor_binding_canary_failed")
    return blockers


def _impl_ready_for(optimizer_type: str, report: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("optimizer_type") == optimizer_type
        and row.get("cuda_kernel_implementation_ready") is True
        for row in report.get("rows", [])
    )


def _case_for_family(family: str, live_probe: Mapping[str, Any]) -> dict[str, Any]:
    return next(
        (dict(case) for case in live_probe.get("cases", []) if isinstance(case, Mapping) and case.get("family") == family),
        {},
    )


def _family(optimizer_type: str) -> str:
    if optimizer_type in {"AutoProdigy", "prodigy", "prodigyplus.ProdigyPlusScheduleFree"}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _case_failed(case_name: str, family: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"schema_version": 1, "case": case_name, "family": family, "ok": False, "reason": reason, "kernel_executed": False, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": [reason], **extra}


def _failed(reason: str, blockers: list[str]) -> dict[str, Any]:
    return {"schema_version": 1, "status": "failed", "reason": reason, "training_tensor_binding_parity_passed": False, "kernel_executed_case_count": 0, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": blockers}


def _skipped(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "status": "skipped", "reason": reason, "training_tensor_binding_parity_passed": False, "kernel_executed_case_count": 0, "training_dispatch": False, "training_path_enabled": False, "native_dispatch_allowed": False, "blocked_reasons": []}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "PROBE_KIND", "build_adaptive_lr_training_tensor_binding_canary_scorecard"]
