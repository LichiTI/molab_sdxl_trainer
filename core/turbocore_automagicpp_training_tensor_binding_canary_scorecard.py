"""Report-only training tensor binding canary for Automagic++."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.lulynx_trainer.automagic_plus_plus_optimizer import AutomagicPlusPlus
from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_automagicpp_native_scratch_kernel_scorecard import (
    build_automagicpp_native_scratch_kernel_scorecard,
)


ENTRYPOINT = "probe_automagicpp_training_tensor_binding_canary_py"
PROBE_KIND = "automagicpp_training_tensor_binding_canary_v0"
FLOAT_TOLERANCE = 5e-6
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_automagicpp_training_tensor_binding_canary_scorecard(
    *,
    scratch_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    scratch = dict(scratch_report or build_automagicpp_native_scratch_kernel_scorecard(workspace_root=workspace_root))
    live_probe = (
        _live_probe(numel=numel, workspace_root=workspace_root, arch=arch)
        if run_live_probe
        else _skipped_live_probe("live_probe_disabled")
    )
    validations = _validations(scratch, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_training_tensor_binding_canary_scorecard_v0",
        "gate": "automagicpp_training_tensor_binding_canary",
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
        "optimizer_kind": "automagicpp",
        "optimizer_family": "factored_custom",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "scratch_summary": dict(scratch.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "kernel_executed": bool(live_probe.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": bool(
                live_probe.get("training_tensor_binding_parity_passed", False)
            ),
            "e2e_no_regression_passed": bool(live_probe.get("e2e_no_regression_passed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_local_lr_diff": live_probe.get("max_local_lr_diff"),
            "max_full_var_diff": live_probe.get("max_full_var_diff"),
            "avg_lr_diff": live_probe.get("avg_lr_diff"),
            "loss_diff": live_probe.get("loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "automagicpp_runtime_dispatch_shadow_missing",
                "automagicpp_training_loop_canary_missing",
                "automagicpp_rollback_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Automagic++ runtime dispatch shadow before TrainingLoop canary"
            if ready
            else "fix Automagic++ training tensor binding canary blockers"
        ),
        "notes": [
            "This canary uses a synthetic isolated training graph and real autograd gradients.",
            "It binds Parameter, grad, local_lr, prev_sign, full_var, has_prev_sign, and avg_lr to the native kernel.",
            "It never touches user training runs and keeps native optimizer dispatch disabled.",
        ],
    }


def _live_probe(*, numel: int, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed_live_probe("lulynx_native_entrypoint_missing", ["automagicpp_training_tensor_binding_entrypoint_missing"])
    try:
        value = torch.linspace(-0.25, 0.35, steps=max(int(numel), 1), device="cuda", dtype=torch.float32)
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
        if torch.is_tensor(state.get("prev_sign")) and state["prev_sign"].dtype != torch.float32:
            state["prev_sign"] = state["prev_sign"].to(device=candidate_param.device, dtype=torch.float32)
        group = candidate_optimizer.param_groups[0]
        launch_config = {
            "beta2": float(group["beta2"]),
            "eps1": float(_eps_values(group["eps"])[0]),
            "clip_threshold": float(group["clip_threshold"]),
            "min_lr": float(candidate_optimizer.min_lr),
            "max_lr": float(candidate_optimizer.max_lr),
            "lr_up": float(candidate_optimizer.lr_up),
            "lr_down": float(candidate_optimizer.lr_down),
            "weight_decay": float(group["weight_decay"]),
            "max_numel": max(int(numel), 1_048_576),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        has_prev = torch.tensor([1 if bool(state.get("has_prev_sign", False)) else 0], device="cuda", dtype=torch.int32)
        avg_lr = state["avg_lr"].detach().reshape(1).contiguous().to(device="cuda", dtype=torch.float32)
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                candidate_param.grad.detach().contiguous(),
                state["local_lr"],
                state["prev_sign"],
                state["full_var"],
                has_prev,
                avg_lr,
                json.dumps(launch_config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _failed_live_probe(
                str(launch.get("reason") or "training_tensor_binding_launch_failed"),
                [f"automagicpp_training_tensor_binding_launch_failed:{launch.get('reason', 'unknown')}"],
                native_launch=launch,
            )
        state["has_prev_sign"] = bool(int(has_prev.detach().cpu().item()))
        state["avg_lr"] = avg_lr.detach().mean()
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_state = reference_optimizer.state[reference_param]
        param_compare = _compare_tensor(reference_param.detach(), candidate_param.detach())
        state_compare = _compare_state(reference_state, state)
        loss_diff = abs(float(reference_loss) - float(candidate_loss))
        finite = all(torch.isfinite(torch.tensor(v)).item() for v in [prime_loss, reference_loss, candidate_loss])
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and finite
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "device": str(candidate_param.device),
            "probe_kind": PROBE_KIND,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": ok,
            "e2e_no_regression_passed": ok,
            "native_launch": launch,
            "isolated_training_graph": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "prime_loss": float(prime_loss),
            "reference_loss": float(reference_loss),
            "candidate_loss": float(candidate_loss),
            "loss_diff": loss_diff,
            "losses_finite": finite,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_local_lr_diff": state_compare.get("max_local_lr_diff"),
            "max_full_var_diff": state_compare.get("max_full_var_diff"),
            "avg_lr_diff": state_compare.get("avg_lr_diff"),
            "prev_sign_mismatch_count": state_compare.get("prev_sign_mismatch_count"),
            "has_prev_sign_match": state_compare.get("has_prev_sign_match"),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, finite),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"automagicpp_training_tensor_binding_probe_failed:{type(exc).__name__}"],
        )


def _make_optimizer(param: torch.nn.Parameter) -> AutomagicPlusPlus:
    return AutomagicPlusPlus(
        [param],
        lr=1e-4,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_up=1.01,
        lr_down=0.95,
        beta2=0.97,
        beta1=0.0,
        weight_decay=0.01,
        max_update_rms_ratio=None,
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
            "p20_automagicpp_scratch_kernel_ready",
            bool(scratch.get("automagicpp_native_kernel_parity", False)),
            "automagicpp_native_scratch_kernel_missing",
        ),
        _validation(
            "training_tensor_binding_probe_or_skip",
            live_ready,
            "automagicpp_training_tensor_binding_canary_failed",
        ),
        _validation(
            "e2e_no_regression_or_skip",
            live_status == "skipped" or bool(live_probe.get("e2e_no_regression_passed", False)),
            "automagicpp_training_tensor_binding_e2e_regression",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "automagicpp_training_tensor_binding_enabled_dispatch",
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


def _compare_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    local_lr_diff = _max_abs(left["local_lr"], right["local_lr"])
    full_var_diff = _max_abs(left["full_var"], right["full_var"])
    avg_lr_diff = abs(float(left["avg_lr"].detach().cpu()) - float(right["avg_lr"].detach().cpu()))
    prev_mismatch = int((left["prev_sign"] != right["prev_sign"]).sum().item())
    has_prev_match = bool(left.get("has_prev_sign", False)) == bool(right.get("has_prev_sign", False))
    ok = (
        local_lr_diff <= FLOAT_TOLERANCE
        and full_var_diff <= FLOAT_TOLERANCE
        and avg_lr_diff <= FLOAT_TOLERANCE
        and prev_mismatch == 0
        and has_prev_match
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "max_local_lr_diff": local_lr_diff,
        "max_full_var_diff": full_var_diff,
        "avg_lr_diff": avg_lr_diff,
        "prev_sign_mismatch_count": prev_mismatch,
        "has_prev_sign_match": has_prev_match,
        "tolerance": FLOAT_TOLERANCE,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _eps_values(value: tuple[float, float] | float) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) >= 2:
            return float(value[0]), float(value[1])
        if len(value) == 1:
            return float(value[0]), 1e-3
    return float(value), 1e-3


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _live_blockers(param_compare: Mapping[str, Any], state_compare: Mapping[str, Any], finite: bool) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("automagicpp_training_tensor_binding_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("automagicpp_training_tensor_binding_state_parity_failed")
    if not finite:
        blockers.append("automagicpp_training_tensor_binding_non_finite_loss")
    return blockers


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed": False,
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


__all__ = ["ENTRYPOINT", "PROBE_KIND", "build_automagicpp_training_tensor_binding_canary_scorecard"]
