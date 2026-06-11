"""Report-only training tensor binding canary for PagedAdamW8bit."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_paged_adamw8bit_bnb_exact_parity_scorecard import (
    FLOAT_TOLERANCE,
    _compare_states,
    _compare_tensor,
    _state_signature,
    _step_int,
)
from core.turbocore_paged_adamw8bit_checkpoint_runtime_adapter_scorecard import (
    export_paged_adamw8bit_runtime_checkpoint_envelope,
    import_paged_adamw8bit_runtime_checkpoint_state_dict,
    build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
)


ENTRYPOINT = "probe_paged_adamw8bit_training_tensor_binding_canary_py"
PROBE_KIND = "paged_adamw8bit_training_tensor_binding_canary_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_paged_adamw8bit_training_tensor_binding_canary_scorecard(
    *,
    checkpoint_runtime_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run an isolated training-tensor binding canary without dispatch."""

    checkpoint_runtime = dict(
        checkpoint_runtime_report
        or build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard(run_live_probe=False, numel=numel)
    )
    live_probe = (
        _live_probe(numel=numel, workspace_root=workspace_root, arch=arch)
        if run_live_probe
        else _skipped_live_probe("live_probe_disabled")
    )
    validations = _validations(checkpoint_runtime, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_training_tensor_binding_canary_scorecard_v0",
        "gate": "paged_adamw8bit_training_tensor_binding_canary",
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
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "checkpoint_runtime_summary": dict(checkpoint_runtime.get("summary") or {}),
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
            "max_state_float_diff": live_probe.get("max_state_float_diff"),
            "state_uint8_mismatch_count": live_probe.get("state_uint8_mismatch_count"),
            "loss_diff": live_probe.get("loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
                "paged_adamw8bit_real_training_matrix_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add PagedAdamW8bit runtime dispatch dry-run selector and real-training matrix gate"
            if ready
            else "fix PagedAdamW8bit training tensor binding canary blockers"
        ),
        "notes": [
            "This canary uses a synthetic isolated training graph and real autograd gradients.",
            "It binds a Parameter and its .grad to the native kernel, but never touches user training runs.",
            "Default training dispatch remains disabled until selector dry-run and real training matrix gates pass.",
        ],
    }


def _live_probe(
    *,
    numel: int,
    workspace_root: str | Path | None,
    arch: str | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    if importlib.util.find_spec("bitsandbytes") is None:
        return _skipped_live_probe("bitsandbytes_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed_live_probe(
            "lulynx_native_entrypoint_missing",
            ["paged_adamw8bit_training_tensor_binding_entrypoint_missing"],
        )
    try:
        value = torch.linspace(-1.0, 1.0, steps=max(int(numel), 1), device="cuda")
        prime = _make_trainer(value)
        optimizer = prime._create_optimizer()
        optimizer_name = type(optimizer).__name__
        if optimizer_name == "AdamW":
            return _failed_live_probe(
                "resolved_to_fallback_adamw",
                ["paged_adamw8bit_resolved_to_fallback_adamw"],
                optimizer_class=optimizer_name,
            )
        prime_loss = _backward_loss(prime.lora_injector.param, scale=0.35)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        live_state = _first_live_state(optimizer)
        envelope = export_paged_adamw8bit_runtime_checkpoint_envelope(
            optimizer_state_dict=optimizer.state_dict(),
            live_state=live_state,
            run_id="p8l_probe",
        )
        imported_state_dict, import_validation = import_paged_adamw8bit_runtime_checkpoint_state_dict(envelope)

        saved_param = prime.lora_injector.param.detach().clone()
        reference = _make_trainer(saved_param)
        reference_optimizer = reference._create_optimizer()
        reference_optimizer.load_state_dict(copy.deepcopy(imported_state_dict))
        ref_loss = _backward_loss(reference.lora_injector.param, scale=0.7)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        candidate = _make_trainer(saved_param)
        candidate_optimizer = candidate._create_optimizer()
        candidate_optimizer.load_state_dict(copy.deepcopy(imported_state_dict))
        candidate_param = candidate.lora_injector.param
        candidate_loss = _backward_loss(candidate_param, scale=0.7)
        candidate_state = _first_live_state(candidate_optimizer)
        group = candidate_optimizer.param_groups[0]
        beta1, beta2 = group["betas"][:2]
        step_index = _step_int(candidate_state.get("step"))
        launch_config = {
            "lr": float(group["lr"]),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "step_index": step_index,
            "max_numel": max(int(numel), 1_048_576),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                candidate_param.grad.detach().contiguous(),
                candidate_state["state1"],
                candidate_state["state2"],
                candidate_state["qmap1"],
                candidate_state["qmap2"],
                candidate_state["absmax1"],
                candidate_state["absmax2"],
                json.dumps(launch_config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _failed_live_probe(
                str(launch.get("reason") or "training_tensor_binding_launch_failed"),
                [f"paged_adamw8bit_training_tensor_binding_launch_failed:{launch.get('reason', 'unknown')}"],
                optimizer_class=optimizer_name,
                native_launch=launch,
            )
        if isinstance(candidate_state, dict):
            candidate_state["step"] = step_index + 1
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_state = _first_live_state(reference_optimizer)
        param_compare = _compare_tensor(
            "param",
            reference.lora_injector.param.detach(),
            candidate_param.detach(),
            tolerance=FLOAT_TOLERANCE,
        )
        state_compare = _compare_states(reference_state, candidate_state, REQUIRED_LIVE_KEYS)
        step_match = _step_int(reference_state.get("step")) == _step_int(candidate_state.get("step"))
        loss_diff = abs(float(ref_loss) - float(candidate_loss))
        finite = all(torch.isfinite(torch.tensor(v)).item() for v in [prime_loss, ref_loss, candidate_loss])
        ok = (
            bool(import_validation.get("ok", False))
            and bool(param_compare["ok"])
            and bool(state_compare["ok"])
            and step_match
            and finite
        )
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(candidate_param.device),
            "probe_kind": PROBE_KIND,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": ok,
            "e2e_no_regression_passed": ok,
            "native_launch": launch,
            "import_validation": import_validation,
            "isolated_training_graph": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "prime_loss": float(prime_loss),
            "reference_loss": float(ref_loss),
            "candidate_loss": float(candidate_loss),
            "loss_diff": loss_diff,
            "losses_finite": finite,
            "reference_step": _step_int(reference_state.get("step")),
            "candidate_step": _step_int(candidate_state.get("step")),
            "step_match": step_match,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_state_float_diff": state_compare.get("max_float_diff"),
            "state_uint8_mismatch_count": state_compare.get("uint8_mismatch_count"),
            "reference_state_signature": _state_signature(reference_state, REQUIRED_LIVE_KEYS),
            "candidate_state_signature": _state_signature(candidate_state, REQUIRED_LIVE_KEYS),
            "blocked_reasons": [] if ok else _live_blockers(import_validation, param_compare, state_compare, step_match, finite),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"paged_adamw8bit_training_tensor_binding_probe_failed:{type(exc).__name__}"],
        )


def _backward_loss(param: torch.nn.Parameter, *, scale: float) -> float:
    if param.grad is not None:
        param.grad = None
    values = param.float()
    loss = ((values * float(scale)).pow(2).mean() + values.mean() * 0.013)
    loss.backward()
    return float(loss.detach().cpu())


def _validations(
    checkpoint_runtime: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(live_probe.get("training_tensor_binding_parity_passed", False))
    return [
        _validation(
            "p8k_checkpoint_runtime_adapter_ready",
            bool(checkpoint_runtime.get("checkpoint_adapter_runtime_ready", False)),
            "paged_adamw8bit_checkpoint_runtime_adapter_missing",
        ),
        _validation(
            "training_tensor_binding_probe_or_skip",
            live_ready,
            "paged_adamw8bit_training_tensor_binding_canary_failed",
        ),
        _validation(
            "e2e_no_regression_or_skip",
            live_status == "skipped" or bool(live_probe.get("e2e_no_regression_passed", False)),
            "paged_adamw8bit_training_tensor_binding_e2e_regression",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "paged_adamw8bit_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _live_blockers(
    import_validation: Mapping[str, Any],
    param_compare: Mapping[str, Any],
    state_compare: Mapping[str, Any],
    step_match: bool,
    finite: bool,
) -> list[str]:
    blockers = []
    if not bool(import_validation.get("ok", False)):
        blockers.append("paged_adamw8bit_training_tensor_binding_checkpoint_import_invalid")
    if not bool(param_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_training_tensor_binding_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_training_tensor_binding_state_parity_failed")
    if not step_match:
        blockers.append("paged_adamw8bit_training_tensor_binding_step_mismatch")
    if not finite:
        blockers.append("paged_adamw8bit_training_tensor_binding_non_finite_loss")
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


__all__ = [
    "ENTRYPOINT",
    "PROBE_KIND",
    "build_paged_adamw8bit_training_tensor_binding_canary_scorecard",
]
