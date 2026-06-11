"""Report-only native live-buffer launch parity for PagedAdamW8bit."""

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
    build_paged_adamw8bit_bnb_exact_parity_scorecard,
    _compare_states,
    _compare_tensor,
    _state_signature,
    _step_int,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
    _step,
)


ENTRYPOINT = "probe_paged_adamw8bit_live_buffer_launch_py"
PROBE_KIND = "paged_adamw8bit_native_live_buffer_launch_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_paged_adamw8bit_native_live_buffer_launch_scorecard(
    *,
    formula_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Launch the native kernel on cloned bnb live buffers only."""

    formula = dict(
        formula_report
        or build_paged_adamw8bit_bnb_exact_parity_scorecard(run_live_probe=False, numel=numel)
    )
    live_probe = (
        _live_launch_probe(numel=numel, workspace_root=workspace_root, arch=arch)
        if run_live_probe
        else _skipped_live_probe("live_probe_disabled")
    )
    validations = _validations(formula, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_native_live_buffer_launch_scorecard_v0",
        "gate": "paged_adamw8bit_native_live_buffer_launch_parity",
        "ok": ready,
        "promotion_ready": False,
        "native_live_launch_probe_ready": str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
        "native_live_launch_parity_ready": bool(live_probe.get("native_live_launch_parity_passed", False)),
        "native_live_tensor_launch_ready": bool(live_probe.get("native_live_launch_parity_passed", False)),
        "training_tensor_binding_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "entrypoint": ENTRYPOINT,
        "probe_kind": PROBE_KIND,
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "formula_summary": dict(formula.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "native_live_launch_parity_passed": bool(
                live_probe.get("native_live_launch_parity_passed", False)
            ),
            "kernel_executed": bool(live_probe.get("kernel_executed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_state_float_diff": live_probe.get("max_state_float_diff"),
            "state_uint8_mismatch_count": live_probe.get("state_uint8_mismatch_count"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_training_tensor_binding_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "runtime_canary_e2e_no_regression_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only PagedAdamW8bit runtime canary shadow using cloned live-buffer launch evidence"
            if ready
            else "fix PagedAdamW8bit native live-buffer launch parity blockers"
        ),
        "notes": [
            "This gate launches the native kernel on cloned PagedAdamW8bit live buffers.",
            "It is still not training dispatch: the original training optimizer tensors are never passed in.",
            "Passing this gate means the next risk boundary is runtime tensor binding and checkpoint adapter runtime.",
        ],
    }


def _live_launch_probe(
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
            ["paged_adamw8bit_native_live_launch_entrypoint_missing"],
        )
    try:
        value = torch.linspace(-1.0, 1.0, steps=max(int(numel), 1), device="cuda")
        trainer = _make_trainer(value)
        optimizer = trainer._create_optimizer()
        optimizer_name = type(optimizer).__name__
        if optimizer_name == "AdamW":
            return _failed_live_probe(
                "resolved_to_fallback_adamw",
                ["paged_adamw8bit_resolved_to_fallback_adamw"],
                optimizer_class=optimizer_name,
            )

        param = trainer.lora_injector.param
        grad1 = torch.linspace(-0.1, 0.1, steps=param.numel(), device=param.device)
        grad2 = torch.linspace(0.05, -0.05, steps=param.numel(), device=param.device)
        _step(param, optimizer, grad1)
        checkpoint = copy.deepcopy(optimizer.state_dict())
        saved_param = param.detach().clone()

        reference = _make_trainer(saved_param)
        reference_optimizer = reference._create_optimizer()
        reference_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        _step(reference.lora_injector.param, reference_optimizer, grad2)

        candidate = _make_trainer(saved_param)
        candidate_optimizer = candidate._create_optimizer()
        candidate_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        candidate_param = candidate.lora_injector.param
        candidate_state = _first_live_state(candidate_optimizer)
        _ensure_contiguous_state(candidate_state)
        group = candidate_optimizer.param_groups[0]
        beta1, beta2 = group["betas"][:2]
        step_index = _step_int(candidate_state.get("step"))
        config = {
            "lr": float(group["lr"]),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "step_index": step_index,
            "max_numel": max(int(numel), 1_048_576),
            "cloned_live_buffers": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                grad2.detach().clone().contiguous(),
                candidate_state["state1"],
                candidate_state["state2"],
                candidate_state["qmap1"],
                candidate_state["qmap2"],
                candidate_state["absmax1"],
                candidate_state["absmax2"],
                json.dumps(config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _failed_live_probe(
                str(launch.get("reason") or "native_live_launch_failed"),
                [f"paged_adamw8bit_native_live_launch_failed:{launch.get('reason', 'unknown')}"],
                optimizer_class=optimizer_name,
                native_launch=launch,
            )
        if isinstance(candidate_state, dict):
            candidate_state["step"] = step_index + 1

        reference_state = _first_live_state(reference_optimizer)
        param_compare = _compare_tensor(
            "param",
            reference.lora_injector.param.detach(),
            candidate_param.detach(),
            tolerance=FLOAT_TOLERANCE,
        )
        state_compare = _compare_states(reference_state, candidate_state, REQUIRED_LIVE_KEYS)
        step_match = _step_int(reference_state.get("step")) == _step_int(candidate_state.get("step"))
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and step_match
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "probe_kind": PROBE_KIND,
            "native_live_launch_parity_passed": ok,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "native_launch": launch,
            "cloned_live_buffers": True,
            "training_tensor_binding": False,
            "training_dispatch": False,
            "training_path_enabled": False,
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
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, step_match),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"paged_adamw8bit_native_live_launch_probe_failed:{type(exc).__name__}"],
        )


def _ensure_contiguous_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    for key in REQUIRED_LIVE_KEYS:
        value = state.get(key)
        if torch.is_tensor(value) and not value.is_contiguous():
            state[key] = value.contiguous()


def _validations(formula: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(live_probe.get("native_live_launch_parity_passed", False))
    return [
        _validation(
            "p8h_formula_parity_ready",
            bool(formula.get("bnb_exact_native_formula_parity_ready", False)),
            "paged_adamw8bit_bnb_exact_formula_parity_missing",
        ),
        _validation(
            "native_live_launch_probe_or_skip",
            live_ready,
            "paged_adamw8bit_native_live_launch_parity_failed",
        ),
        _validation(
            "cloned_buffers_only",
            str(live_probe.get("status", "unknown")) == "skipped"
            or bool(live_probe.get("cloned_live_buffers", False)),
            "paged_adamw8bit_native_live_launch_not_limited_to_clones",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_tensor_binding", True))
            and not bool(live_probe.get("training_path_enabled", True)),
            "paged_adamw8bit_native_live_launch_enabled_training_path",
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
    param_compare: Mapping[str, Any],
    state_compare: Mapping[str, Any],
    step_match: bool,
) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_native_live_launch_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_native_live_launch_state_parity_failed")
    if not step_match:
        blockers.append("paged_adamw8bit_native_live_launch_step_mismatch")
    return blockers


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "native_live_launch_parity_passed": False,
        "kernel_executed": False,
        "cloned_live_buffers": True,
        "training_tensor_binding": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "native_live_launch_parity_passed": False,
        "kernel_executed": False,
        "cloned_live_buffers": True,
        "training_tensor_binding": False,
        "training_dispatch": False,
        "training_path_enabled": False,
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
    "build_paged_adamw8bit_native_live_buffer_launch_scorecard",
]
