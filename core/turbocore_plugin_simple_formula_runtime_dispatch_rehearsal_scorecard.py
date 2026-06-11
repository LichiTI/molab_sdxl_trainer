"""Batch native dispatch rehearsal for selected plugin simple-formula optimizers.

This scorecard executes the guarded runtime path for selected plugin
simple-formula optimizers, but keeps product/request/UI defaults closed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_simple_formula_training_loop_canary import SPECS, SimpleFormulaPluginSpec
from core.turbocore_simple_optimizer_training_executor import build_simple_optimizer_training_executor


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard.json"


def build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    device: torch.device | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Run selected simple-formula native executors through dispatch runtime."""

    selector = _as_dict(selector_report) if selector_report is not None else build_plugin_optimizer_selector_scorecard()
    selected = _selected_specs(selector)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(workspace_root or REPO_ROOT)
    if target_device.type != "cuda":
        report = _blocked(
            "cuda_required_for_plugin_simple_formula_runtime_dispatch_rehearsal",
            selector=selector,
            selected=selected,
            workspace_root=root,
            device=target_device,
        )
    elif not selected:
        report = _blocked(
            "plugin_simple_formula_selected_optimizer_specs_missing",
            selector=selector,
            selected=selected,
            workspace_root=root,
            device=target_device,
        )
    else:
        started = time.perf_counter()
        cases = [_run_case(spec, root, target_device, index) for index, spec in enumerate(selected, start=1)]
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        ready = len(cases) == len(selected) and all(case.get("runtime_dispatch_rehearsal_ready") is True for case in cases)
        report = {
            "schema_version": 1,
            "scorecard": "turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard_v0",
            "gate": "plugin_simple_formula_runtime_dispatch_rehearsal",
            "roadmap": ROADMAP,
            "ok": ready,
            "promotion_ready": False,
            "runtime_dispatch_rehearsal_ready": ready,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_dispatch_ready": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "internal_rehearsal_executed": True,
            "rehearsal_training_dispatch_requested": True,
            "selected_optimizer_family": "simple_formula",
            "workspace_root": str(root.resolve()),
            "device": str(target_device),
            "selector_scorecard": _compact_selector(selector),
            "cases": cases,
            "summary": {
                "selected_optimizer_count": len(selected),
                "case_count": len(cases),
                "runtime_dispatch_rehearsal_ready_count": sum(
                    1 for case in cases if case.get("runtime_dispatch_rehearsal_ready") is True
                ),
                "training_executor_called_count": sum(
                    1 for case in cases if case.get("training_executor_called") is True
                ),
                "native_step_count": sum(1 for case in cases if case.get("native_step_executed") is True),
                "native_kernel_launch_count": sum(1 for case in cases if case.get("native_kernel_launched") is True),
                "skip_pytorch_count": sum(
                    1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False
                ),
                "unique_native_kind_count": len({str(case.get("executor_optimizer_kind") or "") for case in cases}),
                "runtime_dispatch_ready_count": 0,
                "native_dispatch_allowed_count": 0,
                "training_path_enabled_count": 0,
                "product_native_ready_count": 0,
                "elapsed_ms": _elapsed_ms(started),
            },
            "promotion_blockers": _dedupe(
                blockers
                + [
                    "plugin_simple_formula_owner_release_review_missing",
                    "plugin_simple_formula_product_training_route_not_bound",
                ]
            ),
            "blocked_reasons": blockers,
            "recommended_next_step": (
                "bind selected plugin simple-formula rehearsal evidence into guarded product-training canary"
                if ready
                else "fix selected plugin simple-formula runtime dispatch rehearsal blockers"
            ),
            "notes": [
                "This is an explicit CUDA rehearsal, not product dispatch approval.",
                "Request, schema, UI, runtime dispatch, and training defaults remain closed.",
            ],
        }
    if write_artifact:
        _write_artifact(report)
    return report


def _run_case(
    spec: SimpleFormulaPluginSpec,
    workspace_root: Path,
    device: torch.device,
    case_index: int,
) -> dict[str, Any]:
    before: torch.Tensor | None = None
    executor: Any | None = None
    try:
        param = torch.nn.Parameter(_param_tensor(spec, device, case_index))
        before = param.detach().clone()
        loss = ((param.float() * spec.loss_scale) ** 2).mean() + param.float().mean() * spec.loss_bias
        loss.backward()
        executor = build_simple_optimizer_training_executor(
            params=[param],
            config=_executor_config(spec),
            workspace_root=workspace_root,
        )
        runtime = TurboCoreNativeUpdateDispatchRuntime()
        report = runtime.prepare_step(
            step=case_index,
            arming_report={
                "previous_request_requested": True,
                "armed_for_native_dispatch": True,
                "execute_native_step": True,
            },
            kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
            runtime_context={
                "native_update_executor_present": True,
                "native_update_runtime_execution_guard_enabled": True,
                "native_update_training_mutation_guard_enabled": True,
                "native_update_training_dispatch_enabled": True,
                "native_update_runtime_dispatch_available": True,
                "training_path_enabled": True,
            },
            native_executor=executor,
        )
        training_executor = _as_dict(report.get("training_executor"))
        executor_result = _as_dict(training_executor.get("result"))
        param_mutated = before is not None and not torch.equal(before, param.detach())
        ready = (
            bool(report.get("native_step_executed", False))
            and bool(report.get("native_kernel_launched", False))
            and bool(training_executor.get("called", False))
            and bool(training_executor.get("ok", False))
            and bool(executor_result.get("ok", False))
            and str(executor_result.get("optimizer_kind") or "") == spec.native_kind
            and report.get("should_call_pytorch_optimizer_step") is False
            and param_mutated
            and not report.get("blocked_reasons")
        )
        return {
            "schema_version": 1,
            "ok": ready,
            "selected_optimizer_name": spec.kind,
            "optimizer_class": spec.class_name,
            "selected_optimizer_family": "simple_formula",
            "native_route": spec.native_route,
            "expected_native_kind": spec.native_kind,
            "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
            "runtime_dispatch_rehearsal_ready": ready,
            "training_executor_called": bool(training_executor.get("called", False)),
            "training_executor_ok": bool(training_executor.get("ok", False)),
            "native_step_executed": bool(report.get("native_step_executed", False)),
            "native_kernel_launched": bool(report.get("native_kernel_launched", False)),
            "training_parameters_mutated": param_mutated,
            "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
            "fallback_to_pytorch_required": bool(report.get("fallback_to_pytorch_required", True)),
            "internal_training_path_enabled": bool(report.get("training_path_enabled", False)),
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": []
            if ready
            else _dedupe(
                list(report.get("blocked_reasons", []) or [])
                + list(executor_result.get("blocked_reasons", []) or [])
                + [f"plugin_{spec.kind}_runtime_dispatch_rehearsal_missing"]
            ),
        }
    except Exception as exc:  # pragma: no cover - CUDA/native dependent
        return {
            "schema_version": 1,
            "ok": False,
            "selected_optimizer_name": spec.kind,
            "optimizer_class": spec.class_name,
            "selected_optimizer_family": "simple_formula",
            "expected_native_kind": spec.native_kind,
            "runtime_dispatch_rehearsal_ready": False,
            "training_executor_called": False,
            "native_step_executed": False,
            "native_kernel_launched": False,
            "training_parameters_mutated": False,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "native_dispatch_allowed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"plugin_{spec.kind}_runtime_dispatch_rehearsal_failed:{type(exc).__name__}"],
        }
    finally:
        if executor is not None:
            executor.close()


def _selected_specs(selector: Mapping[str, Any]) -> list[SimpleFormulaPluginSpec]:
    selected = {
        str(row.get("optimizer_name") or "").strip().lower()
        for row in selector.get("rows", [])
        if isinstance(row, Mapping) and row.get("native_route_family") == "simple_formula"
    }
    return [spec for name, spec in SPECS.items() if name in selected]


def _param_tensor(spec: SimpleFormulaPluginSpec, device: torch.device, case_index: int) -> torch.Tensor:
    numel = 1
    for dim in spec.param_shape:
        numel *= int(dim)
    start = -0.08 + case_index * 0.0005
    stop = 0.18 + case_index * 0.0005
    return torch.linspace(start, stop, steps=numel, device=device, dtype=torch.float32).view(spec.param_shape)


def _executor_config(spec: SimpleFormulaPluginSpec) -> dict[str, Any]:
    return {
        "optimizer_kind": spec.native_kind,
        "lr": 1e-3,
        "betas": [0.9, 0.99],
        "momentum": 0.9,
        "nu": 0.7,
        "kappa": 1000.0,
        "xi": 10.0,
        "constant": 0.7,
        "alpha": 0.99,
        "beta": 0.965 if spec.kind == "tiger" else 0.9,
        "eps": 1e-8,
        "trust_coefficient": 1e-3,
        "dampening": 0.0,
        "weight_decay": float(spec.weight_decay),
        "block_size": 128,
    }


def _blocked(
    reason: str,
    *,
    selector: Mapping[str, Any],
    selected: list[SimpleFormulaPluginSpec],
    workspace_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard_v0",
        "gate": "plugin_simple_formula_runtime_dispatch_rehearsal",
        "roadmap": ROADMAP,
        "ok": False,
        "promotion_ready": False,
        "runtime_dispatch_rehearsal_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "internal_rehearsal_executed": False,
        "selected_optimizer_family": "simple_formula",
        "workspace_root": str(workspace_root.resolve()),
        "device": str(device),
        "selector_scorecard": _compact_selector(selector),
        "cases": [],
        "summary": {
            "selected_optimizer_count": len(selected),
            "case_count": 0,
            "runtime_dispatch_rehearsal_ready_count": 0,
            "training_executor_called_count": 0,
            "native_step_count": 0,
            "native_kernel_launch_count": 0,
            "skip_pytorch_count": 0,
            "unique_native_kind_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
            "elapsed_ms": 0.0,
        },
        "promotion_blockers": [reason, "plugin_simple_formula_product_training_route_not_bound"],
        "blocked_reasons": [reason],
        "recommended_next_step": "run selected plugin simple-formula runtime dispatch rehearsal on CUDA",
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "simple_formula_count": int(counts.get("simple_formula", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard"]
