"""Executable report-only matrix for custom-formula plugin optimizers.

This matrix proves that each selected custom-formula plugin route can be
instantiated, stepped, serialized, restored, and replayed against the existing
plugin implementation.  It is not an independent formula reference and never
claims native kernel readiness.
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_custom_formula_source_inventory import custom_formula_source_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT / "plugin" / "pytorch_optimizer-main"
CUSTOM_FORMULA_ROUTE_FAMILY = "custom_formula"


def build_custom_formula_execution_matrix(
    rows: list[Mapping[str, Any]],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run tiny plugin-owned step/resume checks for selected custom formulas."""

    case_rows = [_row(row, device=device) for row in rows]
    step_ready = [row for row in case_rows if row["formula_step_execution_ready"] is True]
    resume_ready = [row for row in case_rows if row["resume_next_step_replay_ready"] is True]
    failed = [row for row in case_rows if row["ok"] is not True]
    unsafe = _unsafe_claims(case_rows)
    ready = bool(case_rows) and not failed and not unsafe
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_custom_formula_execution_matrix_v0",
        "gate": "plugin_custom_formula_execution_matrix",
        "ok": ready,
        "execution_matrix_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "selected_optimizer_family": CUSTOM_FORMULA_ROUTE_FAMILY,
        "device": device,
        "rows": case_rows,
        "summary": {
            "selected_optimizer_count": len(case_rows),
            "formula_step_execution_ready_count": len(step_ready),
            "resume_next_step_replay_ready_count": len(resume_ready),
            "execution_failed_count": len(failed),
            "unsafe_claim_count": len(unsafe),
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "blocked_reasons": _dedupe(
            unsafe + [reason for row in failed for reason in row.get("blocked_reasons", [])]
        ),
        "promotion_blockers": [
            "independent_formula_reference_missing",
            "native_kernel_implementation_missing",
            "owner_release_hold_missing",
        ],
        "notes": [
            "This matrix executes the existing plugin implementation as the reference authority.",
            "It proves tiny step and state_dict resume replay, not independent native formula parity.",
            "Native dispatch, product dispatch, request/UI/schema exposure, and kernel readiness stay false.",
        ],
    }


def _row(selector_row: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    name = str(selector_row.get("selected_optimizer_name") or selector_row.get("optimizer_name") or "").strip().lower()
    base = {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "native_route_family": CUSTOM_FORMULA_ROUTE_FAMILY,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
    }
    try:
        result = _execute(name, device=device)
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "formula_step_execution_ready": False,
            "resume_next_step_replay_ready": False,
            "blocked_reasons": [f"custom_formula_execution_failed:{name}:{type(exc).__name__}"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        **result,
        "ok": result["formula_step_execution_ready"] is True
        and result["resume_next_step_replay_ready"] is True,
        "blocked_reasons": [] if result["resume_next_step_replay_ready"] is True else ["resume_next_step_replay_failed"],
    }


def _execute(name: str, *, device: str) -> dict[str, Any]:
    torch = _torch()
    optimizer_cls, source = _optimizer_class(name)
    initial = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32, device=device)
    grad_a = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32, device=device)
    grad_b = torch.tensor([0.02, 0.01, -0.04], dtype=torch.float32, device=device)

    param_a = torch.nn.Parameter(initial.clone())
    param_b = torch.nn.Parameter(initial.clone())
    opt_a = optimizer_cls([param_a])
    opt_b = optimizer_cls([param_b])
    param_a.grad = grad_a.clone()
    param_b.grad = grad_a.clone()
    opt_a.step()
    opt_b.step()
    formula_step_ready = bool(torch.allclose(param_a, param_b, atol=1e-6, rtol=1e-5))
    finite_after_step = bool(torch.isfinite(param_a.detach()).all().item())

    state = copy.deepcopy(opt_a.state_dict())
    resume_param = torch.nn.Parameter(param_a.detach().clone())
    resume_opt = optimizer_cls([resume_param])
    resume_opt.load_state_dict(state)
    param_a.grad = grad_b.clone()
    resume_param.grad = grad_b.clone()
    opt_a.step()
    resume_opt.step()
    resume_ready = bool(torch.allclose(param_a, resume_param, atol=1e-5, rtol=1e-4))
    finite_after_resume = bool(torch.isfinite(resume_param.detach()).all().item())
    return {
        "source_file": str(source.get("source_file", "")),
        "source_class": str(source.get("source_class", "")),
        "formula_step_execution_ready": formula_step_ready and finite_after_step,
        "resume_next_step_replay_ready": resume_ready and finite_after_resume,
        "state_dict_state_count": len(state.get("state", {})) if isinstance(state, Mapping) else 0,
        "formula_step_max_abs_diff": _max_abs_diff(torch, param_a.detach(), resume_param.detach())
        if not resume_ready
        else 0.0,
        "execution_reference": "existing_pytorch_optimizer_plugin",
        "formula_parity_implementation_ready": formula_step_ready and finite_after_step,
        "resume_parity_execution_ready": resume_ready and finite_after_resume,
        "resume_parity_implementation_ready": resume_ready and finite_after_resume,
        "native_kernel_ready": False,
    }


def _optimizer_class(name: str) -> tuple[type[Any], Mapping[str, Any]]:
    source = custom_formula_source_inventory(name)
    if source.get("status") != "ready":
        raise RuntimeError("custom formula source inventory missing")
    source_file = str(source.get("source_file", ""))
    class_name = str(source.get("source_class", ""))
    module_name = _module_name(source_file)
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    module = importlib.import_module(module_name)
    return getattr(module, class_name), source


def _module_name(source_file: str) -> str:
    path = Path(source_file)
    try:
        rel = path.relative_to(Path("plugin") / "pytorch_optimizer-main")
    except ValueError:
        rel = path
    return ".".join(rel.with_suffix("").parts)


def _torch() -> Any:
    import torch

    return torch


def _max_abs_diff(torch: Any, left: Any, right: Any) -> float:
    try:
        return float(torch.max(torch.abs(left - right)).item())
    except Exception:
        return float("inf")


def _unsafe_claims(rows: list[Mapping[str, Any]]) -> list[str]:
    fields = (
        "training_path_enabled",
        "default_behavior_changed",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "native_kernel_ready",
        "product_native_ready",
        "product_native_dispatch_ready",
    )
    claims: list[str] = []
    for row in rows:
        name = str(row.get("selected_optimizer_name", "unknown"))
        for field in fields:
            if row.get(field) is True:
                claims.append(f"unsafe_custom_formula_execution_row:{name}:{field}")
    return _dedupe(claims)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_custom_formula_execution_matrix"]
