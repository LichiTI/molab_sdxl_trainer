"""TrainingLoop contract for post-approval TurboCore optimizer route binding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from core.turbocore_optimizer_product_training_route_binding_preflight import (
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_update_gate import TurboCoreNativeUpdateGate, build_native_update_gate_config
from core.turbocore_update_shadow import TurboCoreUpdateShadow, build_update_shadow_config
from core.lulynx_trainer.training_loop import TrainingLoop


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_training_loop_contract.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def build_optimizer_product_training_route_binding_training_loop_contract(
    *,
    preflight_report: Mapping[str, Any] | None = None,
    synthetic_signed_preflight_report: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    current = _as_dict(preflight_report) or build_optimizer_product_training_route_binding_preflight(
        artifact_dir=directory,
        write_artifact=True,
    )
    candidate_report = _as_dict(synthetic_signed_preflight_report) or _synthetic_signed_preflight(directory)
    candidate = _as_dict(candidate_report.get("post_approval_training_route_binding_candidate"))
    switches = _as_dict(candidate.get("existing_training_loop_switches"))
    current_context = _runtime_context(False, False, False)
    candidate_context = _runtime_context(
        bool(switches.get("turbocore_native_update_dispatch_enabled", False)),
        bool(switches.get("turbocore_native_update_training_path_enabled", False)),
        bool(switches.get("turbocore_native_update_require_native_cuda", False)),
    )
    missing_training_path = _runtime_context(True, False, True)
    missing_cuda = _runtime_context(True, True, False)
    current_closed = _context_closed(current_context)
    candidate_open = _context_open(candidate_context)
    all_three_required = _context_closed(missing_training_path) and _context_closed(missing_cuda)
    payload = {
        "schema_version": 1,
        "contract": "turbocore_optimizer_product_training_route_binding_training_loop_contract_v0",
        "gate": "optimizer_product_training_route_binding_training_loop_contract",
        "ok": bool(current_closed and candidate_open and all_three_required),
        "roadmap": ROADMAP,
        "artifact_first": True,
        "current_real_artifact_training_path_enabled": False,
        "post_approval_training_loop_context_ready": bool(candidate_open),
        "requires_all_three_switches": bool(all_three_required),
        "product_training_route_bound": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "summary": {
            "candidate_switch_count": len(switches),
            "open_training_path_enabled": 1 if candidate_context.get("training_path_enabled") is True else 0,
            "closed_training_path_enabled": 1 if current_context.get("training_path_enabled") is False else 0,
            "missing_training_path_closes_context_count": 1
            if missing_training_path.get("training_path_enabled") is False
            else 0,
            "missing_require_native_cuda_closes_context_count": 1
            if missing_cuda.get("training_path_enabled") is False
            else 0,
            "request_fields_emitted_count": 0,
            "schema_exposure_allowed_count": 0,
            "ui_exposure_allowed_count": 0,
        },
        "candidate_switches": switches,
        "candidate_runtime_context": _compact_context(candidate_context),
        "recommended_next_step": (
            "after real owner approval, bind these existing TrainingLoop switches in a separate product route change"
        ),
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _synthetic_signed_preflight(directory: Path) -> dict[str, Any]:
    return build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap=_read_json(directory / "turbocore_optimizer_native_readiness_gap_scorecard.json"),
        owner_release_review_record={"ok": True, "approval_recorded": True, "release_review_recorded": True},
        owner_release_direction_packet={
            "ok": True,
            "owner_release_direction_recorded": True,
            "owner_release_approval_recorded": True,
        },
        product_exposure_decision={
            "ok": True,
            "evidence_ready": True,
            "ready_for_product_exposure_review": True,
            "product_exposure_decision_recorded": True,
            "post_product_exposure_request_fields": {},
            "product_exposure_allowed": False,
            "training_launch_allowed": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ready_for_ui": False,
            "backend_router_registered": False,
        },
        release_review_package={
            "ok": True,
            "evidence_ready": True,
            "ready_for_review": True,
            "ready_for_owner_release_review": True,
            "release_review_recorded": True,
            "default_off": True,
            "post_release_request_fields": {},
            "release_gate_open": False,
            "training_launch_allowed": False,
            "runtime_dispatch_allowed": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "backend_router_registered": False,
        },
        write_artifact=False,
    )


def _runtime_context(dispatch_enabled: bool, training_path_enabled: bool, require_native_cuda: bool) -> dict[str, Any]:
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.multi_gpu = False
    loop.num_processes = 1
    loop.num_machines = 1
    loop.deepspeed = False
    loop._gradient_release_manager = None
    loop._turbocore_native_update_gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config("native_experimental", dispatch_enabled=dispatch_enabled)
    )
    loop._turbocore_native_update_training_path_enabled = bool(training_path_enabled)
    loop._turbocore_native_update_require_native_cuda = bool(require_native_cuda)
    loop._turbocore_update_shadow = TurboCoreUpdateShadow(build_update_shadow_config("off"))
    loop.optimizer = SimpleNamespace(
        param_groups=[{"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}]
    )
    loop.learning_rate = 1e-3
    loop.max_grad_norm = 1.0
    loop._turbocore_native_update_quantized_optimizer_kind = "adamw"
    loop._turbocore_native_update_defer_state_sync = False
    loop._turbocore_native_update_runtime_synchronization_policy = "context_synchronize"
    return loop._turbocore_native_update_runtime_context()


def _context_open(context: Mapping[str, Any]) -> bool:
    return all(
        context.get(field) is True
        for field in (
            "training_path_enabled",
            "native_update_training_dispatch_enabled",
            "native_update_runtime_dispatch_available",
            "native_update_executor_present",
            "native_update_runtime_execution_guard_enabled",
            "native_update_training_mutation_guard_enabled",
        )
    )


def _context_closed(context: Mapping[str, Any]) -> bool:
    return all(
        context.get(field) is False
        for field in (
            "training_path_enabled",
            "native_update_training_dispatch_enabled",
            "native_update_runtime_dispatch_available",
            "native_update_executor_present",
        )
    )


def _compact_context(context: Mapping[str, Any]) -> dict[str, Any]:
    config = _as_dict(context.get("native_update_training_executor_config"))
    return {
        "training_path_enabled": context.get("training_path_enabled") is True,
        "native_update_training_dispatch_enabled": context.get("native_update_training_dispatch_enabled") is True,
        "native_update_executor_present": context.get("native_update_executor_present") is True,
        "optimizer_kind": str(config.get("optimizer_kind", "") or ""),
        "require_native_cuda": config.get("require_native_cuda") is True,
        "prefer_native_cuda": config.get("prefer_native_cuda") is True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_optimizer_product_training_route_binding_training_loop_contract"]
