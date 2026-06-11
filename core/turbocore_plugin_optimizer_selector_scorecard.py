"""Report-only scorecard for plugin optimizer selector expansion.

``PytorchOptimizer`` and ``GenericOptimizer`` are selectors, not optimizer
formulas.  Native work must classify the selected optimizer name before it can
choose a kernel, state layout, or runtime fallback.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from core.configs import OptimizerType
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report
from core.lulynx_trainer.optimizer_plugin_support import (
    PLUGIN_MUON_FAMILY_OPTIMIZERS,
    PLUGIN_SCHEDULE_FREE_OPTIMIZERS,
    PLUGIN_SPECIAL_HANDLING,
)


_ADAPTIVE_LR = frozenset({
    "prodigy",
    "dadaptadagrad",
    "dadaptadam",
    "dadaptadan",
    "dadaptlion",
    "dadaptsgd",
})

_FACTORED_MEMORY = frozenset({
    "adafactor",
    "came",
    "emofact",
    "galore",
    "scalableshampoo",
    "shampoo",
    "sm3",
    "soap",
})

_CLOSURE_OR_SECOND_ORDER = frozenset({
    "adahessian",
    "alig",
    "bsam",
    "kron",
    "lbfgs",
})

_FUSED_BACKWARD = frozenset({"adalomo", "lomo"})

_MODEL_OR_SHAPE_AWARE = frozenset({
    "adammini",
    "alice",
    "distributedmuon",
    "spectralsphere",
}) | PLUGIN_MUON_FAMILY_OPTIMIZERS

_STATE_ADAPTER_SPECIAL = frozenset({"demo", "sgdsai", "spam"})

_SIMPLE_FORMULA = frozenset({
    "accsgd",
    "aggmo",
    "asgd",
    "fromage",
    "gravity",
    "lars",
    "lion",
    "madgrad",
    "nero",
    "pid",
    "qhm",
    "rmsprop",
    "sgd",
    "sgdp",
    "sgdw",
    "signsgd",
    "tiger",
    "vsgd",
})

_ADAM_LIKE_MARKERS = ("adam", "ranger", "radam", "nadam", "lamb", "yogi", "novograd")

_ROUTE_PRIORITY = {
    "schedule_free_state_machine": "P8A-like",
    "adaptive_lr_state_machine": "P9-like",
    "factored_memory_layout": "P10-like",
    "closure_or_second_order": "manual_contract",
    "fused_backward": "manual_contract",
    "model_or_shape_aware": "manual_contract",
    "state_adapter_special": "manual_contract",
    "simple_formula": "P7-like",
    "adam_like_formula": "P8-like",
    "custom_formula": "manual_contract",
}


def build_plugin_optimizer_selector_scorecard() -> dict[str, Any]:
    """Classify plugin optimizer selectors without instantiating optimizers."""

    capability = _selector_capability(OptimizerType.PYTORCH_OPTIMIZER)
    generic = _selector_capability(OptimizerType.GENERIC)
    plugin_names = [str(name).strip().lower() for name in capability.get("plugin_optimizers", []) if str(name).strip()]
    resume_names = {
        str(name).strip().lower()
        for name in capability.get("plugin_resume_smoke_passed", [])
        if str(name).strip()
    }
    special = {str(key).strip().lower(): str(value) for key, value in capability.get("plugin_special_handling", {}).items()}
    rows = [_plugin_row(name, resume_names, special) for name in sorted(set(plugin_names))]
    missing_resume = sorted(set(plugin_names) - resume_names)
    unclassified = [row["optimizer_name"] for row in rows if row["native_route_family"] == "unclassified"]
    selector_contracts = [_pytorch_selector_contract(capability), _generic_selector_contract(generic)]
    family_counts = Counter(str(row["native_route_family"]) for row in rows)
    direct_native_blocked = all(not bool(row.get("native_dispatch_allowed", True)) for row in rows)
    selector_boundary_ready = all(bool(item.get("selector_boundary_ready", False)) for item in selector_contracts)
    ready = bool(rows) and not missing_resume and not unclassified and direct_native_blocked and selector_boundary_ready

    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_optimizer_selector_scorecard_v0",
        "gate": "plugin_optimizer_selector_expansion",
        "ok": ready,
        "promotion_ready": False,
        "plugin_selector_classification_ready": ready,
        "selector_boundary_ready": selector_boundary_ready,
        "all_discovered_plugins_resume_proven": not missing_resume,
        "missing_classification_count": len(unclassified),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "selector_types": [OptimizerType.PYTORCH_OPTIMIZER.value, OptimizerType.GENERIC.value],
        "selector_contracts": selector_contracts,
        "summary": {
            "plugin_optimizer_count": len(rows),
            "resume_proven_count": len(resume_names),
            "missing_resume_count": len(missing_resume),
            "special_handling_count": len(special),
            "route_family_counts": dict(sorted(family_counts.items())),
            "manual_contract_family_count": sum(
                count for family, count in family_counts.items() if _ROUTE_PRIORITY.get(family) == "manual_contract"
            ),
        },
        "rows": rows,
        "missing_resume_plugins": missing_resume,
        "unclassified_plugins": unclassified,
        "promotion_blockers": (
            [f"plugin_resume_missing:{name}" for name in missing_resume]
            + [f"plugin_unclassified:{name}" for name in unclassified]
            + ["selected_optimizer_native_abi_missing", "owner_release_hold_missing"]
        ),
        "blocked_reasons": [f"plugin_resume_missing:{name}" for name in missing_resume]
        + [f"plugin_unclassified:{name}" for name in unclassified],
        "recommended_next_step": (
            "choose one high-use plugin route family and build a selected-optimizer native ABI gate"
            if ready
            else "fix plugin selector classification or resume coverage gaps"
        ),
        "notes": [
            "This scorecard is report-only and never enables native dispatch.",
            "Selector enums must be resolved to a selected optimizer name before native work.",
            "Rows are structural classifications; no third-party optimizer implementation is copied.",
        ],
    }


def _selector_capability(optimizer: OptimizerType) -> dict[str, Any]:
    report = optimizer_capability_report([optimizer])
    for item in report.get("optimizers", []):
        if isinstance(item, Mapping) and item.get("optimizer_type") == optimizer.value:
            return dict(item)
    return {}


def _plugin_row(name: str, resume_names: set[str], special: Mapping[str, str]) -> dict[str, Any]:
    family = _classify_plugin(name)
    return {
        "optimizer_name": name,
        "selector": OptimizerType.PYTORCH_OPTIMIZER.value,
        "native_route_family": family,
        "native_route_priority": _ROUTE_PRIORITY.get(family, "manual_contract"),
        "resume_proven": name in resume_names,
        "special_handling": str(special.get(name, "")),
        "requires_selected_optimizer_request": True,
        "selector_is_not_optimizer": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "next_gate": _next_gate(family),
    }


def _classify_plugin(name: str) -> str:
    if name in PLUGIN_SCHEDULE_FREE_OPTIMIZERS:
        return "schedule_free_state_machine"
    if name in _ADAPTIVE_LR:
        return "adaptive_lr_state_machine"
    if name in _FACTORED_MEMORY:
        return "factored_memory_layout"
    if name in _CLOSURE_OR_SECOND_ORDER:
        return "closure_or_second_order"
    if name in _FUSED_BACKWARD:
        return "fused_backward"
    if name in _MODEL_OR_SHAPE_AWARE:
        return "model_or_shape_aware"
    if name in _STATE_ADAPTER_SPECIAL:
        return "state_adapter_special"
    if name in _SIMPLE_FORMULA:
        return "simple_formula"
    if any(marker in name for marker in _ADAM_LIKE_MARKERS):
        return "adam_like_formula"
    return "custom_formula"


def _next_gate(family: str) -> str:
    mapping = {
        "schedule_free_state_machine": "reuse P8A-style train/eval state-machine gate for selected plugin optimizer",
        "adaptive_lr_state_machine": "reuse P9-style dynamic LR state-machine gate for selected plugin optimizer",
        "factored_memory_layout": "reuse P10-style state layout and quality matrix gate",
        "closure_or_second_order": "write closure/create_graph training-loop ABI before any native work",
        "fused_backward": "write fused-backward ownership and skip-step ABI before any native work",
        "model_or_shape_aware": "write model-aware param grouping and tensor-shape ABI before any native work",
        "state_adapter_special": "write special state adapter and resume ABI before any native work",
        "simple_formula": "reuse P7-style formula parity before kernel work",
        "adam_like_formula": "prove AdamW compatibility or write separate formula/state layout gate",
        "custom_formula": "write selected optimizer formula/state reference first",
    }
    return mapping.get(family, "write selected optimizer formula/state reference first")


def _pytorch_selector_contract(capability: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selector_boundary_ready": bool(capability.get("plugin_optimizers", [])),
        "requires_optimizer_args_name": True,
        "selected_optimizer_source": "optimizer_args.name",
        "native_dispatch_policy": "disabled_until_selected_optimizer_scorecard_passes",
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
    }


def _generic_selector_contract(capability: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selector_type": OptimizerType.GENERIC.value,
        "selector_boundary_ready": bool(capability),
        "requires_optimizer_args_name": True,
        "selected_optimizer_source": "optimizer_args.name_or_dotted_class",
        "native_dispatch_policy": "disabled_until_request_time_selected_optimizer_is_classified",
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
    }


__all__ = ["build_plugin_optimizer_selector_scorecard"]
