# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Optimizer capability reporting for native training.

This module is deliberately report-only.  It does not instantiate optimizers or
change trainer behavior; it records which optimizer names are real first-class
routes, which routes depend on optional packages, and which ones are selector
bridges rather than concrete optimizer implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Iterable

from .config import OptimizerType
from .optimizer_plugin_support import (
    canonical_plugin_resume_names,
    plugin_special_handling_for_available,
    plugin_support_summary,
)


@dataclass(frozen=True)
class OptimizerCapability:
    optimizer_type: str
    status: str
    family: str
    implementation: str
    dependency: str = ""
    dependency_available: bool = True
    fallback_optimizer: str = ""
    scheduler_policy: str = "standard"
    state_resume: str = "expected"
    notes: tuple[str, ...] = ()
    plugin_optimizers: tuple[str, ...] = field(default_factory=tuple)
    plugin_resume_smoke_passed: tuple[str, ...] = field(default_factory=tuple)
    plugin_special_handling: dict[str, str] = field(default_factory=dict)
    plugin_support_summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        data["plugin_optimizers"] = list(self.plugin_optimizers)
        data["plugin_resume_smoke_passed"] = list(self.plugin_resume_smoke_passed)
        data["plugin_special_handling"] = dict(self.plugin_special_handling)
        data["plugin_support_summary"] = dict(self.plugin_support_summary)
        return data


_LOCAL_OPTIMIZERS: dict[OptimizerType, tuple[str, str, tuple[str, ...]]] = {
    OptimizerType.ADAMW: ("torch", "torch.optim.AdamW", ()),
    OptimizerType.KAHAN_ADAMW_8BIT: (
        "local",
        "core.lulynx_trainer.kahan_adamw8bit.KahanAdamW8bit",
        ("Local 8-bit moment optimizer with Kahan compensation.",),
    ),
    OptimizerType.AUTOMAGIC_PLUS_PLUS: (
        "local",
        "core.lulynx_trainer.automagic_plus_plus_optimizer.AutomagicPlusPlus",
        ("Local Warehouse optimizer; external LR scheduler is forced constant.",),
    ),
    OptimizerType.AUTO_PRODIGY: (
        "local",
        "core.lulynx_trainer.auto_prodigy_optimizer.AutoProdigy",
        ("Local Warehouse optimizer; external LR scheduler is forced constant.",),
    ),
    OptimizerType.ANIMA_FACTORED_ADAMW: (
        "local",
        "core.lulynx_trainer.anima_factored_optimizer.AnimaFactoredAdamW",
        ("Experimental full-finetune optimizer for large Anima DiT matrices.",),
    ),
    OptimizerType.MUON: (
        "local",
        "core.lulynx_trainer.muon_optimizer.Muon",
        ("Newton-Schulz orthogonalized momentum on 2D LoRA factors; AdamW fallback on 1D params.",),
    ),
    OptimizerType.SGD_NESTEROV: ("torch", "torch.optim.SGD(nesterov=True)", ()),
}

_BITSANDBYTES_OPTIMIZERS = frozenset({
    OptimizerType.ADAMW_8BIT,
    OptimizerType.PAGED_ADAMW,
    OptimizerType.PAGED_ADAMW_32BIT,
    OptimizerType.PAGED_ADAMW_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
    OptimizerType.LION_8BIT,
})

_DADAPT_OPTIMIZERS = frozenset({
    OptimizerType.DADAPTATION,
    OptimizerType.DADAPT_ADAM_PREPRINT,
    OptimizerType.DADAPT_ADAGRAD,
    OptimizerType.DADAPT_ADAM,
    OptimizerType.DADAPT_ADAN,
    OptimizerType.DADAPT_ADAN_IP,
    OptimizerType.DADAPT_LION,
    OptimizerType.DADAPT_SGD,
})

_SCHEDULE_FREE_OPTIMIZERS = frozenset({
    OptimizerType.ADAMW_SCHEDULE_FREE,
    OptimizerType.RADAM_SCHEDULE_FREE,
    OptimizerType.SGD_SCHEDULE_FREE,
})

_DEPENDENCY_BY_OPTIMIZER: dict[OptimizerType, str] = {
    OptimizerType.PRODIGY: "prodigyopt",
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE: "prodigyplus",
    OptimizerType.ADAFACTOR: "transformers",
    OptimizerType.LION: "lion_pytorch",
}
for _opt in _BITSANDBYTES_OPTIMIZERS:
    _DEPENDENCY_BY_OPTIMIZER[_opt] = "bitsandbytes"
for _opt in _DADAPT_OPTIMIZERS:
    _DEPENDENCY_BY_OPTIMIZER[_opt] = "dadaptation"
for _opt in _SCHEDULE_FREE_OPTIMIZERS:
    _DEPENDENCY_BY_OPTIMIZER[_opt] = "schedulefree"


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _plugin_optimizer_names() -> tuple[str, ...]:
    try:
        from .optimizer_plugin_bridge import list_pytorch_optimizer_capabilities

        caps = list_pytorch_optimizer_capabilities()
    except Exception:
        return ()
    names = caps.get("optimizers", []) if isinstance(caps, dict) else []
    return tuple(sorted(str(item) for item in names if str(item).strip()))


def optimizer_capability(opt: OptimizerType | str) -> OptimizerCapability:
    optimizer = opt if isinstance(opt, OptimizerType) else OptimizerType(str(opt))
    value = optimizer.value

    if optimizer in _LOCAL_OPTIMIZERS:
        family, implementation, notes = _LOCAL_OPTIMIZERS[optimizer]
        scheduler_policy = "constant" if optimizer in {OptimizerType.AUTOMAGIC_PLUS_PLUS, OptimizerType.AUTO_PRODIGY} else "standard"
        return OptimizerCapability(
            optimizer_type=value,
            status="available",
            family=family,
            implementation=implementation,
            scheduler_policy=scheduler_policy,
            notes=notes,
        )

    if optimizer == OptimizerType.PYTORCH_OPTIMIZER:
        plugin_names = _plugin_optimizer_names()
        smoke_passed = canonical_plugin_resume_names(plugin_names)
        special = plugin_special_handling_for_available(plugin_names)
        return OptimizerCapability(
            optimizer_type=value,
            status="available" if plugin_names else "blocked",
            family="plugin_selector",
            implementation="optimizer_plugin_bridge.create_pytorch_optimizer",
            dependency="plugin:pytorch_optimizer",
            dependency_available=bool(plugin_names),
            state_resume="depends_on_selected_optimizer",
            notes=(
                "Requires optimizer_args name=<optimizer>; this enum is a selector, not one concrete optimizer.",
                "Only plugin_resume_smoke_passed entries have trainer-path step/state/load/next-step parity proof.",
            ),
            plugin_optimizers=plugin_names,
            plugin_resume_smoke_passed=smoke_passed,
            plugin_special_handling=special,
            plugin_support_summary=plugin_support_summary(plugin_names),
        )

    if optimizer == OptimizerType.GENERIC:
        return OptimizerCapability(
            optimizer_type=value,
            status="available",
            family="generic_selector",
            implementation="optimizer_plugin_bridge.create_generic_optimizer",
            dependency="selected optimizer path",
            dependency_available=True,
            state_resume="depends_on_selected_optimizer",
            notes=(
                "Requires optimizer_args name=<torch.optim class | plugin optimizer | dotted class path>.",
            ),
        )

    if optimizer == OptimizerType.LION_8BIT:
        bnb = module_available("bitsandbytes")
        lion = module_available("lion_pytorch")
        return OptimizerCapability(
            optimizer_type=value,
            status="available" if (bnb or lion) else "available_with_fallback",
            family="optional_dependency",
            implementation="bitsandbytes.optim.Lion8bit or lion_pytorch.Lion",
            dependency="bitsandbytes|lion_pytorch",
            dependency_available=bool(bnb or lion),
            fallback_optimizer="AdamW8bit/AdamW",
            notes=("Lion8bit tries bitsandbytes first, then lion_pytorch, then AdamW fallback.",),
        )

    dependency = _DEPENDENCY_BY_OPTIMIZER.get(optimizer, "")
    if dependency:
        available = module_available(dependency)
        family = "optional_dependency"
        scheduler_policy = "constant" if optimizer in _SCHEDULE_FREE_OPTIMIZERS or optimizer == OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE else "standard"
        implementation = _implementation_name(optimizer)
        if optimizer in _DADAPT_OPTIMIZERS and not available:
            plugin_available = bool(_plugin_optimizer_names())
            return OptimizerCapability(
                optimizer_type=value,
                status="available" if plugin_available else "available_with_fallback",
                family=family,
                implementation=f"{implementation} or pytorch_optimizer DAdapt route",
                dependency="dadaptation|plugin:pytorch_optimizer",
                dependency_available=plugin_available,
                fallback_optimizer="pytorch_optimizer DAdaptAdam/DAdaptAdaGrad/DAdaptAdan/DAdaptLion/DAdaptSGD or AdamW",
                scheduler_policy=scheduler_policy,
                notes=(
                    "When dadaptation is missing, Trainer uses the local pytorch_optimizer DAdapt equivalent before AdamW fallback.",
                ),
            )
        return OptimizerCapability(
            optimizer_type=value,
            status="available" if available else "available_with_fallback",
            family=family,
            implementation=implementation,
            dependency=dependency,
            dependency_available=available,
            fallback_optimizer="AdamW",
            scheduler_policy=scheduler_policy,
            notes=("Trainer falls back to torch.optim.AdamW when the optional dependency is missing.",),
        )

    return OptimizerCapability(
        optimizer_type=value,
        status="blocked",
        family="unknown",
        implementation="",
        dependency_available=False,
        state_resume="unknown",
        notes=("No optimizer capability mapping exists for this enum value.",),
    )


def optimizer_capability_report(
    optimizers: Iterable[OptimizerType | str] | None = None,
) -> dict[str, Any]:
    items = [optimizer_capability(opt) for opt in (optimizers or list(OptimizerType))]
    missing = [item.optimizer_type for item in items if item.family == "unknown"]
    return {
        "schema_version": 1,
        "optimizers": [item.as_dict() for item in items],
        "summary": {
            "total": len(items),
            "available": sum(1 for item in items if item.status == "available"),
            "available_with_fallback": sum(1 for item in items if item.status == "available_with_fallback"),
            "blocked": sum(1 for item in items if item.status == "blocked"),
            "missing_capability_mappings": missing,
        },
    }


def _implementation_name(optimizer: OptimizerType) -> str:
    mapping = {
        OptimizerType.ADAMW_8BIT: "bitsandbytes.optim.AdamW8bit",
        OptimizerType.PAGED_ADAMW: "bitsandbytes.optim.PagedAdamW",
        OptimizerType.PAGED_ADAMW_32BIT: "bitsandbytes.optim.PagedAdamW32bit",
        OptimizerType.PAGED_ADAMW_8BIT: "bitsandbytes.optim.PagedAdamW8bit",
        OptimizerType.PAGED_LION_8BIT: "bitsandbytes.optim.PagedLion8bit",
        OptimizerType.SGD_NESTEROV_8BIT: "bitsandbytes.optim.SGD8bit",
        OptimizerType.PRODIGY: "prodigyopt.Prodigy",
        OptimizerType.ADAFACTOR: "transformers.optimization.Adafactor",
        OptimizerType.LION: "lion_pytorch.Lion",
        OptimizerType.DADAPTATION: "dadaptation.experimental.DAdaptAdamPreprint",
        OptimizerType.DADAPT_ADAM_PREPRINT: "dadaptation.experimental.DAdaptAdamPreprint",
        OptimizerType.DADAPT_ADAGRAD: "dadaptation.DAdaptAdaGrad",
        OptimizerType.DADAPT_ADAM: "dadaptation.DAdaptAdam",
        OptimizerType.DADAPT_ADAN: "dadaptation.DAdaptAdan",
        OptimizerType.DADAPT_ADAN_IP: "dadaptation.experimental.DAdaptAdanIP",
        OptimizerType.DADAPT_LION: "dadaptation.DAdaptLion",
        OptimizerType.DADAPT_SGD: "dadaptation.DAdaptSGD",
        OptimizerType.ADAMW_SCHEDULE_FREE: "schedulefree.AdamWScheduleFree",
        OptimizerType.RADAM_SCHEDULE_FREE: "schedulefree.RAdamScheduleFree",
        OptimizerType.SGD_SCHEDULE_FREE: "schedulefree.SGDScheduleFree",
        OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE: "prodigyplus.ProdigyPlusScheduleFree",
    }
    return mapping.get(optimizer, optimizer.value)
