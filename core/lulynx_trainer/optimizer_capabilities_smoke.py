# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test optimizer capability reporting."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.config import OptimizerType
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.optimizer_capabilities import (
    optimizer_capability,
    optimizer_capability_report,
)
from core.lulynx_trainer.optimizer_plugin_support import (
    PLUGIN_PENDING_OR_SPECIAL,
    PLUGIN_RESUME_SMOKE_PASSED,
)


def _by_type(report: dict) -> dict[str, dict]:
    return {str(item["optimizer_type"]): item for item in report["optimizers"]}


def test_every_enum_has_capability_mapping() -> None:
    report = optimizer_capability_report()
    summary = report["summary"]
    assert summary["total"] == len(list(OptimizerType)), summary
    assert summary["missing_capability_mappings"] == [], summary

    by_type = _by_type(report)
    for optimizer in OptimizerType:
        assert optimizer.value in by_type, optimizer
        assert by_type[optimizer.value]["family"] != "unknown", by_type[optimizer.value]


def test_legacy_automagic_alias_normalizes_to_plus_plus() -> None:
    frontend = ConfigAdapter.from_frontend_dict({"schema_id": "sdxl-lora", "optimizer_type": "Automagic"})
    direct = UnifiedTrainingConfig.from_dict({"optimizer_type": "Automagic"})
    assert frontend.optimizer == OptimizerType.AUTOMAGIC_PLUS_PLUS, frontend.optimizer
    assert direct.optimizer == OptimizerType.AUTOMAGIC_PLUS_PLUS, direct.optimizer


def test_selector_and_schedule_policies_are_explicit() -> None:
    report = optimizer_capability_report()
    by_type = _by_type(report)

    assert by_type["PytorchOptimizer"]["family"] == "plugin_selector"
    assert by_type["PytorchOptimizer"]["state_resume"] == "depends_on_selected_optimizer"
    assert by_type["GenericOptimizer"]["family"] == "generic_selector"
    assert by_type["GenericOptimizer"]["state_resume"] == "depends_on_selected_optimizer"

    assert by_type["Automagic++"]["scheduler_policy"] == "constant"
    assert by_type["AutoProdigy"]["scheduler_policy"] == "constant"
    assert by_type["AdamWScheduleFree"]["scheduler_policy"] == "constant"
    assert by_type["prodigyplus.ProdigyPlusScheduleFree"]["scheduler_policy"] == "constant"


def test_plugin_report_exposes_local_pytorch_optimizer_names() -> None:
    capability = optimizer_capability(OptimizerType.PYTORCH_OPTIMIZER).as_dict()
    # The local plugin is enabled_by_default; if discovery regresses this becomes
    # blocked and the UI must not advertise plugin optimizers as concrete routes.
    assert capability["status"] == "available", capability
    names = {str(item).lower() for item in capability["plugin_optimizers"]}
    for expected in ("came", "stableadamw", "ademamix", "muon", "sophiah"):
        assert expected in names, (expected, sorted(list(names))[:20])


def test_plugin_report_distinguishes_resume_proof_from_discovery() -> None:
    capability = optimizer_capability(OptimizerType.PYTORCH_OPTIMIZER).as_dict()
    discovered = {str(item).lower() for item in capability["plugin_optimizers"]}
    resume_passed = {str(item).lower() for item in capability["plugin_resume_smoke_passed"]}
    assert resume_passed.issubset(discovered), (resume_passed - discovered, len(discovered))
    for expected in (
        "came",
        "stableadamw",
        "scion",
        "ademamix",
        "sophiah",
        "apollo",
        "apollodqn",
        "bsam",
        "ranger21",
        "schedulefreeadamw",
        "soap",
        "shampoo",
        "scalableshampoo",
        "muon",
        "adamuon",
        "adago",
        "a2grad",
        "adafactor",
        "adahessian",
        "adammini",
        "adalomo",
        "alig",
        "schedulefreeradam",
        "schedulefreesgd",
        "alice",
        "demo",
        "distributedmuon",
        "kron",
        "lbfgs",
        "lomo",
        "spam",
        "sgdsai",
        "spectralsphere",
    ):
        assert expected in resume_passed, (expected, sorted(resume_passed))
    assert len(resume_passed) == len(PLUGIN_RESUME_SMOKE_PASSED), len(resume_passed)
    for pending in PLUGIN_PENDING_OR_SPECIAL:
        assert pending not in resume_passed, pending

    special = {str(key).lower(): value for key, value in capability["plugin_special_handling"].items()}
    assert "muon" in special and "use_muon" in special["muon"], special
    assert "lbfgs" in special and "recompute closure" in special["lbfgs"], special
    assert "bsam" in special and "initial gradients" in special["bsam"], special
    assert "adammini" in special and "named-parameter" in special["adammini"], special
    assert "adahessian" in special and "create_graph" in special["adahessian"], special
    assert "alig" in special and "loss-only closure" in special["alig"], special
    assert "demo" in special and "identity gather" in special["demo"], special
    assert "adalomo" in special and "Fused-backward" in special["adalomo"], special
    assert "lomo" in special and "Fused-backward" in special["lomo"], special
    assert "distributedmuon" in special and "identity all_gather" in special["distributedmuon"], special
    assert "kron" in special and "einsum expressions" in special["kron"], special

    summary = capability["plugin_support_summary"]
    assert summary["resume_passed_count"] == len(PLUGIN_RESUME_SMOKE_PASSED), summary
    assert summary["pending_or_special_count"] == len(PLUGIN_PENDING_OR_SPECIAL), summary


def main() -> int:
    test_every_enum_has_capability_mapping()
    test_legacy_automagic_alias_normalizes_to_plus_plus()
    test_selector_and_schedule_policies_are_explicit()
    test_plugin_report_exposes_local_pytorch_optimizer_names()
    test_plugin_report_distinguishes_resume_proof_from_discovery()
    print("optimizer_capabilities_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
