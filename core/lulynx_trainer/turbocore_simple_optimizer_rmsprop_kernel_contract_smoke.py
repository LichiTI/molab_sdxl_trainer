"""Smoke for RMSProp first-class simple optimizer kernel contract visibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints  # noqa: E402


ENTRYPOINT = "get_simple_optimizer_kernel_contracts"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def run_smoke() -> dict[str, Any]:
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    assert native is not None, "lulynx_native missing get_simple_optimizer_kernel_contracts"
    payload = native.get_simple_optimizer_kernel_contracts()
    capability = dict(payload) if isinstance(payload, Mapping) else {}
    contracts = [dict(item) for item in capability.get("contracts", []) if isinstance(item, Mapping)]
    validations = [dict(item) for item in capability.get("validations", []) if isinstance(item, Mapping)]
    contract = _contract_for(contracts, "rmsprop")
    validation = _validation_for(validations, "rmsprop")
    input_roles = {str(item.get("role")) for item in contract.get("input_buffers", []) if isinstance(item, Mapping)}
    required_hyperparameters = {str(item) for item in contract.get("required_hyperparameters", [])}
    blocked_reasons = {str(item) for item in contract.get("blocked_reasons", [])}
    numeric_policy = dict(contract.get("numeric_policy", {}))
    branch_contracts = [
        dict(item) for item in contract.get("branch_contracts", []) if isinstance(item, Mapping)
    ]
    branch_ids = {str(item.get("branch_id")) for item in branch_contracts}

    assert capability.get("training_path_enabled") is False, capability
    assert capability.get("native_kernel_present") is False, capability
    assert contract["contract"] == "rmsprop_flat_fp32_cuda_kernel_v0", contract
    assert contract["launch_plan"] == "rmsprop_flat_fp32_launch_plan_v0", contract
    assert contract["kernel_name"] == "rmsprop_flat_fp32_cuda_v0", contract
    assert contract["available"] is False, contract
    assert contract["native_kernel_present"] is False, contract
    assert contract["training_path_enabled"] is False, contract
    assert {"param_flat", "grad_flat", "square_avg"}.issubset(input_roles), contract
    assert {"lr", "alpha", "eps"}.issubset(required_hyperparameters), contract
    assert numeric_policy.get("centered_branch_supported") is True, contract
    assert numeric_policy.get("momentum_branch_supported") is True, contract
    assert branch_ids == {"rmsprop_centered", "rmsprop_momentum"}, branch_contracts
    for branch in branch_contracts:
        assert branch.get("status") == "native_kernel_branch_supported", branch
        assert branch.get("native_kernel_present") is True, branch
        assert branch.get("training_path_enabled") is False, branch
    assert "product_dispatch_review_missing" in blocked_reasons, contract
    assert validation.get("ok") is True, validation
    assert validation.get("training_path_enabled") is False, validation
    assert validation.get("native_kernel_present") is False, validation

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_rmsprop_kernel_contract_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": {
            "rmsprop_first_class_contract_ready_count": 1,
            "rmsprop_first_class_contract_training_path_enabled_count": 0,
            "rmsprop_first_class_contract_native_kernel_present_count": 0,
            "rmsprop_branch_first_class_contract_count": len(branch_contracts),
            "rmsprop_centered_branch_kernel_supported_count": 1,
            "rmsprop_momentum_branch_kernel_supported_count": 1,
        },
        "recommended_next_step": "validate RMSProp centered/momentum native kernel branch parity before product exposure",
    }


def _contract_for(contracts: list[dict[str, Any]], optimizer_kind: str) -> dict[str, Any]:
    for contract in contracts:
        if contract.get("optimizer_kind") == optimizer_kind:
            return contract
    raise AssertionError(f"{optimizer_kind} contract missing")


def _validation_for(validations: list[dict[str, Any]], optimizer_kind: str) -> dict[str, Any]:
    for validation in validations:
        if validation.get("optimizer_kind") == optimizer_kind:
            return validation
    raise AssertionError(f"{optimizer_kind} validation missing")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
