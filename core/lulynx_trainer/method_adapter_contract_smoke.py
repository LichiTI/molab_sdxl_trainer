# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for method-adapter contract resolution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TRAINER_ROOT = Path(__file__).resolve().parent


def _load_local_module(module_name: str, filename: str):
    full_name = f"_lulynx_{module_name}_smoke_target"
    module = sys.modules.get(full_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(full_name, TRAINER_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


contract_mod = _load_local_module("method_adapter_contract", "method_adapter_contract.py")
normalize_adapter_method = contract_mod.normalize_adapter_method
resolve_adapter_method = contract_mod.resolve_adapter_method
adapter_contract_summary = contract_mod.adapter_contract_summary


def test_alias_normalization() -> None:
    assert normalize_adapter_method("lora+") == "lora_plus"
    assert normalize_adapter_method("rslora") == "rs_lora"
    assert normalize_adapter_method("rank_stabilized_lora") == "rs_lora"
    assert normalize_adapter_method("flexrank_lora") == "flexrank"
    assert normalize_adapter_method("lycoris-lokr") == "lokr"
    assert normalize_adapter_method("diag_oft") == "diag-oft"
    assert normalize_adapter_method("oft") == "diag-oft"
    assert normalize_adapter_method("networks.oft") == "diag-oft"
    print("PASS: adapter alias normalization")


def test_lora_family_contracts() -> None:
    methods = {
        "lora": ("lora", "networks.lora"),
        "lora_plus": ("lora", "networks.lora"),
        "rs_lora": ("lora", "networks.lora"),
        "dora": ("lora", "networks.lora"),
        "lora_fa": ("lora", "networks.lora_fa"),
        "vera": ("lora", "networks.vera"),
        "tlora": ("lora", "networks.tlora"),
        "flexrank": ("lora", "networks.flexrank_lora"),
        "hydralora": ("lora", "networks.lora"),
        "fera": ("lora", "networks.lora"),
    }
    for method, (backend, network_module) in methods.items():
        spec = resolve_adapter_method({"lora_type": method, "network_dim": 4, "network_alpha": 4}, family="anima")
        assert spec.supported, spec
        assert spec.backend == backend, spec
        assert spec.network_module == network_module, spec
        if method == "lora_plus":
            assert spec.requires_optimizer_grouping, spec
        if method == "rs_lora":
            assert "rs_lora_enabled" in spec.flags, spec
            assert not spec.safe_merge, spec
        if method == "flexrank":
            assert "flexrank_lora_enabled" in spec.flags, spec
            assert not spec.safe_merge, spec
        if method in {"vera", "hydralora", "fera"}:
            assert spec.requires_special_save, spec
    print("PASS: LoRA-family adapter contracts")


def test_lycoris_contracts() -> None:
    for method in ("loha", "locon", "lokr", "ia3", "full", "diag_oft"):
        spec = resolve_adapter_method({"lora_type": method}, family="newbie")
        expected = "diag-oft" if method == "diag_oft" else method
        assert spec.supported, spec
        assert spec.backend == "lycoris", spec
        assert spec.network_module == "lycoris.locon", spec
        assert spec.lycoris_algo == expected, spec
    print("PASS: LyCORIS adapter contracts")


def test_config_inference_and_summary() -> None:
    spec = resolve_adapter_method(
        {
            "model_type": "newbie",
            "newbie_adapter_type": "",
            "network_module": "lycoris.locon",
            "lycoris_algo": "diag_oft",
        }
    )
    assert spec.family == "newbie", spec
    assert spec.method == "diag-oft", spec
    summary = adapter_contract_summary(spec)
    assert "family=newbie" in summary
    assert "lycoris_algo=diag-oft" in summary

    flexrank = resolve_adapter_method({"network_module": "networks.flexrank_lora"}, family="sdxl")
    assert flexrank.method == "flexrank", flexrank
    assert "flexrank_lora_enabled" in flexrank.flags, flexrank

    legacy_oft = resolve_adapter_method({"network_module": "networks.oft"}, family="sdxl")
    assert legacy_oft.supported, legacy_oft
    assert legacy_oft.method == "diag-oft", legacy_oft
    assert legacy_oft.backend == "lycoris", legacy_oft
    assert legacy_oft.network_module == "lycoris.locon", legacy_oft
    assert legacy_oft.lycoris_algo == "diag-oft", legacy_oft

    unknown = resolve_adapter_method({"lora_type": "mystery"}, family="anima")
    assert not unknown.supported
    assert unknown.warnings
    print("PASS: adapter contract inference and summary")


def main() -> int:
    test_alias_normalization()
    test_lora_family_contracts()
    test_lycoris_contracts()
    test_config_inference_and_summary()
    print("PASS: method adapter contract smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
