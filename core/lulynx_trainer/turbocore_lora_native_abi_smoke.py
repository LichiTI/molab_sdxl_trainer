"""Smoke probe for the Rust/CUDA LoRA fused ABI contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_lora_native_abi import (  # noqa: E402
    REQUIRED_ENTRYPOINTS,
    SCRATCH_KERNEL_ENTRYPOINTS,
    probe_lora_cuda_scratch_kernel,
    probe_lora_fused_native_abi,
)
from core.turbocore_lora_native_runtime import probe_lora_native_training_dispatch  # noqa: E402
from core.services.native_module_loader import ensure_lulynx_native_artifact_path, native_with_entrypoints  # noqa: E402


def run_smoke() -> dict[str, Any]:
    ensure_lulynx_native_artifact_path()
    native = native_with_entrypoints(*(REQUIRED_ENTRYPOINTS + SCRATCH_KERNEL_ENTRYPOINTS))
    assert native is not None, REQUIRED_ENTRYPOINTS

    contract = native.get_lora_fused_kernel_contract()
    assert contract["contract"] == "lora_delta_add_cuda_kernel_v0", contract
    assert contract["native_kernel_present"] is True, contract
    assert contract["training_path_enabled"] is True, contract

    plan = native.build_lora_fused_launch_plan(
        json.dumps([2, 64, 320]),
        json.dumps([4, 320]),
        json.dumps([320, 4]),
        json.dumps([2, 64, 320]),
        "float32",
        4,
        1.0,
    )
    assert plan["plan_kind"] == "lora_delta_add_launch_plan_v0", plan
    assert plan["shape_contract_ok"] is True, plan
    assert plan["launch_allowed"] is True, plan
    assert plan["native_kernel_present"] is True, plan
    assert plan["training_path_enabled"] is True, plan
    assert plan["blocked_reasons"] == [], plan

    validation = native.validate_lora_fused_launch_plan(json.dumps(plan))
    assert validation["ok"] is True, validation
    assert validation["launch_allowed"] is True, validation

    probe = probe_lora_fused_native_abi(x_shape=(2, 64, 320), rank=4, out_features=320, dtype="float32")
    assert probe["ok"] is True, probe
    assert probe["abi_contract_available"] is True, probe
    assert probe["native_kernel_present"] is True, probe
    assert probe["native_dispatch_allowed"] is True, probe
    assert probe["training_path_enabled"] is True, probe

    scratch = probe_lora_cuda_scratch_kernel(workspace_root=PROJECT_ROOT)
    assert scratch["present"] is True, scratch
    assert scratch["entrypoint_present"] is True, scratch
    assert scratch["training_path_enabled"] is False, scratch
    assert scratch["training_dispatch"] is False, scratch
    assert scratch["training_tensor_binding"] is False, scratch
    if scratch["ok"]:
        assert scratch["kernel_executed"] is True, scratch
        assert scratch["parity_ok"] is True, scratch
        assert scratch["native_candidate_repeated_validation_seen"] is True, scratch
    else:
        assert scratch["blocked_reasons"], scratch

    training = probe_lora_native_training_dispatch()
    assert training["ok"] is True, training
    assert training["kernel_executed"] is True, training
    assert training["training_dispatch"] is True, training
    assert training["training_path_enabled"] is True, training
    assert training["forward_parity_ok"] is True, training
    assert training["backward_parity_ok"] is True, training

    return {
        "schema_version": 1,
        "probe": "turbocore_lora_native_abi_smoke",
        "ok": True,
        "entrypoints": list(REQUIRED_ENTRYPOINTS + SCRATCH_KERNEL_ENTRYPOINTS),
        "contract": contract["contract"],
        "launch_plan_kind": plan["plan_kind"],
        "abi_contract_available": probe["abi_contract_available"],
        "scratch_kernel_probe_ok": scratch["ok"],
        "scratch_kernel_executed": scratch["kernel_executed"],
        "scratch_kernel_case_count": scratch["case_count"],
        "native_training_dispatch_ok": training["ok"],
        "native_kernel_present": True,
        "training_path_enabled": True,
    }


def main() -> int:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
