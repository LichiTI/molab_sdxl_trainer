"""Smoke tests for TurboCore candidate registry integration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_candidates import (  # noqa: E402
    TurboCoreCandidate,
    candidate_names,
    list_turbocore_candidates,
    pytorch_lora_delta_candidate,
    register_turbocore_candidate,
)
from core.turbocore_parity import check_lora_delta_parity  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import run_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_readiness_probe import build_readiness_report  # noqa: E402
from core.lulynx_trainer.turbocore_torch_compile_candidate_probe import run_probe  # noqa: E402


def test_default_candidate_registry() -> None:
    candidates = list_turbocore_candidates()
    assert "pytorch_explicit" in candidate_names("lora_fused")
    assert "torch_compile" in candidate_names("lora_fused")
    assert "triton_lora_delta_v0" in candidate_names("lora_fused")
    assert "triton_lora_delta_v1" in candidate_names("lora_fused")
    assert "triton_lora_delta_v2" in candidate_names("lora_fused")
    assert "triton_lora_delta_v2_tc" in candidate_names("lora_fused")
    assert "triton_lora_delta_v3_dispatch" in candidate_names("lora_fused")
    assert "rust_cuda_lora_delta_v0" in candidate_names("lora_fused")
    assert "pytorch_adamw" in candidate_names("native_optimizer")
    assert "rust_cuda_adamw_v0" in candidate_names("native_optimizer")
    assert candidates["lora_fused"][0]["native"] is False
    compile_rows = [row for row in candidates["lora_fused"] if row["name"] == "torch_compile"]
    assert compile_rows and compile_rows[0]["experimental"] is True
    reserved = [row for row in candidates["lora_fused"] if row["name"] == "rust_cuda_lora_delta_v0"]
    assert reserved and reserved[0]["available"] is False


def test_registered_lora_candidate_flows_to_parity_and_benchmark() -> None:
    register_turbocore_candidate(
        TurboCoreCandidate(
            name="smoke_lora_delta",
            feature="lora_fused",
            callable=pytorch_lora_delta_candidate,
            native=False,
            experimental=True,
            description="Smoke alias for the PyTorch LoRA delta candidate.",
        )
    )
    parity = check_lora_delta_parity(
        batch=1,
        tokens=8,
        in_features=16,
        out_features=16,
        rank=4,
        dtype=torch.float32,
        device="cpu",
        candidate_name="smoke_lora_delta",
    )
    assert parity.ok, parity.as_dict()
    assert parity.details and parity.details["candidate"] == "smoke_lora_delta"

    bench = run_benchmark(
        preset="tiny",
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cpu"),
        iters=1,
        warmup=0,
        candidate_name="smoke_lora_delta",
    )
    assert bench["results"][0]["candidate"] == "smoke_lora_delta"
    assert bench["summary"]["native_kernel_present"] is False


def test_torch_compile_probe_unavailable_path_is_non_fatal() -> None:
    import core.lulynx_trainer.turbocore_torch_compile_candidate_probe as probe_mod

    original = probe_mod.get_turbocore_candidate
    probe_mod.get_turbocore_candidate = lambda _feature, _name=None: None
    try:
        payload = run_probe(device=torch.device("cpu"), dtype=torch.float32, preset="tiny", rank=4, iters=1, warmup=0)
    finally:
        probe_mod.get_turbocore_candidate = original
    assert payload["non_fatal"] is True
    assert payload["available"] is False
    assert payload["reason"] == "candidate_unavailable"


def test_readiness_probe_without_torch_compile() -> None:
    payload = build_readiness_report(
        device=torch.device("cpu"),
        dtype=torch.float32,
        preset="tiny",
        rank=4,
        iters=1,
        warmup=0,
        include_torch_compile=False,
    )
    assert payload["probe"] == "turbocore_readiness"
    assert payload["summary"]["ready_for_ui"] is False
    assert "parity" in payload["sections"]
    assert payload["sections"]["parity"]["summary"]["ok"] is True
    assert "native_update_training_loop_dispatch_smoke" in payload["sections"]
    dispatch_smoke = payload["sections"]["native_update_training_loop_dispatch_smoke"]
    assert dispatch_smoke["ok"] is True
    assert dispatch_smoke["training_path_enabled"] is False
    assert payload["summary"]["native_update_training_executor_available"] is False
    assert payload["summary"]["native_update_training_loop_dispatch_smoke_ok"] is False


if __name__ == "__main__":
    test_default_candidate_registry()
    test_registered_lora_candidate_flows_to_parity_and_benchmark()
    test_torch_compile_probe_unavailable_path_is_non_fatal()
    test_readiness_probe_without_torch_compile()
    print("turbocore_candidates_smoke: ok")
