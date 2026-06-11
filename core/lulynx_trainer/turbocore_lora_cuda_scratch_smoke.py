"""Smoke probe for the scratch-only Rust/CUDA LoRA kernel path."""

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

from core.services.native_module_loader import ensure_lulynx_native_artifact_path, native_with_entrypoints  # noqa: E402
from core.turbocore_lora_native_abi import SCRATCH_KERNEL_ENTRYPOINTS, probe_lora_cuda_scratch_kernel  # noqa: E402


def _assert_closed(payload: dict[str, Any]) -> None:
    assert payload["training_path_enabled"] is False, payload
    assert payload["training_dispatch"] is False, payload
    assert payload["training_tensor_binding"] is False, payload
    assert payload["training_parameters_mutated"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload


def run_smoke() -> dict[str, Any]:
    ensure_lulynx_native_artifact_path()
    native = native_with_entrypoints(*SCRATCH_KERNEL_ENTRYPOINTS)
    assert native is not None, SCRATCH_KERNEL_ENTRYPOINTS

    probe = probe_lora_cuda_scratch_kernel(workspace_root=PROJECT_ROOT)
    _assert_closed(probe)
    assert probe["present"] is True, probe
    assert probe["entrypoint_present"] is True, probe
    assert probe["scratch_kernel_probe_available"] is True, probe
    assert probe["scratch_buffers_only"] is True, probe

    if probe["ok"]:
        assert probe["kernel_executed"] is True, probe
        assert probe["parity_ok"] is True, probe
        assert probe["scratch_kernel_present"] is True, probe
        assert probe["native_kernel_present"] is True, probe
        assert int(probe["case_count"]) >= 4, probe
        assert int(probe["passed_case_count"]) == int(probe["case_count"]), probe
        assert int(probe["rank_count"]) >= 4, probe
        assert probe["native_candidate_repeated_validation_seen"] is True, probe
        assert float(probe["max_abs_diff"] or 0.0) <= 5e-5, probe
    else:
        assert probe["kernel_executed"] is False, probe
        assert probe["blocked_reasons"], probe

    return {
        "schema_version": 1,
        "probe": "turbocore_lora_cuda_scratch_smoke",
        "ok": True,
        "scratch_probe_ok": bool(probe["ok"]),
        "kernel_executed": bool(probe["kernel_executed"]),
        "case_count": int(probe["case_count"]),
        "passed_case_count": int(probe["passed_case_count"]),
        "rank_count": int(probe["rank_count"]),
        "native_candidate_repeated_validation_seen": bool(probe["native_candidate_repeated_validation_seen"]),
        "parity_ok": bool(probe["parity_ok"]),
        "max_abs_diff": probe.get("max_abs_diff"),
        "blocked_reasons": list(probe.get("blocked_reasons", []) or []),
        "training_path_enabled": False,
    }


def main() -> int:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
