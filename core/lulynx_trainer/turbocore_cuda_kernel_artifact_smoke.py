"""Developer smoke for TurboCore AdamW CUDA kernel artifact probes."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METADATA_PATH = PROJECT_ROOT / "backend" / "native" / "target" / "cuda" / "adamw_flat_fp32_cuda_v0.build.json"


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _read_artifact_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {
            "present": False,
            "path": str(METADATA_PATH),
            "ok": False,
            "reason": "artifact_metadata_missing",
        }
    with METADATA_PATH.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    payload["present"] = True
    payload["path"] = str(METADATA_PATH)
    return payload


def _validate_artifact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("present"):
        return {"ok": True, "skipped": True, "reason": "artifact_metadata_missing"}
    assert payload.get("schema_version") == 2, payload
    assert payload.get("native_kernel_present") is False, payload
    assert payload.get("training_path_enabled") is False, payload
    assert payload.get("performance_test_ready") is False, payload
    assert payload.get("artifact_only") is True, payload
    if payload.get("ok") is True:
        assert payload.get("ptx_exists") is True, payload
        assert int(payload.get("ptx_size_bytes") or 0) > 0, payload
        if not payload.get("ptx_only"):
            assert payload.get("object_exists") is True, payload
            assert int(payload.get("object_size_bytes") or 0) > 0, payload
    else:
        assert payload.get("stage") in {"preflight", "compile_ptx", "compile_obj", "verify_outputs"}, payload
        assert payload.get("failure"), payload
    return {
        "ok": True,
        "skipped": False,
        "artifact_ok": bool(payload.get("ok")),
        "stage": payload.get("stage"),
        "ptx_exists": bool(payload.get("ptx_exists")),
        "object_exists": bool(payload.get("object_exists")),
    }


def _probe_nvrtc() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    import lulynx_native  # type: ignore

    if not hasattr(lulynx_native, "probe_adamw_cuda_nvrtc_compile_py"):
        return {"ok": True, "skipped": True, "reason": "nvrtc_probe_entrypoint_missing"}
    probe = lulynx_native.probe_adamw_cuda_nvrtc_compile_py(str(PROJECT_ROOT), "compute_89")
    assert probe["training_path_enabled"] is False, probe
    assert probe["native_kernel_present"] is False, probe
    assert probe["performance_test_ready"] is False, probe
    assert probe["artifact_only"] is True, probe
    assert probe["kernel_executed"] is False, probe
    assert probe["parameters_mutated"] is False, probe
    if probe.get("ok"):
        assert int(probe.get("ptx_size_bytes") or 0) > 0, probe
        preview = str(probe.get("ptx_preview", ""))
        assert preview and ("NVIDIA" in preview or ".version" in preview), probe
    return {
        "ok": True,
        "skipped": False,
        "probe_ok": bool(probe.get("ok")),
        "reason": probe.get("reason", ""),
        "ptx_size_bytes": int(probe.get("ptx_size_bytes") or 0),
        "origin": str(getattr(spec, "origin", "")),
    }


def _probe_driver_ptx_load() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    import lulynx_native  # type: ignore

    if not hasattr(lulynx_native, "probe_adamw_cuda_driver_ptx_load_py"):
        return {"ok": True, "skipped": True, "reason": "driver_ptx_probe_entrypoint_missing"}
    probe = lulynx_native.probe_adamw_cuda_driver_ptx_load_py(str(PROJECT_ROOT), "compute_89")
    assert probe["training_path_enabled"] is False, probe
    assert probe["native_kernel_present"] is False, probe
    assert probe["performance_test_ready"] is False, probe
    assert probe["artifact_only"] is True, probe
    assert probe["kernel_executed"] is False, probe
    assert probe["parameters_mutated"] is False, probe
    if probe.get("ok"):
        assert probe.get("ptx_loaded") is True, probe
        assert probe.get("function_resolved") is True, probe
        assert int(probe.get("ptx_size_bytes") or 0) > 0, probe
    return {
        "ok": True,
        "skipped": False,
        "probe_ok": bool(probe.get("ok")),
        "reason": probe.get("reason", ""),
        "driver_version": int(probe.get("driver_version") or 0),
        "device_name": str(probe.get("device_name", "")),
        "ptx_loaded": bool(probe.get("ptx_loaded", False)),
        "function_resolved": bool(probe.get("function_resolved", False)),
        "origin": str(getattr(spec, "origin", "")),
    }


def _probe_scratch_launch() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    import lulynx_native  # type: ignore

    if not hasattr(lulynx_native, "probe_adamw_cuda_scratch_launch_py"):
        return {"ok": True, "skipped": True, "reason": "scratch_launch_probe_entrypoint_missing"}
    probe = lulynx_native.probe_adamw_cuda_scratch_launch_py(str(PROJECT_ROOT), "compute_89")
    assert probe["training_path_enabled"] is False, probe
    assert probe["native_kernel_present"] is False, probe
    assert probe["performance_test_ready"] is False, probe
    assert probe["scratch_buffers_only"] is True, probe
    assert probe["training_tensor_binding"] is False, probe
    assert probe["training_parameters_mutated"] is False, probe
    if probe.get("ok"):
        assert probe.get("kernel_executed") is True, probe
        assert probe.get("parameters_mutated") is True, probe
        assert probe.get("parity_ok") is True, probe
        max_abs_diff = probe.get("max_abs_diff")
        assert max_abs_diff is not None and float(max_abs_diff) <= 5e-6, probe
    return {
        "ok": True,
        "skipped": False,
        "probe_ok": bool(probe.get("ok")),
        "reason": probe.get("reason", ""),
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parity_ok": bool(probe.get("parity_ok", False)),
        "max_abs_diff": float(probe.get("max_abs_diff") or 0.0),
        "origin": str(getattr(spec, "origin", "")),
    }


def run_smoke() -> dict[str, Any]:
    metadata = _read_artifact_metadata()
    artifact = _validate_artifact_metadata(metadata)
    nvrtc = _probe_nvrtc()
    driver_ptx = _probe_driver_ptx_load()
    scratch_launch = _probe_scratch_launch()
    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_kernel_artifact_smoke",
        "ok": bool(artifact.get("ok"))
        and bool(nvrtc.get("ok"))
        and bool(driver_ptx.get("ok"))
        and bool(scratch_launch.get("ok")),
        "artifact_metadata": artifact,
        "nvrtc_compile_probe": nvrtc,
        "driver_ptx_load_probe": driver_ptx,
        "scratch_launch_probe": scratch_launch,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
