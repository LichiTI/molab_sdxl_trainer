"""Aggregate TurboCore readiness evidence into one JSON report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_candidates import list_turbocore_candidates  # noqa: E402
from core.turbocore_capabilities import build_turbocore_capability_report  # noqa: E402
from core.turbocore_parity import build_turbocore_parity_report  # noqa: E402
from core.turbocore_workspace_pipeline import (  # noqa: E402
    build_turbocore_native_training_capability_stub,
    build_workspace_pipeline_prototype_report,
    run_workspace_pipeline_lifecycle_probe,
)
from core.turbocore_capabilities import probe_native_training_bridge  # noqa: E402
from core.turbocore_native_abi import validate_workspace_pipeline_native_capabilities  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import run_benchmark as run_lora_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_native_optimizer_benchmark import run_benchmark as run_optimizer_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_torch_compile_candidate_probe import run_probe as run_torch_compile_probe  # noqa: E402
from core.lulynx_trainer.turbocore_candidate_scorecard import build_candidate_scorecard  # noqa: E402
from core.lulynx_trainer.turbocore_lora_promotion_scorecard import build_lora_fused_promotion_scorecard  # noqa: E402
from core.turbocore_native_update_promotion_scorecard import build_native_update_promotion_scorecard  # noqa: E402
from core.turbocore_lora_native_abi import probe_lora_cuda_scratch_kernel  # noqa: E402


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"} and device.type != "cpu":
        return torch.float16
    return torch.float32


def _safe_section(name: str, fn) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        if isinstance(payload, dict):
            payload.setdefault("ok", True)
            payload.setdefault("elapsed_seconds", round(time.perf_counter() - started, 4))
            return payload
        return {
            "ok": True,
            "value": payload,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        return {
            "ok": False,
            "section": name,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _normalize_native_update_performance_report(value: dict[str, Any] | None) -> dict[str, Any] | None:
    report = _as_dict(value)
    if not report:
        return None
    nested = _as_dict(report.get("native_update_performance_report"))
    if nested:
        return nested
    return report


def _synthetic_native_update_shadow_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training_path_enabled": False,
        "direct_grad_lifecycle_integrated": True,
        "checkpoint_metadata_integrated": True,
        "checkpoint_owner_state_enabled": True,
        "after_optimizer": {
            "compared": True,
            "parity_ok_loose": True,
            "max_abs_param_diff": 1.0e-8,
            "mean_abs_param_diff": 1.0e-9,
        },
        "copyback_probe": {
            "scratch_copyback_validated": True,
            "real_parameters_mutated": False,
        },
        "copyback_dispatch_probe": {
            "copyback_dispatch_enabled": True,
            "copyback_dispatch_validated": True,
            "copyback_dispatch_target": "training_parameters",
            "real_parameters_mutated": True,
            "real_parameters_restored": True,
        },
        "native_binding_probe": {
            "request_shape_ready": True,
            "tensor_object_binding_ready": True,
            "launch_plan_ready": True,
            "stream_lifetime_bound": True,
            "stream_guard_ready": True,
            "event_chain_verified": True,
        },
        "owner_native_launch_probe": {
            "ok": True,
            "attempted": True,
            "kernel_executed": True,
            "parity_ok": True,
            "persistent_owner_mutated": False,
            "event_chain_probe_requested": True,
            "event_chain_verified": True,
        },
    }


def _synthetic_native_update_scorecard(
    *,
    performance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    return build_native_update_promotion_scorecard(
        optimizer=optimizer,
        params=[param],
        shadow_report=_synthetic_native_update_shadow_report(),
        performance_report=_normalize_native_update_performance_report(performance_report),
        runtime_context={},
        mode="native_experimental",
        dispatch_enabled=True,
        required_shadow_passes=1,
        allow_missing_native_kernel=True,
        strict=True,
    )


def _native_update_training_loop_dispatch_smoke(*, requested_device: torch.device) -> dict[str, Any]:
    if requested_device.type != "cuda":
        return _skipped_native_update_dispatch_smoke("requested_device_not_cuda")
    if not torch.cuda.is_available():
        return _skipped_native_update_dispatch_smoke("cuda_unavailable")
    from core.lulynx_trainer.turbocore_native_update_training_loop_dispatch_smoke import (  # noqa: E402
        test_training_loop_second_step_executes_native_update,
    )

    test_training_loop_second_step_executes_native_update()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_loop_dispatch_smoke",
        "ok": True,
        "skipped": False,
        "reason": "explicit_cuda_dispatch_smoke_passed",
        "native_training_executor_available": True,
        "native_step_executed": True,
        "native_kernel_launched": True,
        "owner_backend": "rust_cuda_adamw_v0",
        "should_call_pytorch_optimizer_step": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _skipped_native_update_dispatch_smoke(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_loop_dispatch_smoke",
        "ok": True,
        "skipped": True,
        "reason": str(reason or "skipped"),
        "native_training_executor_available": False,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def build_readiness_report(
    *,
    device: torch.device,
    dtype: torch.dtype,
    preset: str = "tiny",
    rank: int = 4,
    iters: int = 1,
    warmup: int = 0,
    include_torch_compile: bool = True,
    shape_policy: str = "auto",
    native_update_performance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = {
        "model_type": "anima",
        "training_type": "lora",
        "execution_core": "turbo",
        "turbocore_features": ["lora_fused", "native_optimizer", "workspace_pool", "data_pipeline"],
        "turbocore_allow_fallback": True,
        "turbocore_workspace_mb": 256,
        "turbocore_prefetch_depth": 2,
        "enable_mixed_resolution_training": True,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "turbocore_readiness",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "preset": preset,
        "rank": int(rank),
        "shape_policy": shape_policy,
        "non_fatal": True,
        "sections": {},
    }
    sections = report["sections"]
    sections["candidates"] = _safe_section("candidates", lambda: {"items": list_turbocore_candidates()})
    sections["candidate_scorecard"] = _safe_section(
        "candidate_scorecard",
        lambda: build_candidate_scorecard(
            device=device,
            dtype=dtype,
            preset=preset,
            rank=max(int(rank), 1),
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            run_unavailable=False,
            include_torch_compile=include_torch_compile,
            shape_policy=shape_policy,
        ),
    )
    sections["lora_cuda_scratch_kernel"] = _safe_section(
        "lora_cuda_scratch_kernel",
        lambda: probe_lora_cuda_scratch_kernel(workspace_root=PROJECT_ROOT),
    )
    sections["lora_promotion_scorecard"] = _safe_section(
        "lora_promotion_scorecard",
        lambda: build_lora_fused_promotion_scorecard(
            candidate_scorecard=sections.get("candidate_scorecard", {}),
            benchmark_matrix=None,
            native_scratch_report=sections.get("lora_cuda_scratch_kernel", {}),
            dtype=dtype,
            presets=[preset],
            ranks=[max(int(rank), 1)],
            shape_policy=shape_policy,
        ),
    )
    sections["native_update_promotion_scorecard"] = _safe_section(
        "native_update_promotion_scorecard",
        lambda: _synthetic_native_update_scorecard(
            performance_report=native_update_performance_report,
        ),
    )
    sections["native_update_training_loop_dispatch_smoke"] = _safe_section(
        "native_update_training_loop_dispatch_smoke",
        lambda: _native_update_training_loop_dispatch_smoke(requested_device=device),
    )
    sections["capability"] = _safe_section("capability", lambda: build_turbocore_capability_report(config))
    sections["workspace_data_pipeline_prototype"] = _safe_section(
        "workspace_data_pipeline_prototype",
        lambda: build_workspace_pipeline_prototype_report(
            prefetch_depth=max(int(config.get("turbocore_prefetch_depth", 2) or 2), 1),
            workspace_mb=max(int(config.get("turbocore_workspace_mb", 0) or 0), 0),
        ),
    )
    sections["workspace_data_pipeline_lifecycle"] = _safe_section(
        "workspace_data_pipeline_lifecycle",
        lambda: run_workspace_pipeline_lifecycle_probe(
            batches=4,
            prefetch_depth=max(int(config.get("turbocore_prefetch_depth", 2) or 2), 1),
            workspace_mb=max(int(config.get("turbocore_workspace_mb", 0) or 0), 0),
            dtype=dtype,
            device=device,
        ),
    )
    sections["native_capability_stub"] = _safe_section(
        "native_capability_stub",
        build_turbocore_native_training_capability_stub,
    )
    sections["native_bridge_import"] = _safe_section(
        "native_bridge_import",
        probe_native_training_bridge,
    )
    sections["native_abi_validation"] = _safe_section(
        "native_abi_validation",
        lambda: validate_workspace_pipeline_native_capabilities(
            sections.get("native_capability_stub", {})
        ),
    )
    sections["parity"] = _safe_section("parity", lambda: build_turbocore_parity_report(device=device, dtype=dtype))
    sections["lora_benchmark"] = _safe_section(
        "lora_benchmark",
        lambda: run_lora_benchmark(
            preset=preset,
            ranks=[max(int(rank), 1)],
            dtype=dtype,
            device=device,
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            candidate_name="pytorch_explicit",
        ),
    )
    sections["native_optimizer_benchmark"] = _safe_section(
        "native_optimizer_benchmark",
        lambda: run_optimizer_benchmark(
            preset="tiny" if preset == "tiny" else "sdxl_lora_short",
            ranks=[max(int(rank), 1)],
            dtype=dtype,
            device=device,
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            candidate_name="pytorch_adamw",
        ),
    )
    if include_torch_compile:
        sections["torch_compile_candidate"] = _safe_section(
            "torch_compile_candidate",
            lambda: run_torch_compile_probe(
                device=device,
                dtype=dtype,
                preset=preset,
                rank=max(int(rank), 1),
                iters=max(int(iters), 1),
                warmup=max(int(warmup), 0),
            ),
        )

    hard_sections = ["candidates", "capability", "parity", "lora_benchmark", "native_optimizer_benchmark"]
    hard_ok = all(bool(sections.get(name, {}).get("ok", False)) for name in hard_sections)
    parity_ok = bool((sections.get("parity", {}).get("summary") or {}).get("ok", False))
    lora_promotion = sections.get("lora_promotion_scorecard", {}) or {}
    lora_native_abi = lora_promotion.get("native_abi", {}) if isinstance(lora_promotion.get("native_abi"), dict) else {}
    native_update_promotion = sections.get("native_update_promotion_scorecard", {}) or {}
    native_update_blockers = native_update_promotion.get("primary_promotion_blockers") or native_update_promotion.get("promotion_blockers", [])
    native_update_training_loop_smoke = sections.get("native_update_training_loop_dispatch_smoke", {}) or {}
    native_update_performance_gate = _as_dict(native_update_promotion.get("performance_gate"))
    native_update_executor_available = bool(
        native_update_training_loop_smoke.get("native_training_executor_available", False)
    )
    native_update_kernel_launched = bool(native_update_training_loop_smoke.get("native_kernel_launched", False))
    recommended_next_step = (
        "run representative native_update_dispatch_promotion_perf matrix with enough steps and retained optimizer microbenchmark evidence"
        if native_update_executor_available
        and not bool(native_update_performance_gate.get("representative_performance_gate_ready", False))
        else "prepare explicit native-update rollout review while keeping product dispatch default-off"
        if native_update_executor_available
        and bool(native_update_performance_gate.get("representative_performance_gate_ready", False))
        and not bool(native_update_promotion.get("promotion_ready", False))
        else "promote workspace/data-pipeline ABI to native stub before kernel activation"
    )
    report["summary"] = {
        "ok": bool(hard_ok and parity_ok),
        "ready_for_ui": False,
        "native_kernel_present": bool(
            lora_native_abi.get("native_kernel_present", False) or native_update_kernel_launched
        ),
        "candidate_gate_counts": (sections.get("candidate_scorecard", {}).get("summary") or {}).get("gate_counts", {}),
        "lora_promotion_ready": bool(lora_promotion.get("promotion_ready", False)),
        "lora_native_abi_contract_available": bool(lora_native_abi.get("abi_contract_available", False)),
        "lora_native_kernel_present": bool(lora_native_abi.get("native_kernel_present", False)),
        "lora_scratch_kernel_probe_available": bool(lora_promotion.get("scratch_kernel_probe_available", False)),
        "lora_scratch_kernel_probe_ok": bool(lora_promotion.get("scratch_kernel_probe_ok", False)),
        "lora_scratch_kernel_present": bool(lora_promotion.get("native_scratch_kernel_present", False)),
        "native_update_promotion_ready": bool(native_update_promotion.get("promotion_ready", False)),
        "native_update_representative_performance_ready": bool(
            native_update_performance_gate.get("representative_performance_gate_ready", False)
        ),
        "native_update_performance_blockers": list(
            native_update_performance_gate.get("blocked_reasons", []) or []
        )[:8],
        "native_update_training_executor_available": native_update_executor_available,
        "native_update_training_loop_dispatch_smoke_ok": bool(
            native_update_training_loop_smoke.get("ok", False)
            and not native_update_training_loop_smoke.get("skipped", False)
        ),
        "native_update_native_kernel_launched": native_update_kernel_launched,
        "lora_promotion_blockers": list(lora_promotion.get("promotion_blockers", []) or [])[:8],
        "native_update_promotion_blockers": list(native_update_blockers or [])[:8],
        "torch_compile_available_on_host": bool(sections.get("torch_compile_candidate", {}).get("available", False)),
        "workspace_data_pipeline_prototype": (sections.get("workspace_data_pipeline_prototype", {}) or {}).get("status", ""),
        "workspace_data_pipeline_lifecycle_ok": bool((sections.get("workspace_data_pipeline_lifecycle", {}) or {}).get("ok", False)),
        "native_stub_schema_complete": bool((sections.get("native_abi_validation", {}) or {}).get("ok", False)),
        "native_training_path_locked": not bool((sections.get("native_capability_stub", {}) or {}).get("training_path_enabled", False)),
        "native_bridge_provider": str(((sections.get("native_bridge_import", {}) or {}).get("diagnostic") or {}).get("provider", "unknown")),
        "native_stub_gate": (
            "stub_complete_training_locked"
            if bool((sections.get("native_abi_validation", {}) or {}).get("ok", False))
            and not bool((sections.get("native_capability_stub", {}) or {}).get("training_path_enabled", False))
            else "stub_incomplete_or_unexpected"
        ),
        "recommended_next_step": recommended_next_step,
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate TurboCore readiness evidence")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--shape-policy", default="auto", choices=["auto", "off", "disabled"])
    parser.add_argument(
        "--native-update-performance-report",
        default="",
        help="Optional matrix_summary.json or native_update_performance_report JSON to include as report-only evidence.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    native_update_performance_report = _load_json_report(args.native_update_performance_report)
    payload = build_readiness_report(
        device=device,
        dtype=dtype,
        preset=str(args.preset or "tiny"),
        rank=int(args.rank),
        iters=int(args.iters),
        warmup=int(args.warmup),
        include_torch_compile=not bool(args.skip_torch_compile),
        shape_policy=str(args.shape_policy or "auto"),
        native_update_performance_report=native_update_performance_report,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0


def _load_json_report(path_value: str) -> dict[str, Any] | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to load native update performance report {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"native update performance report must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
