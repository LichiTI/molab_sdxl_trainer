"""V2 scorecard for native AdamW multi-tensor update evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.turbocore_native_update_training_executor import build_native_update_training_executor


DEFAULT_DTYPES = ("float32", "float16")


def build_native_update_multitensor_scorecard(
    *,
    first_round_report: Mapping[str, Any] | None = None,
    dtype_cases: Sequence[str] = DEFAULT_DTYPES,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Build a V2 promotion scorecard for multi-tensor native optimizer update.

    The scorecard runs a tiny real native-update dispatch when CUDA is present.
    It uses multiple trainable tensors in one flat owner and verifies that the
    native executor mutates training parameters while PyTorch optimizer.step()
    remains skipped.
    """

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_round = dict(first_round_report or {})
    dtype_reports = {
        _normalize_dtype(dtype): _run_dtype_case(_normalize_dtype(dtype), target_device)
        for dtype in dtype_cases
    }
    ready_reports = [report for report in dtype_reports.values() if _dtype_case_ready(report)]
    first_round_ready = bool(first_round.get("promotion_ready", False))
    representative_perf_ready = bool(
        _as_dict(first_round.get("performance_gate")).get("representative_performance_gate_ready", False)
    )
    blockers: list[str] = []
    if target_device.type != "cuda":
        blockers.append("cuda_required_for_optimizer_multitensor_update")
    if not first_round_ready:
        blockers.append("native_optimizer_update_first_round_not_ready")
    if not representative_perf_ready:
        blockers.append("native_optimizer_representative_performance_missing")
    if not ready_reports:
        blockers.append("optimizer_multitensor_native_step_missing")
    if max((int(report.get("tensor_count", 0) or 0) for report in ready_reports), default=0) < 2:
        blockers.append("optimizer_multitensor_tensor_count_too_small")
    if len({str(report.get("dtype", "")) for report in ready_reports}) < 2:
        blockers.append("optimizer_mixed_precision_dtype_coverage_missing")
    ready = not blockers
    max_tensor_count = max((int(report.get("tensor_count", 0) or 0) for report in ready_reports), default=0)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_update_multitensor_scorecard_v0",
        "gate": "optimizer_multitensor_update",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": ready,
        "native_dispatch_allowed": ready,
        "native_step_executed": bool(ready_reports),
        "should_call_pytorch_optimizer_step": not bool(ready_reports),
        "bucketed_launch_plan": bool(max_tensor_count >= 2 and ready_reports),
        "param_group_count": max((int(report.get("param_group_count", 0) or 0) for report in ready_reports), default=0),
        "tensor_count": max_tensor_count,
        "dtype_bucket_count": len({str(report.get("dtype", "")) for report in ready_reports}),
        "ready_dtype_count": len(ready_reports),
        "required_dtypes": list(dtype_cases),
        "dtype_reports": dtype_reports,
        "first_round_native_update": {
            "promotion_ready": first_round_ready,
            "native_step_executed": bool(first_round.get("native_step_executed", False)),
            "training_path_enabled": bool(first_round.get("training_path_enabled", False)),
            "representative_performance_gate_ready": representative_perf_ready,
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _run_dtype_case(dtype_name: str, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return _blocked(dtype_name, "cuda_required_for_native_update_multitensor_case")
    torch_dtype = _torch_dtype(dtype_name)
    if torch_dtype is None:
        return _blocked(dtype_name, f"unsupported_optimizer_multitensor_dtype:{dtype_name}")
    try:
        torch.manual_seed(2027)
        params = [
            torch.nn.Parameter((torch.randn(8, 4, device=device) * 0.1).to(torch_dtype)),
            torch.nn.Parameter((torch.randn(3, 5, device=device) * 0.1).to(torch_dtype)),
        ]
        for param in params:
            param.grad = (torch.randn_like(param.float()) * 0.01).to(torch_dtype)
        before = [param.detach().clone().float() for param in params]
        optimizer = torch.optim.AdamW([{"params": params, "lr": 1e-3, "weight_decay": 0.01}])
        executor = build_native_update_training_executor(
            optimizer=optimizer,
            params=params,
            config={
                "lr": 1e-3,
                "weight_decay": 0.01,
                "max_grad_norm": 0.0,
                "finite_check": True,
                "prefer_native_cuda": True,
                "require_native_cuda": True,
                "prefer_triton": False,
                "block_size": 256,
            },
        )
        try:
            report = executor({"training_dispatch": True, "training_path_enabled": True})
        finally:
            executor.close()
        after = [param.detach().clone().float() for param in params]
        max_param_delta = max(float((new - old).abs().max().detach().cpu()) for old, new in zip(before, after))
        native_step = bool(report.get("native_step_executed", False))
        skipped_pytorch = not bool(report.get("should_call_pytorch_optimizer_step", True))
        training_mutated = bool(report.get("training_parameters_mutated", False)) and max_param_delta > 0.0
        blocked = []
        if not native_step:
            blocked.append("optimizer_multitensor_native_step_not_executed")
        if not skipped_pytorch:
            blocked.append("optimizer_multitensor_pytorch_step_not_skipped")
        if not training_mutated:
            blocked.append("optimizer_multitensor_parameters_not_mutated")
        if not bool(report.get("native_kernel_launched", False)):
            blocked.append("optimizer_multitensor_native_kernel_not_launched")
        update_report = _as_dict(report.get("update_report"))
        if str(update_report.get("owner_backend", "") or "") != "rust_cuda_adamw_v0":
            blocked.append("optimizer_multitensor_native_backend_not_rust_cuda_adamw")
        return {
            "schema_version": 1,
            "ok": not blocked,
            "dtype": dtype_name,
            "device": str(device),
            "param_group_count": len(optimizer.param_groups),
            "tensor_count": len(params),
            "flat_numel": sum(int(param.numel()) for param in params),
            "bucketed_launch_plan": len(params) >= 2,
            "dtype_bucket_count": 1,
            "runtime_native_step": native_step,
            "native_step_executed": native_step,
            "native_kernel_launched": bool(report.get("native_kernel_launched", False)),
            "owner_backend": str(update_report.get("owner_backend", "") or ""),
            "training_parameters_mutated": training_mutated,
            "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
            "max_param_delta": max_param_delta,
            "training_path_enabled": bool(report.get("training_path_enabled", False)),
            "executor_report": report,
            "blocked_reasons": _dedupe(blocked),
        }
    except Exception as exc:
        return _blocked(dtype_name, f"optimizer_multitensor_case_failed:{type(exc).__name__}: {exc}")


def _dtype_case_ready(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("ok", False)
        and report.get("native_step_executed", False)
        and report.get("native_kernel_launched", False)
        and report.get("owner_backend") == "rust_cuda_adamw_v0"
        and report.get("training_parameters_mutated", False)
        and not report.get("should_call_pytorch_optimizer_step", True)
        and int(report.get("tensor_count", 0) or 0) >= 2
    )


def _torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return None


def _normalize_dtype(value: str) -> str:
    normalized = str(value or "").replace("torch.", "").strip().lower()
    return {"fp16": "float16", "half": "float16", "fp32": "float32", "bf16": "bfloat16"}.get(
        normalized,
        normalized,
    )


def _blocked(dtype_name: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "dtype": dtype_name,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "training_path_enabled": False,
        "blocked_reasons": [reason],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_update_multitensor_scorecard"]
