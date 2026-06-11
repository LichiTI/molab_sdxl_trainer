"""Native Rust/CUDA LoRA forward bridge with PyTorch autograd fallback math."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from core.services.native_module_loader import native_with_entrypoints


F32_RUNTIME_ENTRYPOINTS = (
    "create_lora_cuda_kernel_runtime_session_py",
    "lora_cuda_kernel_runtime_session_snapshot_py",
    "launch_lora_delta_f32_runtime_session_py",
    "destroy_lora_cuda_kernel_runtime_session_py",
)
F16_RUNTIME_ENTRYPOINTS = (
    "create_lora_cuda_f16_kernel_runtime_session_py",
    "lora_cuda_f16_kernel_runtime_session_snapshot_py",
    "launch_lora_delta_f16_runtime_session_py",
    "destroy_lora_cuda_f16_kernel_runtime_session_py",
)
RUNTIME_ENTRYPOINTS = F32_RUNTIME_ENTRYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_dtype_name(value: str | torch.dtype) -> str:
    return str(value).replace("torch.", "").strip().lower()


class _NativeLoraRuntime:
    def __init__(self, *, dtype: str = "float32") -> None:
        self.dtype = _normalize_dtype_name(dtype)
        self._native: Any | None = None
        self._session_id: int | None = None
        if self.dtype == "float16":
            self._entrypoints = F16_RUNTIME_ENTRYPOINTS
            self._create_name = "create_lora_cuda_f16_kernel_runtime_session_py"
            self._destroy_name = "destroy_lora_cuda_f16_kernel_runtime_session_py"
            self._launch_name = "launch_lora_delta_f16_runtime_session_py"
        else:
            self._entrypoints = F32_RUNTIME_ENTRYPOINTS
            self._create_name = "create_lora_cuda_kernel_runtime_session_py"
            self._destroy_name = "destroy_lora_cuda_kernel_runtime_session_py"
            self._launch_name = "launch_lora_delta_f32_runtime_session_py"

    def native(self) -> Any:
        native = self._native or native_with_entrypoints(*self._entrypoints)
        if native is None:
            raise RuntimeError(f"lulynx_native LoRA {self.dtype} runtime entrypoints are unavailable")
        self._native = native
        return native

    def session_id(self) -> int:
        if self._session_id is not None:
            return self._session_id
        native = self.native()
        created = dict(getattr(native, self._create_name)(str(PROJECT_ROOT), None))
        if not bool(created.get("ok", False)):
            raise RuntimeError(f"native LoRA {self.dtype} runtime session creation failed: {created}")
        self._session_id = int(created["runtime_session_id"])
        return self._session_id

    def close(self) -> dict[str, Any]:
        if self._session_id is None:
            return {"ok": True, "destroyed": False}
        native = self.native()
        session_id = self._session_id
        self._session_id = None
        return dict(getattr(native, self._destroy_name)(session_id))

    def launch(
        self,
        x_2d: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        base_2d: torch.Tensor,
        output_2d: torch.Tensor,
        *,
        scale: float,
        training_dispatch: bool,
        training_path_enabled: bool,
    ) -> dict[str, Any]:
        batch_tokens = int(x_2d.shape[0])
        in_features = int(x_2d.shape[1])
        rank = int(down_weight.shape[0])
        out_features = int(up_weight.shape[0])
        config = {
            "batch_tokens": batch_tokens,
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "scale": float(scale),
            "block_size": 128,
            "training_dispatch": bool(training_dispatch),
            "training_path_enabled": bool(training_path_enabled),
            "dtype": self.dtype,
        }
        report = dict(
            getattr(self.native(), self._launch_name)(
                self.session_id(),
                x_2d,
                down_weight,
                up_weight,
                base_2d,
                output_2d,
                json.dumps(config),
            )
        )
        if not bool(report.get("ok", False)):
            raise RuntimeError(f"native LoRA {self.dtype} runtime launch failed: {report}")
        return report


_RUNTIMES = {
    "float32": _NativeLoraRuntime(dtype="float32"),
    "float16": _NativeLoraRuntime(dtype="float16"),
}
_LAST_FORWARD_REPORT: dict[str, Any] = {}


def close_native_lora_runtime() -> dict[str, Any]:
    reports = {dtype: runtime.close() for dtype, runtime in _RUNTIMES.items()}
    return {"ok": all(bool(item.get("ok", False)) for item in reports.values()), "reports": reports}


def last_native_lora_forward_report() -> dict[str, Any]:
    return dict(_LAST_FORWARD_REPORT)


def native_lora_delta_forward(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
    *,
    training_dispatch: bool = True,
    training_path_enabled: bool = True,
) -> torch.Tensor:
    _validate_native_inputs(x, down_weight, up_weight, base_output)
    dtype_name = _normalize_dtype_name(x.dtype)
    original_shape = tuple(base_output.shape)
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    x_2d = x.reshape(-1, in_features).contiguous()
    down = down_weight.contiguous()
    up = up_weight.contiguous()
    base_2d = base_output.reshape(-1, out_features).contiguous()
    output_2d = torch.empty_like(base_2d)
    report = _RUNTIMES[dtype_name].launch(
        x_2d,
        down,
        up,
        base_2d,
        output_2d,
        scale=float(scale),
        training_dispatch=training_dispatch,
        training_path_enabled=training_path_enabled,
    )
    global _LAST_FORWARD_REPORT
    _LAST_FORWARD_REPORT = report
    return output_2d.reshape(original_shape)


class _NativeLoraDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        base_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        out = native_lora_delta_forward(
            x,
            down_weight,
            up_weight,
            base_output,
            float(scale),
            training_dispatch=True,
            training_path_enabled=True,
        )
        ctx.save_for_backward(x, down_weight, up_weight)
        ctx.scale = float(scale)
        ctx.x_shape = tuple(x.shape)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        x, down_weight, up_weight = ctx.saved_tensors
        scale = float(ctx.scale)
        in_features = int(x.shape[-1])
        out_features = int(up_weight.shape[0])
        grad_2d = grad_output.reshape(-1, out_features).contiguous().float()
        x_2d = x.reshape(-1, in_features).contiguous().float()
        down_f = down_weight.float()
        up_f = up_weight.float()
        hidden = F.linear(x_2d, down_f)
        grad_hidden = grad_2d.matmul(up_f).mul(scale)
        grad_x = grad_hidden.matmul(down_f).reshape(ctx.x_shape).to(dtype=x.dtype)
        grad_down = grad_hidden.transpose(0, 1).matmul(x_2d).to(dtype=down_weight.dtype)
        grad_up = grad_2d.transpose(0, 1).matmul(hidden).mul(scale).to(dtype=up_weight.dtype)
        grad_base = grad_output
        return grad_x, grad_down, grad_up, grad_base, None


def native_lora_delta_autograd(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _NativeLoraDeltaFunction.apply(x, down_weight, up_weight, base_output, float(scale))


def probe_lora_native_training_dispatch(
    *,
    x_shape: Sequence[int] = (2, 64, 320),
    rank: int = 4,
    out_features: int | None = None,
    device: torch.device | None = None,
    dtype: str | torch.dtype = "float32",
) -> dict[str, Any]:
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = _normalize_dtype_name(dtype)
    if target_device.type != "cuda":
        return _blocked_training_report("cuda_required_for_native_lora_training_dispatch", dtype=dtype_name)
    if dtype_name not in {"float32", "float16"}:
        return _blocked_training_report(f"unsupported_lora_native_training_dtype:{dtype_name}", dtype=dtype_name)
    try:
        torch.manual_seed(1337)
        shape = tuple(int(dim) for dim in x_shape)
        in_features = int(shape[-1])
        resolved_out = int(out_features or in_features)
        resolved_rank = int(rank)
        scale = 1.0 / max(resolved_rank, 1)
        torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
        native_inputs = _make_case(shape, resolved_rank, resolved_out, target_device, torch_dtype)
        ref_inputs = [item.detach().clone().requires_grad_(True) for item in native_inputs]
        native_out = native_lora_delta_autograd(*native_inputs, scale)
        ref_hidden = F.linear(ref_inputs[0].float(), ref_inputs[1].float())
        ref_delta = F.linear(ref_hidden, ref_inputs[2].float()) * scale
        ref_out = (ref_inputs[3].float() + ref_delta).to(dtype=torch_dtype)
        forward_diff = float((native_out.float() - ref_out.float()).abs().max().detach().cpu())
        native_loss = native_out.square().mean()
        ref_loss = ref_out.square().mean()
        native_loss.backward()
        ref_loss.backward()
        grad_diffs = [
            float((left.grad.float() - right.grad.float()).abs().max().detach().cpu())
            for left, right in zip(native_inputs, ref_inputs)
            if left.grad is not None and right.grad is not None
        ]
        forward_report = last_native_lora_forward_report()
        tolerance = 2e-4 if dtype_name == "float32" else 5e-2
        backward_ok = bool(grad_diffs and max(grad_diffs) <= tolerance)
        forward_ok = forward_diff <= tolerance
        return {
            "schema_version": 1,
            "report": "turbocore_lora_native_training_dispatch_probe_v0",
            "ok": bool(forward_ok and backward_ok and forward_report.get("ok", False)),
            "candidate": "rust_cuda_lora_delta_v0",
            "native_kernel_present": bool(forward_report.get("native_kernel_present", False)),
            "kernel_executed": bool(forward_report.get("kernel_executed", False)),
            "kernel_launch_count": int(forward_report.get("kernel_launch_count", 0) or 0),
            "output_mutated": bool(forward_report.get("output_mutated", False)),
            "training_tensor_binding": bool(forward_report.get("training_tensor_binding", False)),
            "training_dispatch": bool(forward_report.get("training_dispatch", False)),
            "training_path_enabled": bool(forward_report.get("training_path_enabled", False)),
            "autograd_binding": True,
            "forward_backward_training_integration": True,
            "forward_parity_ok": forward_ok,
            "backward_parity_ok": backward_ok,
            "max_abs_forward_diff": forward_diff,
            "max_abs_grad_diff": max(grad_diffs) if grad_diffs else None,
            "stream_lifetime_bound": bool(forward_report.get("stream_lifetime_bound", False)),
            "runtime_recovery_ready": bool(forward_report.get("runtime_recovery_ready", False)),
            "fallback_to_pytorch_lora": False,
            "pytorch_lora_path_authoritative": False,
            "dtype": dtype_name,
            "tolerance": tolerance,
            "device": str(target_device),
            "x_shape": list(shape),
            "rank": resolved_rank,
            "out_features": resolved_out,
            "forward_report": forward_report,
            "blocked_reasons": [],
        }
    except Exception as exc:
        return _blocked_training_report(
            f"native_lora_training_dispatch_probe_failed:{type(exc).__name__}: {exc}",
            dtype=dtype_name,
        )
    finally:
        close_native_lora_runtime()


def _make_case(
    shape: tuple[int, ...],
    rank: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    in_features = int(shape[-1])
    batch_tokens = 1
    for dim in shape[:-1]:
        batch_tokens *= int(dim)
    scale = 0.25 if dtype is torch.float16 else 1.0
    x = (torch.randn(*shape, dtype=torch.float32, device=device) * scale).to(dtype).requires_grad_(True)
    down = (torch.randn(rank, in_features, dtype=torch.float32, device=device) * scale).to(dtype).requires_grad_(True)
    up = (torch.randn(out_features, rank, dtype=torch.float32, device=device) * scale).to(dtype).requires_grad_(True)
    base = (
        torch.randn(batch_tokens, out_features, dtype=torch.float32, device=device) * scale
    ).to(dtype).reshape(*shape[:-1], out_features)
    base.requires_grad_(True)
    return [x, down, up, base]


def _validate_native_inputs(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
) -> None:
    tensors = [x, down_weight, up_weight, base_output]
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("native LoRA runtime requires CUDA tensors")
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) != 1 or next(iter(dtypes)) not in {torch.float32, torch.float16}:
        raise ValueError("native LoRA runtime currently supports float32 or float16 tensors")
    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("x and base_output must have at least 2 dimensions")
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    if tuple(x.shape[:-1]) != tuple(base_output.shape[:-1]):
        raise ValueError("x and base_output batch dimensions must match")
    if tuple(down_weight.shape) != (rank, in_features):
        raise ValueError("down_weight must have shape [rank, in_features]")
    if tuple(up_weight.shape) != (out_features, rank):
        raise ValueError("up_weight must have shape [out_features, rank]")


def _blocked_training_report(reason: str, *, dtype: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_training_dispatch_probe_v0",
        "ok": False,
        "candidate": "rust_cuda_lora_delta_v0",
        "native_kernel_present": False,
        "kernel_executed": False,
        "output_mutated": False,
        "training_tensor_binding": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "autograd_binding": False,
        "forward_backward_training_integration": False,
        "forward_parity_ok": False,
        "backward_parity_ok": False,
        "stream_lifetime_bound": False,
        "runtime_recovery_ready": False,
        "fallback_to_pytorch_lora": True,
        "pytorch_lora_path_authoritative": True,
        "dtype": dtype,
        "blocked_reasons": [reason],
    }


__all__ = [
    "F16_RUNTIME_ENTRYPOINTS",
    "F32_RUNTIME_ENTRYPOINTS",
    "RUNTIME_ENTRYPOINTS",
    "close_native_lora_runtime",
    "last_native_lora_forward_report",
    "native_lora_delta_autograd",
    "native_lora_delta_forward",
    "probe_lora_native_training_dispatch",
]
