"""Candidate registry for TurboCore benchmark and parity probes.

Native TurboCore kernels do not exist yet, but the benchmark/parity harnesses
should not grow one-off integration code for every future experiment.  This
registry gives each feature a small stable callable shape and ships only safe
PyTorch candidates by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F


LoraDeltaCandidate = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
NativeOptimizerCandidate = Callable[[list[torch.nn.Parameter], float, float, float], None]


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class TurboCoreCandidate:
    name: str
    feature: str
    callable: Callable[..., Any]
    native: bool = False
    experimental: bool = False
    description: str = ""
    available: bool = True
    reason: str = ""
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature": self.feature,
            "native": bool(self.native),
            "experimental": bool(self.experimental),
            "description": self.description,
            "available": bool(self.available),
            "reason": self.reason,
            "notes": list(self.notes),
        }


def pytorch_lora_delta_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return base_output + F.linear(F.linear(x, down_weight), up_weight) * float(scale)


def pytorch_native_optimizer_candidate(
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
) -> None:
    if float(max_grad_norm or 0.0) > 0:
        torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    optimizer.step()


_COMPILED_LORA_DELTA: LoraDeltaCandidate | None = None


def torch_compile_lora_delta_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """torch.compile candidate for LoRA delta/add experiments.

    This is intentionally a benchmark/parity candidate only.  If compilation
    fails for a local PyTorch/backend combination, the caller sees the error in
    the benchmark instead of training silently changing behavior.
    """

    global _COMPILED_LORA_DELTA
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    if _COMPILED_LORA_DELTA is None:
        _COMPILED_LORA_DELTA = torch.compile(
            pytorch_lora_delta_candidate,
            dynamic=False,
            fullgraph=False,
        )
    return _COMPILED_LORA_DELTA(x, down_weight, up_weight, base_output, scale)


def unavailable_native_candidate(*_args: Any, **_kwargs: Any) -> Any:
    """Placeholder callable for registered future native candidates."""

    raise RuntimeError("TurboCore native candidate is registered for discovery only and is not available")


_TRITON_AVAILABLE = _module_available("triton")


def _load_triton_lora_delta_candidates() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any], bool, str]:
    try:
        from core.turbocore_triton_lora import (  # type: ignore
            triton_lora_delta_available,
            triton_lora_delta_candidate,
            triton_lora_delta_v1_candidate,
            triton_lora_delta_v2_candidate,
            triton_lora_delta_v2_tc_candidate,
            triton_lora_delta_v3_dispatch_candidate,
            triton_lora_delta_unavailable_reason,
        )

        return (
            triton_lora_delta_candidate,
            triton_lora_delta_v1_candidate,
            triton_lora_delta_v2_candidate,
            triton_lora_delta_v2_tc_candidate,
            triton_lora_delta_v3_dispatch_candidate,
            bool(triton_lora_delta_available()),
            str(triton_lora_delta_unavailable_reason()),
        )
    except Exception as exc:
        reason = "triton_available_probe_only" if _TRITON_AVAILABLE else "triton_unavailable"
        return (
            unavailable_native_candidate,
            unavailable_native_candidate,
            unavailable_native_candidate,
            unavailable_native_candidate,
            unavailable_native_candidate,
            False,
            f"{reason}: {type(exc).__name__}: {exc}",
        )


(
    _TRITON_LORA_V0_CALLABLE,
    _TRITON_LORA_V1_CALLABLE,
    _TRITON_LORA_V2_CALLABLE,
    _TRITON_LORA_V2_TC_CALLABLE,
    _TRITON_LORA_V3_CALLABLE,
    _TRITON_LORA_AVAILABLE,
    _TRITON_LORA_REASON,
) = _load_triton_lora_delta_candidates()


def _load_triton_optimizer_probe_state() -> tuple[bool, str]:
    try:
        from core.turbocore_triton_optimizer import (  # type: ignore
            triton_adamw_flat_available,
            triton_adamw_flat_unavailable_reason,
        )

        return bool(triton_adamw_flat_available()), str(triton_adamw_flat_unavailable_reason())
    except Exception as exc:
        reason = "triton_available_probe_only" if _TRITON_AVAILABLE else "triton_unavailable"
        return False, f"{reason}: {type(exc).__name__}: {exc}"


_TRITON_OPTIMIZER_AVAILABLE, _TRITON_OPTIMIZER_REASON = _load_triton_optimizer_probe_state()


_CANDIDATES: Dict[str, Dict[str, TurboCoreCandidate]] = {
    "lora_fused": {
        "pytorch_explicit": TurboCoreCandidate(
            name="pytorch_explicit",
            feature="lora_fused",
            callable=pytorch_lora_delta_candidate,
            native=False,
            experimental=False,
            description="PyTorch explicit LoRA delta/add reference candidate.",
        ),
        "torch_compile": TurboCoreCandidate(
            name="torch_compile",
            feature="lora_fused",
            callable=torch_compile_lora_delta_candidate,
            native=False,
            experimental=True,
            description="torch.compile candidate for LoRA delta/add benchmark and parity experiments.",
            available=hasattr(torch, "compile"),
            reason="torch_compile_available" if hasattr(torch, "compile") else "torch_compile_unavailable",
        ),
        "triton_lora_delta_v0": TurboCoreCandidate(
            name="triton_lora_delta_v0",
            feature="lora_fused",
            callable=_TRITON_LORA_V0_CALLABLE,
            native=False,
            experimental=True,
            description="Triton LoRA delta/add v0 candidate: PyTorch down projection plus Triton up projection/base add.",
            available=_TRITON_LORA_AVAILABLE,
            reason=_TRITON_LORA_REASON,
        ),
        "triton_lora_delta_v1": TurboCoreCandidate(
            name="triton_lora_delta_v1",
            feature="lora_fused",
            callable=_TRITON_LORA_V1_CALLABLE,
            native=False,
            experimental=True,
            description="Triton LoRA delta/add v1 candidate: full fused x/down/up/base-add for rank <= 32.",
            available=_TRITON_LORA_AVAILABLE,
            reason=_TRITON_LORA_REASON,
            notes=(
                "research_only_not_training_dispatcher",
                "allow_width_le_768_with_repeated_benchmark_evidence",
                "skip_width_ge_1024_by_default_after_negative_matrix_evidence",
                "skip_dit_presets_by_default_after_negative_matrix_evidence",
                "rank_gt_32_unsupported",
            ),
        ),
        "triton_lora_delta_v2": TurboCoreCandidate(
            name="triton_lora_delta_v2",
            feature="lora_fused",
            callable=_TRITON_LORA_V2_CALLABLE,
            native=False,
            experimental=True,
            description="Triton LoRA delta/add v2 candidate: large-width tuned Triton up/add with PyTorch down projection.",
            available=_TRITON_LORA_AVAILABLE,
            reason=_TRITON_LORA_REASON,
            notes=(
                "research_only_not_training_dispatcher",
                "focus_width_ge_1024_and_dit_shapes",
                "rank_gt_32_unsupported",
                "requires_repeated_parity_and_benchmark_gates",
            ),
        ),
        "triton_lora_delta_v2_tc": TurboCoreCandidate(
            name="triton_lora_delta_v2_tc",
            feature="lora_fused",
            callable=_TRITON_LORA_V2_TC_CALLABLE,
            native=False,
            experimental=True,
            description="Triton LoRA delta/add v2.2 candidate: fp16/bf16 TensorCore-friendly up/add research path.",
            available=_TRITON_LORA_AVAILABLE,
            reason=_TRITON_LORA_REASON,
            notes=(
                "research_only_not_training_dispatcher",
                "targets_fp16_bf16_tensorcore_probe",
                "focus_width_ge_1024_and_dit_shapes",
                "rank_gt_32_unsupported",
                "requires_repeated_parity_and_benchmark_gates",
            ),
        ),
        "triton_lora_delta_v3_dispatch": TurboCoreCandidate(
            name="triton_lora_delta_v3_dispatch",
            feature="lora_fused",
            callable=_TRITON_LORA_V3_CALLABLE,
            native=False,
            experimental=True,
            description="Triton LoRA delta/add v3 research dispatcher over v1, v2_tc, and PyTorch fallback.",
            available=_TRITON_LORA_AVAILABLE,
            reason=_TRITON_LORA_REASON,
            notes=(
                "research_only_not_training_dispatcher",
                "shape_rank_dtype_dispatcher",
                "uses_pytorch_fallback_for_negative_research_shapes",
                "requires_route_level_repeated_validation",
            ),
        ),
        "rust_cuda_lora_delta_v0": TurboCoreCandidate(
            name="rust_cuda_lora_delta_v0",
            feature="lora_fused",
            callable=unavailable_native_candidate,
            native=True,
            experimental=True,
            description="Reserved Rust/CUDA LoRA fused delta candidate; unavailable until native bridge exposes it.",
            available=False,
            reason="native_training_bridge_not_implemented",
        ),
    },
    "native_optimizer": {
        "pytorch_adamw": TurboCoreCandidate(
            name="pytorch_adamw",
            feature="native_optimizer",
            callable=pytorch_native_optimizer_candidate,
            native=False,
            experimental=False,
            description="PyTorch AdamW step candidate for LoRA-sized parameter groups.",
        ),
        "rust_cuda_adamw_v0": TurboCoreCandidate(
            name="rust_cuda_adamw_v0",
            feature="native_optimizer",
            callable=unavailable_native_candidate,
            native=True,
            experimental=True,
            description="Reserved Rust/CUDA LoRA-sized AdamW/update candidate; unavailable until native bridge exposes it.",
            available=False,
            reason="native_training_bridge_not_implemented",
        ),
        "triton_adamw_flat_v0": TurboCoreCandidate(
            name="triton_adamw_flat_v0",
            feature="native_optimizer",
            callable=unavailable_native_candidate,
            native=True,
            experimental=True,
            description="Triton flat fp32 AdamW update research kernel; benchmark-only until the optimizer ABI has flat persistent buffers.",
            available=False,
            reason="flat_state_contract_not_wired_to_candidate_registry" if _TRITON_OPTIMIZER_AVAILABLE else _TRITON_OPTIMIZER_REASON,
            notes=(
                "research_only_not_training_dispatcher",
                "flat_contiguous_fp32_buffers_only",
                "use_turbocore_triton_adamw_flat_benchmark_for_performance",
                "old_list_param_optimizer_candidate_contract_not_supported",
            ),
        ),
        "python_update_executor_v0": TurboCoreCandidate(
            name="python_update_executor_v0",
            feature="native_optimizer",
            callable=unavailable_native_candidate,
            native=False,
            experimental=True,
            description="Python prototype for the TurboCore update executor: persistent flat AdamW owner plus optional direct-gradient hooks.",
            available=False,
            reason="executor_contract_probe_only_not_candidate_callable",
            notes=(
                "research_only_not_training_dispatcher",
                "persistent_flat_owner",
                "direct_grad_to_flat_owner_probe",
                "copy_back_required_until_training_owns_flat_params",
                "training_path_enabled_false",
            ),
        ),
    },
}


def list_turbocore_candidates(feature: str | None = None) -> dict[str, list[dict[str, Any]]]:
    if feature:
        key = str(feature).strip().lower()
        return {key: [candidate.as_dict() for candidate in _CANDIDATES.get(key, {}).values()]}
    return {
        key: [candidate.as_dict() for candidate in values.values()]
        for key, values in sorted(_CANDIDATES.items())
    }


def get_turbocore_candidate(feature: str, name: str | None = None) -> TurboCoreCandidate | None:
    feature_key = str(feature or "").strip().lower()
    candidates = _CANDIDATES.get(feature_key, {})
    if not candidates:
        return None
    if name:
        candidate = candidates.get(str(name).strip().lower())
        if candidate is not None and not candidate.available:
            return None
        return candidate
    for candidate in candidates.values():
        if candidate.available:
            return candidate
    return None


def register_turbocore_candidate(candidate: TurboCoreCandidate) -> None:
    """Register a process-local candidate for experiments and tests."""

    feature = str(candidate.feature or "").strip().lower()
    name = str(candidate.name or "").strip().lower()
    if not feature or not name:
        raise ValueError("TurboCore candidate requires non-empty feature and name")
    _CANDIDATES.setdefault(feature, {})[name] = candidate


def candidate_names(feature: str) -> list[str]:
    return sorted(_CANDIDATES.get(str(feature or "").strip().lower(), {}))


__all__ = [
    "LoraDeltaCandidate",
    "NativeOptimizerCandidate",
    "TurboCoreCandidate",
    "candidate_names",
    "get_turbocore_candidate",
    "list_turbocore_candidates",
    "pytorch_lora_delta_candidate",
    "pytorch_native_optimizer_candidate",
    "register_turbocore_candidate",
    "torch_compile_lora_delta_candidate",
    "unavailable_native_candidate",
]
