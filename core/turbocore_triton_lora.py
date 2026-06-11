"""Triton research candidates for TurboCore LoRA delta/add.

The v0 candidate fuses only the small-rank up projection and base add.  The v1
candidate is the first full-fused research path for ``x @ down.T @ up.T + base``
on common small LoRA ranks.  Both are scorecard-only and never activate the
training path by themselves.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


try:  # pragma: no cover - import availability is host-specific
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def triton_lora_delta_available() -> bool:
    return triton is not None and tl is not None and bool(torch.cuda.is_available())


def triton_lora_delta_unavailable_reason() -> str:
    if triton is None or tl is None:
        return "triton_unavailable"
    if not bool(torch.cuda.is_available()):
        return "cuda_unavailable"
    return "available"


V2_LAUNCH_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "dit_3072_wide",
        "min_out_features": 2048,
        "block_m": 16,
        "block_n": 128,
        "num_warps": 8,
        "num_stages": 4,
    },
    {
        "name": "dit_1536_midwide",
        "min_out_features": 1408,
        "block_m": 16,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "name": "sdxl_1280_midwide",
        "min_out_features": 1024,
        "block_m": 16,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "name": "fallback_small_probe",
        "min_out_features": 0,
        "block_m": 16,
        "block_n": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
)


V2_TC_SWEEP_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "tc_m16_n32_w4_s2", "block_m": 16, "block_n": 32, "num_warps": 4, "num_stages": 2},
    {"name": "tc_m16_n64_w4_s3", "block_m": 16, "block_n": 64, "num_warps": 4, "num_stages": 3},
    {"name": "tc_m16_n128_w8_s4", "block_m": 16, "block_n": 128, "num_warps": 8, "num_stages": 4},
    {"name": "tc_m32_n64_w4_s3", "block_m": 32, "block_n": 64, "num_warps": 4, "num_stages": 3},
    {"name": "tc_m32_n128_w8_s4", "block_m": 32, "block_n": 128, "num_warps": 8, "num_stages": 4},
)


def _normalize_v2_launch_config(
    config: dict[str, Any],
    *,
    out_features: int,
    rank: int,
) -> dict[str, Any]:
    block_r = int(config.get("block_r") or (16 if int(rank) <= 16 else 32))
    return {
        "name": str(config.get("name") or "custom"),
        "out_features": int(out_features),
        "rank": int(rank),
        "block_m": int(config.get("block_m") or 16),
        "block_n": int(config.get("block_n") or 64),
        "block_r": block_r,
        "num_warps": int(config.get("num_warps") or 4),
        "num_stages": int(config.get("num_stages") or 3),
    }


def triton_lora_delta_v2_config_for_shape(
    *,
    out_features: int,
    rank: int,
) -> dict[str, Any]:
    config = next(
        item for item in V2_LAUNCH_CONFIGS
        if int(out_features) >= int(item["min_out_features"])
    )
    return _normalize_v2_launch_config(config, out_features=out_features, rank=rank)


def triton_lora_delta_v2_tc_config_candidates_for_shape(
    *,
    out_features: int,
    rank: int,
) -> list[dict[str, Any]]:
    auto = triton_lora_delta_v2_config_for_shape(out_features=out_features, rank=rank)
    candidates = [auto]
    seen = {auto["name"]}
    for config in V2_TC_SWEEP_CONFIGS:
        normalized = _normalize_v2_launch_config(config, out_features=out_features, rank=rank)
        if normalized["name"] in seen:
            continue
        candidates.append(normalized)
        seen.add(normalized["name"])
    return candidates


V3_TC_ROUTE_CONFIGS: dict[tuple[int, int], str] = {}


V3_TC_DISABLED_ROUTE_CONFIGS: dict[tuple[int, int], str] = {
    (1152, 8): "tc_m16_n64_w4_s3",
    (1152, 16): "tc_m32_n128_w8_s4",
    (1280, 4): "tc_m16_n32_w4_s2",
    (1280, 16): "tc_m16_n64_w4_s3",
    (1536, 4): "tc_m16_n64_w4_s3",
    (1536, 8): "tc_m16_n128_w8_s4",
    (1536, 16): "tc_m32_n64_w4_s3",
    (3072, 8): "dit_3072_wide",
    (3072, 16): "tc_m16_n128_w8_s4",
}


def _v2_tc_config_by_name(
    name: str,
    *,
    out_features: int,
    rank: int,
) -> dict[str, Any] | None:
    for config in triton_lora_delta_v2_tc_config_candidates_for_shape(out_features=out_features, rank=rank):
        if str(config.get("name")) == str(name):
            return config
    return None


def triton_lora_delta_v3_decision_for_shape(
    *,
    dtype: torch.dtype,
    out_features: int,
    rank: int,
) -> dict[str, Any]:
    if int(rank) > 32:
        return {
            "path": "pytorch_explicit",
            "reason": "rank_gt_32_fallback",
            "config": None,
        }
    if int(out_features) <= 768:
        return {
            "path": "triton_lora_delta_v1",
            "reason": "small_width_v1_positive_matrix",
            "config": None,
        }
    if dtype in {torch.float16, torch.bfloat16}:
        config_name = V3_TC_ROUTE_CONFIGS.get((int(out_features), int(rank)))
        if config_name:
            config = _v2_tc_config_by_name(config_name, out_features=out_features, rank=rank)
            if config is not None:
                return {
                    "path": "triton_lora_delta_v2_tc",
                    "reason": "v2_tc_sweep_selected_config",
                    "config": config,
                }
        if (int(out_features), int(rank)) in V3_TC_DISABLED_ROUTE_CONFIGS:
            return {
                "path": "pytorch_explicit",
                "reason": "v2_tc_route_disabled_by_paired_benchmark",
                "config": None,
            }
    return {
        "path": "pytorch_explicit",
        "reason": "no_positive_research_route_fallback",
        "config": None,
    }


if triton is not None and tl is not None:  # pragma: no branch - definition guard

    @triton.jit
    def _lora_up_add_kernel(
        hidden_ptr,
        up_ptr,
        base_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        out_features: tl.constexpr,
        rank: tl.constexpr,
        scale: tl.constexpr,
        stride_hidden_n: tl.constexpr,
        stride_hidden_r: tl.constexpr,
        stride_up_o: tl.constexpr,
        stride_up_r: tl.constexpr,
        stride_base_n: tl.constexpr,
        stride_base_o: tl.constexpr,
        stride_out_n: tl.constexpr,
        stride_out_o: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)

        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hidden_n + offs_r[None, :] * stride_hidden_r,
            mask=(offs_m[:, None] < n_elements) & (offs_r[None, :] < rank),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            up_ptr + offs_n[None, :] * stride_up_o + offs_r[:, None] * stride_up_r,
            mask=(offs_n[None, :] < out_features) & (offs_r[:, None] < rank),
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(hidden, up, input_precision="ieee") * scale
        base = tl.load(
            base_ptr + offs_m[:, None] * stride_base_n + offs_n[None, :] * stride_base_o,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
            other=0.0,
        )
        tl.store(
            out_ptr + offs_m[:, None] * stride_out_n + offs_n[None, :] * stride_out_o,
            base + acc,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
        )

    @triton.jit
    def _lora_up_add_tc_kernel(
        hidden_ptr,
        up_ptr,
        base_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        out_features: tl.constexpr,
        rank: tl.constexpr,
        scale: tl.constexpr,
        stride_hidden_n: tl.constexpr,
        stride_hidden_r: tl.constexpr,
        stride_up_o: tl.constexpr,
        stride_up_r: tl.constexpr,
        stride_base_n: tl.constexpr,
        stride_base_o: tl.constexpr,
        stride_out_n: tl.constexpr,
        stride_out_o: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)

        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hidden_n + offs_r[None, :] * stride_hidden_r,
            mask=(offs_m[:, None] < n_elements) & (offs_r[None, :] < rank),
            other=0.0,
        )
        up = tl.load(
            up_ptr + offs_n[None, :] * stride_up_o + offs_r[:, None] * stride_up_r,
            mask=(offs_n[None, :] < out_features) & (offs_r[:, None] < rank),
            other=0.0,
        )
        acc = tl.dot(hidden, up) * scale
        base = tl.load(
            base_ptr + offs_m[:, None] * stride_base_n + offs_n[None, :] * stride_base_o,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
            other=0.0,
        )
        tl.store(
            out_ptr + offs_m[:, None] * stride_out_n + offs_n[None, :] * stride_out_o,
            base + acc,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
        )

    @triton.jit
    def _lora_full_fused_kernel(
        x_ptr,
        down_ptr,
        up_ptr,
        base_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        rank: tl.constexpr,
        scale: tl.constexpr,
        stride_x_n: tl.constexpr,
        stride_x_k: tl.constexpr,
        stride_down_r: tl.constexpr,
        stride_down_k: tl.constexpr,
        stride_up_o: tl.constexpr,
        stride_up_r: tl.constexpr,
        stride_base_n: tl.constexpr,
        stride_base_o: tl.constexpr,
        stride_out_n: tl.constexpr,
        stride_out_o: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)

        hidden = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
        for k0 in range(0, in_features, BLOCK_K):
            k_idxs = k0 + offs_k
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_x_n + k_idxs[None, :] * stride_x_k,
                mask=(offs_m[:, None] < n_elements) & (k_idxs[None, :] < in_features),
                other=0.0,
            )
            down_tile = tl.load(
                down_ptr + offs_r[None, :] * stride_down_r + k_idxs[:, None] * stride_down_k,
                mask=(offs_r[None, :] < rank) & (k_idxs[:, None] < in_features),
                other=0.0,
            )
            hidden += tl.dot(x_tile, down_tile, input_precision="ieee")

        up = tl.load(
            up_ptr + offs_n[None, :] * stride_up_o + offs_r[:, None] * stride_up_r,
            mask=(offs_n[None, :] < out_features) & (offs_r[:, None] < rank),
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(hidden, up, input_precision="ieee") * scale
        base = tl.load(
            base_ptr + offs_m[:, None] * stride_base_n + offs_n[None, :] * stride_base_o,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
            other=0.0,
        )
        tl.store(
            out_ptr + offs_m[:, None] * stride_out_n + offs_n[None, :] * stride_out_o,
            base + acc,
            mask=(offs_m[:, None] < n_elements) & (offs_n[None, :] < out_features),
        )


def triton_lora_delta_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return ``base + (x @ down.T) @ up.T * scale`` using a Triton up/add v0."""

    if not triton_lora_delta_available():
        raise RuntimeError(f"Triton LoRA delta candidate unavailable: {triton_lora_delta_unavailable_reason()}")
    if x.device.type != "cuda" or base_output.device.type != "cuda":
        raise RuntimeError("Triton LoRA delta candidate requires CUDA tensors")
    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("LoRA tensors must have at least 2 dimensions")

    original_shape = tuple(base_output.shape)
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    if rank > 32:
        raise RuntimeError("Triton LoRA delta v0 currently supports rank <= 32")
    if int(up_weight.shape[0]) != out_features or int(up_weight.shape[1]) != rank:
        raise ValueError("up_weight must have shape [out_features, rank]")
    if int(down_weight.shape[1]) != in_features:
        raise ValueError("down_weight must have shape [rank, in_features]")

    x_2d = x.reshape(-1, in_features).contiguous()
    base_2d = base_output.reshape(-1, out_features).contiguous()
    hidden = F.linear(x_2d, down_weight).contiguous()
    up = up_weight.contiguous()
    out = torch.empty_like(base_2d)

    n_elements = int(x_2d.shape[0])
    block_m = 16
    block_n = 32
    # tl.dot requires K >= 16; rank smaller than that is handled by masks.
    block_r = 16 if rank <= 16 else 32
    num_warps = 4
    num_stages = 2
    grid = (triton.cdiv(n_elements, block_m), triton.cdiv(out_features, block_n))
    _lora_up_add_kernel[grid](
        hidden,
        up,
        base_2d,
        out,
        n_elements,
        out_features,
        rank,
        float(scale),
        hidden.stride(0),
        hidden.stride(1),
        up.stride(0),
        up.stride(1),
        base_2d.stride(0),
        base_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(original_shape)


def triton_lora_delta_v2_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Large-width-oriented Triton v2 candidate.

    v2 keeps the reliable PyTorch down projection and focuses optimization on the
    up-projection/add phase with launch settings tuned for wide output features.
    This remains benchmark/scorecard-only and is not a training dispatcher.
    """

    if not triton_lora_delta_available():
        raise RuntimeError(f"Triton LoRA delta candidate unavailable: {triton_lora_delta_unavailable_reason()}")
    if x.device.type != "cuda" or base_output.device.type != "cuda":
        raise RuntimeError("Triton LoRA delta candidate requires CUDA tensors")
    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("LoRA tensors must have at least 2 dimensions")

    original_shape = tuple(base_output.shape)
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    if rank > 32:
        raise RuntimeError("Triton LoRA delta v2 currently supports rank <= 32")
    if int(up_weight.shape[0]) != out_features or int(up_weight.shape[1]) != rank:
        raise ValueError("up_weight must have shape [out_features, rank]")
    if int(down_weight.shape[1]) != in_features:
        raise ValueError("down_weight must have shape [rank, in_features]")

    x_2d = x.reshape(-1, in_features).contiguous()
    base_2d = base_output.reshape(-1, out_features).contiguous()
    hidden = F.linear(x_2d, down_weight).contiguous()
    up = up_weight.contiguous()
    out = torch.empty_like(base_2d)

    n_elements = int(x_2d.shape[0])
    config = triton_lora_delta_v2_config_for_shape(out_features=out_features, rank=rank)
    block_m = int(config["block_m"])
    block_n = int(config["block_n"])
    block_r = int(config["block_r"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])

    grid = (triton.cdiv(n_elements, block_m), triton.cdiv(out_features, block_n))
    _lora_up_add_kernel[grid](
        hidden,
        up,
        base_2d,
        out,
        n_elements,
        out_features,
        rank,
        float(scale),
        hidden.stride(0),
        hidden.stride(1),
        up.stride(0),
        up.stride(1),
        base_2d.stride(0),
        base_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(original_shape)


def triton_lora_delta_v2_tc_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """TensorCore-friendly v2.2 candidate for fp16/bf16 research probes."""

    return triton_lora_delta_v2_tc_candidate_with_config(
        x,
        down_weight,
        up_weight,
        base_output,
        scale,
        launch_config=None,
    )


def triton_lora_delta_v2_tc_candidate_with_config(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
    *,
    launch_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """TensorCore-friendly v2.2 candidate with an explicit launch config."""

    if not triton_lora_delta_available():
        raise RuntimeError(f"Triton LoRA delta candidate unavailable: {triton_lora_delta_unavailable_reason()}")
    if x.device.type != "cuda" or base_output.device.type != "cuda":
        raise RuntimeError("Triton LoRA delta candidate requires CUDA tensors")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("Triton LoRA delta v2_tc currently targets fp16/bf16 TensorCore probes")
    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("LoRA tensors must have at least 2 dimensions")

    original_shape = tuple(base_output.shape)
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    if rank > 32:
        raise RuntimeError("Triton LoRA delta v2_tc currently supports rank <= 32")
    if int(up_weight.shape[0]) != out_features or int(up_weight.shape[1]) != rank:
        raise ValueError("up_weight must have shape [out_features, rank]")
    if int(down_weight.shape[1]) != in_features:
        raise ValueError("down_weight must have shape [rank, in_features]")

    x_2d = x.reshape(-1, in_features).contiguous()
    base_2d = base_output.reshape(-1, out_features).contiguous()
    hidden = F.linear(x_2d, down_weight).contiguous()
    up = up_weight.contiguous()
    out = torch.empty_like(base_2d)

    n_elements = int(x_2d.shape[0])
    config = (
        _normalize_v2_launch_config(launch_config, out_features=out_features, rank=rank)
        if launch_config is not None
        else triton_lora_delta_v2_config_for_shape(out_features=out_features, rank=rank)
    )
    block_m = int(config["block_m"])
    block_n = int(config["block_n"])
    block_r = int(config["block_r"])
    grid = (triton.cdiv(n_elements, block_m), triton.cdiv(out_features, block_n))
    _lora_up_add_tc_kernel[grid](
        hidden,
        up,
        base_2d,
        out,
        n_elements,
        out_features,
        rank,
        float(scale),
        hidden.stride(0),
        hidden.stride(1),
        up.stride(0),
        up.stride(1),
        base_2d.stride(0),
        base_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    return out.reshape(original_shape)


def triton_lora_delta_v3_dispatch_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Shape-aware V3 research dispatcher over V1/V2_TC/PyTorch paths."""

    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("LoRA tensors must have at least 2 dimensions")
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    decision = triton_lora_delta_v3_decision_for_shape(
        dtype=x.dtype,
        out_features=out_features,
        rank=rank,
    )
    path = str(decision.get("path") or "pytorch_explicit")
    if path == "triton_lora_delta_v1":
        return triton_lora_delta_v1_candidate(x, down_weight, up_weight, base_output, scale)
    if path == "triton_lora_delta_v2_tc":
        return triton_lora_delta_v2_tc_candidate_with_config(
            x,
            down_weight,
            up_weight,
            base_output,
            scale,
            launch_config=decision.get("config"),
        )
    return base_output + F.linear(F.linear(x, down_weight), up_weight) * float(scale)


def triton_lora_delta_v1_candidate(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    base_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Full-fused Triton v1 for common small LoRA ranks."""

    if not triton_lora_delta_available():
        raise RuntimeError(f"Triton LoRA delta candidate unavailable: {triton_lora_delta_unavailable_reason()}")
    if x.device.type != "cuda" or base_output.device.type != "cuda":
        raise RuntimeError("Triton LoRA delta candidate requires CUDA tensors")
    if x.dim() < 2 or base_output.dim() < 2:
        raise ValueError("LoRA tensors must have at least 2 dimensions")

    original_shape = tuple(base_output.shape)
    in_features = int(x.shape[-1])
    out_features = int(base_output.shape[-1])
    rank = int(down_weight.shape[0])
    if rank > 32:
        raise RuntimeError("Triton LoRA delta v1 currently supports rank <= 32")
    if int(up_weight.shape[0]) != out_features or int(up_weight.shape[1]) != rank:
        raise ValueError("up_weight must have shape [out_features, rank]")
    if int(down_weight.shape[1]) != in_features:
        raise ValueError("down_weight must have shape [rank, in_features]")

    x_2d = x.reshape(-1, in_features).contiguous()
    base_2d = base_output.reshape(-1, out_features).contiguous()
    down = down_weight.contiguous()
    up = up_weight.contiguous()
    out = torch.empty_like(base_2d)

    n_elements = int(x_2d.shape[0])
    block_m = 16
    block_n = 32
    block_k = 32
    block_r = 16 if rank <= 16 else 32
    grid = (triton.cdiv(n_elements, block_m), triton.cdiv(out_features, block_n))
    _lora_full_fused_kernel[grid](
        x_2d,
        down,
        up,
        base_2d,
        out,
        n_elements,
        in_features,
        out_features,
        rank,
        float(scale),
        x_2d.stride(0),
        x_2d.stride(1),
        down.stride(0),
        down.stride(1),
        up.stride(0),
        up.stride(1),
        base_2d.stride(0),
        base_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_R=block_r,
    )
    return out.reshape(original_shape)


def triton_lora_delta_metadata() -> dict[str, Any]:
    return {
        "name": "triton_lora_delta_v0",
        "available": triton_lora_delta_available(),
        "reason": triton_lora_delta_unavailable_reason(),
        "fusion_scope": "pytorch_down_projection_plus_triton_up_projection_add",
        "training_path_enabled": False,
    }


def triton_lora_delta_v1_metadata() -> dict[str, Any]:
    return {
        "name": "triton_lora_delta_v1",
        "available": triton_lora_delta_available(),
        "reason": triton_lora_delta_unavailable_reason(),
        "fusion_scope": "full_fused_x_down_up_base_add_rank_le_32",
        "training_path_enabled": False,
    }


def triton_lora_delta_v2_metadata() -> dict[str, Any]:
    return {
        "name": "triton_lora_delta_v2",
        "available": triton_lora_delta_available(),
        "reason": triton_lora_delta_unavailable_reason(),
        "fusion_scope": "pytorch_down_projection_plus_triton_up_add_large_width_tuned",
        "launch_configs": list(V2_LAUNCH_CONFIGS),
        "training_path_enabled": False,
    }


def triton_lora_delta_v2_tc_metadata() -> dict[str, Any]:
    return {
        "name": "triton_lora_delta_v2_tc",
        "available": triton_lora_delta_available(),
        "reason": triton_lora_delta_unavailable_reason(),
        "fusion_scope": "pytorch_down_projection_plus_tensorcore_friendly_triton_up_add",
        "launch_configs": list(V2_LAUNCH_CONFIGS),
        "sweep_configs": list(V2_TC_SWEEP_CONFIGS),
        "training_path_enabled": False,
    }


def triton_lora_delta_v3_metadata() -> dict[str, Any]:
    return {
        "name": "triton_lora_delta_v3_dispatch",
        "available": triton_lora_delta_available(),
        "reason": triton_lora_delta_unavailable_reason(),
        "fusion_scope": "shape_aware_research_dispatch_over_v1_v2_tc_pytorch",
        "tc_route_configs": {f"{width}:r{rank}": config for (width, rank), config in V3_TC_ROUTE_CONFIGS.items()},
        "disabled_tc_route_configs": {f"{width}:r{rank}": config for (width, rank), config in V3_TC_DISABLED_ROUTE_CONFIGS.items()},
        "training_path_enabled": False,
    }


__all__ = [
    "triton_lora_delta_available",
    "triton_lora_delta_candidate",
    "triton_lora_delta_metadata",
    "triton_lora_delta_v1_candidate",
    "triton_lora_delta_v1_metadata",
    "triton_lora_delta_v2_candidate",
    "triton_lora_delta_v2_config_for_shape",
    "triton_lora_delta_v2_tc_config_candidates_for_shape",
    "triton_lora_delta_v2_metadata",
    "triton_lora_delta_v2_tc_candidate",
    "triton_lora_delta_v2_tc_candidate_with_config",
    "triton_lora_delta_v2_tc_metadata",
    "triton_lora_delta_v3_decision_for_shape",
    "triton_lora_delta_v3_dispatch_candidate",
    "triton_lora_delta_v3_metadata",
    "triton_lora_delta_unavailable_reason",
]
