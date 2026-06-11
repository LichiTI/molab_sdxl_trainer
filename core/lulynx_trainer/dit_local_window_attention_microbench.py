"""Micro benchmark for local/window attention in DiT training research.

The current primitive uses dense attention scores plus a local mask. This bench
therefore records theoretical window reduction and measured behavior, while
explicitly blocking acceleration claims until a sparse/window kernel exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .dit_local_window_attention import DiTLocalWindowAttentionPolicy, dit_local_window_attention


@dataclass(frozen=True)
class DiTLocalWindowMicrobenchConfig:
    batch_size: int = 1
    grid_h: int = 4
    grid_w: int = 4
    hidden_size: int = 32
    num_heads: int = 4
    window_h: int = 3
    window_w: int = 3
    one_sided: bool = False
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    device: str = "cpu"
    dtype: str = "float32"

    def normalized(self) -> "DiTLocalWindowMicrobenchConfig":
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        return DiTLocalWindowMicrobenchConfig(
            batch_size=max(int(self.batch_size), 1),
            grid_h=max(int(self.grid_h), 1),
            grid_w=max(int(self.grid_w), 1),
            hidden_size=max(int(self.hidden_size), 2),
            num_heads=max(int(self.num_heads), 1),
            window_h=max(int(self.window_h), 1),
            window_w=max(int(self.window_w), 1),
            one_sided=bool(self.one_sided),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            device=str(self.device or "cpu"),
            dtype=dtype,
        )


class TinyLocalWindowAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        policy: DiTLocalWindowAttentionPolicy | None,
    ) -> tuple[torch.Tensor, Any]:
        batch, token_count, hidden = tokens.shape
        qkv = self.qkv(self.norm(tokens))
        qkv = qkv.view(batch, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if policy is None:
            mixed = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            plan = None
        else:
            mixed, plan = dit_local_window_attention(query, key, value, policy)
        mixed = mixed.transpose(1, 2).reshape(batch, token_count, hidden)
        return tokens + self.proj(mixed), plan


def run_dit_local_window_attention_microbench(
    config: DiTLocalWindowMicrobenchConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    block = TinyLocalWindowAttentionBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    local_block = TinyLocalWindowAttentionBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    local_block.load_state_dict(block.state_dict())
    token_count = cfg.grid_h * cfg.grid_w
    tokens = torch.randn(cfg.batch_size, token_count, cfg.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125
    policy = DiTLocalWindowAttentionPolicy(
        enabled=True,
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        window_h=cfg.window_h,
        window_w=cfg.window_w,
        one_sided=cfg.one_sided,
    )
    disabled_parity_ok = _disabled_parity_ok(block, tokens)

    for _ in range(cfg.warmup_steps):
        _step(block, tokens, target, policy=None)
        _step(local_block, tokens, target, policy=policy)

    full_samples = [_step(block, tokens, target, policy=None) for _ in range(cfg.steps)]
    local_samples = [_step(local_block, tokens, target, policy=policy) for _ in range(cfg.steps)]
    plan = local_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    local_step_ms = _mean(sample["step_ms"] for sample in local_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    local_loss = _mean(sample["loss"] for sample in local_samples)
    blockers = []
    if not bool(plan.enabled):
        blockers.append("local_window_not_enabled")
    if tuple(local_samples[-1]["output_shape"]) != tuple(tokens.shape):
        blockers.append("shape_stability_missing")
    if local_samples[-1]["grad_norm"] <= 0.0:
        blockers.append("grad_flow_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_missing")
    if float(plan.estimated_attention_fraction) >= 1.0:
        blockers.append("window_reduction_missing")
    blockers.append("sparse_masked_kernel_missing")

    measurement_ready = blockers == ["sparse_masked_kernel_missing"]
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_micro_ab_bench_v0",
        "reducer_id": "local_window_attention",
        "ok": False,
        "measurement_ready": bool(measurement_ready),
        "kernel_acceleration_ready": False,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "config": _config_payload(cfg, device, dtype),
        "plan": plan.as_dict(),
        "full_step_ms": full_step_ms,
        "local_window_step_ms": local_step_ms,
        "observed_speedup": full_step_ms / local_step_ms if local_step_ms > 0 else 0.0,
        "full_loss": full_loss,
        "local_window_loss": local_loss,
        "observed_loss_delta": abs(local_loss - full_loss),
        "estimated_attention_fraction": float(plan.estimated_attention_fraction),
        "estimated_attention_reduction": float(1.0 - plan.estimated_attention_fraction),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "local_window_grad_norm": float(local_samples[-1]["grad_norm"]),
        "blocked_reasons": blockers,
        "recommended_next_step": "add sparse/window attention kernel before performance promotion",
    }


def _step(
    block: TinyLocalWindowAttentionBlock,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: DiTLocalWindowAttentionPolicy | None,
) -> dict[str, Any]:
    block.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    _sync_if_cuda(work.device)
    start = time.perf_counter()
    output, plan = block(work, policy)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    _sync_if_cuda(work.device)
    return {
        "loss": float(loss.detach().item()),
        "step_ms": float((time.perf_counter() - start) * 1000.0),
        "grad_norm": float(work.grad.detach().float().norm().item()) if work.grad is not None else 0.0,
        "output_shape": tuple(output.shape),
        "plan": plan,
    }


def _disabled_parity_ok(block: TinyLocalWindowAttentionBlock, tokens: torch.Tensor) -> bool:
    with torch.no_grad():
        full, _ = block(tokens, None)
        local, plan = block(tokens, DiTLocalWindowAttentionPolicy(enabled=False))
    return bool(not plan.enabled and torch.allclose(full, local, atol=1e-5, rtol=1e-5))


def _config(
    config: DiTLocalWindowMicrobenchConfig | Mapping[str, Any] | None,
) -> DiTLocalWindowMicrobenchConfig:
    if isinstance(config, DiTLocalWindowMicrobenchConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiTLocalWindowMicrobenchConfig(**dict(config))
    else:
        cfg = DiTLocalWindowMicrobenchConfig()
    cfg = cfg.normalized()
    if cfg.hidden_size % cfg.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    return cfg


def _resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: Any) -> float:
    payload = [float(value) for value in values]
    return float(sum(payload) / max(len(payload), 1))


def _config_payload(
    cfg: DiTLocalWindowMicrobenchConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "grid_h": int(cfg.grid_h),
        "grid_w": int(cfg.grid_w),
        "token_count": int(cfg.grid_h * cfg.grid_w),
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_heads),
        "window_h": int(cfg.window_h),
        "window_w": int(cfg.window_w),
        "one_sided": bool(cfg.one_sided),
        "steps": int(cfg.steps),
        "warmup_steps": int(cfg.warmup_steps),
        "seed": int(cfg.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }


__all__ = [
    "DiTLocalWindowMicrobenchConfig",
    "TinyLocalWindowAttentionBlock",
    "run_dit_local_window_attention_microbench",
]
