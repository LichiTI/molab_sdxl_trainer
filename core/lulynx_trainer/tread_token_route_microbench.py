"""Micro A/B benchmark for TREAD-style DiT token routing.

The benchmark is intentionally trainer-outside. It measures a tiny
self-attention + MLP token block with full tokens versus routed tokens so the
frontier idea has concrete step-time, loss-delta, and gradient-flow evidence
before any production trainer wiring is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .tread_token_routing import TreadTokenRoutePolicy, run_tread_routed_block


@dataclass(frozen=True)
class TreadMicrobenchConfig:
    batch_size: int = 1
    token_count: int = 64
    hidden_size: int = 32
    num_heads: int = 4
    keep_ratio: float = 0.5
    min_keep_tokens: int = 4
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    device: str = "cpu"
    dtype: str = "float32"

    def normalized(self) -> "TreadMicrobenchConfig":
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        return TreadMicrobenchConfig(
            batch_size=max(int(self.batch_size), 1),
            token_count=max(int(self.token_count), 2),
            hidden_size=max(int(self.hidden_size), 2),
            num_heads=max(int(self.num_heads), 1),
            keep_ratio=min(max(float(self.keep_ratio), 0.0), 1.0),
            min_keep_tokens=max(int(self.min_keep_tokens), 1),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            device=str(self.device or "cpu"),
            dtype=dtype,
        )


class TinyTreadDiTBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, hidden = tokens.shape
        qkv = self.qkv(self.norm1(tokens))
        qkv = qkv.view(batch, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * scale, dim=-1)
        mixed = torch.matmul(attn, value).transpose(1, 2).reshape(batch, token_count, hidden)
        tokens = tokens + self.proj(mixed)
        return tokens + self.mlp(self.norm2(tokens))


def run_tread_token_route_microbench(
    config: TreadMicrobenchConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    block = TinyTreadDiTBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    routed_block = TinyTreadDiTBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    routed_block.load_state_dict(block.state_dict())
    tokens = torch.randn(cfg.batch_size, cfg.token_count, cfg.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125
    policy = TreadTokenRoutePolicy(
        enabled=True,
        keep_ratio=cfg.keep_ratio,
        min_keep_tokens=cfg.min_keep_tokens,
        score_mode="l2",
    )

    disabled_parity_ok = _disabled_parity_ok(block, tokens)
    for _ in range(cfg.warmup_steps):
        _step(block, tokens, target, policy=None)
        _step(routed_block, tokens, target, policy=policy)

    full_samples = [_step(block, tokens, target, policy=None) for _ in range(cfg.steps)]
    routed_samples = [_step(routed_block, tokens, target, policy=policy) for _ in range(cfg.steps)]
    plan = routed_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    routed_step_ms = _mean(sample["step_ms"] for sample in routed_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    routed_loss = _mean(sample["loss"] for sample in routed_samples)
    speedup = full_step_ms / routed_step_ms if routed_step_ms > 0 else 0.0
    loss_delta = abs(routed_loss - full_loss)
    blocker_flags = {
        "route_enabled": bool(plan.enabled),
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in routed_samples),
        "grad_flow_ok": all(sample["grad_norm"] > 0.0 for sample in routed_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(routed_loss),
        "compute_reduced": float(plan.estimated_attention_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
    }
    blockers = [f"{name}_missing" for name, ok in blocker_flags.items() if not ok]

    return {
        "schema_version": 1,
        "scorecard": "tread_token_route_micro_ab_bench_v0",
        "reducer_id": "tread",
        "ok": not blockers,
        "micro_ab_ready": not blockers,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "config": _config_payload(cfg, device, dtype),
        "plan": plan.as_dict(),
        "full_step_ms": full_step_ms,
        "routed_step_ms": routed_step_ms,
        "observed_speedup": speedup,
        "full_loss": full_loss,
        "routed_loss": routed_loss,
        "observed_loss_delta": loss_delta,
        "estimated_attention_fraction": float(plan.estimated_attention_fraction),
        "estimated_attention_reduction": float(1.0 - plan.estimated_attention_fraction),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "routed_grad_norm": float(routed_samples[-1]["grad_norm"]),
        "blocked_reasons": blockers,
        "recommended_next_step": "run representative Anima/Newbie TREAD A/B on cached samples",
    }


def _step(
    block: TinyTreadDiTBlock,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: TreadTokenRoutePolicy | None,
) -> dict[str, Any]:
    block.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    _sync_if_cuda(work.device)
    start = time.perf_counter()
    if policy is None:
        output = block(work)
        plan = None
    else:
        output, plan = run_tread_routed_block(work, block, policy)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    _sync_if_cuda(work.device)
    step_ms = (time.perf_counter() - start) * 1000.0
    grad_norm = float(work.grad.detach().float().norm().item()) if work.grad is not None else 0.0
    return {
        "loss": float(loss.detach().item()),
        "step_ms": float(step_ms),
        "grad_norm": grad_norm,
        "output_shape": tuple(output.shape),
        "plan": plan,
    }


def _disabled_parity_ok(block: TinyTreadDiTBlock, tokens: torch.Tensor) -> bool:
    with torch.no_grad():
        full = block(tokens)
        routed, plan = run_tread_routed_block(tokens, block, TreadTokenRoutePolicy(enabled=False))
    return bool(not plan.enabled and torch.equal(full, routed))


def _config(config: TreadMicrobenchConfig | Mapping[str, Any] | None) -> TreadMicrobenchConfig:
    if isinstance(config, TreadMicrobenchConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = TreadMicrobenchConfig(**dict(config))
    else:
        cfg = TreadMicrobenchConfig()
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


def _config_payload(cfg: TreadMicrobenchConfig, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "token_count": int(cfg.token_count),
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_heads),
        "keep_ratio": float(cfg.keep_ratio),
        "min_keep_tokens": int(cfg.min_keep_tokens),
        "steps": int(cfg.steps),
        "warmup_steps": int(cfg.warmup_steps),
        "seed": int(cfg.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }


__all__ = [
    "TinyTreadDiTBlock",
    "TreadMicrobenchConfig",
    "run_tread_token_route_microbench",
]
