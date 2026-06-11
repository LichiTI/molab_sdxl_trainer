"""Micro A/B benchmark for DiffCR-style DiT token compression.

This is a trainer-outside research bench. It compares full-token execution with
DiffCR compressed-token execution on a tiny self-attention + MLP block and keeps
all production training paths disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .diffcr_token_compression import DiffCRTokenCompressionPolicy, run_diffcr_compressed_block


@dataclass(frozen=True)
class DiffCRMicrobenchConfig:
    batch_size: int = 1
    token_count: int = 64
    hidden_size: int = 32
    num_heads: int = 4
    compression_ratio: float = 0.5
    min_tokens: int = 4
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    device: str = "cpu"
    dtype: str = "float32"
    score_mode: str = "sequential"

    def normalized(self) -> "DiffCRMicrobenchConfig":
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        score_mode = str(self.score_mode or "sequential").strip().lower()
        if score_mode not in {"l2", "abs_mean", "sequential"}:
            score_mode = "sequential"
        return DiffCRMicrobenchConfig(
            batch_size=max(int(self.batch_size), 1),
            token_count=max(int(self.token_count), 2),
            hidden_size=max(int(self.hidden_size), 2),
            num_heads=max(int(self.num_heads), 1),
            compression_ratio=min(max(float(self.compression_ratio), 0.0), 1.0),
            min_tokens=max(int(self.min_tokens), 1),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            device=str(self.device or "cpu"),
            dtype=dtype,
            score_mode=score_mode,
        )


class TinyDiffCRDiTBlock(torch.nn.Module):
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
            torch.nn.SiLU(),
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


def run_diffcr_token_compression_microbench(
    config: DiffCRMicrobenchConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    block = TinyDiffCRDiTBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    compressed_block = TinyDiffCRDiTBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    compressed_block.load_state_dict(block.state_dict())
    tokens = torch.randn(cfg.batch_size, cfg.token_count, cfg.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125
    policy = DiffCRTokenCompressionPolicy(
        enabled=True,
        compression_ratio=cfg.compression_ratio,
        min_tokens=cfg.min_tokens,
        score_mode=cfg.score_mode,
    )

    disabled_parity_ok = _disabled_parity_ok(block, tokens)
    for _ in range(cfg.warmup_steps):
        _step(block, tokens, target, policy=None)
        _step(compressed_block, tokens, target, policy=policy)

    full_samples = [_step(block, tokens, target, policy=None) for _ in range(cfg.steps)]
    compressed_samples = [_step(compressed_block, tokens, target, policy=policy) for _ in range(cfg.steps)]
    plan = compressed_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    compressed_step_ms = _mean(sample["step_ms"] for sample in compressed_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    compressed_loss = _mean(sample["loss"] for sample in compressed_samples)
    speedup = full_step_ms / compressed_step_ms if compressed_step_ms > 0 else 0.0
    loss_delta = abs(compressed_loss - full_loss)
    checks = {
        "compression_enabled": bool(plan.enabled),
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in compressed_samples),
        "grad_flow_ok": all(sample["grad_norm"] > 0.0 for sample in compressed_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(compressed_loss),
        "compute_reduced": float(plan.estimated_attention_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
    }
    blockers = [f"{name}_missing" for name, ok in checks.items() if not ok]

    return {
        "schema_version": 1,
        "scorecard": "diffcr_token_compression_micro_ab_bench_v0",
        "reducer_id": "diffcr",
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
        "compressed_step_ms": compressed_step_ms,
        "observed_speedup": speedup,
        "full_loss": full_loss,
        "compressed_loss": compressed_loss,
        "observed_loss_delta": loss_delta,
        "estimated_attention_fraction": float(plan.estimated_attention_fraction),
        "estimated_attention_reduction": float(1.0 - plan.estimated_attention_fraction),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "compressed_grad_norm": float(compressed_samples[-1]["grad_norm"]),
        "blocked_reasons": blockers,
        "recommended_next_step": "run representative 2K+ cached Anima/Newbie DiffCR A/B",
    }


def _step(
    block: TinyDiffCRDiTBlock,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: DiffCRTokenCompressionPolicy | None,
) -> dict[str, Any]:
    block.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    _sync_if_cuda(work.device)
    start = time.perf_counter()
    if policy is None:
        output = block(work)
        plan = None
    else:
        output, plan = run_diffcr_compressed_block(work, block, policy)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    _sync_if_cuda(work.device)
    grad_norm = float(work.grad.detach().float().norm().item()) if work.grad is not None else 0.0
    return {
        "loss": float(loss.detach().item()),
        "step_ms": float((time.perf_counter() - start) * 1000.0),
        "grad_norm": grad_norm,
        "output_shape": tuple(output.shape),
        "plan": plan,
    }


def _disabled_parity_ok(block: TinyDiffCRDiTBlock, tokens: torch.Tensor) -> bool:
    with torch.no_grad():
        full = block(tokens)
        compressed, plan = run_diffcr_compressed_block(
            tokens,
            block,
            DiffCRTokenCompressionPolicy(enabled=False, compression_ratio=0.5),
        )
    return bool(not plan.enabled and torch.equal(full, compressed))


def _config(config: DiffCRMicrobenchConfig | Mapping[str, Any] | None) -> DiffCRMicrobenchConfig:
    if isinstance(config, DiffCRMicrobenchConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiffCRMicrobenchConfig(**dict(config))
    else:
        cfg = DiffCRMicrobenchConfig()
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


def _config_payload(cfg: DiffCRMicrobenchConfig, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "token_count": int(cfg.token_count),
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_heads),
        "compression_ratio": float(cfg.compression_ratio),
        "min_tokens": int(cfg.min_tokens),
        "steps": int(cfg.steps),
        "warmup_steps": int(cfg.warmup_steps),
        "seed": int(cfg.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "score_mode": cfg.score_mode,
    }


__all__ = [
    "DiffCRMicrobenchConfig",
    "TinyDiffCRDiTBlock",
    "run_diffcr_token_compression_microbench",
]
