"""Micro A/B benchmark for DiT-BlockSkip training research."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .dit_blockskip_training_spike import DiTBlockSkipPolicy, build_dit_blockskip_plan, run_dit_blockskip_sequence


@dataclass(frozen=True)
class DiTBlockSkipMicrobenchConfig:
    batch_size: int = 1
    token_count: int = 32
    hidden_size: int = 24
    num_heads: int = 4
    block_count: int = 4
    skip_ratio: float = 0.5
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    device: str = "cpu"
    dtype: str = "float32"

    def normalized(self) -> "DiTBlockSkipMicrobenchConfig":
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        return DiTBlockSkipMicrobenchConfig(
            batch_size=max(int(self.batch_size), 1),
            token_count=max(int(self.token_count), 2),
            hidden_size=max(int(self.hidden_size), 2),
            num_heads=max(int(self.num_heads), 1),
            block_count=max(int(self.block_count), 1),
            skip_ratio=min(max(float(self.skip_ratio), 0.0), 0.95),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            device=str(self.device or "cpu"),
            dtype=dtype,
        )


class TinyBlockSkipDiTBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.calls = 0
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
        self.calls += 1
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


def run_dit_blockskip_microbench(
    config: DiTBlockSkipMicrobenchConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    blocks = _make_blocks(cfg, device, dtype)
    skipped_blocks = _make_blocks(cfg, device, dtype)
    skipped_blocks.load_state_dict(blocks.state_dict())
    tokens = torch.randn(cfg.batch_size, cfg.token_count, cfg.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125
    policy = DiTBlockSkipPolicy(enabled=True, skip_ratio=cfg.skip_ratio)
    disabled_parity_ok = _disabled_parity_ok(blocks, tokens)

    for _ in range(cfg.warmup_steps):
        _step(blocks, tokens, target, policy=None)
        _step(skipped_blocks, tokens, target, policy=policy)

    full_samples = [_step(blocks, tokens, target, policy=None) for _ in range(cfg.steps)]
    skipped_samples = [_step(skipped_blocks, tokens, target, policy=policy) for _ in range(cfg.steps)]
    plan = skipped_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    skipped_step_ms = _mean(sample["step_ms"] for sample in skipped_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    skipped_loss = _mean(sample["loss"] for sample in skipped_samples)
    speedup = full_step_ms / skipped_step_ms if skipped_step_ms > 0 else 0.0
    loss_delta = abs(skipped_loss - full_loss)
    checks = {
        "skip_enabled": bool(plan.enabled),
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in skipped_samples),
        "grad_flow_ok": all(sample["grad_norm"] > 0.0 for sample in skipped_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(skipped_loss),
        "compute_reduced": float(plan.estimated_block_compute_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
    }
    blockers = [f"{name}_missing" for name, ok in checks.items() if not ok]

    return {
        "schema_version": 1,
        "scorecard": "dit_blockskip_micro_ab_bench_v0",
        "reducer_id": "blockskip",
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
        "skipped_step_ms": skipped_step_ms,
        "observed_speedup": speedup,
        "full_loss": full_loss,
        "skipped_loss": skipped_loss,
        "observed_loss_delta": loss_delta,
        "estimated_block_compute_fraction": float(plan.estimated_block_compute_fraction),
        "estimated_block_compute_reduction": float(1.0 - plan.estimated_block_compute_fraction),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "skipped_grad_norm": float(skipped_samples[-1]["grad_norm"]),
        "executed_block_calls": int(skipped_samples[-1]["executed_block_calls"]),
        "blocked_reasons": blockers,
        "recommended_next_step": "run representative cached Anima/Newbie BlockSkip A/B",
    }


def _step(
    blocks: torch.nn.ModuleList,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: DiTBlockSkipPolicy | None,
) -> dict[str, Any]:
    blocks.zero_grad(set_to_none=True)
    _reset_calls(blocks)
    work = tokens.detach().clone().requires_grad_(True)
    _sync_if_cuda(work.device)
    start = time.perf_counter()
    if policy is None:
        output = work
        for block in blocks:
            output = block(output)
        plan = None
    else:
        plan = build_dit_blockskip_plan(total_blocks=len(blocks), step_index=0, total_steps=1, policy=policy)
        output = run_dit_blockskip_sequence(work, list(blocks), plan)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    _sync_if_cuda(work.device)
    return {
        "loss": float(loss.detach().item()),
        "step_ms": float((time.perf_counter() - start) * 1000.0),
        "grad_norm": float(work.grad.detach().float().norm().item()) if work.grad is not None else 0.0,
        "output_shape": tuple(output.shape),
        "plan": plan,
        "executed_block_calls": sum(int(getattr(block, "calls", 0)) for block in blocks),
    }


def _disabled_parity_ok(blocks: torch.nn.ModuleList, tokens: torch.Tensor) -> bool:
    with torch.no_grad():
        full = tokens
        for block in blocks:
            full = block(full)
        plan = build_dit_blockskip_plan(
            total_blocks=len(blocks),
            step_index=0,
            total_steps=1,
            policy=DiTBlockSkipPolicy(enabled=False, skip_ratio=0.5),
        )
        skipped = run_dit_blockskip_sequence(tokens, list(blocks), plan)
    return bool(not plan.enabled and torch.equal(full, skipped))


def _make_blocks(cfg: DiTBlockSkipMicrobenchConfig, device: torch.device, dtype: torch.dtype) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [TinyBlockSkipDiTBlock(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.block_count)]
    ).to(device=device, dtype=dtype)


def _reset_calls(blocks: torch.nn.ModuleList) -> None:
    for block in blocks:
        if hasattr(block, "calls"):
            block.calls = 0


def _config(config: DiTBlockSkipMicrobenchConfig | Mapping[str, Any] | None) -> DiTBlockSkipMicrobenchConfig:
    if isinstance(config, DiTBlockSkipMicrobenchConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiTBlockSkipMicrobenchConfig(**dict(config))
    else:
        cfg = DiTBlockSkipMicrobenchConfig()
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


def _config_payload(cfg: DiTBlockSkipMicrobenchConfig, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "token_count": int(cfg.token_count),
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_heads),
        "block_count": int(cfg.block_count),
        "skip_ratio": float(cfg.skip_ratio),
        "steps": int(cfg.steps),
        "warmup_steps": int(cfg.warmup_steps),
        "seed": int(cfg.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }


__all__ = [
    "DiTBlockSkipMicrobenchConfig",
    "TinyBlockSkipDiTBlock",
    "run_dit_blockskip_microbench",
]
