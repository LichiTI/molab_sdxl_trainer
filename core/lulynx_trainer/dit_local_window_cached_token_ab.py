"""Cached-token local/window attention A/B replay for DiT training research.

The current primitive uses dense masked attention, so this replay can only
prove shape/gradient behavior and theoretical attention-pair reduction. It
keeps a sparse-kernel blocker until a real window/sparse kernel exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .diffcr_cached_token_ab import build_diffcr_cached_token_replay
from .dit_local_window_attention import DiTLocalWindowAttentionPolicy, dit_local_window_attention_gather
from .dit_local_window_attention_microbench import TinyLocalWindowAttentionBlock


@dataclass(frozen=True)
class DiTLocalWindowCachedTokenABConfig:
    family: str = "anima"
    window_h: int = 3
    window_w: int = 3
    one_sided: bool = False
    num_heads: int = 4
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    dtype: str = "float32"

    def normalized(self) -> "DiTLocalWindowCachedTokenABConfig":
        family = str(self.family or "anima").strip().lower()
        if family not in {"anima", "newbie"}:
            family = "anima"
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        return DiTLocalWindowCachedTokenABConfig(
            family=family,
            window_h=max(int(self.window_h), 1),
            window_w=max(int(self.window_w), 1),
            one_sided=bool(self.one_sided),
            num_heads=max(int(self.num_heads), 1),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            dtype=dtype,
        )


def run_dit_local_window_cached_token_ab(
    latents: torch.Tensor,
    config: DiTLocalWindowCachedTokenABConfig | Mapping[str, Any] | None = None,
    *,
    padding_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    replay = build_diffcr_cached_token_replay(latents, family=cfg.family, padding_mask=padding_mask)
    dtype = _resolve_dtype(cfg.dtype, replay.tokens.device)
    tokens = replay.tokens.to(dtype=dtype)
    if replay.hidden_size % cfg.num_heads != 0:
        raise ValueError("cached token hidden size must be divisible by num_heads")
    torch.manual_seed(cfg.seed)

    full_block = TinyLocalWindowAttentionBlock(replay.hidden_size, cfg.num_heads).to(
        device=tokens.device,
        dtype=dtype,
    )
    local_block = TinyLocalWindowAttentionBlock(replay.hidden_size, cfg.num_heads).to(
        device=tokens.device,
        dtype=dtype,
    )
    local_block.load_state_dict(full_block.state_dict())
    target = torch.randn_like(tokens) * 0.125
    grid_h, grid_w = replay.token_grid
    policy = DiTLocalWindowAttentionPolicy(
        enabled=True,
        grid_h=grid_h,
        grid_w=grid_w,
        window_h=cfg.window_h,
        window_w=cfg.window_w,
        one_sided=cfg.one_sided,
    )
    disabled_parity_ok = _disabled_parity_ok(full_block, tokens, grid_h=grid_h, grid_w=grid_w)

    for _ in range(cfg.warmup_steps):
        _step(full_block, tokens, target, policy=None)
        _step(local_block, tokens, target, policy=policy)

    full_samples = [_step(full_block, tokens, target, policy=None) for _ in range(cfg.steps)]
    local_samples = [_step(local_block, tokens, target, policy=policy) for _ in range(cfg.steps)]
    plan = local_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    local_step_ms = _mean(sample["step_ms"] for sample in local_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    local_loss = _mean(sample["loss"] for sample in local_samples)
    loss_delta = abs(local_loss - full_loss)
    checks = {
        "local_window_enabled": bool(plan.enabled),
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in local_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(local_loss),
        "input_grad_flow_ok": all(sample["input_grad_norm"] > 0.0 for sample in local_samples),
        "param_grad_flow_ok": all(sample["param_grad_norm"] > 0.0 for sample in local_samples),
        "compute_reduced": float(plan.estimated_attention_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
        "default_off": _default_off_ok(),
    }
    blockers = [f"{name}_missing" for name, ok in checks.items() if not ok]
    blockers.append("optimized_cuda_window_kernel_missing")
    replay_ready = blockers == ["optimized_cuda_window_kernel_missing"]

    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_cached_token_ab_replay_v0",
        "reducer_id": "local_window_attention",
        "ok": False,
        "cached_token_ab_ready": bool(replay_ready),
        "kernel_acceleration_ready": False,
        "window_gather_kernel_ready": True,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "cache_replay": replay.as_dict(),
        "policy": _policy_payload(cfg),
        "plan": plan.as_dict(),
        "kernel": dict(local_samples[-1]["kernel"] or {}),
        "full_step_ms": full_step_ms,
        "local_window_step_ms": local_step_ms,
        "observed_speedup": full_step_ms / local_step_ms if local_step_ms > 0 else 0.0,
        "observed_step_reduction_ms": full_step_ms - local_step_ms,
        "full_loss": full_loss,
        "local_window_loss": local_loss,
        "observed_loss_delta": loss_delta,
        "estimated_attention_fraction": float(plan.estimated_attention_fraction),
        "estimated_attention_reduction": float(1.0 - plan.estimated_attention_fraction),
        "full_grad_norm": float(full_samples[-1]["input_grad_norm"]),
        "local_window_grad_norm": float(local_samples[-1]["input_grad_norm"]),
        "local_window_param_grad_norm": float(local_samples[-1]["param_grad_norm"]),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "default_off_scorecard": _default_off_scorecard(),
        "blocked_reasons": blockers,
        "recommended_next_step": "replace torch gather prototype with optimized CUDA/window kernel before trainer A/B promotion",
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
    if policy is None:
        output, plan = block(work, policy)
        kernel = {"kernel": "full_attention_baseline", "dense_score_matrix_built": True}
    else:
        output, plan, kernel = _block_gather_attention(block, work, policy)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    _sync_if_cuda(work.device)
    return {
        "loss": float(loss.detach().item()),
        "step_ms": float((time.perf_counter() - start) * 1000.0),
        "input_grad_norm": _input_grad_norm(work),
        "param_grad_norm": _param_grad_norm(block),
        "output_shape": tuple(output.shape),
        "plan": plan,
        "kernel": kernel,
    }


def _disabled_parity_ok(block: TinyLocalWindowAttentionBlock, tokens: torch.Tensor, *, grid_h: int, grid_w: int) -> bool:
    with torch.no_grad():
        full, _ = block(tokens, None)
        local, plan = block(
            tokens,
            DiTLocalWindowAttentionPolicy(enabled=False, grid_h=grid_h, grid_w=grid_w),
        )
    return bool(not plan.enabled and torch.allclose(full, local, atol=1e-5, rtol=1e-5))


def _block_gather_attention(
    block: TinyLocalWindowAttentionBlock,
    tokens: torch.Tensor,
    policy: DiTLocalWindowAttentionPolicy,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    batch, token_count, hidden = tokens.shape
    qkv = block.qkv(block.norm(tokens))
    qkv = qkv.view(batch, token_count, 3, block.num_heads, block.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    mixed, plan, kernel = dit_local_window_attention_gather(query, key, value, policy)
    mixed = mixed.transpose(1, 2).reshape(batch, token_count, hidden)
    return tokens + block.proj(mixed), plan, kernel


def _config(
    config: DiTLocalWindowCachedTokenABConfig | Mapping[str, Any] | None,
) -> DiTLocalWindowCachedTokenABConfig:
    if isinstance(config, DiTLocalWindowCachedTokenABConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiTLocalWindowCachedTokenABConfig(**dict(config))
    else:
        cfg = DiTLocalWindowCachedTokenABConfig()
    return cfg.normalized()


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


def _input_grad_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().float().norm().item())


def _param_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            value = float(param.grad.detach().float().norm().item())
            total += value * value
    return float(math.sqrt(total))


def _mean(values: Any) -> float:
    payload = [float(value) for value in values]
    return float(sum(payload) / max(len(payload), 1))


def _policy_payload(cfg: DiTLocalWindowCachedTokenABConfig) -> dict[str, Any]:
    return {
        "family": cfg.family,
        "window_h": int(cfg.window_h),
        "window_w": int(cfg.window_w),
        "one_sided": bool(cfg.one_sided),
        "num_heads": int(cfg.num_heads),
        "steps": int(cfg.steps),
        "warmup_steps": int(cfg.warmup_steps),
        "seed": int(cfg.seed),
        "dtype": cfg.dtype,
    }


def _default_off_ok() -> bool:
    return not any(
        (
            False,  # training_path_enabled
            False,  # trainer_wiring_allowed
            False,  # request_fields_emitted
            False,  # runtime_activation_enabled
            False,  # default_behavior_changed
            False,  # promotion_ready
        )
    )


def _default_off_scorecard() -> dict[str, Any]:
    return {
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


__all__ = [
    "DiTLocalWindowCachedTokenABConfig",
    "run_dit_local_window_cached_token_ab",
]
