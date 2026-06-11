"""Cached-token DiffCR A/B replay for DiT training research.

This module is intentionally trainer-outside. It converts cache-first latents
into DiT visual tokens, then compares full-token execution with DiffCR
compressed-token execution on a tiny DiT-shaped block. Runtime/trainer
activation stays default-off.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .diffcr_token_compression import DiffCRTokenCompressionPolicy, run_diffcr_compressed_block
from .diffcr_token_compression_microbench import TinyDiffCRDiTBlock


@dataclass(frozen=True)
class DiffCRCachedTokenABConfig:
    family: str = "anima"
    compression_ratio: float = 0.5
    min_tokens: int = 4
    num_heads: int = 4
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    dtype: str = "float32"
    score_mode: str = "l2"
    layer_index: int = -1
    timestep_index: int = -1

    def normalized(self) -> "DiffCRCachedTokenABConfig":
        family = str(self.family or "anima").strip().lower()
        if family not in {"anima", "newbie"}:
            family = "anima"
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        score_mode = str(self.score_mode or "l2").strip().lower()
        if score_mode not in {"l2", "abs_mean", "provided", "sequential"}:
            score_mode = "l2"
        return DiffCRCachedTokenABConfig(
            family=family,
            compression_ratio=min(max(float(self.compression_ratio), 0.0), 1.0),
            min_tokens=max(int(self.min_tokens), 1),
            num_heads=max(int(self.num_heads), 1),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            dtype=dtype,
            score_mode=score_mode,
            layer_index=int(self.layer_index),
            timestep_index=int(self.timestep_index),
        )


@dataclass(frozen=True)
class CachedTokenReplay:
    tokens: torch.Tensor
    valid_mask: torch.Tensor | None
    family: str
    latent_shape: tuple[int, int, int, int]
    token_grid: tuple[int, int]

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.tokens.shape[2])

    @property
    def valid_token_fraction(self) -> float:
        if self.valid_mask is None:
            return 1.0
        return float(self.valid_mask.float().mean().item())

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "latent_shape": list(self.latent_shape),
            "token_grid": list(self.token_grid),
            "token_count": int(self.token_count),
            "hidden_size": int(self.hidden_size),
            "valid_token_fraction": float(self.valid_token_fraction),
        }


def build_diffcr_cached_token_replay(
    latents: torch.Tensor,
    *,
    family: str = "anima",
    padding_mask: torch.Tensor | None = None,
) -> CachedTokenReplay:
    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError("latents must have shape [batch, channels, height, width]")
    batch, channels, height, width = (int(dim) for dim in latents.shape)
    if min(batch, channels, height, width) <= 0:
        raise ValueError("latents must have non-empty dimensions")
    normalized_family = str(family or "anima").strip().lower()
    if normalized_family == "newbie":
        tokens = latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        token_grid = (height, width)
        valid_mask = _flatten_padding_mask(padding_mask, batch=batch, height=height, width=width)
    else:
        if height % 2 or width % 2:
            raise ValueError("Anima cached latents must have even height and width for 2x2 patch tokens")
        tokens = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(batch, (height // 2) * (width // 2), channels * 4)
        token_grid = (height // 2, width // 2)
        valid_mask = _patch_padding_mask(padding_mask, batch=batch, height=height, width=width)
        normalized_family = "anima"
    if valid_mask is not None:
        tokens = tokens * valid_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return CachedTokenReplay(
        tokens=tokens.contiguous(),
        valid_mask=valid_mask,
        family=normalized_family,
        latent_shape=(batch, channels, height, width),
        token_grid=token_grid,
    )


def run_diffcr_cached_token_ab(
    latents: torch.Tensor,
    config: DiffCRCachedTokenABConfig | Mapping[str, Any] | None = None,
    *,
    padding_mask: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    replay = build_diffcr_cached_token_replay(latents, family=cfg.family, padding_mask=padding_mask)
    dtype = _resolve_dtype(cfg.dtype, replay.tokens.device)
    tokens = replay.tokens.to(dtype=dtype)
    if replay.hidden_size % cfg.num_heads != 0:
        raise ValueError("cached token hidden size must be divisible by num_heads")
    score_values = _normalize_scores(scores, tokens)
    torch.manual_seed(cfg.seed)

    block = TinyDiffCRDiTBlock(replay.hidden_size, cfg.num_heads).to(device=tokens.device, dtype=dtype)
    compressed_block = TinyDiffCRDiTBlock(replay.hidden_size, cfg.num_heads).to(device=tokens.device, dtype=dtype)
    compressed_block.load_state_dict(block.state_dict())
    target = torch.randn_like(tokens) * 0.125
    policy = DiffCRTokenCompressionPolicy(
        enabled=True,
        compression_ratio=cfg.compression_ratio,
        min_tokens=cfg.min_tokens,
        score_mode=cfg.score_mode,
    )
    disabled_parity_ok = _disabled_parity_ok(block, tokens)

    for _ in range(cfg.warmup_steps):
        _step(block, tokens, target, policy=None, cfg=cfg, scores=None)
        _step(compressed_block, tokens, target, policy=policy, cfg=cfg, scores=score_values)

    full_samples = [_step(block, tokens, target, policy=None, cfg=cfg, scores=None) for _ in range(cfg.steps)]
    compressed_samples = [
        _step(compressed_block, tokens, target, policy=policy, cfg=cfg, scores=score_values)
        for _ in range(cfg.steps)
    ]
    plan = compressed_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    compressed_step_ms = _mean(sample["step_ms"] for sample in compressed_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    compressed_loss = _mean(sample["loss"] for sample in compressed_samples)
    loss_delta = abs(compressed_loss - full_loss)
    checks = {
        "compression_enabled": bool(plan.enabled),
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in compressed_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(compressed_loss),
        "input_grad_flow_ok": all(sample["input_grad_norm"] > 0.0 for sample in compressed_samples),
        "param_grad_flow_ok": all(sample["param_grad_norm"] > 0.0 for sample in compressed_samples),
        "compute_reduced": float(plan.estimated_attention_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
        "default_off": _default_off_ok(),
    }
    blockers = [f"{name}_missing" for name, ok in checks.items() if not ok]

    return {
        "schema_version": 1,
        "scorecard": "diffcr_cached_token_ab_replay_v0",
        "reducer_id": "diffcr",
        "ok": not blockers,
        "cached_token_ab_ready": not blockers,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "cache_replay": replay.as_dict(),
        "policy": _policy_payload(cfg),
        "plan": plan.as_dict(),
        "full_step_ms": full_step_ms,
        "compressed_step_ms": compressed_step_ms,
        "observed_speedup": full_step_ms / compressed_step_ms if compressed_step_ms > 0 else 0.0,
        "observed_step_reduction_ms": full_step_ms - compressed_step_ms,
        "full_loss": full_loss,
        "compressed_loss": compressed_loss,
        "observed_loss_delta": loss_delta,
        "estimated_attention_fraction": float(plan.estimated_attention_fraction),
        "estimated_attention_reduction": float(1.0 - plan.estimated_attention_fraction),
        "full_grad_norm": float(full_samples[-1]["input_grad_norm"]),
        "compressed_grad_norm": float(compressed_samples[-1]["input_grad_norm"]),
        "compressed_param_grad_norm": float(compressed_samples[-1]["param_grad_norm"]),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "default_off_scorecard": _default_off_scorecard(),
        "blocked_reasons": blockers,
        "recommended_next_step": "run representative real cached Anima/Newbie DiffCR A/B",
    }


def _step(
    block: TinyDiffCRDiTBlock,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: DiffCRTokenCompressionPolicy | None,
    cfg: DiffCRCachedTokenABConfig,
    scores: torch.Tensor | None,
) -> dict[str, Any]:
    block.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    _sync_if_cuda(work.device)
    start = time.perf_counter()
    if policy is None:
        output = block(work)
        plan = None
    else:
        output, plan = run_diffcr_compressed_block(
            work,
            block,
            policy,
            layer_index=cfg.layer_index,
            timestep_index=cfg.timestep_index,
            scores=scores,
        )
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


def _normalize_scores(scores: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor | None:
    if scores is None:
        return None
    if scores.shape != tokens.shape[:2]:
        raise ValueError("scores must have shape [batch, tokens]")
    return scores.to(device=tokens.device, dtype=torch.float32)


def _patch_padding_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    flat = _normalize_padding_mask(mask, batch=batch, height=height, width=width)
    valid = ~flat.reshape(batch, height // 2, 2, width // 2, 2).any(dim=(2, 4))
    return valid.reshape(batch, (height // 2) * (width // 2))


def _flatten_padding_mask(mask: torch.Tensor | None, *, batch: int, height: int, width: int) -> torch.Tensor | None:
    if mask is None:
        return None
    return ~_normalize_padding_mask(mask, batch=batch, height=height, width=width).reshape(batch, height * width)


def _normalize_padding_mask(mask: torch.Tensor, *, batch: int, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or tuple(mask.shape) != (batch, height, width):
        raise ValueError("padding_mask must have shape [batch, 1, height, width] or [batch, height, width]")
    return mask.to(dtype=torch.bool)


def _config(config: DiffCRCachedTokenABConfig | Mapping[str, Any] | None) -> DiffCRCachedTokenABConfig:
    if isinstance(config, DiffCRCachedTokenABConfig):
        return config.normalized()
    if isinstance(config, Mapping):
        return DiffCRCachedTokenABConfig(**dict(config)).normalized()
    return DiffCRCachedTokenABConfig().normalized()


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


def _input_grad_norm(tokens: torch.Tensor) -> float:
    if tokens.grad is None:
        return 0.0
    return float(tokens.grad.detach().float().norm().item())


def _param_grad_norm(block: torch.nn.Module) -> float:
    total = 0.0
    for param in block.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def _policy_payload(cfg: DiffCRCachedTokenABConfig) -> dict[str, Any]:
    return {
        "compression_ratio": float(cfg.compression_ratio),
        "min_tokens": int(cfg.min_tokens),
        "num_heads": int(cfg.num_heads),
        "score_mode": cfg.score_mode,
        "layer_index": int(cfg.layer_index),
        "timestep_index": int(cfg.timestep_index),
    }


def _default_off_scorecard() -> dict[str, bool]:
    return {
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


def _default_off_ok() -> bool:
    return not any(_default_off_scorecard().values())


__all__ = [
    "CachedTokenReplay",
    "DiffCRCachedTokenABConfig",
    "build_diffcr_cached_token_replay",
    "run_diffcr_cached_token_ab",
]
