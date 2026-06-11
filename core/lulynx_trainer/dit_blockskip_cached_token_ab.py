"""Cached-token DiT-BlockSkip A/B replay for compute-reducer research.

The replay stays outside the production trainer. It consumes cache-first
latents, converts them to Anima/Newbie visual-token layouts, then compares a
tiny full block sequence against the scheduled BlockSkip sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from .dit_blockskip_microbench import TinyBlockSkipDiTBlock
from .dit_blockskip_training_spike import DiTBlockSkipPolicy, build_dit_blockskip_plan, run_dit_blockskip_sequence


FAMILIES = ("anima", "newbie")


@dataclass(frozen=True)
class DiTBlockSkipCachedTokenABConfig:
    family: str = "both"
    batch_size: int = 1
    latent_channels: int = 8
    latent_height: int = 8
    latent_width: int = 8
    num_heads: int = 4
    block_count: int = 4
    skip_ratio: float = 0.5
    steps: int = 3
    warmup_steps: int = 1
    seed: int = 20260605
    device: str = "cpu"
    dtype: str = "float32"

    def normalized(self) -> "DiTBlockSkipCachedTokenABConfig":
        family = str(self.family or "both").strip().lower()
        if family not in {*FAMILIES, "both"}:
            family = "both"
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        height = max(int(self.latent_height), 2)
        width = max(int(self.latent_width), 2)
        if height % 2:
            height += 1
        if width % 2:
            width += 1
        return DiTBlockSkipCachedTokenABConfig(
            family=family,
            batch_size=max(int(self.batch_size), 1),
            latent_channels=max(int(self.latent_channels), 1),
            latent_height=height,
            latent_width=width,
            num_heads=max(int(self.num_heads), 1),
            block_count=max(int(self.block_count), 1),
            skip_ratio=min(max(float(self.skip_ratio), 0.0), 0.95),
            steps=max(int(self.steps), 1),
            warmup_steps=max(int(self.warmup_steps), 0),
            seed=int(self.seed),
            device=str(self.device or "cpu"),
            dtype=dtype,
        )


@dataclass(frozen=True)
class BlockSkipCachedTokenReplay:
    tokens: torch.Tensor
    valid_mask: torch.Tensor | None
    family: str
    latent_shape: tuple[int, int, int, int]
    token_grid: tuple[int, int]
    token_semantics: str

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
            "token_semantics": self.token_semantics,
            "valid_token_fraction": float(self.valid_token_fraction),
        }


def build_blockskip_cached_token_replay(
    latents: torch.Tensor,
    *,
    family: str = "anima",
    padding_mask: torch.Tensor | None = None,
) -> BlockSkipCachedTokenReplay:
    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError("latents must have shape [batch, channels, height, width]")
    if not torch.is_floating_point(latents):
        latents = latents.float()
    batch, channels, height, width = (int(dim) for dim in latents.shape)
    if min(batch, channels, height, width) <= 0:
        raise ValueError("latents must have non-empty dimensions")

    normalized_family = str(family or "anima").strip().lower()
    if normalized_family == "newbie":
        tokens = latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        valid_mask = _flatten_padding_mask(padding_mask, batch=batch, height=height, width=width)
        replay = BlockSkipCachedTokenReplay(
            tokens=tokens,
            valid_mask=valid_mask,
            family="newbie",
            latent_shape=(batch, channels, height, width),
            token_grid=(height, width),
            token_semantics="per_pixel_latent_token",
        )
    else:
        if height % 2 or width % 2:
            raise ValueError("Anima cached latents must have even height and width for 2x2 patch tokens")
        tokens = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(batch, (height // 2) * (width // 2), channels * 4)
        valid_mask = _patch_padding_mask(padding_mask, batch=batch, height=height, width=width)
        replay = BlockSkipCachedTokenReplay(
            tokens=tokens,
            valid_mask=valid_mask,
            family="anima",
            latent_shape=(batch, channels, height, width),
            token_grid=(height // 2, width // 2),
            token_semantics="2x2_patch_latent_token",
        )
    return _apply_valid_mask(replay)


def run_dit_blockskip_cached_token_ab(
    latents: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    config: DiTBlockSkipCachedTokenABConfig | Mapping[str, Any] | None = None,
    *,
    padding_masks: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    requested = FAMILIES if cfg.family == "both" else (cfg.family,)
    family_reports = [
        _run_family_ab(
            family=family,
            latents=_latents_for_family(latents, family=family, cfg=cfg, device=device),
            padding_mask=_padding_mask_for_family(padding_masks, family),
            cfg=cfg,
            dtype=dtype,
            seed_offset=index,
        )
        for index, family in enumerate(requested)
    ]
    blockers = [f"{row['family']}:{reason}" for row in family_reports for reason in row["blocked_reasons"]]
    mean_fraction = _mean(row["estimated_block_compute_fraction"] for row in family_reports)
    mean_reduction = _mean(row["estimated_block_compute_reduction"] for row in family_reports)
    return {
        "schema_version": 1,
        "scorecard": "dit_blockskip_cached_token_ab_replay_v0",
        "reducer_id": "blockskip",
        "ok": not blockers,
        "cached_token_ab_ready": not blockers,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "config": _config_payload(cfg, device, dtype),
        "families": list(requested),
        "family_count": len(family_reports),
        "family_reports": family_reports,
        "estimated_block_compute_fraction": float(mean_fraction),
        "estimated_block_compute_reduction": float(mean_reduction),
        "blocked_reasons": blockers,
        "recommended_next_step": "ingest real cached Anima/Newbie BlockSkip A/B results before trainer wiring",
    }


def _run_family_ab(
    *,
    family: str,
    latents: torch.Tensor,
    padding_mask: torch.Tensor | None,
    cfg: DiTBlockSkipCachedTokenABConfig,
    dtype: torch.dtype,
    seed_offset: int,
) -> dict[str, Any]:
    replay = build_blockskip_cached_token_replay(latents, family=family, padding_mask=padding_mask)
    tokens = replay.tokens.to(dtype=dtype)
    if replay.hidden_size % cfg.num_heads != 0:
        raise ValueError(f"{family} cached token hidden size must be divisible by num_heads")
    torch.manual_seed(cfg.seed + seed_offset)
    blocks = _make_blocks(replay.hidden_size, cfg, tokens.device, dtype)
    skipped_blocks = _make_blocks(replay.hidden_size, cfg, tokens.device, dtype)
    skipped_blocks.load_state_dict(blocks.state_dict())
    target = torch.randn_like(tokens) * 0.125
    policy = DiTBlockSkipPolicy(enabled=True, skip_ratio=cfg.skip_ratio)
    disabled_parity_ok = _disabled_parity_ok(blocks, tokens)

    for step_index in range(cfg.warmup_steps):
        _step(blocks, tokens, target, policy=None, step_index=step_index, total_steps=cfg.steps)
        _step(skipped_blocks, tokens, target, policy=policy, step_index=step_index, total_steps=cfg.steps)

    full_samples = [
        _step(blocks, tokens, target, policy=None, step_index=step_index, total_steps=cfg.steps)
        for step_index in range(cfg.steps)
    ]
    skipped_samples = [
        _step(skipped_blocks, tokens, target, policy=policy, step_index=step_index, total_steps=cfg.steps)
        for step_index in range(cfg.steps)
    ]
    plan = skipped_samples[-1]["plan"]
    full_step_ms = _mean(sample["step_ms"] for sample in full_samples)
    skipped_step_ms = _mean(sample["step_ms"] for sample in skipped_samples)
    full_loss = _mean(sample["loss"] for sample in full_samples)
    skipped_loss = _mean(sample["loss"] for sample in skipped_samples)
    checks = {
        "skip_enabled": bool(plan.enabled),
        "cache_first_tokens": replay.token_count > 0 and replay.hidden_size > 0,
        "shape_stable": all(tuple(sample["output_shape"]) == tuple(tokens.shape) for sample in skipped_samples),
        "grad_flow_ok": all(sample["grad_norm"] > 0.0 for sample in skipped_samples),
        "finite_loss": math.isfinite(full_loss) and math.isfinite(skipped_loss),
        "compute_reduced": float(plan.estimated_block_compute_fraction) < 1.0,
        "disabled_parity_ok": bool(disabled_parity_ok),
    }
    blockers = [f"{name}_missing" for name, ok in checks.items() if not ok]
    return {
        "family": family,
        "ok": not blockers,
        "cache_replay": replay.as_dict(),
        "plan": plan.as_dict(),
        "full_step_ms": full_step_ms,
        "skipped_step_ms": skipped_step_ms,
        "observed_speedup": full_step_ms / skipped_step_ms if skipped_step_ms > 0 else 0.0,
        "full_loss": full_loss,
        "skipped_loss": skipped_loss,
        "observed_loss_delta": abs(skipped_loss - full_loss),
        "estimated_block_compute_fraction": float(plan.estimated_block_compute_fraction),
        "estimated_block_compute_reduction": float(1.0 - plan.estimated_block_compute_fraction),
        "full_grad_norm": float(full_samples[-1]["grad_norm"]),
        "skipped_grad_norm": float(skipped_samples[-1]["grad_norm"]),
        "executed_block_calls": int(skipped_samples[-1]["executed_block_calls"]),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
    }


def _step(
    blocks: torch.nn.ModuleList,
    tokens: torch.Tensor,
    target: torch.Tensor,
    *,
    policy: DiTBlockSkipPolicy | None,
    step_index: int,
    total_steps: int,
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
        plan = build_dit_blockskip_plan(
            total_blocks=len(blocks),
            step_index=step_index,
            total_steps=total_steps,
            policy=policy,
        )
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


def _make_blocks(
    hidden_size: int,
    cfg: DiTBlockSkipCachedTokenABConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [TinyBlockSkipDiTBlock(hidden_size, cfg.num_heads) for _ in range(cfg.block_count)]
    ).to(device=device, dtype=dtype)


def _latents_for_family(
    latents: torch.Tensor | Mapping[str, torch.Tensor] | None,
    *,
    family: str,
    cfg: DiTBlockSkipCachedTokenABConfig,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(latents, Mapping):
        if family not in latents:
            raise ValueError(f"latents mapping must include {family}")
        return latents[family].to(device=device)
    if isinstance(latents, torch.Tensor):
        return latents.to(device=device)
    total = cfg.batch_size * cfg.latent_channels * cfg.latent_height * cfg.latent_width
    values = torch.linspace(-1.0, 1.0, steps=total, device=device, dtype=torch.float32)
    offset = 0.125 if family == "newbie" else 0.0
    return values.reshape(cfg.batch_size, cfg.latent_channels, cfg.latent_height, cfg.latent_width) + offset


def _padding_mask_for_family(
    padding_masks: Mapping[str, torch.Tensor] | torch.Tensor | None,
    family: str,
) -> torch.Tensor | None:
    if isinstance(padding_masks, Mapping):
        return padding_masks.get(family)
    return padding_masks


def _apply_valid_mask(replay: BlockSkipCachedTokenReplay) -> BlockSkipCachedTokenReplay:
    if replay.valid_mask is None:
        return BlockSkipCachedTokenReplay(
            tokens=replay.tokens.contiguous(),
            valid_mask=None,
            family=replay.family,
            latent_shape=replay.latent_shape,
            token_grid=replay.token_grid,
            token_semantics=replay.token_semantics,
        )
    mask = replay.valid_mask.to(device=replay.tokens.device, dtype=replay.tokens.dtype).unsqueeze(-1)
    return BlockSkipCachedTokenReplay(
        tokens=(replay.tokens * mask).contiguous(),
        valid_mask=replay.valid_mask,
        family=replay.family,
        latent_shape=replay.latent_shape,
        token_grid=replay.token_grid,
        token_semantics=replay.token_semantics,
    )


def _patch_padding_mask(mask: torch.Tensor | None, *, batch: int, height: int, width: int) -> torch.Tensor | None:
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


def _config(
    config: DiTBlockSkipCachedTokenABConfig | Mapping[str, Any] | None,
) -> DiTBlockSkipCachedTokenABConfig:
    if isinstance(config, DiTBlockSkipCachedTokenABConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiTBlockSkipCachedTokenABConfig(**dict(config))
    else:
        cfg = DiTBlockSkipCachedTokenABConfig()
    return cfg.normalized()


def _reset_calls(blocks: torch.nn.ModuleList) -> None:
    for block in blocks:
        if hasattr(block, "calls"):
            block.calls = 0


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
    cfg: DiTBlockSkipCachedTokenABConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "family": cfg.family,
        "batch_size": int(cfg.batch_size),
        "latent_channels": int(cfg.latent_channels),
        "latent_height": int(cfg.latent_height),
        "latent_width": int(cfg.latent_width),
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
    "BlockSkipCachedTokenReplay",
    "DiTBlockSkipCachedTokenABConfig",
    "build_blockskip_cached_token_replay",
    "run_dit_blockskip_cached_token_ab",
]
