# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Fine-grained per-layer LoRA rank allocation policy (FG-LoRA, frontier #4).

This is the *orthogonal* rank-budget direction of ``fg_lora_rank_policy``: every
matched target layer is kept and only each layer's LoRA rank is reallocated by a
configurable depth profile (optionally conserving the total parameter budget).
It is the complement of ``adapter_target_policy`` (which *prunes* layers and
couples rank to a selection score); the trainer routes ``coupled_prune`` to that
existing engine and ``orthogonal_redistribute`` / ``profiled`` here.

Design contract:
  * ``uniform`` / ``flat`` profile / degenerate input -> every layer gets
    ``base_rank`` == bitwise parity with legacy uniform injection.
  * Pure functions, no torch dependency, CPU unit-testable. The trainer supplies
    the matched target full-path names (from the live model) so this module turns
    ``names + base_rank + config`` into a ``{full_path_name: rank}`` map. The LoRA
    injector resolves a full-path key first, then a module-type key, then the
    uniform rank -- so a per-block map drives true per-layer rank without any
    forward-math change and stays bitwise-parity when absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SUPPORTED_FG_LORA_RANK_POLICIES = {"uniform", "coupled_prune", "orthogonal_redistribute"}
SUPPORTED_FG_LORA_RANK_PROFILES = {"center_peak", "ascending", "descending", "flat"}

# Common DiT / UNet block-index segment names; the captured integer is the depth.
_BLOCK_INDEX_RE = re.compile(
    r"(?:^|\.)(?:blocks|layers|transformer_blocks|single_blocks|double_blocks|"
    r"down_blocks|up_blocks|block)\.(\d+)(?:\.|$)"
)


@dataclass(frozen=True)
class FgLoraRankPolicyConfig:
    """Config for the orthogonal/profiled per-layer rank reallocation."""

    policy: str = "uniform"
    min_rank: int = 1
    max_rank: int = 64
    profile: str = "center_peak"
    conserve_budget: bool = True

    def normalized(self) -> "FgLoraRankPolicyConfig":
        policy = str(self.policy or "uniform").strip().lower().replace("-", "_")
        if policy not in SUPPORTED_FG_LORA_RANK_POLICIES:
            policy = "uniform"
        profile = str(self.profile or "center_peak").strip().lower().replace("-", "_")
        if profile not in SUPPORTED_FG_LORA_RANK_PROFILES:
            profile = "center_peak"
        min_rank = max(int(self.min_rank or 1), 1)
        max_rank = max(int(self.max_rank or min_rank), min_rank)
        return FgLoraRankPolicyConfig(
            policy=policy, min_rank=min_rank, max_rank=max_rank,
            profile=profile, conserve_budget=bool(self.conserve_budget),
        )


def parse_block_depth(name: str) -> int | None:
    """Parse the block/layer index out of a module path, or None if absent."""
    match = _BLOCK_INDEX_RE.search(name or "")
    return int(match.group(1)) if match else None


def select_target_full_paths(
    module_names: Iterable[str],
    target_names: Sequence[str],
    exclude_substrings: Iterable[str] = (),
) -> list[str]:
    """Mirror the LoRA injector's match rule on names only (no torch needed).

    A module matches when its leaf segment is a target, or any target string is a
    substring of the full path (covers dotted targets like ``cross_attn.v_proj``).
    ``exclude_substrings`` drops frozen sibling subtrees (e.g. ``llm_adapter``),
    exactly as the injector does, so the produced keys line up with the names the
    injector will see.
    """
    targets = [str(t) for t in target_names if str(t).strip()]
    excludes = tuple(str(s) for s in exclude_substrings if str(s).strip())
    out: list[str] = []
    for raw in module_names:
        name = str(raw)
        if excludes and any(s in name for s in excludes):
            continue
        leaf = name.split(".")[-1]
        if leaf in targets or any(t in name for t in targets):
            out.append(name)
    return out


def _profile_weight(profile: str, pos: float) -> float:
    """Map a depth position in [0,1] to a strictly-positive profile weight."""
    if profile == "flat":
        return 1.0
    if profile == "ascending":
        return 0.1 + 0.9 * pos
    if profile == "descending":
        return 0.1 + 0.9 * (1.0 - pos)
    # center_peak: triangular, peak at the middle depth.
    return 0.1 + 0.9 * (1.0 - abs(2.0 * pos - 1.0))


def _depth_positions(names: Sequence[str]) -> dict[str, float]:
    """Position each name in [0,1] by its parsed block index.

    Names without a parseable index get a neutral 0.5 so they sit mid-budget.
    """
    parsed = {name: parse_block_depth(name) for name in names}
    valid = [depth for depth in parsed.values() if depth is not None]
    if not valid:
        return {name: 0.5 for name in names}
    low, high = min(valid), max(valid)
    span = (high - low) or 1
    return {
        name: (0.5 if depth is None else (depth - low) / span)
        for name, depth in parsed.items()
    }


def build_orthogonal_rank_map(
    target_names: Sequence[str],
    base_rank: int,
    config: FgLoraRankPolicyConfig | Mapping | None = None,
) -> dict[str, int]:
    """All layers kept; per-layer rank reallocated by the depth profile.

    ``conserve_budget`` keeps ``sum(rank) ~= len(names) * base_rank`` (a pure
    redistribution of the uniform budget, modulo integer rounding and clipping);
    otherwise rank is mapped into ``[min_rank, max_rank]`` by the normalized
    profile weight. ``flat`` profile (or degenerate input) returns ``base_rank``
    for every layer -> bitwise parity with uniform.
    """
    cfg = _config(config)
    names = [str(n) for n in target_names if str(n).strip()]
    if not names:
        return {}
    base = max(int(base_rank or 1), 1)
    if cfg.profile == "flat":
        return {name: _clip(base, cfg) for name in names}

    positions = _depth_positions(names)
    weights = {n: max(_profile_weight(cfg.profile, positions[n]), 1e-6) for n in names}
    return _allocate(names, weights, base, cfg)


def build_profiled_rank_map(
    target_names: Sequence[str],
    base_rank: int,
    importance: Mapping[str, float],
    config: FgLoraRankPolicyConfig | Mapping | None = None,
) -> dict[str, int]:
    """All layers kept; rank scaled by per-layer importance from a profile.

    ``importance`` maps a full path OR a module-type suffix to a score. Layers
    with missing/zero importance fall back to the mean score so unseen layers stay
    near ``base_rank`` instead of collapsing to ``min_rank``.
    """
    cfg = _config(config)
    names = [str(n) for n in target_names if str(n).strip()]
    if not names:
        return {}
    base = max(int(base_rank or 1), 1)
    scored = {name: _importance_for(name, importance) for name in names}
    positive = [score for score in scored.values() if score > 0.0]
    if not positive:
        return {name: _clip(base, cfg) for name in names}
    mean = sum(positive) / len(positive)
    weights = {name: (score if score > 0.0 else mean) for name, score in scored.items()}
    return _allocate(names, weights, base, cfg)


def _allocate(
    names: Sequence[str],
    weights: Mapping[str, float],
    base: int,
    cfg: FgLoraRankPolicyConfig,
) -> dict[str, int]:
    """Turn positive per-name weights into an integer rank map."""
    total = sum(weights.values()) or 1.0
    if cfg.conserve_budget:
        budget = base * len(names)
        return {name: _clip(round(budget * weights[name] / total), cfg) for name in names}
    peak = max(weights.values()) or 1.0
    span = cfg.max_rank - cfg.min_rank
    return {
        name: _clip(round(cfg.min_rank + span * (weights[name] / peak)), cfg)
        for name in names
    }


def _importance_for(name: str, importance: Mapping[str, float]) -> float:
    if name in importance:
        return max(float(importance[name] or 0.0), 0.0)
    leaf = name.split(".")[-1]
    for key, value in importance.items():
        key = str(key)
        if name.endswith(key) or leaf == key:
            return max(float(value or 0.0), 0.0)
    return 0.0


def _clip(rank: int, cfg: FgLoraRankPolicyConfig) -> int:
    return min(max(int(rank), cfg.min_rank), cfg.max_rank)


def _config(config: FgLoraRankPolicyConfig | Mapping | None) -> FgLoraRankPolicyConfig:
    if isinstance(config, Mapping):
        return FgLoraRankPolicyConfig(**config).normalized()
    return (config or FgLoraRankPolicyConfig()).normalized()


__all__ = [
    "FgLoraRankPolicyConfig",
    "SUPPORTED_FG_LORA_RANK_POLICIES",
    "SUPPORTED_FG_LORA_RANK_PROFILES",
    "build_orthogonal_rank_map",
    "build_profiled_rank_map",
    "parse_block_depth",
    "select_target_full_paths",
]
