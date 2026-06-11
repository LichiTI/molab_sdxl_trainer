"""Read-only PCIe cache candidate profiler.

This module only ranks transfer-cache candidates. It never mutates modules or
training tensors, so it is safe to leave wired into reports before the real
cache path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class PcieCacheCandidate:
    name: str
    block_index: int
    parameter_count: int
    transfer_mb: float
    format: str
    packed: bool
    reason: str
    sparse_decision: str
    prefetch_submitted: int
    prefetch_consumed: int
    prefetch_missed: int
    prefetch_errors: int
    pack_errors: int
    decode_errors: int
    score: float
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "block_index": int(self.block_index),
            "parameter_count": int(self.parameter_count),
            "transfer_mb": round(float(self.transfer_mb), 3),
            "format": self.format,
            "packed": bool(self.packed),
            "reason": self.reason,
            "sparse_decision": self.sparse_decision,
            "prefetch_submitted": int(self.prefetch_submitted),
            "prefetch_consumed": int(self.prefetch_consumed),
            "prefetch_missed": int(self.prefetch_missed),
            "prefetch_errors": int(self.prefetch_errors),
            "pack_errors": int(self.pack_errors),
            "decode_errors": int(self.decode_errors),
            "score": round(float(self.score), 3),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class PcieCacheProfile:
    enabled: bool
    family: str
    mode: str
    scope: str
    candidate_count: int
    total_transfer_mb: float
    estimated_cache_mb: float
    high_value_count: int
    medium_value_count: int
    low_value_count: int
    candidates: tuple[PcieCacheCandidate, ...]
    notes: tuple[str, ...]

    def as_dict(self, *, sample_limit: int = 24) -> dict[str, Any]:
        sample = list(self.candidates[: max(int(sample_limit or 0), 0)])
        submitted = sum(int(item.prefetch_submitted) for item in self.candidates)
        consumed = sum(int(item.prefetch_consumed) for item in self.candidates)
        missed = sum(int(item.prefetch_missed) for item in self.candidates)
        errors = sum(
            int(item.prefetch_errors) + int(item.pack_errors) + int(item.decode_errors)
            for item in self.candidates
        )
        top_blocks = _top_block_summary(self.candidates)
        next_action = _next_action(
            enabled=self.enabled,
            candidate_count=self.candidate_count,
            high_value_count=self.high_value_count,
            errors=errors,
            missed=missed,
            total_transfer_mb=self.total_transfer_mb,
        )
        return {
            "enabled": bool(self.enabled),
            "family": self.family,
            "mode": self.mode,
            "scope": self.scope,
            "candidate_count": int(self.candidate_count),
            "total_transfer_mb": round(float(self.total_transfer_mb), 3),
            "estimated_cache_mb": round(float(self.estimated_cache_mb), 3),
            "high_value_count": int(self.high_value_count),
            "medium_value_count": int(self.medium_value_count),
            "low_value_count": int(self.low_value_count),
            "prefetch_submitted_total": int(submitted),
            "prefetch_consumed_total": int(consumed),
            "prefetch_missed_total": int(missed),
            "error_count": int(errors),
            "top_blocks": top_blocks,
            "next_action": next_action,
            "summary_text": _summary_text(
                enabled=self.enabled,
                family=self.family,
                mode=self.mode,
                candidate_count=self.candidate_count,
                high_value_count=self.high_value_count,
                medium_value_count=self.medium_value_count,
                total_transfer_mb=self.total_transfer_mb,
                estimated_cache_mb=self.estimated_cache_mb,
                missed=missed,
                errors=errors,
                next_action=next_action,
            ),
            "sample": [item.as_dict() for item in sample],
            "notes": list(self.notes),
        }


def _top_block_summary(candidates: Iterable[PcieCacheCandidate], *, limit: int = 6) -> list[dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    for item in candidates:
        block_index = int(item.block_index)
        if block_index < 0:
            continue
        current = blocks.setdefault(
            block_index,
            {"block_index": block_index, "candidate_count": 0, "transfer_mb": 0.0, "high_value_count": 0, "score": 0.0},
        )
        current["candidate_count"] += 1
        current["transfer_mb"] += float(item.transfer_mb)
        current["score"] += float(item.score)
        if item.recommendation == "high_value_cache_candidate":
            current["high_value_count"] += 1
    ranked = sorted(blocks.values(), key=lambda row: (-float(row["score"]), -float(row["transfer_mb"]), int(row["block_index"])))
    result = []
    for row in ranked[: max(int(limit or 0), 0)]:
        result.append(
            {
                "block_index": int(row["block_index"]),
                "candidate_count": int(row["candidate_count"]),
                "high_value_count": int(row["high_value_count"]),
                "transfer_mb": round(float(row["transfer_mb"]), 3),
                "score": round(float(row["score"]), 3),
            }
        )
    return result


def _next_action(
    *,
    enabled: bool,
    candidate_count: int,
    high_value_count: int,
    errors: int,
    missed: int,
    total_transfer_mb: float,
) -> str:
    if not enabled:
        return "disabled"
    if errors > 0:
        return "fix_transfer_errors_before_cache"
    if candidate_count <= 0 or total_transfer_mb <= 0:
        return "no_cache_candidate"
    observed = max(int(missed), 0)
    # When block-level prefetch already covers the hot transfer path, cache_v0
    # can still be A/B tested but should not be presented as the next obvious
    # recommendation. It costs VRAM and our long test showed near-neutral speed.
    if high_value_count > 0 and observed == 0:
        return "prefetch_covered_cache_v0_ab_only"
    if high_value_count >= 8 and total_transfer_mb >= 128:
        return "cache_v0_manual_candidate"
    if missed > 0 and high_value_count > 0:
        return "observe_more_steps"
    return "keep_observing"


def _summary_text(
    *,
    enabled: bool,
    family: str,
    mode: str,
    candidate_count: int,
    high_value_count: int,
    medium_value_count: int,
    total_transfer_mb: float,
    estimated_cache_mb: float,
    missed: int,
    errors: int,
    next_action: str,
) -> str:
    if not enabled:
        return f"PCIe Delta/Cache observe disabled for {family or 'unknown'}."
    return (
        "PCIe Delta/Cache observe: "
        f"family={family or 'unknown'} mode={mode or 'unknown'} "
        f"candidates={int(candidate_count)} high={int(high_value_count)} medium={int(medium_value_count)} "
        f"transfer={float(total_transfer_mb):.1f}MB estimated_cache={float(estimated_cache_mb):.1f}MB "
        f"prefetch_missed={int(missed)} errors={int(errors)} next={next_action}"
    )


def _module_stats(module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    transfer = {}
    prefetch = {}
    getter = getattr(module, "get_transfer_format_stats", None)
    if callable(getter):
        try:
            transfer = dict(getter())
        except Exception as exc:
            transfer = {"error": f"{type(exc).__name__}: {exc}"}
    getter = getattr(module, "get_cpu_pinned_prefetch_stats", None)
    if callable(getter):
        try:
            prefetch = dict(getter())
        except Exception as exc:
            prefetch = {"errors": 1, "error": f"{type(exc).__name__}: {exc}"}
    return transfer, prefetch


def _recommendation(score: float, *, errors: int, transfer_mb: float) -> str:
    if errors > 0:
        return "skip_errors"
    if transfer_mb <= 0:
        return "skip_no_transfer"
    if score >= 24.0:
        return "high_value_cache_candidate"
    if score >= 8.0:
        return "medium_value_observe"
    return "low_value_observe"


def _score_candidate(
    *,
    transfer_mb: float,
    sparse_decision: str,
    prefetch_consumed: int,
    prefetch_missed: int,
    errors: int,
) -> float:
    if errors > 0:
        return 0.0
    observations = max(int(prefetch_consumed) + int(prefetch_missed), 0)
    reuse_boost = 1.0 + min(observations, 64) / 16.0
    miss_rate = float(prefetch_missed) / max(observations, 1)
    sparse_boost = 1.25 if sparse_decision == "cold_on_demand" else 1.0
    return float(transfer_mb) * reuse_boost * sparse_boost + miss_rate * 8.0


def _build_profile(
    *,
    enabled: bool,
    family: str,
    mode: str,
    scope: str,
    candidates: list[PcieCacheCandidate],
) -> PcieCacheProfile:
    candidates.sort(key=lambda item: (-item.score, -item.transfer_mb, item.block_index, item.name))
    high = sum(1 for item in candidates if item.recommendation == "high_value_cache_candidate")
    medium = sum(1 for item in candidates if item.recommendation == "medium_value_observe")
    low = sum(1 for item in candidates if item.recommendation == "low_value_observe")
    total_transfer = sum(float(item.transfer_mb) for item in candidates)
    notes = [
        "read-only profiler; no cache is allocated and no tensor path is changed",
        "high score means repeated, large, stable CPU-pinned frozen Linear transfers may be worth manual cache_v0 testing",
        "cache_v0 is most useful when streaming prefetch coverage is poor/off; perfect prefetch coverage often leaves little speed upside",
        "when next_action=prefetch_covered_cache_v0_ab_only, cache_v0 should remain an A/B test rather than a recommended default",
        "estimated_cache_mb is a planning hint, not current VRAM usage",
    ]
    return PcieCacheProfile(
        enabled=bool(enabled),
        family=str(family or ""),
        mode=str(mode or ""),
        scope=str(scope or ""),
        candidate_count=len(candidates),
        total_transfer_mb=float(total_transfer),
        estimated_cache_mb=float(total_transfer),
        high_value_count=high,
        medium_value_count=medium,
        low_value_count=low,
        candidates=tuple(candidates),
        notes=tuple(notes),
    )


def build_dit_pcie_cache_profile(
    plan: Any,
    *,
    enabled: bool,
    family: str,
    mode: str,
) -> PcieCacheProfile:
    candidates: list[PcieCacheCandidate] = []
    if not enabled:
        return _build_profile(enabled=False, family=family, mode=mode, scope="dit_blocks", candidates=[])
    for unit in getattr(plan, "units", []) or []:
        if not bool(getattr(unit, "cpu_pinned", False)):
            continue
        module = getattr(unit, "module", None)
        if module is None:
            continue
        transfer, prefetch = _module_stats(module)
        transfer_mb = float(transfer.get("transfer_mb", 0.0) or 0.0)
        prefetch_consumed = int(prefetch.get("consumed", 0) or 0)
        prefetch_missed = int(prefetch.get("missed", 0) or 0)
        errors = int(transfer.get("pack_errors", 0) or 0) + int(transfer.get("decode_errors", 0) or 0) + int(prefetch.get("errors", 0) or 0)
        sparse_decision = str(getattr(unit, "sparse_decision", "") or "")
        score = _score_candidate(
            transfer_mb=transfer_mb,
            sparse_decision=sparse_decision,
            prefetch_consumed=prefetch_consumed,
            prefetch_missed=prefetch_missed,
            errors=errors,
        )
        candidates.append(
            PcieCacheCandidate(
                name=str(getattr(unit, "module_name", "")),
                block_index=int(getattr(unit, "block_index", -1)),
                parameter_count=int(getattr(unit, "parameter_count", 0) or 0),
                transfer_mb=transfer_mb,
                format=str(transfer.get("format", "")),
                packed=bool(transfer.get("packed", False)),
                reason=str(getattr(unit, "reason", "")),
                sparse_decision=sparse_decision,
                prefetch_submitted=int(prefetch.get("submitted", 0) or 0),
                prefetch_consumed=prefetch_consumed,
                prefetch_missed=prefetch_missed,
                prefetch_errors=int(prefetch.get("errors", 0) or 0),
                pack_errors=int(transfer.get("pack_errors", 0) or 0),
                decode_errors=int(transfer.get("decode_errors", 0) or 0),
                score=score,
                recommendation=_recommendation(score, errors=errors, transfer_mb=transfer_mb),
            )
        )
    return _build_profile(enabled=True, family=family, mode=mode, scope="dit_blocks", candidates=candidates)


def _infer_block_index(name: str) -> int:
    for pattern in (r"(?:^|\.)blocks\.(\d+)(?:\.|$)", r"(?:^|\.)layers\.(\d+)(?:\.|$)"):
        match = re.search(pattern, str(name or ""))
        if match:
            return int(match.group(1))
    return -1


def build_module_pcie_cache_profile(
    units: Iterable[tuple[str, Any, int] | tuple[str, Any, int, int]],
    *,
    enabled: bool,
    family: str,
    mode: str,
) -> PcieCacheProfile:
    candidates: list[PcieCacheCandidate] = []
    if not enabled:
        return _build_profile(enabled=False, family=family, mode=mode, scope="modules", candidates=[])
    for unit in units:
        if len(unit) >= 4:
            name, module, parameter_count, block_index = unit[:4]
        else:
            name, module, parameter_count = unit[:3]
            block_index = _infer_block_index(str(name))
        transfer, prefetch = _module_stats(module)
        transfer_mb = float(transfer.get("transfer_mb", 0.0) or 0.0)
        prefetch_consumed = int(prefetch.get("consumed", 0) or 0)
        prefetch_missed = int(prefetch.get("missed", 0) or 0)
        errors = int(transfer.get("pack_errors", 0) or 0) + int(transfer.get("decode_errors", 0) or 0) + int(prefetch.get("errors", 0) or 0)
        score = _score_candidate(
            transfer_mb=transfer_mb,
            sparse_decision="",
            prefetch_consumed=prefetch_consumed,
            prefetch_missed=prefetch_missed,
            errors=errors,
        )
        candidates.append(
            PcieCacheCandidate(
                name=str(name),
                block_index=int(block_index),
                parameter_count=int(parameter_count or 0),
                transfer_mb=transfer_mb,
                format=str(transfer.get("format", "")),
                packed=bool(transfer.get("packed", False)),
                reason="cpu_pinned",
                sparse_decision="",
                prefetch_submitted=int(prefetch.get("submitted", 0) or 0),
                prefetch_consumed=prefetch_consumed,
                prefetch_missed=prefetch_missed,
                prefetch_errors=int(prefetch.get("errors", 0) or 0),
                pack_errors=int(transfer.get("pack_errors", 0) or 0),
                decode_errors=int(transfer.get("decode_errors", 0) or 0),
                score=score,
                recommendation=_recommendation(score, errors=errors, transfer_mb=transfer_mb),
            )
        )
    return _build_profile(enabled=True, family=family, mode=mode, scope="modules", candidates=candidates)


def build_active_module_pcie_cache_profile(
    root: Any,
    *,
    enabled: bool,
    family: str,
    mode: str,
) -> PcieCacheProfile:
    """Build an observe-only profile from currently active managed modules."""

    units: list[tuple[str, Any, int, int]] = []
    if not enabled or root is None:
        return _build_profile(enabled=False, family=family, mode=mode, scope="modules", candidates=[])
    named_modules = getattr(root, "named_modules", None)
    if not callable(named_modules):
        return _build_profile(enabled=True, family=family, mode=mode, scope="modules", candidates=[])
    for name, module in named_modules():
        if not bool(getattr(module, "lulynx_weight_residency_active", False)):
            continue
        if not callable(getattr(module, "get_transfer_format_stats", None)):
            continue
        parameter_count = 0
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            try:
                parameter_count = sum(int(param.numel()) for param in parameters(recurse=False))
            except Exception:
                parameter_count = 0
        units.append((str(name), module, int(parameter_count), _infer_block_index(str(name))))
    return build_module_pcie_cache_profile(units, enabled=True, family=family, mode=mode)
