"""Compile target detection for route-specific training cores.

This detector is intentionally read-only.  It reports likely stable compile
targets without mutating the model, so later per-block/full-core compile code
can make decisions from a visible contract instead of guessing at attribute
names in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CompileTargetCandidate:
    route: str
    path: str
    target_type: str
    eligible: bool
    reason: str = ""

    def log_line(self) -> str:
        status = "yes" if self.eligible else "no"
        suffix = f" reason=\"{self.reason}\"" if self.reason else ""
        return (
            "[compile-target] "
            f"route={self.route} target={self.path} type={self.target_type} eligible={status}{suffix}"
        )


def _has_dynamic_hooks(obj: Any) -> bool:
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"):
        hooks = getattr(obj, name, None)
        try:
            if hooks and len(hooks) > 0:
                return True
        except Exception:
            if hooks:
                return True
    return False


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _candidate_for_target(route: str, path: str, target: Any, target_type: str) -> CompileTargetCandidate:
    if target is None:
        return CompileTargetCandidate(route, path, target_type, False, "missing target")
    if _has_dynamic_hooks(target):
        return CompileTargetCandidate(route, path, target_type, False, "dynamic hooks detected")
    if not callable(target):
        return CompileTargetCandidate(route, path, target_type, False, "target is not callable")
    return CompileTargetCandidate(route, path, target_type, True)


def _normalize_target_strategy(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    return normalized if normalized in {"auto", "block", "inner_forward"} else "auto"


def _resolve_inner_forward_target(block: Any) -> tuple[str, Any] | tuple[None, None]:
    for attr in ("_forward_impl", "_forward"):
        target = getattr(block, attr, None)
        if callable(target):
            return attr, target
    return None, None


def _iter_block_candidates(
    route: str,
    model: Any,
    *,
    collection_name: str,
    prefer_inner_forward: bool = True,
    target_strategy: str = "auto",
) -> Iterable[CompileTargetCandidate]:
    blocks = model
    for part in collection_name.split("."):
        blocks = getattr(blocks, part, None)
        if blocks is None:
            break
    if blocks is None and hasattr(model, "unet"):
        blocks = getattr(model, "unet")
        for part in collection_name.split("."):
            blocks = getattr(blocks, part, None)
            if blocks is None:
                break
    if blocks is None or not _is_sequence(blocks):
        return
    normalized_strategy = _normalize_target_strategy(target_strategy)
    for index in range(len(blocks)):
        block = blocks[index]
        if prefer_inner_forward and normalized_strategy in {"auto", "inner_forward"}:
            attr, target = _resolve_inner_forward_target(block)
            if attr is not None:
                yield _candidate_for_target(
                    route,
                    f"{collection_name}.{index}.{attr}",
                    target,
                    "inner_forward",
                )
                continue
            if normalized_strategy == "inner_forward":
                yield CompileTargetCandidate(
                    route,
                    f"{collection_name}.{index}._forward_impl",
                    "inner_forward",
                    False,
                    "missing inner forward target",
                )
                continue
        yield _candidate_for_target(route, f"{collection_name}.{index}", block, "block")


def detect_compile_targets(
    model: Any,
    *,
    route: str,
    target_strategy: str = "auto",
) -> list[CompileTargetCandidate]:
    """Return route-specific compile candidates without mutating ``model``."""

    route_name = str(route or "sdxl").strip().lower()
    normalized_strategy = _normalize_target_strategy(target_strategy)
    candidates: list[CompileTargetCandidate] = []

    if route_name == "anima":
        run_blocks = getattr(model, "_run_blocks", None)
        if run_blocks is None and hasattr(model, "transformer"):
            run_blocks = getattr(getattr(model, "transformer"), "_run_blocks", None)
        if run_blocks is None and hasattr(model, "unet"):
            run_blocks = getattr(getattr(model, "unet"), "_run_blocks", None)
        if run_blocks is not None:
            candidates.append(_candidate_for_target(route_name, "_run_blocks", run_blocks, "full_core"))
        candidates.extend(
            _iter_block_candidates(
                route_name,
                model,
                collection_name="blocks",
                target_strategy=normalized_strategy,
            )
        )
        candidates.extend(
            _iter_block_candidates(
                route_name,
                model,
                collection_name="net.blocks",
                target_strategy=normalized_strategy,
            )
        )
        return candidates

    if route_name == "newbie":
        candidates.extend(
            _iter_block_candidates(
                route_name,
                model,
                collection_name="transformer_blocks",
                target_strategy=normalized_strategy,
            )
        )
        candidates.extend(
            _iter_block_candidates(
                route_name,
                model,
                collection_name="blocks",
                target_strategy=normalized_strategy,
            )
        )
        return candidates

    if route_name == "flux":
        transformer = getattr(model, "transformer", model)
        candidates.extend(
            _iter_block_candidates(
                route_name,
                transformer,
                collection_name="transformer_blocks",
                prefer_inner_forward=False,
                target_strategy="block",
            )
        )
        candidates.extend(
            _iter_block_candidates(
                route_name,
                transformer,
                collection_name="single_transformer_blocks",
                prefer_inner_forward=False,
                target_strategy="block",
            )
        )
        return candidates

    if route_name in {"sdxl", "sd15"}:
        unet = getattr(model, "unet", model)
        candidates.extend(
            _iter_block_candidates(
                route_name,
                unet,
                collection_name="down_blocks",
                prefer_inner_forward=False,
                target_strategy="block",
            )
        )
        mid_block = getattr(unet, "mid_block", None)
        if mid_block is not None:
            candidates.append(_candidate_for_target(route_name, "mid_block", mid_block, "block"))
        candidates.extend(
            _iter_block_candidates(
                route_name,
                unet,
                collection_name="up_blocks",
                prefer_inner_forward=False,
                target_strategy="block",
            )
        )
        return candidates

    return candidates


def compile_target_log_lines(candidates: Iterable[CompileTargetCandidate]) -> list[str]:
    return [candidate.log_line() for candidate in candidates]
