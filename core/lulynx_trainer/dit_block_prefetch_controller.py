"""Forward-order async prefetch controller for CPU-pinned DiT Linear weights.

Split out of ``dit_residency_planner.py``: the planner decides hot/cold
residency, this module executes the runtime prefetch.  It owns the opt-in
``prefetch_mode`` axis (``original`` = today's fixed-depth forward prefetch;
``adaptive`` = blockskip-aware + online depth-adaptive).  The adaptive policy
only changes WHICH/HOW-DEEP blocks are staged H2D, never the forward math.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .dit_residency_planner import DitResidencyPlan, DitResidencyUnit


# Stream-offload prefetch prediction modes (opt-in axis, orthogonal to the
# residency mode). "original" = today's fixed-depth forward-order prefetch
# (default; behaviourally unchanged). "adaptive" = blockskip-aware + online
# depth-adaptive prefetch — same H2D math, only WHICH/HOW-DEEP blocks are
# prefetched changes, so it never alters the forward output.
VALID_DIT_PREFETCH_MODES = {"original", "adaptive"}
# A prefetch "miss" = a CPU-pinned Linear ran before its async H2D copy was
# ready (a stall). Above this per-forward miss-rate, adaptive mode deepens the
# prefetch window by one block. VRAM-headroom upper bound keeps staging bounded.
ADAPTIVE_PREFETCH_MISS_RATE_THRESHOLD = 0.05
ADAPTIVE_PREFETCH_MAX_DEPTH = 4


def normalize_dit_prefetch_mode(value: Any) -> str:
    mode = str(value or "original").strip().lower().replace("-", "_")
    aliases = {"": "original", "off": "original", "fixed": "original",
               "static": "original", "auto": "adaptive", "adapt": "adaptive"}
    mode = aliases.get(mode, mode)
    return mode if mode in VALID_DIT_PREFETCH_MODES else "original"


class DitBlockPrefetchController:
    """Forward-order async prefetch for CPU-pinned frozen DiT Linear weights."""

    def __init__(
        self,
        blocks: list[nn.Module],
        plan: DitResidencyPlan,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None = None,
        depth: int = 1,
        prefetch_mode: str = "original",
        install_hooks: bool = True,
    ) -> None:
        self.blocks = list(blocks)
        self.plan = plan
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.depth = max(int(depth or 0), 0)
        self.policy = normalize_dit_prefetch_mode(prefetch_mode)
        self.adaptive = self.policy == "adaptive"
        self.enabled = False
        self.reason = "disabled"
        self._handles: list[Any] = []
        self._stream: torch.cuda.Stream | None = None
        self._module_to_index: dict[int, int] = {id(block): index for index, block in enumerate(self.blocks)}
        self._units_by_block: dict[int, list[DitResidencyUnit]] = {}
        self._before_calls = 0
        self._prefetch_calls = 0
        self._submitted = 0
        self._skipped = 0
        self._errors = 0
        self._last_error = ""
        # Adaptive-mode state (inert when policy == "original").
        self._adaptive_depth = max(self.depth, 1)
        self._max_depth = max(self.depth, ADAPTIVE_PREFETCH_MAX_DEPTH)
        self._skipped_blockskip = 0
        self._last_consumed = 0
        self._last_missed = 0
        self._last_block_index = -1
        self._depth_grew = 0

        for unit in plan.units:
            if unit.cpu_pinned and unit.sparse_decision != "cold_on_demand":
                self._units_by_block.setdefault(int(unit.block_index), []).append(unit)

        if self.depth <= 0:
            self.reason = "prefetch depth is 0"
            return
        if self.device is None or self.device.type != "cuda":
            self.reason = "prefetch requires a CUDA training device"
            return
        if not torch.cuda.is_available():
            self.reason = "CUDA is not available"
            return
        if not self._units_by_block:
            self.reason = "no CPU-pinned Linear units are planned"
            return
        try:
            self._stream = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            self.reason = f"failed to create CUDA prefetch stream: {type(exc).__name__}: {exc}"
            self._errors += 1
            self._last_error = self.reason
            return
        self.enabled = True
        self.reason = "active"
        if install_hooks:
            self.install_hooks()

    @property
    def planned_block_count(self) -> int:
        return len(self._units_by_block)

    @property
    def planned_linear_count(self) -> int:
        return sum(len(units) for units in self._units_by_block.values())

    def install_hooks(self) -> None:
        if not self.enabled or self._handles:
            return

        def _make_hook(block_index: int) -> Any:
            def _hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
                self.before_block(block_index, *args)

            return _hook

        for index, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(_make_hook(index)))

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        for units in self._units_by_block.values():
            for unit in units:
                unit.module.clear_cpu_pinned_prefetch()

    def before_block_module(self, block: nn.Module, *inputs: Any) -> None:
        block_index = self._module_to_index.get(id(block))
        if block_index is not None:
            self.before_block(block_index, *inputs)

    def before_block(self, block_index: int, *inputs: Any) -> None:
        if not self.enabled:
            return
        self._before_calls += 1
        block_index = int(block_index)
        if self.adaptive and block_index <= self._last_block_index:
            # Wrapped back to the start of a new forward -> reconsider depth from
            # the previous forward's miss feedback.
            self._adapt_depth()
        self._last_block_index = block_index
        device, dtype = self._resolve_target(inputs)
        if device is None or device.type != "cuda":
            self._skipped += 1
            return
        for target in self._prefetch_targets(block_index):
            self._prefetch_block(target, device=device, dtype=dtype)

    def _prefetch_targets(self, block_index: int) -> list[int]:
        """Block indices to stage from ``block_index`` (pure: no CUDA, no IO).

        ``original`` -> ``[i .. i+depth]``.  ``adaptive`` -> ``[i .. i+adaptive_depth]``
        minus any block a live blockskip seam will identity-skip this forward
        (its Linears never run, so staging it wastes PCIe).  Excluded blocks bump
        ``_skipped_blockskip``.  Kept side-effect-free of CUDA so it is unit
        testable on CPU; the actual H2D copy stays in ``_prefetch_block``.
        """
        eff_depth = self._adaptive_depth if self.adaptive else self.depth
        skip_fn = self._blockskip_predicate() if self.adaptive else None
        targets: list[int] = []
        for offset in range(0, eff_depth + 1):
            target = block_index + offset
            if skip_fn is not None and skip_fn(target):
                self._skipped_blockskip += 1
                continue
            targets.append(target)
        return targets

    def _blockskip_predicate(self):
        """Live blockskip predicate for adaptive prefetch.

        Reads the SAME ContextVar-published compute-reducer seam that
        ``anima_native_dit._run_blocks`` consults, so a block that will be
        identity-skipped this forward is not prefetched. Returns ``None`` when no
        blockskip seam is active (then adaptive prefetch behaves like original
        for coverage, just with the adaptive depth). Degrades safely on import
        failure.
        """
        try:
            from .dit_compute_reducer_seam import get_active_compute_reducer_seam
        except Exception:
            return None
        seam = get_active_compute_reducer_seam()
        if seam is None or getattr(seam, "strategy", "") != "blockskip":
            return None
        return lambda idx: bool(seam.should_skip_block(idx))

    def _aggregate_consume_stats(self) -> tuple[int, int]:
        consumed = missed = 0
        for units in self._units_by_block.values():
            for unit in units:
                stats = unit.module.get_cpu_pinned_prefetch_stats()
                consumed += int(stats.get("consumed", 0))
                missed += int(stats.get("missed", 0))
        return consumed, missed

    def _adapt_depth(self) -> None:
        """Deepen the prefetch window when the previous forward saw misses.

        A miss = a CPU-pinned Linear ran before its async H2D copy was ready, so
        it stalled on a synchronous copy. A high per-forward miss-rate means the
        prefetch is too shallow/late -> grow depth by one (bounded by
        ``_max_depth`` for VRAM headroom). Zero new H2D math; only how far ahead
        the controller stages.
        """
        consumed, missed = self._aggregate_consume_stats()
        self._adapt_depth_from(consumed, missed)

    def _adapt_depth_from(self, consumed: int, missed: int) -> None:
        """Pure depth-adaptation step from cumulative ``consumed``/``missed``.

        Splits the previous forward's delta off the running totals and grows
        ``_adaptive_depth`` by one when its miss-rate exceeds the threshold and
        headroom remains. No CUDA/IO, so the policy is unit testable on CPU.
        """
        d_consumed = consumed - self._last_consumed
        d_missed = missed - self._last_missed
        self._last_consumed = consumed
        self._last_missed = missed
        total = d_consumed + d_missed
        if total <= 0:
            return
        if (d_missed / total) > ADAPTIVE_PREFETCH_MISS_RATE_THRESHOLD and self._adaptive_depth < self._max_depth:
            self._adaptive_depth += 1
            self._depth_grew += 1

    def _resolve_target(self, inputs: Any) -> tuple[torch.device | None, torch.dtype | None]:
        tensor = self._first_tensor(inputs)
        if tensor is not None and tensor.device.type == "cuda":
            return tensor.device, tensor.dtype if tensor.is_floating_point() else self.dtype
        return self.device, self.dtype

    def _first_tensor(self, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = self._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._first_tensor(item)
                if found is not None:
                    return found
        return None

    def _prefetch_block(self, block_index: int, *, device: torch.device, dtype: torch.dtype | None) -> None:
        units = self._units_by_block.get(int(block_index))
        if not units:
            return
        for unit in units:
            self._prefetch_calls += 1
            try:
                submitted = unit.module.prefetch_cpu_pinned_residency(
                    device=device,
                    dtype=dtype or self.dtype,
                    stream=self._stream,
                )
                if submitted:
                    self._submitted += 1
                else:
                    self._skipped += 1
            except Exception as exc:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"

    def as_dict(self) -> dict[str, Any]:
        submitted = consumed = missed = errors = pending = 0
        for units in self._units_by_block.values():
            for unit in units:
                stats = unit.module.get_cpu_pinned_prefetch_stats()
                submitted += int(stats.get("submitted", 0))
                consumed += int(stats.get("consumed", 0))
                missed += int(stats.get("missed", 0))
                errors += int(stats.get("errors", 0))
                pending += int(stats.get("pending", 0))
        return {
            "enabled": bool(self.enabled),
            "reason": self.reason,
            "family": self.plan.family,
            "mode": self.plan.mode,
            "policy": self.policy,
            "depth": int(self.depth),
            "adaptive_depth": int(self._adaptive_depth),
            "depth_grew": int(self._depth_grew),
            "skipped_blockskip": int(self._skipped_blockskip),
            "device": str(self.device or ""),
            "planned_block_count": int(self.planned_block_count),
            "planned_linear_count": int(self.planned_linear_count),
            "before_block_calls": int(self._before_calls),
            "prefetch_calls": int(self._prefetch_calls),
            "submitted": int(submitted),
            "submitted_controller": int(self._submitted),
            "consumed": int(consumed),
            "missed": int(missed),
            "skipped": int(self._skipped),
            "errors": int(errors + self._errors),
            "pending": int(pending),
            "last_error": self._last_error,
        }


def clear_dit_block_prefetch_controller(model: nn.Module) -> None:
    controller = getattr(model, "_lulynx_dit_prefetch_controller", None)
    if controller is not None and hasattr(controller, "close"):
        try:
            controller.close()
        except Exception:
            pass
    try:
        setattr(model, "_lulynx_dit_prefetch_controller", None)
    except Exception:
        pass


def install_dit_block_prefetch_controller(
    model: nn.Module,
    blocks: list[nn.Module],
    plan: DitResidencyPlan,
    *,
    enabled: bool,
    depth: int,
    device: torch.device | str | None,
    dtype: torch.dtype | None = None,
    install_hooks: bool = True,
    prefetch_mode: str = "original",
) -> DitBlockPrefetchController | None:
    clear_dit_block_prefetch_controller(model)
    if not enabled:
        return None
    controller = DitBlockPrefetchController(
        blocks,
        plan,
        device=device,
        dtype=dtype,
        depth=depth,
        install_hooks=install_hooks,
        prefetch_mode=prefetch_mode,
    )
    try:
        setattr(model, "_lulynx_dit_prefetch_controller", controller)
    except Exception:
        controller.close()
        return None
    return controller

