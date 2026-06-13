"""Opt-in torch profiler summary for Newbie backward autograd probes."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

MATMUL_EVENT_KEYS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear"}


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _shape_repr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_shape_repr(item) for item in value) + "]"
    return str(value)


def _event_row(event: Any) -> dict[str, Any]:
    self_cuda_us = _safe_float(
        getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0))
    )
    cuda_us = _safe_float(getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0)))
    self_cpu_us = _safe_float(getattr(event, "self_cpu_time_total", 0.0))
    cpu_us = _safe_float(getattr(event, "cpu_time_total", 0.0))
    return {
        "key": str(getattr(event, "key", "") or ""),
        "count": _safe_int(getattr(event, "count", 0)),
        "self_cuda_ms": round(self_cuda_us / 1000.0, 4),
        "cuda_ms": round(cuda_us / 1000.0, 4),
        "self_cpu_ms": round(self_cpu_us / 1000.0, 4),
        "cpu_ms": round(cpu_us / 1000.0, 4),
    }


def _sort_rows(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], bool]:
    has_cuda_time = any(float(row["self_cuda_ms"]) > 0.0 for row in rows)
    sort_key = "self_cuda_ms" if has_cuda_time else "self_cpu_ms"
    rows.sort(
        key=lambda row: (
            float(row.get(sort_key, 0.0) or 0.0),
            float(row.get("self_cpu_ms", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return sort_key, rows, has_cuda_time


def summarize_backward_profiler_events(events: Any, *, top_k: int = 12) -> dict[str, Any]:
    """Project torch profiler key averages into compact JSON-safe rows."""

    rows: list[dict[str, Any]] = []
    for event in events or ():
        key = str(getattr(event, "key", "") or "")
        if not key:
            continue
        rows.append(_event_row(event))

    sort_key, rows, has_cuda_time = _sort_rows(rows)
    limit = max(int(top_k or 0), 1)
    return {
        "report": "newbie_backward_op_profile_v0",
        "status": "profiled",
        "sort_key": sort_key,
        "event_count": len(rows),
        "top_k": limit,
        "top_ops": rows[:limit],
        "cuda_activity_available": has_cuda_time,
    }


def summarize_backward_profiler_shape_events(events: Any, *, top_k: int = 12) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for event in events or ():
        key = str(getattr(event, "key", "") or "")
        if not key:
            continue
        row = _event_row(event)
        row["input_shapes"] = _shape_repr(getattr(event, "input_shapes", None))
        rows.append(row)

    _, rows, _ = _sort_rows(rows)
    matmul_rows = [row for row in rows if row.get("key") in MATMUL_EVENT_KEYS]
    limit = max(int(top_k or 0), 1)
    return {
        "shape_group_count": len(rows),
        "top_shape_groups": rows[:limit],
        "top_matmul_shape_groups": matmul_rows[:limit],
    }


def profile_backward_autograd_call(
    backward_call: Callable[[], None],
    *,
    top_k: int = 12,
    record_shapes: bool = False,
) -> dict[str, Any]:
    """Run one backward call under torch.profiler and return a compact report."""

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    backward_started = False
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=bool(record_shapes),
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            backward_started = True
            backward_call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report = summarize_backward_profiler_events(profiler.key_averages(), top_k=top_k)
        report["record_shapes"] = bool(record_shapes)
        if bool(record_shapes):
            report.update(
                summarize_backward_profiler_shape_events(
                    profiler.key_averages(group_by_input_shape=True),
                    top_k=top_k,
                )
            )
        report["activities"] = ["cuda" if item == torch.profiler.ProfilerActivity.CUDA else "cpu" for item in activities]
        return report
    except Exception as exc:
        if backward_started:
            raise
        return {
            "report": "newbie_backward_op_profile_v0",
            "status": "profile_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "top_k": max(int(top_k or 0), 1),
            "record_shapes": bool(record_shapes),
            "top_ops": [],
            "event_count": 0,
            "cuda_activity_available": False,
        }


class NewbieModuleTimingProfiler:
    """Opt-in single-step module timing for Newbie compute-path attribution."""

    def __init__(self, root: Any, *, top_k: int = 12, max_modules: int = 256) -> None:
        self.root = root
        self.top_k = max(int(top_k or 0), 1)
        self.max_modules = max(int(max_modules or 0), 1)
        self.enabled = False
        self._handles: list[Any] = []
        self._active: dict[tuple[int, str], list[tuple[float, Any]]] = {}
        self._cuda_events: list[tuple[str, str, Any, Any]] = []
        self._groups: dict[str, dict[str, Any]] = {}
        self._module_count = 0

    def attach(self) -> "NewbieModuleTimingProfiler":
        named_modules = getattr(self.root, "named_modules", None)
        if not callable(named_modules):
            return self
        for name, module in named_modules():
            if not name:
                continue
            group = _classify_newbie_timing_module(name, module)
            if not group:
                continue
            if not _is_newbie_timing_leaf(module):
                continue
            self._ensure_group(group, name)
            self._handles.append(module.register_forward_pre_hook(self._make_start_hook(group, "forward")))
            self._handles.append(module.register_forward_hook(self._make_end_hook(group, "forward")))
            self._handles.append(module.register_full_backward_pre_hook(self._make_start_hook(group, "backward")))
            self._handles.append(module.register_full_backward_hook(self._make_end_hook(group, "backward")))
            self._module_count += 1
            if self._module_count >= self.max_modules:
                break
        self.enabled = bool(self._handles)
        return self

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        self.enabled = False

    def snapshot(self, *, step: int = 0) -> dict[str, Any]:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        for group, phase, start_event, end_event in self._cuda_events:
            try:
                elapsed = float(start_event.elapsed_time(end_event))
            except Exception:
                elapsed = 0.0
            self._groups[group][f"{phase}_cuda_ms"] += elapsed
        self._cuda_events = []
        rows = [self._row(group, stats) for group, stats in self._groups.items()]
        rows.sort(
            key=lambda row: (
                float(row.get("backward_cuda_ms", 0.0) or 0.0)
                + float(row.get("forward_cuda_ms", 0.0) or 0.0),
                float(row.get("backward_cpu_ms", 0.0) or 0.0)
                + float(row.get("forward_cpu_ms", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return {
            "report": "newbie_module_timing_profile_v0",
            "status": "profiled" if self._module_count else "no_modules_matched",
            "step": int(step or 0),
            "top_k": self.top_k,
            "tracked_module_count": self._module_count,
            "hook_count": len(self._handles),
            "cuda_activity_available": bool(torch.cuda.is_available()),
            "top_groups": rows[: self.top_k],
            "group_count": len(rows),
            "runtime_default_change": False,
            "probe_only": True,
        }

    def _make_start_hook(self, group: str, phase: str) -> Callable[..., None]:
        def _hook(module: Any, *args: Any) -> None:
            event = None
            if torch.cuda.is_available():
                try:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                except Exception:
                    event = None
            key = (id(module), phase)
            self._active.setdefault(key, []).append((time.perf_counter(), event))

        return _hook

    def _make_end_hook(self, group: str, phase: str) -> Callable[..., None]:
        def _hook(module: Any, *args: Any) -> None:
            key = (id(module), phase)
            stack = self._active.get(key)
            if not stack:
                return
            start_cpu, start_event = stack.pop()
            if not stack:
                self._active.pop(key, None)
            stats = self._ensure_group(group, "")
            stats[f"{phase}_count"] += 1
            stats[f"{phase}_cpu_ms"] += max((time.perf_counter() - start_cpu) * 1000.0, 0.0)
            if start_event is not None and torch.cuda.is_available():
                try:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    self._cuda_events.append((group, phase, start_event, end_event))
                except Exception:
                    pass

        return _hook

    def _ensure_group(self, group: str, module_name: str) -> dict[str, Any]:
        stats = self._groups.setdefault(
            group,
            {
                "group": group,
                "module_count": 0,
                "module_name_examples": [],
                "forward_count": 0,
                "backward_count": 0,
                "forward_cpu_ms": 0.0,
                "backward_cpu_ms": 0.0,
                "forward_cuda_ms": 0.0,
                "backward_cuda_ms": 0.0,
            },
        )
        if module_name:
            stats["module_count"] += 1
            examples = stats["module_name_examples"]
            if len(examples) < 5:
                examples.append(module_name)
        return stats

    @staticmethod
    def _row(group: str, stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "group": group,
            "module_count": _safe_int(stats.get("module_count")),
            "forward_count": _safe_int(stats.get("forward_count")),
            "backward_count": _safe_int(stats.get("backward_count")),
            "forward_cpu_ms": round(_safe_float(stats.get("forward_cpu_ms")), 4),
            "backward_cpu_ms": round(_safe_float(stats.get("backward_cpu_ms")), 4),
            "forward_cuda_ms": round(_safe_float(stats.get("forward_cuda_ms")), 4),
            "backward_cuda_ms": round(_safe_float(stats.get("backward_cuda_ms")), 4),
            "module_name_examples": [str(item) for item in stats.get("module_name_examples", [])],
        }


def _linear_features(module: Any) -> tuple[int, int] | None:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    try:
        return int(in_features), int(out_features)
    except (TypeError, ValueError):
        return None


def _is_newbie_timing_leaf(module: Any) -> bool:
    if _linear_features(module) is not None:
        return True
    children = getattr(module, "children", None)
    return callable(children) and not any(True for _ in children())


def _classify_newbie_timing_module(name: str, module: Any) -> str:
    lowered = str(name or "").lower()
    class_name = type(module).__name__.lower()
    features = _linear_features(module)
    if features is not None:
        in_features, out_features = features
        small = min(in_features, out_features)
        large = max(in_features, out_features)
        if small > 0 and large == small * 4 and small >= 1024:
            return "newbie.ffn.expand_4x_linear" if in_features < out_features else "newbie.ffn.down_4x_linear"
        if small <= 64 and large >= 1024:
            return "newbie.lora.low_rank_linear"
        if any(token in lowered for token in ("feed_forward", "feedforward", ".ffn", ".mlp")):
            return "newbie.ffn.named_linear"
        if any(token in lowered for token in ("attention", ".attn", "q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out")):
            return "newbie.attention.linear"
        if large >= 1024:
            return "newbie.other.large_linear"
    if "lora" in lowered or "lora" in class_name:
        return "newbie.lora.module"
    return ""
