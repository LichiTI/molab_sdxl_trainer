"""Experimental LXCS cache prefetch queue probe.

P3 needs evidence about overlap and queue stalls, not just single-batch decode
timings.  This module reads real cache tensor payloads on a background thread,
hands them to a bounded queue, and reports whether a consumer would wait for
decode.  It is intentionally diagnostic-only and stays out of the trainer hot
path unless a caller explicitly wires it in for an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterable, Sequence

try:
    from .lossless_cache_sidecar import (
        load_lossless_cache_sidecar_arrays_from_file,
        load_numpy_cache_entries,
        sidecar_path_for_cache,
        write_lossless_cache_sidecar_file,
    )
    from .lossless_cache_flat_sidecar import (
        load_flat_lossless_cache_sidecar_arrays_from_file,
        write_flat_lossless_cache_sidecar_file,
    )
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_sidecar import (
        load_lossless_cache_sidecar_arrays_from_file,
        load_numpy_cache_entries,
        sidecar_path_for_cache,
        write_lossless_cache_sidecar_file,
    )
    from lossless_cache_flat_sidecar import (
        load_flat_lossless_cache_sidecar_arrays_from_file,
        write_flat_lossless_cache_sidecar_file,
    )
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS


ConsumerCallback = Callable[["LosslessCachePrefetchPayload"], None]
_CUDA_HANDOFF_WARMED = False


@dataclass(frozen=True)
class LosslessCachePrefetchQueueConfig:
    enabled: bool = True
    prefetch_depth: int = 2
    sidecar_enabled: bool = True
    sidecar_strict: bool = False
    fallback_to_raw: bool = True
    sidecar_format: str = "lxcs"
    sidecar_suffix: str = ".lxcs"
    sidecar_dir: str | None = None
    max_entries: int = 0
    verify_sidecar: bool = True
    copy_arrays: bool = True
    consumer_delay_ms: float = 0.0
    handoff: str = "none"  # none | torch | pin | cuda | cuda_coalesced


@dataclass
class LosslessCachePrefetchPayload:
    index: int
    path: Path
    arrays: dict[str, Any]
    report: dict[str, Any]
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.report.get("ok", False))


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(started: float) -> float:
    return (_now() - started) * 1000.0


def _round_ms(value: float) -> float:
    return round(float(value), 4)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    items = sorted(float(item) for item in values)
    if len(items) == 1:
        return items[0]
    rank = (len(items) - 1) * min(max(float(q), 0.0), 1.0)
    low = int(rank)
    high = min(low + 1, len(items) - 1)
    frac = rank - low
    return items[low] * (1.0 - frac) + items[high] * frac


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "total_ms": _round_ms(sum(values)),
        "p50_ms": _round_ms(_percentile(values, 0.50)),
        "p95_ms": _round_ms(_percentile(values, 0.95)),
        "max_ms": _round_ms(max(values)),
    }


def _path_key(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{path.name}.{digest}"


def _sidecar_path(path: Path, config: LosslessCachePrefetchQueueConfig) -> Path:
    if config.sidecar_dir:
        return Path(config.sidecar_dir) / f"{_path_key(path)}{config.sidecar_suffix}"
    return sidecar_path_for_cache(path, suffix=config.sidecar_suffix)


def _arrays_from_raw_cache(path: Path, *, max_entries: int = 0) -> dict[str, Any]:
    import numpy as np

    entries = list(load_numpy_cache_entries(path))
    if max_entries > 0:
        entries = entries[: max_entries]
    arrays: dict[str, Any] = {}
    for entry in entries:
        metadata = dict(entry.metadata or {})
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if not dtype or not isinstance(shape, list):
            continue
        arrays[str(entry.name)] = np.frombuffer(bytes(entry.data), dtype=np.dtype(str(dtype))).reshape(
            tuple(int(item) for item in shape)
        ).copy()
    return arrays


def _trim_arrays(arrays: dict[str, Any], max_entries: int) -> dict[str, Any]:
    if max_entries <= 0 or len(arrays) <= max_entries:
        return arrays
    return {name: arrays[name] for name in list(arrays)[:max_entries]}


def _array_summary(name: str, array: Any) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    return {
        "name": str(name),
        "dtype": str(contiguous.dtype),
        "shape": [int(item) for item in contiguous.shape],
        "nbytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _summarize_arrays(arrays: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    rows = [_array_summary(name, array) for name, array in arrays.items()]
    return sum(int(row["nbytes"]) for row in rows), rows


def _warm_cuda_handoff(torch: Any, mode: str) -> None:
    global _CUDA_HANDOFF_WARMED
    if mode not in {"cuda", "cuda_coalesced"} or not torch.cuda.is_available() or _CUDA_HANDOFF_WARMED:
        return
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    finally:
        _CUDA_HANDOFF_WARMED = True


def _torch_handoff(arrays: dict[str, Any], mode: str) -> dict[str, Any]:
    resolved = str(mode or "none").lower().replace("-", "_")
    if resolved == "none":
        return {"mode": "none", "enabled": False, "handoff_ms": 0.0}
    try:
        import torch
        import numpy as np
    except Exception as exc:
        return {
            "mode": resolved,
            "enabled": False,
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "handoff_ms": 0.0,
        }

    _warm_cuda_handoff(torch, resolved)

    started = _now()
    if resolved == "cuda_coalesced":
        segments: list[dict[str, Any]] = []
        buffer = bytearray()
        for name, array in arrays.items():
            contiguous = np.ascontiguousarray(array)
            offset = len(buffer)
            raw = contiguous.tobytes(order="C")
            buffer.extend(raw)
            segments.append(
                {
                    "name": str(name),
                    "offset": offset,
                    "nbytes": len(raw),
                    "dtype": str(contiguous.dtype),
                    "shape": [int(item) for item in contiguous.shape],
                }
            )
        report: dict[str, Any] = {
            "mode": resolved,
            "enabled": True,
            "ok": True,
            "tensor_count": len(arrays),
            "tensor_bytes": len(buffer),
            "cuda_available": bool(torch.cuda.is_available()),
            "coalesced": True,
            "segment_count": len(segments),
            "h2d_tensor_count": 0,
            "view_reconstruction": "not_attempted",
        }
        if not torch.cuda.is_available():
            report.update({"ok": False, "reason": "cuda_unavailable"})
            report["handoff_ms"] = _round_ms(_elapsed_ms(started))
            return report
        try:
            host = torch.frombuffer(buffer, dtype=torch.uint8)
            pinned = host.pin_memory()
        except Exception as exc:
            report.update({"ok": False, "reason": f"coalesced_pin_memory_failed: {type(exc).__name__}: {exc}"})
            report["handoff_ms"] = _round_ms(_elapsed_ms(started))
            return report
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            copied = pinned.to("cuda", non_blocking=True)
        stream.synchronize()
        report.update(
            {
                "pin_memory": True,
                "cuda_device": torch.cuda.get_device_name(0),
                "h2d_tensor_count": 1,
                "coalesced_bytes": int(copied.nelement() * copied.element_size()),
                "segments": segments,
                "handoff_ms": _round_ms(_elapsed_ms(started)),
            }
        )
        return report

    tensors = {name: torch.from_numpy(array.copy()) for name, array in arrays.items()}
    report: dict[str, Any] = {
        "mode": resolved,
        "enabled": True,
        "ok": True,
        "tensor_count": len(tensors),
        "tensor_bytes": sum(int(tensor.element_size() * tensor.nelement()) for tensor in tensors.values()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved == "torch":
        report["handoff_ms"] = _round_ms(_elapsed_ms(started))
        return report
    if resolved in {"pin", "cuda"}:
        try:
            pinned = {name: tensor.pin_memory() for name, tensor in tensors.items()}
        except Exception as exc:
            report.update({"ok": False, "reason": f"pin_memory_failed: {type(exc).__name__}: {exc}"})
            report["handoff_ms"] = _round_ms(_elapsed_ms(started))
            return report
        report["pin_memory"] = True
        if resolved == "pin":
            report["handoff_ms"] = _round_ms(_elapsed_ms(started))
            return report
        if not torch.cuda.is_available():
            report.update({"ok": False, "reason": "cuda_unavailable"})
            report["handoff_ms"] = _round_ms(_elapsed_ms(started))
            return report
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            copied = {name: tensor.to("cuda", non_blocking=True) for name, tensor in pinned.items()}
        stream.synchronize()
        report.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "h2d_tensor_count": len(copied),
                "handoff_ms": _round_ms(_elapsed_ms(started)),
            }
        )
        return report
    report.update({"ok": False, "reason": f"unsupported_handoff_mode:{resolved}"})
    report["handoff_ms"] = _round_ms(_elapsed_ms(started))
    return report


def load_lossless_cache_prefetch_payload(
    path: str | Path,
    *,
    index: int = 0,
    config: LosslessCachePrefetchQueueConfig | None = None,
) -> LosslessCachePrefetchPayload:
    cfg = config or LosslessCachePrefetchQueueConfig()
    cache_path = Path(path)
    started = _now()
    sidecar = _sidecar_path(cache_path, cfg)
    source = "raw_cache"
    fallback = False
    reason = ""
    arrays: dict[str, Any] = {}

    try:
        if cfg.sidecar_enabled:
            if sidecar.is_file():
                sidecar_format = str(cfg.sidecar_format or "lxcs").lower()
                if sidecar_format == "lxfs":
                    arrays = _trim_arrays(
                        load_flat_lossless_cache_sidecar_arrays_from_file(
                            sidecar,
                            verify_crc32=bool(cfg.verify_sidecar),
                            copy_arrays=bool(cfg.copy_arrays),
                        ),
                        int(cfg.max_entries),
                    )
                    source = "lxfs_sidecar"
                else:
                    arrays = _trim_arrays(
                        load_lossless_cache_sidecar_arrays_from_file(
                            sidecar,
                            verify_sha256=bool(cfg.verify_sidecar),
                            copy_arrays=bool(cfg.copy_arrays),
                        ),
                        int(cfg.max_entries),
                    )
                    source = "lxcs_sidecar"
                reason = "sidecar_loaded"
            elif cfg.sidecar_strict:
                raise FileNotFoundError(f"LXCS sidecar missing: {sidecar}")
            else:
                reason = "sidecar_missing"
        else:
            reason = "sidecar_disabled"

        if not arrays:
            if not cfg.fallback_to_raw:
                raise FileNotFoundError(f"LXCS sidecar unavailable and raw fallback disabled: {cache_path}")
            arrays = _arrays_from_raw_cache(cache_path, max_entries=max(int(cfg.max_entries), 0))
            fallback = True
            source = "raw_cache"
            reason = reason or "raw_fallback"

        raw_bytes, entries = _summarize_arrays(arrays)
        handoff = _torch_handoff(arrays, cfg.handoff)
        decode_ms = _round_ms(_elapsed_ms(started) - float(handoff.get("handoff_ms") or 0.0))
        report = {
            "ok": True,
            "index": int(index),
            "source": str(cache_path),
            "provider": "lxcs_prefetch_payload_v1",
            "payload_source": source,
            "sidecar_format": str(cfg.sidecar_format or "lxcs").lower(),
            "sidecar_path": str(sidecar),
            "reason": reason,
            "entry_count": len(arrays),
            "raw_bytes": raw_bytes,
            "decode_ms": max(decode_ms, 0.0),
            "handoff_ms": _round_ms(float(handoff.get("handoff_ms") or 0.0)),
            "fallback_to_raw_cache": bool(fallback),
            "entries": entries,
            "handoff": handoff,
            "reads_tensor_payloads": True,
            "training_path_enabled": False,
        }
        return LosslessCachePrefetchPayload(index=int(index), path=cache_path, arrays=arrays, report=report)
    except Exception as exc:
        report = {
            "ok": False,
            "index": int(index),
            "source": str(cache_path),
            "provider": "lxcs_prefetch_payload_v1",
            "sidecar_format": str(cfg.sidecar_format or "lxcs").lower(),
            "sidecar_path": str(sidecar),
            "error": f"{type(exc).__name__}: {exc}",
            "decode_ms": _round_ms(_elapsed_ms(started)),
            "fallback_to_raw_cache": bool(fallback),
            "reads_tensor_payloads": True,
            "training_path_enabled": False,
        }
        return LosslessCachePrefetchPayload(index=int(index), path=cache_path, arrays={}, report=report, error=report["error"])


def prepare_lossless_cache_prefetch_sidecars(
    paths: Iterable[str | Path],
    *,
    config: LosslessCachePrefetchQueueConfig | None = None,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
    min_saving: float = 0.02,
) -> dict[str, Any]:
    cfg = config or LosslessCachePrefetchQueueConfig()
    reports: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        sidecar = _sidecar_path(path, cfg)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        started = _now()
        try:
            if str(cfg.sidecar_format or "lxcs").lower() == "lxfs":
                report = write_flat_lossless_cache_sidecar_file(
                    path,
                    sidecar_path=sidecar,
                    codecs=codecs,
                    min_saving=float(min_saving),
                )
            else:
                report = write_lossless_cache_sidecar_file(
                    path,
                    sidecar_path=sidecar,
                    chunk_size=max(int(chunk_size), 1),
                    codecs=codecs,
                    min_saving=float(min_saving),
                )
            report.update({"ok": True, "prepare_ms": _round_ms(_elapsed_ms(started))})
        except Exception as exc:
            report = {
                "ok": False,
                "source": str(path),
                "sidecar_path": str(sidecar),
                "error": f"{type(exc).__name__}: {exc}",
                "prepare_ms": _round_ms(_elapsed_ms(started)),
            }
        reports.append(report)
    return {
        "ok": all(bool(report.get("ok")) for report in reports),
        "provider": "lxcs_prefetch_sidecar_prepare_v1",
        "sidecar_format": str(cfg.sidecar_format or "lxcs").lower(),
        "case_count": len(reports),
        "ok_count": sum(1 for report in reports if report.get("ok")),
        "reports": reports,
        "training_path_enabled": False,
    }


def _empty_metrics(paths: Sequence[Path], config: LosslessCachePrefetchQueueConfig) -> dict[str, Any]:
    return {
        "provider": "lxcs_prefetch_queue_v1",
        "ok": False,
        "path_count": len(paths),
        "prefetch_depth": max(int(config.prefetch_depth), 1),
        "consumer_delay_ms": _round_ms(float(config.consumer_delay_ms)),
        "handoff_mode": str(config.handoff or "none"),
        "reads_tensor_payloads": True,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
        "shadow_only": True,
    }


def run_lossless_cache_prefetch_queue(
    paths: Iterable[str | Path],
    *,
    config: LosslessCachePrefetchQueueConfig | None = None,
    consumer: ConsumerCallback | None = None,
) -> dict[str, Any]:
    cfg = config or LosslessCachePrefetchQueueConfig()
    cache_paths = [Path(path) for path in paths]
    report = _empty_metrics(cache_paths, cfg)
    if not cfg.enabled:
        report.update({"ok": True, "skipped": True, "reason": "disabled", "p3_prefetch_queue_probe_ready": False})
        return report
    if not cache_paths:
        report.update({"reason": "no_cache_paths", "p3_prefetch_queue_probe_ready": False})
        return report

    work_queue: queue.Queue[LosslessCachePrefetchPayload | None] = queue.Queue(maxsize=max(int(cfg.prefetch_depth), 1))
    errors: list[str] = []
    cases: list[dict[str, Any]] = []
    decode_samples: list[float] = []
    handoff_samples: list[float] = []
    callback_samples: list[float] = []
    queue_empty_wait_ms = 0.0
    queue_full_stall_ms = 0.0
    full_lock = threading.Lock()

    def producer() -> None:
        nonlocal queue_full_stall_ms
        for index, path in enumerate(cache_paths):
            payload = load_lossless_cache_prefetch_payload(path, index=index, config=cfg)
            put_started = _now()
            work_queue.put(payload)
            with full_lock:
                queue_full_stall_ms += _elapsed_ms(put_started)
        work_queue.put(None)

    started = _now()
    thread = threading.Thread(target=producer, name="lxcs-prefetch-probe", daemon=True)
    thread.start()
    consumed = 0
    while True:
        get_started = _now()
        payload = work_queue.get()
        queue_empty_wait_ms += _elapsed_ms(get_started)
        if payload is None:
            break
        cases.append({k: v for k, v in payload.report.items() if k != "entries"} | {"entries": payload.report.get("entries", [])})
        decode_samples.append(float(payload.report.get("decode_ms") or 0.0))
        handoff_samples.append(float(payload.report.get("handoff_ms") or 0.0))
        if payload.error:
            errors.append(payload.error)
        else:
            callback_started = _now()
            if consumer is not None:
                consumer(payload)
            callback_samples.append(_elapsed_ms(callback_started))
            consumed += 1
            if float(cfg.consumer_delay_ms) > 0.0:
                time.sleep(float(cfg.consumer_delay_ms) / 1000.0)
    thread.join(timeout=5.0)

    fallback_count = sum(1 for case in cases if case.get("fallback_to_raw_cache"))
    raw_bytes = sum(int(case.get("raw_bytes") or 0) for case in cases)
    wall_ms = _elapsed_ms(started)
    serial_payload_ms = sum(decode_samples) + sum(handoff_samples)
    report.update(
        {
            "ok": not errors and consumed == len(cache_paths),
            "submitted": len(cache_paths),
            "decoded": len(cases),
            "consumed": consumed,
            "error_count": len(errors),
            "errors": errors,
            "fallback_count": fallback_count,
            "raw_bytes": raw_bytes,
            "wall_ms": _round_ms(wall_ms),
            "serial_payload_ms": _round_ms(serial_payload_ms),
            "queue_empty_wait_ms": _round_ms(queue_empty_wait_ms),
            "queue_full_stall_ms": _round_ms(queue_full_stall_ms),
            "decode": _timing_summary(decode_samples),
            "handoff": _timing_summary(handoff_samples),
            "consumer_callback": _timing_summary(callback_samples),
            "cases": cases,
            "p3_prefetch_queue_probe_ready": not errors and consumed == len(cache_paths),
            "p3_prefetch_h2d_ready": False,
            "readiness_blockers": [
                "diagnostic_queue_only",
                "not_attached_to_real_dataloader_workers",
                "no_training_wall_clock_ab",
                "gpu_idle_and_step_p95_not_verified",
            ],
        }
    )
    return report


def run_lossless_cache_serial_probe(
    paths: Iterable[str | Path],
    *,
    config: LosslessCachePrefetchQueueConfig | None = None,
    consumer: ConsumerCallback | None = None,
) -> dict[str, Any]:
    cfg = config or LosslessCachePrefetchQueueConfig()
    cache_paths = [Path(path) for path in paths]
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    decode_samples: list[float] = []
    handoff_samples: list[float] = []
    started = _now()
    consumed = 0
    for index, path in enumerate(cache_paths):
        payload = load_lossless_cache_prefetch_payload(path, index=index, config=cfg)
        cases.append(payload.report)
        decode_samples.append(float(payload.report.get("decode_ms") or 0.0))
        handoff_samples.append(float(payload.report.get("handoff_ms") or 0.0))
        if payload.error:
            errors.append(payload.error)
            continue
        if consumer is not None:
            consumer(payload)
        consumed += 1
        if float(cfg.consumer_delay_ms) > 0.0:
            time.sleep(float(cfg.consumer_delay_ms) / 1000.0)
    return {
        "provider": "lxcs_prefetch_serial_probe_v1",
        "ok": not errors and consumed == len(cache_paths),
        "path_count": len(cache_paths),
        "consumed": consumed,
        "error_count": len(errors),
        "errors": errors,
        "fallback_count": sum(1 for case in cases if case.get("fallback_to_raw_cache")),
        "wall_ms": _round_ms(_elapsed_ms(started)),
        "decode": _timing_summary(decode_samples),
        "handoff": _timing_summary(handoff_samples),
        "cases": cases,
        "reads_tensor_payloads": True,
        "training_path_enabled": False,
        "shadow_only": True,
    }


__all__ = [
    "LosslessCachePrefetchPayload",
    "LosslessCachePrefetchQueueConfig",
    "load_lossless_cache_prefetch_payload",
    "prepare_lossless_cache_prefetch_sidecars",
    "run_lossless_cache_prefetch_queue",
    "run_lossless_cache_serial_probe",
]
