# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe-only prefetch queue for .lynx manifest batch reads.

This stays outside the trainer hot path. It exists to prove ordered batch
delivery, explicit fallback accounting, and queue stall metrics before any
runtime/request integration is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterable, Sequence

try:
    from .lossless_cache_tensor_manifest_adapter import (
        LosslessTensorManifestAdapterConfig,
        LosslessTensorManifestDatasetAdapter,
    )
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_tensor_manifest_adapter import (
        LosslessTensorManifestAdapterConfig,
        LosslessTensorManifestDatasetAdapter,
    )


BatchConsumer = Callable[["LosslessTensorManifestPrefetchBatch"], None]


@dataclass(frozen=True)
class LosslessTensorManifestPrefetchConfig:
    enabled: bool = False
    prefetch_depth: int = 2
    strict_adapter: bool = False
    verify_crc32: bool = True
    eager_readers: bool = False
    copy_arrays: bool = True
    use_tensor_records: bool = True
    consumer_delay_ms: float = 0.0


@dataclass
class LosslessTensorManifestPrefetchBatch:
    index: int
    label: str
    sample_ids: list[str]
    rows: list[dict[str, Any]]
    adapter_report: dict[str, Any]
    load_ms: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.adapter_report.get("ok"))


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(started: float) -> float:
    return (_now() - started) * 1000.0


def _round_ms(value: float) -> float:
    return round(float(value), 4)


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    items = sorted(float(item) for item in values)

    def pct(q: float) -> float:
        if len(items) == 1:
            return items[0]
        rank = (len(items) - 1) * min(max(q, 0.0), 1.0)
        low = int(rank)
        high = min(low + 1, len(items) - 1)
        frac = rank - low
        return items[low] * (1.0 - frac) + items[high] * frac

    return {
        "count": len(items),
        "total_ms": _round_ms(sum(items)),
        "p50_ms": _round_ms(pct(0.50)),
        "p95_ms": _round_ms(pct(0.95)),
        "max_ms": _round_ms(max(items)),
    }


def _normalize_batches(batches: Iterable[tuple[str, Iterable[str]]]) -> list[tuple[str, list[str]]]:
    return [(str(label), [str(item) for item in sample_ids]) for label, sample_ids in batches]


def _empty_report(batch_count: int, cfg: LosslessTensorManifestPrefetchConfig) -> dict[str, Any]:
    return {
        "provider": "lynx_manifest_prefetch_queue_v1",
        "ok": False,
        "batch_count": int(batch_count),
        "prefetch_depth": max(int(cfg.prefetch_depth), 1),
        "consumer_delay_ms": _round_ms(float(cfg.consumer_delay_ms)),
        "manifest_prefetch_ready": False,
        "training_path_enabled": False,
        "resource_center_allowed": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "shadow_only": True,
    }


def _adapter_config(cfg: LosslessTensorManifestPrefetchConfig) -> LosslessTensorManifestAdapterConfig:
    return LosslessTensorManifestAdapterConfig(
        enabled=True,
        strict=bool(cfg.strict_adapter),
        verify_crc32=bool(cfg.verify_crc32),
        eager_readers=bool(cfg.eager_readers),
        copy_arrays=bool(cfg.copy_arrays),
        use_tensor_records=bool(cfg.use_tensor_records),
    )


def _load_batch(
    adapter: LosslessTensorManifestDatasetAdapter,
    *,
    index: int,
    label: str,
    sample_ids: list[str],
) -> LosslessTensorManifestPrefetchBatch:
    started = _now()
    try:
        rows, report = adapter.load_batch(sample_ids)
        return LosslessTensorManifestPrefetchBatch(
            index=int(index),
            label=str(label),
            sample_ids=list(sample_ids),
            rows=rows or [],
            adapter_report=report,
            load_ms=_round_ms(_elapsed_ms(started)),
        )
    except Exception as exc:
        return LosslessTensorManifestPrefetchBatch(
            index=int(index),
            label=str(label),
            sample_ids=list(sample_ids),
            rows=[],
            adapter_report={
                "ok": False,
                "reason": "manifest_prefetch_decode_failed",
                "fallback_to_raw_npz": True,
                "training_path_enabled": False,
                "resource_center_allowed": False,
            },
            load_ms=_round_ms(_elapsed_ms(started)),
            error=f"{type(exc).__name__}: {exc}",
        )


def _case_row(payload: LosslessTensorManifestPrefetchBatch) -> dict[str, Any]:
    delivered = [str(row.get("sample_id") or "") for row in payload.rows]
    return {
        "index": int(payload.index),
        "label": payload.label,
        "ok": payload.ok,
        "requested_sample_ids": list(payload.sample_ids),
        "delivered_sample_ids": delivered,
        "order_preserved": delivered == payload.sample_ids,
        "sample_count": len(payload.sample_ids),
        "tensor_count": sum(int(row.get("tensor_count") or 0) for row in payload.rows),
        "load_ms": _round_ms(payload.load_ms),
        "fallback_to_raw_npz": bool(payload.adapter_report.get("fallback_to_raw_npz")),
        "adapter_report_ok": bool(payload.adapter_report.get("ok")),
        "adapter_training_path_enabled": bool(payload.adapter_report.get("training_path_enabled")),
        "adapter_resource_center_allowed": bool(payload.adapter_report.get("resource_center_allowed")),
        "uses_raw_tensor_records": bool(payload.adapter_report.get("uses_raw_tensor_records")),
        "batch_plan_shard_count": int(payload.adapter_report.get("batch_plan_shard_count") or 0),
        "batch_plan_read_round_count": int(payload.adapter_report.get("batch_plan_read_round_count") or 0),
        "batch_plan_missing_sample_count": int(
            payload.adapter_report.get("batch_plan_missing_sample_count") or 0
        ),
        "batch_plan_request_shard_transition_count": int(
            payload.adapter_report.get("batch_plan_request_shard_transition_count") or 0
        ),
        "error": payload.error,
    }


def run_lossless_tensor_manifest_serial_batches(
    manifest_path: str | Path,
    batches: Iterable[tuple[str, Iterable[str]]],
    *,
    config: LosslessTensorManifestPrefetchConfig | None = None,
    consumer: BatchConsumer | None = None,
) -> dict[str, Any]:
    cfg = config or LosslessTensorManifestPrefetchConfig()
    batch_rows = _normalize_batches(batches)
    report = _empty_report(len(batch_rows), cfg) | {"provider": "lynx_manifest_serial_batches_v1"}
    if not cfg.enabled:
        report.update({"ok": True, "skipped": True, "reason": "disabled"})
        return report
    if not batch_rows:
        report["reason"] = "no_batches"
        return report

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    load_samples: list[float] = []
    started = _now()
    with LosslessTensorManifestDatasetAdapter(manifest_path, config=_adapter_config(cfg)) as adapter:
        for index, (label, sample_ids) in enumerate(batch_rows):
            payload = _load_batch(adapter, index=index, label=label, sample_ids=sample_ids)
            cases.append(_case_row(payload))
            load_samples.append(payload.load_ms)
            if payload.error:
                errors.append(payload.error)
                continue
            if consumer is not None:
                consumer(payload)
            if float(cfg.consumer_delay_ms) > 0.0:
                time.sleep(float(cfg.consumer_delay_ms) / 1000.0)

    fallback_count = sum(1 for case in cases if case.get("fallback_to_raw_npz"))
    batch_plan_missing_count = sum(int(case.get("batch_plan_missing_sample_count") or 0) for case in cases)
    report.update(
        {
            "ok": not errors and len(cases) == len(batch_rows),
            "submitted": len(batch_rows),
            "consumed": len(cases) - len(errors),
            "error_count": len(errors),
            "errors": errors,
            "fallback_to_raw_npz_count": fallback_count,
            "order_preserved_count": sum(1 for case in cases if case.get("order_preserved")),
            "adapter_training_path_enabled_count": sum(
                1 for case in cases if case.get("adapter_training_path_enabled")
            ),
            "adapter_resource_center_allowed_count": sum(
                1 for case in cases if case.get("adapter_resource_center_allowed")
            ),
            "raw_tensor_record_case_count": sum(
                1 for case in cases if case.get("uses_raw_tensor_records")
            ),
            "batch_plan_read_round_count": sum(
                int(case.get("batch_plan_read_round_count") or 0) for case in cases
            ),
            "batch_plan_max_shard_count": max(
                [int(case.get("batch_plan_shard_count") or 0) for case in cases] or [0]
            ),
            "batch_plan_missing_sample_count": batch_plan_missing_count,
            "batch_plan_request_shard_transition_count": sum(
                int(case.get("batch_plan_request_shard_transition_count") or 0) for case in cases
            ),
            "wall_ms": _round_ms(_elapsed_ms(started)),
            "load": _timing_summary(load_samples),
            "cases": cases,
            "manifest_prefetch_ready": not errors and fallback_count == 0,
            "batch_locality_prefetch_ready": not errors
            and fallback_count == 0
            and batch_plan_missing_count == 0
            and sum(1 for case in cases if case.get("order_preserved")) == len(cases),
        }
    )
    return report


def run_lossless_tensor_manifest_prefetch_queue(
    manifest_path: str | Path,
    batches: Iterable[tuple[str, Iterable[str]]],
    *,
    config: LosslessTensorManifestPrefetchConfig | None = None,
    consumer: BatchConsumer | None = None,
) -> dict[str, Any]:
    cfg = config or LosslessTensorManifestPrefetchConfig()
    batch_rows = _normalize_batches(batches)
    report = _empty_report(len(batch_rows), cfg)
    if not cfg.enabled:
        report.update({"ok": True, "skipped": True, "reason": "disabled"})
        return report
    if not batch_rows:
        report["reason"] = "no_batches"
        return report

    work_queue: queue.Queue[LosslessTensorManifestPrefetchBatch | None] = queue.Queue(
        maxsize=max(int(cfg.prefetch_depth), 1)
    )
    errors: list[str] = []
    cases: list[dict[str, Any]] = []
    load_samples: list[float] = []
    consumer_samples: list[float] = []
    queue_empty_wait_ms = 0.0
    queue_full_stall_ms = 0.0
    stall_lock = threading.Lock()

    def producer() -> None:
        nonlocal queue_full_stall_ms
        try:
            with LosslessTensorManifestDatasetAdapter(manifest_path, config=_adapter_config(cfg)) as adapter:
                for index, (label, sample_ids) in enumerate(batch_rows):
                    payload = _load_batch(adapter, index=index, label=label, sample_ids=sample_ids)
                    put_started = _now()
                    work_queue.put(payload)
                    with stall_lock:
                        queue_full_stall_ms += _elapsed_ms(put_started)
        except Exception as exc:
            payload = LosslessTensorManifestPrefetchBatch(
                index=-1,
                label="producer",
                sample_ids=[],
                rows=[],
                adapter_report={
                    "ok": False,
                    "reason": "manifest_prefetch_producer_failed",
                    "fallback_to_raw_npz": True,
                    "training_path_enabled": False,
                    "resource_center_allowed": False,
                },
                load_ms=0.0,
                error=f"producer:{type(exc).__name__}: {exc}",
            )
            work_queue.put(payload)
        finally:
            work_queue.put(None)

    started = _now()
    thread = threading.Thread(target=producer, name="lynx-manifest-prefetch-probe", daemon=True)
    thread.start()
    consumed = 0
    while True:
        get_started = _now()
        payload = work_queue.get()
        queue_empty_wait_ms += _elapsed_ms(get_started)
        if payload is None:
            break
        cases.append(_case_row(payload))
        load_samples.append(payload.load_ms)
        if payload.error:
            errors.append(payload.error)
            continue
        callback_started = _now()
        if consumer is not None:
            try:
                consumer(payload)
            except Exception as exc:
                errors.append(f"consumer:{type(exc).__name__}: {exc}")
        consumer_samples.append(_elapsed_ms(callback_started))
        consumed += 1
        if float(cfg.consumer_delay_ms) > 0.0:
            time.sleep(float(cfg.consumer_delay_ms) / 1000.0)
    thread.join(timeout=5.0)
    if thread.is_alive():
        errors.append("producer_thread_join_timeout")

    fallback_count = sum(1 for case in cases if case.get("fallback_to_raw_npz"))
    batch_plan_missing_count = sum(int(case.get("batch_plan_missing_sample_count") or 0) for case in cases)
    report.update(
        {
            "ok": not errors and consumed == len(batch_rows),
            "submitted": len(batch_rows),
            "decoded": len(cases),
            "consumed": consumed,
            "error_count": len(errors),
            "errors": errors,
            "fallback_to_raw_npz_count": fallback_count,
            "order_preserved_count": sum(1 for case in cases if case.get("order_preserved")),
            "adapter_training_path_enabled_count": sum(
                1 for case in cases if case.get("adapter_training_path_enabled")
            ),
            "adapter_resource_center_allowed_count": sum(
                1 for case in cases if case.get("adapter_resource_center_allowed")
            ),
            "raw_tensor_record_case_count": sum(
                1 for case in cases if case.get("uses_raw_tensor_records")
            ),
            "batch_plan_read_round_count": sum(
                int(case.get("batch_plan_read_round_count") or 0) for case in cases
            ),
            "batch_plan_max_shard_count": max(
                [int(case.get("batch_plan_shard_count") or 0) for case in cases] or [0]
            ),
            "batch_plan_missing_sample_count": batch_plan_missing_count,
            "batch_plan_request_shard_transition_count": sum(
                int(case.get("batch_plan_request_shard_transition_count") or 0) for case in cases
            ),
            "wall_ms": _round_ms(_elapsed_ms(started)),
            "queue_empty_wait_ms": _round_ms(queue_empty_wait_ms),
            "queue_full_stall_ms": _round_ms(queue_full_stall_ms),
            "record_mode": "raw_tensor_records" if cfg.use_tensor_records else "numpy_arrays",
            "load": _timing_summary(load_samples),
            "consumer_callback": _timing_summary(consumer_samples),
            "cases": cases,
            "manifest_prefetch_ready": not errors
            and fallback_count == 0
            and batch_plan_missing_count == 0
            and consumed == len(batch_rows),
            "batch_locality_prefetch_ready": not errors
            and fallback_count == 0
            and batch_plan_missing_count == 0
            and consumed == len(batch_rows)
            and sum(1 for case in cases if case.get("order_preserved")) == len(cases),
            "readiness_blockers": [
                "diagnostic_queue_only",
                "not_attached_to_real_dataloader_workers",
                "no_training_wall_clock_ab",
                "gpu_idle_and_step_p95_not_verified",
            ],
        }
    )
    return report


__all__ = [
    "LosslessTensorManifestPrefetchBatch",
    "LosslessTensorManifestPrefetchConfig",
    "run_lossless_tensor_manifest_prefetch_queue",
    "run_lossless_tensor_manifest_serial_batches",
]
