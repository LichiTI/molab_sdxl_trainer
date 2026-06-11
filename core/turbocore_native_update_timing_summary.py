"""Timing summary helpers for TurboCore native update dispatch reports."""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping


def summarize_native_update_timing(
    dispatch_runtime_reports: Iterable[Mapping[str, Any]] | None,
    loop_timings: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Flatten native update executor timings into matrix-friendly fields."""

    results = _training_executor_results(dispatch_runtime_reports)
    executed = [item for item in results if bool(item.get("native_step_executed", False))]
    update_reports = [_as_dict(item.get("update_report")) for item in executed]
    owner_steps = [_as_dict(item.get("owner_step")) for item in update_reports]
    native_reports = [_as_dict(item.get("native_report")) for item in owner_steps]
    native_reports = [item for item in native_reports if item]

    summary: dict[str, Any] = {
        "native_dispatch_training_executor_reports": len(results),
        "native_dispatch_training_executor_executed_reports": len(executed),
        "native_dispatch_training_executor_timing_present": any(
            bool(_as_dict(item.get("timing"))) for item in executed
        ),
        "native_dispatch_update_report_present": any(bool(item) for item in update_reports),
        "native_dispatch_owner_native_report_present": bool(native_reports),
    }
    _add_timing_fields(
        summary,
        "native_dispatch_training_executor",
        [_as_dict(item.get("timing")) for item in executed],
        ("elapsed_ms", "state_sync_ms", "param_sync_ms", "executor_step_ms", "optimizer_state_sync_ms"),
    )
    _add_timing_fields(
        summary,
        "native_dispatch_update_executor",
        [_as_dict(item.get("timing")) for item in update_reports],
        ("elapsed_ms", "grad_sync_ms", "owner_step_ms", "copyback_ms", "zero_grad_ms"),
    )
    _add_timing_fields(
        summary,
        "native_dispatch_loop",
        [_as_dict(item) for item in (loop_timings or [])],
        (
            "executor_get_ms",
            "dispatch_runtime_prepare_ms",
            "gate_update_ms",
            "arming_observe_ms",
            "runtime_profile_refresh_ms",
            "shadow_prepare_ms",
            "shadow_compare_ms",
        ),
    )
    last_update = next((item for item in reversed(update_reports) if item), {})
    last_native = next((item for item in reversed(native_reports) if item), {})
    if last_update:
        summary.update(
            {
                "native_dispatch_update_executor_owner_backend": str(last_update.get("owner_backend", "") or ""),
                "native_dispatch_update_executor_used_direct_grad": bool(last_update.get("used_direct_grad", False)),
                "native_dispatch_update_executor_native_kernel_present": bool(
                    last_update.get("native_kernel_present", False)
                ),
            }
        )
    if last_native:
        borrowed_policy = _as_dict(last_native.get("borrowed_stream_policy"))
        borrowed_lease = _as_dict(last_native.get("borrowed_stream_runtime_lease"))
        summary.update(
            {
                "native_dispatch_owner_native_runtime_synchronization": str(
                    last_native.get("runtime_synchronization", "") or ""
                ),
                "native_dispatch_owner_native_runtime_stream_binding": str(
                    last_native.get("runtime_launch_stream_binding", "") or ""
                ),
                "native_dispatch_owner_native_stream_lifetime_bound": bool(
                    last_native.get("stream_lifetime_bound", False)
                ),
                "native_dispatch_owner_native_stream_synchronization_bound": bool(
                    last_native.get("stream_synchronization_bound", False)
                ),
                "native_dispatch_owner_native_ctx_synchronize_skipped": bool(
                    last_native.get("ctx_synchronize_skipped", False)
                ),
                "native_dispatch_owner_native_borrowed_stream_policy_allowed": borrowed_policy.get("allowed")
                if borrowed_policy
                else None,
                "native_dispatch_owner_native_borrowed_stream_handle_nonzero": bool(
                    borrowed_policy.get("stream_handle_nonzero", False)
                )
                if borrowed_policy
                else None,
                "native_dispatch_owner_native_borrowed_stream_runtime_lease_ok": borrowed_lease.get("ok")
                if borrowed_lease
                else None,
            }
        )
    return summary


def _training_executor_results(
    dispatch_runtime_reports: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for report in dispatch_runtime_reports or []:
        runtime = _as_dict(report)
        training_executor = _as_dict(runtime.get("training_executor"))
        result = _as_dict(training_executor.get("result"))
        if result:
            results.append(result)
    return results


def _add_timing_fields(
    summary: dict[str, Any],
    prefix: str,
    timings: Iterable[Mapping[str, Any]],
    fields: tuple[str, ...],
) -> None:
    timing_rows = [_as_dict(item) for item in timings]
    for field in fields:
        values = [_float(item.get(field)) for item in timing_rows]
        values = [item for item in values if item is not None]
        if not values:
            continue
        summary[f"{prefix}_{field}_mean"] = round(float(statistics.fmean(values)), 4)
        summary[f"{prefix}_{field}_last"] = round(float(values[-1]), 4)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["summarize_native_update_timing"]
