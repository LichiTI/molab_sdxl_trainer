"""Timing bottleneck triage for TurboCore V5 native update samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


def build_v5_timing_bottleneck_triage(
    *,
    matrix_summary: Mapping[str, Any] | None = None,
    native_case_name: str = "",
) -> dict[str, Any]:
    """Classify native-update timing overhead without changing runtime behavior."""

    payload = _as_dict(matrix_summary)
    native_case = _native_case(payload, native_case_name)
    summary = _as_dict(native_case.get("summary"))
    timing_present = bool(summary.get("native_dispatch_training_executor_timing_present", False))
    native_executed = bool(summary.get("native_dispatch_executed", False))
    metrics = _metrics(summary)
    bottlenecks = _rank_bottlenecks(metrics)
    blocked = _blockers(payload, native_case, native_executed, timing_present)
    return {
        "schema_version": 1,
        "triage": "turbocore_v5_timing_bottleneck_triage_v0",
        "gate": "v5_timing_bottleneck_triage",
        "ok": not blocked,
        "timing_triage_ready": not blocked,
        "native_case": str(_as_dict(native_case.get("case")).get("name", "") or ""),
        "native_dispatch_executed": native_executed,
        "timing_summary_present": timing_present,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manual_wider_canary_allowed": False,
        "metrics": metrics,
        "primary_bottleneck": bottlenecks[0]["id"] if bottlenecks else "",
        "bottlenecks": bottlenecks,
        "engineering_backlog": _engineering_backlog(bottlenecks),
        "blocked_reasons": blocked,
        "recommended_next_step": _recommended_next_step(blocked, bottlenecks),
        "notes": [
            "This triage only ranks engineering work; it does not enable native dispatch.",
            "Global CUDA context synchronization is treated as a high-priority overhead even when step speedup is positive.",
            "Default and auto rollout remain disabled.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}


def _native_case(payload: Mapping[str, Any], native_case_name: str) -> dict[str, Any]:
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        return {}
    if native_case_name:
        for item in cases:
            case = _as_dict(item)
            if str(_as_dict(case.get("case")).get("name", "") or "") == native_case_name:
                return case
    for item in cases:
        case = _as_dict(item)
        summary = _as_dict(case.get("summary"))
        if bool(summary.get("native_dispatch_executed", False)):
            return case
    for item in cases:
        case = _as_dict(item)
        name = str(_as_dict(case.get("case")).get("name", "") or "")
        if "native_update_dispatch" in name:
            return case
    return {}


def _metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    training_elapsed = _float(summary.get("native_dispatch_training_executor_elapsed_ms_mean"))
    update_elapsed = _float(summary.get("native_dispatch_update_executor_elapsed_ms_mean"))
    prepare = _float(summary.get("native_dispatch_loop_dispatch_runtime_prepare_ms_mean"))
    grad_sync = _float(summary.get("native_dispatch_update_executor_grad_sync_ms_mean"))
    owner_step = _float(summary.get("native_dispatch_update_executor_owner_step_ms_mean"))
    copyback = _float(summary.get("native_dispatch_update_executor_copyback_ms_mean"))
    state_sync = _float(summary.get("native_dispatch_training_executor_state_sync_ms_mean"))
    state_sync_last = _float(summary.get("native_dispatch_training_executor_state_sync_ms_last"))
    runtime_sync = str(summary.get("native_dispatch_owner_native_runtime_synchronization", "") or "")
    stream_binding = str(summary.get("native_dispatch_owner_native_runtime_stream_binding", "") or "")
    global_sync = _global_sync(runtime_sync, stream_binding)
    return {
        "steady_mean_step_ms": _float(summary.get("steady_mean_step_ms")),
        "mean_step_ms": _float(summary.get("mean_step_ms")),
        "training_executor_elapsed_ms_mean": training_elapsed,
        "training_executor_state_sync_ms_mean": state_sync,
        "training_executor_state_sync_ms_last": state_sync_last,
        "update_executor_elapsed_ms_mean": update_elapsed,
        "update_executor_grad_sync_ms_mean": grad_sync,
        "update_executor_owner_step_ms_mean": owner_step,
        "update_executor_copyback_ms_mean": copyback,
        "loop_dispatch_runtime_prepare_ms_mean": prepare,
        "runtime_synchronization": runtime_sync,
        "runtime_stream_binding": stream_binding,
        "global_context_sync": global_sync,
        "used_direct_grad": bool(summary.get("native_dispatch_update_executor_used_direct_grad", False)),
        "native_kernel_present": bool(summary.get("native_dispatch_update_executor_native_kernel_present", False)),
        "prepare_vs_training_elapsed_ratio": _ratio(prepare, training_elapsed),
        "grad_sync_vs_update_elapsed_ratio": _ratio(grad_sync, update_elapsed),
        "copyback_vs_update_elapsed_ratio": _ratio(copyback, update_elapsed),
        "state_sync_warmup_heavy": bool(
            state_sync is not None
            and state_sync_last is not None
            and state_sync >= 1.0
            and state_sync_last <= max(0.05, state_sync * 0.05)
        ),
    }


def _rank_bottlenecks(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if bool(metrics.get("global_context_sync", False)):
        candidates.append(
            _candidate(
                "stream_event_chain_sync_fast_path",
                100,
                "Replace per-step cuCtxSynchronize/default-stream synchronization with a bound stream/event chain.",
                ["runtime_synchronization", "runtime_stream_binding"],
            )
        )
    prepare = _float(metrics.get("loop_dispatch_runtime_prepare_ms_mean"))
    training_elapsed = _float(metrics.get("training_executor_elapsed_ms_mean"))
    prepare_ratio = _ratio(prepare, training_elapsed)
    if prepare is not None and (prepare >= 5.0 or _at_least(prepare_ratio, 0.35)):
        score = 80 + min(15, int(prepare))
        candidates.append(
            _candidate(
                "dispatch_prepare_cache_fast_path",
                score,
                "Cache dispatch prepare inputs and avoid rebuilding arming/runtime context each step.",
                ["loop_dispatch_runtime_prepare_ms_mean"],
            )
        )
    grad_sync = _float(metrics.get("update_executor_grad_sync_ms_mean"))
    update_elapsed = _float(metrics.get("update_executor_elapsed_ms_mean"))
    grad_ratio = _ratio(grad_sync, update_elapsed)
    if grad_sync is not None and (grad_sync >= 2.0 or _at_least(grad_ratio, 0.25)):
        candidates.append(
            _candidate(
                "direct_grad_owner_buffer_fast_path",
                70 + min(10, int(grad_sync)),
                "Write gradients directly into the native owner buffer and reduce per-step grad sync.",
                ["update_executor_grad_sync_ms_mean", "used_direct_grad"],
            )
        )
    copyback = _float(metrics.get("update_executor_copyback_ms_mean"))
    copyback_ratio = _ratio(copyback, update_elapsed)
    if copyback is not None and (copyback >= 2.0 or _at_least(copyback_ratio, 0.20)):
        candidates.append(
            _candidate(
                "copyback_defer_or_owner_state_snapshot",
                62 + min(10, int(copyback)),
                "Defer parameter/state copyback to checkpoint/fallback boundaries where safe.",
                ["update_executor_copyback_ms_mean"],
            )
        )
    if bool(metrics.get("state_sync_warmup_heavy", False)):
        candidates.append(
            _candidate(
                "state_sync_warmup_amortization",
                45,
                "Keep state sync out of steady-step triage; current evidence suggests warmup-heavy cost.",
                ["training_executor_state_sync_ms_mean", "training_executor_state_sync_ms_last"],
            )
        )
    candidates.sort(key=lambda item: (-int(item["priority"]), str(item["id"])))
    return candidates


def _engineering_backlog(bottlenecks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "id": str(item.get("id") or ""),
            "priority": int(item.get("priority", 0) or 0),
            "summary": str(item.get("summary") or ""),
            "status": "recommended",
        }
        for index, item in enumerate(bottlenecks)
    ]


def _candidate(identifier: str, priority: int, summary: str, evidence_fields: list[str]) -> dict[str, Any]:
    return {
        "id": identifier,
        "priority": int(priority),
        "summary": summary,
        "evidence_fields": evidence_fields,
    }


def _blockers(
    payload: Mapping[str, Any],
    native_case: Mapping[str, Any],
    native_executed: bool,
    timing_present: bool,
) -> list[str]:
    blocked: list[str] = []
    if not payload:
        blocked.append("v5_p7_matrix_summary_missing")
    elif payload.get("matrix") != "turbocore_update_benchmark_matrix_v0":
        blocked.append("v5_p7_matrix_schema_invalid")
    if not native_case:
        blocked.append("v5_p7_native_case_missing")
    if native_case and not native_executed:
        blocked.append("v5_p7_native_dispatch_not_executed")
    if native_case and not timing_present:
        blocked.append("v5_p7_timing_summary_missing")
    return _dedupe(blocked)


def _recommended_next_step(blocked: list[str], bottlenecks: list[Mapping[str, Any]]) -> str:
    if blocked:
        return "rerun V5 timing-enabled native dispatch matrix before triage"
    if not bottlenecks:
        return "no dominant native timing bottleneck detected; collect longer canary timing"
    primary = str(bottlenecks[0].get("id") or "")
    if primary == "stream_event_chain_sync_fast_path":
        return "prototype stream/event-chain synchronization fast path before widening canary"
    if primary == "dispatch_prepare_cache_fast_path":
        return "cache dispatch prepare inputs before optimizing kernel math"
    if primary == "direct_grad_owner_buffer_fast_path":
        return "prototype direct-gradient owner-buffer write path"
    if primary == "copyback_defer_or_owner_state_snapshot":
        return "prototype deferred copyback/owner-state snapshot policy"
    return "work through the ranked timing backlog"


def _global_sync(runtime_sync: str, stream_binding: str) -> bool:
    sync = runtime_sync.strip().lower()
    binding = stream_binding.strip().lower()
    return bool(
        "cuctxsynchronize" in sync
        or "context_synchronize" in sync
        or "default_stream_null_synchronized" in binding
    )


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return round(float(numerator / denominator), 4)


def _at_least(value: float | None, threshold: float) -> bool:
    return bool(value is not None and value >= threshold)


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if output >= 0.0 else None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build TurboCore V5 timing bottleneck triage.")
    parser.add_argument("--matrix-summary", required=True, help="Path to matrix_summary.json.")
    parser.add_argument("--native-case", default="", help="Optional native case name.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)

    report = build_v5_timing_bottleneck_triage(
        matrix_summary=load_json(args.matrix_summary),
        native_case_name=args.native_case,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_timing_bottleneck_triage", "load_json"]
