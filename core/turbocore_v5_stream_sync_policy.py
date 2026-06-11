"""Default-off stream/event-chain sync policy for TurboCore V5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


VALID_MODES = {"off", "observe", "event_chain_experimental"}


def build_v5_stream_sync_policy(
    *,
    timing_triage: Mapping[str, Any] | None = None,
    stream_guard: Mapping[str, Any] | None = None,
    native_runtime: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
    requested_mode: str = "off",
) -> dict[str, Any]:
    """Plan the sync fast path without changing native runtime behavior."""

    triage = _as_dict(timing_triage)
    guard = _as_dict(stream_guard)
    runtime = _as_dict(native_runtime)
    rollback = _rollback_policy(rollback_policy)
    mode = _mode(requested_mode)
    requested = mode != "off"
    explicit_experimental = mode == "event_chain_experimental"
    progress_gates = {
        "default_off": not requested,
        "explicit_experimental_request": explicit_experimental,
        "timing_triage_ready": bool(triage.get("timing_triage_ready", False)),
        "sync_is_primary_bottleneck": str(triage.get("primary_bottleneck") or "")
        == "stream_event_chain_sync_fast_path",
        "native_runtime_borrowed_stream_launch_supported": bool(
            runtime.get("adamw_launch_on_borrowed_stream_supported", False)
        ),
        "native_runtime_ctx_sync_free_step_supported": bool(
            runtime.get("ctx_synchronize_free_training_step_supported", False)
        ),
        "stream_guard_event_chain_verified": bool(guard.get("event_chain_verified", False)),
        "stream_guard_nonzero_external_stream": bool(guard.get("stream_handle_nonzero", False))
        and str(guard.get("stream_handle_kind", "") or "") == "external_cuda_stream_handle",
        "stream_lifetime_bound": bool(guard.get("stream_lifetime_bound", False)),
        "stream_ordering_verified": bool(guard.get("pre_launch_ordering_verified", False))
        and bool(guard.get("post_launch_ordering_verified", False))
        and bool(guard.get("stream_wait_event_verified", False)),
        "rollback_policy_ready": _rollback_ready(rollback),
        "default_and_auto_rollout_off": True,
    }
    allowed = bool(explicit_experimental and all(_required_gates(progress_gates).values()))
    blocked = _blockers(progress_gates, mode)
    return {
        "schema_version": 1,
        "policy": "turbocore_v5_stream_sync_policy_v0",
        "gate": "v5_stream_event_chain_sync_policy",
        "ok": allowed,
        "sync_fast_path_allowed": allowed,
        "requested_mode": mode,
        "requested": requested,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manual_wider_canary_allowed": False,
        "requires_explicit_opt_in": True,
        "current_runtime_synchronization": str(_as_dict(triage.get("metrics")).get("runtime_synchronization", "") or ""),
        "current_runtime_stream_binding": str(_as_dict(triage.get("metrics")).get("runtime_stream_binding", "") or ""),
        "required_native_runtime_capabilities": {
            "adamw_launch_on_borrowed_stream_supported": True,
            "ctx_synchronize_free_training_step_supported": True,
            "event_chain_synchronization_supported": True,
        },
        "native_runtime": {
            "adamw_launch_on_borrowed_stream_supported": bool(
                runtime.get("adamw_launch_on_borrowed_stream_supported", False)
            ),
            "ctx_synchronize_free_training_step_supported": bool(
                runtime.get("ctx_synchronize_free_training_step_supported", False)
            ),
            "event_chain_synchronization_supported": bool(
                runtime.get("event_chain_synchronization_supported", False)
            ),
        },
        "stream_guard_summary": _stream_guard_summary(guard),
        "rollback_policy": rollback,
        "progress_gates": progress_gates,
        "required_gates": _required_gates(progress_gates),
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(allowed, blocked),
        "notes": [
            "This policy does not remove cuCtxSynchronize by itself.",
            "Skipping context synchronization is blocked until the native AdamW runtime can launch on a verified borrowed stream.",
            "Default and auto rollout remain disabled.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}


def _required_gates(progress_gates: Mapping[str, bool]) -> dict[str, bool]:
    return {
        key: bool(value)
        for key, value in progress_gates.items()
        if key not in {"default_off"}
    }


def _blockers(progress_gates: Mapping[str, bool], mode: str) -> list[str]:
    if mode == "off":
        return ["v5_p8_stream_sync_policy_default_off"]
    if mode == "observe":
        return ["v5_p8_stream_sync_policy_observe_only"]
    return _dedupe(
        [
            f"v5_p8_{name}_missing"
            for name, ok in _required_gates(progress_gates).items()
            if not ok
        ]
    )


def _stream_guard_summary(guard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(guard),
        "stream_handle_kind": str(guard.get("stream_handle_kind", "") or ""),
        "stream_handle_reported": bool(guard.get("stream_handle_reported", False)),
        "stream_handle_nonzero": bool(guard.get("stream_handle_nonzero", False)),
        "event_chain_verified": bool(guard.get("event_chain_verified", False)),
        "pre_launch_ordering_verified": bool(guard.get("pre_launch_ordering_verified", False)),
        "post_launch_ordering_verified": bool(guard.get("post_launch_ordering_verified", False)),
        "stream_wait_event_verified": bool(guard.get("stream_wait_event_verified", False)),
        "stream_lifetime_bound": bool(guard.get("stream_lifetime_bound", False)),
        "blocked_reasons": list(guard.get("blocked_reasons", []) or []),
    }


def _rollback_policy(source: Mapping[str, Any] | None) -> dict[str, Any]:
    value = _as_dict(source)
    return {
        "schema_version": 1,
        "policy": "v5_stream_sync_fast_path_rollback_policy_v0",
        "fallback_authoritative": bool(value.get("fallback_authoritative", True)),
        "fallback_backend": str(value.get("fallback_backend", "pytorch_adamw") or "pytorch_adamw"),
        "disable_for_run_on_native_error": bool(value.get("disable_for_run_on_native_error", True)),
        "disable_for_run_on_state_sync_failure": bool(value.get("disable_for_run_on_state_sync_failure", True)),
        "disable_for_run_on_stream_ordering_failure": bool(
            value.get("disable_for_run_on_stream_ordering_failure", True)
        ),
        "disable_for_run_on_non_finite": bool(value.get("disable_for_run_on_non_finite", True)),
        "rollback_on_performance_regression": bool(value.get("rollback_on_performance_regression", True)),
        "default_training_path_enabled": False,
    }


def _rollback_ready(rollback: Mapping[str, Any]) -> bool:
    return bool(
        rollback.get("fallback_authoritative", False)
        and rollback.get("disable_for_run_on_native_error", False)
        and rollback.get("disable_for_run_on_state_sync_failure", False)
        and rollback.get("disable_for_run_on_stream_ordering_failure", False)
        and rollback.get("disable_for_run_on_non_finite", False)
        and rollback.get("rollback_on_performance_regression", False)
    )


def _recommended_next_step(allowed: bool, blocked: list[str]) -> str:
    if allowed:
        return "run an explicit stream/event-chain native AdamW canary; default and auto remain off"
    if "v5_p8_stream_sync_policy_default_off" in blocked:
        return "keep sync fast path disabled unless explicitly requested"
    if any("native_runtime_borrowed_stream_launch_supported" in item for item in blocked):
        return "implement native AdamW borrowed-stream launch support before removing cuCtxSynchronize"
    if any("stream_guard" in item or "stream_lifetime" in item or "stream_ordering" in item for item in blocked):
        return "complete stream guard/event-chain lifetime evidence before sync fast path"
    return "resolve stream sync policy blockers before canary"


def _mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in VALID_MODES else "off"


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
    parser = argparse.ArgumentParser(description="Build TurboCore V5 stream sync policy report.")
    parser.add_argument("--timing-triage", default="", help="Path to V5 P7 timing triage JSON.")
    parser.add_argument("--stream-guard", default="", help="Optional stream guard evidence JSON.")
    parser.add_argument("--native-runtime", default="", help="Optional native runtime capability JSON.")
    parser.add_argument("--rollback-policy", default="", help="Optional rollback policy JSON.")
    parser.add_argument("--mode", default="off", help="off | observe | event_chain_experimental")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)

    report = build_v5_stream_sync_policy(
        timing_triage=load_json(args.timing_triage) if args.timing_triage else None,
        stream_guard=load_json(args.stream_guard) if args.stream_guard else None,
        native_runtime=load_json(args.native_runtime) if args.native_runtime else None,
        rollback_policy=load_json(args.rollback_policy) if args.rollback_policy else None,
        requested_mode=args.mode,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_stream_sync_policy", "load_json"]
