"""Short benchmark matrix for TurboCore update-path evidence.

The matrix is intentionally conservative: it defaults to dry-run and only runs
real training when ``--run`` is passed.  Each case delegates to
``native_runtime_profile_benchmark.py`` so the product trainer path remains the
single source of runtime evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (str(repo_root), str(backend_root)):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

from core.turbocore_native_update_performance import build_native_update_performance_gate
from core.turbocore_native_update_timing_summary import summarize_native_update_timing
from core.turbocore_v5_borrowed_stream_canary_audit import (
    BORROWED_STREAM_POLICY,
    audit_training_executor_reports,
    benchmark_requested_sync_policy,
    summarize_borrowed_stream_matrix,
)
from core.turbocore_v5_ctx_sync_free_matrix_compare import summarize_ctx_sync_free_matrix
from core.turbocore_optimizer_benchmark_artifact import (
    load_optimizer_performance_artifact,
    optimizer_artifact_summary,
)


@dataclass(frozen=True)
class UpdateMatrixCase:
    name: str
    label: str
    description: str
    flags: tuple[str, ...] = field(default_factory=tuple)
    evidence_role: str = "diagnostic"
    performance_sample: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL = "4194304"
OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL = "67108864"


MATRIX_CASES: tuple[UpdateMatrixCase, ...] = (
    UpdateMatrixCase(
        name="baseline_phase",
        label="PyTorch fused AdamW baseline",
        description="No TurboCore shadow/gate; measures normal product trainer phase share.",
        evidence_role="baseline",
    ),
    UpdateMatrixCase(
        name="shadow_full",
        label="Full shadow parity",
        description="Runs full TurboCore update shadow beside PyTorch optimizer.",
        flags=("--turbocore-update-shadow", "shadow"),
    ),
    UpdateMatrixCase(
        name="shadow_sampled_auto",
        label="Sampled shadow auto-stop",
        description="Samples parity comparison and stops shadow work after repeated passes.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
        ),
    ),
    UpdateMatrixCase(
        name="direct_grad_audit",
        label="Direct-grad audit",
        description="Adds direct-gradient hook auditing while PyTorch gradients remain authoritative.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-direct-grad",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
        ),
    ),
    UpdateMatrixCase(
        name="contract_copyback",
        label="Checkpoint/copyback probes",
        description="Adds checkpoint roundtrip and scratch copyback validation probes.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-direct-grad",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-checkpoint-contract",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-save-owner-state",
        ),
    ),
    UpdateMatrixCase(
        name="gate_profile",
        label="Gate/readiness profile",
        description="Adds native update gate/readiness/fallback reporting; still no native dispatch.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-direct-grad",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
            "--turbocore-update-shadow-checkpoint-contract",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "profile",
            "--turbocore-native-update-required-shadow-passes",
            "2",
        ),
    ),
    UpdateMatrixCase(
        name="native_update_dispatch",
        label="Native dispatch rehearsal",
        description="Requests default-off native dispatch and records arming/runtime reports while PyTorch stays authoritative.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-direct-grad",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-checkpoint-contract",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-copyback-dispatch-experimental",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-owner-native-launch-probe",
            "--turbocore-update-shadow-owner-native-launch-max-numel",
            OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL,
            "--turbocore-update-shadow-owner-native-event-chain-probe",
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "native_experimental",
            "--turbocore-native-update-required-shadow-passes",
            "2",
            "--turbocore-native-update-allow-missing-kernel",
            "--turbocore-native-update-dispatch-enabled",
            "--turbocore-native-update-training-path-enabled",
            "--turbocore-native-update-require-native-cuda",
            "--turbocore-native-update-diagnostic-executor-replay",
        ),
        evidence_role="safety",
    ),
    UpdateMatrixCase(
        name="native_update_dispatch_perf",
        label="Native dispatch perf sample",
        description="Measures explicit native dispatch after initial safety evidence, with shadow probes auto-stopped after promotion.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
            "--turbocore-update-shadow-checkpoint-contract",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-copyback-dispatch-experimental",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-owner-native-launch-probe",
            "--turbocore-update-shadow-owner-native-launch-max-numel",
            OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL,
            "--turbocore-update-shadow-owner-native-event-chain-probe",
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "native_experimental",
            "--turbocore-native-update-required-shadow-passes",
            "2",
            "--turbocore-native-update-allow-missing-kernel",
            "--turbocore-native-update-dispatch-enabled",
            "--turbocore-native-update-training-path-enabled",
            "--turbocore-native-update-require-native-cuda",
            "--turbocore-native-update-defer-state-sync",
        ),
        evidence_role="performance",
        performance_sample=True,
    ),
    UpdateMatrixCase(
        name="native_update_dispatch_promotion_perf",
        label="Native dispatch promotion perf sample",
        description=(
            "Measures explicit native dispatch with the minimum current-run arming probes; "
            "drops checkpoint and replay diagnostics while reducing shadow comparison sampling."
        ),
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-compare-sample-params",
            "8",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-copyback-dispatch-experimental",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-owner-native-launch-probe",
            "--turbocore-update-shadow-owner-native-launch-max-numel",
            OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL,
            "--turbocore-update-shadow-owner-native-event-chain-probe",
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "native_experimental",
            "--turbocore-native-update-required-shadow-passes",
            "2",
            "--turbocore-native-update-allow-missing-kernel",
            "--turbocore-native-update-dispatch-enabled",
            "--turbocore-native-update-training-path-enabled",
            "--turbocore-native-update-require-native-cuda",
            "--turbocore-native-update-defer-state-sync",
        ),
        evidence_role="performance",
        performance_sample=True,
    ),
    UpdateMatrixCase(
        name="native_update_dispatch_ctx_sync_free_canary",
        label="Native dispatch ctx-sync-free canary",
        description=(
            "Explicit borrowed-stream event-chain canary for measuring ctx-sync-free native dispatch; "
            "kept default-off and outside representative promotion priority."
        ),
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-compare-sample-params",
            "8",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-copyback-dispatch-experimental",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-owner-native-launch-probe",
            "--turbocore-update-shadow-owner-native-launch-max-numel",
            OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL,
            "--turbocore-update-shadow-owner-native-event-chain-probe",
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "native_experimental",
            "--turbocore-native-update-required-shadow-passes",
            "2",
            "--turbocore-native-update-allow-missing-kernel",
            "--turbocore-native-update-dispatch-enabled",
            "--turbocore-native-update-training-path-enabled",
            "--turbocore-native-update-require-native-cuda",
            "--turbocore-native-update-defer-state-sync",
            "--turbocore-native-update-runtime-synchronization-policy",
            "borrowed_stream_event_chain",
        ),
        evidence_role="performance_canary",
        performance_sample=True,
    ),
    UpdateMatrixCase(
        name="owner_native_launch_small",
        label="Owner-native launch diagnostic",
        description="Raises the owner-native launch probe cap for small LoRA owners so diagnostic CUDA launch/parity can be measured without enabling dispatch.",
        flags=(
            "--turbocore-update-shadow",
            "shadow",
            "--turbocore-update-shadow-direct-grad",
            "--turbocore-update-shadow-compare-sample-params",
            "32",
            "--turbocore-update-shadow-stop-after-consecutive-passes",
            "2",
            "--turbocore-update-shadow-checkpoint-contract",
            "--turbocore-update-shadow-copyback-probe",
            "--turbocore-update-shadow-copyback-dispatch-experimental",
            "--turbocore-update-shadow-native-binding-probe",
            "--turbocore-update-shadow-owner-native-launch-probe",
            "--turbocore-update-shadow-owner-native-launch-max-numel",
            OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL,
            "--turbocore-update-shadow-save-owner-state",
            "--turbocore-native-update-mode",
            "profile",
            "--turbocore-native-update-required-shadow-passes",
            "2",
        ),
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _benchmark_script() -> Path:
    return Path(__file__).with_name("native_runtime_profile_benchmark.py")


def _case_map() -> dict[str, UpdateMatrixCase]:
    return {case.name: case for case in MATRIX_CASES}


def _command_for_case(
    case: UpdateMatrixCase,
    *,
    python: Path,
    repo: Path,
    out_root: Path,
    family: str,
    profiles: list[str],
    steps: int,
    steady_warmup: int,
    samples: int,
    resolution: int,
    network_dim: int,
    train_batch_size: int,
    source_data: Path,
) -> list[str]:
    return [
        str(python),
        str(_benchmark_script()),
        "--family",
        family,
        "--profiles",
        *profiles,
        "--steps",
        str(max(int(steps), 1)),
        "--steady-warmup",
        str(max(int(steady_warmup), 0)),
        "--samples",
        str(max(int(samples), 1)),
        "--resolution",
        str(max(int(resolution), 1)),
        "--network-dim",
        str(max(int(network_dim), 1)),
        "--train-batch-size",
        str(max(int(train_batch_size), 1)),
        "--fused-adamw",
        "--phase-profile",
        "--source-data",
        str(source_data),
        "--out",
        str(out_root / case.name),
        *case.flags,
    ]


def _summary_path(out_root: Path, case: UpdateMatrixCase, family: str) -> Path:
    return out_root / case.name / f"{family}_summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_shadow_probe(shadow_reports: list[Any], key: str, score_fn) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    for item in shadow_reports:
        report = item if isinstance(item, dict) else {}
        probe = report.get(key) if isinstance(report.get(key), dict) else {}
        if not probe:
            continue
        score = int(score_fn(probe))
        if score > best_score:
            best = dict(probe)
            best_score = score
    return best


def _best_after_optimizer_shadow(shadow_reports: list[Any]) -> dict[str, Any]:
    def _score(after: dict[str, Any]) -> int:
        return (
            (8 if bool(after.get("parity_ok_loose", False)) else 0)
            + (4 if bool(after.get("compared", False)) else 0)
            + (2 if bool(after.get("sampled", False)) else 0)
            + (1 if bool(after.get("auto_stopped_after_this_step", False)) else 0)
        )

    return _best_shadow_probe(shadow_reports, "after_optimizer", _score)


def _best_probe_shadow_report(shadow_reports: list[Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_after = _best_after_optimizer_shadow(shadow_reports)
    if best_after:
        best["after_optimizer"] = best_after
    owner_native = _best_shadow_probe(shadow_reports, "owner_native_launch_probe", _owner_native_probe_score)
    if owner_native:
        best["owner_native_launch_probe"] = owner_native
    copyback = _best_shadow_probe(
        shadow_reports,
        "copyback_probe",
        lambda probe: (4 if probe.get("scratch_copyback_validated") else 0)
        + (2 if not probe.get("real_parameters_mutated", False) else 0),
    )
    if copyback:
        best["copyback_probe"] = copyback
    native_binding = _best_shadow_probe(
        shadow_reports,
        "native_binding_probe",
        lambda probe: sum(
            1
            for key in (
                "request_shape_ready",
                "tensor_object_binding_ready",
                "launch_plan_ready",
                "stream_lifetime_bound",
                "event_chain_verified",
                "pre_launch_ordering_verified",
                "post_launch_ordering_verified",
                "stream_wait_event_verified",
            )
            if bool(probe.get(key, False))
        ),
    )
    if native_binding:
        best["native_binding_probe"] = native_binding
    direct_grad = _best_shadow_probe(
        shadow_reports,
        "direct_grad_audit",
        lambda probe: (4 if probe.get("parity_ok") is True else 0)
        + min(int((probe.get("snapshot") or {}).get("writes", 0) or 0), 3),
    )
    if direct_grad:
        best["direct_grad_audit"] = direct_grad
    checkpoint = _best_shadow_probe(
        shadow_reports,
        "checkpoint_contract",
        lambda probe: 2 if probe.get("roundtrip_ok") is True else 0,
    )
    if checkpoint:
        best["checkpoint_contract"] = checkpoint
    return best


def _shadow_auto_stopped(shadow_reports: list[Any]) -> bool:
    for item in shadow_reports:
        report = item if isinstance(item, dict) else {}
        after = report.get("after_optimizer") if isinstance(report.get("after_optimizer"), dict) else {}
        if bool(after.get("auto_stopped_after_this_step", False)):
            return True
        if str(report.get("reason", "") or after.get("reason", "") or "") == "auto_stopped_after_consecutive_passes":
            return True
    return False


def _summarize_benchmark_summary(
    summary: dict[str, Any],
    *,
    profile: str = "standard",
    optimizer_performance_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = ((summary.get("runs") or {}).get(profile) or {}) if isinstance(summary, dict) else {}
    phase = run.get("steady_phase_summary") or {}
    shadow_reports = run.get("update_shadow_reports") if isinstance(run.get("update_shadow_reports"), list) else []
    gate_reports = run.get("native_update_gate_reports") if isinstance(run.get("native_update_gate_reports"), list) else []
    readiness = run.get("native_update_readiness") if isinstance(run.get("native_update_readiness"), dict) else {}
    last_shadow = shadow_reports[-1] if shadow_reports else {}
    last_after = last_shadow.get("after_optimizer") if isinstance(last_shadow.get("after_optimizer"), dict) else {}
    best_shadow = _best_probe_shadow_report(shadow_reports)
    best_after = best_shadow.get("after_optimizer") if isinstance(best_shadow.get("after_optimizer"), dict) else last_after
    last_gate = gate_reports[-1] if gate_reports else {}
    retained_gate_reports = [
        item for item in gate_reports
        if isinstance(item, dict) and bool(item.get("retained_probe_evidence", False))
    ]
    direct_grad = best_shadow.get("direct_grad_audit") if isinstance(best_shadow.get("direct_grad_audit"), dict) else {}
    checkpoint = best_shadow.get("checkpoint_contract") if isinstance(best_shadow.get("checkpoint_contract"), dict) else {}
    copyback = best_shadow.get("copyback_probe") if isinstance(best_shadow.get("copyback_probe"), dict) else {}
    native_binding = best_shadow.get("native_binding_probe") if isinstance(best_shadow.get("native_binding_probe"), dict) else {}
    owner_native = best_shadow.get("owner_native_launch_probe") if isinstance(best_shadow.get("owner_native_launch_probe"), dict) else {}
    dispatch_runtime_reports = run.get("native_update_dispatch_runtime_reports") if isinstance(run.get("native_update_dispatch_runtime_reports"), list) else []
    native_update_loop_timings = run.get("native_update_loop_timings") if isinstance(run.get("native_update_loop_timings"), list) else []
    native_timing_summary = summarize_native_update_timing(dispatch_runtime_reports, native_update_loop_timings)
    last_dispatch_runtime = dispatch_runtime_reports[-1] if dispatch_runtime_reports and isinstance(dispatch_runtime_reports[-1], dict) else {}
    native_dispatch_requested = any(
        bool(item.get("requested", False)) for item in dispatch_runtime_reports if isinstance(item, dict)
    )
    native_dispatch_executed = any(
        bool(item.get("native_step_executed", False)) for item in dispatch_runtime_reports if isinstance(item, dict)
    )
    native_dispatch_runtime_blockers = (
        list(last_dispatch_runtime.get("blocked_reasons", []))
        if isinstance(last_dispatch_runtime.get("blocked_reasons"), list)
        else []
    )
    requested_sync_policy = benchmark_requested_sync_policy(summary)
    training_executor_audit = audit_training_executor_reports(dispatch_runtime_reports)
    borrowed_stream_requested = requested_sync_policy == BORROWED_STREAM_POLICY
    execution_plan = last_dispatch_runtime.get("execution_plan") if isinstance(last_dispatch_runtime.get("execution_plan"), dict) else {}
    executor_probe = last_dispatch_runtime.get("executor_probe") if isinstance(last_dispatch_runtime.get("executor_probe"), dict) else {}
    arming_reports = run.get("native_update_dispatch_arming_reports") if isinstance(run.get("native_update_dispatch_arming_reports"), list) else []
    retained_arming_reports = [
        item for item in arming_reports
        if isinstance(item, dict) and bool(item.get("retained_probe_evidence", False))
    ]
    last_arming = arming_reports[-1] if arming_reports and isinstance(arming_reports[-1], dict) else {}
    arming_preconditions = last_arming.get("promotion_preconditions") if isinstance(last_arming.get("promotion_preconditions"), dict) else {}
    recovery_observations = run.get("native_update_runtime_recovery_observations") if isinstance(run.get("native_update_runtime_recovery_observations"), list) else []
    last_recovery_observation = recovery_observations[-1] if recovery_observations and isinstance(recovery_observations[-1], dict) else {}
    recovery_observation_integration = last_recovery_observation.get("integration") if isinstance(last_recovery_observation.get("integration"), dict) else {}
    diagnostic_replay_reports = run.get("native_update_diagnostic_replay_reports") if isinstance(run.get("native_update_diagnostic_replay_reports"), list) else []
    last_diagnostic_replay = diagnostic_replay_reports[-1] if diagnostic_replay_reports and isinstance(diagnostic_replay_reports[-1], dict) else {}
    diagnostic_replay_plan = last_diagnostic_replay.get("execution_plan") if isinstance(last_diagnostic_replay.get("execution_plan"), dict) else {}
    diagnostic_replay_probe = last_diagnostic_replay.get("executor_probe") if isinstance(last_diagnostic_replay.get("executor_probe"), dict) else {}
    dispatch_request = last_gate.get("dispatch_request") if isinstance(last_gate.get("dispatch_request"), dict) else {}
    performance_gate = _profile_performance_gate(
        summary,
        readiness=readiness,
        shadow=best_shadow or last_shadow,
        optimizer_performance_gate=optimizer_performance_gate,
    )
    return {
        "success": bool(run.get("success", False)),
        "steps_completed": int(run.get("steps_completed", 0) or 0),
        "mean_step_ms": float(run.get("mean_step_ms", 0.0) or 0.0),
        "steady_mean_step_ms": float(run.get("steady_mean_step_ms", 0.0) or 0.0),
        "peak_vram_mb": float(run.get("peak_vram_mb", 0.0) or 0.0),
        "optimizer_step_share_pct": float(phase.get("optimizer_step_share_mean", 0.0) or 0.0) * 100.0,
        "optimizer_update_share_pct": float(phase.get("optimizer_update_share_mean", 0.0) or 0.0) * 100.0,
        "shadow_reports": len(shadow_reports),
        "shadow_sampled": bool(best_after.get("sampled", False)),
        "shadow_sample_parameter_tensors": int(best_after.get("sample_parameter_tensors", 0) or 0),
        "shadow_total_parameter_tensors": int(best_after.get("total_parameter_tensors", 0) or 0),
        "shadow_auto_stopped": _shadow_auto_stopped(shadow_reports),
        "shadow_max_abs_param_diff": _float_or_none(best_after.get("max_abs_param_diff")),
        "direct_grad_parity_ok": direct_grad.get("parity_ok") if direct_grad else None,
        "direct_grad_writes": int(((direct_grad.get("snapshot") or {}).get("writes", 0) if direct_grad else 0) or 0),
        "checkpoint_roundtrip_ok": checkpoint.get("roundtrip_ok") if checkpoint else None,
        "copyback_probe_ms": _float_or_none(copyback.get("elapsed_ms") if copyback else None),
        "copyback_scratch_validated": copyback.get("scratch_copyback_validated") if copyback else None,
        "copyback_scratch_max_abs_diff": _float_or_none(copyback.get("scratch_max_abs_diff") if copyback else None),
        "copyback_real_parameters_mutated": copyback.get("real_parameters_mutated") if copyback else None,
        "native_binding_request_shape_ready": native_binding.get("request_shape_ready") if native_binding else None,
        "native_binding_tensor_object_ready": native_binding.get("tensor_object_binding_ready") if native_binding else None,
        "native_binding_launch_plan_ready": native_binding.get("launch_plan_ready") if native_binding else None,
        "native_binding_stream_lifetime_bound": native_binding.get("stream_lifetime_bound") if native_binding else None,
        "native_binding_stream_lifetime_ownership_bound": native_binding.get("stream_lifetime_bound") if native_binding else None,
        "native_binding_event_chain_verified": native_binding.get("event_chain_verified") if native_binding else None,
        "native_binding_pre_launch_ordering_verified": native_binding.get("pre_launch_ordering_verified") if native_binding else None,
        "native_binding_post_launch_ordering_verified": native_binding.get("post_launch_ordering_verified") if native_binding else None,
        "native_binding_stream_wait_event_verified": native_binding.get("stream_wait_event_verified") if native_binding else None,
        "native_binding_probe_ms": _float_or_none(native_binding.get("elapsed_ms") if native_binding else None),
        "owner_native_launch_probe_present": bool(owner_native),
        "owner_native_launch_attempted": bool(owner_native.get("attempted", False)) if owner_native else None,
        "owner_native_launch_ok": bool(owner_native.get("ok", False)) if owner_native else None,
        "owner_native_kernel_executed": bool(owner_native.get("kernel_executed", False)) if owner_native else None,
        "owner_native_parity_ok": bool(owner_native.get("parity_ok", False)) if owner_native else None,
        "owner_native_event_chain_verified": bool(owner_native.get("event_chain_verified", False)) if owner_native else None,
        "owner_native_pre_launch_ordering_verified": bool(owner_native.get("pre_launch_ordering_verified", False)) if owner_native else None,
        "owner_native_post_launch_ordering_verified": bool(owner_native.get("post_launch_ordering_verified", False)) if owner_native else None,
        "owner_native_stream_wait_event_verified": bool(owner_native.get("stream_wait_event_verified", False)) if owner_native else None,
        "owner_native_numel": int(owner_native.get("owner_numel", 0) or 0) if owner_native else None,
        "owner_native_elapsed_ms": _float_or_none(owner_native.get("elapsed_ms") if owner_native else None),
        "native_dispatch_requested_runtime_synchronization_policy": requested_sync_policy,
        "native_dispatch_requested_borrowed_stream_event_chain": borrowed_stream_requested,
        "native_dispatch_gate_requested": bool(dispatch_request.get("requested", False)),
        "native_dispatch_requested": native_dispatch_requested,
        "native_dispatch_executed": native_dispatch_executed,
        "native_dispatch_runtime_reports": len(dispatch_runtime_reports),
        "native_dispatch_runtime_blocked_reasons": native_dispatch_runtime_blockers,
        "native_dispatch_arming_reports": len(arming_reports),
        "native_dispatch_probe_cache_retained": bool(retained_gate_reports or retained_arming_reports),
        "native_dispatch_probe_cache_retained_gate_reports": len(retained_gate_reports),
        "native_dispatch_probe_cache_retained_arming_reports": len(retained_arming_reports),
        "native_dispatch_probe_cache_source": str(last_gate.get("probe_cache_source", "") or last_arming.get("probe_cache_source", "") or ""),
        "native_dispatch_probe_cache_reused_steps": int(
            last_gate.get("probe_cache_reused_steps", last_arming.get("probe_cache_reused_steps", 0)) or 0
        ),
        "native_dispatch_runtime_recovery_observations": len(recovery_observations),
        "native_dispatch_recovery_observation_bridge_ready": bool(
            recovery_observation_integration.get("recovery_observation_bridge_ready", False)
            or recovery_observation_integration.get("default_off_recovery_bridge_ready", False)
        ) if recovery_observation_integration else None,
        "native_dispatch_recovery_runtime_disabled_for_run": bool(last_recovery_observation.get("disabled_for_run", False)) if last_recovery_observation else None,
        "native_dispatch_training_dispatch_recovery_ready": bool(arming_preconditions.get("training_dispatch_recovery_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_dispatch_recovery_blocked": bool(arming_preconditions.get("training_dispatch_recovery_blocked", False)) if arming_preconditions else None,
        "native_dispatch_direct_gradient_write_boundary_ready": bool(arming_preconditions.get("direct_gradient_write_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_direct_gradient_write_native_supported": bool(arming_preconditions.get("direct_gradient_write_native_supported", False)) if arming_preconditions else None,
        "native_dispatch_direct_gradient_write_lifecycle_ready": bool(arming_preconditions.get("direct_gradient_write_lifecycle_ready", False)) if arming_preconditions else None,
        "native_dispatch_direct_gradient_write_bound": bool(arming_preconditions.get("direct_gradient_write_bound", False)) if arming_preconditions else None,
        "native_dispatch_direct_gradient_write_default_off": bool(arming_preconditions.get("direct_gradient_write_default_off", False)) if arming_preconditions else None,
        "native_dispatch_stream_ordering_ready": bool(arming_preconditions.get("stream_ordering_ready", False)) if arming_preconditions else None,
        "native_dispatch_stream_lifetime_ownership_boundary_ready": bool(arming_preconditions.get("stream_lifetime_ownership_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_stream_lifetime_ownership_evidence_bound": bool(arming_preconditions.get("stream_lifetime_ownership_evidence_bound", False)) if arming_preconditions else None,
        "native_dispatch_stream_lifetime_ownership_default_off": bool(arming_preconditions.get("stream_lifetime_ownership_default_off", False)) if arming_preconditions else None,
        "native_dispatch_stream_lifetime_ownership_ready": bool(arming_preconditions.get("stream_lifetime_ownership_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_flat_owner_boundary_ready": bool(arming_preconditions.get("training_flat_owner_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_flat_owner_reference_ready": bool(arming_preconditions.get("training_flat_owner_reference_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_flat_owner_bound": bool(arming_preconditions.get("training_flat_owner_bound", False)) if arming_preconditions else None,
        "native_dispatch_training_flat_owner_default_off": bool(arming_preconditions.get("training_flat_owner_default_off", False)) if arming_preconditions else None,
        "native_dispatch_training_dispatch_kernel_boundary_ready": bool(arming_preconditions.get("training_dispatch_kernel_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_dispatch_kernel_evidence_present": bool(arming_preconditions.get("training_dispatch_kernel_evidence_present", False)) if arming_preconditions else None,
        "native_dispatch_training_dispatch_kernel_bound": bool(arming_preconditions.get("training_dispatch_kernel_bound", False)) if arming_preconditions else None,
        "native_dispatch_training_dispatch_kernel_default_off": bool(arming_preconditions.get("training_dispatch_kernel_default_off", False)) if arming_preconditions else None,
        "native_dispatch_training_runtime_executor_boundary_ready": bool(arming_preconditions.get("training_runtime_executor_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_training_runtime_executor_bound": bool(arming_preconditions.get("training_runtime_executor_bound", False)) if arming_preconditions else None,
        "native_dispatch_training_runtime_executor_default_off": bool(arming_preconditions.get("training_runtime_executor_default_off", False)) if arming_preconditions else None,
        "native_dispatch_training_path_request_boundary_ready": bool(arming_preconditions.get("training_path_request_boundary_ready", False)) if arming_preconditions else None,
        "native_dispatch_explicit_training_path_requested": bool(arming_preconditions.get("explicit_training_path_requested", False)) if arming_preconditions else None,
        "native_dispatch_training_path_default_off": bool(arming_preconditions.get("training_path_default_off", False)) if arming_preconditions else None,
        "native_dispatch_rehearsal_evidence_ready": bool(last_arming.get("native_dispatch_rehearsal_evidence_ready", False)) if last_arming else None,
        "native_dispatch_training_promotion_preconditions_ready": bool(last_arming.get("native_dispatch_training_promotion_preconditions_ready", False)) if last_arming else None,
        "native_dispatch_missing_for_training_promotion": list(arming_preconditions.get("missing_for_training_promotion", [])) if isinstance(arming_preconditions.get("missing_for_training_promotion"), list) else [],
        "native_dispatch_disabled_for_run": bool((last_dispatch_runtime.get("state") or {}).get("disabled_for_run", False)) if isinstance(last_dispatch_runtime.get("state"), dict) else False,
        "native_dispatch_disable_reason": str((last_dispatch_runtime.get("state") or {}).get("disable_reason", "") or "") if isinstance(last_dispatch_runtime.get("state"), dict) else "",
        "native_dispatch_execution_plan_present": bool(execution_plan),
        "native_dispatch_executor_preconditions_ready": bool(execution_plan.get("executor_preconditions_ready", False)) if execution_plan else None,
        "native_dispatch_training_executor_preconditions_ready": bool(execution_plan.get("training_executor_preconditions_ready", False)) if execution_plan else None,
        "native_dispatch_diagnostic_executor_preconditions_ready": bool(execution_plan.get("diagnostic_executor_preconditions_ready", False)) if execution_plan else None,
        "native_dispatch_execution_allowed": bool(execution_plan.get("execution_allowed", False)) if execution_plan else None,
        "native_dispatch_execution_blocked_reasons": list(execution_plan.get("blocked_reasons", [])) if isinstance(execution_plan.get("blocked_reasons"), list) else [],
        "native_dispatch_diagnostic_executor_blocked_reasons": list(execution_plan.get("diagnostic_executor_blocked_reasons", [])) if isinstance(execution_plan.get("diagnostic_executor_blocked_reasons"), list) else [],
        "native_dispatch_executor_probe_present": bool(executor_probe),
        "native_dispatch_executor_probe_called": bool(executor_probe.get("called", False)) if executor_probe else None,
        "native_dispatch_executor_probe_ok": bool(executor_probe.get("ok", False)) if executor_probe else None,
        "native_dispatch_executor_probe_blocked_reasons": list(executor_probe.get("blocked_reasons", [])) if isinstance(executor_probe.get("blocked_reasons"), list) else [],
        "native_dispatch_diagnostic_replay_reports": len(diagnostic_replay_reports),
        "native_dispatch_diagnostic_replay_called": bool(diagnostic_replay_probe.get("called", False)) if diagnostic_replay_probe else None,
        "native_dispatch_diagnostic_replay_ok": bool(diagnostic_replay_probe.get("ok", False)) if diagnostic_replay_probe else None,
        "native_dispatch_diagnostic_replay_native_step_executed": any(
            bool(item.get("native_step_executed", False)) for item in diagnostic_replay_reports if isinstance(item, dict)
        ),
        "native_dispatch_diagnostic_replay_training_path_enabled": any(
            bool(item.get("training_path_enabled", False)) for item in diagnostic_replay_reports if isinstance(item, dict)
        ),
        "native_dispatch_diagnostic_replay_preconditions_ready": bool(diagnostic_replay_plan.get("diagnostic_executor_preconditions_ready", False)) if diagnostic_replay_plan else None,
        "native_dispatch_diagnostic_replay_blocked_reasons": list(diagnostic_replay_probe.get("blocked_reasons", [])) if isinstance(diagnostic_replay_probe.get("blocked_reasons"), list) else [],
        "native_dispatch_training_executor_attempted_reports": int(
            training_executor_audit.get("attempted_reports", 0) or 0
        ),
        "native_dispatch_training_executor_called_reports": int(
            training_executor_audit.get("called_reports", 0) or 0
        ),
        "native_dispatch_training_executor_ok_reports": int(training_executor_audit.get("ok_reports", 0) or 0),
        "native_dispatch_training_executor_last_reason": str(training_executor_audit.get("last_reason", "") or ""),
        "native_dispatch_training_executor_last_error": str(training_executor_audit.get("last_error", "") or ""),
        "native_dispatch_training_executor_blocked_reasons": list(
            training_executor_audit.get("blocked_reasons", [])
        ),
        "native_dispatch_borrowed_stream_native_policy_blocked_reasons": list(
            training_executor_audit.get("native_policy_blocked_reasons", [])
        ),
        "native_dispatch_borrowed_stream_launch_evidence_blocked_reasons": list(
            training_executor_audit.get("launch_evidence_blocked_reasons", [])
        ),
        "native_dispatch_borrowed_stream_stream_guard_blocked_reasons": list(
            training_executor_audit.get("stream_guard_blocked_reasons", [])
        ),
        "native_dispatch_borrowed_stream_lease_blocked_reasons": list(
            training_executor_audit.get("lease_blocked_reasons", [])
        ),
        "native_dispatch_borrowed_stream_native_policy_allowed": training_executor_audit.get(
            "native_policy_allowed"
        ),
        "native_dispatch_borrowed_stream_runtime_stream_guard_evidence_ready": training_executor_audit.get(
            "runtime_stream_guard_evidence_ready"
        ),
        "native_dispatch_borrowed_stream_runtime_stream_lifetime_lease_ready": training_executor_audit.get(
            "runtime_stream_lifetime_lease_ready"
        ),
        "native_dispatch_borrowed_stream_stream_handle_nonzero": training_executor_audit.get(
            "stream_handle_nonzero"
        ),
        "native_dispatch_borrowed_stream_event_chain_verified": training_executor_audit.get(
            "event_chain_verified"
        ),
        "native_dispatch_borrowed_stream_lifetime_bound": training_executor_audit.get(
            "stream_lifetime_bound"
        ),
        "native_dispatch_borrowed_stream_blocked_before_native_step": bool(
            borrowed_stream_requested and native_dispatch_requested and not native_dispatch_executed
        ),
        "native_dispatch_borrowed_stream_policy_not_allowed": bool(
            training_executor_audit.get("policy_not_allowed", False)
        ),
        **native_timing_summary,
        "gate_reports": len(gate_reports),
        "gate_would_enable": bool(last_gate.get("would_enable_native_update", False)),
        "gate_blocked_reasons": list(last_gate.get("blocked_reasons", [])) if isinstance(last_gate.get("blocked_reasons"), list) else [],
        "performance_gate_ready": bool(performance_gate.get("representative_performance_gate_ready", False)),
        "performance_gate_blocked_reasons": list(performance_gate.get("blocked_reasons", [])) if isinstance(performance_gate.get("blocked_reasons"), list) else [],
        "readiness_blockers": list(readiness.get("blocked_reasons", [])) if isinstance(readiness.get("blocked_reasons"), list) else [],
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_performance_gate(
    summary: dict[str, Any],
    *,
    readiness: dict[str, Any],
    shadow: dict[str, Any],
    optimizer_performance_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    performance_report = summary.get("native_update_performance_report") if isinstance(summary.get("native_update_performance_report"), dict) else {}
    performance_gate = performance_report.get("performance_gate") if isinstance(performance_report.get("performance_gate"), dict) else {}
    optimizer_gate = dict(optimizer_performance_gate or {})
    if not optimizer_gate:
        return dict(performance_gate)
    matrix = performance_report.get("benchmark_matrix") if isinstance(performance_report.get("benchmark_matrix"), dict) else {}
    evidence: dict[str, Any] = {"optimizer_performance_gate": optimizer_gate}
    if matrix:
        evidence["benchmark_matrix"] = matrix
    return build_native_update_performance_gate(
        readiness_report=readiness,
        shadow_report=shadow,
        performance_report=evidence,
    )


def build_matrix_payload(args: argparse.Namespace, *, run: bool) -> dict[str, Any]:
    repo = _repo_root()
    out_root = Path(args.out) if args.out else repo / "temp" / "turbocore_update_benchmark_matrix" / time.strftime("%Y%m%d-%H%M%S")
    source_data = Path(args.source_data) if args.source_data else repo / "sucai" / "6_lulu"
    python = Path(args.python) if args.python else Path(sys.executable)
    case_map = _case_map()
    selected_names = list(args.cases or [case.name for case in MATRIX_CASES])
    cases = [case_map[name] for name in selected_names]
    profiles = list(args.profiles or ["standard"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": bool(run),
        "out_root": str(out_root),
        "family": str(args.family),
        "profiles": profiles,
        "cases": [],
    }
    optimizer_report_path = str(getattr(args, "optimizer_performance_report", "") or "").strip()
    if optimizer_report_path:
        payload["optimizer_performance_artifact"] = load_optimizer_performance_artifact(optimizer_report_path)
    optimizer_gate = _optimizer_gate_from_artifact(_optimizer_artifact(payload))
    if run:
        out_root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        command = _command_for_case(
            case,
            python=python,
            repo=repo,
            out_root=out_root,
            family=str(args.family),
            profiles=profiles,
            steps=int(args.steps),
            steady_warmup=int(args.steady_warmup),
            samples=int(args.samples),
            resolution=int(args.resolution),
            network_dim=int(args.network_dim),
            train_batch_size=int(args.train_batch_size),
            source_data=source_data,
        )
        entry: dict[str, Any] = {
            "case": case.as_dict(),
            "command": command,
            "command_text": _join_command(command),
            "summary_path": str(_summary_path(out_root, case, str(args.family))),
        }
        if run:
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=str(repo), text=True)
            entry["returncode"] = int(completed.returncode)
            entry["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            summary_path = _summary_path(out_root, case, str(args.family))
            if summary_path.exists():
                entry["summary"] = _summarize_benchmark_summary(
                    _load_json(summary_path),
                    profile=profiles[0],
                    optimizer_performance_gate=optimizer_gate,
                )
            else:
                entry["summary_missing"] = True
        payload["cases"].append(entry)
        if run and int(entry.get("returncode", 0)) != 0 and not bool(args.keep_going):
            payload["stopped_after_failure"] = case.name
            break
    payload["summary"] = _summarize_matrix(payload)
    payload["native_update_performance_report"] = _build_matrix_performance_report(payload)
    payload["summary"]["native_update_performance_gate"] = _performance_report_summary(
        payload["native_update_performance_report"]
    )
    if run or bool(args.write_dry_run):
        out_root.mkdir(parents=True, exist_ok=True)
        summary_path = out_root / "matrix_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["matrix_summary_path"] = str(summary_path)
    return payload


def _summarize_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    entries = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    executed = [entry for entry in entries if "returncode" in entry]
    summaries = [entry.get("summary") for entry in executed if isinstance(entry.get("summary"), dict)]
    summary = {
        "case_count": len(entries),
        "executed_count": len(executed),
        "all_success": all(int(entry.get("returncode", 1)) == 0 for entry in executed) if executed else None,
        "mean_step_ms_by_case": {
            str(entry.get("case", {}).get("name")): round(float(entry.get("summary", {}).get("mean_step_ms", 0.0) or 0.0), 4)
            for entry in executed
            if isinstance(entry.get("summary"), dict)
        },
        "peak_vram_mb_by_case": {
            str(entry.get("case", {}).get("name")): round(float(entry.get("summary", {}).get("peak_vram_mb", 0.0) or 0.0), 1)
            for entry in executed
            if isinstance(entry.get("summary"), dict)
        },
        "gate_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list(entry.get("summary", {}).get("gate_blocked_reasons", []))
            for entry in executed
            if isinstance(entry.get("summary"), dict) and entry.get("summary", {}).get("gate_blocked_reasons")
        },
        "performance_gate_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list(entry.get("summary", {}).get("performance_gate_blocked_reasons", []))
            for entry in executed
            if isinstance(entry.get("summary"), dict) and entry.get("summary", {}).get("performance_gate_blocked_reasons")
        },
        "native_dispatch_requested_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_requested", False))
        ],
        "native_dispatch_gate_requested_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_gate_requested", False))
        ],
        "native_dispatch_executed_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_executed", False))
        ],
        "native_dispatch_owner_native_runtime_synchronization_by_case": {
            str(entry.get("case", {}).get("name")): str(
                (entry.get("summary") or {}).get("native_dispatch_owner_native_runtime_synchronization", "")
            )
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_owner_native_runtime_synchronization")
        },
        "native_dispatch_ctx_sync_free_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if str(
                (entry.get("summary") or {}).get("native_dispatch_owner_native_runtime_synchronization", "")
                or ""
            )
            == "borrowed_stream_event_chain_no_ctx_sync"
        ],
        "native_dispatch_context_synchronize_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if "synchronize" in str(
                (entry.get("summary") or {}).get("native_dispatch_owner_native_runtime_synchronization", "")
                or ""
            ).lower()
        ],
        **summarize_borrowed_stream_matrix(executed),
        **summarize_ctx_sync_free_matrix(executed),
        "native_dispatch_probe_cache_retained_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_probe_cache_retained", False))
        ],
        "native_dispatch_disabled_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_disabled_for_run", False))
        ],
        "native_dispatch_rehearsal_evidence_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_rehearsal_evidence_ready", False))
        ],
        "native_dispatch_recovery_observation_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_recovery_observation_bridge_ready", False))
        ],
        "native_dispatch_training_recovery_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_recovery_ready", False))
        ],
        "native_dispatch_training_recovery_blocked_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_recovery_blocked", False))
        ],
        "native_dispatch_direct_gradient_write_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_direct_gradient_write_boundary_ready", False))
        ],
        "native_dispatch_direct_gradient_write_native_supported_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_direct_gradient_write_native_supported", False))
        ],
        "native_dispatch_direct_gradient_write_lifecycle_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_direct_gradient_write_lifecycle_ready", False))
        ],
        "native_dispatch_direct_gradient_write_bound_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_direct_gradient_write_bound", False))
        ],
        "native_dispatch_direct_gradient_write_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_direct_gradient_write_default_off", False))
        ],
        "native_dispatch_stream_ordering_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_stream_ordering_ready", False))
        ],
        "native_dispatch_stream_lifetime_ownership_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_stream_lifetime_ownership_boundary_ready", False))
        ],
        "native_dispatch_stream_lifetime_ownership_evidence_bound_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_stream_lifetime_ownership_evidence_bound", False))
        ],
        "native_dispatch_stream_lifetime_ownership_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_stream_lifetime_ownership_default_off", False))
        ],
        "native_dispatch_stream_lifetime_ownership_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_stream_lifetime_ownership_ready", False))
        ],
        "native_dispatch_training_flat_owner_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_flat_owner_boundary_ready", False))
        ],
        "native_dispatch_training_flat_owner_reference_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_flat_owner_reference_ready", False))
        ],
        "native_dispatch_training_flat_owner_bound_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_flat_owner_bound", False))
        ],
        "native_dispatch_training_flat_owner_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_flat_owner_default_off", False))
        ],
        "native_dispatch_training_dispatch_kernel_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_kernel_boundary_ready", False))
        ],
        "native_dispatch_training_dispatch_kernel_evidence_present_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_kernel_evidence_present", False))
        ],
        "native_dispatch_training_dispatch_kernel_bound_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_kernel_bound", False))
        ],
        "native_dispatch_training_dispatch_kernel_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_dispatch_kernel_default_off", False))
        ],
        "native_dispatch_training_runtime_executor_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_runtime_executor_boundary_ready", False))
        ],
        "native_dispatch_training_runtime_executor_bound_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_runtime_executor_bound", False))
        ],
        "native_dispatch_training_runtime_executor_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_runtime_executor_default_off", False))
        ],
        "native_dispatch_training_path_request_boundary_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_path_request_boundary_ready", False))
        ],
        "native_dispatch_explicit_training_path_requested_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_explicit_training_path_requested", False))
        ],
        "native_dispatch_training_path_default_off_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_path_default_off", False))
        ],
        "native_dispatch_missing_for_training_promotion_by_case": {
            str(entry.get("case", {}).get("name")): list((entry.get("summary") or {}).get("native_dispatch_missing_for_training_promotion", []))
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_missing_for_training_promotion")
        },
        "native_dispatch_execution_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list((entry.get("summary") or {}).get("native_dispatch_execution_blocked_reasons", []))
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_execution_blocked_reasons")
        },
        "native_dispatch_training_executor_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_training_executor_preconditions_ready", False))
        ],
        "native_dispatch_diagnostic_executor_ready_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_diagnostic_executor_preconditions_ready", False))
        ],
        "native_dispatch_diagnostic_executor_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list((entry.get("summary") or {}).get("native_dispatch_diagnostic_executor_blocked_reasons", []))
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_diagnostic_executor_blocked_reasons")
        },
        "native_dispatch_executor_probe_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list((entry.get("summary") or {}).get("native_dispatch_executor_probe_blocked_reasons", []))
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_executor_probe_blocked_reasons")
        },
        "native_dispatch_diagnostic_replay_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if int((entry.get("summary") or {}).get("native_dispatch_diagnostic_replay_reports", 0) or 0) > 0
        ],
        "native_dispatch_diagnostic_replay_ok_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("native_dispatch_diagnostic_replay_ok", False))
        ],
        "native_dispatch_performance_sample_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("case") or {}).get("performance_sample", False))
        ],
        "native_dispatch_diagnostic_replay_blockers_by_case": {
            str(entry.get("case", {}).get("name")): list((entry.get("summary") or {}).get("native_dispatch_diagnostic_replay_blocked_reasons", []))
            for entry in executed
            if (entry.get("summary") or {}).get("native_dispatch_diagnostic_replay_blocked_reasons")
        },
        "shadow_auto_stopped_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if bool((entry.get("summary") or {}).get("shadow_auto_stopped", False))
        ],
        "direct_grad_cases": [
            str(entry.get("case", {}).get("name"))
            for entry in executed
            if (entry.get("summary") or {}).get("direct_grad_parity_ok") is True
        ],
        "summary_count": len(summaries),
    }
    optimizer_artifact = _optimizer_artifact(payload)
    if optimizer_artifact:
        summary["optimizer_performance_artifact"] = optimizer_artifact_summary(optimizer_artifact)
    return summary


def _build_matrix_performance_report(payload: dict[str, Any]) -> dict[str, Any]:
    performance_evidence: dict[str, Any] = {"benchmark_matrix": payload}
    optimizer_artifact = _optimizer_artifact(payload)
    optimizer_gate = _optimizer_gate_from_artifact(optimizer_artifact)
    if optimizer_gate:
        performance_evidence["optimizer_performance_gate"] = optimizer_gate
    gate = build_native_update_performance_gate(
        shadow_report=_matrix_shadow_evidence(payload),
        performance_report=performance_evidence,
    )
    report = {
        "schema_version": 1,
        "report": "turbocore_update_benchmark_matrix_performance_report_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "runtime_dispatch_allowed": False,
        "performance_gate": gate,
        "blocked_reasons": list(gate.get("blocked_reasons", [])) if isinstance(gate.get("blocked_reasons"), list) else [],
    }
    if optimizer_artifact:
        report["optimizer_performance_artifact"] = optimizer_artifact_summary(optimizer_artifact)
    return report


def _optimizer_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = payload.get("optimizer_performance_artifact")
    return dict(artifact) if isinstance(artifact, dict) else {}


def _optimizer_gate_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    gate = artifact.get("optimizer_performance_gate") if artifact else None
    return dict(gate) if isinstance(gate, dict) else {}


def _performance_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    gate = report.get("performance_gate") if isinstance(report.get("performance_gate"), dict) else {}
    optimizer = (gate.get("evidence") or {}).get("optimizer_microbenchmark") if isinstance(gate.get("evidence"), dict) else {}
    return {
        "ready": bool(gate.get("representative_performance_gate_ready", False)),
        "blocked_reasons": list(gate.get("blocked_reasons", [])) if isinstance(gate.get("blocked_reasons"), list) else [],
        "optimizer_evidence_present": bool((optimizer or {}).get("present", False)) if isinstance(optimizer, dict) else False,
        "optimizer_evidence_quality": str((optimizer or {}).get("evidence_quality", "") or "") if isinstance(optimizer, dict) else "",
    }


def _matrix_shadow_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    for entry in _executed_entries(payload):
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        if bool(summary.get("owner_native_launch_probe_present", False)):
            probe = {
                "ok": bool(summary.get("owner_native_launch_ok", False)),
                "attempted": bool(summary.get("owner_native_launch_attempted", False)),
                "kernel_executed": bool(summary.get("owner_native_kernel_executed", False)),
                "parity_ok": bool(summary.get("owner_native_parity_ok", False)),
                "persistent_owner_mutated": False,
                "owner_numel": int(summary.get("owner_native_numel", 0) or 0),
                "elapsed_ms": _float_or_none(summary.get("owner_native_elapsed_ms")),
                "source_case": str(entry.get("case", {}).get("name", "") or ""),
            }
            score = _owner_native_probe_score(probe)
            if score > best_score:
                best = probe
                best_score = score
    return {"owner_native_launch_probe": best} if best else {}


def _owner_native_probe_score(probe: dict[str, Any]) -> int:
    return (
        (8 if bool(probe.get("ok", False)) else 0)
        + (4 if bool(probe.get("kernel_executed", False)) else 0)
        + (2 if bool(probe.get("parity_ok", False)) else 0)
        + (1 if bool(probe.get("attempted", False)) else 0)
    )


def _executed_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    return [entry for entry in entries if isinstance(entry, dict) and "returncode" in entry]


def _join_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Actually execute benchmark cases. Default is dry-run.")
    parser.add_argument("--write-dry-run", action="store_true", help="Write matrix_summary.json even without --run.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a case fails.")
    parser.add_argument("--family", choices=("anima", "newbie", "sdxl"), default="anima")
    parser.add_argument("--cases", nargs="+", choices=tuple(_case_map().keys()), default=None)
    parser.add_argument("--profiles", nargs="+", choices=("standard", "aggressive"), default=("standard",))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--steady-warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--network-dim", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--source-data", default="")
    parser.add_argument("--python", default="")
    parser.add_argument(
        "--optimizer-performance-report",
        default="",
        help="Optional optimizer benchmark JSON with performance_gate evidence.",
    )
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = build_matrix_payload(args, run=bool(args.run))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if bool(args.run):
        return 0 if bool(payload.get("summary", {}).get("all_success", False)) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
