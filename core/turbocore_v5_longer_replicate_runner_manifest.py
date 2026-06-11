"""Explicit longer-replicate run manifest for TurboCore V5-P30.

This module turns a P29 owner next-stage package into a manual run plan. It
never launches training, never emits request-adapter fields, and never changes
the default training path.
"""

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

from core.turbocore_v5_owner_review_evidence_package import load_json


P29_READY_DECISION = "owner_next_stage_package_ready_default_off"
DEFAULT_MIN_RUNS = 5
DEFAULT_STEPS = 768
DEFAULT_SAMPLES = 3
DEFAULT_STEADY_WARMUP = 32
DEFAULT_MIN_SPEEDUP = 1.05
DEFAULT_MAX_SPREAD = 0.30
DEFAULT_CASES = ("baseline_phase", "native_update_dispatch_promotion_perf")
DEFAULT_PROFILES = ("standard",)
DEFAULT_PYTHON = "backend\\env\\python-flashattention\\python.exe"
DEFAULT_RUNNER = "backend\\core\\lulynx_trainer\\turbocore_update_benchmark_matrix.py"
DEFAULT_OUTPUT_ROOT = "temp\\turbocore_v5_p30_longer_replicate_manual_plan"


def build_v5_longer_replicate_runner_manifest(
    *,
    owner_next_stage_package: Mapping[str, Any] | None = None,
    run_plan_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = _as_dict(owner_next_stage_package)
    options = _normalize_options(_as_dict(run_plan_options))
    p29_summary = _p29_summary(package)
    rollback = _rollback_policy(options)
    plan = _run_plan(options=options, rollback=rollback)
    blocked = _dedupe(
        _p29_blockers(p29_summary)
        + _option_blockers(options)
        + _rollback_blockers(rollback)
    )
    ready = not blocked
    decision = (
        "longer_replicate_runner_manifest_ready_default_off"
        if ready
        else "longer_replicate_runner_manifest_blocked_default_off"
    )
    return {
        "schema_version": 1,
        "manifest": "turbocore_v5_longer_replicate_runner_manifest_v0",
        "gate": "v5_longer_replicate_runner_manifest",
        "ok": ready,
        "run_manifest_ready": ready,
        "explicit_run_plan_ready": ready,
        "decision": decision,
        "gate_decision": decision,
        "manual_run_required": True,
        "manual_execution_plan_ready": ready,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_manifest_request_fields": {},
        "owner_next_stage_package_summary": p29_summary,
        "run_plan_options": options,
        "run_plan": plan,
        "rollback_policy": rollback,
        "collector_followup": _collector_followup(options, plan),
        "audit_expectations": _audit_expectations(options),
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, p29_summary),
        "notes": [
            "This manifest is a manual run plan only; it never launches training.",
            "Run outputs should be fed back into the P28 collector after manual execution.",
            "Default rollout, auto launch, and request-adapter mapping remain disabled.",
        ],
    }


def _p29_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    summaries = _as_dict(package.get("evidence_summaries"))
    p28 = _as_dict(summaries.get("p28_evidence_bundle"))
    p26 = _as_dict(summaries.get("p26_gate"))
    p27 = _as_dict(summaries.get("p27_decision"))
    return {
        "present": bool(package),
        "source_path": str(package.get("_source_path") or package.get("source_path") or ""),
        "ok": bool(package.get("ok", False)),
        "package_ready": bool(package.get("package_ready", False)),
        "ready_for_owner_archive": bool(package.get("ready_for_owner_archive", False)),
        "decision": str(package.get("decision") or package.get("gate_decision") or ""),
        "default_off": _default_off_confirmed(package),
        "request_adapter_off": _request_adapter_off(package),
        "post_fields_empty": not bool(_as_dict(package.get("post_package_request_fields"))),
        "p28_ready": bool(p28.get("ok", False)) and bool(p28.get("longer_replicate_evidence_ready", False)),
        "p28_run_count": int(p28.get("run_count", 0) or 0),
        "p28_ready_run_count": int(p28.get("ready_run_count", 0) or 0),
        "p28_min_speedup": p28.get("min_speedup"),
        "p28_speedup_spread_ratio": p28.get("speedup_spread_ratio"),
        "p26_ready": bool(p26.get("ok", False))
        and str(p26.get("decision") or "") == "longer_replicate_failure_history_review_ready",
        "p27_approved": bool(p27.get("ok", False))
        and str(p27.get("decision") or "") == "signed_next_stage_review_recorded_default_off",
        "p28_default_off": bool(p28.get("default_off", False)),
        "p26_default_off": bool(p26.get("default_off", False)),
        "p27_default_off": bool(p27.get("default_off", False)),
        "p28_request_adapter_off": bool(p28.get("request_adapter_off", False)),
        "p26_request_adapter_off": bool(p26.get("request_adapter_off", False)),
        "p27_request_adapter_off": bool(p27.get("request_adapter_off", False)),
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
    }


def _normalize_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    min_runs = _int_option(raw, ("min_runs", "run_count", "min_longer_replicate_runs"), DEFAULT_MIN_RUNS)
    steps = _int_option(raw, ("steps", "representative_steps", "min_representative_steps"), DEFAULT_STEPS)
    return {
        "plan_id": _str_option(raw, ("plan_id",), "turbocore_v5_p30_longer_replicate_manual_plan"),
        "family": _str_option(raw, ("family",), "anima"),
        "profiles": _string_list_option(raw, ("profiles", "profile"), list(DEFAULT_PROFILES)),
        "cases": _string_list_option(raw, ("cases", "case"), list(DEFAULT_CASES)),
        "min_runs": max(min_runs, 0),
        "steps": max(steps, 0),
        "samples": max(_int_option(raw, ("samples",), DEFAULT_SAMPLES), 0),
        "steady_warmup": max(_int_option(raw, ("steady_warmup", "steady-warmup"), DEFAULT_STEADY_WARMUP), 0),
        "resolution": max(_int_option(raw, ("resolution",), 64), 0),
        "network_dim": max(_int_option(raw, ("network_dim", "network-dim"), 1), 0),
        "train_batch_size": max(_int_option(raw, ("train_batch_size", "train-batch-size"), 1), 0),
        "source_data": _str_option(raw, ("source_data", "source-data"), "sucai\\6_lulu"),
        "python": _str_option(raw, ("python", "python_executable"), DEFAULT_PYTHON),
        "runner": _str_option(raw, ("runner", "benchmark_runner"), DEFAULT_RUNNER),
        "output_root": _str_option(raw, ("output_root", "out_root"), DEFAULT_OUTPUT_ROOT),
        "optimizer_report_root": _str_option(raw, ("optimizer_report_root",), ""),
        "min_end_to_end_speedup": _float_option(
            raw,
            ("min_end_to_end_speedup", "min_speedup", "min_representative_speedup"),
            DEFAULT_MIN_SPEEDUP,
        ),
        "max_speedup_spread_ratio": _float_option(
            raw,
            ("max_speedup_spread_ratio", "max_spread"),
            DEFAULT_MAX_SPREAD,
        ),
        "seeds": _seeds(raw.get("seeds"), min_runs),
        "auto_launch_requested": _bool_option(raw, ("auto_launch", "auto_launch_requested", "launch_training")),
        "default_training_requested": _bool_option(
            raw,
            ("default_training_path_enabled", "training_path_enabled", "default_training_requested"),
        ),
        "default_rollout_requested": _bool_option(
            raw,
            ("default_rollout_allowed", "auto_rollout_allowed", "default_rollout_requested"),
        ),
        "request_adapter_requested": _bool_option(
            raw,
            ("request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_requested"),
        ),
        "rollback_policy_acknowledged": raw.get("rollback_policy_acknowledged", True) is not False,
    }


def _run_plan(*, options: Mapping[str, Any], rollback: Mapping[str, Any]) -> dict[str, Any]:
    run_specs = [_run_spec(index=index, options=options) for index in range(1, int(options["min_runs"]) + 1)]
    return {
        "schema_version": 1,
        "plan_id": str(options["plan_id"]),
        "runner_kind": "turbocore_update_benchmark_matrix_manual",
        "manual_only": True,
        "auto_launch_allowed": False,
        "run_count": len(run_specs),
        "expected_summary_kind": "matrix_summary_json",
        "expected_collector": "build_v5_longer_replicate_evidence_bundle",
        "cases": list(options["cases"]),
        "profiles": list(options["profiles"]),
        "required_thresholds": {
            "min_runs": int(options["min_runs"]),
            "min_representative_steps": int(options["steps"]),
            "min_end_to_end_speedup": float(options["min_end_to_end_speedup"]),
            "max_speedup_spread_ratio": float(options["max_speedup_spread_ratio"]),
        },
        "rollback_triggers": list(rollback["rollback_triggers"]),
        "runs": run_specs,
    }


def _run_spec(*, index: int, options: Mapping[str, Any]) -> dict[str, Any]:
    profile_values = list(options["profiles"]) or list(DEFAULT_PROFILES)
    profile = str(profile_values[(index - 1) % len(profile_values)])
    output_root = str(options["output_root"]).rstrip("\\/")
    output_dir = f"{output_root}\\run_{index:02d}"
    optimizer_root = str(options["optimizer_report_root"] or f"{output_root}\\optimizer_reports").rstrip("\\/")
    optimizer_report = f"{optimizer_root}\\run_{index:02d}.json"
    cases = [str(item) for item in list(options["cases"])]
    args = [
        "--run",
        "--keep-going",
        "--family",
        str(options["family"]),
        "--cases",
        *cases,
        "--profiles",
        profile,
        "--steps",
        str(options["steps"]),
        "--steady-warmup",
        str(options["steady_warmup"]),
        "--samples",
        str(options["samples"]),
        "--resolution",
        str(options["resolution"]),
        "--network-dim",
        str(options["network_dim"]),
        "--train-batch-size",
        str(options["train_batch_size"]),
        "--source-data",
        str(options["source_data"]),
        "--optimizer-performance-report",
        optimizer_report,
        "--out",
        output_dir,
    ]
    seed_values = list(options["seeds"]) or _seeds(None, int(options["min_runs"]))
    seed = int(seed_values[(index - 1) % len(seed_values)])
    return {
        "schema_version": 1,
        "run_id": f"{options['plan_id']}_run_{index:02d}",
        "manual_only": True,
        "profile": profile,
        "seed": seed,
        "output_dir": output_dir,
        "matrix_summary_path": f"{output_dir}\\matrix_summary.json",
        "optimizer_performance_report": optimizer_report,
        "expected_cases": cases,
        "expected_steps": int(options["steps"]),
        "expected_runtime_evidence": _required_runtime_evidence(),
        "command": {
            "executable": str(options["python"]),
            "script": str(options["runner"]),
            "args": args,
        },
    }


def _collector_followup(options: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    run_paths = [
        str(_as_dict(item).get("matrix_summary_path") or "")
        for item in list(plan.get("runs", []) or [])
        if str(_as_dict(item).get("matrix_summary_path") or "")
    ]
    return {
        "schema_version": 1,
        "manual_step": "after all planned runs finish, feed matrix summaries into P28 collector",
        "collector_module": "core.turbocore_v5_longer_replicate_evidence_collector",
        "collector_function": "build_v5_longer_replicate_evidence_bundle",
        "run_summary_paths": run_paths,
        "thresholds": {
            "min_runs": int(options["min_runs"]),
            "min_representative_steps": int(options["steps"]),
            "min_end_to_end_speedup": float(options["min_end_to_end_speedup"]),
            "max_speedup_spread_ratio": float(options["max_speedup_spread_ratio"]),
        },
    }


def _audit_expectations(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "required_runtime_evidence": _required_runtime_evidence(),
        "required_run_result_fields": [
            "success",
            "steps_completed",
            "representative_end_to_end_speedup",
            "native_dispatch_executed",
            "checkpoint_resume_native_state_boundary",
            "rollback_events",
        ],
        "blocked_if": [
            "any run emits default/request-adapter fields",
            "any run reports rollback events",
            "any run misses native dispatch execution evidence",
            "aggregate min speedup or spread fails thresholds",
        ],
        "expected_min_runs": int(options["min_runs"]),
    }


def _rollback_policy(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": "v5_p30_longer_replicate_manual_run_rollback_policy_v0",
        "fallback_authoritative": True,
        "fallback_backend": "pytorch_adamw",
        "manual_abort_required_on_failure": True,
        "disable_for_run_on_native_error": True,
        "disable_for_run_on_state_sync_failure": True,
        "disable_for_run_on_checkpoint_resume_mismatch": True,
        "disable_for_run_on_config_mismatch": True,
        "disable_for_run_on_non_finite": True,
        "rollback_on_resume_mismatch": True,
        "rollback_on_performance_regression": True,
        "rollback_policy_acknowledged": bool(options.get("rollback_policy_acknowledged", False)),
        "default_training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "rollback_triggers": [
            "native_error",
            "state_sync_failure",
            "checkpoint_resume_mismatch",
            "config_mismatch",
            "non_finite",
            "performance_regression",
            "run_summary_missing",
            "p28_collector_blocked",
        ],
    }


def _p29_blockers(summary: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(summary.get("present", False)):
        blocked.append("v5_p30_p29_package_missing")
    if (
        not bool(summary.get("ok", False))
        or not bool(summary.get("package_ready", False))
        or not bool(summary.get("ready_for_owner_archive", False))
        or str(summary.get("decision") or "") != P29_READY_DECISION
    ):
        blocked.append("v5_p30_p29_package_not_ready")
        blocked.extend(_string_list(summary.get("blocked_reasons")))
    if not bool(summary.get("default_off", False)):
        blocked.append("v5_p30_p29_default_off_violation")
    if not bool(summary.get("request_adapter_off", False)):
        blocked.append("v5_p30_p29_request_adapter_violation")
    if not bool(summary.get("post_fields_empty", False)):
        blocked.append("v5_p30_p29_post_fields_present")
    for name in ("p28", "p26", "p27"):
        if not bool(summary.get(f"{name}_ready" if name != "p27" else "p27_approved", False)):
            blocked.append(f"v5_p30_{name}_summary_not_ready")
        if not bool(summary.get(f"{name}_default_off", False)):
            blocked.append(f"v5_p30_{name}_summary_default_off_violation")
        if not bool(summary.get(f"{name}_request_adapter_off", False)):
            blocked.append(f"v5_p30_{name}_summary_request_adapter_violation")
    return blocked


def _option_blockers(options: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if int(options.get("min_runs", 0) or 0) < DEFAULT_MIN_RUNS:
        blocked.append("v5_p30_min_runs_too_low")
    if int(options.get("steps", 0) or 0) < DEFAULT_STEPS:
        blocked.append("v5_p30_representative_steps_too_low")
    if int(options.get("samples", 0) or 0) < 1:
        blocked.append("v5_p30_samples_too_low")
    if float(options.get("min_end_to_end_speedup", 0.0) or 0.0) < 1.03:
        blocked.append("v5_p30_min_speedup_too_low")
    if float(options.get("max_speedup_spread_ratio", 1.0) or 1.0) > DEFAULT_MAX_SPREAD:
        blocked.append("v5_p30_max_speedup_spread_too_high")
    cases = set(str(item) for item in list(options.get("cases", []) or []))
    if not set(DEFAULT_CASES).issubset(cases):
        blocked.append("v5_p30_required_cases_missing")
    for field, reason in (
        ("auto_launch_requested", "v5_p30_auto_launch_requested"),
        ("default_training_requested", "v5_p30_default_training_requested"),
        ("default_rollout_requested", "v5_p30_default_rollout_requested"),
        ("request_adapter_requested", "v5_p30_request_adapter_requested"),
    ):
        if bool(options.get(field, False)):
            blocked.append(reason)
    for field in ("family", "source_data", "python", "runner", "output_root"):
        if not str(options.get(field) or ""):
            blocked.append(f"v5_p30_{field}_missing")
    return blocked


def _rollback_blockers(rollback: Mapping[str, Any]) -> list[str]:
    required = (
        "fallback_authoritative",
        "manual_abort_required_on_failure",
        "disable_for_run_on_native_error",
        "disable_for_run_on_state_sync_failure",
        "disable_for_run_on_checkpoint_resume_mismatch",
        "rollback_on_resume_mismatch",
        "rollback_on_performance_regression",
        "rollback_policy_acknowledged",
    )
    return [f"v5_p30_rollback_{name}_missing" for name in required if not bool(rollback.get(name, False))]


def _required_runtime_evidence() -> list[str]:
    return [
        "native_dispatch_requested",
        "native_dispatch_executed",
        "native_dispatch_training_executor_timing_present",
        "native_dispatch_update_report_present",
        "native_dispatch_owner_native_report_present",
        "native_dispatch_probe_cache_retained",
        "native_dispatch_owner_native_runtime_synchronization",
        "native_dispatch_training_executor_last_error_empty",
        "fallback_state_sync_on_close_or_recovery",
        "checkpoint_resume_native_state_boundary",
    ]


def _recommended_next_step(ready: bool, p29_summary: Mapping[str, Any]) -> str:
    if ready:
        return "manual-only: run the listed longer-replicate plan, then feed summaries to the P28 collector"
    if not bool(p29_summary.get("package_ready", False)):
        return "produce a ready P29 owner next-stage package before planning longer replicates"
    return "repair P30 run-plan blockers before any manual experiment"


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("request_adapter_mapping_allowed") is False
        and value.get("request_fields_emitted") is False
    )


def _int_option(raw: Mapping[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in raw:
            try:
                return int(raw.get(key))
            except (TypeError, ValueError):
                return 0
    return int(default)


def _float_option(raw: Mapping[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in raw:
            try:
                return float(raw.get(key))
            except (TypeError, ValueError):
                return 0.0
    return float(default)


def _bool_option(raw: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(bool(raw.get(key, False)) for key in keys)


def _str_option(raw: Mapping[str, Any], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        if key in raw:
            return str(raw.get(key) or "")
    return str(default)


def _string_list_option(raw: Mapping[str, Any], keys: tuple[str, ...], default: list[str]) -> list[str]:
    for key in keys:
        if key in raw:
            value = raw.get(key)
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value if str(item)]
    return list(default)


def _seeds(value: Any, count: int) -> list[int]:
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return [6101 + index for index in range(max(int(count), 1))]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _load_plan_options(path: str) -> dict[str, Any]:
    if not path:
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P30 manual longer-replicate run manifest.")
    parser.add_argument("--p29-package", default="", help="P29 owner next-stage package JSON.")
    parser.add_argument("--plan-options", default="", help="Optional JSON file with run-plan options.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=load_json(args.p29_package) if args.p29_package else None,
        run_plan_options=_load_plan_options(args.plan_options),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_longer_replicate_runner_manifest"]
