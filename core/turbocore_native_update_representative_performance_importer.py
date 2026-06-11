"""Import existing representative native-update performance evidence.

This module normalizes older P19/P31 evidence into the current standard
``native_update_performance_report.json`` and
``native_update_performance_gate.json`` artifacts.  It does not launch training
and marks the source quality explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from core.turbocore_native_update_performance import build_native_update_performance_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = REPO_ROOT / "temp"
ARTIFACT_DIR = TEMP_ROOT / "turbocore_optimizer"
REPORT_ARTIFACT = ARTIFACT_DIR / "native_update_performance_report.json"
GATE_ARTIFACT = ARTIFACT_DIR / "native_update_performance_gate.json"
IMPORT_SUMMARY_ARTIFACT = ARTIFACT_DIR / "native_update_representative_performance_import_summary.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"

P19_OWNER_EVIDENCE = TEMP_ROOT / "turbocore_v5_p19_owner_review_evidence_gate_ready_pending_20260601.json"
P31_AUDIT_READY = TEMP_ROOT / "turbocore_v5_p31_longer_replicate_manual_run_audit_ready_20260601.json"
MULTITENSOR_RELEASE_HOLD = ARTIFACT_DIR / "native_update_optimizer_multitensor_release_hold.json"


def build_native_update_representative_performance_import(
    *,
    temp_root: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    root = Path(temp_root) if temp_root is not None else TEMP_ROOT
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    p19 = _read_json(root / P19_OWNER_EVIDENCE.name)
    p31 = _read_json(root / P31_AUDIT_READY.name)
    multitensor = _read_json(out_dir / MULTITENSOR_RELEASE_HOLD.name)

    p19_perf = _as_dict(p19.get("performance_matrix_summary"))
    p31_bundle = _as_dict(p31.get("collector_bundle"))
    p31_aggregate = _as_dict(p31_bundle.get("aggregate"))
    p31_samples = _sample_rows(p31_bundle.get("samples"))

    blockers = _validate_sources(p19=p19, p19_perf=p19_perf, p31=p31, p31_aggregate=p31_aggregate, samples=p31_samples)
    benchmark_matrix = _benchmark_matrix_from_p31(p31_samples, aggregate=p31_aggregate)
    optimizer_gate = _optimizer_gate_from_p19(p19_perf)
    owner_probe = _owner_probe_from_sources(p19_perf=p19_perf, multitensor=multitensor)
    performance_gate = build_native_update_performance_gate(
        shadow_report={
            "owner_native_launch_probe": owner_probe,
            "optimizer_performance_gate": optimizer_gate,
        },
        performance_report={"benchmark_matrix": benchmark_matrix},
    )

    ready = not blockers and bool(performance_gate.get("representative_performance_gate_ready", False))
    report = {
        "schema_version": 1,
        "report": "turbocore_native_update_imported_representative_performance_report_v0",
        "roadmap": ROADMAP,
        "source": "existing_p19_p31_performance_evidence",
        "source_evidence_quality": "existing_imported_owner_review_and_manual_replicate_artifacts",
        "fresh_live_run": False,
        "promotion_grade_current_run": False,
        "standard_artifact_import": True,
        "full_matrix_source_artifact_present": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "benchmark_matrix": benchmark_matrix,
        "optimizer_performance_gate": optimizer_gate,
        "owner_native_launch_probe": owner_probe,
        "performance_gate": performance_gate,
        "blocked_reasons": _dedupe(blockers + _strings(performance_gate.get("blocked_reasons"))),
        "source_artifacts": {
            "owner_review_evidence": str(root / P19_OWNER_EVIDENCE.name),
            "manual_replicate_audit": str(root / P31_AUDIT_READY.name),
            "multitensor_release_hold": str(out_dir / MULTITENSOR_RELEASE_HOLD.name),
        },
        "notes": [
            "This importer normalizes existing evidence only; it does not launch training.",
            "The original P18 full matrix path is not present in this workspace, so source quality is compact/imported.",
            "Product request/UI/schema/runtime/native/training dispatch remains disabled.",
        ],
    }
    report["ok"] = ready
    report["release_performance_evidence_complete"] = ready

    summary = {
        "schema_version": 1,
        "package": "native_update_representative_performance_import_summary_v0",
        "roadmap": ROADMAP,
        "ok": ready,
        "performance_report_artifact": str(out_dir / REPORT_ARTIFACT.name),
        "performance_gate_artifact": str(out_dir / GATE_ARTIFACT.name),
        "source_evidence_quality": report["source_evidence_quality"],
        "fresh_live_run": False,
        "promotion_grade_current_run": False,
        "representative_performance_gate_ready": bool(
            performance_gate.get("representative_performance_gate_ready", False)
        ),
        "blocked_reasons": report["blocked_reasons"],
        "summary": {
            "manual_replicate_run_count": len(p31_samples),
            "manual_replicate_ready_run_count": int(p31_aggregate.get("ready_run_count", 0) or 0),
            "manual_replicate_min_speedup": p31_aggregate.get("min_speedup"),
            "manual_replicate_median_speedup": p31_aggregate.get("median_speedup"),
            "manual_replicate_steps": benchmark_matrix["summary"]["representative_steps"],
            "optimizer_best_speedup_vs_baseline": p19_perf.get("optimizer_best_speedup_vs_baseline"),
            "native_case": p19_perf.get("representative_native_case"),
        },
    }

    if write_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / REPORT_ARTIFACT.name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / GATE_ARTIFACT.name).write_text(
            json.dumps(performance_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / IMPORT_SUMMARY_ARTIFACT.name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _validate_sources(
    *,
    p19: Mapping[str, Any],
    p19_perf: Mapping[str, Any],
    p31: Mapping[str, Any],
    p31_aggregate: Mapping[str, Any],
    samples: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not p19:
        blockers.append("p19_owner_review_evidence_missing")
    if not p31:
        blockers.append("p31_manual_replicate_audit_missing")
    if p19_perf.get("performance_gate_ready") is not True:
        blockers.append("p19_performance_gate_not_ready")
    if p19_perf.get("optimizer_evidence_quality") != "promotion_benchmark":
        blockers.append("p19_optimizer_evidence_not_promotion_benchmark")
    if p19_perf.get("promotion_gate_ok") is not True:
        blockers.append("p19_optimizer_promotion_gate_not_ok")
    if p19_perf.get("report_only_runtime_dispatch_off") is not True:
        blockers.append("p19_runtime_dispatch_not_report_only_off")
    if p31.get("collector_evidence_ready") is not True:
        blockers.append("p31_collector_evidence_not_ready")
    if p31_aggregate.get("ready") is not True:
        blockers.append("p31_manual_replicate_aggregate_not_ready")
    if int(p31_aggregate.get("ready_run_count", 0) or 0) < 5:
        blockers.append("p31_manual_replicate_ready_run_count_too_low")
    if not samples:
        blockers.append("p31_manual_replicate_samples_missing")
    if any(_default_off_blocked(sample) for sample in samples):
        blockers.append("p31_manual_replicate_default_off_contract_failed")
    if any(float(sample.get("representative_end_to_end_speedup", 0.0) or 0.0) < 1.03 for sample in samples):
        blockers.append("p31_manual_replicate_speedup_below_current_gate")
    if any(int(sample.get("representative_steps", 0) or 0) < 20 for sample in samples):
        blockers.append("p31_manual_replicate_steps_below_current_gate")
    return _dedupe(blockers)


def _benchmark_matrix_from_p31(samples: list[dict[str, Any]], *, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_ms = mean(float(item.get("baseline_mean_step_ms", 0.0) or 0.0) for item in samples)
    native_ms = mean(float(item.get("native_mean_step_ms", 0.0) or 0.0) for item in samples)
    steps = min(int(item.get("representative_steps", 0) or 0) for item in samples)
    speedup = baseline_ms / native_ms if native_ms > 0 else 0.0
    cases = [
        {
            "case": {"name": "baseline_phase"},
            "summary": {"success": True, "steps_completed": steps, "mean_step_ms": round(baseline_ms, 4)},
        },
        {
            "case": {"name": "native_update_dispatch_promotion_perf"},
            "summary": {
                "success": True,
                "steps_completed": steps,
                "mean_step_ms": round(native_ms, 4),
                "native_dispatch_executed": True,
            },
        },
    ]
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "source": "existing_p31_manual_replicate_audit",
        "run": True,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "executed_count": len(cases),
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": round(baseline_ms, 4),
                "native_update_dispatch_promotion_perf": round(native_ms, 4),
            },
            "representative_steps": steps,
            "representative_end_to_end_speedup": round(speedup, 4),
            "manual_replicate_ready_run_count": int(aggregate.get("ready_run_count", 0) or 0),
            "manual_replicate_min_speedup": aggregate.get("min_speedup"),
            "manual_replicate_median_speedup": aggregate.get("median_speedup"),
        },
    }


def _optimizer_gate_from_p19(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate": "turbocore_optimizer_performance_gate",
        "status": "imported_existing_owner_review_summary",
        "ok": bool(summary.get("optimizer_evidence_present", False)),
        "promotion_gate_ok": bool(summary.get("promotion_gate_ok", False)),
        "runtime_dispatch_allowed": False,
        "evidence_quality": str(summary.get("optimizer_evidence_quality", "") or ""),
        "best_candidate": {
            "optimizer": "turbocore_adamw_cuda_runtime_session",
            "speedup_vs_baseline": summary.get("optimizer_best_speedup_vs_baseline"),
        },
    }


def _owner_probe_from_sources(*, p19_perf: Mapping[str, Any], multitensor: Mapping[str, Any]) -> dict[str, Any]:
    mt_summary = _as_dict(multitensor.get("summary"))
    nested = _as_dict(multitensor.get("nested_multitensor_evidence"))
    dtype_reports = _as_dict(nested.get("dtype_reports"))
    owner_numel = max((int(_as_dict(item).get("flat_numel", 0) or 0) for item in dtype_reports.values()), default=0)
    return {
        "ok": bool(p19_perf.get("performance_gate_ready", False)),
        "attempted": True,
        "kernel_executed": int(mt_summary.get("native_kernel_launch_count", 0) or 0) > 0,
        "parity_ok": bool(p19_perf.get("performance_gate_ready", False)),
        "persistent_owner_mutated": False,
        "owner_numel": owner_numel,
        "elapsed_ms": None,
        "source": "p19_compact_performance_gate_plus_multitensor_release_hold",
    }


def _sample_rows(value: Any) -> list[dict[str, Any]]:
    return [_as_dict(item) for item in value] if isinstance(value, list) else []


def _default_off_blocked(sample: Mapping[str, Any]) -> bool:
    return any(
        bool(sample.get(key, False))
        for key in (
            "default_behavior_changed",
            "default_training_path_enabled",
            "training_path_enabled",
            "default_rollout_allowed",
            "auto_rollout_allowed",
            "request_adapter_mapping_allowed",
            "request_fields_emitted",
        )
    ) or bool(sample.get("post_gate_request_fields"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_native_update_representative_performance_import"]
