"""Compact representative-performance evidence summary for native update."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "native_update_representative_performance_summary.json"
PERFORMANCE_REPORT_ARTIFACT = ARTIFACT_DIR / "native_update_performance_report.json"
PERFORMANCE_GATE_ARTIFACT = ARTIFACT_DIR / "native_update_performance_gate.json"
MULTITENSOR_RELEASE_HOLD_ARTIFACT = ARTIFACT_DIR / "native_update_optimizer_multitensor_release_hold.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def build_native_update_representative_performance_summary(
    *,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    performance_report = _read_json_if_exists(directory / PERFORMANCE_REPORT_ARTIFACT.name)
    performance_gate = _read_json_if_exists(directory / PERFORMANCE_GATE_ARTIFACT.name)
    multitensor = _read_json_if_exists(directory / MULTITENSOR_RELEASE_HOLD_ARTIFACT.name)
    gate = _performance_gate(performance_report, performance_gate)
    multitensor_summary = _as_dict(multitensor.get("summary"))
    representative_ready = bool(gate.get("representative_performance_gate_ready", False))
    performance_artifact_present = bool(performance_report or performance_gate)
    artifact_ready_or_absent = (not performance_artifact_present) or representative_ready
    native_launch_ready = bool(
        multitensor.get("ok") is True
        and multitensor.get("evidence_ready") is True
        and int(multitensor_summary.get("native_kernel_launch_count", 0) or 0) > 0
        and int(multitensor_summary.get("training_parameter_mutation_count", 0) or 0) > 0
        and int(multitensor_summary.get("top_level_native_dispatch_allowed_count", 0) or 0) == 0
        and int(multitensor_summary.get("top_level_training_path_enabled_count", 0) or 0) == 0
    )
    blockers = _performance_blockers(
        gate,
        performance_artifact_present=performance_artifact_present,
        representative_ready=representative_ready,
    )
    evidence_complete = performance_artifact_present and representative_ready
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_representative_performance_summary_v0",
        "gate": "native_update_representative_performance_summary",
        "ok": native_launch_ready and artifact_ready_or_absent,
        "roadmap": ROADMAP,
        "performance_artifact_present": performance_artifact_present,
        "representative_performance_gate_ready": representative_ready,
        "native_launch_evidence_ready": native_launch_ready,
        "release_performance_evidence_complete": evidence_complete,
        "source_evidence_quality": str(performance_report.get("source_evidence_quality", "") or ""),
        "fresh_live_run": performance_report.get("fresh_live_run"),
        "promotion_grade_current_run": performance_report.get("promotion_grade_current_run"),
        "standard_artifact_import": bool(performance_report.get("standard_artifact_import", False)),
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_allowed": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "summary": {
            "performance_artifact_present_count": 1 if performance_artifact_present else 0,
            "representative_performance_gate_ready_count": 1 if representative_ready else 0,
            "native_launch_evidence_ready_count": 1 if native_launch_ready else 0,
            "native_kernel_launch_count": int(multitensor_summary.get("native_kernel_launch_count", 0) or 0),
            "training_parameter_mutation_count": int(
                multitensor_summary.get("training_parameter_mutation_count", 0) or 0
            ),
            "top_level_native_dispatch_allowed_count": int(
                multitensor_summary.get("top_level_native_dispatch_allowed_count", 0) or 0
            ),
            "top_level_training_path_enabled_count": int(
                multitensor_summary.get("top_level_training_path_enabled_count", 0) or 0
            ),
        },
        "performance_gate": _compact_gate(gate),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "continue owner release review with imported performance artifact labeled"
            if evidence_complete
            else "run or ingest representative native_update_dispatch_promotion_perf performance evidence"
        ),
        "notes": [
            "This summary does not treat multi-tensor native launch evidence as representative performance.",
            "It keeps product dispatch and request/UI/schema exposure closed.",
            "Imported performance artifacts are labeled with source quality and fresh_live_run=false.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _performance_gate(performance_report: Mapping[str, Any], performance_gate: Mapping[str, Any]) -> dict[str, Any]:
    if performance_gate:
        return _as_dict(performance_gate)
    return _as_dict(performance_report.get("performance_gate"))


def _performance_blockers(
    gate: Mapping[str, Any],
    *,
    performance_artifact_present: bool,
    representative_ready: bool,
) -> list[str]:
    blockers = _strings(gate.get("blocked_reasons"))
    if not performance_artifact_present:
        blockers.append("native_update_performance_artifact_missing")
    elif not representative_ready:
        blockers.append("native_update_representative_performance_not_ready")
    return _dedupe(blockers)


def _compact_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _as_dict(gate.get("evidence"))
    training_matrix = _as_dict(evidence.get("training_matrix"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    owner = _as_dict(evidence.get("owner_native_kernel"))
    return {
        "present": bool(gate),
        "representative_performance_gate_ready": bool(gate.get("representative_performance_gate_ready", False)),
        "performance_test_ready": bool(gate.get("performance_test_ready", False)),
        "training_matrix_native_case": str(training_matrix.get("native_case", "") or ""),
        "training_matrix_end_to_end_speedup": training_matrix.get("end_to_end_speedup"),
        "training_matrix_representative_steps": int(training_matrix.get("representative_steps", 0) or 0),
        "optimizer_best_speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
        "owner_kernel_executed": bool(owner.get("kernel_executed", False)),
        "blocked_reasons": _strings(gate.get("blocked_reasons")),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_dict(payload)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_native_update_representative_performance_summary"]
