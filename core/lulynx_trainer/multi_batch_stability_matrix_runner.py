"""Protected runner for Lulynx native multi-batch stability plans.

The runner consumes ``lulynx_multi_batch_stability_matrix_plan_v0``.  It is
dry-run by default and only executes candidates that the plan marked ready.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .multi_batch_stability_matrix_plan import LULYNX_MULTI_BATCH_STABILITY_MATRIX_PLAN


LULYNX_MULTI_BATCH_STABILITY_MATRIX_RUNNER = "lulynx_multi_batch_stability_matrix_runner_v0"


def run_lulynx_multi_batch_stability_matrix_plan(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    selected_ids: Sequence[str] | None = None,
    reuse_existing: bool = False,
    cwd: Path | str | None = None,
    subprocess_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run or dry-run a multi-batch stability matrix plan.

    ``execute=False`` never starts subprocesses.  ``execute=True`` is still
    conservative: blocked candidates, release-claim candidates, and malformed
    plans are refused.
    """

    matrix_plan = plan if isinstance(plan, Mapping) else {}
    selected = {str(item) for item in (selected_ids or []) if str(item)}
    validation_blockers = _plan_validation_blockers(matrix_plan)
    results: list[dict[str, Any]] = []
    if validation_blockers:
        return _runner_report(
            execute=execute,
            status="blocked_invalid_plan",
            validation_blockers=validation_blockers,
            results=[],
        )

    runner = subprocess_runner or subprocess.run
    for raw_candidate in matrix_plan.get("candidates", []):
        candidate = raw_candidate if isinstance(raw_candidate, Mapping) else {}
        candidate_id = str(candidate.get("id") or "")
        if selected and candidate_id not in selected:
            results.append(_candidate_result(candidate, status="skipped_not_selected"))
            continue
        blockers = _candidate_blockers(candidate)
        expected_report = _expected_gpu_bubble_report(candidate)
        if reuse_existing and expected_report is not None and expected_report.is_file():
            results.append(
                _candidate_result(
                    candidate,
                    status="reused_existing",
                    blockers=blockers,
                    expected_report=str(expected_report),
                    evidence=_candidate_evidence_summary(candidate),
                )
            )
            continue
        if blockers:
            results.append(_candidate_result(candidate, status="blocked", blockers=blockers))
            continue
        command = [str(part) for part in candidate.get("protected_command", [])]
        if not execute:
            results.append(_candidate_result(candidate, status="dry_run", command=command))
            continue
        if not command:
            results.append(_candidate_result(candidate, status="blocked", blockers=["missing_protected_command"]))
            continue
        completed = runner(command, cwd=str(cwd) if cwd is not None else None, check=False)
        returncode = int(getattr(completed, "returncode", 1))
        results.append(
            _candidate_result(
                candidate,
                status="executed" if returncode == 0 else "failed",
                command=command,
                returncode=returncode,
                evidence=_candidate_evidence_summary(candidate),
            )
        )
        if returncode != 0:
            break

    return _runner_report(
        execute=execute,
        status=_overall_status(results, execute=execute),
        validation_blockers=[],
        results=results,
    )


def _plan_validation_blockers(plan: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(plan.get("report") or "") != LULYNX_MULTI_BATCH_STABILITY_MATRIX_PLAN:
        blockers.append("unsupported_plan_report")
    if bool(plan.get("safe_to_auto_start")):
        blockers.append("plan_safe_to_auto_start_must_remain_false")
    if bool(plan.get("release_claim_allowed")):
        blockers.append("plan_release_claim_allowed_must_remain_false")
    if not isinstance(plan.get("candidates"), Sequence) or isinstance(plan.get("candidates"), (str, bytes)):
        blockers.append("missing_candidate_list")
    return blockers


def _candidate_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(candidate.get("status") or "") != "ready_to_schedule":
        blockers.append("candidate_not_ready_to_schedule")
    if bool(candidate.get("safe_to_auto_start")):
        blockers.append("candidate_safe_to_auto_start_must_remain_false")
    if bool(candidate.get("release_claim_allowed")):
        blockers.append("candidate_release_claim_allowed_must_remain_false")
    blockers.extend(_string_list(candidate.get("blockers")))
    return _dedupe(blockers)


def _expected_gpu_bubble_report(candidate: Mapping[str, Any]) -> Path | None:
    reports = candidate.get("expected_reports")
    if not isinstance(reports, Mapping):
        return None
    path = reports.get("gpu_bubble_report")
    return Path(str(path)) if path else None


def _candidate_evidence_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reports = candidate.get("expected_reports")
    expected = reports if isinstance(reports, Mapping) else {}
    gpu_report_path = Path(str(expected.get("gpu_bubble_report") or "")) if expected.get("gpu_bubble_report") else None
    benchmark_path = Path(str(expected.get("benchmark_summary") or "")) if expected.get("benchmark_summary") else None
    manifest_path = Path(str(expected.get("run_manifest_glob") or "")) if expected.get("run_manifest_glob") else None
    gpu_report = _load_json(gpu_report_path) if gpu_report_path and gpu_report_path.is_file() else {}
    benchmark_summary = _load_json(benchmark_path) if benchmark_path and benchmark_path.is_file() else {}
    run_manifest = _load_json(manifest_path) if manifest_path and manifest_path.is_file() else {}
    fresh_gate = _manifest_mapping(run_manifest, "multi_batch_promotion_gate")
    fresh_dataloader = _manifest_mapping(run_manifest, "multi_batch_dataloader")
    run_summaries = gpu_report.get("run_summaries") if isinstance(gpu_report.get("run_summaries"), list) else []
    first_run = run_summaries[0] if run_summaries and isinstance(run_summaries[0], Mapping) else {}
    classification = gpu_report.get("classification") if isinstance(gpu_report.get("classification"), Mapping) else {}
    gpu_return_code = _as_int(gpu_report.get("return_code"), 0) if gpu_report else None
    first_run_success = bool(first_run.get("success", False)) if first_run else False
    fresh_gate_ready = bool(fresh_gate.get("ready_for_long_window_probe", False)) if fresh_gate else False
    fresh_gate_blockers = _string_list(fresh_gate.get("blockers")) if fresh_gate else []
    evidence_complete = bool(
        gpu_report_path and gpu_report_path.is_file()
        and benchmark_path and benchmark_path.is_file()
        and manifest_path and manifest_path.is_file()
        and gpu_report
        and benchmark_summary
        and run_manifest
        and gpu_return_code == 0
        and run_summaries
        and first_run_success
        and fresh_gate_ready
        and not fresh_gate_blockers
    )
    completion_blockers = _evidence_completion_blockers(
        gpu_report_exists=bool(gpu_report_path and gpu_report_path.is_file()),
        benchmark_summary_exists=bool(benchmark_path and benchmark_path.is_file()),
        run_manifest_exists=bool(manifest_path and manifest_path.is_file()),
        benchmark_summary_parsed=bool(benchmark_summary),
        run_manifest_parsed=bool(run_manifest),
        gpu_return_code=gpu_return_code,
        run_summary_count=len(run_summaries),
        first_run_success=first_run_success,
        fresh_gate_ready=fresh_gate_ready,
        fresh_gate_blockers=fresh_gate_blockers,
    )
    return {
        "schema_version": 1,
        "report": "lulynx_multi_batch_stability_candidate_evidence_v0",
        "release_claim_allowed": False,
        "gpu_bubble_report_exists": bool(gpu_report_path and gpu_report_path.is_file()),
        "benchmark_summary_exists": bool(benchmark_path and benchmark_path.is_file()),
        "run_manifest_exists": bool(manifest_path and manifest_path.is_file()),
        "benchmark_summary_parsed": bool(benchmark_summary),
        "run_manifest_parsed": bool(run_manifest),
        "fresh_promotion_gate_status": str(fresh_gate.get("status") or "missing") if fresh_gate else "missing",
        "fresh_promotion_gate_ready": fresh_gate_ready,
        "fresh_promotion_gate_blockers": fresh_gate_blockers,
        "fresh_dataloader_contract_status": _fresh_dataloader_status(fresh_dataloader),
        "fresh_dataloader_drop_last": bool(fresh_dataloader.get("drop_last")) if fresh_dataloader else False,
        "gpu_bubble_return_code": gpu_return_code,
        "benchmark_summary_missing": bool(gpu_report.get("benchmark_summary_missing", False)) if gpu_report else True,
        "run_summary_count": len(run_summaries),
        "first_run_success": first_run_success,
        "steps_completed": _as_int(first_run.get("steps_completed"), 0) if first_run else 0,
        "steady_samples_per_second": _as_float(first_run.get("steady_samples_per_second"), 0.0) if first_run else 0.0,
        "peak_vram_mb": _as_float(first_run.get("peak_vram_mb"), 0.0) if first_run else 0.0,
        "final_loss": _as_float(first_run.get("final_loss"), 0.0) if first_run else 0.0,
        "failure_stage": _first_nonempty(
            first_run.get("failure_stage") if first_run else "",
            first_run.get("failed_stage") if first_run else "",
            first_run.get("error_stage") if first_run else "",
            classification.get("failure_stage") if classification else "",
            classification.get("failed_stage") if classification else "",
            gpu_report.get("failure_stage") if gpu_report else "",
        ),
        "classification_status": str(classification.get("status") or "") if classification else "",
        "active_gpu_util_pct_mean": _as_float(classification.get("active_gpu_util_pct_mean"), 0.0) if classification else 0.0,
        "completion_blockers": completion_blockers,
        "evidence_complete_for_review": evidence_complete,
    }


def _candidate_result(
    candidate: Mapping[str, Any],
    *,
    status: str,
    blockers: Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    returncode: int | None = None,
    expected_report: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "id": str(candidate.get("id") or ""),
        "family": str(candidate.get("family") or ""),
        "physical_batch_size": _as_int(candidate.get("physical_batch_size"), 0),
        "status": status,
        "release_claim_allowed": False,
        "blockers": list(blockers or []),
    }
    if command is not None:
        item["command"] = [str(part) for part in command]
    if returncode is not None:
        item["returncode"] = int(returncode)
    if expected_report:
        item["expected_report"] = expected_report
    if evidence is not None:
        item["evidence"] = dict(evidence)
    return item


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    direct = manifest.get(key)
    if isinstance(direct, Mapping):
        return direct
    extra = manifest.get("extra")
    if isinstance(extra, Mapping):
        value = extra.get(key)
        if isinstance(value, Mapping):
            return value
        runtime_features = extra.get("runtime_features")
        if isinstance(runtime_features, Mapping):
            value = runtime_features.get(key)
            if isinstance(value, Mapping):
                return value
    runtime_features = manifest.get("runtime_features")
    if isinstance(runtime_features, Mapping):
        value = runtime_features.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _fresh_dataloader_status(dataloader: Mapping[str, Any]) -> str:
    if not dataloader:
        return "missing"
    return "ok" if bool(dataloader.get("ok")) else "blocked"


def _evidence_completion_blockers(
    *,
    gpu_report_exists: bool,
    benchmark_summary_exists: bool,
    run_manifest_exists: bool,
    benchmark_summary_parsed: bool,
    run_manifest_parsed: bool,
    gpu_return_code: int | None,
    run_summary_count: int,
    first_run_success: bool,
    fresh_gate_ready: bool,
    fresh_gate_blockers: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not gpu_report_exists:
        blockers.append("missing_gpu_bubble_report")
    if not benchmark_summary_exists:
        blockers.append("missing_benchmark_summary")
    elif not benchmark_summary_parsed:
        blockers.append("benchmark_summary_unparsed")
    if not run_manifest_exists:
        blockers.append("missing_run_manifest")
    elif not run_manifest_parsed:
        blockers.append("run_manifest_unparsed")
    if gpu_return_code not in (None, 0):
        blockers.append("gpu_bubble_return_code_nonzero")
    if run_summary_count <= 0:
        blockers.append("missing_run_summary")
    elif not first_run_success:
        blockers.append("first_run_failed")
    if not fresh_gate_ready:
        blockers.append("fresh_promotion_gate_not_ready")
    if fresh_gate_blockers:
        blockers.append("fresh_promotion_gate_blockers")
    return _dedupe(blockers)


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _runner_report(
    *,
    execute: bool,
    status: str,
    validation_blockers: Sequence[str],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report": LULYNX_MULTI_BATCH_STABILITY_MATRIX_RUNNER,
        "status": status,
        "execute_requested": bool(execute),
        "release_claim_allowed": False,
        "selected_count": sum(1 for item in results if str(item.get("status") or "") != "skipped_not_selected"),
        "executed_count": sum(1 for item in results if str(item.get("status") or "") == "executed"),
        "failed_count": sum(1 for item in results if str(item.get("status") or "") == "failed"),
        "blocked_count": sum(1 for item in results if str(item.get("status") or "") == "blocked"),
        "validation_blockers": list(validation_blockers),
        "candidates": [dict(item) for item in results],
    }


def _overall_status(results: Sequence[Mapping[str, Any]], *, execute: bool) -> str:
    statuses = [str(item.get("status") or "") for item in results]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "blocked" for status in statuses):
        return "blocked_with_skips"
    if execute and any(status == "executed" for status in statuses):
        return "executed"
    if any(status == "dry_run" for status in statuses):
        return "dry_run"
    if any(status == "reused_existing" for status in statuses):
        return "reused_existing"
    return "no_candidates_selected"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_MULTI_BATCH_STABILITY_MATRIX_RUNNER",
    "run_lulynx_multi_batch_stability_matrix_plan",
]
