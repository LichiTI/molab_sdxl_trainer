"""Audit manual longer-replicate run results for TurboCore V5-P31.

This module binds manually produced matrix summaries back to the P30 manifest.
It does not launch training, emit request-adapter fields, or change default
training behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_longer_replicate_evidence_collector import (
    build_v5_longer_replicate_evidence_bundle,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P30_READY_DECISION = "longer_replicate_runner_manifest_ready_default_off"


def build_v5_longer_replicate_manual_run_audit(
    *,
    runner_manifest: Mapping[str, Any] | None = None,
    run_payloads: Iterable[Mapping[str, Any]] | None = None,
    run_summary_paths: Iterable[str | Path] | None = None,
    run_result_payloads: Iterable[Mapping[str, Any]] | None = None,
    matrix_summary_payloads: Iterable[Mapping[str, Any]] | None = None,
    run_result_paths: Iterable[str | Path] | None = None,
    matrix_summary_paths: Iterable[str | Path] | None = None,
    emit_collector_bundle: bool = True,
) -> dict[str, Any]:
    manifest = _as_dict(runner_manifest)
    thresholds = _collector_thresholds(manifest)
    loaded = _load_inputs(
        _chain_payloads(run_payloads, run_result_payloads, matrix_summary_payloads),
        _chain_paths(run_summary_paths, run_result_paths, matrix_summary_paths),
    )
    matched = _match_planned_runs(_planned_runs(manifest), loaded)
    run_audits = _run_audits(matched["matched_runs"], thresholds)
    collector = build_v5_longer_replicate_evidence_bundle(
        run_payloads=[item["payload"] for item in matched["matched_inputs"]],
        min_runs=thresholds["min_runs"],
        min_representative_steps=thresholds["min_representative_steps"],
        min_end_to_end_speedup=thresholds["min_end_to_end_speedup"],
        max_speedup_spread_ratio=thresholds["max_speedup_spread_ratio"],
    )
    manifest_summary = _manifest_summary(manifest)
    blockers = _dedupe(
        _manifest_blockers(manifest_summary)
        + _input_blockers(loaded, matched)
        + [reason for audit in run_audits for reason in audit["blocked_reasons"]]
        + _collector_blockers(collector)
    )
    ready = not blockers
    decision = (
        "longer_replicate_manual_run_audit_ready_default_off"
        if ready
        else "longer_replicate_manual_run_audit_blocked_default_off"
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_longer_replicate_manual_run_audit_v0",
        "gate": "v5_longer_replicate_manual_run_audit",
        "ok": ready,
        "manual_run_audit_ready": ready,
        "collector_evidence_ready": bool(collector.get("longer_replicate_evidence_ready", False)),
        "decision": decision,
        "gate_decision": decision,
        "manual_run_required": True,
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
        "post_audit_request_fields": {},
        "runner_manifest_summary": manifest_summary,
        "input_summary": {
            "input_count": len(loaded),
            "load_error_count": len([item for item in loaded if item.get("load_error")]),
            "sources": [str(item.get("source") or "") for item in loaded],
        },
        "plan_match_summary": {
            "planned_run_count": matched["planned_run_count"],
            "input_count": len(loaded),
            "matched_run_count": len(matched["matched_runs"]),
            "missing_planned_runs": matched["missing_planned_runs"],
            "unexpected_inputs": matched["unexpected_inputs"],
            "duplicate_planned_runs": matched["duplicate_planned_runs"],
            "identity_conflicts": matched["identity_conflicts"],
        },
        "run_audits": run_audits,
        "matched_runs": matched["matched_runs"],
        "collector_invocation": _collector_invocation(thresholds, matched),
        "collector_ready_run_payloads": [item["payload"] for item in matched["matched_inputs"]],
        "p28_collector_bundle": collector if emit_collector_bundle else {},
        "collector_bundle": collector,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(ready, collector, matched),
        "notes": [
            "This audit consumes manual run summaries only; it does not launch training.",
            "The collector_bundle can be passed to P26 after all P31 checks pass.",
            "Default rollout and request-adapter mapping remain disabled.",
        ],
    }


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    plan = _as_dict(manifest.get("run_plan"))
    return {
        "present": bool(manifest),
        "source_path": str(manifest.get("_source_path") or manifest.get("source_path") or ""),
        "ok": bool(manifest.get("ok", False)),
        "run_manifest_ready": bool(manifest.get("run_manifest_ready", False)),
        "explicit_run_plan_ready": bool(manifest.get("explicit_run_plan_ready", False)),
        "decision": str(manifest.get("decision") or manifest.get("gate_decision") or ""),
        "manual_run_required": bool(manifest.get("manual_run_required", False)),
        "training_launch_allowed": bool(manifest.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(manifest.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(manifest.get("runs_dispatched", True)),
        "default_off": _default_off_confirmed(manifest),
        "request_adapter_off": _request_adapter_off(manifest),
        "post_fields_empty": not bool(_as_dict(manifest.get("post_manifest_request_fields"))),
        "run_plan_present": bool(plan),
        "planned_run_count": len(_planned_runs(manifest)),
        "run_plan_count": int(plan.get("run_count", 0) or 0),
        "blocked_reasons": _string_list(manifest.get("blocked_reasons")),
    }


def _manifest_blockers(summary: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(summary.get("present", False)):
        blocked.append("v5_p31_p30_manifest_missing")
    if (
        not bool(summary.get("ok", False))
        or not bool(summary.get("run_manifest_ready", False))
        or not bool(summary.get("explicit_run_plan_ready", False))
        or str(summary.get("decision") or "") != P30_READY_DECISION
    ):
        blocked.append("v5_p31_p30_manifest_not_ready")
        blocked.extend(_string_list(summary.get("blocked_reasons")))
    if not bool(summary.get("manual_run_required", False)):
        blocked.append("v5_p31_manual_run_required_missing")
    for field, reason in (
        ("training_launch_allowed", "v5_p31_training_launch_allowed_violation"),
        ("auto_launch_allowed", "v5_p31_auto_launch_allowed_violation"),
        ("runs_dispatched", "v5_p31_runs_dispatched_violation"),
    ):
        if bool(summary.get(field, False)):
            blocked.append(reason)
    if not bool(summary.get("default_off", False)):
        blocked.append("v5_p31_p30_default_off_violation")
    if not bool(summary.get("request_adapter_off", False)):
        blocked.append("v5_p31_p30_request_adapter_violation")
    if not bool(summary.get("post_fields_empty", False)):
        blocked.append("v5_p31_p30_post_fields_present")
    if not bool(summary.get("run_plan_present", False)):
        blocked.append("v5_p31_run_plan_missing")
    if int(summary.get("planned_run_count", 0) or 0) < 1:
        blocked.append("v5_p31_planned_runs_missing")
    if int(summary.get("run_plan_count", 0) or 0) != int(summary.get("planned_run_count", 0) or 0):
        blocked.append("v5_p31_run_plan_count_mismatch")
    return blocked


def _input_blockers(loaded: list[dict[str, Any]], matched: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    load_errors = [item["load_error"] for item in loaded if item.get("load_error")]
    blocked.extend(str(error) for error in load_errors)
    if not loaded:
        blocked.append("v5_p31_run_results_missing")
    if matched["missing_planned_runs"]:
        blocked.append("v5_p31_planned_run_result_missing")
        blocked.extend(f"missing:{item}" for item in matched["missing_planned_runs"])
    if matched["unexpected_inputs"]:
        blocked.append("v5_p31_unexpected_run_result")
        blocked.extend(f"unexpected:{item}" for item in matched["unexpected_inputs"])
    if matched["duplicate_planned_runs"]:
        blocked.append("v5_p31_duplicate_planned_run_result")
        blocked.extend(f"duplicate:{item}" for item in matched["duplicate_planned_runs"])
    if matched["identity_conflicts"]:
        blocked.append("v5_p31_run_result_identity_conflict")
        blocked.extend(f"conflict:{item}" for item in matched["identity_conflicts"])
    return blocked


def _collector_blockers(collector: Mapping[str, Any]) -> list[str]:
    if bool(collector.get("longer_replicate_evidence_ready", False)):
        return []
    blocked = ["v5_p31_p28_collector_not_ready"]
    blocked.extend(_string_list(collector.get("blocked_reasons")))
    blocked.extend(_string_list(collector.get("promotion_blockers")))
    return _dedupe(blocked)


def _planned_runs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = _as_dict(manifest.get("run_plan"))
    runs = plan.get("runs")
    if not isinstance(runs, list):
        return []
    return [_as_dict(item) for item in runs if isinstance(item, Mapping)]


def _match_planned_runs(planned: list[Mapping[str, Any]], loaded: list[dict[str, Any]]) -> dict[str, Any]:
    planned_records = [
        {"index": index, "label": _run_label(run), "keys": _run_keys(run), "run": run}
        for index, run in enumerate(planned)
    ]
    matches_by_plan: dict[int, list[dict[str, Any]]] = {record["index"]: [] for record in planned_records}
    unexpected: list[str] = []
    identity_conflicts: list[str] = []
    for item in loaded:
        if item.get("load_error"):
            continue
        matched_records = [record for record in planned_records if record["keys"].intersection(_input_keys(item))]
        if not matched_records:
            unexpected.append(str(item["source"]))
            continue
        if len(matched_records) > 1:
            identity_conflicts.append(str(item["source"]))
            continue
        matches_by_plan[matched_records[0]["index"]].append(item)

    matched_runs: list[dict[str, Any]] = []
    matched_inputs: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicate: list[str] = []
    for record in planned_records:
        run = _as_dict(record["run"])
        inputs = matches_by_plan[record["index"]]
        if not inputs:
            missing.append(str(record["label"]))
            continue
        if len(inputs) > 1:
            duplicate.append(str(record["label"]))
        item = inputs[0]
        payload = _as_dict(item["payload"])
        matched_inputs.append(item)
        matched_runs.append(
            {
                "run_id": str(run.get("run_id") or ""),
                "expected_matrix_summary_path": str(run.get("matrix_summary_path") or ""),
                "expected_cases": [str(value) for value in list(run.get("expected_cases") or []) if str(value)],
                "expected_steps": int(run.get("expected_steps", 0) or 0),
                "source": str(item["source"]),
                "result_run_id": str(payload.get("run_id") or ""),
                "payload": payload,
                "matched": True,
            }
        )
    return {
        "planned_run_count": len(planned),
        "matched_runs": matched_runs,
        "matched_inputs": matched_inputs,
        "missing_planned_runs": missing,
        "unexpected_inputs": unexpected,
        "duplicate_planned_runs": duplicate,
        "identity_conflicts": identity_conflicts,
    }


def _run_keys(run: Mapping[str, Any]) -> set[str]:
    return {
        value
        for value in (
            str(run.get("run_id") or ""),
            str(run.get("matrix_summary_path") or ""),
            str(run.get("output_dir") or ""),
        )
        if value
    }


def _input_keys(item: Mapping[str, Any]) -> set[str]:
    payload = _as_dict(item.get("payload"))
    return {
        value
        for value in (
            str(item.get("source") or ""),
            str(payload.get("_source_path") or ""),
            str(payload.get("source_path") or ""),
            str(payload.get("matrix_summary_path") or ""),
            str(payload.get("output_dir") or ""),
            str(payload.get("run_id") or ""),
        )
        if value
    }


def _run_label(run: Mapping[str, Any]) -> str:
    return str(run.get("run_id") or run.get("matrix_summary_path") or "planned_run")


def _run_audits(matched_runs: list[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_run_audit(run, thresholds) for run in matched_runs]


def _run_audit(run: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(run.get("payload"))
    expected_cases = [str(item) for item in list(run.get("expected_cases") or []) if str(item)]
    observed_cases = _observed_cases(payload)
    blocked: list[str] = []
    if expected_cases:
        missing = [case for case in expected_cases if case not in observed_cases]
        if missing:
            blocked.append("v5_p31_expected_cases_missing")
            blocked.extend(f"missing_case:{case}" for case in missing)
    representative_steps = _representative_steps(payload)
    expected_steps = int(run.get("expected_steps", 0) or thresholds.get("min_representative_steps", 0) or 0)
    if expected_steps > 0 and representative_steps > 0 and representative_steps < expected_steps:
        blocked.append("v5_p31_representative_steps_below_planned")
    if payload and not _native_dispatch_executed(payload):
        blocked.append("v5_p31_native_dispatch_not_executed")
    return {
        "run_id": str(run.get("run_id") or ""),
        "source": str(run.get("source") or ""),
        "expected_cases": expected_cases,
        "observed_cases": observed_cases,
        "expected_steps": expected_steps,
        "representative_steps": representative_steps,
        "blocked_reasons": _dedupe(blocked),
    }


def _observed_cases(payload: Mapping[str, Any]) -> list[str]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return []
    out: list[str] = []
    for item in cases:
        entry = _as_dict(item)
        case = _as_dict(entry.get("case"))
        name = str(case.get("name") or entry.get("name") or "")
        if name:
            out.append(name)
    return out


def _representative_steps(payload: Mapping[str, Any]) -> int:
    summary = _as_dict(payload.get("summary"))
    return _first_int(payload.get("representative_steps"), payload.get("steps_completed"), summary.get("steps_completed"))


def _native_dispatch_executed(payload: Mapping[str, Any]) -> bool:
    summary = _as_dict(payload.get("summary"))
    if bool(payload.get("native_dispatch_executed", False)) or bool(summary.get("native_dispatch_executed", False)):
        return True
    for item in payload.get("cases", []) if isinstance(payload.get("cases"), list) else []:
        case_summary = _as_dict(_as_dict(item).get("summary"))
        if bool(case_summary.get("native_dispatch_executed", False)):
            return True
    return False


def _collector_invocation(thresholds: Mapping[str, Any], matched: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collector_module": "core.turbocore_v5_longer_replicate_evidence_collector",
        "collector_function": "build_v5_longer_replicate_evidence_bundle",
        "run_summary_sources": [str(item.get("source") or "") for item in matched["matched_inputs"]],
        "thresholds": dict(thresholds),
    }


def _collector_thresholds(manifest: Mapping[str, Any]) -> dict[str, Any]:
    followup = _as_dict(manifest.get("collector_followup"))
    thresholds = _as_dict(followup.get("thresholds"))
    plan = _as_dict(manifest.get("run_plan"))
    required = _as_dict(plan.get("required_thresholds"))
    return {
        "min_runs": _first_int(thresholds.get("min_runs"), required.get("min_runs"), 5),
        "min_representative_steps": _first_int(
            thresholds.get("min_representative_steps"),
            required.get("min_representative_steps"),
            768,
        ),
        "min_end_to_end_speedup": _first_float(
            thresholds.get("min_end_to_end_speedup"),
            required.get("min_end_to_end_speedup"),
            1.05,
        ),
        "max_speedup_spread_ratio": _first_float(
            thresholds.get("max_speedup_spread_ratio"),
            required.get("max_speedup_spread_ratio"),
            0.30,
        ),
    }


def _load_inputs(
    run_payloads: Iterable[Mapping[str, Any]] | None,
    run_summary_paths: Iterable[str | Path] | None,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for index, payload in enumerate(run_payloads or []):
        loaded.append({"source": f"provided_payload:{index}", "payload": dict(payload), "load_error": ""})
    for raw_path in run_summary_paths or []:
        path = Path(raw_path)
        if not path.exists():
            loaded.append({"source": str(path), "payload": {}, "load_error": "v5_p31_run_summary_path_missing"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("_source_path", str(path))
            loaded.append({"source": str(path), "payload": _as_dict(payload), "load_error": ""})
        except Exception as exc:
            loaded.append(
                {
                    "source": str(path),
                    "payload": {},
                    "load_error": f"v5_p31_run_summary_path_error:{type(exc).__name__}",
                }
            )
    return loaded


def _chain_payloads(*groups: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for group in groups:
        out.extend(list(group or []))
    return out


def _chain_paths(*groups: Iterable[str | Path] | None) -> list[str | Path]:
    out: list[str | Path] = []
    for group in groups:
        out.extend(list(group or []))
    return out


def _recommended_next_step(ready: bool, collector: Mapping[str, Any], matched: Mapping[str, Any]) -> str:
    if ready:
        return "feed collector_bundle into P26 and continue the signed default-off review chain"
    if matched.get("missing_planned_runs"):
        return "collect every planned P30 matrix summary before P31 audit"
    if not bool(collector.get("longer_replicate_evidence_ready", False)):
        return "repair run-result blockers until the P28 collector bundle is ready"
    return "hold manual longer-replicate evidence until P31 audit blockers clear"


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


def _first_int(*values: Any) -> int:
    for value in values:
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    return 0


def _first_float(*values: Any) -> float:
    for value in values:
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if out > 0.0:
            return out
    return 0.0


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V5 P30 manual longer-replicate run results.")
    parser.add_argument(
        "--manifest",
        "--runner-manifest",
        dest="manifest",
        default="",
        help="P30 longer-replicate runner manifest JSON.",
    )
    parser.add_argument("--run-summary", action="append", default=[], help="Run or matrix summary JSON path.")
    parser.add_argument("--run-result", action="append", default=[], help="Run result JSON path.")
    parser.add_argument("--matrix-summary", action="append", default=[], help="Matrix summary JSON path.")
    parser.add_argument("--collector-ready-only", action="store_true", help="Omit embedded P28 collector bundle.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=load_json(args.manifest) if args.manifest else None,
        run_summary_paths=args.run_summary,
        run_result_paths=args.run_result,
        matrix_summary_paths=args.matrix_summary,
        emit_collector_bundle=not bool(args.collector_ready_only),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_longer_replicate_manual_run_audit"]
