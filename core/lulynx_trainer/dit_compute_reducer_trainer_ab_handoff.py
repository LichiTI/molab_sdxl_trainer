"""Request/dispatch handoff for DiT compute reducer trainer A/B cases.

This mirrors the T-LoRA A/B handoff shape, but remains conservative: it emits
auditable request patches and dispatch payloads without submitting jobs,
registering request fields, or enabling the training path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .dit_compute_reducer_ab_result_ingestion import build_dit_compute_reducer_ab_result_ingestion
from .dit_compute_reducer_quality_review_decision import build_dit_compute_reducer_quality_review_decision
from .dit_compute_reducer_trainer_ab_manifest import build_dit_compute_reducer_trainer_ab_result_gate


APPROVED_DECISIONS = {"approved", "approve", "approved_for_dispatch"}


def build_dit_compute_reducer_trainer_ab_request_patch_plan(
    manifest: Mapping[str, Any],
    *,
    base_request: Mapping[str, Any] | None = None,
    include_dry_run_flag: bool = True,
) -> dict[str, Any]:
    plan = dict(manifest)
    base = dict(base_request or {})
    cases = [dict(case) for case in plan.get("cases", ()) if isinstance(case, Mapping)]
    blockers: list[str] = []
    if plan.get("scorecard") != "dit_compute_reducer_trainer_ab_manifest_v0":
        blockers.append("unexpected_trainer_ab_manifest")
    if not bool(plan.get("trainer_ab_manifest_ready", plan.get("runner_ready", plan.get("ok", False)))):
        blockers.append("trainer_ab_manifest_not_ready")
    if _unsafe_flags(plan):
        blockers.append("unsafe_trainer_ab_manifest_flag")
    if not cases:
        blockers.append("manifest_cases_missing")

    rows = []
    for case in cases:
        case_blockers = _case_blockers(case)
        blockers.extend(case_blockers)
        if case_blockers:
            continue
        rows.append(_patch_row(case, "baseline", base, include_dry_run_flag, plan))
        rows.append(_patch_row(case, "candidate", base, include_dry_run_flag, plan))

    ready = bool(rows) and not blockers
    return {
        "schema_version": 1,
        "plan": "dit_compute_reducer_trainer_ab_request_patch_plan_v0",
        "ok": ready,
        "dry_run_request_patches_emitted": ready,
        "request_fields_emitted": False,
        "dry_run_only": True,
        "case_count": len(cases),
        "patch_count": len(rows),
        "patches": rows,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "perform manual owner review before dispatch handoff"
            if ready
            else "complete compute-reducer trainer A/B request patch prerequisites"
        ),
    }


def build_dit_compute_reducer_trainer_ab_manual_dispatch_review(
    *,
    request_patch_plan: Mapping[str, Any],
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(request_patch_plan)
    payload = dict(review or {})
    blockers: list[str] = []
    reviewed_case_ids = tuple(str(item) for item in payload.get("reviewed_case_ids", ()) or ())
    case_count = int(plan.get("case_count") or 0)
    decision = str(payload.get("decision") or "").strip().lower()

    if plan.get("plan") != "dit_compute_reducer_trainer_ab_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_plan")
    if not bool(plan.get("dry_run_request_patches_emitted", plan.get("ok", False))):
        blockers.append("request_patch_plan_not_ready")
    if not bool(plan.get("dry_run_only", False)):
        blockers.append("dry_run_boundary_missing")
    if _unsafe_flags(plan):
        blockers.append("unsafe_request_patch_plan_flag")
    if decision not in APPROVED_DECISIONS:
        blockers.append("manual_decision_not_approved")
    if not str(payload.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(payload.get("artifact_digest") or payload.get("package_digest") or "").strip():
        blockers.append("artifact_digest_missing")
    if not bool(payload.get("risk_acknowledged", False)):
        blockers.append("risk_acknowledgement_missing")
    if case_count > 0 and reviewed_case_ids and len(set(reviewed_case_ids)) != case_count:
        blockers.append("reviewed_case_count_mismatch")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_manual_dispatch_review_v0",
        "ok": ready,
        "review_ready": ready,
        "dispatch_handoff_allowed": ready,
        "real_dispatch_allowed": False,
        "case_count": case_count,
        "reviewed_case_count": len(set(reviewed_case_ids)),
        "review": _summary(payload, ("decision", "reviewer", "artifact_digest", "package_digest", "risk_acknowledged")),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build dispatch payload manifest for normal trainer A/B execution"
            if ready
            else "complete owner review before dispatch handoff"
        ),
    }


def build_dit_compute_reducer_trainer_ab_dispatch_manifest(
    *,
    manual_review: Mapping[str, Any],
    request_patch_plan: Mapping[str, Any],
    dispatcher_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    review = dict(manual_review)
    plan = dict(request_patch_plan)
    contract = dict(dispatcher_contract or {})
    patches = [dict(item) for item in plan.get("patches", ()) if isinstance(item, Mapping)]
    blockers: list[str] = []

    if review.get("scorecard") != "dit_compute_reducer_trainer_ab_manual_dispatch_review_v0":
        blockers.append("unexpected_manual_review")
    if not bool(review.get("review_ready", review.get("ok", False))):
        blockers.append("manual_review_not_ready")
    if not bool(review.get("dispatch_handoff_allowed", False)):
        blockers.append("manual_review_handoff_not_allowed")
    if plan.get("plan") != "dit_compute_reducer_trainer_ab_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_plan")
    if not bool(plan.get("dry_run_request_patches_emitted", False)):
        blockers.append("request_patches_not_emitted")
    if _unsafe_flags(review, plan):
        blockers.append("unsafe_child_flag")
    if bool(contract.get("auto_submit", False)):
        blockers.append("auto_submit_not_allowed")
    if not str(contract.get("dispatcher") or "").strip():
        blockers.append("dispatcher_missing")
    if not patches:
        blockers.append("dispatch_patches_missing")

    payloads = [_dispatch_payload(row, contract) for row in patches]
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_dispatch_manifest_v0",
        "ok": ready,
        "dispatch_manifest_ready": ready,
        "dispatch_payloads_emitted": ready,
        "execution_performed": False,
        "case_count": int(plan.get("case_count") or review.get("case_count") or 0),
        "payload_count": len(payloads),
        "dispatcher": str(contract.get("dispatcher") or ""),
        "payloads": payloads if ready else [],
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "submit payloads through the normal trainer dispatcher and ingest results"
            if ready
            else "complete manual review and dispatcher contract before handoff"
        ),
    }


def build_dit_compute_reducer_trainer_ab_dispatch_result_ingestion(
    *,
    dispatch_manifest: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(dispatch_manifest)
    payloads = [dict(item) for item in manifest.get("payloads", ()) if isinstance(item, Mapping)]
    blockers: list[str] = []
    if manifest.get("scorecard") != "dit_compute_reducer_trainer_ab_dispatch_manifest_v0":
        blockers.append("unexpected_dispatch_manifest")
    if not bool(manifest.get("dispatch_manifest_ready", manifest.get("ok", False))):
        blockers.append("dispatch_manifest_not_ready")
    if bool(manifest.get("execution_performed", False)):
        blockers.append("dispatch_manifest_claims_execution")
    if _unsafe_flags(manifest):
        blockers.append("unsafe_dispatch_manifest_flag")
    if not payloads:
        blockers.append("dispatch_payloads_missing")

    result_gate = build_dit_compute_reducer_trainer_ab_result_gate(
        manifest=_manifest_from_payloads(payloads, thresholds=thresholds or manifest.get("thresholds")),
        case_results=case_results,
        thresholds=thresholds,
    )
    if not bool(result_gate.get("trainer_ab_result_ready", result_gate.get("ok", False))):
        blockers.append("trainer_ab_result_gate_not_ready")
    if _unsafe_flags(result_gate):
        blockers.append("unsafe_result_gate_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_dispatch_result_ingestion_v0",
        "ok": ready,
        "result_ingestion_ready": ready,
        "case_count": int(result_gate.get("case_count") or manifest.get("case_count") or 0),
        "result_count": int(result_gate.get("result_count") or 0),
        "result_gate": result_gate,
        "result_summaries_for_ingestion": list(result_gate.get("result_summaries_for_ingestion") or []),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed passed dispatch results into compute-reducer A/B result ingestion"
            if ready
            else "collect complete dispatch results before result ingestion"
        ),
    }


def build_dit_compute_reducer_trainer_ab_quality_review_decision(
    *,
    dispatch_result_ingestion: Mapping[str, Any],
    quality_review: Mapping[str, Any] | None = None,
    evidence_package: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dispatch_ingestion = dict(dispatch_result_ingestion)
    result_summaries = [
        dict(item)
        for item in dispatch_ingestion.get("result_summaries_for_ingestion", ())
        if isinstance(item, Mapping)
    ]
    package = dict(evidence_package or _evidence_package_from_dispatch_ingestion(dispatch_ingestion))
    blockers: list[str] = []

    if dispatch_ingestion.get("scorecard") != "dit_compute_reducer_trainer_ab_dispatch_result_ingestion_v0":
        blockers.append("unexpected_dispatch_result_ingestion")
    if not bool(dispatch_ingestion.get("result_ingestion_ready", dispatch_ingestion.get("ok", False))):
        blockers.append("dispatch_result_ingestion_not_ready")
    if _unsafe_flags(dispatch_ingestion, package):
        blockers.append("unsafe_dispatch_or_package_flag")
    if not result_summaries:
        blockers.append("result_summaries_for_ingestion_missing")

    ab_ingestion = build_dit_compute_reducer_ab_result_ingestion(
        evidence_package=package,
        result_summaries=result_summaries,
        thresholds=thresholds,
    )
    quality_decision = build_dit_compute_reducer_quality_review_decision(
        result_ingestion=ab_ingestion,
        quality_review=quality_review,
    )
    if not bool(ab_ingestion.get("ab_result_ingestion_ready", ab_ingestion.get("ok", False))):
        blockers.append("ab_result_ingestion_not_ready")
    if not bool(quality_decision.get("quality_review_ready", quality_decision.get("ok", False))):
        blockers.append("quality_review_not_ready")
    if _unsafe_flags(ab_ingestion, quality_decision):
        blockers.append("unsafe_review_child_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_quality_review_decision_v0",
        "ok": ready,
        "trainer_ab_quality_review_ready": ready,
        "quality_review_ready": ready,
        "passed_reducers": list(quality_decision.get("passed_reducers") or []),
        "passed_reducer_count": int(quality_decision.get("passed_reducer_count") or 0),
        "dispatch_result_ingestion_scorecard": str(dispatch_ingestion.get("scorecard") or ""),
        "ab_result_ingestion": ab_ingestion,
        "quality_decision": quality_decision,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off compute reducer rollout proposal after signed trainer A/B review"
            if ready
            else "complete trainer A/B result ingestion and signed quality review"
        ),
    }


def _patch_row(
    case: Mapping[str, Any],
    arm: str,
    base_request: Mapping[str, Any],
    include_dry_run_flag: bool,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    reducer_id = str(case.get("reducer_id") or "")
    request = dict(base_request)
    request.update(
        {
            "dit_compute_reducer_ab_case_id": str(case.get("case_id") or ""),
            "dit_compute_reducer_ab_arm": arm,
            "model_family": str(case.get("family") or "anima"),
            "max_train_steps": int(case.get("max_train_steps") or 1),
            "resolution": int(case.get("resolution") or 1),
            "train_batch_size": int(case.get("batch_size") or 1),
            "seed": int(case.get("seed") or 0),
            "cache_first": True,
            "native_dit_required": True,
            "dit_compute_reducer_ab_contract": "dit_compute_reducer_trainer_ab_manifest_v0",
            "dit_compute_reducer_selected": reducer_id,
            "dit_compute_reducer_thresholds": dict(manifest.get("thresholds") or {}),
        }
    )
    request.update(dict(case.get(f"{arm}_request_overrides") or {}))
    if arm == "candidate":
        request["dit_compute_reducer_policy"] = {
            "enabled": True,
            "reducer_id": reducer_id,
            "mode": "report_only_candidate",
        }
        result_path = str(case.get("candidate_result_path") or "")
    else:
        request["dit_compute_reducer_policy"] = {"enabled": False, "reducer_id": "disabled"}
        result_path = str(case.get("baseline_result_path") or "")
    if include_dry_run_flag:
        request["dry_run"] = True
    return {
        "case_id": str(case.get("case_id") or ""),
        "arm": arm,
        "family": str(case.get("family") or ""),
        "reducer_id": reducer_id,
        "request_patch": request,
        "expected_result_path": result_path,
    }


def _dispatch_payload(row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(row.get("request_patch") or {})
    request["dry_run"] = False
    request["dit_compute_reducer_ab_dispatch_reviewed"] = True
    request["dit_compute_reducer_ab_dispatcher"] = str(contract.get("dispatcher") or "")
    return {
        "case_id": str(row.get("case_id") or ""),
        "arm": str(row.get("arm") or ""),
        "family": str(row.get("family") or ""),
        "reducer_id": str(row.get("reducer_id") or ""),
        "expected_result_path": str(row.get("expected_result_path") or ""),
        "request": request,
    }


def _manifest_from_payloads(payloads: Sequence[Mapping[str, Any]], *, thresholds: Any) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        case_id = str(payload.get("case_id") or "")
        if not case_id:
            continue
        request = dict(payload.get("request") or {})
        row = by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "reducer_id": str(payload.get("reducer_id") or ""),
                "family": str(payload.get("family") or "anima"),
                "max_train_steps": int(request.get("max_train_steps") or 1),
                "warmup_steps": 0,
                "thresholds": dict(thresholds or {}),
            },
        )
        if str(payload.get("arm") or "") == "baseline":
            row["baseline_result_path"] = str(payload.get("expected_result_path") or "")
        elif str(payload.get("arm") or "") == "candidate":
            row["candidate_result_path"] = str(payload.get("expected_result_path") or "")
    return {
        "manifest": "dit_compute_reducer_trainer_ab_manifest_v0",
        "scorecard": "dit_compute_reducer_trainer_ab_manifest_v0",
        "ok": bool(by_case),
        "runner_ready": bool(by_case),
        "trainer_ab_manifest_ready": bool(by_case),
        "thresholds": dict(thresholds or {}),
        "cases": list(by_case.values()),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


def _evidence_package_from_dispatch_ingestion(dispatch_ingestion: Mapping[str, Any]) -> dict[str, Any]:
    result_summaries = [
        dict(item)
        for item in dispatch_ingestion.get("result_summaries_for_ingestion", ())
        if isinstance(item, Mapping)
    ]
    selected = sorted({str(item.get("reducer_id") or "") for item in result_summaries if item.get("reducer_id")})
    thresholds = dict((dispatch_ingestion.get("result_gate") or {}).get("thresholds") or {})
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_ab_evidence_package_v0",
        "ok": bool(selected),
        "evidence_package_ready": bool(selected),
        "selected_reducers": selected,
        "thresholds": thresholds,
        "source_scorecard": str(dispatch_ingestion.get("scorecard") or ""),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
    }


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    case_id = str(case.get("case_id") or "")
    blockers: list[str] = []
    if not case_id:
        blockers.append("case_id_missing")
    if str(case.get("reducer_id") or "") not in {"tread", "diffcr", "blockskip", "local_window_attention"}:
        blockers.append(f"{case_id}:unsupported_reducer")
    if not bool(case.get("cache_first_required", False)):
        blockers.append(f"{case_id}:cache_first_required_missing")
    if not bool(case.get("native_dit_required", False)):
        blockers.append(f"{case_id}:native_dit_required_missing")
    return blockers


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "request_fields_emitted",
        "request_adapter_registered",
        "runtime_activation_enabled",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_dit_compute_reducer_trainer_ab_dispatch_manifest",
    "build_dit_compute_reducer_trainer_ab_dispatch_result_ingestion",
    "build_dit_compute_reducer_trainer_ab_quality_review_decision",
    "build_dit_compute_reducer_trainer_ab_manual_dispatch_review",
    "build_dit_compute_reducer_trainer_ab_request_patch_plan",
]
