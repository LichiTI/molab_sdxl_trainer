"""Result ingestion gate for reviewed T-LoRA A/B dispatch payloads."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .tlora_ab_runner_manifest import build_tlora_ab_result_gate


def build_tlora_ab_dispatch_result_ingestion(
    *,
    dispatch_manifest: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(dispatch_manifest)
    payloads = [dict(item) for item in manifest.get("payloads", []) if isinstance(item, Mapping)]
    blockers: list[str] = []
    if manifest.get("scorecard") != "tlora_ab_dispatch_manifest_v0":
        blockers.append("unexpected_dispatch_manifest")
    if not bool(manifest.get("dispatch_manifest_ready", manifest.get("ok", False))):
        blockers.append("dispatch_manifest_not_ready")
    if bool(manifest.get("execution_performed", False)):
        blockers.append("dispatch_manifest_claims_execution")
    if bool(manifest.get("training_path_enabled", False)):
        blockers.append("unsafe_manifest_training_path_enabled")
    if not payloads:
        blockers.append("dispatch_payloads_missing")

    result_gate = build_tlora_ab_result_gate(
        _manifest_from_payloads(payloads),
        case_results,
        thresholds=thresholds,
    )
    if not bool(result_gate.get("ab_result_ready", result_gate.get("ok", False))):
        blockers.append("ab_result_gate_not_ready")
    if bool(result_gate.get("training_path_enabled", False)) or bool(result_gate.get("promotion_ready", False)):
        blockers.append("unsafe_result_gate_flag")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_dispatch_result_ingestion_v0",
        "ok": ready,
        "result_ingestion_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "case_count": int(result_gate.get("case_count") or manifest.get("case_count") or 0),
        "result_count": int(result_gate.get("result_count") or 0),
        "result_gate": result_gate,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "review ingested representative T-LoRA A/B results for promotion decision"
            if ready
            else "collect all reviewed dispatch results before promotion review"
        ),
    }


def _manifest_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        case_id = str(payload.get("case_id") or "")
        if not case_id:
            continue
        row = by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "family": str(payload.get("family") or "anima"),
            },
        )
        arm = str(payload.get("arm") or "")
        if arm == "baseline":
            row["baseline_result_path"] = str(payload.get("expected_result_path") or "")
        elif arm == "tlora":
            row["tlora_result_path"] = str(payload.get("expected_result_path") or "")
    return {
        "manifest": "tlora_ab_runner_manifest_v0",
        "ok": bool(by_case),
        "runner_ready": bool(by_case),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "case_count": len(by_case),
        "cases": list(by_case.values()),
    }


__all__ = ["build_tlora_ab_dispatch_result_ingestion"]
