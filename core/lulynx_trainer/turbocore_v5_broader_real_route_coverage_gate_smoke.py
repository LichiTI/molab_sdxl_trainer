"""Smoke checks for V5-P37 broader real-route coverage gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_broader_real_route_coverage_gate import (  # noqa: E402
    build_v5_broader_real_route_coverage_gate,
)
from core.turbocore_v5_post_archive_rollout_scope import build_v5_post_archive_rollout_scope  # noqa: E402
from lulynx_trainer.turbocore_v5_post_archive_rollout_scope_smoke import (  # noqa: E402
    _p35_ready,
    _readiness_report,
    _scope_review,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p36_ready = _p36_ready()
    ready = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(),
    )
    assert ready["ok"] is True, ready
    assert ready["broader_real_route_coverage_ready"] is True, ready
    assert ready["decision"] == "broader_real_route_coverage_ready_default_off", ready
    assert ready["evidence_quality_summary"]["ready_route_count"] == 3, ready
    assert ready["evidence_quality_summary"]["digest_count"] == 3, ready
    assert ready["covered_route_ids"] == ["native_update_dispatch", "lora_fused_dispatch", "native_data_prefetch"], ready
    assert ready["post_p37_request_fields"] == {}, ready
    _assert_default_off(ready)

    missing_route = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(drop="native_data_prefetch"),
    )
    _assert_blocked(missing_route, "route_missing", "native_data_prefetch")

    smoke_only = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(quality="smoke"),
    )
    _assert_blocked(smoke_only, "quality_insufficient")

    p36_missing = build_v5_broader_real_route_coverage_gate(route_evidence=_route_evidence())
    _assert_blocked(p36_missing, "p36", "missing")

    p36_dirty = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification={**p36_ready, "ui_exposure_allowed": True},
        route_evidence=_route_evidence(),
    )
    _assert_blocked(p36_dirty, "p36", "not_ready")

    p36_default_changed = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification={**p36_ready, "default_behavior_changed": True},
        route_evidence=_route_evidence(),
    )
    _assert_blocked(p36_default_changed, "default_behavior_changed")

    p36_review_missing = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification={**p36_ready, "manual_review_required": False},
        route_evidence=_route_evidence(),
    )
    _assert_blocked(p36_review_missing, "p36", "not_ready")

    route_missing_ready_flag = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{key: value for key, value in item.items() if key != "coverage_ready"} for item in _route_evidence()],
    )
    _assert_blocked(route_missing_ready_flag, "route_not_ready")

    route_not_ok = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "ok": False} for item in _route_evidence()],
    )
    _assert_blocked(route_not_ok, "route_not_ok")

    route_missing_digest = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{key: value for key, value in item.items() if key != "source_report_digest"} for item in _route_evidence()],
    )
    _assert_blocked(route_missing_digest, "digest_missing")

    route_missing_source = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{key: value for key, value in item.items() if key != "source"} for item in _route_evidence()],
    )
    _assert_blocked(route_missing_source, "source_missing")

    route_zero_metrics = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "representative_steps": 0, "sample_count": 0} for item in _route_evidence()],
    )
    _assert_blocked(route_zero_metrics, "steps_insufficient", "sample_count_insufficient")

    route_default_violation = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "default_rollout_allowed": True} for item in _route_evidence()],
    )
    _assert_blocked(route_default_violation, "default_off")

    route_request_adapter = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "request_adapter_mapping_allowed": True} for item in _route_evidence()],
    )
    _assert_blocked(route_request_adapter, "request_adapter")

    route_launch_requested = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "training_launch_allowed": True} for item in _route_evidence()],
    )
    _assert_blocked(route_launch_requested, "launch")

    route_runs_dispatched = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "runs_dispatched": True} for item in _route_evidence()],
    )
    _assert_blocked(route_runs_dispatched, "runs_dispatched")

    route_manual_only_missing = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=[{**item, "manual_only": False} for item in _route_evidence()],
    )
    _assert_blocked(route_manual_only_missing, "manual_only")

    failure_history = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(),
        failure_history=[{"reason": "native_dispatch_timeout", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    closed_failure_history = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(),
        failure_history=[{"reason": "closed_transient", "status": "closed", "severity": "high"}],
    )
    assert closed_failure_history["ok"] is True, closed_failure_history

    rollback_history = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(),
        rollback_history=[{"kind": "route_rollback", "cooldown_active": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")

    unexpected_route = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence() + [_route("unknown_route")],
    )
    _assert_blocked(unexpected_route, "unexpected")

    duplicate_route = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence() + [_route("native_update_dispatch", digest_suffix="duplicate")],
    )
    _assert_blocked(duplicate_route, "duplicate")

    malformed_route = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence() + [{**_route("temporary_route"), "route_id": ""}],
    )
    _assert_blocked(malformed_route, "route_id_missing")

    blank_required_routes = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=p36_ready,
        route_evidence=_route_evidence(),
        required_routes=[" "],
    )
    _assert_blocked(blank_required_routes, "required_routes_empty")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p37_broader_real_route_coverage_gate_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_route": _summary(missing_route),
        "smoke_only": _summary(smoke_only),
        "p36_missing": _summary(p36_missing),
        "p36_dirty": _summary(p36_dirty),
        "p36_default_changed": _summary(p36_default_changed),
        "p36_review_missing": _summary(p36_review_missing),
        "route_missing_ready_flag": _summary(route_missing_ready_flag),
        "route_not_ok": _summary(route_not_ok),
        "route_missing_digest": _summary(route_missing_digest),
        "route_missing_source": _summary(route_missing_source),
        "route_zero_metrics": _summary(route_zero_metrics),
        "route_default_violation": _summary(route_default_violation),
        "route_request_adapter": _summary(route_request_adapter),
        "route_launch_requested": _summary(route_launch_requested),
        "route_runs_dispatched": _summary(route_runs_dispatched),
        "route_manual_only_missing": _summary(route_manual_only_missing),
        "failure_history": _summary(failure_history),
        "closed_failure_history": _summary(closed_failure_history),
        "rollback_history": _summary(rollback_history),
        "unexpected_route": _summary(unexpected_route),
        "duplicate_route": _summary(duplicate_route),
        "malformed_route": _summary(malformed_route),
        "blank_required_routes": _summary(blank_required_routes),
    }


def _p36_ready() -> dict[str, Any]:
    return build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=_p35_ready(),
        readiness_report=_readiness_report(broader=True, packaging=True),
        rollout_policy_review=_scope_review(),
    )


def _route_evidence(*, quality: str = "real_route_manual_replay", drop: str = "") -> list[dict[str, Any]]:
    routes = ("native_update_dispatch", "lora_fused_dispatch", "native_data_prefetch")
    return [_route(route, quality=quality) for route in routes if route != drop]


def _route(
    route_id: str,
    *,
    quality: str = "real_route_manual_replay",
    digest_suffix: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "route_id": route_id,
        "ok": True,
        "coverage_ready": True,
        "promotion_ready": True,
        "evidence_quality": quality,
        "source_report_digest": f"sha256:{route_id}:{digest_suffix or 'ready'}",
        "source": f"temp/turbocore_v5_p37_{route_id}_{digest_suffix or 'ready'}.json",
        "manual_only": True,
        "real_route": True,
        "default_off": True,
        "request_adapter_off": True,
        "representative_steps": 768,
        "sample_count": 5,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "blocked_reasons": [],
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["training_launch_allowed"] is False, report
    assert report["auto_launch_allowed"] is False, report
    assert report["runs_dispatched"] is False, report
    assert report["ui_exposure_allowed"] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["broader_real_route_coverage_ready"] is False, report
    assert report["decision"] == "broader_real_route_coverage_blocked_default_off", report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "broader_real_route_coverage_ready": bool(report.get("broader_real_route_coverage_ready", False)),
        "ready_route_count": int((report.get("evidence_quality_summary") or {}).get("ready_route_count", 0) or 0),
        "required_route_count": int((report.get("evidence_quality_summary") or {}).get("required_route_count", 0) or 0),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
