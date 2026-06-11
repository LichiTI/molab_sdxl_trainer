"""Smoke checks for V5-P38 packaging / observability evidence gate."""

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
from core.turbocore_v5_packaging_observability_evidence_gate import (  # noqa: E402
    build_v5_packaging_observability_evidence_gate,
)
from lulynx_trainer.turbocore_v5_broader_real_route_coverage_gate_smoke import (  # noqa: E402
    _p36_ready,
    _route_evidence,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)
PACKAGES = ("archive_ledger", "offline_runtime_pack", "evidence_bundle")
CHANNELS = ("state", "logs", "events", "reports")


def run_smoke() -> dict[str, Any]:
    p37_ready = _p37_ready()
    ready = _gate(p37_ready)
    assert ready["ok"] is True, ready
    assert ready["packaging_observability_ready"] is True, ready
    assert ready["decision"] == "packaging_observability_evidence_gate_ready_default_off", ready
    assert ready["artifact_digest_summary"]["ready_package_count"] == 3, ready
    assert ready["telemetry_channel_summary"]["ready_channel_count"] == 4, ready
    assert ready["report_surface_summary"]["ready_report_count"] == 1, ready
    assert ready["post_p38_request_fields"] == {}, ready
    _assert_default_off(ready)

    p37_missing = _gate(None)
    _assert_blocked(p37_missing, "p37", "missing")

    p37_decision_mismatch = _gate({**p37_ready, "decision": "wrong"})
    _assert_blocked(p37_decision_mismatch, "p37", "not_ready")

    p37_unsafe = _gate({**p37_ready, "ui_exposure_allowed": True})
    _assert_blocked(p37_unsafe, "ui_exposure_allowed")

    p37_post_fields = _gate({**p37_ready, "post_p37_request_fields": {"bad": True}})
    _assert_blocked(p37_post_fields, "post_p37_request_fields")

    package_missing = _gate(p37_ready, packaging=[_package(item) for item in PACKAGES if item != "evidence_bundle"])
    _assert_blocked(package_missing, "package_missing", "evidence_bundle")

    package_not_ready = _gate(p37_ready, packaging=[{**_package(item), "packaging_evidence_ready": False} for item in PACKAGES])
    _assert_blocked(package_not_ready, "package_evidence_not_ready")

    package_missing_digest = _gate(p37_ready, packaging=[_without(_package(item), "sha256") for item in PACKAGES])
    _assert_blocked(package_missing_digest, "package_digest_missing")

    package_incomplete = _gate(p37_ready, packaging=[{**_package(item), "complete": False, "artifact_count": 0} for item in PACKAGES])
    _assert_blocked(package_incomplete, "package_incomplete", "artifact_count")

    package_unsafe = _gate(p37_ready, packaging=[{**_package(item), "default_rollout_allowed": True} for item in PACKAGES])
    _assert_blocked(package_unsafe, "default_rollout_allowed")

    telemetry_missing = _gate(p37_ready, observability=[_telemetry(item) for item in CHANNELS if item != "events"])
    _assert_blocked(telemetry_missing, "telemetry_channel_missing", "events")

    telemetry_not_ready = _gate(
        p37_ready,
        observability=[{**_telemetry(item), "observability_evidence_ready": False} for item in CHANNELS],
    )
    _assert_blocked(telemetry_not_ready, "telemetry_evidence_not_ready")

    telemetry_missing_digest = _gate(p37_ready, observability=[_without(_telemetry(item), "sha256") for item in CHANNELS])
    _assert_blocked(telemetry_missing_digest, "telemetry_digest_missing")

    telemetry_claim_only = _gate(p37_ready, observability=[{**_telemetry(item), "report_only": False} for item in CHANNELS])
    _assert_blocked(telemetry_claim_only, "report_only")

    telemetry_fallback = _gate(
        p37_ready,
        observability=[{**_telemetry(item), "available": False, "fallback_available": True} for item in CHANNELS],
    )
    assert telemetry_fallback["ok"] is True, telemetry_fallback
    assert telemetry_fallback["telemetry_channel_summary"]["degraded_channel_count"] == 4, telemetry_fallback

    telemetry_unsafe = _gate(p37_ready, observability=[{**_telemetry(item), "training_launch_allowed": True} for item in CHANNELS])
    _assert_blocked(telemetry_unsafe, "training_launch_allowed")

    report_missing = _gate(p37_ready, reports=[])
    _assert_blocked(report_missing, "report_missing")

    report_not_ready = _gate(p37_ready, reports=[{**_report(), "report_evidence_ready": False}])
    _assert_blocked(report_not_ready, "report_evidence_not_ready")

    report_missing_section = _gate(p37_ready, reports=[{**_report(), "missing_sections": ["recovery"]}])
    _assert_blocked(report_missing_section, "section_missing", "recovery")

    report_missing_cross_digest = _gate(p37_ready, reports=[_without(_report(), "packaging_digest")])
    _assert_blocked(report_missing_cross_digest, "cross_digest_missing", "packaging_digest")

    report_blocked = _gate(p37_ready, reports=[{**_report(), "blocked_reasons": ["report_schema_dirty"]}])
    _assert_blocked(report_blocked, "report_schema_dirty")

    report_post_fields = _gate(p37_ready, reports=[{**_report(), "post_report_request_fields": {"bad": True}}])
    _assert_blocked(report_post_fields, "post_report_request_fields")

    failure_history = _gate(
        p37_ready,
        failure_history=[{"reason": "packaging_manifest_mismatch", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    closed_failure = _gate(
        p37_ready,
        failure_history=[{"reason": "closed_packaging_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure

    rollback_history = _gate(
        p37_ready,
        rollback_history=[{"kind": "observability_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p38_packaging_observability_evidence_gate_smoke",
        "ok": True,
        "ready": _summary(ready),
        "p37_missing": _summary(p37_missing),
        "p37_decision_mismatch": _summary(p37_decision_mismatch),
        "p37_unsafe": _summary(p37_unsafe),
        "p37_post_fields": _summary(p37_post_fields),
        "package_missing": _summary(package_missing),
        "package_not_ready": _summary(package_not_ready),
        "package_missing_digest": _summary(package_missing_digest),
        "package_incomplete": _summary(package_incomplete),
        "package_unsafe": _summary(package_unsafe),
        "telemetry_missing": _summary(telemetry_missing),
        "telemetry_not_ready": _summary(telemetry_not_ready),
        "telemetry_missing_digest": _summary(telemetry_missing_digest),
        "telemetry_claim_only": _summary(telemetry_claim_only),
        "telemetry_fallback": _summary(telemetry_fallback),
        "telemetry_unsafe": _summary(telemetry_unsafe),
        "report_missing": _summary(report_missing),
        "report_not_ready": _summary(report_not_ready),
        "report_missing_section": _summary(report_missing_section),
        "report_missing_cross_digest": _summary(report_missing_cross_digest),
        "report_blocked": _summary(report_blocked),
        "report_post_fields": _summary(report_post_fields),
        "failure_history": _summary(failure_history),
        "closed_failure": _summary(closed_failure),
        "rollback_history": _summary(rollback_history),
    }


def _gate(
    p37: dict[str, Any] | None,
    *,
    packaging: list[dict[str, Any]] | None = None,
    observability: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    return build_v5_packaging_observability_evidence_gate(
        p37_coverage_gate=p37,
        packaging_evidence=_packages() if packaging is None else packaging,
        observability_evidence=_telemetry_rows() if observability is None else observability,
        report_evidence=[_report()] if reports is None else reports,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p37_ready() -> dict[str, Any]:
    return build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=_p36_ready(),
        route_evidence=_route_evidence(),
    )


def _packages() -> list[dict[str, Any]]:
    return [_package(item) for item in PACKAGES]


def _package(package_id: str) -> dict[str, Any]:
    return {
        "package_id": package_id,
        "kind": "offline_runtime_pack" if package_id == "offline_runtime_pack" else "archive_artifact",
        "ok": True,
        "packaging_evidence_ready": True,
        "complete": True,
        "portable": True,
        "artifact_count": 2,
        "sha256": f"sha256:{package_id}:ready",
        "source": f"temp/turbocore_v5_p38_{package_id}.json",
        **_safe_row_flags(),
    }


def _telemetry_rows() -> list[dict[str, Any]]:
    return [_telemetry(item) for item in CHANNELS]


def _telemetry(channel_id: str) -> dict[str, Any]:
    return {
        "channel_id": channel_id,
        "provider": "file_telemetry_reader",
        "ok": True,
        "observability_evidence_ready": True,
        "report_only": True,
        "available": True,
        "fallback_available": False,
        "sample_run_count": 2,
        "sha256": f"sha256:{channel_id}:ready",
        "source": f"temp/turbocore_v5_p38_telemetry_{channel_id}.json",
        **_safe_row_flags(),
    }


def _report() -> dict[str, Any]:
    return {
        "report_id": "training_report",
        "ok": True,
        "report_evidence_ready": True,
        "schema_version": 1,
        "required_sections": ["metrics", "diagnostics", "recovery", "reproducibility"],
        "available_sections": ["metrics", "diagnostics", "recovery", "reproducibility"],
        "missing_sections": [],
        "sha256": "sha256:training_report:ready",
        "source": "temp/turbocore_v5_p38_training_report.json",
        "p37_digest": "sha256:p37:ready",
        "packaging_digest": "sha256:packaging:ready",
        "observability_digest": "sha256:observability:ready",
        **_safe_row_flags(),
    }


def _safe_row_flags() -> dict[str, Any]:
    return {
        "default_off": True,
        "request_adapter_off": True,
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
        "promotion_blockers": [],
    }


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    copied = dict(value)
    copied.pop(key, None)
    return copied


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["training_launch_allowed"] is False, report
    assert report["auto_launch_allowed"] is False, report
    assert report["runs_dispatched"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["post_p38_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["packaging_observability_ready"] is False, report
    assert report["decision"] == "packaging_observability_evidence_gate_blocked_default_off", report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "packaging_observability_ready": bool(report.get("packaging_observability_ready", False)),
        "ready_package_count": int((report.get("artifact_digest_summary") or {}).get("ready_package_count", 0) or 0),
        "ready_channel_count": int((report.get("telemetry_channel_summary") or {}).get("ready_channel_count", 0) or 0),
        "ready_report_count": int((report.get("report_surface_summary") or {}).get("ready_report_count", 0) or 0),
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
