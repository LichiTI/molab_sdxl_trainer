"""Stable first-release scope for TurboCore optimizer default-off delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_owner_release_review_packet import (
    build_native_update_owner_release_review_packet,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_run_local_staging import (
    build_optimizer_product_training_route_binding_run_local_staging,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_stable_first_release_scope.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
READY_NEXT_STEP = (
    "ship stable baseline with TurboCore optimizer default-off, or collect owner/product approvals "
    "for native exposure"
)


def build_turbocore_optimizer_stable_first_release_scope(
    *,
    owner_packet: Mapping[str, Any] | None = None,
    route_preflight: Mapping[str, Any] | None = None,
    run_local_staging: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    packet = _artifact_or_build(
        owner_packet,
        directory / "native_update_owner_release_review_packet.json",
        build_native_update_owner_release_review_packet,
    )
    preflight = _artifact_or_build(
        route_preflight,
        directory / "turbocore_optimizer_product_training_route_binding_preflight.json",
        build_optimizer_product_training_route_binding_preflight,
    )
    staging = _artifact_or_build(
        run_local_staging,
        directory / "turbocore_optimizer_product_training_route_binding_run_local_staging.json",
        build_optimizer_product_training_route_binding_run_local_staging,
    )
    packet_evidence = _as_dict(packet.get("compact_evidence"))
    unsafe = _unsafe_reasons(packet, preflight, staging)
    ready = not unsafe and packet.get("ready_for_owner_signature") is True
    payload = {
        "schema_version": 1,
        "artifact": "turbocore_optimizer_stable_first_release_scope_v0",
        "gate": "optimizer_stable_first_release_default_off_scope",
        "ok": True,
        "roadmap": ROADMAP,
        "stable_first_release_scope": "stable_baseline_with_turbocore_optimizer_default_off",
        "stable_first_release_blocked_by_turbocore_optimizer": not ready,
        "turbocore_optimizer_default_off_release_scope_ready": ready,
        "owner_signature_ready": packet.get("ready_for_owner_signature") is True,
        "owner_release_approval_recorded": False,
        "product_exposure_decision_recorded": False,
        "product_training_route_binding_ready": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "release_claim_allowed": ready,
        "native_training_claim_allowed": False,
        "blocked_reasons": unsafe,
        "summary": {
            "stable_first_release_turbocore_optimizer_blocker_count": 0 if ready else 1,
            "turbocore_optimizer_default_off_release_scope_ready_count": 1 if ready else 0,
            "owner_signature_ready_count": 1 if packet.get("ready_for_owner_signature") is True else 0,
            "owner_release_approval_recorded_count": 0,
            "owner_release_direction_recorded_count": int(
                packet_evidence.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                packet_evidence.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "product_exposure_decision_recorded_count": 0,
            "product_training_route_binding_ready_count": 0,
            "run_local_adapter_staged_count": _summary_int(staging, "run_local_adapter_staged_count"),
            "runtime_config_patch_applied_count": _summary_int(staging, "runtime_config_patch_applied_count"),
            "training_path_enabled_count": 0,
        },
        "recommended_next_step": (
            READY_NEXT_STEP
            if ready
            else "repair TurboCore optimizer default-off release evidence before first release"
        ),
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _artifact_or_build(
    supplied: Mapping[str, Any] | None,
    path: Path,
    builder: Any,
) -> dict[str, Any]:
    if isinstance(supplied, Mapping):
        return dict(supplied)
    if path.exists():
        try:
            return _as_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}
    return _as_dict(builder(artifact_dir=path.parent, write_artifact=True))


def _unsafe_reasons(
    packet: Mapping[str, Any],
    preflight: Mapping[str, Any],
    staging: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if packet.get("ready_for_owner_signature") is not True:
        reasons.append("owner_signature_packet_not_ready")
    packet_evidence = _as_dict(packet.get("compact_evidence"))
    for field in (
        "owner_release_direction_recorded_count",
        "owner_release_direction_approval_recorded_count",
    ):
        if int(packet_evidence.get(field, 0) or 0) != 0:
            reasons.append(f"{field}_before_stable_first_release")
    for name, report in (("owner_packet", packet), ("route_preflight", preflight), ("run_local_staging", staging)):
        for field in _MUST_REMAIN_FALSE:
            if report.get(field) is True:
                reasons.append(f"{name}_{field}_unexpectedly_enabled")
    if preflight.get("product_training_route_binding_preflight_ready") is True:
        reasons.append("route_binding_ready_before_stable_first_release")
    if staging.get("run_local_adapter_staged") is True:
        reasons.append("run_local_adapter_staged_before_stable_first_release")
    return reasons


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    return int(summary.get(key, 0) or 0)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


_MUST_REMAIN_FALSE = (
    "product_exposure_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "ui_exposure_allowed",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "training_path_enabled",
    "training_launch_executed",
)


__all__ = ["build_turbocore_optimizer_stable_first_release_scope"]
