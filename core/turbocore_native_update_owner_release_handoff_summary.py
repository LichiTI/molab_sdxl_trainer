"""Compact owner-release handoff summary for TurboCore native update."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_release_review_package import (
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_release_review_package,
    load_gate_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
RELEASE_PACKAGE_ARTIFACT = ARTIFACT_DIR / "native_update_release_review_package.json"
MULTITENSOR_RELEASE_HOLD_ARTIFACT = ARTIFACT_DIR / "native_update_optimizer_multitensor_release_hold.json"
REPRESENTATIVE_PERFORMANCE_SUMMARY_ARTIFACT = (
    ARTIFACT_DIR / "native_update_representative_performance_summary.json"
)
NATIVE_READINESS_GAP_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_native_readiness_gap_scorecard.json"
PRODUCT_ROUTE_PREFLIGHT_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_preflight.json"
)
TRAINING_LOOP_ROUTE_CONTRACT_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_training_loop_contract.json"
)
CONFIG_ADAPTER_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_config_adapter.json"
PRODUCT_ROUTE_ADAPTER_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_product_route_adapter.json"
)
RUNTIME_APPLIER_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_runtime_applier.json"
)
RUN_LOCAL_STAGING_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_run_local_staging.json"
)
STABLE_FIRST_RELEASE_SCOPE_ARTIFACT = (
    ARTIFACT_DIR / "turbocore_optimizer_stable_first_release_scope.json"
)
OWNER_RELEASE_DIRECTION_RECORD_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_record.json"
OWNER_RELEASE_DIRECTION_ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_direction_packet.json"
ARTIFACT = ARTIFACT_DIR / "native_update_owner_release_handoff_summary.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def build_native_update_owner_release_handoff_summary(
    *,
    release_package: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    package = _release_package(release_package, artifact_dir=artifact_dir)
    handoff = _as_dict(package.get("owner_release_review_handoff"))
    review_template = _as_dict(package.get("release_review_template"))
    supplemental = _as_dict(package.get("supplemental_gate_summaries"))
    optimizer = _as_dict(supplemental.get("optimizer_family_coverage"))
    optimizer_counts = _as_dict(optimizer.get("optimizer_family_counts"))
    multitensor = _as_dict(supplemental.get("native_update_optimizer_multitensor_release_hold"))
    multitensor_counts = _multitensor_release_hold_counts(artifact_dir=artifact_dir)
    performance_summary = _representative_performance_summary(artifact_dir=artifact_dir)
    performance_gate = _as_dict(performance_summary.get("performance_gate"))
    native_readiness = _native_readiness_gap_summary(artifact_dir=artifact_dir)
    route_preflight = _product_route_preflight_summary(artifact_dir=artifact_dir)
    training_loop_route = _training_loop_route_contract_summary(artifact_dir=artifact_dir)
    config_adapter = _config_adapter_summary(artifact_dir=artifact_dir)
    product_route_adapter = _product_route_adapter_summary(artifact_dir=artifact_dir)
    runtime_applier = _runtime_applier_summary(artifact_dir=artifact_dir)
    run_local_staging = _run_local_staging_summary(artifact_dir=artifact_dir)
    stable_first_release = _stable_first_release_scope_summary(artifact_dir=artifact_dir)
    owner_release_direction = _owner_release_direction_summary(artifact_dir=artifact_dir)
    blocked_reasons = _strings(package.get("blocked_reasons"))
    required_review_fields = _strings(handoff.get("required_review_fields"))
    must_remain_false = _strings(handoff.get("must_remain_false"))
    must_remain_empty = _strings(handoff.get("must_remain_empty"))
    technical_evidence_ready = bool(
        package.get("ok") is True
        and package.get("evidence_ready") is True
        and package.get("ready_for_owner_release_review") is True
        and package.get("default_off") is True
        and package.get("expected_gate_count") == package.get("present_gate_count")
        and package.get("expected_gate_count") == package.get("default_off_gate_count")
        and package.get("supplemental_gate_count") == package.get("present_supplemental_gate_count")
        and package.get("supplemental_gate_count") == package.get("default_off_supplemental_gate_count")
    )
    review_recorded = package.get("release_review_recorded") is True
    decision = str(package.get("decision", "") or "")
    ok = bool(
        technical_evidence_ready
        and decision in {HOLD_DECISION, READY_DECISION}
        and not _unsafe_top_level_enabled(package)
        and bool(handoff.get("release_review_template_digest"))
    )
    summary = {
        "expected_gate_count": int(package.get("expected_gate_count", 0) or 0),
        "present_gate_count": int(package.get("present_gate_count", 0) or 0),
        "default_off_gate_count": int(package.get("default_off_gate_count", 0) or 0),
        "supplemental_gate_count": int(package.get("supplemental_gate_count", 0) or 0),
        "present_supplemental_gate_count": int(package.get("present_supplemental_gate_count", 0) or 0),
        "default_off_supplemental_gate_count": int(package.get("default_off_supplemental_gate_count", 0) or 0),
        "plugin_optimizer_count": int(optimizer_counts.get("plugin_optimizer_count", 0) or 0),
        "plugin_selected_native_ready_count": int(optimizer_counts.get("plugin_selected_native_ready_count", 0) or 0),
        "optimizer_inventory_source_ready_count": int(
            optimizer_counts.get("optimizer_native_kernel_inventory_source_ready_count", 0) or 0
        ),
        "optimizer_inventory_probe_ready_count": int(
            optimizer_counts.get("optimizer_native_kernel_inventory_probe_ready_count", 0) or 0
        ),
        "optimizer_family_contract_ready_count": int(
            optimizer_counts.get("optimizer_family_kernel_contract_ready_count", 0) or 0
        ),
        "multitensor_native_kernel_launch_count": int(
            multitensor_counts.get("native_kernel_launch_count", 0) or 0
        ),
        "multitensor_top_level_native_dispatch_allowed_count": int(
            multitensor_counts.get("top_level_native_dispatch_allowed_count", 0) or 0
        ),
        "multitensor_training_parameter_mutation_count": int(
            multitensor_counts.get("training_parameter_mutation_count", 0) or 0
        ),
        "representative_performance_artifact_present_count": 1
        if performance_summary.get("performance_artifact_present") is True
        else 0,
        "representative_performance_gate_ready_count": 1
        if performance_summary.get("representative_performance_gate_ready") is True
        else 0,
        "representative_performance_fresh_live_run_count": 1
        if performance_summary.get("fresh_live_run") is True
        else 0,
        "representative_performance_training_matrix_steps": int(
            performance_gate.get("training_matrix_representative_steps", 0) or 0
        ),
        "representative_performance_end_to_end_speedup": performance_gate.get(
            "training_matrix_end_to_end_speedup"
        ),
        "native_readiness_runtime_launch_coverage_ready_family_count": int(
            native_readiness.get("runtime_launch_coverage_ready_family_count", 0) or 0
        ),
        "native_readiness_runtime_launch_adapter_ready_family_count": int(
            native_readiness.get("family_specific_runtime_launch_adapter_ready_family_count", 0) or 0
        ),
        "native_readiness_runtime_launch_adapter_ready_optimizer_count": int(
            native_readiness.get("family_specific_runtime_launch_adapter_ready_optimizer_count", 0) or 0
        ),
        "native_readiness_owner_release_hold_ready_family_count": int(
            native_readiness.get("owner_release_hold_ready_family_count", 0) or 0
        ),
        "native_readiness_request_schema_ui_non_exposure_ready_family_count": int(
            native_readiness.get("request_schema_ui_non_exposure_ready_family_count", 0) or 0
        ),
        "native_readiness_family_specific_runtime_launch_missing_count": int(
            native_readiness.get("family_specific_runtime_launch_missing_count", 0) or 0
        ),
        "product_route_binding_preflight_ready_count": int(
            route_preflight.get("product_training_route_binding_ready_count", 0) or 0
        ),
        "product_route_binding_candidate_count": int(
            route_preflight.get("post_approval_training_route_binding_candidate_count", 0) or 0
        ),
        "product_route_binding_owner_approval_recorded_count": int(
            route_preflight.get("owner_release_approval_recorded_count", 0) or 0
        ),
        "owner_release_direction_ready_for_signature_count": int(
            owner_release_direction.get("owner_release_direction_ready_for_signature_count", 0) or 0
        ),
        "owner_release_direction_recorded_count": int(
            owner_release_direction.get("owner_release_direction_recorded_count", 0) or 0
        ),
        "owner_release_direction_approval_recorded_count": int(
            _first_count(
                owner_release_direction,
                "owner_release_direction_approval_recorded_count",
                "owner_release_approval_recorded_count",
            )
        ),
        "product_route_binding_exposure_decision_recorded_count": int(
            route_preflight.get("product_exposure_decision_recorded_count", 0) or 0
        ),
        "training_loop_route_candidate_switch_count": int(
            training_loop_route.get("candidate_switch_count", 0) or 0
        ),
        "training_loop_route_open_training_path_enabled_count": int(
            training_loop_route.get("open_training_path_enabled", 0) or 0
        ),
        "training_loop_route_request_fields_emitted_count": int(
            training_loop_route.get("request_fields_emitted_count", 0) or 0
        ),
        "training_loop_route_schema_exposure_allowed_count": int(
            training_loop_route.get("schema_exposure_allowed_count", 0) or 0
        ),
        "training_loop_route_ui_exposure_allowed_count": int(
            training_loop_route.get("ui_exposure_allowed_count", 0) or 0
        ),
        "route_binding_config_patch_ready_count": int(
            config_adapter.get("product_training_route_binding_config_patch_ready_count", 0) or 0
        ),
        "route_binding_constructor_switch_field_count": int(
            config_adapter.get("training_loop_constructor_switch_field_count", 0) or 0
        ),
        "route_binding_kwargs_patch_field_count": int(
            config_adapter.get("training_loop_kwargs_patch_field_count", 0) or 0
        ),
        "product_route_binding_product_route_count": int(
            product_route_adapter.get("product_training_route_count", 0) or 0
        ),
        "product_route_binding_kwargs_wired_count": int(
            product_route_adapter.get("product_training_route_binding_kwargs_wired_count", 0) or 0
        ),
        "route_binding_runtime_config_patch_applied_count": int(
            runtime_applier.get("runtime_config_patch_applied_count", 0) or 0
        ),
        "route_binding_runtime_config_patch_field_count": int(
            runtime_applier.get("runtime_config_patch_field_count", 0) or 0
        ),
        "route_binding_run_local_adapter_staged_count": int(
            run_local_staging.get("run_local_adapter_staged_count", 0) or 0
        ),
        "product_launch_staging_wired_count": int(
            run_local_staging.get("product_launch_staging_wired_count", 0) or 0
        ),
        "stable_first_release_turbocore_optimizer_blocker_count": int(
            stable_first_release.get("stable_first_release_turbocore_optimizer_blocker_count", 0) or 0
        ),
        "turbocore_optimizer_default_off_release_scope_ready_count": int(
            stable_first_release.get("turbocore_optimizer_default_off_release_scope_ready_count", 0) or 0
        ),
    }
    payload = {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_handoff_summary_v0",
        "gate": "native_update_owner_release_handoff_summary",
        "ok": ok,
        "roadmap": ROADMAP,
        "technical_evidence_ready": technical_evidence_ready,
        "representative_performance_evidence_complete": bool(
            performance_summary.get("release_performance_evidence_complete", False)
        ),
        "representative_performance_source_evidence_quality": str(
            performance_summary.get("source_evidence_quality", "") or ""
        ),
        "representative_performance_fresh_live_run": performance_summary.get("fresh_live_run"),
        "ready_for_owner_release_review": technical_evidence_ready,
        "release_review_recorded": review_recorded,
        "owner_action_required": not review_recorded,
        "decision": decision,
        "source_release_package_decision": decision,
        "source_release_package_digest": _as_dict(handoff).get("release_review_template_digest", ""),
        "action_required": handoff.get("action_required", package.get("recommended_next_step", "")),
        "blocked_reasons": blocked_reasons,
        "owner_blockers": [item for item in blocked_reasons if "owner" in item or "review" in item],
        "required_review_fields": required_review_fields,
        "required_requested_scope": str(handoff.get("required_requested_scope", "") or ""),
        "required_gate_acknowledgement_count": len(_strings(handoff.get("required_gate_acknowledgements"))),
        "required_supplemental_acknowledgements": _strings(
            handoff.get("required_supplemental_acknowledgements")
        ),
        "review_template_for_owner": {
            "reviewer": review_template.get("reviewer", ""),
            "reviewed_at": review_template.get("reviewed_at", ""),
            "requested_scope": review_template.get("requested_scope", ""),
            "approve_native_update_release_review_package": False,
            "acknowledge_all_expected_gates_present": bool(
                review_template.get("acknowledge_all_expected_gates_present", False)
            ),
            "acknowledge_all_gates_default_off": bool(
                review_template.get("acknowledge_all_gates_default_off", False)
            ),
            "acknowledge_no_request_ui_schema_exposure": bool(
                review_template.get("acknowledge_no_request_ui_schema_exposure", False)
            ),
            "acknowledge_no_training_launch_or_native_execution": bool(
                review_template.get("acknowledge_no_training_launch_or_native_execution", False)
            ),
            "acknowledge_product_exposure_requires_separate_owner_direction": bool(
                review_template.get("acknowledge_product_exposure_requires_separate_owner_direction", False)
            ),
            "acknowledged_gate_count": len(_as_dict(review_template.get("acknowledged_gates"))),
            "acknowledged_supplemental_gate_count": len(
                _as_dict(review_template.get("acknowledged_supplemental_gates"))
            ),
        },
        "must_remain_false": must_remain_false,
        "must_remain_empty": must_remain_empty,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "summary": summary,
        "recommended_next_step": str(package.get("recommended_next_step", "") or ""),
        "notes": [
            "This summary is report-only and does not record owner approval.",
            "The included review template keeps approval false by default.",
            "A separate owner release direction is still required before product exposure work.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _release_package(
    release_package: Mapping[str, Any] | None,
    *,
    artifact_dir: str | Path | None,
) -> dict[str, Any]:
    if release_package is not None:
        return _as_dict(release_package)
    source = RELEASE_PACKAGE_ARTIFACT if artifact_dir is None else Path(artifact_dir) / RELEASE_PACKAGE_ARTIFACT.name
    if source.exists():
        return _read_json(source)
    directory = ARTIFACT_DIR if artifact_dir is None else Path(artifact_dir)
    return build_native_update_release_review_package(gate_artifacts=load_gate_artifacts(directory))


def _multitensor_release_hold_counts(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        MULTITENSOR_RELEASE_HOLD_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / MULTITENSOR_RELEASE_HOLD_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _representative_performance_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        REPRESENTATIVE_PERFORMANCE_SUMMARY_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / REPRESENTATIVE_PERFORMANCE_SUMMARY_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _read_json(source)


def _native_readiness_gap_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        NATIVE_READINESS_GAP_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / NATIVE_READINESS_GAP_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _product_route_preflight_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        PRODUCT_ROUTE_PREFLIGHT_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / PRODUCT_ROUTE_PREFLIGHT_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _training_loop_route_contract_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        TRAINING_LOOP_ROUTE_CONTRACT_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / TRAINING_LOOP_ROUTE_CONTRACT_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _config_adapter_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = CONFIG_ADAPTER_ARTIFACT if artifact_dir is None else Path(artifact_dir) / CONFIG_ADAPTER_ARTIFACT.name
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _product_route_adapter_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        PRODUCT_ROUTE_ADAPTER_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / PRODUCT_ROUTE_ADAPTER_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _runtime_applier_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = RUNTIME_APPLIER_ARTIFACT if artifact_dir is None else Path(artifact_dir) / RUNTIME_APPLIER_ARTIFACT.name
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _run_local_staging_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        RUN_LOCAL_STAGING_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / RUN_LOCAL_STAGING_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _stable_first_release_scope_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    source = (
        STABLE_FIRST_RELEASE_SCOPE_ARTIFACT
        if artifact_dir is None
        else Path(artifact_dir) / STABLE_FIRST_RELEASE_SCOPE_ARTIFACT.name
    )
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _owner_release_direction_summary(*, artifact_dir: str | Path | None) -> dict[str, Any]:
    directory = ARTIFACT_DIR if artifact_dir is None else Path(artifact_dir)
    record_source = directory / OWNER_RELEASE_DIRECTION_RECORD_ARTIFACT.name
    if record_source.exists():
        summary = _as_dict(_read_json(record_source).get("summary"))
        if summary:
            return summary
    source = directory / OWNER_RELEASE_DIRECTION_ARTIFACT.name
    if not source.exists():
        return {}
    return _as_dict(_read_json(source).get("summary"))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_dict(payload)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_count(value: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in value:
            return int(value.get(key, 0) or 0)
    return 0


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _unsafe_top_level_enabled(package: Mapping[str, Any]) -> bool:
    for key in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        if package.get(key) is True:
            return True
    return bool(package.get("post_release_request_fields"))


__all__ = ["build_native_update_owner_release_handoff_summary"]
