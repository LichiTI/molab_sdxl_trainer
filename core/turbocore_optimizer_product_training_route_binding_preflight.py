"""Default-off product training-route binding preflight for TurboCore optimizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_preflight.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"

INPUT_ARTIFACTS = {
    "native_readiness_gap": ARTIFACT_DIR / "turbocore_optimizer_native_readiness_gap_scorecard.json",
    "owner_release_review_record": ARTIFACT_DIR / "native_update_owner_release_review_record.json",
    "product_exposure_decision": ARTIFACT_DIR / "native_update_product_exposure_decision.json",
    "release_review_package": ARTIFACT_DIR / "native_update_release_review_package.json",
}

UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "product_exposure_enabled",
    "product_exposure_approved",
    "release_gate_open",
    "runtime_dispatch_allowed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "training_path_enabled",
    "training_dispatch",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "request_submission_allowed",
    "request_submitted",
    "request_payload_materialized",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_product_exposure_request_fields",
    "post_release_request_fields",
    "post_training_route_request_fields",
    "post_training_launch_request_fields",
    "product_exposure_request",
    "training_launch_request",
    "request_payload",
    "request_adapter_fields",
    "request_schema_fields",
    "backend_router_registration",
    "ui_route_registration",
)


def build_optimizer_product_training_route_binding_preflight(
    *,
    native_readiness_gap: Mapping[str, Any] | None = None,
    owner_release_review_record: Mapping[str, Any] | None = None,
    product_exposure_decision: Mapping[str, Any] | None = None,
    release_review_package: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    gap = _source(native_readiness_gap, directory, "native_readiness_gap")
    owner = _source(owner_release_review_record, directory, "owner_release_review_record")
    exposure = _source(product_exposure_decision, directory, "product_exposure_decision")
    release = _source(release_review_package, directory, "release_review_package")

    gap_summary = _as_dict(gap.get("summary"))
    family_count = int(gap_summary.get("route_family_count", 0) or 0)
    launch_ready = int(gap_summary.get("runtime_launch_coverage_ready_family_count", 0) or 0)
    owner_hold = int(gap_summary.get("owner_release_hold_ready_family_count", 0) or 0)
    non_exposure = int(gap_summary.get("request_schema_ui_non_exposure_ready_family_count", 0) or 0)
    readiness_ready = bool(
        gap
        and gap.get("ok") is True
        and family_count > 0
        and launch_ready == family_count
        and owner_hold == family_count
        and non_exposure == family_count
        and int(gap_summary.get("family_specific_runtime_launch_missing_count", 0) or 0) == 0
    )
    owner_recorded = bool(owner.get("approval_recorded") is True and owner.get("release_review_recorded") is True)
    release_recorded = bool(
        release.get("release_review_recorded") is True or owner.get("release_review_recorded") is True
    )
    product_exposure_recorded = bool(exposure.get("product_exposure_decision_recorded") is True)
    unsafe_claims = _unsafe_claims(gap, "native_readiness_gap")
    unsafe_claims += _unsafe_claims(owner, "owner_release_review_record")
    unsafe_claims += _unsafe_claims(exposure, "product_exposure_decision")
    unsafe_claims += _unsafe_claims(release, "release_review_package")
    defaults_closed = not unsafe_claims
    blockers = _dedupe(
        ([] if readiness_ready else ["native_readiness_gap_not_ready"])
        + ([] if owner_recorded else ["owner_release_approval_missing"])
        + ([] if release_recorded else ["release_review_record_missing"])
        + ([] if product_exposure_recorded else ["product_exposure_decision_not_recorded"])
        + unsafe_claims
    )
    preflight_ready = bool(readiness_ready and owner_recorded and release_recorded and product_exposure_recorded and defaults_closed)
    candidate = _route_binding_candidate(gap_summary) if preflight_ready else {}
    payload = {
        "schema_version": 1,
        "preflight": "turbocore_optimizer_product_training_route_binding_preflight_v0",
        "gate": "optimizer_product_training_route_binding_preflight",
        "ok": defaults_closed and bool(gap),
        "roadmap": ROADMAP,
        "artifact_first": True,
        "product_training_route_binding_preflight_ready": preflight_ready,
        "product_training_route_binding_candidate_ready": preflight_ready,
        "product_training_route_bound": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "post_approval_training_route_binding_candidate": candidate,
        "summary": {
            "route_family_count": family_count,
            "plugin_optimizer_count": int(gap_summary.get("plugin_optimizer_count", 0) or 0),
            "runtime_launch_coverage_ready_family_count": launch_ready,
            "family_specific_runtime_launch_adapter_ready_optimizer_count": int(
                gap_summary.get("family_specific_runtime_launch_adapter_ready_optimizer_count", 0) or 0
            ),
            "owner_release_hold_ready_family_count": owner_hold,
            "request_schema_ui_non_exposure_ready_family_count": non_exposure,
            "owner_release_approval_recorded_count": 1 if owner_recorded else 0,
            "release_review_recorded_count": 1 if release_recorded else 0,
            "product_exposure_decision_recorded_count": 1 if product_exposure_recorded else 0,
            "product_training_route_binding_ready_count": family_count if preflight_ready else 0,
            "post_approval_training_route_binding_candidate_count": 1 if candidate else 0,
            "training_path_enabled_count": 0,
            "runtime_dispatch_ready_family_count": 0,
            "native_dispatch_allowed_family_count": 0,
        },
        "blocked_reasons": blockers,
        "promotion_blockers": _dedupe(blockers + ["training_path_dispatch_not_enabled"]),
        "input_artifacts": {name: str((directory / path.name) if artifact_dir else path) for name, path in INPUT_ARTIFACTS.items()},
        "recommended_next_step": (
            "route binding candidate is ready for a separate implementation step; keep this preflight report-only"
            if preflight_ready
            else "collect signed owner release/product exposure decisions before product training-route binding"
        ),
        "notes": [
            "This preflight does not create request fields, register routes, expose UI, submit jobs, or launch training.",
            "A signed owner record can make the binding candidate eligible, but a separate code change is required to bind training.",
        ],
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _route_binding_candidate(gap_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate": "post_approval_optimizer_training_route_binding_v0",
        "report_only": True,
        "requires_separate_code_change": True,
        "existing_training_loop_switches": {
            "turbocore_native_update_dispatch_enabled": True,
            "turbocore_native_update_training_path_enabled": True,
            "turbocore_native_update_require_native_cuda": True,
        },
        "runtime_context_patch": {
            "training_path_enabled": True,
            "native_update_training_dispatch_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
        },
        "request_ui_schema_contract": {
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "backend_router_registered": False,
            "post_training_route_request_fields": {},
        },
        "optimizer_scope": {
            "route_family_count": int(gap_summary.get("route_family_count", 0) or 0),
            "plugin_optimizer_count": int(gap_summary.get("plugin_optimizer_count", 0) or 0),
            "runtime_launch_coverage_ready_family_count": int(
                gap_summary.get("runtime_launch_coverage_ready_family_count", 0) or 0
            ),
        },
    }


def _source(value: Mapping[str, Any] | None, directory: Path, name: str) -> dict[str, Any]:
    if value is not None:
        return _as_dict(value)
    return _read_json(directory / INPUT_ARTIFACTS[name].name)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_dict(payload)


def _read_json_if_supplied(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _read_json(Path(path))


def _unsafe_claims(report: Mapping[str, Any], label: str) -> list[str]:
    if not report:
        return [f"{label}_missing"]
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if report.get(field) is True:
            blocked.append(f"{label}_unsafe:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        value = report.get(field)
        if value not in (None, {}, [], "", ()):
            blocked.append(f"{label}_unsafe_non_empty:{field}")
    return blocked


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-readiness-gap", default="")
    parser.add_argument("--owner-release-review-record", default="")
    parser.add_argument("--product-exposure-decision", default="")
    parser.add_argument("--release-review-package", default="")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--no-artifact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap=_read_json_if_supplied(args.native_readiness_gap),
        owner_release_review_record=_read_json_if_supplied(args.owner_release_review_record),
        product_exposure_decision=_read_json_if_supplied(args.product_exposure_decision),
        release_review_package=_read_json_if_supplied(args.release_review_package),
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_product_training_route_binding_preflight"]


if __name__ == "__main__":
    raise SystemExit(main())
