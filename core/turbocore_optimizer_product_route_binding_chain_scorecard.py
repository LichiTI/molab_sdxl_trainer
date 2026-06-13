"""V2 O4 aggregate scorecard for product route-binding contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_native_update_owner_release_handoff_summary import (
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_owner_release_review_packet import (
    build_native_update_owner_release_review_packet,
)
from core.turbocore_native_update_release_review_archive import (
    build_native_update_release_review_archive,
)
from core.turbocore_optimizer_product_training_route_binding_config_adapter import (
    build_optimizer_product_training_route_binding_config_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_product_route_adapter import (
    build_optimizer_product_training_route_binding_product_route_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_run_local_staging import (
    build_optimizer_product_training_route_binding_run_local_staging,
)
from core.turbocore_optimizer_product_training_route_binding_runtime_applier import (
    apply_optimizer_product_training_route_binding_runtime_patch,
)
from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (
    build_optimizer_product_training_route_binding_training_loop_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_optimizer_product_route_binding_chain_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
STAGE_COUNT = 7


def build_optimizer_product_route_binding_chain_scorecard(
    *,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate O4 route-binding evidence without enabling product dispatch."""

    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT.parent
    preflight = build_optimizer_product_training_route_binding_preflight(
        artifact_dir=directory,
        write_artifact=True,
    )
    training_loop = build_optimizer_product_training_route_binding_training_loop_contract(
        preflight_report=preflight,
        artifact_dir=directory,
        write_artifact=True,
    )
    config_adapter = build_optimizer_product_training_route_binding_config_adapter(
        preflight_report=preflight,
        training_loop_contract=training_loop,
        artifact_dir=directory,
        write_artifact=True,
    )
    product_route = build_optimizer_product_training_route_binding_product_route_adapter(
        config_adapter_report=config_adapter,
        artifact_dir=directory,
        refresh_config_adapter_artifact=False,
        write_artifact=True,
    )
    runtime_applier = apply_optimizer_product_training_route_binding_runtime_patch(
        {},
        config_adapter_report=config_adapter,
        artifact_dir=directory,
        refresh_config_adapter_artifact=False,
        write_artifact=True,
    )
    staging = build_optimizer_product_training_route_binding_run_local_staging(
        config_adapter_report=config_adapter,
        artifact_dir=directory,
        write_artifact=True,
        write_run_local_adapter=False,
        refresh_config_adapter_artifact=False,
    )
    handoff = build_native_update_owner_release_handoff_summary(
        artifact_dir=directory,
        write_artifact=True,
    )
    packet = build_native_update_owner_release_review_packet(
        handoff_summary=handoff,
        artifact_dir=directory,
        write_artifact=True,
    )
    archive = build_native_update_release_review_archive(write_artifact=True)
    exposure = _read_json(directory / "native_update_product_exposure_decision.json")

    rows = [
        _stage_row(
            "O4-1",
            "product route binding preflight",
            preflight,
            ready=preflight.get("ok") is True and _summary_int(preflight, "runtime_launch_coverage_ready_optimizer_count") == 124,
            blocked=_prefixed_blockers(preflight, fallback="product_route_binding_preflight_not_ready"),
        ),
        _stage_row(
            "O4-2",
            "config adapter patch contract",
            config_adapter,
            ready=config_adapter.get("ok") is True and product_route.get("product_training_route_binding_kwargs_wired") is True,
            blocked=_dedupe(
                _prefixed_blockers(config_adapter, fallback="route_binding_config_adapter_not_ready")
                + _prefixed_blockers(product_route, fallback="product_route_kwargs_adapter_not_ready")
            ),
        ),
        _stage_row(
            "O4-3",
            "owner handoff summary",
            handoff,
            ready=handoff.get("technical_evidence_ready") is True and handoff.get("ready_for_owner_release_review") is True,
            blocked=_prefixed_blockers(handoff, fallback="owner_handoff_summary_not_ready"),
        ),
        _stage_row(
            "O4-4",
            "signed review packet",
            packet,
            ready=packet.get("ready_for_owner_signature") is True and packet.get("approval_recorded") is False,
            blocked=_prefixed_blockers(packet, fallback="owner_release_review_packet_not_ready"),
        ),
        _stage_row(
            "O4-5",
            "review archive",
            archive,
            ready=archive.get("evidence_ready") is True and archive.get("ready_for_review") is True,
            blocked=_archive_blockers(archive),
        ),
        _stage_row(
            "O4-6",
            "promotion scorecard default-off guard",
            exposure,
            ready=exposure.get("ok") is True and exposure.get("ready_for_product_exposure_review") is True,
            blocked=_prefixed_blockers(exposure, fallback="product_exposure_decision_evidence_not_ready"),
        ),
        _stage_row(
            "O4-7",
            "training_loop contract",
            training_loop,
            ready=training_loop.get("ok") is True and training_loop.get("post_approval_training_loop_context_ready") is True,
            blocked=_prefixed_blockers(training_loop, fallback="training_loop_route_binding_contract_not_ready"),
        ),
    ]
    ready_count = sum(1 for row in rows if row["stage_ready"])
    unsafe_count = sum(1 for row in rows if _unsafe(row))
    approval_blockers = _dedupe(
        _strings(preflight.get("blocked_reasons"))
        + _strings(config_adapter.get("blocked_reasons"))
        + _strings(runtime_applier.get("blocked_reasons"))
        + _strings(staging.get("blocked_reasons"))
        + _strings(archive.get("blocked_reasons"))
        + _strings(exposure.get("blocked_reasons"))
        + [
            "owner_release_approval_missing",
            "owner_release_direction_not_recorded",
            "product_exposure_decision_not_recorded",
        ]
    )
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_product_route_binding_chain_scorecard_v0",
        "gate": "optimizer_product_route_binding_chain",
        "roadmap": ROADMAP,
        "roadmap_section": "O4",
        "ok": ready_count == STAGE_COUNT and unsafe_count == 0,
        "product_route_binding_chain_ready": False,
        "promotion_ready": False,
        "report_only": True,
        "product_training_route_bound": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "rows": rows,
        "summary": {
            "product_route_binding_chain_stage_count": STAGE_COUNT,
            "product_route_binding_chain_ready_stage_count": ready_count,
            "product_route_binding_chain_open_stage_count": STAGE_COUNT - ready_count,
            "product_route_binding_chain_contract_ready_count": ready_count,
            "product_route_binding_chain_approval_missing_count": len(approval_blockers),
            "product_route_binding_chain_product_training_route_bound_count": 0,
            "product_route_binding_chain_runtime_dispatch_ready_count": 0,
            "product_route_binding_chain_native_dispatch_allowed_count": 0,
            "product_route_binding_chain_training_path_enabled_count": 0,
            "product_route_binding_chain_default_behavior_changed_count": 0,
            "product_route_binding_chain_product_native_ready_count": 0,
            "product_training_route_binding_kwargs_wired_count": _summary_int(
                product_route,
                "product_training_route_binding_kwargs_wired_count",
            ),
            "product_launch_staging_wired_count": _summary_int(staging, "product_launch_staging_wired_count"),
            "run_local_adapter_staged_count": _summary_int(staging, "run_local_adapter_staged_count"),
            "runtime_config_patch_applied_count": _summary_int(runtime_applier, "runtime_config_patch_applied_count"),
            "training_loop_contract_candidate_switch_count": _summary_int(training_loop, "candidate_switch_count"),
            "training_loop_contract_open_training_path_enabled_count": _summary_int(training_loop, "open_training_path_enabled"),
        },
        "blocked_reasons": [],
        "promotion_blockers": approval_blockers,
        "recommended_next_step": "record owner/product approvals before binding product training route; keep defaults closed",
        "notes": [
            "O4 aggregates existing route-binding contracts only.",
            "It does not emit request fields, register UI/schema, bind product training, or launch native dispatch.",
            "Synthetic post-approval TrainingLoop context evidence is not a default product route.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _stage_row(
    roadmap_item: str,
    title: str,
    source: Mapping[str, Any],
    *,
    ready: bool,
    blocked: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "roadmap_item": roadmap_item,
        "title": title,
        "source_gate": str(source.get("gate") or ""),
        "source_scorecard": str(source.get("scorecard") or source.get("preflight") or source.get("adapter") or source.get("contract") or source.get("package") or ""),
        "stage_ready": bool(ready),
        "runtime_dispatch_ready": source.get("runtime_dispatch_ready") is True or source.get("runtime_dispatch_allowed") is True,
        "native_dispatch_allowed": source.get("native_dispatch_allowed") is True,
        "training_path_enabled": source.get("training_path_enabled") is True,
        "default_behavior_changed": source.get("default_behavior_changed") is True,
        "product_training_route_bound": source.get("product_training_route_bound") is True,
        "product_native_ready_count": _summary_int(source, "product_native_ready_count"),
        "blocked_reasons": [] if ready else blocked,
    }


def _archive_blockers(archive: Mapping[str, Any]) -> list[str]:
    if archive.get("evidence_ready") is True and archive.get("ready_for_review") is True:
        return []
    return _prefixed_blockers(archive, fallback="release_review_archive_not_ready")


def _prefixed_blockers(source: Mapping[str, Any], *, fallback: str) -> list[str]:
    reasons = _strings(source.get("blocked_reasons"))
    return reasons or [fallback]


def _unsafe(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(field) is True
        for field in (
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "training_path_enabled",
            "default_behavior_changed",
            "product_training_route_bound",
        )
    ) or int(row.get("product_native_ready_count", 0) or 0) > 0


def _summary_int(source: Mapping[str, Any], key: str) -> int:
    summary = source.get("summary")
    if not isinstance(summary, Mapping):
        return 0
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item or "")]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_optimizer_product_route_binding_chain_scorecard"]
