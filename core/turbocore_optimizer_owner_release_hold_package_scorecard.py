"""V2 O2 owner/release hold package for TurboCore optimizer families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.turbocore_factored_custom_owner_release_hold_scorecard import (
    build_factored_custom_owner_release_hold_scorecard,
)
from core.turbocore_plugin_adamlike_owner_release_hold_scorecard import (
    build_plugin_adamlike_owner_release_hold_scorecard,
)
from core.turbocore_plugin_factored_memory_owner_release_hold_scorecard import (
    build_plugin_factored_memory_owner_release_hold_scorecard,
)
from core.turbocore_plugin_schedulefree_owner_release_hold_scorecard import (
    build_plugin_schedulefree_owner_release_hold_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_owner_release_hold_scorecard import (
    build_simple_optimizer_schedulefree_owner_release_hold_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_optimizer_owner_release_hold_package_scorecard.json"


FamilyBuilder = Callable[[], dict[str, Any]]

FAMILY_BUILDERS: tuple[tuple[str, str, FamilyBuilder], ...] = (
    ("adam_like", "O2-1", build_plugin_adamlike_owner_release_hold_scorecard),
    ("schedule_free", "O2-2", build_plugin_schedulefree_owner_release_hold_scorecard),
    ("factored_memory", "O2-3", build_plugin_factored_memory_owner_release_hold_scorecard),
    ("factored_custom", "O2-4", build_factored_custom_owner_release_hold_scorecard),
    ("simple_variant_selected_route", "O2-5", build_simple_optimizer_schedulefree_owner_release_hold_scorecard),
)


def build_optimizer_owner_release_hold_package_scorecard(
    *,
    family_reports: Mapping[str, Mapping[str, Any]] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate O2 hold evidence without recording approval or enabling dispatch."""

    reports = family_reports or {}
    rows = [
        _family_row(family_id, roadmap_item, reports.get(family_id) or builder())
        for family_id, roadmap_item, builder in FAMILY_BUILDERS
    ]
    blockers = _dedupe(reason for row in rows for reason in row["blocked_reasons"])
    ready = all(row["owner_release_hold_ready"] for row in rows) and not blockers
    approval_missing_count = sum(1 for row in rows if not row["owner_approval_recorded"] or not row["release_approval_recorded"])
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_owner_release_hold_package_scorecard_v0",
        "gate": "optimizer_owner_release_hold_package",
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design_v2.md",
        "roadmap_section": "O2",
        "ok": ready,
        "owner_release_hold_package_ready": ready,
        "promotion_ready": False,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "summary": {
            "owner_release_hold_package_family_count": len(rows),
            "owner_release_hold_package_ready_family_count": sum(
                1 for row in rows if row["owner_release_hold_ready"]
            ),
            "owner_release_hold_package_manual_review_required_count": sum(
                1 for row in rows if row["manual_review_required"]
            ),
            "owner_release_hold_package_owner_approval_missing_count": approval_missing_count,
            "owner_release_hold_package_release_approval_missing_count": approval_missing_count,
            "owner_release_hold_package_runtime_dispatch_ready_count": sum(
                1 for row in rows if row["runtime_dispatch_ready"]
            ),
            "owner_release_hold_package_native_dispatch_allowed_count": sum(
                1 for row in rows if row["native_dispatch_allowed"]
            ),
            "owner_release_hold_package_training_path_enabled_count": sum(
                1 for row in rows if row["training_path_enabled"]
            ),
            "owner_release_hold_package_default_behavior_changed_count": sum(
                1 for row in rows if row["default_behavior_changed"]
            ),
            "owner_release_hold_package_product_native_ready_count": sum(
                row["product_native_ready_count"] for row in rows
            ),
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "optimizer_owner_release_approval_missing",
                "optimizer_release_approval_missing",
                "optimizer_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "move O2 hold package into owner/release review artifacts while keeping default-off"
            if ready
            else "fix O2 owner/release hold package blockers"
        ),
        "notes": [
            "This package makes O2 signable and reviewable; it does not approve product dispatch.",
            "All families remain off/observe only until owner and release approval are recorded elsewhere.",
            "Request, schema, UI, runtime dispatch, native dispatch, and training path remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _family_row(family_id: str, roadmap_item: str, report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    blockers = _dedupe(report.get("blocked_reasons", []))
    return {
        "schema_version": 1,
        "family_id": family_id,
        "roadmap_item": roadmap_item,
        "source_scorecard": str(report.get("scorecard") or ""),
        "source_gate": str(report.get("gate") or ""),
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "default_behavior_changed": report.get("default_behavior_changed") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "blocked_reasons": blockers,
        "promotion_blockers": _dedupe(report.get("promotion_blockers", [])),
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_optimizer_owner_release_hold_package_scorecard"]
