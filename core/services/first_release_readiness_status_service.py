from __future__ import annotations

from pathlib import Path
from typing import Any

from ..lulynx_trainer.lulynx_first_release_readiness import (
    build_default_first_release_readiness,
)
from .first_release_evidence_refresh_service import (
    refresh_first_release_evidence_status,
)


def build_first_release_readiness_status(project_root: Path) -> dict[str, Any]:
    repo_root = Path(project_root).resolve()
    report = build_default_first_release_readiness(repo_root)
    summary = dict(report.get("summary") or {})
    release_blockers = list(report.get("release_blockers") or [])
    deferred_research_blockers = list(report.get("deferred_research_blockers") or [])
    return {
        "schema_version": 1,
        "id": "lulynx_first_release_readiness",
        "display_name_zh": "首发发布状态",
        "display_name_en": "First Release Readiness",
        "readiness": "ready" if bool(report.get("release_ready")) else "blocked",
        "release_scope": str(report.get("release_scope") or "first_release_stable_baseline"),
        "stable_baseline_ready": bool(report.get("release_ready")),
        "release_blockers": release_blockers,
        "deferred_research_blockers": deferred_research_blockers,
        "deferred_research_blocker_groups": dict(report.get("deferred_research_blocker_groups") or {}),
        "release_validation_todo": list(report.get("release_validation_todo") or []),
        "summary": summary,
        "core_release_smoke": dict(report.get("core_release_smoke") or {}),
        "batch1_handler_parity_smoke": dict(report.get("batch1_handler_parity_smoke") or {}),
        "experimental_claim_gate_evidence": dict(report.get("experimental_claim_gate_evidence") or {}),
        "research_artifact_gates": dict(report.get("research_artifact_gates") or {}),
        "gated_experimental_features": list(report.get("gated_experimental_features") or []),
        "sources": dict(report.get("sources") or {}),
        "note_zh": (
            "该状态只评估第一版 stable baseline 是否可发布；"
            "batch2/4/8、lossless、compile/attention 泛化收益、98/99% GPU 利用率等后续路线仍保持受保护关闭。"
        ),
        "note_en": (
            "This status only evaluates whether the first stable baseline can ship. "
            "Later paths such as batch2/4/8, lossless, generalized compile/attention gains, "
            "and 98/99% GPU utilization claims remain gated off."
        ),
        "raw_report": report,
    }


def refresh_first_release_readiness_status(project_root: Path) -> dict[str, Any]:
    refresh_report = refresh_first_release_evidence_status(project_root)
    payload = build_first_release_readiness_status(project_root)
    return {
        **payload,
        "refresh_report": refresh_report,
    }


__all__ = [
    "build_first_release_readiness_status",
    "refresh_first_release_readiness_status",
]
