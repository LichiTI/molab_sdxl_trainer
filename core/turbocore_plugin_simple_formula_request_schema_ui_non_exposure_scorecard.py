"""Request/schema/UI non-exposure audit for selected simple-formula plugin native work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.turbocore_plugin_simple_formula_owner_release_hold_scorecard import (
    build_plugin_simple_formula_owner_release_hold_scorecard,
)


AUDIT_KIND = "plugin_simple_formula_request_schema_ui_non_exposure_v0"
FORBIDDEN_BOUNDARY_TOKENS = (
    "plugin_simple_formula_native",
    "plugin_simple_formula_dispatch",
    "plugin_simple_formula_canary",
    "enable_plugin_simple_formula_native",
    "enable_simple_formula_native",
    "simple_formula_native",
    "simple_formula_turbocore",
)
BOUNDARY_PATHS = (
    "resources/web/routers",
    "backend/routers",
    "backend/resources/web/routers",
    "backend/backend_native.py",
    "backend/lulynx_launcher/app/launcher_api.py",
    "backend/core/contracts",
    "backend/lulynx_launcher/domain/training_models.py",
    "backend/lulynx_launcher/domain/task_models.py",
    "backend/contracts",
    "backend/training_models",
    "backend/task_models",
    "backend/training_configs",
    "plugin/lora-scripts-ui-main",
)
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "build_new_launcher",
    "dist_new_launcher",
    "dist_wpf_launcher",
}
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".vue",
    ".yaml",
    ".yml",
}


def build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard(
    *,
    owner_release_hold_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Audit product boundaries while selected simple-formula native dispatch stays disabled."""

    root = Path(workspace_root).resolve() if workspace_root is not None else Path(__file__).resolve().parents[2]
    hold = _as_dict(owner_release_hold_report or build_plugin_simple_formula_owner_release_hold_scorecard())
    boundary = _boundary_inventory(root)
    findings = _scan_boundaries(root, boundary["present_paths"])
    validations = _validations(hold, boundary, findings)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    hold_summary = _as_dict(hold.get("summary"))
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_simple_formula_request_schema_ui_non_exposure_scorecard_v0",
        "gate": "plugin_simple_formula_request_schema_ui_non_exposure",
        "ok": ready,
        "promotion_ready": False,
        "request_schema_ui_non_exposure_ready": ready,
        "owner_release_hold_ready": hold.get("owner_release_hold_ready") is True,
        "dispatch_review_gate_ready": hold.get("dispatch_review_gate_ready") is True,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "request_adapter_enabled": False,
        "backend_router_registered": False,
        "product_native_dispatch_ready": False,
        "product_native_ready_count": 0,
        "audit_kind": AUDIT_KIND,
        "boundary_inventory": boundary,
        "forbidden_boundary_tokens": list(FORBIDDEN_BOUNDARY_TOKENS),
        "boundary_findings": findings,
        "owner_release_hold_summary": hold_summary,
        "validations": validations,
        "summary": {
            "request_schema_ui_non_exposure_ready": ready,
            "owner_release_hold_ready": hold.get("owner_release_hold_ready") is True,
            "dispatch_review_gate_ready": hold.get("dispatch_review_gate_ready") is True,
            "optimizer_count": int(hold_summary.get("optimizer_count", 0) or 0),
            "present_boundary_path_count": len(boundary["present_paths"]),
            "missing_boundary_path_count": len(boundary["missing_paths"]),
            "scanned_file_count": len(findings["scanned_files"]),
            "forbidden_token_hit_count": len(findings["forbidden_token_hits"]),
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_simple_formula_owner_approval_missing",
                "plugin_simple_formula_release_approval_missing",
                "plugin_simple_formula_request_schema_ui_exposure_not_approved",
                "plugin_simple_formula_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "await explicit owner/release approval before simple-formula request/schema/UI or dispatch wiring"
            if ready
            else "fix simple-formula request/schema/UI non-exposure blockers"
        ),
        "notes": [
            "This audit is report-only and does not register request fields, schema fields, UI controls, or routes.",
            "Boundary scanning checks product-facing paths for selected simple-formula native exposure tokens.",
            "Owner and release approval remain required before product exposure can be implemented.",
        ],
    }
    if write_artifact:
        _write_artifact(root, report)
    return report


def _boundary_inventory(root: Path) -> dict[str, Any]:
    present: list[str] = []
    missing: list[str] = []
    for value in BOUNDARY_PATHS:
        path = root / value
        if path.exists():
            present.append(value)
        else:
            missing.append(value)
    return {
        "schema_version": 1,
        "present_paths": present,
        "missing_paths": missing,
        "boundary_path_count": len(BOUNDARY_PATHS),
    }


def _scan_boundaries(root: Path, paths: Sequence[str]) -> dict[str, Any]:
    scanned: list[str] = []
    hits: list[dict[str, Any]] = []
    for value in paths:
        path = root / value
        files = [path] if path.is_file() else _iter_text_files(path)
        for file_path in files:
            rel = _rel(root, file_path)
            scanned.append(rel)
            lowered = _read_text(file_path).lower()
            for token in FORBIDDEN_BOUNDARY_TOKENS:
                if token.lower() in lowered:
                    hits.append({"path": rel, "token": token})
    return {
        "schema_version": 1,
        "scanned_files": sorted(set(scanned)),
        "forbidden_token_hits": hits,
    }


def _iter_text_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES and not _excluded(child)
    ]


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _validations(
    hold: Mapping[str, Any],
    boundary: Mapping[str, Any],
    findings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "owner_release_hold_ready",
            hold.get("owner_release_hold_ready") is True,
            "plugin_simple_formula_owner_release_hold_missing",
        ),
        _validation(
            "optimizer_set_complete",
            int(_as_dict(hold.get("summary")).get("optimizer_count", 0) or 0) == 18,
            "plugin_simple_formula_request_schema_ui_optimizer_set_incomplete",
        ),
        _validation(
            "owner_release_approval_not_recorded",
            hold.get("owner_approval_recorded") is False and hold.get("release_approval_recorded") is False,
            "plugin_simple_formula_request_schema_ui_unexpected_approval",
        ),
        _validation(
            "hold_kept_product_boundaries_off",
            hold.get("request_fields_emitted") is False
            and hold.get("schema_exposure_allowed") is False
            and hold.get("ui_exposure_allowed") is False
            and hold.get("runtime_dispatch_ready") is False
            and hold.get("native_dispatch_allowed") is False
            and hold.get("training_path_enabled") is False,
            "plugin_simple_formula_request_schema_ui_hold_enabled_boundary",
        ),
        _validation(
            "boundary_inventory_present",
            len(boundary.get("present_paths", [])) > 0,
            "plugin_simple_formula_request_schema_ui_boundary_inventory_missing",
        ),
        _validation(
            "forbidden_tokens_absent",
            not findings.get("forbidden_token_hits"),
            "plugin_simple_formula_request_schema_ui_forbidden_boundary_token_found",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_artifact(root: Path, report: Mapping[str, Any]) -> None:
    temp_dir = root / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_simple_formula_request_schema_ui_non_exposure_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard"]
