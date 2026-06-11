"""Artifact-first native readiness gap summary for TurboCore optimizers.

This report does not execute CUDA and does not enable dispatch.  It joins the
selected-family batch, native inventory, and family contract artifacts so the
next implementation work can see the remaining native-readiness gap by family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_native_readiness_gap_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"

INPUT_ARTIFACTS = {
    "coverage": ARTIFACT_DIR / "turbocore_optimizer_coverage_scorecard.json",
    "plugin_family_batch": ARTIFACT_DIR / "turbocore_plugin_optimizer_family_batch_scorecard.json",
    "native_inventory": ARTIFACT_DIR / "turbocore_optimizer_native_kernel_inventory_scorecard.json",
    "family_contract": ARTIFACT_DIR / "turbocore_optimizer_family_kernel_contract_scorecard.json",
    "selected_family_owner_release_hold": ARTIFACT_DIR
    / "turbocore_plugin_selected_family_owner_release_hold_scorecard.json",
    "selected_family_request_schema_ui_non_exposure": ARTIFACT_DIR
    / "turbocore_plugin_selected_family_request_schema_ui_non_exposure_scorecard.json",
}
RUNTIME_REHEARSAL_ARTIFACTS = {
    "adam_like_formula": ARTIFACT_DIR / "turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard.json",
    "adaptive_lr_state_machine": ARTIFACT_DIR / "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard.json",
    "schedule_free_state_machine": ARTIFACT_DIR / "turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard.json",
    "simple_formula": ARTIFACT_DIR / "turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard.json",
}
PRECONDITION_REHEARSAL_ARTIFACTS = {
    "closure_or_second_order": ARTIFACT_DIR
    / "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard.json",
    "custom_formula": ARTIFACT_DIR / "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard.json",
    "factored_memory_layout": ARTIFACT_DIR
    / "turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard.json",
    "fused_backward": ARTIFACT_DIR / "turbocore_plugin_fused_backward_runtime_precondition_rehearsal_scorecard.json",
    "model_or_shape_aware": ARTIFACT_DIR
    / "turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard.json",
    "state_adapter_special": ARTIFACT_DIR
    / "turbocore_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard.json",
}


def build_optimizer_native_readiness_gap_scorecard(
    *,
    coverage_report: Mapping[str, Any] | None = None,
    plugin_family_batch_report: Mapping[str, Any] | None = None,
    native_inventory_report: Mapping[str, Any] | None = None,
    family_contract_report: Mapping[str, Any] | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    coverage = _as_dict(coverage_report) or _read_json(INPUT_ARTIFACTS["coverage"])
    plugin_batch = _as_dict(plugin_family_batch_report) or _read_json(INPUT_ARTIFACTS["plugin_family_batch"])
    inventory = _as_dict(native_inventory_report) or _read_json(INPUT_ARTIFACTS["native_inventory"])
    family_contract = _as_dict(family_contract_report) or _read_json(INPUT_ARTIFACTS["family_contract"])
    owner_hold = _read_optional_json(INPUT_ARTIFACTS["selected_family_owner_release_hold"])
    non_exposure = _read_optional_json(INPUT_ARTIFACTS["selected_family_request_schema_ui_non_exposure"])

    route_counts = _route_family_counts(coverage, plugin_batch)
    batch_rows = _by_family(plugin_batch.get("family_rows"))
    inventory_rows = _by_family(inventory.get("family_rows"))
    contract_rows = _by_family(family_contract.get("contracts"))
    runtime_rehearsals = _read_family_artifacts(RUNTIME_REHEARSAL_ARTIFACTS)
    precondition_rehearsals = _read_family_artifacts(PRECONDITION_REHEARSAL_ARTIFACTS)

    rows = [
        _family_row(
            family,
            optimizer_count,
            batch_rows.get(family, {}),
            inventory_rows.get(family, {}),
            contract_rows.get(family, {}),
            runtime_rehearsals.get(family, {}),
            precondition_rehearsals.get(family, {}),
            owner_hold,
            non_exposure,
        )
        for family, optimizer_count in sorted(route_counts.items())
    ]
    blocked_reasons = _summary_blockers(rows, route_counts)
    summary = {
        "route_family_count": len(rows),
        "plugin_optimizer_count": sum(row["optimizer_count"] for row in rows),
        "selected_optimizer_gate_ready_family_count": sum(
            1 for row in rows if row["selected_optimizer_gate_ready"]
        ),
        "kernel_source_ready_optimizer_count": sum(row["kernel_source_present_count"] for row in rows),
        "rust_probe_ready_optimizer_count": sum(row["rust_probe_present_count"] for row in rows),
        "family_contract_ready_count": sum(1 for row in rows if row["family_contract_ready"]),
        "family_evidence_ready_count": sum(1 for row in rows if row["family_evidence_ready"]),
        "runtime_rehearsal_ready_family_count": sum(1 for row in rows if row["runtime_rehearsal_ready"]),
        "runtime_precondition_ready_family_count": sum(
            1 for row in rows if row["runtime_precondition_rehearsal_ready"]
        ),
        "family_specific_runtime_launch_adapter_ready_family_count": sum(
            1 for row in rows if row["family_specific_runtime_launch_adapter_ready"]
        ),
        "family_specific_runtime_launch_adapter_ready_optimizer_count": sum(
            int(row["family_specific_runtime_launch_adapter_ready_count"]) for row in rows
        ),
        "runtime_launch_coverage_ready_family_count": sum(
            1
            for row in rows
            if row["runtime_rehearsal_ready"] or row["family_specific_runtime_launch_adapter_ready"]
        ),
        "owner_release_hold_ready_family_count": sum(1 for row in rows if row["owner_release_hold_ready"]),
        "request_schema_ui_non_exposure_ready_family_count": sum(
            1 for row in rows if row["request_schema_ui_non_exposure_ready"]
        ),
        "representative_runtime_rehearsal_ready_count": sum(
            int(row["representative_runtime_rehearsal_ready_count"]) for row in rows
        ),
        "representative_runtime_ready_family_count": sum(
            1 for row in rows if int(row["representative_runtime_rehearsal_ready_count"]) > 0
        ),
        "runtime_launch_absent_family_count": sum(
            1
            for row in rows
            if not row["runtime_rehearsal_ready"] and int(row["representative_runtime_rehearsal_ready_count"]) == 0
        ),
        "runtime_dispatch_ready_family_count": sum(1 for row in rows if row["runtime_dispatch_ready"]),
        "native_dispatch_allowed_family_count": sum(1 for row in rows if row["native_dispatch_allowed"]),
        "training_path_enabled_family_count": sum(1 for row in rows if row["training_path_enabled"]),
        "product_native_ready_family_count": sum(1 for row in rows if row["product_native_ready"]),
        "family_specific_runtime_launch_missing_count": sum(
            1 for row in rows if "family_specific_runtime_launch_missing" in row["blocked_reasons"]
        ),
        "product_training_route_missing_count": sum(
            1 for row in rows if "product_training_route_not_bound" in row["blocked_reasons"]
        ),
        "owner_release_approval_missing_count": sum(
            1 for row in rows if "owner_release_approval_missing" in row["blocked_reasons"]
        ),
    }
    family_specific_missing = summary["family_specific_runtime_launch_missing_count"] > 0
    owner_hold_missing = summary["owner_release_hold_ready_family_count"] != summary["route_family_count"]
    non_exposure_missing = (
        summary["request_schema_ui_non_exposure_ready_family_count"] != summary["route_family_count"]
    )
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_native_readiness_gap_scorecard_v0",
        "gate": "optimizer_native_readiness_gap",
        "ok": not blocked_reasons,
        "promotion_ready": False,
        "roadmap": ROADMAP,
        "artifact_first": True,
        "cuda_executed": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "summary": summary,
        "route_family_counts": route_counts,
        "rows": rows,
        "runtime_gap_plan": _runtime_gap_plan(rows),
        "blocked_reasons": blocked_reasons,
        "promotion_blockers": _dedupe(
            (
                ["family_specific_runtime_launch_missing"]
                if family_specific_missing
                else []
            )
            + (["owner_release_hold_missing"] if owner_hold_missing else [])
            + (["request_schema_ui_non_exposure_missing"] if non_exposure_missing else [])
            + [
                "owner_release_approval_missing",
                "training_path_dispatch_not_enabled",
            ]
        ),
        "recommended_next_step": (
            "implement family-specific runtime launch/adapter support from the rows with "
            "family_evidence_ready=true while keeping product dispatch default-off"
            if family_specific_missing
            else "prepare selected-family owner/release hold and request/schema/UI non-exposure evidence"
            if owner_hold_missing or non_exposure_missing
            else "await explicit owner/release approval before product training-route binding or dispatch wiring"
        ),
        "input_artifacts": {name: str(path) for name, path in INPUT_ARTIFACTS.items()},
        "runtime_rehearsal_artifacts": {
            family: str(path) for family, path in {**RUNTIME_REHEARSAL_ARTIFACTS, **PRECONDITION_REHEARSAL_ARTIFACTS}.items()
        },
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _family_row(
    family: str,
    optimizer_count: int,
    batch: Mapping[str, Any],
    inventory: Mapping[str, Any],
    contract: Mapping[str, Any],
    runtime_rehearsal: Mapping[str, Any],
    precondition_rehearsal: Mapping[str, Any],
    owner_hold: Mapping[str, Any],
    non_exposure: Mapping[str, Any],
) -> dict[str, Any]:
    source_count = int(inventory.get("kernel_source_present_count", 0) or 0)
    probe_count = int(inventory.get("rust_probe_present_count", 0) or 0)
    gate_ready = batch.get("selected_optimizer_gate_ready") is True
    contract_ready = contract.get("native_kernel_present") is True
    evidence_ready = (
        gate_ready
        and source_count == optimizer_count
        and probe_count == optimizer_count
        and contract_ready
    )
    runtime_rehearsal_ready = _runtime_rehearsal_ready(runtime_rehearsal, optimizer_count)
    precondition_rehearsal_ready = _precondition_rehearsal_ready(precondition_rehearsal, optimizer_count)
    runtime_adapter_ready = _family_runtime_launch_adapter_ready(precondition_rehearsal, optimizer_count)
    runtime_adapter_ready_count = _family_runtime_launch_adapter_ready_count(precondition_rehearsal)
    owner_hold_ready = _owner_release_hold_ready(owner_hold, family)
    non_exposure_ready = _request_schema_ui_non_exposure_ready(non_exposure, family)
    runtime_ready = batch.get("runtime_dispatch_ready") is True or contract.get("runtime_dispatch_ready") is True
    native_allowed = batch.get("native_dispatch_allowed") is True or contract.get("native_dispatch_allowed") is True
    training_enabled = batch.get("training_path_enabled") is True or contract.get("training_path_enabled") is True
    product_ready = runtime_ready and native_allowed and training_enabled
    blocked = []
    if not gate_ready:
        blocked.append("selected_optimizer_gate_not_ready")
    if source_count != optimizer_count:
        blocked.append("kernel_source_inventory_incomplete")
    if probe_count != optimizer_count:
        blocked.append("rust_probe_inventory_incomplete")
    if not contract_ready:
        blocked.append("family_contract_not_ready")
    if not runtime_rehearsal_ready and not precondition_rehearsal_ready:
        blocked.append("runtime_rehearsal_artifact_missing")
    if not runtime_rehearsal_ready and not runtime_adapter_ready:
        blocked.append("family_specific_runtime_launch_missing")
    if not owner_hold_ready:
        blocked.append("owner_release_hold_missing")
    if not non_exposure_ready:
        blocked.append("request_schema_ui_non_exposure_missing")
    blocked.append("product_training_route_not_bound")
    if not native_allowed:
        blocked.append("owner_release_approval_missing")
    if not training_enabled:
        blocked.append("training_path_dispatch_not_enabled")
    return {
        "schema_version": 1,
        "native_route_family": family,
        "optimizer_count": optimizer_count,
        "selected_optimizer_gate": str(batch.get("selected_optimizer_gate", "") or ""),
        "selected_optimizer_gate_ready": gate_ready,
        "selected_native_canary_ready_count": int(batch.get("selected_native_canary_ready_count", 0) or 0),
        "kernel_source_present_count": source_count,
        "rust_probe_present_count": probe_count,
        "family_contract_ready": contract_ready,
        "family_evidence_ready": evidence_ready,
        "runtime_rehearsal_ready": runtime_rehearsal_ready,
        "runtime_precondition_rehearsal_ready": precondition_rehearsal_ready,
        "runtime_rehearsal_mode": "dispatch" if runtime_rehearsal_ready else "precondition" if precondition_rehearsal_ready else "missing",
        "family_specific_runtime_launch_adapter_ready": runtime_adapter_ready,
        "family_specific_runtime_launch_adapter_ready_count": runtime_adapter_ready_count,
        "owner_release_hold_ready": owner_hold_ready,
        "request_schema_ui_non_exposure_ready": non_exposure_ready,
        "runtime_rehearsal_case_count": _case_count(runtime_rehearsal or precondition_rehearsal),
        "runtime_rehearsal_readiness_count": _readiness_count(runtime_rehearsal or precondition_rehearsal),
        "representative_runtime_rehearsal_ready_count": _representative_runtime_count(
            runtime_rehearsal or precondition_rehearsal
        ),
        "runtime_rehearsal_native_step_count": int(
            _as_dict((runtime_rehearsal or precondition_rehearsal).get("summary")).get("native_step_count", 0) or 0
        ),
        "runtime_gap_stage": _runtime_gap_stage(
            runtime_rehearsal_ready,
            precondition_rehearsal_ready,
            runtime_adapter_ready,
            runtime_rehearsal or precondition_rehearsal,
        ),
        "next_runtime_adapter_step": _next_runtime_adapter_step(
            family,
            runtime_rehearsal_ready,
            precondition_rehearsal_ready,
            runtime_adapter_ready,
            runtime_rehearsal or precondition_rehearsal,
        ),
        "runtime_dispatch_ready": runtime_ready,
        "native_dispatch_allowed": native_allowed,
        "training_path_enabled": training_enabled,
        "product_native_ready": product_ready,
        "next_gate": str(batch.get("next_gate", "") or "implement family-specific runtime launch adapter"),
        "blocked_reasons": blocked,
    }


def _runtime_gap_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["runtime_rehearsal_ready"] or row["family_specific_runtime_launch_adapter_ready"]:
            continue
        out.append(
            {
                "native_route_family": row["native_route_family"],
                "optimizer_count": row["optimizer_count"],
                "runtime_gap_stage": row["runtime_gap_stage"],
                "representative_runtime_rehearsal_ready_count": row[
                    "representative_runtime_rehearsal_ready_count"
                ],
                "next_runtime_adapter_step": row["next_runtime_adapter_step"],
                "product_native_ready": False,
            }
        )
    return out


def _summary_blockers(rows: list[dict[str, Any]], route_counts: Mapping[str, int]) -> list[str]:
    blockers: list[str] = []
    if len(route_counts) != 10:
        blockers.append("route_family_count_mismatch")
    if sum(route_counts.values()) != 124:
        blockers.append("plugin_optimizer_count_mismatch")
    if any(not row["family_evidence_ready"] for row in rows):
        blockers.append("family_evidence_not_ready")
    if any(not row["runtime_rehearsal_ready"] and not row["runtime_precondition_rehearsal_ready"] for row in rows):
        blockers.append("runtime_rehearsal_artifact_missing")
    if any(row["product_native_ready"] for row in rows):
        blockers.append("unexpected_product_native_ready")
    return blockers


def _route_family_counts(*reports: Mapping[str, Any]) -> dict[str, int]:
    for report in reports:
        candidates = (
            _as_dict(_as_dict(report.get("summary")).get("route_family_counts")),
            _as_dict(report.get("route_family_counts")),
            _as_dict(_as_dict(report.get("plugin_optimizer_summary")).get("plugin_selector_route_family_counts")),
            _as_dict(_as_dict(report.get("plugin_optimizer_family_batch")).get("route_family_counts")),
        )
        for candidate in candidates:
            out = _int_dict(candidate)
            if out:
                return out
    return {}


def _by_family(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if isinstance(row, Mapping):
            family = str(row.get("native_route_family", "") or "")
            if family:
                out[family] = dict(row)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} payload is {type(payload).__name__}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_family_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    out = {}
    for family, path in paths.items():
        if path.exists():
            out[family] = _read_json(path)
    return out


def _runtime_rehearsal_ready(report: Mapping[str, Any], optimizer_count: int) -> bool:
    summary = _as_dict(report.get("summary"))
    return (
        report.get("ok") is True
        and report.get("runtime_dispatch_rehearsal_ready") is True
        and int(summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0) == optimizer_count
        and int(summary.get("native_step_count", 0) or 0) > 0
    )


def _precondition_rehearsal_ready(report: Mapping[str, Any], optimizer_count: int) -> bool:
    summary = _as_dict(report.get("summary"))
    return (
        report.get("ok") is True
        and report.get("runtime_precondition_rehearsal_ready") is True
        and int(summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0) == optimizer_count
    )


def _family_runtime_launch_adapter_ready(report: Mapping[str, Any], optimizer_count: int) -> bool:
    summary = _as_dict(report.get("summary"))
    return (
        report.get("ok") is True
        and report.get("family_specific_runtime_launch_adapter_ready") is True
        and int(summary.get("family_specific_runtime_launch_adapter_ready_count", 0) or 0) == optimizer_count
        and int(summary.get("native_step_count", 0) or 0) > 0
        and int(summary.get("native_kernel_launch_count", 0) or 0) > 0
    )


def _family_runtime_launch_adapter_ready_count(report: Mapping[str, Any]) -> int:
    summary = _as_dict(report.get("summary"))
    return int(summary.get("family_specific_runtime_launch_adapter_ready_count", 0) or 0)


def _owner_release_hold_ready(report: Mapping[str, Any], family: str) -> bool:
    if report.get("ok") is not True or report.get("owner_release_hold_ready") is not True:
        return False
    manifest = _as_dict(report.get("hold_manifest"))
    rows = manifest.get("family_rows")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if (
            isinstance(row, Mapping)
            and str(row.get("native_route_family", "") or "") == family
            and row.get("selected_optimizer_gate_ready") is True
            and row.get("runtime_dispatch_ready") is False
            and row.get("native_dispatch_allowed") is False
            and row.get("training_path_enabled") is False
        ):
            return True
    return False


def _request_schema_ui_non_exposure_ready(report: Mapping[str, Any], family: str) -> bool:
    del family
    summary = _as_dict(report.get("summary"))
    return (
        report.get("ok") is True
        and report.get("request_schema_ui_non_exposure_ready") is True
        and int(summary.get("family_count", 0) or 0) == 10
        and int(summary.get("plugin_optimizer_count", 0) or 0) == 124
        and int(summary.get("forbidden_token_hit_count", 0) or 0) == 0
        and report.get("request_fields_emitted") is False
        and report.get("schema_exposure_allowed") is False
        and report.get("ui_exposure_allowed") is False
        and report.get("runtime_dispatch_ready") is False
        and report.get("native_dispatch_allowed") is False
        and report.get("training_path_enabled") is False
    )


def _case_count(report: Mapping[str, Any]) -> int:
    summary = _as_dict(report.get("summary"))
    return int(summary.get("case_count", summary.get("selected_optimizer_count", 0)) or 0)


def _readiness_count(report: Mapping[str, Any]) -> int:
    summary = _as_dict(report.get("summary"))
    return int(
        summary.get(
            "runtime_dispatch_rehearsal_ready_count",
            summary.get("runtime_precondition_rehearsal_ready_count", 0),
        )
        or 0
    )


def _representative_runtime_count(report: Mapping[str, Any]) -> int:
    summary = _as_dict(report.get("summary"))
    return int(summary.get("representative_runtime_dispatch_rehearsal_ready_count", 0) or 0)


def _runtime_gap_stage(
    runtime_rehearsal_ready: bool,
    precondition_rehearsal_ready: bool,
    runtime_adapter_ready: bool,
    report: Mapping[str, Any],
) -> str:
    if runtime_rehearsal_ready:
        return "family_runtime_dispatch_rehearsal_ready"
    if runtime_adapter_ready:
        return "family_runtime_launch_adapter_ready"
    if _representative_runtime_count(report) > 0:
        return "representative_runtime_rehearsal_ready"
    if precondition_rehearsal_ready:
        return "precondition_only_runtime_launch_missing"
    return "runtime_rehearsal_artifact_missing"


def _next_runtime_adapter_step(
    family: str,
    runtime_rehearsal_ready: bool,
    precondition_rehearsal_ready: bool,
    runtime_adapter_ready: bool,
    report: Mapping[str, Any],
) -> str:
    if runtime_rehearsal_ready:
        return "hold product dispatch default-off until owner/release approval"
    if runtime_adapter_ready:
        return f"bind {family} adapter coverage into explicit owner/release hold while product dispatch stays default-off"
    if _representative_runtime_count(report) > 0:
        return f"expand {family} representative runtime canary into family-specific runtime launch coverage"
    if precondition_rehearsal_ready:
        return f"add first {family} representative native TrainingLoop canary before full-family runtime launch"
    return f"refresh {family} runtime/precondition artifact before adding native adapter"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_dict(value: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, item in value.items():
        try:
            out[str(key)] = int(item or 0)
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_optimizer_native_readiness_gap_scorecard"]
