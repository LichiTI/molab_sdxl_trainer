"""Dry-run request patches for T-LoRA A/B benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TLoRAABRequestPatch:
    case_id: str
    arm: str
    family: str
    request_patch: Mapping[str, Any]
    expected_result_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm": self.arm,
            "family": self.family,
            "request_patch": dict(self.request_patch),
            "expected_result_path": self.expected_result_path,
        }


def build_tlora_ab_request_patch_plan(
    manifest: Mapping[str, Any],
    *,
    base_request: Mapping[str, Any] | None = None,
    include_dry_run_flag: bool = True,
) -> dict[str, Any]:
    base = dict(base_request or {})
    cases = [dict(case) for case in manifest.get("cases", []) if isinstance(case, Mapping)]
    blockers: list[str] = []
    if not bool(manifest.get("runner_ready", manifest.get("ok", False))):
        blockers.append("manifest_not_runner_ready")
    if not cases:
        blockers.append("manifest_cases_missing")
    rows: list[TLoRAABRequestPatch] = []
    for case in cases:
        case_blockers = _case_blockers(case)
        blockers.extend(case_blockers)
        if case_blockers:
            continue
        rows.append(_patch_row(case, "baseline", base, include_dry_run_flag))
        rows.append(_patch_row(case, "tlora", base, include_dry_run_flag))
    request_fields_emitted = bool(rows) and not blockers
    return {
        "schema_version": 1,
        "plan": "tlora_ab_request_patch_plan_v0",
        "ok": request_fields_emitted,
        "request_fields_emitted": request_fields_emitted,
        "dry_run_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "case_count": len(cases),
        "patch_count": len(rows),
        "patches": [row.as_dict() for row in rows],
        "blocked_reasons": blockers,
        "recommended_next_step": "review dry-run request patches before dispatching real trainer A/B cases",
    }


def build_tlora_ab_request_patch_scorecard(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    blockers = list(payload.get("blocked_reasons") or [])
    if not bool(payload.get("request_fields_emitted")):
        blockers.append("request_fields_not_emitted")
    if not bool(payload.get("dry_run_only")):
        blockers.append("dry_run_boundary_missing")
    if bool(payload.get("training_path_enabled")):
        blockers.append("unsafe_training_path_enabled")
    if bool(payload.get("default_behavior_changed")):
        blockers.append("default_behavior_changed")
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_request_patch_plan_v0",
        "ok": bool(payload.get("request_fields_emitted")) and not blockers,
        "request_fields_emitted": bool(payload.get("request_fields_emitted")),
        "dry_run_only": bool(payload.get("dry_run_only")),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "plan": payload,
        "blocked_reasons": blockers,
        "recommended_next_step": payload.get("recommended_next_step"),
    }


def _patch_row(
    case: Mapping[str, Any],
    arm: str,
    base_request: Mapping[str, Any],
    include_dry_run_flag: bool,
) -> TLoRAABRequestPatch:
    family = str(case.get("family") or "anima").lower()
    patch = dict(base_request)
    patch.update(
        {
            "tlora_ab_case_id": case["case_id"],
            "tlora_ab_arm": arm,
            "model_family": family,
            "max_train_steps": int(case.get("max_train_steps") or 1),
            "resolution": int(case.get("resolution") or 1),
            "seed": int(case.get("seed") or 0),
            "train_image_count": int(case.get("image_count") or 1),
        }
    )
    if include_dry_run_flag:
        patch["dry_run"] = True
    if arm == "baseline":
        patch.update(_adapter_fields(family, "lora", str(case.get("baseline_network_module") or "networks.lora")))
        result_path = str(case.get("baseline_result_path") or "")
    else:
        patch.update(_adapter_fields(family, "tlora", str(case.get("tlora_network_module") or "networks.tlora")))
        patch["tlora_min_rank"] = int(case.get("tlora_min_rank") or 1)
        patch["tlora_rank_schedule"] = str(case.get("tlora_rank_schedule") or "linear")
        patch["tlora_orthogonal_init"] = bool(case.get("tlora_orthogonal_init", False))
        result_path = str(case.get("tlora_result_path") or "")
    return TLoRAABRequestPatch(
        case_id=str(case["case_id"]),
        arm=arm,
        family=family,
        request_patch=patch,
        expected_result_path=result_path,
    )


def _adapter_fields(family: str, adapter_type: str, network_module: str) -> dict[str, Any]:
    fields = {
        "network_module": network_module,
        "adapter_type": adapter_type,
        "lora_type": adapter_type,
    }
    if family == "newbie":
        fields["newbie_adapter_type"] = adapter_type
    return fields


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    case_id = str(case.get("case_id") or "")
    blockers: list[str] = []
    if not case_id:
        blockers.append("case_id_missing")
    if str(case.get("baseline_network_module") or "") == str(case.get("tlora_network_module") or ""):
        blockers.append(f"{case_id}:baseline_and_tlora_modules_match")
    if not str(case.get("tlora_network_module") or "").endswith("tlora"):
        blockers.append(f"{case_id}:tlora_network_module_not_tlora")
    return blockers


__all__ = [
    "TLoRAABRequestPatch",
    "build_tlora_ab_request_patch_plan",
    "build_tlora_ab_request_patch_scorecard",
]
