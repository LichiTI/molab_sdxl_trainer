"""Apply bubble advisor patches to a next-run request safely."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ALLOWED_BUBBLE_PATCH_PATHS = {
    "adaptive_step_logging_enabled",
    "anima_block_prefetch",
    "anima_block_prefetch_depth",
    "cached_dataloader_auto_policy",
    "cached_dataloader_pin_memory",
    "cached_dataloader_prefetch_factor",
    "cached_dataloader_workers",
    "data_transfer_non_blocking",
    "data_transfer_profile_mode",
    "eval_every_n_steps",
    "gradient_accumulation_steps",
    "layer_monitor_interval",
    "newbie_block_prefetch",
    "newbie_block_prefetch_depth",
    "optimizer_backend",
    "pin_memory",
    "save_every_n_epochs",
    "save_every_n_steps",
    "step_phase_profile_enabled",
    "tensorboard_flush_interval_steps",
    "train_batch_size",
}

BUBBLE_ADVISOR_LEDGER_FIELD = "bubble_advisor_action_ledger"
BUBBLE_ADVISOR_HISTORY_FIELD = "bubble_advisor_action_history"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _action_plan_from(report_or_plan: Mapping[str, Any]) -> Mapping[str, Any]:
    report = _mapping(report_or_plan)
    if report.get("plan") == "bubble_runtime_action_plan_v0":
        return report
    return _mapping(report.get("action_plan"))


def _stable_action_id(plan: Mapping[str, Any], mutations: list[Mapping[str, Any]]) -> str:
    payload = {
        "phase": plan.get("phase"),
        "domain": plan.get("domain"),
        "action_kind": plan.get("action_kind"),
        "mutations": [
            {
                "path": item.get("path"),
                "recommended": item.get("recommended"),
            }
            for item in mutations
        ],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"bubble-advisor-{digest}"


def _source_report_summary(report_or_plan: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    report = _mapping(report_or_plan)
    if not report or report.get("plan") == "bubble_runtime_action_plan_v0":
        return {
            "source": "action_plan",
            "phase": str(plan.get("phase") or ""),
            "domain": str(plan.get("domain") or ""),
            "action_kind": str(plan.get("action_kind") or ""),
        }
    diagnosis = _mapping(report.get("diagnosis"))
    evidence = _mapping(diagnosis.get("evidence"))
    snapshot = _mapping(report.get("snapshot"))
    step_phase = _mapping(snapshot.get("step_phase"))
    gpu = _mapping(snapshot.get("gpu"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    return {
        "source": str(report.get("controller") or report.get("report") or "bubble_controller_report"),
        "phase": str(report.get("phase") or plan.get("phase") or ""),
        "status": str(report.get("status") or ""),
        "diagnosis_kind": str(diagnosis.get("kind") or ""),
        "domain": str(plan.get("domain") or ""),
        "action_kind": str(plan.get("action_kind") or ""),
        "evidence": dict(evidence),
        "snapshot": {
            "step_phase": {
                "dominant_bottleneck": step_phase.get("dominant_bottleneck"),
                "bubble_ratio_estimate": step_phase.get("bubble_ratio_estimate"),
                "data_wait_share": step_phase.get("data_wait_share"),
                "h2d_transfer_share": step_phase.get("h2d_transfer_share"),
                "optimizer_share": step_phase.get("optimizer_share"),
                "host_gap_share": step_phase.get("host_gap_share"),
                "mean_step_ms": step_phase.get("mean_step_ms"),
            },
            "gpu": {
                "active_gpu_util_pct_mean": gpu.get("active_gpu_util_pct_mean"),
                "active_gpu_saturated_sample_ratio": gpu.get("active_gpu_saturated_sample_ratio"),
                "memory_used_mb_max": gpu.get("memory_used_mb_max"),
                "memory_total_mb": gpu.get("memory_total_mb"),
            },
            "runtime": {
                "train_batch_size": runtime.get("train_batch_size"),
                "gradient_accumulation_steps": runtime.get("gradient_accumulation_steps"),
                "workers": runtime.get("workers"),
                "prefetch_factor": runtime.get("prefetch_factor"),
                "pin_memory": runtime.get("pin_memory"),
                "optimizer_backend": runtime.get("optimizer_backend"),
            },
            "safety": {
                "memory_ratio": safety.get("memory_ratio"),
                "vram_safe": safety.get("vram_safe"),
            },
        },
    }


def _history_from_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_history = request.get(BUBBLE_ADVISOR_HISTORY_FIELD)
    if not isinstance(raw_history, list):
        return []
    return [dict(item) for item in raw_history if isinstance(item, Mapping)]


def _append_ledger_history(
    request: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    max_history: int,
) -> list[dict[str, Any]]:
    history = _history_from_request(request)
    action_id = str(ledger.get("action_id") or "")
    if action_id and any(str(item.get("action_id") or "") == action_id for item in history):
        return history[-max(max_history, 1) :]
    history.append(dict(ledger))
    return history[-max(max_history, 1) :]


def _empty_result(
    *,
    base_request: Mapping[str, Any],
    plan: Mapping[str, Any],
    status: str,
    reason: str,
    blocked_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "patch": "bubble_advisor_patch_apply_v0",
        "status": status,
        "reason": reason,
        "can_apply_to_next_request": False,
        "can_apply_during_current_run": False,
        "patched_request": dict(base_request),
        "applied_overlay": {},
        "skipped": [],
        "blocked_reasons": list(blocked_reasons or []),
        "action_ledger": {
            "schema_version": 1,
            "ledger": "bubble_advisor_action_ledger_v0",
            "status": status,
            "phase": str(plan.get("phase") or ""),
            "domain": str(plan.get("domain") or ""),
            "action_kind": str(plan.get("action_kind") or ""),
            "reason": reason,
            "applied_overlay": {},
            "rollback": {},
        },
    }


def apply_bubble_advisor_patch_to_request(
    base_request: Mapping[str, Any],
    report_or_plan: Mapping[str, Any],
    *,
    action_id: str | None = None,
    allow_current_mismatch: bool = False,
) -> dict[str, Any]:
    """Return a patched copy of *base_request* plus an auditable ledger entry.

    This helper is intentionally next-run only.  It never mutates the current
    trainer config and rejects unknown patch paths so controller reports cannot
    become arbitrary request writes.
    """

    request = dict(base_request or {})
    plan = _action_plan_from(report_or_plan)
    if not plan:
        return _empty_result(
            base_request=request,
            plan={},
            status="blocked",
            reason="bubble action_plan is missing",
            blocked_reasons=["missing_action_plan"],
        )

    if plan.get("status") != "advisor_patch_ready":
        return _empty_result(
            base_request=request,
            plan=plan,
            status="blocked",
            reason=f"action_plan status is {plan.get('status')!r}, not advisor_patch_ready",
            blocked_reasons=["action_plan_not_ready"],
        )
    if plan.get("apply_mode") != "advisor_patch":
        return _empty_result(
            base_request=request,
            plan=plan,
            status="blocked",
            reason=f"action_plan apply_mode is {plan.get('apply_mode')!r}, not advisor_patch",
            blocked_reasons=["apply_mode_not_advisor_patch"],
        )
    if not bool(plan.get("can_apply_to_next_request", False)):
        return _empty_result(
            base_request=request,
            plan=plan,
            status="blocked",
            reason="action_plan cannot apply to the next request",
            blocked_reasons=["next_request_apply_not_allowed"],
        )
    if bool(plan.get("can_apply_during_current_run", False)):
        return _empty_result(
            base_request=request,
            plan=plan,
            status="blocked",
            reason="action_plan unexpectedly claims current-run mutation support",
            blocked_reasons=["current_run_mutation_not_allowed"],
        )

    mutations = [item for item in plan.get("mutations", []) if isinstance(item, Mapping)]
    if not mutations:
        return _empty_result(
            base_request=request,
            plan=plan,
            status="blocked",
            reason="action_plan has no mutations to apply",
            blocked_reasons=["missing_mutations"],
        )

    patched = dict(request)
    applied: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    blocked: list[str] = []
    for mutation in mutations:
        if str(mutation.get("op") or "set") != "set":
            blocked.append("unsupported_mutation_op")
            skipped.append({"path": mutation.get("path"), "reason": "unsupported_mutation_op"})
            continue
        path = str(mutation.get("path") or "")
        if path not in ALLOWED_BUBBLE_PATCH_PATHS:
            blocked.append("unsupported_patch_path")
            skipped.append({"path": path, "reason": "unsupported_patch_path"})
            continue
        recommended = mutation.get("recommended")
        expected_current = mutation.get("current")
        if path in request and request.get(path) not in {expected_current, recommended} and not allow_current_mismatch:
            blocked.append("current_mismatch")
            skipped.append(
                {
                    "path": path,
                    "current": request.get(path),
                    "expected_current": expected_current,
                    "recommended": recommended,
                    "reason": "current_mismatch",
                }
            )
            continue
        if request.get(path) == recommended:
            skipped.append({"path": path, "reason": "already_recommended", "recommended": recommended})
            continue
        patched[path] = recommended
        applied[path] = recommended

    if blocked:
        return {
            "schema_version": 1,
            "patch": "bubble_advisor_patch_apply_v0",
            "status": "blocked",
            "reason": "one or more advisor patch mutations were blocked",
            "can_apply_to_next_request": False,
            "can_apply_during_current_run": False,
            "patched_request": dict(request),
            "applied_overlay": {},
            "skipped": skipped,
            "blocked_reasons": sorted(set(blocked)),
            "action_ledger": {
                "schema_version": 1,
                "ledger": "bubble_advisor_action_ledger_v0",
                "status": "blocked",
                "phase": str(plan.get("phase") or ""),
                "domain": str(plan.get("domain") or ""),
                "action_kind": str(plan.get("action_kind") or ""),
                "reason": "one or more advisor patch mutations were blocked",
                "applied_overlay": {},
                "rollback": {},
                "skipped": skipped,
            },
        }

    result_status = "applied" if applied else "no_change"
    action_ledger = {
        "schema_version": 1,
        "ledger": "bubble_advisor_action_ledger_v0",
        "action_id": action_id or _stable_action_id(plan, mutations),
        "status": result_status,
        "phase": str(plan.get("phase") or ""),
        "domain": str(plan.get("domain") or ""),
        "action_kind": str(plan.get("action_kind") or ""),
        "apply_scope": "next_request",
        "applied_overlay": applied,
        "rollback": {
            "restore": {str(item.get("path")): item.get("current") for item in mutations if item.get("path") in applied},
            "policy": _mapping(plan.get("rollback")),
        },
        "source_plan_status": str(plan.get("status") or ""),
        "source_report": _source_report_summary(report_or_plan, plan),
        "current_run_mutation": False,
    }
    return {
        "schema_version": 1,
        "patch": "bubble_advisor_patch_apply_v0",
        "status": result_status,
        "reason": "advisor patch applied to next request" if applied else "request already matches advisor patch",
        "can_apply_to_next_request": bool(applied),
        "can_apply_during_current_run": False,
        "patched_request": patched,
        "applied_overlay": applied,
        "skipped": skipped,
        "blocked_reasons": [],
        "action_ledger": action_ledger,
    }


def prepare_bubble_advisor_next_request(
    base_request: Mapping[str, Any],
    report_or_plan: Mapping[str, Any],
    *,
    action_id: str | None = None,
    allow_current_mismatch: bool = False,
    embed_ledger: bool = True,
    enable_next_run_observation: bool = True,
    max_history: int = 20,
) -> dict[str, Any]:
    """Prepare an auditable next-run request from an advisor patch.

    This is the S5 helper used by UI/API layers: it reuses the strict patch
    allowlist, then carries the ledger/history in the next request so the next
    run manifest can prove what was applied.
    """

    result = apply_bubble_advisor_patch_to_request(
        base_request,
        report_or_plan,
        action_id=action_id,
        allow_current_mismatch=allow_current_mismatch,
    )
    prepared_request = dict(result.get("patched_request") or {})
    next_request_overlay: dict[str, Any] = dict(_mapping(result.get("applied_overlay")))
    if result.get("status") not in {"applied", "no_change"}:
        return {
            **result,
            "prepared_request": prepared_request,
            "next_request_overlay": next_request_overlay,
            "prepared_next_request_changed": False,
        }

    if enable_next_run_observation:
        observation_overlay = {
            "bubble_controller_enabled": True,
            "bubble_controller_mode": "advisor_patch",
        }
        for key, value in observation_overlay.items():
            if prepared_request.get(key) != value:
                prepared_request[key] = value
                next_request_overlay[key] = value

    if embed_ledger:
        ledger = dict(_mapping(result.get("action_ledger")))
        ledger["prepared_next_request"] = True
        history = _append_ledger_history(prepared_request, ledger, max_history=max_history)
        prepared_request[BUBBLE_ADVISOR_LEDGER_FIELD] = ledger
        prepared_request[BUBBLE_ADVISOR_HISTORY_FIELD] = history
        next_request_overlay[BUBBLE_ADVISOR_LEDGER_FIELD] = ledger
        next_request_overlay[BUBBLE_ADVISOR_HISTORY_FIELD] = history

    return {
        **result,
        "patched_request": prepared_request,
        "prepared_request": prepared_request,
        "next_request_overlay": next_request_overlay,
        "prepared_next_request_changed": bool(next_request_overlay),
    }


__all__ = [
    "ALLOWED_BUBBLE_PATCH_PATHS",
    "BUBBLE_ADVISOR_HISTORY_FIELD",
    "BUBBLE_ADVISOR_LEDGER_FIELD",
    "apply_bubble_advisor_patch_to_request",
    "prepare_bubble_advisor_next_request",
]
