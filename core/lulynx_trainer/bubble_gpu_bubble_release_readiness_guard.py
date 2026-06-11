"""Fail-closed release guard for GPU-bubble readiness artifacts.

This guard cross-checks the top-level readiness report and terminal self-check.
It is read-only and does not make release evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.core.lulynx_trainer.bubble_post_input_refresh_contract import (
    POST_INPUT_REFRESH_SEQUENCE,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = "bubble_gpu_bubble_release_readiness_guard_v0"
READINESS_REPORT = "gpu_bubble_experiment_readiness_next_actions_v0"
TERMINAL_REPORT = "bubble_gpu_bubble_readiness_terminal_self_check_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
BLOCKED_RELEASE_POLICY = "ship_stable_baseline_without_gpu_bubble_gain_claim"
CLAIM_WORDING_POLICY = "case_specific_non_release_only"
REQUIRED_METRIC_IDS = (
    "step_wall_ms",
    "samples_per_second",
    "data_wait_share",
    "transfer_step_share",
    "train_step_total_share",
    "optimizer_update_total_share",
    "host_gap_share",
    "peak_vram_mb",
    "cache_prefetch_hit_miss",
    "loss_delta",
)
REQUIRED_EXPERIMENT_BATCH_IDS = (
    "batch_0_baseline_observation",
    "batch_1_data_supply",
    "batch_2_h2d_transfer",
    "batch_3_optimizer_backward",
    "batch_4_compute_kernel",
    "batch_5_scheduling_checkpoint",
)
REQUIRED_ACCEPTANCE_GATE_IDS = (
    "same_data_seed_steps_batch_accumulation",
    "steady_window_after_warmup",
    "steady_step_or_samples_per_second_gain",
    "dominant_bottleneck_or_key_share_reduced",
    "peak_vram_within_target_tier",
    "loss_and_artifact_normal",
    "manifest_runtime_snapshot_records_resolved_strategy",
    "sd15_lora_512_release_coverage",
    "natural_load_canary_release_gate",
    "case_specific_release_wording_only_after_rebuild",
)
NORMALIZED_EVIDENCE_GATE_IDS = (
    "steady_step_or_samples_per_second_gain",
    "dominant_bottleneck_or_key_share_reduced",
    "peak_vram_within_target_tier",
    "loss_and_artifact_normal",
    "manifest_runtime_snapshot_records_resolved_strategy",
    "sd15_lora_512_release_coverage",
    "natural_load_canary_release_gate",
    "case_specific_release_wording_only_after_rebuild",
)
REQUIRED_ATTRIBUTION_RULE_IDS = (
    "data_bound",
    "transfer_bound",
    "optimizer_bound",
    "logging_checkpoint_bound",
    "host_scheduling_bound",
    "compute_bound",
)
REQUIRED_FAMILY_STRATEGY_IDS = ("sd15", "sdxl", "anima", "newbie", "flux_dit")
REQUIRED_PROGRESSION_PHASE_IDS = (
    "phase_1_observation_foundation",
    "phase_2_short_benchmark_baselines",
    "phase_3_single_factor_ab",
    "phase_4_combined_strategy",
    "phase_5_long_training_validation",
    "phase_6_ui_advisor",
)
FORBIDDEN_CLAIM_WORDING_TOKENS = (
    "gpu bubble release gain claim",
    "gpu bubble gain claim is ready",
    "public release claim",
    "product release claim",
    "product release gain",
    "global gpu utilization claim",
    "universal gpu utilization claim",
    "all training jobs reach 99",
    "release gain claim allowed",
)
def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]


def _post_input_refresh_sequence(value: Any = None) -> list[str]:
    sequence = [item for item in _strings(value) if item]
    if not sequence:
        return list(POST_INPUT_REFRESH_SEQUENCE)
    if "refresh_external_input_handoff_packet" not in sequence:
        try:
            insert_at = sequence.index("refresh_source_cache_axis_pipeline_readiness") + 1
        except ValueError:
            insert_at = max(len(sequence) - 2, 0)
        sequence.insert(insert_at, "refresh_external_input_handoff_packet")
    return sequence


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _unique(values: Sequence[Any]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _count_string_field(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _release_exit_related_action_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "related_family_ids": _unique([row.get("family") for row in rows])[:20],
        "related_family_counts": _count_string_field(rows, "family"),
        "related_readiness_state_counts": _count_string_field(rows, "readiness_state"),
        "related_blocker_kind_counts": _count_string_field(rows, "readiness_blocker_kind"),
        "related_external_input_action_count": sum(
            1 for row in rows if bool(row.get("requires_external_input"))
        ),
        "related_manual_gpu_action_count": sum(
            1
            for row in rows
            if bool(row.get("current_action_requires_gpu"))
            or bool(_mapping(row.get("resolution_contract")).get("requires_manual_gpu"))
        ),
        "related_protected_followup_action_count": sum(
            1
            for row in rows
            if bool(row.get("followup_requires_gpu_heavy_run"))
            or bool(_mapping(row.get("resolution_contract")).get("requires_protected_runner"))
        ),
    }


def _action_by_id(readiness: Mapping[str, Any], action_id: str) -> Mapping[str, Any]:
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        if str(action.get("id") or "") == action_id:
            return action
    return {}


def _failure(failures: list[dict[str, Any]], check_id: str, reason: str, *, value: Any = None) -> None:
    failures.append({"id": check_id, "reason": reason, "value": value})


def _manual_gpu_counts_from_actions(readiness: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "gpu_related_action_count": 0,
        "current_gpu_heavy_action_count": 0,
        "followup_gpu_required_action_count": 0,
        "protected_manual_gpu_ready_action_count": 0,
        "blocked_missing_prerequisite_gpu_action_count": 0,
        "waiting_manual_gpu_evidence_action_count": 0,
    }
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        state = str(action.get("readiness_state") or "")
        current = bool(action.get("requires_gpu_heavy_run"))
        followup = bool(action.get("followup_requires_gpu_heavy_run"))
        is_gpu_related = current or followup or state in {
            "protected_manual_gpu_ready",
            "followup_axis_preparation_ready",
            "blocked_missing_prerequisite",
            "waiting_manual_gpu_evidence",
        }
        if not is_gpu_related:
            continue
        counts["gpu_related_action_count"] += 1
        if current:
            counts["current_gpu_heavy_action_count"] += 1
        if followup:
            counts["followup_gpu_required_action_count"] += 1
        if state == "protected_manual_gpu_ready":
            counts["protected_manual_gpu_ready_action_count"] += 1
        if state == "blocked_missing_prerequisite":
            counts["blocked_missing_prerequisite_gpu_action_count"] += 1
        if state == "waiting_manual_gpu_evidence":
            counts["waiting_manual_gpu_evidence_action_count"] += 1
    return counts


def _next_action_machine_summary_from_actions(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    state_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    unsafe_action_ids: list[str] = []
    missing_field_action_ids: list[str] = []
    for action in actions:
        action_id = str(action.get("id") or "")
        state = str(action.get("readiness_state") or "unknown")
        blocker = str(action.get("readiness_blocker_kind") or "")
        state_counts[state] = state_counts.get(state, 0) + 1
        if blocker:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if (
            bool(action.get("safe_to_auto_start"))
            or bool(action.get("release_claim_allowed_after_success"))
            or not bool(action.get("not_release_evidence"))
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe_action_ids.append(action_id)
        if (
            not action_id
            or not state
            or "readiness_blocker_kind" not in action
            or "primary_blocker" not in action
            or "secondary_readiness_blockers" not in action
        ):
            missing_field_action_ids.append(action_id or str(action.get("dedupe_key") or ""))
    return {
        "unique_action_count": len(actions),
        "readiness_state_counts": dict(sorted(state_counts.items())),
        "readiness_blocker_kind_counts": dict(sorted(blocker_counts.items())),
        "json_ready_action_count": state_counts.get("json_ready", 0),
        "json_closed_action_count": state_counts.get("json_closed", 0),
        "unsafe_action_count": len(unsafe_action_ids),
        "unsafe_action_ids": _unique(unsafe_action_ids),
        "missing_machine_field_action_count": len(missing_field_action_ids),
        "missing_machine_field_action_ids": _unique(missing_field_action_ids),
    }


def _next_action_contract_summary_from_actions(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    required_keys = (
        "id",
        "status",
        "readiness_state",
        "readiness_blocker_kind",
        "action_kind",
        "primary_blocker",
        "blocker_id",
        "requires_gpu_if_executed",
        "requires_external_input",
        "manual_start_required",
        "safe_to_auto_start",
        "release_claim_allowed_after_success",
        "not_release_evidence",
        "roadmap",
    )
    required_string_keys = (
        "id",
        "status",
        "readiness_state",
        "action_kind",
        "primary_blocker",
        "blocker_id",
        "roadmap",
    )
    missing_by_action: list[dict[str, Any]] = []
    for action in actions:
        missing = [
            key
            for key in required_keys
            if key not in action
            or (key in required_string_keys and not str(action.get(key) or ""))
        ]
        if missing:
            missing_by_action.append(
                {
                    "id": str(action.get("id") or action.get("dedupe_key") or ""),
                    "missing_keys": missing,
                }
            )
    unsafe_ids = [
        str(action.get("id") or "")
        for action in actions
        if bool(action.get("safe_to_auto_start"))
        or bool(action.get("release_claim_allowed_after_success"))
        or not bool(action.get("not_release_evidence"))
        or str(action.get("roadmap") or "") != ROADMAP
    ]
    return {
        "action_count": len(actions),
        "contract_complete_action_count": len(actions) - len(missing_by_action),
        "missing_contract_action_count": len(missing_by_action),
        "missing_contract_action_ids": [str(item.get("id") or "") for item in missing_by_action],
        "missing_contract_fields_by_action": missing_by_action,
        "release_or_auto_start_unsafe_action_count": len(unsafe_ids),
        "release_or_auto_start_unsafe_action_ids": _unique(unsafe_ids),
        "contract_ok": not missing_by_action and not unsafe_ids,
    }


def _remaining_work_summary_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    first_release = _mapping(readiness.get("first_release_scope"))
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    json_ready: list[str] = []
    json_closed: list[str] = []
    external_input: list[str] = []
    missing_prerequisite: list[str] = []
    manual_gpu_evidence: list[str] = []
    protected_manual_gpu_ready: list[str] = []
    followup_gpu_required: list[str] = []
    current_gpu_heavy: list[str] = []
    cache_axis_not_ready: list[str] = []
    duplicate_or_stale_axis: list[str] = []
    unsafe_action_ids: list[str] = []
    for action in actions:
        action_id = str(action.get("id") or "")
        if not action_id:
            continue
        state = str(action.get("readiness_state") or "")
        blocker = str(action.get("readiness_blocker_kind") or "")
        if state == "json_review_ready":
            json_ready.append(action_id)
        if state == "json_closed":
            json_closed.append(action_id)
        if state == "waiting_external_input" or bool(action.get("requires_external_input")):
            external_input.append(action_id)
        if state == "blocked_missing_prerequisite" or blocker == "sd15_checkpoint":
            missing_prerequisite.append(action_id)
        if state == "waiting_manual_gpu_evidence":
            manual_gpu_evidence.append(action_id)
        if state == "protected_manual_gpu_ready":
            protected_manual_gpu_ready.append(action_id)
        if bool(action.get("followup_requires_gpu_heavy_run")):
            followup_gpu_required.append(action_id)
        if bool(action.get("requires_gpu_heavy_run")):
            current_gpu_heavy.append(action_id)
        if state == "cache_axis_not_ready":
            cache_axis_not_ready.append(action_id)
        if state == "blocked_duplicate_or_stale_axis":
            duplicate_or_stale_axis.append(action_id)
        if (
            bool(action.get("safe_to_auto_start"))
            or bool(action.get("release_claim_allowed_after_success"))
            or not bool(action.get("not_release_evidence"))
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe_action_ids.append(action_id)
    release_gate_ids = _strings(
        first_release.get("gpu_bubble_release_hard_gate_ids")
    ) or _strings(unblocker.get("gpu_bubble_release_hard_gate_ids"))
    release_gate_actions = _unique(
        [*missing_prerequisite, *external_input, *manual_gpu_evidence, *cache_axis_not_ready]
    )
    recommended_next = (
        "provide_external_inputs_or_refresh_json_admission"
        if external_input or missing_prerequisite
        else "review_json_ready_actions"
        if json_ready
        else "prepare_manual_gpu_evidence_plan"
        if manual_gpu_evidence or protected_manual_gpu_ready or followup_gpu_required
        else "no_json_only_work_remaining"
    )
    return {
        "total_action_count": len(actions),
        "stable_first_release_blocked_by_this_artifact": bool(
            first_release.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_blocked": bool(
            first_release.get("gpu_bubble_release_claim_blocked")
        ),
        "gpu_bubble_release_hard_gate_count": len(release_gate_ids),
        "gpu_bubble_release_hard_gate_ids": release_gate_ids[:20],
        "json_ready_action_count": len(json_ready),
        "json_ready_action_ids": json_ready[:50],
        "json_closed_action_count": len(json_closed),
        "json_closed_action_ids": json_closed[:50],
        "external_input_action_count": len(external_input),
        "external_input_action_ids": external_input[:50],
        "missing_prerequisite_action_count": len(missing_prerequisite),
        "missing_prerequisite_action_ids": missing_prerequisite[:50],
        "manual_gpu_evidence_action_count": len(manual_gpu_evidence),
        "manual_gpu_evidence_action_ids": manual_gpu_evidence[:50],
        "protected_manual_gpu_ready_action_count": len(protected_manual_gpu_ready),
        "protected_manual_gpu_ready_action_ids": protected_manual_gpu_ready[:50],
        "followup_gpu_required_action_count": len(followup_gpu_required),
        "followup_gpu_required_action_ids": followup_gpu_required[:50],
        "current_gpu_heavy_action_count": len(current_gpu_heavy),
        "current_gpu_heavy_action_ids": current_gpu_heavy[:50],
        "cache_axis_not_ready_action_count": len(cache_axis_not_ready),
        "cache_axis_not_ready_action_ids": cache_axis_not_ready[:50],
        "duplicate_or_stale_axis_action_count": len(duplicate_or_stale_axis),
        "duplicate_or_stale_axis_action_ids": duplicate_or_stale_axis[:50],
        "release_gate_related_action_count": len(release_gate_actions),
        "release_gate_related_action_ids": release_gate_actions[:50],
        "recommended_release_policy": str(first_release.get("recommended_release_policy") or ""),
        "recommended_next_non_gpu_focus": recommended_next,
        "unsafe_action_count": len(unsafe_action_ids),
        "unsafe_action_ids": _unique(unsafe_action_ids),
        "fail_closed": not unsafe_action_ids,
    }


def _action_text(action: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "id",
        "status",
        "family",
        "action_type",
        "description",
        "rationale",
        "primary_blocker",
        "readiness_blocker_kind",
        "blocker_id",
        "expected_next_artifact",
    ):
        value = action.get(key)
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
    for key in (
        "secondary_readiness_blockers",
        "required_inputs",
        "blocked_by",
        "depends_on",
        "related_artifacts",
    ):
        parts.extend(_strings(action.get(key)))
    return " ".join(parts).lower()


def _manual_review_outcome_kind_from_action(action: Mapping[str, Any]) -> str:
    if bool(action.get("followup_requires_gpu_heavy_run")):
        return "followup_gpu_plan"
    text = _action_text(action)
    blocked_tokens = (
        "threshold_not_met",
        "below_threshold",
        "regressed",
        "execution_failed",
        "candidate_execution_failed",
        "missing_summaries",
        "no_candidate",
        "rollback",
    )
    if any(token in text for token in blocked_tokens):
        return "blocked_or_regression_review"
    action_type = str(action.get("action_type") or "")
    if (
        "diagnostic" in action_type
        or "promotion" in action_type
        or bool(action.get("diagnostic_only"))
        or "debug_repeat" in text
        or "debug_candidate" in text
    ):
        return "diagnostic_or_promotion_review"
    positive_tokens = ("throughput_gain_met", "candidate_found", "candidate_passed", "candidate_review")
    if any(token in text for token in positive_tokens):
        return "positive_candidate_review"
    return "manual_review"


def _manual_review_queue_summary_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    manual_review = [
        action for action in actions if str(action.get("readiness_state") or "") == "manual_review_ready"
    ]
    closed_blocked_regression = [
        action
        for action in actions
        if str(action.get("readiness_state") or "") == "json_closed"
        and bool(action.get("closed_as_blocked_or_regression"))
    ]
    closed_diagnostic_promotion = [
        action
        for action in actions
        if str(action.get("readiness_state") or "") == "json_closed"
        and bool(action.get("closed_as_diagnostic_promotion_recorded"))
    ]
    review_only = [
        action
        for action in manual_review
        if not bool(action.get("requires_gpu_heavy_run")) and not bool(action.get("followup_requires_gpu_heavy_run"))
    ]
    followup_gpu = [
        action for action in manual_review if bool(action.get("followup_requires_gpu_heavy_run"))
    ]
    current_gpu = [
        action for action in manual_review if bool(action.get("requires_gpu_heavy_run"))
    ]
    outcome_ids: dict[str, list[str]] = {}
    for action in manual_review:
        outcome = _manual_review_outcome_kind_from_action(action)
        outcome_ids.setdefault(outcome, []).append(str(action.get("id") or ""))
    unsafe_action_ids = [
        str(action.get("id") or "")
        for action in manual_review
        if bool(action.get("safe_to_auto_start"))
        or bool(action.get("release_claim_allowed_after_success"))
        or not bool(action.get("not_release_evidence"))
        or str(action.get("roadmap") or "") != ROADMAP
    ]
    return {
        "manual_review_ready_count": len(manual_review),
        "closed_blocked_or_regression_count": len(closed_blocked_regression),
        "closed_diagnostic_or_promotion_count": len(closed_diagnostic_promotion),
        "review_only_action_count": len(review_only),
        "followup_gpu_action_count": len(followup_gpu),
        "current_gpu_heavy_action_count": len(current_gpu),
        "closed_blocked_or_regression_action_ids": [
            str(action.get("id") or "") for action in closed_blocked_regression[:20]
        ],
        "closed_diagnostic_or_promotion_action_ids": [
            str(action.get("id") or "") for action in closed_diagnostic_promotion[:20]
        ],
        "review_only_action_ids": [str(action.get("id") or "") for action in review_only[:20]],
        "followup_gpu_action_ids": [str(action.get("id") or "") for action in followup_gpu[:20]],
        "current_gpu_heavy_action_ids": [str(action.get("id") or "") for action in current_gpu[:20]],
        "review_outcome_counts": {
            outcome: len(ids) for outcome, ids in sorted(outcome_ids.items())
        },
        "review_outcome_action_ids": {
            outcome: ids[:20] for outcome, ids in sorted(outcome_ids.items())
        },
        "unsafe_action_count": len(unsafe_action_ids),
        "unsafe_action_ids": _unique(unsafe_action_ids)[:20],
        "fail_closed": len(unsafe_action_ids) == 0,
    }


def _protected_followup_gpu_queue_from_actions(readiness: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unsafe_ids: list[str] = []
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        if not bool(action.get("followup_requires_gpu_heavy_run")):
            continue
        action_id = str(action.get("id") or "")
        row = {
            "id": action_id,
            "family": str(action.get("family") or ""),
            "readiness_state": str(action.get("readiness_state") or ""),
            "readiness_blocker_kind": str(action.get("readiness_blocker_kind") or ""),
            "current_action_requires_gpu": bool(action.get("requires_gpu_heavy_run")),
            "followup_requires_gpu_heavy_run": True,
            "followup_manual_start_required": True,
            "current_action_manual_start_required": bool(action.get("manual_start_required")),
            "requires_external_input": bool(action.get("requires_external_input")),
            "safe_to_auto_start": bool(action.get("safe_to_auto_start")),
            "release_claim_allowed_after_success": bool(
                action.get("release_claim_allowed_after_success")
            ),
            "not_release_evidence": bool(action.get("not_release_evidence")),
        }
        if (
            row["current_action_requires_gpu"]
            or row["current_action_manual_start_required"]
            or row["safe_to_auto_start"]
            or row["release_claim_allowed_after_success"]
            or not row["not_release_evidence"]
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe_ids.append(action_id)
        rows.append(row)
    family_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family") or "unknown")
        blocker = str(row.get("readiness_blocker_kind") or "unknown")
        state = str(row.get("readiness_state") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "followup_gpu_required_action_count": len(rows),
        "followup_gpu_required_action_ids": [str(row.get("id") or "") for row in rows[:20]],
        "family_counts": dict(sorted(family_counts.items())),
        "readiness_state_counts": dict(sorted(state_counts.items())),
        "readiness_blocker_kind_counts": dict(sorted(blocker_counts.items())),
        "current_action_gpu_count": sum(1 for row in rows if bool(row.get("current_action_requires_gpu"))),
        "current_action_manual_start_count": sum(
            1 for row in rows if bool(row.get("current_action_manual_start_required"))
        ),
        "followup_manual_start_required_count": len(rows),
        "requires_external_input_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "unsafe_action_count": len(unsafe_ids),
        "unsafe_action_ids": _unique(unsafe_ids),
        "row_ids": [str(row.get("id") or "") for row in rows],
    }


def _remaining_release_blocker_matrix_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    remaining = _mapping(readiness.get("remaining_work_summary"))
    unclosed: list[Mapping[str, Any]] = []
    state_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    family_action_counts: dict[str, int] = {}
    family_external_input_counts: dict[str, int] = {}
    family_manual_gpu_counts: dict[str, int] = {}
    family_protected_followup_counts: dict[str, int] = {}
    family_source_cache_blocked_counts: dict[str, int] = {}
    family_unsafe_counts: dict[str, int] = {}
    external_input: list[str] = []
    manual_gpu: list[str] = []
    protected_followup: list[str] = []
    source_cache: list[str] = []
    sd15_checkpoint: list[str] = []
    duplicate_or_stale: list[str] = []
    unsafe: list[str] = []
    for action in actions:
        action_id = str(action.get("id") or "")
        state = str(action.get("readiness_state") or "")
        blocker = str(action.get("readiness_blocker_kind") or "")
        primary = str(action.get("primary_blocker") or "")
        if not action_id or state == "json_closed":
            continue
        family = str(action.get("family") or "unknown")
        unclosed.append(action)
        state_counts[state or "unknown"] = state_counts.get(state or "unknown", 0) + 1
        blocker_counts[blocker or "unknown"] = blocker_counts.get(blocker or "unknown", 0) + 1
        family_action_counts[family] = family_action_counts.get(family, 0) + 1
        if state == "waiting_external_input" or bool(action.get("requires_external_input")):
            external_input.append(action_id)
            family_external_input_counts[family] = family_external_input_counts.get(family, 0) + 1
        if (
            bool(action.get("requires_gpu_heavy_run"))
            or state in {"waiting_manual_gpu_evidence", "protected_manual_gpu_ready"}
        ):
            manual_gpu.append(action_id)
            family_manual_gpu_counts[family] = family_manual_gpu_counts.get(family, 0) + 1
        if bool(action.get("followup_requires_gpu_heavy_run")):
            protected_followup.append(action_id)
            family_protected_followup_counts[family] = (
                family_protected_followup_counts.get(family, 0) + 1
            )
        if state == "cache_axis_not_ready" or blocker in {
            "source_cache_axis",
            "source_cache_axis_pipeline",
            "source_cache_axis_manual_canary",
        }:
            source_cache.append(action_id)
            family_source_cache_blocked_counts[family] = (
                family_source_cache_blocked_counts.get(family, 0) + 1
            )
        if blocker == "sd15_checkpoint" or "sd15" in action_id or "sd15" in primary:
            sd15_checkpoint.append(action_id)
        if state == "blocked_duplicate_or_stale_axis":
            duplicate_or_stale.append(action_id)
        if (
            bool(action.get("safe_to_auto_start"))
            or bool(action.get("release_claim_allowed_after_success"))
            or not bool(action.get("not_release_evidence"))
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe.append(action_id)
            family_unsafe_counts[family] = family_unsafe_counts.get(family, 0) + 1
    missing_inputs = _strings(unblocker.get("missing_external_inputs"))
    next_unlock_inputs = _unique(
        [
            *missing_inputs,
            *(["manual_gpu_evidence"] if manual_gpu else []),
            *(["protected_followup_manual_gpu_run"] if protected_followup else []),
            *(["source_cache_axis_refresh"] if source_cache else []),
        ]
    )
    return {
        "total_unclosed_action_count": len(unclosed),
        "unclosed_action_ids": [str(action.get("id") or "") for action in unclosed[:50]],
        "json_ready_action_count": state_counts.get("json_review_ready", 0),
        "readiness_state_counts": dict(sorted(state_counts.items())),
        "blocked_by_kind_counts": dict(sorted(blocker_counts.items())),
        "family_count": len(family_action_counts),
        "family_ids": sorted(family_action_counts),
        "family_action_counts": dict(sorted(family_action_counts.items())),
        "family_external_input_required_counts": dict(sorted(family_external_input_counts.items())),
        "family_manual_gpu_required_counts": dict(sorted(family_manual_gpu_counts.items())),
        "family_protected_followup_gpu_required_counts": dict(
            sorted(family_protected_followup_counts.items())
        ),
        "family_source_cache_blocked_counts": dict(sorted(family_source_cache_blocked_counts.items())),
        "unsafe_family_count": len(family_unsafe_counts),
        "family_unsafe_action_counts": dict(sorted(family_unsafe_counts.items())),
        "release_hard_gate_ids": _strings(
            unblocker.get("gpu_bubble_release_hard_gate_ids")
        ) or _strings(remaining.get("gpu_bubble_release_hard_gate_ids")),
        "external_input_required_action_count": len(external_input),
        "external_input_required_action_ids": external_input[:50],
        "missing_external_inputs": missing_inputs[:20],
        "manual_gpu_required_action_count": len(manual_gpu),
        "manual_gpu_required_action_ids": manual_gpu[:50],
        "protected_followup_gpu_required_action_count": len(protected_followup),
        "protected_followup_gpu_required_action_ids": protected_followup[:50],
        "source_cache_blocked_action_count": len(source_cache),
        "source_cache_blocked_action_ids": source_cache[:50],
        "sd15_checkpoint_action_count": len(sd15_checkpoint),
        "sd15_checkpoint_action_ids": sd15_checkpoint[:50],
        "duplicate_or_stale_source_axis_action_count": len(duplicate_or_stale),
        "duplicate_or_stale_source_axis_action_ids": duplicate_or_stale[:50],
        "next_unlock_inputs": next_unlock_inputs[:50],
        "unsafe_action_count": len(unsafe),
        "unsafe_action_ids": _unique(unsafe),
    }


def _remaining_blocker_resolution_handoff_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    input_resolution = _mapping(unblocker.get("input_resolution_summary"))
    missing_inputs = _strings(unblocker.get("missing_external_inputs"))
    refresh_commands = _post_input_refresh_sequence(input_resolution.get("next_json_refresh_sequence"))
    rows: list[dict[str, Any]] = []
    unsafe_ids: list[str] = []
    bucket_counts: dict[str, int] = {}
    resolution_bad_ids: list[str] = []

    for action in actions:
        action_id = str(action.get("id") or "")
        state = str(action.get("readiness_state") or "")
        blocker = str(action.get("readiness_blocker_kind") or "")
        if not action_id or state == "json_closed":
            continue
        bucket = "external_input"
        unlock_inputs: list[str] = []
        if blocker == "sd15_checkpoint" or "sd15" in action_id:
            bucket = "sd15_checkpoint_then_manual_gpu"
            unlock_inputs = ["sd15_checkpoint", "manual_gpu_evidence"]
        elif bool(action.get("followup_requires_gpu_heavy_run")) or blocker == "protected_followup_axis":
            bucket = "protected_followup_manual_gpu"
            unlock_inputs = ["protected_followup_manual_gpu_run", "manual_gpu_evidence"]
        elif state == "waiting_manual_gpu_evidence" or blocker == "manual_gpu_evidence":
            bucket = "manual_gpu_evidence_then_rebuild"
            unlock_inputs = [*missing_inputs, "manual_gpu_evidence"]
        elif blocker in {"duplicate_or_stale_source_axis", "source_cache_axis_not_ready"} or state in {
            "blocked_duplicate_or_stale_axis",
            "cache_axis_not_ready",
        }:
            bucket = "source_cache_axis_refresh"
            unlock_inputs = ["new_source_root", "source_cache_axis_refresh"]
        elif bool(action.get("requires_external_input")):
            bucket = "external_input_refresh"
            unlock_inputs = missing_inputs
        requires_external = bool(action.get("requires_external_input"))
        requires_current_gpu = bool(action.get("requires_gpu_heavy_run"))
        requires_followup_gpu = bool(action.get("followup_requires_gpu_heavy_run"))
        resolution_contract = {
            "resolution_kind": bucket,
            "can_resolve_json_only_now": False,
            "requires_external_input": requires_external,
            "requires_manual_gpu": bool(
                requires_current_gpu
                or requires_followup_gpu
                or bucket in {
                    "sd15_checkpoint_then_manual_gpu",
                    "protected_followup_manual_gpu",
                    "manual_gpu_evidence_then_rebuild",
                }
            ),
            "requires_protected_runner": bool(requires_followup_gpu),
            "required_input_ids": _unique(unlock_inputs)[:20],
            "missing_input_ids": _unique(unlock_inputs)[:20],
            "post_unlock_refresh_command_ids": refresh_commands[:20],
            "post_unlock_required_artifact_ids": [
                "gpu_bubble_experiment_readiness_next_actions",
                "gpu_bubble_readiness_terminal_self_check",
                "gpu_bubble_release_readiness_guard_report",
            ],
            "terminal_guard_required": True,
            "release_claim_after_resolution_allowed": False,
            "safe_to_auto_start_after_resolution": False,
            "not_release_evidence": True,
        }
        if (
            not resolution_contract["resolution_kind"]
            or "refresh_gpu_bubble_readiness_next_actions" not in refresh_commands
            or "refresh_gpu_bubble_terminal_self_check" not in refresh_commands
            or "run_gpu_bubble_release_readiness_guard" not in refresh_commands
            or bool(resolution_contract["release_claim_after_resolution_allowed"])
            or bool(resolution_contract["safe_to_auto_start_after_resolution"])
            or not bool(resolution_contract["not_release_evidence"])
        ):
            resolution_bad_ids.append(action_id)
        row = {
            "id": action_id,
            "family": str(action.get("family") or ""),
            "readiness_state": state,
            "readiness_blocker_kind": blocker,
            "blocker_bucket": bucket,
            "resolution_contract": resolution_contract,
            "next_unlock_input_ids": _unique(unlock_inputs)[:20],
            "required_refresh_command_ids": refresh_commands[:20],
            "requires_external_input": requires_external,
            "current_action_requires_gpu": requires_current_gpu,
            "followup_requires_gpu_heavy_run": requires_followup_gpu,
            "safe_to_auto_start": bool(action.get("safe_to_auto_start")),
            "release_claim_allowed_after_success": bool(
                action.get("release_claim_allowed_after_success")
            ),
            "not_release_evidence": bool(action.get("not_release_evidence")),
        }
        if (
            row["safe_to_auto_start"]
            or row["release_claim_allowed_after_success"]
            or not row["not_release_evidence"]
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe_ids.append(action_id)
        rows.append(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return {
        "row_count": len(rows),
        "row_ids": [str(row.get("id") or "") for row in rows[:50]],
        "rows": rows,
        "resolution_contract_ok": not resolution_bad_ids,
        "resolution_contract_bad_count": len(resolution_bad_ids),
        "resolution_contract_bad_ids": _unique(resolution_bad_ids),
        "resolution_bucket_counts": dict(sorted(bucket_counts.items())),
        "json_only_resolution_available_count": 0,
        "external_input_required_count": sum(
            1 for row in rows if bool(_mapping(row.get("resolution_contract")).get("requires_external_input"))
        ),
        "manual_gpu_required_count": sum(
            1 for row in rows if bool(_mapping(row.get("resolution_contract")).get("requires_manual_gpu"))
        ),
        "protected_runner_required_count": sum(
            1 for row in rows if bool(_mapping(row.get("resolution_contract")).get("requires_protected_runner"))
        ),
        "release_claim_after_resolution_allowed": False,
        "blocker_bucket_counts": dict(sorted(bucket_counts.items())),
        "next_unlock_input_ids": _unique(
            [input_id for row in rows for input_id in _strings(row.get("next_unlock_input_ids"))]
        )[:50],
        "required_refresh_command_ids": refresh_commands[:20],
        "external_input_row_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "current_gpu_row_count": sum(1 for row in rows if bool(row.get("current_action_requires_gpu"))),
        "protected_followup_gpu_row_count": sum(
            1 for row in rows if bool(row.get("followup_requires_gpu_heavy_run"))
        ),
        "unsafe_row_count": len(unsafe_ids),
        "unsafe_row_ids": _unique(unsafe_ids),
    }


def _remaining_action_dependency_graph_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    handoff = _remaining_blocker_resolution_handoff_from_readiness(readiness)
    input_resolution = _mapping(unblocker.get("input_resolution_summary"))
    refresh_commands = _strings(input_resolution.get("next_json_refresh_sequence")) or _strings(
        handoff.get("required_refresh_command_ids")
    )
    if not refresh_commands:
        refresh_commands = [
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ]
    missing_inputs = _strings(unblocker.get("missing_external_inputs"))
    release_hard_gates = _strings(unblocker.get("gpu_bubble_release_hard_gate_ids"))
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    dependency_kind_counts: dict[str, int] = {}
    action_state_counts: dict[str, int] = {}
    blocker_kind_counts: dict[str, int] = {}
    unsafe_ids: list[str] = []

    for action in actions:
        action_id = str(action.get("id") or "")
        state = str(action.get("readiness_state") or "")
        if not action_id or state == "json_closed":
            continue
        blocker = str(action.get("readiness_blocker_kind") or action.get("primary_blocker") or "")
        dependency_ids: list[str] = []
        dependency_kinds: list[str] = []
        if bool(action.get("requires_external_input")):
            for input_id in missing_inputs or ["external_input"]:
                dependency_ids.append(f"external_input:{input_id}")
                dependency_kinds.append("external_input")
        if blocker == "sd15_checkpoint" or "sd15" in action_id:
            dependency_ids.append("external_input:sd15_checkpoint")
            dependency_ids.append("manual_gpu:sd15_lora512_ab_matrix")
            dependency_kinds.extend(["external_input", "manual_gpu_evidence"])
        if blocker == "manual_gpu_evidence" or state == "waiting_manual_gpu_evidence":
            dependency_ids.append("manual_gpu:post_manual_evidence")
            dependency_kinds.append("manual_gpu_evidence")
        if bool(action.get("requires_gpu_heavy_run")):
            dependency_ids.append(f"manual_gpu:{action_id}")
            dependency_kinds.append("manual_gpu_evidence")
        if bool(action.get("followup_requires_gpu_heavy_run")) or blocker == "protected_followup_axis":
            dependency_ids.append(f"protected_followup_gpu:{action_id}")
            dependency_ids.append("manual_gpu:post_manual_evidence")
            dependency_kinds.extend(["protected_followup_gpu", "manual_gpu_evidence"])
        if blocker in {"duplicate_or_stale_source_axis", "source_cache_axis_not_ready"} or state in {
            "blocked_duplicate_or_stale_axis",
            "cache_axis_not_ready",
        }:
            dependency_ids.append("source_cache_axis:new_or_repaired_axis")
            dependency_kinds.append("source_cache_axis")
        for gate_id in release_hard_gates:
            dependency_ids.append(f"release_hard_gate:{gate_id}")
            dependency_kinds.append("release_hard_gate")
        dependency_ids = _unique(dependency_ids)
        dependency_kinds = _unique(dependency_kinds)
        for dep_id in dependency_ids:
            edges.append(
                {
                    "from_dependency_id": dep_id,
                    "to_action_id": action_id,
                    "dependency_kind": dep_id.split(":", 1)[0],
                }
            )
        for dep_kind in dependency_kinds:
            dependency_kind_counts[dep_kind] = dependency_kind_counts.get(dep_kind, 0) + 1
        action_state_counts[state] = action_state_counts.get(state, 0) + 1
        blocker_kind_counts[blocker] = blocker_kind_counts.get(blocker, 0) + 1
        unsafe = (
            bool(action.get("safe_to_auto_start"))
            or bool(action.get("release_claim_allowed_after_success"))
            or not bool(action.get("not_release_evidence"))
            or str(action.get("roadmap") or "") != ROADMAP
        )
        if unsafe:
            unsafe_ids.append(action_id)
        rows.append(
            {
                "action_id": action_id,
                "readiness_state": state,
                "readiness_blocker_kind": blocker,
                "dependency_ids": dependency_ids[:30],
                "dependency_kinds": dependency_kinds[:20],
                "requires_external_input": bool(action.get("requires_external_input")),
                "requires_current_gpu": bool(action.get("requires_gpu_heavy_run")),
                "followup_requires_gpu_heavy_run": bool(action.get("followup_requires_gpu_heavy_run")),
                "safe_to_auto_start": bool(action.get("safe_to_auto_start")),
                "release_claim_allowed_after_success": bool(
                    action.get("release_claim_allowed_after_success")
                ),
                "not_release_evidence": bool(action.get("not_release_evidence")),
                "unsafe": unsafe,
            }
        )

    dependency_node_ids = _unique(edge.get("from_dependency_id") for edge in edges)
    refresh_ok = all(
        command_id in refresh_commands
        for command_id in [
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ]
    )
    return {
        "action_node_count": len(rows),
        "action_node_ids": [row["action_id"] for row in rows[:50]],
        "dependency_node_count": len(dependency_node_ids),
        "dependency_node_ids": dependency_node_ids[:80],
        "edge_count": len(edges),
        "edge_sample": edges[:80],
        "action_state_counts": dict(sorted(action_state_counts.items())),
        "blocker_kind_counts": dict(sorted(blocker_kind_counts.items())),
        "dependency_kind_counts": dict(sorted(dependency_kind_counts.items())),
        "missing_external_inputs": missing_inputs[:20],
        "release_hard_gate_ids": release_hard_gates[:20],
        "required_refresh_command_ids": refresh_commands[:20],
        "refresh_sequence_terminal_guard_ok": refresh_ok,
        "rows": rows,
        "unsafe_action_count": len(unsafe_ids),
        "unsafe_action_ids": _unique(unsafe_ids)[:50],
        "fail_closed": len(unsafe_ids) == 0 and refresh_ok,
    }


def _remaining_action_unblock_sequence_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    graph = _remaining_action_dependency_graph_from_readiness(readiness)
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    blocker_handoff = _mapping(readiness.get("remaining_blocker_resolution_handoff_summary"))
    missing_inputs = _strings(unblocker.get("missing_external_inputs"))
    hard_gates = _strings(unblocker.get("gpu_bubble_release_hard_gate_ids"))
    refresh_commands = _strings(graph.get("required_refresh_command_ids"))
    graph_rows = [_mapping(item) for item in _list(graph.get("rows"))]
    external_action_ids = [
        str(row.get("action_id") or "")
        for row in graph_rows
        if "external_input" in _strings(row.get("dependency_kinds"))
        or bool(row.get("requires_external_input"))
    ]
    source_cache_action_ids = [
        str(row.get("action_id") or "")
        for row in graph_rows
        if "source_cache_axis" in _strings(row.get("dependency_kinds"))
    ]
    manual_gpu_action_ids = [
        str(row.get("action_id") or "")
        for row in graph_rows
        if "manual_gpu_evidence" in _strings(row.get("dependency_kinds"))
        or bool(row.get("requires_current_gpu"))
    ]
    protected_gpu_action_ids = [
        str(row.get("action_id") or "")
        for row in graph_rows
        if "protected_followup_gpu" in _strings(row.get("dependency_kinds"))
        or bool(row.get("followup_requires_gpu_heavy_run"))
    ]
    terminal_command_ids = [
        command_id
        for command_id in refresh_commands
        if command_id in {"refresh_gpu_bubble_terminal_self_check", "run_gpu_bubble_release_readiness_guard"}
    ]
    rows = [
        {
            "stage_id": "provide_external_inputs",
            "stage_order": 10,
            "stage_kind": "external_input",
            "status": "blocked_waiting_external_input" if missing_inputs else "ready_for_json_refresh",
            "required_input_ids": missing_inputs[:20],
            "related_action_ids": _unique(external_action_ids)[:50],
            "requires_external_input": bool(missing_inputs),
            "requires_manual_gpu": False,
            "requires_protected_runner": False,
            "required_refresh_command_ids": [],
            "terminal_guard_required_after_stage": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_stage": False,
            "not_release_evidence": True,
        },
        {
            "stage_id": "refresh_json_admission_chain",
            "stage_order": 20,
            "stage_kind": "json_refresh",
            "status": "blocked_until_external_inputs" if missing_inputs else "json_refresh_required",
            "required_input_ids": missing_inputs[:20],
            "related_action_ids": _unique([*external_action_ids, *source_cache_action_ids])[:50],
            "requires_external_input": bool(missing_inputs),
            "requires_manual_gpu": False,
            "requires_protected_runner": False,
            "required_refresh_command_ids": refresh_commands[:20],
            "terminal_guard_required_after_stage": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_stage": False,
            "not_release_evidence": True,
        },
        {
            "stage_id": "collect_manual_gpu_evidence",
            "stage_order": 30,
            "stage_kind": "manual_gpu_evidence",
            "status": "manual_or_protected_gpu_required",
            "required_input_ids": _unique(["manual_gpu_evidence", "protected_followup_manual_gpu_run"])[:20],
            "related_action_ids": _unique([*manual_gpu_action_ids, *protected_gpu_action_ids])[:50],
            "requires_external_input": False,
            "requires_manual_gpu": True,
            "requires_protected_runner": bool(protected_gpu_action_ids),
            "required_refresh_command_ids": [],
            "terminal_guard_required_after_stage": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_stage": False,
            "not_release_evidence": True,
        },
        {
            "stage_id": "rebuild_post_manual_evidence_chain",
            "stage_order": 40,
            "stage_kind": "post_manual_json_rebuild",
            "status": "blocked_until_manual_gpu_evidence",
            "required_input_ids": _unique(["manual_gpu_evidence", *missing_inputs])[:20],
            "related_action_ids": _unique(manual_gpu_action_ids)[:50],
            "requires_external_input": bool(missing_inputs),
            "requires_manual_gpu": True,
            "requires_protected_runner": False,
            "required_refresh_command_ids": refresh_commands[:20],
            "terminal_guard_required_after_stage": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_stage": False,
            "not_release_evidence": True,
        },
        {
            "stage_id": "terminal_guard_release_claim_check",
            "stage_order": 50,
            "stage_kind": "terminal_guard",
            "status": "blocked_until_hard_gates_clear",
            "required_input_ids": hard_gates[:20],
            "related_action_ids": [
                str(row.get("action_id") or "")
                for row in graph_rows
                if "release_hard_gate" in _strings(row.get("dependency_kinds"))
            ][:50],
            "requires_external_input": bool(missing_inputs),
            "requires_manual_gpu": True,
            "requires_protected_runner": bool(protected_gpu_action_ids),
            "required_refresh_command_ids": terminal_command_ids[:20],
            "terminal_guard_required_after_stage": True,
            "safe_to_auto_start": False,
            "release_claim_allowed_after_stage": False,
            "not_release_evidence": True,
        },
    ]
    unsafe_stage_ids = [
        str(row.get("stage_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed_after_stage"))
        or not bool(row.get("not_release_evidence"))
    ]
    refresh_ok = all(
        command_id in refresh_commands
        for command_id in [
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ]
    )
    next_required_input_ids = _unique(
        [
            *(
                input_id
                for row in rows
                for input_id in _strings(row.get("required_input_ids"))
            ),
            *_strings(blocker_handoff.get("next_unlock_input_ids")),
        ]
    )
    return {
        "stage_count": len(rows),
        "stage_ids": [str(row.get("stage_id") or "") for row in rows],
        "current_stage_id": "provide_external_inputs" if missing_inputs else "collect_manual_gpu_evidence",
        "next_required_input_ids": next_required_input_ids[:50],
        "manual_gpu_stage_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "external_input_stage_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "protected_runner_stage_count": sum(1 for row in rows if bool(row.get("requires_protected_runner"))),
        "release_hard_gate_ids": hard_gates[:20],
        "terminal_guard_required": True,
        "required_refresh_command_ids": refresh_commands[:20],
        "refresh_sequence_terminal_guard_ok": refresh_ok,
        "rows": rows,
        "unsafe_stage_count": len(unsafe_stage_ids),
        "unsafe_stage_ids": unsafe_stage_ids[:20],
        "fail_closed": len(unsafe_stage_ids) == 0 and refresh_ok,
    }


def _path_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.exists()


def _remaining_blocker_artifact_presence_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_expected_ids: list[str] = []
    missing_evidence_ids: list[str] = []
    unsafe_ids: list[str] = []
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        action_id = str(action.get("id") or "")
        state = str(action.get("readiness_state") or "")
        if not action_id or state == "json_closed":
            continue
        expected_outputs = _strings(action.get("expected_outputs"))
        evidence_paths = _strings(action.get("evidence_paths"))
        expected_missing = [path for path in expected_outputs if not _path_exists(path)]
        evidence_missing = [path for path in evidence_paths if not _path_exists(path)]
        if expected_missing:
            missing_expected_ids.append(action_id)
        if evidence_missing:
            missing_evidence_ids.append(action_id)
        if (
            bool(action.get("safe_to_auto_start"))
            or bool(action.get("release_claim_allowed_after_success"))
            or not bool(action.get("not_release_evidence"))
            or str(action.get("roadmap") or "") != ROADMAP
        ):
            unsafe_ids.append(action_id)
        rows.append(
            {
                "id": action_id,
                "expected_output_count": len(expected_outputs),
                "expected_output_existing_count": len(expected_outputs) - len(expected_missing),
                "expected_output_missing_count": len(expected_missing),
                "evidence_path_count": len(evidence_paths),
                "evidence_path_existing_count": len(evidence_paths) - len(evidence_missing),
                "evidence_path_missing_count": len(evidence_missing),
            }
        )
    return {
        "row_count": len(rows),
        "row_ids": [str(row.get("id") or "") for row in rows[:50]],
        "expected_output_action_count": sum(1 for row in rows if _safe_int(row.get("expected_output_count")) > 0),
        "expected_output_missing_action_count": len(missing_expected_ids),
        "expected_output_missing_action_ids": _unique(missing_expected_ids),
        "evidence_path_action_count": sum(1 for row in rows if _safe_int(row.get("evidence_path_count")) > 0),
        "evidence_path_missing_action_count": len(missing_evidence_ids),
        "evidence_path_missing_action_ids": _unique(missing_evidence_ids),
        "unsafe_row_count": len(unsafe_ids),
        "unsafe_row_ids": _unique(unsafe_ids),
    }


def _release_claim_exit_criteria_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    handoff = _remaining_blocker_resolution_handoff_from_readiness(readiness)
    presence = _remaining_blocker_artifact_presence_from_readiness(readiness)
    remaining = _mapping(readiness.get("remaining_work_summary"))
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    hard_gate_ids = (
        _strings(unblocker.get("gpu_bubble_release_hard_gate_ids"))
        or _strings(remaining.get("gpu_bubble_release_hard_gate_ids"))
        or _strings(readiness.get("gpu_bubble_release_hard_gate_ids"))
    )
    if not hard_gate_ids:
        hard_gate_ids = _unique(
            [
                str(_mapping(gate).get("case_id") or _mapping(gate).get("id") or "")
                for gate in _list(readiness.get("release_hard_gates"))
            ]
        )
    handoff_rows = [_mapping(row) for row in _list(handoff.get("rows"))]
    missing_expected_ids = set(_strings(presence.get("expected_output_missing_action_ids")))
    missing_evidence_ids = set(_strings(presence.get("evidence_path_missing_action_ids")))
    rows: list[dict[str, Any]] = []
    for gate_id in _unique(hard_gate_ids):
        if gate_id == "sd15_lora_512":
            action_ids = [
                str(row.get("id") or "")
                for row in handoff_rows
                if "sd15" in str(row.get("id") or "")
                or "sd15_checkpoint" in _strings(row.get("next_unlock_input_ids"))
            ]
            required_inputs = ["sd15_checkpoint", "manual_gpu_evidence"]
            required_outputs = [
                "real_ab_matrix_sd15_data_smoke/ab_matrix_results.json",
                "real_ab_matrix_sd15_data_smoke/evidence_pack/bubble_runtime_evidence_pack.json",
                "real_ab_matrix_sd15_data_smoke/sd15_data_workers_smoke/bubble_advisor_ab_evidence.json",
            ]
            manual_gpu_required = True
        elif gate_id == "natural_load_canary_pending":
            action_ids = [
                str(row.get("id") or "")
                for row in handoff_rows
                if str(row.get("blocker_bucket") or "")
                in {"source_cache_axis_refresh", "manual_gpu_evidence_then_rebuild"}
            ]
            required_inputs = ["new_source_root", "source_cache_axis_refresh", "manual_gpu_evidence"]
            required_outputs = [
                "current_combined/natural_load_canary.json",
                "current_combined/release_claims.json",
                "current_combined/evidence_pack.json",
            ]
            manual_gpu_required = True
        else:
            action_ids = []
            required_inputs = []
            required_outputs = []
            manual_gpu_required = False
        action_id_set = set(_unique(action_ids))
        related_rows = [
            row for row in handoff_rows if str(row.get("id") or "") in action_id_set
        ]
        rows.append(
            {
                "gate_id": gate_id,
                "gate_status": "blocked_pending_external_input_or_manual_gpu",
                "related_action_ids": _unique(action_ids)[:50],
                **_release_exit_related_action_summary(related_rows),
                "required_input_ids": required_inputs,
                "required_output_ids": required_outputs,
                "missing_declared_output_action_ids": sorted(
                    action_id_set & (missing_expected_ids | missing_evidence_ids)
                )[:50],
                "manual_gpu_required": manual_gpu_required,
                "protected_runner_required": any(
                    bool(row.get("followup_requires_gpu_heavy_run"))
                    for row in handoff_rows
                    if str(row.get("id") or "") in action_id_set
                ),
                "json_only_exit_available": False,
                "terminal_guard_required": True,
                "release_claim_allowed_after_exit": False,
                "safe_to_auto_start": False,
                "not_release_evidence": True,
            }
        )
    unsafe_gate_ids = [
        str(row.get("gate_id") or "")
        for row in rows
        if bool(row.get("release_claim_allowed_after_exit"))
        or bool(row.get("safe_to_auto_start"))
        or not bool(row.get("not_release_evidence"))
    ]
    return {
        "release_hard_gate_count": len(rows),
        "release_hard_gate_ids": [str(row.get("gate_id") or "") for row in rows],
        "gate_row_count": len(rows),
        "json_only_exit_available_count": sum(1 for row in rows if bool(row.get("json_only_exit_available"))),
        "manual_gpu_required_gate_count": sum(1 for row in rows if bool(row.get("manual_gpu_required"))),
        "protected_runner_required_gate_count": sum(
            1 for row in rows if bool(row.get("protected_runner_required"))
        ),
        "missing_declared_output_gate_count": sum(
            1 for row in rows if _strings(row.get("missing_declared_output_action_ids"))
        ),
        "unsafe_gate_count": len(unsafe_gate_ids),
        "unsafe_gate_ids": _unique(unsafe_gate_ids),
        "rows": rows,
        "fail_closed": len(unsafe_gate_ids) == 0,
    }


def _release_gate_input_kind(input_id: str) -> str:
    if input_id == "sd15_checkpoint":
        return "external_checkpoint"
    if input_id in {"new_source_root", "warm_cache_axis", "caption_repair_axis"}:
        return "external_source_or_cache_axis"
    if input_id == "source_cache_axis_refresh":
        return "source_cache_json_refresh"
    if input_id in {"manual_gpu_evidence", "protected_followup_manual_gpu_run"}:
        return "manual_gpu_evidence"
    return "release_gate_input"


def _release_gate_input_dependency_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    release_exit = _release_claim_exit_criteria_from_readiness(readiness)
    release_unblocker = _mapping(readiness.get("release_unblocker_summary"))
    gate_rows = [_mapping(row) for row in _list(release_exit.get("rows"))]
    hard_gate_ids = _strings(release_exit.get("release_hard_gate_ids"))
    missing_external_inputs = _strings(release_unblocker.get("missing_external_inputs"))
    manual_blocking = _mapping(release_unblocker.get("manual_evidence_blocking_summary"))
    required_input_ids = _unique(
        [
            *missing_external_inputs,
            *[
                input_id
                for gate_row in gate_rows
                for input_id in _strings(gate_row.get("required_input_ids"))
            ],
        ]
    )
    rows: list[dict[str, Any]] = []
    for input_id in required_input_ids:
        related_gate_rows = [
            gate_row
            for gate_row in gate_rows
            if input_id in _strings(gate_row.get("required_input_ids"))
        ]
        if not related_gate_rows and input_id in {
            "new_source_root",
            "warm_cache_axis",
            "caption_repair_axis",
            "source_cache_axis_refresh",
        }:
            related_gate_rows = [
                gate_row
                for gate_row in gate_rows
                if str(gate_row.get("gate_id") or "") == "natural_load_canary_pending"
            ]
        if not related_gate_rows and input_id == "sd15_checkpoint":
            related_gate_rows = [
                gate_row for gate_row in gate_rows if str(gate_row.get("gate_id") or "") == "sd15_lora_512"
            ]
        if not related_gate_rows and input_id == "manual_gpu_evidence":
            related_gate_rows = gate_rows
        kind = _release_gate_input_kind(input_id)
        missing = (
            input_id in missing_external_inputs
            or (
                input_id == "manual_gpu_evidence"
                and bool(manual_blocking.get("manual_gpu_evidence_required", True))
            )
            or input_id == "source_cache_axis_refresh"
        )
        rows.append(
            {
                "input_id": input_id,
                "input_kind": kind,
                "dependency_status": "missing_or_manual_required" if missing else "declared_required",
                "related_gate_ids": _unique(
                    [str(gate_row.get("gate_id") or "") for gate_row in related_gate_rows]
                )[:20],
                "related_action_ids": _unique(
                    [
                        action_id
                        for gate_row in related_gate_rows
                        for action_id in _strings(gate_row.get("related_action_ids"))
                    ]
                )[:50],
                "affected_family_ids": _unique(
                    [
                        family_id
                        for gate_row in related_gate_rows
                        for family_id in _strings(gate_row.get("related_family_ids"))
                    ]
                )[:20],
                "missing": missing,
                "requires_external_input": kind in {
                    "external_checkpoint",
                    "external_source_or_cache_axis",
                },
                "requires_manual_gpu": kind == "manual_gpu_evidence",
                "requires_source_cache_refresh": kind == "source_cache_json_refresh",
                "json_only_resolution_available": False,
                "terminal_guard_required_after_input": True,
                "release_claim_allowed_after_input": False,
                "safe_to_auto_start": False,
                "not_release_evidence": True,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("json_only_resolution_available"))
        or bool(row.get("release_claim_allowed_after_input"))
        or bool(row.get("safe_to_auto_start"))
        or not bool(row.get("not_release_evidence"))
    ]
    missing_input_ids = [str(row.get("input_id") or "") for row in rows if bool(row.get("missing"))]
    return {
        "release_hard_gate_count": len(hard_gate_ids),
        "release_hard_gate_ids": hard_gate_ids[:20],
        "dependency_row_count": len(rows),
        "required_input_ids": required_input_ids[:50],
        "missing_input_count": len(missing_input_ids),
        "missing_input_ids": _unique(missing_input_ids)[:50],
        "external_input_dependency_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "manual_gpu_dependency_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "source_cache_refresh_dependency_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "json_only_resolution_available_count": sum(
            1 for row in rows if bool(row.get("json_only_resolution_available"))
        ),
        "unsafe_input_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_post_input_refresh_plan_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _release_gate_input_dependency_from_readiness(readiness)
    release_unblocker = _mapping(readiness.get("release_unblocker_summary"))
    input_resolution = _mapping(release_unblocker.get("input_resolution_summary"))
    refresh_commands = _post_input_refresh_sequence(input_resolution.get("next_json_refresh_sequence"))
    terminal_guard_commands = [
        command_id
        for command_id in refresh_commands
        if command_id
        in {"refresh_gpu_bubble_terminal_self_check", "run_gpu_bubble_release_readiness_guard"}
    ]
    downstream_artifacts = [
        "gpu_bubble_experiment_readiness_next_actions",
        "gpu_bubble_readiness_terminal_self_check",
        "gpu_bubble_release_readiness_guard_report",
    ]
    rows: list[dict[str, Any]] = []
    for dependency_row in (_mapping(row) for row in _list(dependency.get("rows"))):
        input_id = str(dependency_row.get("input_id") or "")
        if not input_id:
            continue
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(dependency_row.get("input_kind") or ""),
                "plan_status": "blocked_until_input_available"
                if bool(dependency_row.get("missing"))
                else "json_refresh_required_after_input",
                "related_gate_ids": _strings(dependency_row.get("related_gate_ids"))[:20],
                "related_action_ids": _strings(dependency_row.get("related_action_ids"))[:50],
                "affected_family_ids": _strings(dependency_row.get("affected_family_ids"))[:20],
                "input_missing": bool(dependency_row.get("missing")),
                "external_input_required_before_refresh": bool(
                    dependency_row.get("requires_external_input")
                ),
                "manual_gpu_evidence_required_before_refresh": bool(
                    dependency_row.get("requires_manual_gpu")
                ),
                "source_cache_refresh_input": bool(
                    dependency_row.get("requires_source_cache_refresh")
                ),
                "required_refresh_command_ids": refresh_commands[:20],
                "terminal_guard_command_ids": terminal_guard_commands[:10],
                "post_refresh_required_artifact_ids": downstream_artifacts,
                "terminal_guard_required_after_refresh": True,
                "safe_to_auto_start_refresh": False,
                "release_claim_allowed_after_refresh": False,
                "not_release_evidence": True,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start_refresh"))
        or bool(row.get("release_claim_allowed_after_refresh"))
        or not bool(row.get("not_release_evidence"))
        or "refresh_gpu_bubble_readiness_next_actions"
        not in _strings(row.get("required_refresh_command_ids"))
        or "refresh_gpu_bubble_terminal_self_check"
        not in _strings(row.get("required_refresh_command_ids"))
        or "run_gpu_bubble_release_readiness_guard"
        not in _strings(row.get("required_refresh_command_ids"))
    ]
    return {
        "plan_row_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "blocked_input_count": sum(1 for row in rows if bool(row.get("input_missing"))),
        "blocked_input_ids": [
            str(row.get("input_id") or "") for row in rows if bool(row.get("input_missing"))
        ][:50],
        "external_input_plan_count": sum(
            1 for row in rows if bool(row.get("external_input_required_before_refresh"))
        ),
        "manual_gpu_evidence_plan_count": sum(
            1 for row in rows if bool(row.get("manual_gpu_evidence_required_before_refresh"))
        ),
        "source_cache_refresh_plan_count": sum(
            1 for row in rows if bool(row.get("source_cache_refresh_input"))
        ),
        "required_refresh_command_ids": refresh_commands[:20],
        "terminal_guard_command_ids": terminal_guard_commands[:10],
        "post_refresh_required_artifact_ids": downstream_artifacts,
        "unsafe_plan_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_input_detection_source_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _release_gate_input_dependency_from_readiness(readiness)
    release_unblocker = _mapping(readiness.get("release_unblocker_summary"))
    input_resolution = _mapping(release_unblocker.get("input_resolution_summary"))
    manual_blocking = _mapping(release_unblocker.get("manual_evidence_blocking_summary"))
    missing_inputs = set(_strings(release_unblocker.get("missing_external_inputs")))
    detector_map = {
        "sd15_checkpoint": [
            "external_input_intake_registry",
            "external_input_handoff_packet",
            "sd15_lora512_release_gap_readiness",
        ],
        "new_source_root": [
            "external_input_intake_registry",
            "source_axis_freshness_dedupe_audit",
            "source_cache_axis_identity_registry",
            "source_cache_axis_pipeline_readiness",
        ],
        "warm_cache_axis": [
            "newbie_warm_cache_inventory",
            "source_cache_axis_admission_preflight",
            "source_cache_axis_pipeline_readiness",
        ],
        "caption_repair_axis": [
            "p60_source_axis_requirement",
            "source_axis_freshness_dedupe_audit",
            "source_cache_axis_admission_preflight",
        ],
        "manual_gpu_evidence": [
            "remaining_blocker_artifact_presence_summary",
            "post_manual_evidence_rebuild_plan",
            "manual_protected_gpu_command_surface_summary",
        ],
        "source_cache_axis_refresh": [
            "source_cache_axis_pipeline_readiness",
            "source_cache_axis_admission_preflight",
            "source_cache_axis_manual_canary_plan",
        ],
    }
    refresh_map = {
        "sd15_checkpoint": [
            "refresh_external_input_intake_registry",
            "refresh_gpu_bubble_readiness_next_actions",
        ],
        "new_source_root": [
            "refresh_external_input_intake_registry",
            "refresh_source_axis_freshness_dedupe_audit",
            "refresh_source_cache_axis_identity_registry",
            "refresh_source_cache_axis_pipeline_readiness",
        ],
        "warm_cache_axis": [
            "refresh_source_cache_axis_identity_registry",
            "refresh_source_cache_axis_pipeline_readiness",
        ],
        "caption_repair_axis": [
            "refresh_source_axis_freshness_dedupe_audit",
            "refresh_source_cache_axis_pipeline_readiness",
        ],
        "manual_gpu_evidence": [
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ],
        "source_cache_axis_refresh": [
            "refresh_source_cache_axis_pipeline_readiness",
            "refresh_gpu_bubble_readiness_next_actions",
        ],
    }
    rows: list[dict[str, Any]] = []
    for dependency_row in (_mapping(row) for row in _list(dependency.get("rows"))):
        input_id = str(dependency_row.get("input_id") or "")
        if not input_id:
            continue
        if input_id == "manual_gpu_evidence":
            detected = not bool(manual_blocking.get("manual_gpu_evidence_required", True))
        elif input_id == "source_cache_axis_refresh":
            detected = False
        elif input_id == "sd15_checkpoint":
            detected = bool(input_resolution.get("sd15_checkpoint_exists"))
        elif input_id == "new_source_root":
            detected = _safe_int(input_resolution.get("new_source_root_count")) > 0
        else:
            detected = input_id not in missing_inputs and not bool(dependency_row.get("missing"))
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(dependency_row.get("input_kind") or ""),
                "detection_status": "detected_refresh_required" if detected else "missing_or_unverified",
                "detector_artifact_ids": detector_map.get(input_id, ["release_unblocker_summary"]),
                "required_refresh_command_ids": refresh_map.get(
                    input_id,
                    ["refresh_gpu_bubble_readiness_next_actions"],
                ),
                "related_gate_ids": _strings(dependency_row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(dependency_row.get("affected_family_ids"))[:20],
                "input_missing": bool(dependency_row.get("missing")),
                "detected": detected,
                "requires_external_input": bool(dependency_row.get("requires_external_input")),
                "requires_manual_gpu": bool(dependency_row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    dependency_row.get("requires_source_cache_refresh")
                ),
                "terminal_guard_required_after_detection": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_detection": False,
                "not_release_evidence": True,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed_after_detection"))
        or not bool(row.get("not_release_evidence"))
        or not _strings(row.get("detector_artifact_ids"))
        or not _strings(row.get("required_refresh_command_ids"))
    ]
    return {
        "detection_row_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "missing_or_unverified_input_count": sum(
            1 for row in rows if str(row.get("detection_status") or "") == "missing_or_unverified"
        ),
        "missing_or_unverified_input_ids": [
            str(row.get("input_id") or "")
            for row in rows
            if str(row.get("detection_status") or "") == "missing_or_unverified"
        ][:50],
        "detected_input_count": sum(1 for row in rows if bool(row.get("detected"))),
        "external_input_detector_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "manual_gpu_detector_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "source_cache_refresh_detector_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "unsafe_detector_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_input_acceptance_criteria_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _release_gate_input_dependency_from_readiness(readiness)
    detection = _release_gate_input_detection_source_from_readiness(readiness)
    dependency_by_input = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(dependency.get("rows"))
    }
    detection_by_input = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(detection.get("rows"))
    }
    criteria_map = {
        "sd15_checkpoint": [
            "checkpoint_file_detected",
            "sd15_release_gap_readiness_refreshed",
            "terminal_and_guard_refreshed_after_detection",
        ],
        "new_source_root": [
            "new_source_root_detected",
            "source_axis_identity_registered",
            "source_cache_pipeline_refreshed",
            "terminal_and_guard_refreshed_after_detection",
        ],
        "warm_cache_axis": [
            "cache_ready_axis_detected",
            "warm_cache_inventory_refreshed",
            "source_cache_preflight_refreshed",
        ],
        "caption_repair_axis": [
            "caption_repair_axis_detected",
            "source_axis_freshness_refreshed",
            "source_cache_preflight_refreshed",
        ],
        "manual_gpu_evidence": [
            "manual_gpu_evidence_artifacts_present",
            "post_manual_rebuild_plan_refreshed",
            "terminal_and_guard_refreshed_after_evidence",
        ],
        "source_cache_axis_refresh": [
            "source_cache_preflight_admitted",
            "manual_canary_plan_refreshed",
            "readiness_terminal_guard_refreshed",
        ],
    }
    evidence_map = {
        "sd15_checkpoint": [
            "external_input_intake_registry",
            "sd15_lora512_release_gap_readiness",
        ],
        "new_source_root": [
            "source_axis_freshness_dedupe_audit",
            "source_cache_axis_identity_registry",
            "source_cache_axis_pipeline_readiness",
        ],
        "warm_cache_axis": [
            "newbie_warm_cache_inventory",
            "source_cache_axis_admission_preflight",
        ],
        "caption_repair_axis": [
            "p60_source_axis_requirement",
            "source_axis_freshness_dedupe_audit",
            "source_cache_axis_admission_preflight",
        ],
        "manual_gpu_evidence": [
            "remaining_blocker_artifact_presence_summary",
            "post_manual_evidence_rebuild_plan",
            "release_claim_exit_criteria_summary",
        ],
        "source_cache_axis_refresh": [
            "source_cache_axis_pipeline_readiness",
            "source_cache_axis_admission_preflight",
            "source_cache_axis_manual_canary_plan",
        ],
    }
    rows: list[dict[str, Any]] = []
    for input_id, dependency_row in dependency_by_input.items():
        detection_row = detection_by_input.get(input_id, {})
        detected = bool(detection_row.get("detected"))
        missing = bool(dependency_row.get("missing"))
        accepted = detected and not missing
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(dependency_row.get("input_kind") or ""),
                "acceptance_status": "accepted_pending_refresh" if accepted else "not_satisfied",
                "acceptance_criteria_ids": criteria_map.get(input_id, ["input_detected_and_refreshed"]),
                "required_evidence_artifact_ids": evidence_map.get(input_id, ["release_unblocker_summary"]),
                "detector_artifact_ids": _strings(detection_row.get("detector_artifact_ids"))[:20],
                "required_refresh_command_ids": _strings(detection_row.get("required_refresh_command_ids"))[:20],
                "related_gate_ids": _strings(dependency_row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(dependency_row.get("affected_family_ids"))[:20],
                "input_missing": missing,
                "detected": detected,
                "accepted": accepted,
                "requires_external_input": bool(dependency_row.get("requires_external_input")),
                "requires_manual_gpu": bool(dependency_row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    dependency_row.get("requires_source_cache_refresh")
                ),
                "terminal_guard_required_after_acceptance": True,
                "release_claim_allowed_after_acceptance": False,
                "safe_to_auto_start": False,
                "not_release_evidence": True,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("release_claim_allowed_after_acceptance"))
        or bool(row.get("safe_to_auto_start"))
        or not bool(row.get("not_release_evidence"))
        or not _strings(row.get("acceptance_criteria_ids"))
        or not _strings(row.get("required_evidence_artifact_ids"))
    ]
    unsatisfied_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if str(row.get("acceptance_status") or "") != "accepted_pending_refresh"
    ]
    return {
        "acceptance_row_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "accepted_input_count": sum(1 for row in rows if bool(row.get("accepted"))),
        "accepted_input_ids": [
            str(row.get("input_id") or "") for row in rows if bool(row.get("accepted"))
        ][:50],
        "unsatisfied_input_count": len(unsatisfied_input_ids),
        "unsatisfied_input_ids": _unique(unsatisfied_input_ids)[:50],
        "external_input_acceptance_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "manual_gpu_acceptance_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "source_cache_refresh_acceptance_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "unsafe_acceptance_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_input_refresh_readiness_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = _release_gate_input_acceptance_criteria_from_readiness(readiness)
    refresh_plan = _release_gate_post_input_refresh_plan_from_readiness(readiness)
    refresh_plan_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(refresh_plan.get("rows"))
    }
    rows: list[dict[str, Any]] = []
    for acceptance_row in (_mapping(row) for row in _list(acceptance.get("rows"))):
        input_id = str(acceptance_row.get("input_id") or "")
        if not input_id:
            continue
        plan_row = refresh_plan_rows.get(input_id, {})
        accepted = bool(acceptance_row.get("accepted"))
        unsafe = (
            bool(acceptance_row.get("safe_to_auto_start"))
            or bool(acceptance_row.get("release_claim_allowed_after_acceptance"))
            or not bool(acceptance_row.get("not_release_evidence"))
            or bool(plan_row.get("safe_to_auto_start_refresh"))
            or bool(plan_row.get("release_claim_allowed_after_refresh"))
            or not bool(plan_row.get("not_release_evidence"))
        )
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(acceptance_row.get("input_kind") or ""),
                "refresh_readiness_status": (
                    "refresh_ready_after_accepted_input"
                    if accepted and not unsafe
                    else "blocked_waiting_input_acceptance"
                ),
                "accepted": accepted,
                "input_missing": bool(acceptance_row.get("input_missing")),
                "detected": bool(acceptance_row.get("detected")),
                "refresh_ready": accepted and not unsafe,
                "blocked_refresh": not accepted or unsafe,
                "acceptance_status": str(acceptance_row.get("acceptance_status") or ""),
                "plan_status": str(plan_row.get("plan_status") or ""),
                "acceptance_criteria_ids": _strings(
                    acceptance_row.get("acceptance_criteria_ids")
                )[:20],
                "required_evidence_artifact_ids": _strings(
                    acceptance_row.get("required_evidence_artifact_ids")
                )[:20],
                "required_refresh_command_ids": _strings(
                    plan_row.get("required_refresh_command_ids")
                    or acceptance_row.get("required_refresh_command_ids")
                )[:20],
                "terminal_guard_command_ids": _strings(
                    plan_row.get("terminal_guard_command_ids")
                )[:10],
                "post_refresh_required_artifact_ids": _strings(
                    plan_row.get("post_refresh_required_artifact_ids")
                )[:20],
                "related_gate_ids": _strings(acceptance_row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(acceptance_row.get("affected_family_ids"))[:20],
                "requires_external_input": bool(acceptance_row.get("requires_external_input")),
                "requires_manual_gpu": bool(acceptance_row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    acceptance_row.get("requires_source_cache_refresh")
                ),
                "terminal_guard_required_after_refresh": True,
                "safe_to_auto_start_refresh": False,
                "release_claim_allowed_after_refresh": False,
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    blocked_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("blocked_refresh"))
    ]
    refresh_ready_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("refresh_ready"))
    ]
    return {
        "refresh_row_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "accepted_input_count": sum(1 for row in rows if bool(row.get("accepted"))),
        "accepted_input_ids": [
            str(row.get("input_id") or "") for row in rows if bool(row.get("accepted"))
        ][:50],
        "refresh_ready_input_count": len(refresh_ready_input_ids),
        "refresh_ready_input_ids": _unique(refresh_ready_input_ids)[:50],
        "blocked_refresh_input_count": len(blocked_input_ids),
        "blocked_refresh_input_ids": _unique(blocked_input_ids)[:50],
        "external_input_refresh_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "manual_gpu_refresh_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "source_cache_refresh_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "unsafe_refresh_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_input_refresh_blocker_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    refresh_readiness = _release_gate_input_refresh_readiness_from_readiness(readiness)
    rows: list[dict[str, Any]] = []
    for refresh_row in (_mapping(row) for row in _list(refresh_readiness.get("rows"))):
        input_id = str(refresh_row.get("input_id") or "")
        if not input_id:
            continue
        blocked_reason_ids: list[str] = []
        if bool(refresh_row.get("input_missing")):
            blocked_reason_ids.append("input_missing")
        if not bool(refresh_row.get("detected")):
            blocked_reason_ids.append("input_not_detected")
        if not bool(refresh_row.get("accepted")):
            blocked_reason_ids.append("input_not_accepted")
        if not bool(refresh_row.get("refresh_ready")):
            blocked_reason_ids.append("post_input_refresh_not_ready")
        if bool(refresh_row.get("requires_external_input")):
            blocked_reason_ids.append("external_input_required")
        if bool(refresh_row.get("requires_manual_gpu")):
            blocked_reason_ids.append("manual_gpu_evidence_required")
        if bool(refresh_row.get("requires_source_cache_refresh")):
            blocked_reason_ids.append("source_cache_refresh_required")
        if bool(refresh_row.get("terminal_guard_required_after_refresh")):
            blocked_reason_ids.append("terminal_guard_required_after_refresh")
        unsafe = (
            bool(refresh_row.get("unsafe"))
            or bool(refresh_row.get("safe_to_auto_start"))
            or bool(refresh_row.get("release_claim_allowed"))
            or bool(refresh_row.get("safe_to_auto_start_refresh"))
            or bool(refresh_row.get("release_claim_allowed_after_refresh"))
            or not bool(refresh_row.get("not_release_evidence"))
        )
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(refresh_row.get("input_kind") or ""),
                "blocker_status": (
                    "blocked_waiting_input_acceptance"
                    if bool(refresh_row.get("blocked_refresh"))
                    else "no_refresh_blocker"
                ),
                "blocked_reason_ids": _unique(blocked_reason_ids)[:20],
                "blocked_refresh": bool(refresh_row.get("blocked_refresh")),
                "refresh_ready": bool(refresh_row.get("refresh_ready")),
                "accepted": bool(refresh_row.get("accepted")),
                "detected": bool(refresh_row.get("detected")),
                "input_missing": bool(refresh_row.get("input_missing")),
                "requires_external_input": bool(refresh_row.get("requires_external_input")),
                "requires_manual_gpu": bool(refresh_row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    refresh_row.get("requires_source_cache_refresh")
                ),
                "required_refresh_command_ids": _strings(
                    refresh_row.get("required_refresh_command_ids")
                )[:20],
                "terminal_guard_command_ids": _strings(
                    refresh_row.get("terminal_guard_command_ids")
                )[:10],
                "related_gate_ids": _strings(refresh_row.get("related_gate_ids"))[:20],
                "affected_family_ids": _strings(refresh_row.get("affected_family_ids"))[:20],
                "terminal_guard_required_after_refresh": bool(
                    refresh_row.get("terminal_guard_required_after_refresh")
                ),
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    unsafe_input_ids = [
        str(row.get("input_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    blocked_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("blocked_refresh"))
    ]
    return {
        "blocker_row_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "blocked_input_count": len(blocked_input_ids),
        "blocked_input_ids": _unique(blocked_input_ids)[:50],
        "refresh_ready_input_count": sum(1 for row in rows if bool(row.get("refresh_ready"))),
        "refresh_ready_input_ids": [
            str(row.get("input_id") or "") for row in rows if bool(row.get("refresh_ready"))
        ][:50],
        "missing_input_blocker_count": sum(1 for row in rows if bool(row.get("input_missing"))),
        "undetected_input_blocker_count": sum(1 for row in rows if not bool(row.get("detected"))),
        "unaccepted_input_blocker_count": sum(1 for row in rows if not bool(row.get("accepted"))),
        "external_input_blocker_count": sum(1 for row in rows if bool(row.get("requires_external_input"))),
        "manual_gpu_blocker_count": sum(1 for row in rows if bool(row.get("requires_manual_gpu"))),
        "source_cache_refresh_blocker_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "terminal_guard_required_count": sum(
            1 for row in rows if bool(row.get("terminal_guard_required_after_refresh"))
        ),
        "unsafe_blocker_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _release_gate_input_lifecycle_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _release_gate_input_dependency_from_readiness(readiness)
    detection = _release_gate_input_detection_source_from_readiness(readiness)
    acceptance = _release_gate_input_acceptance_criteria_from_readiness(readiness)
    refresh = _release_gate_input_refresh_readiness_from_readiness(readiness)
    blocker = _release_gate_input_refresh_blocker_from_readiness(readiness)
    dependency_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(dependency.get("rows"))
    }
    detection_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(detection.get("rows"))
    }
    acceptance_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(acceptance.get("rows"))
    }
    refresh_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(refresh.get("rows"))
    }
    blocker_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(blocker.get("rows"))
    }
    input_ids = _unique(
        [
            *_strings(dependency.get("required_input_ids")),
            *_strings(detection.get("input_ids")),
            *_strings(acceptance.get("input_ids")),
            *_strings(refresh.get("input_ids")),
            *_strings(blocker.get("input_ids")),
        ]
    )
    rows: list[dict[str, Any]] = []
    for input_id in input_ids:
        dependency_row = dependency_rows.get(input_id, {})
        detection_row = detection_rows.get(input_id, {})
        acceptance_row = acceptance_rows.get(input_id, {})
        refresh_row = refresh_rows.get(input_id, {})
        blocker_row = blocker_rows.get(input_id, {})
        detected = bool(detection_row.get("detected"))
        accepted = bool(acceptance_row.get("accepted"))
        refresh_ready = bool(refresh_row.get("refresh_ready"))
        blocked_refresh = bool(
            blocker_row.get("blocked_refresh", refresh_row.get("blocked_refresh"))
        )
        if refresh_ready:
            lifecycle_stage = "refresh_ready"
        elif accepted:
            lifecycle_stage = "accepted_pending_refresh"
        elif detected:
            lifecycle_stage = "detected_waiting_acceptance"
        else:
            lifecycle_stage = "waiting_detection"
        unsafe = (
            bool(dependency_row.get("safe_to_auto_start"))
            or bool(detection_row.get("safe_to_auto_start"))
            or bool(acceptance_row.get("safe_to_auto_start"))
            or bool(refresh_row.get("safe_to_auto_start"))
            or bool(blocker_row.get("safe_to_auto_start"))
            or bool(dependency_row.get("release_claim_allowed_after_input"))
            or bool(detection_row.get("release_claim_allowed_after_detection"))
            or bool(acceptance_row.get("release_claim_allowed_after_acceptance"))
            or bool(refresh_row.get("release_claim_allowed"))
            or bool(blocker_row.get("release_claim_allowed"))
            or not bool(dependency_row.get("not_release_evidence", True))
            or not bool(detection_row.get("not_release_evidence", True))
            or not bool(acceptance_row.get("not_release_evidence", True))
            or not bool(refresh_row.get("not_release_evidence", True))
            or not bool(blocker_row.get("not_release_evidence", True))
        )
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(
                    dependency_row.get("input_kind")
                    or refresh_row.get("input_kind")
                    or ""
                ),
                "lifecycle_stage": lifecycle_stage,
                "dependency_status": str(
                    dependency_row.get("dependency_status") or ""
                ),
                "detection_status": str(detection_row.get("detection_status") or ""),
                "acceptance_status": str(
                    acceptance_row.get("acceptance_status") or ""
                ),
                "refresh_readiness_status": str(
                    refresh_row.get("refresh_readiness_status") or ""
                ),
                "blocker_status": str(blocker_row.get("blocker_status") or ""),
                "missing": bool(dependency_row.get("missing")),
                "detected": detected,
                "accepted": accepted,
                "refresh_ready": refresh_ready,
                "blocked_refresh": blocked_refresh,
                "blocked_reason_ids": _strings(
                    blocker_row.get("blocked_reason_ids")
                )[:20],
                "required_refresh_command_ids": _strings(
                    refresh_row.get("required_refresh_command_ids")
                    or blocker_row.get("required_refresh_command_ids")
                    or detection_row.get("required_refresh_command_ids")
                )[:20],
                "terminal_guard_command_ids": _strings(
                    refresh_row.get("terminal_guard_command_ids")
                    or blocker_row.get("terminal_guard_command_ids")
                )[:10],
                "related_gate_ids": _strings(
                    dependency_row.get("related_gate_ids")
                    or refresh_row.get("related_gate_ids")
                    or blocker_row.get("related_gate_ids")
                )[:20],
                "affected_family_ids": _strings(
                    dependency_row.get("affected_family_ids")
                    or refresh_row.get("affected_family_ids")
                    or blocker_row.get("affected_family_ids")
                )[:20],
                "requires_external_input": bool(
                    dependency_row.get("requires_external_input")
                ),
                "requires_manual_gpu": bool(dependency_row.get("requires_manual_gpu")),
                "requires_source_cache_refresh": bool(
                    dependency_row.get("requires_source_cache_refresh")
                ),
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    unsafe_input_ids = [str(row.get("input_id") or "") for row in rows if row["unsafe"]]
    blocked_input_ids = [
        str(row.get("input_id") or "") for row in rows if row["blocked_refresh"]
    ]
    detected_unaccepted_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if row["detected"] and not row["accepted"]
    ]
    accepted_pending_refresh_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if row["accepted"] and not row["refresh_ready"]
    ]
    refresh_ready_ids = [
        str(row.get("input_id") or "") for row in rows if row["refresh_ready"]
    ]
    stage_counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("lifecycle_stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    if refresh_ready_ids and not blocked_input_ids:
        lifecycle_status = "ready_for_post_input_json_refresh"
    elif accepted_pending_refresh_ids:
        lifecycle_status = "accepted_pending_post_input_refresh"
    elif detected_unaccepted_ids:
        lifecycle_status = "detected_waiting_acceptance"
    else:
        lifecycle_status = "waiting_for_input_detection"
    return {
        "lifecycle_status": lifecycle_status,
        "input_count": len(rows),
        "input_ids": [str(row.get("input_id") or "") for row in rows],
        "lifecycle_stage_counts": dict(sorted(stage_counts.items())),
        "detected_input_count": sum(1 for row in rows if row["detected"]),
        "detected_input_ids": [
            str(row.get("input_id") or "") for row in rows if row["detected"]
        ][:50],
        "detected_unaccepted_input_count": len(detected_unaccepted_ids),
        "detected_unaccepted_input_ids": detected_unaccepted_ids[:50],
        "accepted_input_count": sum(1 for row in rows if row["accepted"]),
        "accepted_input_ids": [
            str(row.get("input_id") or "") for row in rows if row["accepted"]
        ][:50],
        "accepted_pending_refresh_input_count": len(accepted_pending_refresh_ids),
        "accepted_pending_refresh_input_ids": accepted_pending_refresh_ids[:50],
        "refresh_ready_input_count": len(refresh_ready_ids),
        "refresh_ready_input_ids": refresh_ready_ids[:50],
        "blocked_input_count": len(blocked_input_ids),
        "blocked_input_ids": blocked_input_ids[:50],
        "external_input_count": sum(
            1 for row in rows if row["requires_external_input"]
        ),
        "manual_gpu_input_count": sum(1 for row in rows if row["requires_manual_gpu"]),
        "source_cache_refresh_input_count": sum(
            1 for row in rows if row["requires_source_cache_refresh"]
        ),
        "unsafe_input_count": len(unsafe_input_ids),
        "unsafe_input_ids": unsafe_input_ids[:50],
        "rows": rows,
        "fail_closed": len(unsafe_input_ids) == 0,
    }


def _external_input_release_gate_alignment_from_readiness(
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    transition = _mapping(readiness.get("external_input_transition_table")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("external_input_transition_table")
    )
    lifecycle = _release_gate_input_lifecycle_from_readiness(readiness)
    dependency = _release_gate_input_dependency_from_readiness(readiness)
    transition_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(transition.get("rows"))
    }
    dependency_rows = {
        str(row.get("input_id") or ""): _mapping(row)
        for row in _list(dependency.get("rows"))
    }
    lifecycle_rows = [_mapping(row) for row in _list(lifecycle.get("rows"))]
    external_input_ids = _strings(transition.get("missing_external_inputs")) or [
        input_id for input_id, row in transition_rows.items() if bool(row.get("required"))
    ]
    release_gate_input_ids = _strings(lifecycle.get("input_ids")) or [
        str(row.get("input_id") or "") for row in lifecycle_rows
    ]
    external_set = set(external_input_ids)
    release_set = set(release_gate_input_ids)
    rows: list[dict[str, Any]] = []
    for lifecycle_row in lifecycle_rows:
        input_id = str(lifecycle_row.get("input_id") or "")
        if not input_id:
            continue
        dependency_row = dependency_rows.get(input_id, {})
        transition_row = transition_rows.get(input_id, {})
        requires_external = bool(
            lifecycle_row.get("requires_external_input")
            or dependency_row.get("requires_external_input")
        )
        requires_manual_gpu = bool(
            lifecycle_row.get("requires_manual_gpu")
            or dependency_row.get("requires_manual_gpu")
        )
        requires_source_cache_refresh = bool(
            lifecycle_row.get("requires_source_cache_refresh")
            or dependency_row.get("requires_source_cache_refresh")
        )
        if requires_external:
            alignment_kind = "external_input"
        elif requires_manual_gpu:
            alignment_kind = "manual_gpu_evidence"
        elif requires_source_cache_refresh:
            alignment_kind = "source_cache_refresh"
        else:
            alignment_kind = "release_gate_internal"
        transition_present = input_id in transition_rows
        expected_transition_present = requires_external
        mismatch = (
            (requires_external and not transition_present)
            or (not requires_external and transition_present)
            or (requires_external and input_id not in external_set)
        )
        unsafe = (
            mismatch
            or bool(lifecycle_row.get("safe_to_auto_start"))
            or bool(lifecycle_row.get("release_claim_allowed"))
            or not bool(lifecycle_row.get("not_release_evidence", True))
            or bool(transition_row.get("safe_to_auto_start"))
            or bool(transition_row.get("release_claim_allowed"))
            or bool(transition_row.get("release_claim_allowed_after_success"))
            or (
                bool(transition_row)
                and not bool(transition_row.get("not_release_evidence", True))
            )
        )
        rows.append(
            {
                "input_id": input_id,
                "input_kind": str(lifecycle_row.get("input_kind") or ""),
                "alignment_kind": alignment_kind,
                "requires_external_input": requires_external,
                "requires_manual_gpu": requires_manual_gpu,
                "requires_source_cache_refresh": requires_source_cache_refresh,
                "in_external_transition_table": transition_present,
                "expected_in_external_transition_table": expected_transition_present,
                "external_input_missing": input_id in external_set,
                "release_gate_input_present": input_id in release_set,
                "lifecycle_stage": str(lifecycle_row.get("lifecycle_stage") or ""),
                "transition_state": str(transition_row.get("transition_state") or ""),
                "missing": bool(lifecycle_row.get("missing")),
                "detected": bool(lifecycle_row.get("detected")),
                "accepted": bool(lifecycle_row.get("accepted")),
                "blocked_refresh": bool(lifecycle_row.get("blocked_refresh")),
                "related_gate_ids": _strings(lifecycle_row.get("related_gate_ids"))[:20],
                "blocked_reason_ids": _strings(
                    lifecycle_row.get("blocked_reason_ids")
                )[:20],
                "handoff_step_id": str(transition_row.get("handoff_step_id") or ""),
                "replay_command_ids": _strings(
                    transition_row.get("replay_command_ids")
                )[:20],
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    external_missing_from_release = [
        input_id for input_id in external_input_ids if input_id not in release_set
    ]
    release_external_missing_from_transition = [
        str(row.get("input_id") or "")
        for row in rows
        if bool(row.get("requires_external_input"))
        and not bool(row.get("in_external_transition_table"))
    ]
    non_external_release_gate_input_ids = [
        str(row.get("input_id") or "")
        for row in rows
        if not bool(row.get("requires_external_input"))
    ]
    unsafe_ids = [str(row.get("input_id") or "") for row in rows if bool(row.get("unsafe"))]
    alignment_ok = (
        not external_missing_from_release
        and not release_external_missing_from_transition
        and not unsafe_ids
    )
    return {
        "alignment_status": (
            "external_inputs_aligned_with_release_gate_inputs"
            if alignment_ok
            else "external_input_release_gate_alignment_drift_detected"
        ),
        "alignment_ok": alignment_ok,
        "external_input_count": len(external_input_ids),
        "external_input_ids": external_input_ids[:50],
        "release_gate_input_count": len(release_gate_input_ids),
        "release_gate_input_ids": release_gate_input_ids[:50],
        "external_release_gate_input_count": sum(
            1 for row in rows if bool(row.get("requires_external_input"))
        ),
        "manual_gpu_release_gate_input_count": sum(
            1 for row in rows if bool(row.get("requires_manual_gpu"))
        ),
        "source_cache_refresh_release_gate_input_count": sum(
            1 for row in rows if bool(row.get("requires_source_cache_refresh"))
        ),
        "non_external_release_gate_input_count": len(non_external_release_gate_input_ids),
        "non_external_release_gate_input_ids": non_external_release_gate_input_ids[:50],
        "external_missing_from_release_gate_count": len(external_missing_from_release),
        "external_missing_from_release_gate_ids": external_missing_from_release[:50],
        "release_external_missing_from_transition_count": len(
            release_external_missing_from_transition
        ),
        "release_external_missing_from_transition_ids": (
            release_external_missing_from_transition[:50]
        ),
        "blocked_input_count": sum(1 for row in rows if bool(row.get("blocked_refresh"))),
        "unsafe_alignment_count": len(unsafe_ids),
        "unsafe_input_ids": unsafe_ids[:50],
        "rows": rows,
        "fail_closed": alignment_ok,
    }


def _release_gate_post_input_refresh_command_surface_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    refresh_plan = _release_gate_post_input_refresh_plan_from_readiness(readiness)
    refresh_blocker = _release_gate_input_refresh_blocker_from_readiness(readiness)
    required_command_ids = _strings(refresh_plan.get("required_refresh_command_ids"))
    terminal_guard_command_ids = _strings(refresh_plan.get("terminal_guard_command_ids"))
    blocked_input_ids = _strings(refresh_blocker.get("blocked_input_ids"))
    plan_rows = [_mapping(row) for row in _list(refresh_plan.get("rows"))]
    rows: list[dict[str, Any]] = []
    for index, command_id in enumerate(required_command_ids, start=1):
        related_input_ids = [
            str(row.get("input_id") or "")
            for row in plan_rows
            if command_id in _strings(row.get("required_refresh_command_ids"))
        ]
        terminal_guard_command = command_id in terminal_guard_command_ids
        blocked = bool(blocked_input_ids)
        rows.append(
            {
                "command_id": command_id,
                "command_order": index,
                "command_kind": "terminal_guard" if terminal_guard_command else "json_refresh",
                "command_status": (
                    "blocked_waiting_input_acceptance"
                    if blocked
                    else "ready_after_inputs_accepted"
                ),
                "related_input_ids": _unique(related_input_ids)[:50],
                "blocked_input_ids": blocked_input_ids[:50],
                "blocked_input_count": len(blocked_input_ids),
                "refresh_ready_input_count": _safe_int(
                    refresh_blocker.get("refresh_ready_input_count")
                ),
                "terminal_guard_command": terminal_guard_command,
                "required_after_input_acceptance": True,
                "blocked_until_input_acceptance": blocked,
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": False,
            }
        )
    unsafe_command_ids = [
        str(row.get("command_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed"))
        or not bool(row.get("not_release_evidence"))
        or not str(row.get("command_id") or "")
    ]
    blocked_command_ids = [
        str(row.get("command_id") or "")
        for row in rows
        if bool(row.get("blocked_until_input_acceptance"))
    ]
    return {
        "command_row_count": len(rows),
        "required_command_count": len(required_command_ids),
        "required_command_ids": required_command_ids[:50],
        "json_refresh_command_count": sum(
            1 for row in rows if str(row.get("command_kind") or "") == "json_refresh"
        ),
        "terminal_guard_command_count": sum(
            1 for row in rows if bool(row.get("terminal_guard_command"))
        ),
        "blocked_command_count": len(blocked_command_ids),
        "blocked_command_ids": _unique(blocked_command_ids)[:50],
        "ready_command_count": sum(1 for row in rows if not bool(row.get("blocked_until_input_acceptance"))),
        "ready_command_ids": [
            str(row.get("command_id") or "")
            for row in rows
            if not bool(row.get("blocked_until_input_acceptance"))
        ][:50],
        "blocked_input_count": len(blocked_input_ids),
        "blocked_input_ids": blocked_input_ids[:50],
        "unsafe_command_count": len(unsafe_command_ids),
        "unsafe_command_ids": _unique(unsafe_command_ids)[:50],
        "rows": rows,
        "fail_closed": len(unsafe_command_ids) == 0,
    }


def _release_gate_post_input_refresh_sequence_integrity_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    command_surface = _release_gate_post_input_refresh_command_surface_from_readiness(readiness)
    rows = [_mapping(row) for row in _list(command_surface.get("rows"))]
    observed_command_ids = [str(row.get("command_id") or "") for row in rows]
    expected_command_ids = list(POST_INPUT_REFRESH_SEQUENCE)
    duplicate_command_ids = [
        command_id
        for command_id in _unique(observed_command_ids)
        if observed_command_ids.count(command_id) > 1
    ]
    missing_command_ids = [
        command_id for command_id in expected_command_ids if command_id not in observed_command_ids
    ]
    unexpected_command_ids = [
        command_id for command_id in observed_command_ids if command_id not in expected_command_ids
    ]
    terminal_guard_command_ids = [
        str(row.get("command_id") or "")
        for row in rows
        if bool(row.get("terminal_guard_command"))
    ]
    unsafe_command_ids = [
        str(row.get("command_id") or "")
        for row in rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed"))
        or not bool(row.get("not_release_evidence"))
        or bool(row.get("unsafe"))
    ]
    order_matches_expected = observed_command_ids == expected_command_ids
    terminal_guard_tail_ok = terminal_guard_command_ids == expected_command_ids[-2:]
    blocked_until_input_acceptance = all(
        bool(row.get("blocked_until_input_acceptance")) for row in rows
    )
    sequence_ok = (
        order_matches_expected
        and terminal_guard_tail_ok
        and blocked_until_input_acceptance
        and not missing_command_ids
        and not unexpected_command_ids
        and not duplicate_command_ids
        and not unsafe_command_ids
    )
    return {
        "sequence_ok": sequence_ok,
        "expected_command_count": len(expected_command_ids),
        "observed_command_count": len(observed_command_ids),
        "expected_command_ids": expected_command_ids,
        "observed_command_ids": observed_command_ids,
        "missing_command_count": len(missing_command_ids),
        "missing_command_ids": missing_command_ids,
        "unexpected_command_count": len(unexpected_command_ids),
        "unexpected_command_ids": unexpected_command_ids,
        "duplicate_command_count": len(duplicate_command_ids),
        "duplicate_command_ids": duplicate_command_ids,
        "order_matches_expected": order_matches_expected,
        "terminal_guard_tail_ok": terminal_guard_tail_ok,
        "terminal_guard_command_ids": terminal_guard_command_ids,
        "blocked_until_input_acceptance": blocked_until_input_acceptance,
        "blocked_command_count": _safe_int(command_surface.get("blocked_command_count")),
        "ready_command_count": _safe_int(command_surface.get("ready_command_count")),
        "unsafe_sequence_count": len(unsafe_command_ids),
        "unsafe_command_ids": _unique(unsafe_command_ids)[:50],
        "fail_closed": not unsafe_command_ids and sequence_ok,
    }


def _release_gate_post_input_refresh_terminal_guard_dependency_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    command_surface = _release_gate_post_input_refresh_command_surface_from_readiness(readiness)
    sequence_integrity = _release_gate_post_input_refresh_sequence_integrity_from_readiness(readiness)
    rows = [_mapping(row) for row in _list(command_surface.get("rows"))]
    terminal_guard_rows = [row for row in rows if bool(row.get("terminal_guard_command"))]
    terminal_guard_command_ids = [
        str(row.get("command_id") or "") for row in terminal_guard_rows
    ]
    terminal_guard_command_orders = [
        _safe_int(row.get("command_order")) for row in terminal_guard_rows
    ]
    expected_terminal_guard_command_ids = [
        "refresh_gpu_bubble_terminal_self_check",
        "run_gpu_bubble_release_readiness_guard",
    ]
    dependency_rows: list[dict[str, Any]] = []
    for index, row in enumerate(terminal_guard_rows, start=1):
        command_id = str(row.get("command_id") or "")
        guard_kind = (
            "terminal_self_check"
            if command_id == "refresh_gpu_bubble_terminal_self_check"
            else "release_readiness_guard"
        )
        unsafe = (
            bool(row.get("safe_to_auto_start"))
            or bool(row.get("release_claim_allowed"))
            or not bool(row.get("not_release_evidence"))
            or bool(row.get("unsafe"))
            or not bool(row.get("blocked_until_input_acceptance"))
        )
        dependency_rows.append(
            {
                "command_id": command_id,
                "dependency_order": index,
                "command_order": _safe_int(row.get("command_order")),
                "guard_kind": guard_kind,
                "depends_on_json_refresh_sequence": True,
                "required_after_json_refresh": True,
                "blocked_until_input_acceptance": bool(
                    row.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    unsafe_command_ids = [
        str(row.get("command_id") or "")
        for row in dependency_rows
        if bool(row.get("unsafe"))
    ]
    command_row_count = _safe_int(command_surface.get("command_row_count"))
    expected_terminal_guard_command_orders = [
        max(command_row_count - 1, 0),
        command_row_count,
    ]
    all_json_refresh_commands_before_terminal_guard = (
        terminal_guard_command_orders == expected_terminal_guard_command_orders
    )
    dependency_ok = (
        terminal_guard_command_ids == expected_terminal_guard_command_ids
        and bool(sequence_integrity.get("terminal_guard_tail_ok"))
        and all_json_refresh_commands_before_terminal_guard
        and bool(sequence_integrity.get("sequence_ok"))
        and bool(sequence_integrity.get("blocked_until_input_acceptance"))
        and not unsafe_command_ids
    )
    return {
        "dependency_ok": dependency_ok,
        "terminal_guard_required": True,
        "terminal_guard_command_count": len(terminal_guard_command_ids),
        "expected_terminal_guard_command_count": len(expected_terminal_guard_command_ids),
        "terminal_guard_command_ids": terminal_guard_command_ids,
        "expected_terminal_guard_command_ids": expected_terminal_guard_command_ids,
        "terminal_guard_command_orders": terminal_guard_command_orders,
        "expected_terminal_guard_command_orders": expected_terminal_guard_command_orders,
        "terminal_self_check_required": (
            "refresh_gpu_bubble_terminal_self_check" in terminal_guard_command_ids
        ),
        "release_guard_required": (
            "run_gpu_bubble_release_readiness_guard" in terminal_guard_command_ids
        ),
        "terminal_guard_tail_ok": bool(sequence_integrity.get("terminal_guard_tail_ok")),
        "all_json_refresh_commands_before_terminal_guard": (
            all_json_refresh_commands_before_terminal_guard
        ),
        "json_refresh_command_count": _safe_int(command_surface.get("json_refresh_command_count")),
        "blocked_until_input_acceptance": bool(
            sequence_integrity.get("blocked_until_input_acceptance")
        ),
        "blocked_command_count": _safe_int(sequence_integrity.get("blocked_command_count")),
        "ready_command_count": _safe_int(sequence_integrity.get("ready_command_count")),
        "unsafe_dependency_count": len(unsafe_command_ids),
        "unsafe_command_ids": _unique(unsafe_command_ids)[:50],
        "rows": dependency_rows,
        "fail_closed": dependency_ok,
    }


def _release_gate_post_input_refresh_artifact_coverage_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    refresh_plan = _release_gate_post_input_refresh_plan_from_readiness(readiness)
    terminal_guard_dependency = (
        _release_gate_post_input_refresh_terminal_guard_dependency_from_readiness(readiness)
    )
    expected_artifact_ids = [
        "gpu_bubble_experiment_readiness_next_actions",
        "gpu_bubble_readiness_terminal_self_check",
        "gpu_bubble_release_readiness_guard_report",
    ]
    rows: list[dict[str, Any]] = []
    for plan_row in (_mapping(row) for row in _list(refresh_plan.get("rows"))):
        input_id = str(plan_row.get("input_id") or "")
        artifact_ids = _strings(plan_row.get("post_refresh_required_artifact_ids"))
        missing_artifact_ids = [
            artifact_id for artifact_id in expected_artifact_ids if artifact_id not in artifact_ids
        ]
        covered = not missing_artifact_ids
        unsafe = (
            not covered
            or bool(plan_row.get("safe_to_auto_start_refresh"))
            or bool(plan_row.get("release_claim_allowed_after_refresh"))
            or not bool(plan_row.get("not_release_evidence"))
        )
        rows.append(
            {
                "input_id": input_id,
                "artifact_ids": artifact_ids[:20],
                "artifact_count": len(artifact_ids),
                "missing_artifact_ids": missing_artifact_ids,
                "readiness_artifact_required": (
                    "gpu_bubble_experiment_readiness_next_actions" in artifact_ids
                ),
                "terminal_artifact_required": (
                    "gpu_bubble_readiness_terminal_self_check" in artifact_ids
                ),
                "release_guard_artifact_required": (
                    "gpu_bubble_release_readiness_guard_report" in artifact_ids
                ),
                "covered": covered,
                "blocked_until_input_acceptance": bool(
                    terminal_guard_dependency.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    missing_coverage_input_ids = [
        str(row.get("input_id") or "") for row in rows if not bool(row.get("covered"))
    ]
    unsafe_input_ids = [
        str(row.get("input_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    coverage_ok = (
        not missing_coverage_input_ids
        and not unsafe_input_ids
        and bool(terminal_guard_dependency.get("dependency_ok"))
    )
    return {
        "coverage_ok": coverage_ok,
        "required_artifact_count": len(expected_artifact_ids),
        "required_artifact_ids": expected_artifact_ids,
        "input_row_count": len(rows),
        "covered_input_count": sum(1 for row in rows if bool(row.get("covered"))),
        "covered_input_ids": [
            str(row.get("input_id") or "") for row in rows if bool(row.get("covered"))
        ][:50],
        "missing_coverage_input_count": len(missing_coverage_input_ids),
        "missing_coverage_input_ids": _unique(missing_coverage_input_ids)[:50],
        "readiness_artifact_required": all(
            bool(row.get("readiness_artifact_required")) for row in rows
        ),
        "terminal_artifact_required": all(
            bool(row.get("terminal_artifact_required")) for row in rows
        ),
        "release_guard_artifact_required": all(
            bool(row.get("release_guard_artifact_required")) for row in rows
        ),
        "terminal_guard_dependency_ok": bool(terminal_guard_dependency.get("dependency_ok")),
        "blocked_until_input_acceptance": bool(
            terminal_guard_dependency.get("blocked_until_input_acceptance")
        ),
        "unsafe_artifact_coverage_count": len(unsafe_input_ids),
        "unsafe_input_ids": _unique(unsafe_input_ids)[:50],
        "rows": rows,
        "fail_closed": coverage_ok,
    }


def _release_gate_post_input_refresh_command_artifact_link_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    command_surface = _release_gate_post_input_refresh_command_surface_from_readiness(readiness)
    artifact_coverage = _release_gate_post_input_refresh_artifact_coverage_from_readiness(readiness)
    output_artifacts_by_command = {
        "refresh_gpu_bubble_readiness_next_actions": [
            "gpu_bubble_experiment_readiness_next_actions"
        ],
        "refresh_gpu_bubble_terminal_self_check": [
            "gpu_bubble_readiness_terminal_self_check"
        ],
        "run_gpu_bubble_release_readiness_guard": [
            "gpu_bubble_release_readiness_guard_report"
        ],
    }
    required_artifact_ids = _strings(artifact_coverage.get("required_artifact_ids"))
    rows: list[dict[str, Any]] = []
    for command_row in (
        _mapping(row) for row in _list(command_surface.get("rows"))
    ):
        command_id = str(command_row.get("command_id") or "")
        artifact_ids = output_artifacts_by_command.get(command_id, [])
        unsafe = (
            bool(command_row.get("safe_to_auto_start"))
            or bool(command_row.get("release_claim_allowed"))
            or not bool(command_row.get("not_release_evidence"))
            or bool(command_row.get("unsafe"))
        )
        rows.append(
            {
                "command_id": command_id,
                "command_order": _safe_int(command_row.get("command_order")),
                "command_kind": str(command_row.get("command_kind") or ""),
                "output_artifact_ids": artifact_ids,
                "output_artifact_count": len(artifact_ids),
                "produces_required_post_refresh_artifact": bool(artifact_ids),
                "blocked_until_input_acceptance": bool(
                    command_row.get("blocked_until_input_acceptance")
                ),
                "safe_to_auto_start": False,
                "release_claim_allowed": False,
                "not_release_evidence": True,
                "unsafe": unsafe,
            }
        )
    linked_artifact_ids = _unique(
        [
            artifact_id
            for row in rows
            for artifact_id in _strings(row.get("output_artifact_ids"))
        ]
    )
    missing_link_artifact_ids = [
        artifact_id for artifact_id in required_artifact_ids if artifact_id not in linked_artifact_ids
    ]
    extra_link_artifact_ids = [
        artifact_id for artifact_id in linked_artifact_ids if artifact_id not in required_artifact_ids
    ]
    unsafe_command_ids = [
        str(row.get("command_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    command_ids_with_artifacts = [
        str(row.get("command_id") or "")
        for row in rows
        if bool(row.get("produces_required_post_refresh_artifact"))
    ]
    link_ok = (
        not missing_link_artifact_ids
        and not extra_link_artifact_ids
        and not unsafe_command_ids
        and bool(artifact_coverage.get("coverage_ok"))
    )
    return {
        "link_ok": link_ok,
        "command_row_count": len(rows),
        "required_artifact_count": len(required_artifact_ids),
        "required_artifact_ids": required_artifact_ids,
        "linked_artifact_count": len(linked_artifact_ids),
        "linked_artifact_ids": linked_artifact_ids,
        "missing_link_artifact_count": len(missing_link_artifact_ids),
        "missing_link_artifact_ids": missing_link_artifact_ids,
        "extra_link_artifact_count": len(extra_link_artifact_ids),
        "extra_link_artifact_ids": extra_link_artifact_ids,
        "command_artifact_link_count": len(command_ids_with_artifacts),
        "command_ids_with_artifacts": command_ids_with_artifacts,
        "command_ids_without_artifacts": [
            str(row.get("command_id") or "")
            for row in rows
            if not bool(row.get("produces_required_post_refresh_artifact"))
        ],
        "readiness_command_id": "refresh_gpu_bubble_readiness_next_actions",
        "terminal_command_id": "refresh_gpu_bubble_terminal_self_check",
        "release_guard_command_id": "run_gpu_bubble_release_readiness_guard",
        "artifact_coverage_ok": bool(artifact_coverage.get("coverage_ok")),
        "blocked_until_input_acceptance": all(
            bool(row.get("blocked_until_input_acceptance")) for row in rows
        ),
        "unsafe_link_count": len(unsafe_command_ids),
        "unsafe_command_ids": _unique(unsafe_command_ids)[:50],
        "rows": rows,
        "fail_closed": link_ok,
    }


def _release_gate_post_input_refresh_guard_consumption_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    terminal_guard_dependency = (
        _release_gate_post_input_refresh_terminal_guard_dependency_from_readiness(readiness)
    )
    artifact_coverage = _release_gate_post_input_refresh_artifact_coverage_from_readiness(readiness)
    command_artifact_link = (
        _release_gate_post_input_refresh_command_artifact_link_from_readiness(readiness)
    )
    blocker_presence = _remaining_blocker_artifact_presence_from_readiness(readiness)
    source_downstream_contract = _source_and_downstream_contract_from_readiness(readiness)
    source_freshness = _mapping(readiness.get("source_axis_freshness_dedupe_audit")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_axis_freshness_dedupe_audit")
    )
    evidence_summary = _mapping(readiness.get("evidence_summary"))
    external_runner_manifest_summary = _mapping(
        readiness.get("external_input_json_refresh_runner_manifest_summary")
    ) or _mapping(
        evidence_summary.get("external_input_json_refresh_runner_manifest_summary")
    )
    required_input_artifact_ids = [
        "gpu_bubble_experiment_readiness_next_actions",
        "gpu_bubble_readiness_terminal_self_check",
    ]
    produced_guard_artifact_id = "gpu_bubble_release_readiness_guard_report"
    required_consumed_summary_ids = [
        "release_gate_post_input_refresh_terminal_guard_dependency_summary",
        "release_gate_post_input_refresh_artifact_coverage_summary",
        "release_gate_post_input_refresh_command_artifact_link_summary",
        "remaining_blocker_artifact_presence_summary",
        "source_and_downstream_artifact_contract_summary",
        "source_axis_freshness_dedupe_audit",
        "external_input_json_refresh_runner_manifest_summary",
        "roadmap_lineage_audit",
    ]
    summary_sources = {
        "release_gate_post_input_refresh_terminal_guard_dependency_summary": terminal_guard_dependency,
        "release_gate_post_input_refresh_artifact_coverage_summary": artifact_coverage,
        "release_gate_post_input_refresh_command_artifact_link_summary": command_artifact_link,
        "remaining_blocker_artifact_presence_summary": blocker_presence,
        "source_and_downstream_artifact_contract_summary": source_downstream_contract,
        "source_axis_freshness_dedupe_audit": source_freshness,
        "external_input_json_refresh_runner_manifest_summary": external_runner_manifest_summary,
    }
    terminal_only_summary_ids = ["roadmap_lineage_audit"]
    rows: list[dict[str, Any]] = []
    for summary_id in required_consumed_summary_ids:
        summary = _mapping(summary_sources.get(summary_id))
        terminal_only = summary_id in terminal_only_summary_ids
        present = terminal_only or bool(summary)
        fail_closed = terminal_only or present
        if summary_id == "release_gate_post_input_refresh_terminal_guard_dependency_summary":
            fail_closed = fail_closed and bool(summary.get("dependency_ok"))
        elif summary_id == "release_gate_post_input_refresh_artifact_coverage_summary":
            fail_closed = fail_closed and bool(summary.get("coverage_ok"))
        elif summary_id == "release_gate_post_input_refresh_command_artifact_link_summary":
            fail_closed = fail_closed and bool(summary.get("link_ok"))
        elif summary_id == "external_input_json_refresh_runner_manifest_summary":
            expected_count = len(POST_INPUT_REFRESH_SEQUENCE)
            fail_closed = (
                fail_closed
                and bool(summary.get("fail_closed"))
                and bool(summary.get("manifest_ok"))
                and bool(summary.get("runner_ready"))
                and bool(summary.get("execution_ok"))
                and bool(summary.get("row_execution_consistent"))
                and bool(summary.get("sequence_ok"))
                and bool(summary.get("stage_manifest_ok"))
                and bool(summary.get("not_release_evidence"))
                and _safe_int(summary.get("expected_command_count")) == expected_count
                and _safe_int(summary.get("command_count")) == expected_count
                and _safe_int(summary.get("executed_count")) == expected_count
                and _safe_int(summary.get("row_count")) == expected_count
                and _safe_int(summary.get("executed_row_count")) == expected_count
                and _safe_int(summary.get("failure_count")) == 0
                and _safe_int(summary.get("failed_row_count")) == 0
                and _safe_int(summary.get("output_missing_count")) == 0
                and _safe_int(summary.get("missing_output_row_count")) == 0
                and _safe_int(summary.get("forbidden_heavy_flag_count")) == 0
                and _safe_int(summary.get("row_forbidden_heavy_flag_count")) == 0
                and _safe_int(summary.get("unsafe_row_count")) == 0
                and _safe_int(summary.get("validation_issue_count")) == 0
                and _safe_int(summary.get("stage_manifest_issue_count")) == 0
                and _safe_int(summary.get("stage_count")) == expected_count
                and _safe_int(summary.get("script_count")) == expected_count
                and _safe_int(summary.get("expected_output_count")) == expected_count
                and _safe_int(summary.get("stage_manifest_forbidden_heavy_flag_count")) == 0
            )
        unsafe = (
            not present
            or not fail_closed
            or bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
        )
        row = {
            "summary_id": summary_id,
            "consumption_stage": "terminal" if terminal_only else "readiness",
            "required_for_guard": True,
            "present": present,
            "fail_closed": fail_closed,
            "terminal_only": terminal_only,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": unsafe,
        }
        if summary_id == "external_input_json_refresh_runner_manifest_summary":
            row.update(
                {
                    "manifest_ok": bool(summary.get("manifest_ok")),
                    "runner_ready": bool(summary.get("runner_ready")),
                    "execution_ok": bool(summary.get("execution_ok")),
                    "row_execution_consistent": bool(
                        summary.get("row_execution_consistent")
                    ),
                    "expected_command_count": _safe_int(
                        summary.get("expected_command_count")
                    ),
                    "row_count": _safe_int(summary.get("row_count")),
                    "executed_row_count": _safe_int(summary.get("executed_row_count")),
                    "failed_row_count": _safe_int(summary.get("failed_row_count")),
                    "missing_output_row_count": _safe_int(
                        summary.get("missing_output_row_count")
                    ),
                    "row_forbidden_heavy_flag_count": _safe_int(
                        summary.get("row_forbidden_heavy_flag_count")
                    ),
                    "unsafe_row_count": _safe_int(summary.get("unsafe_row_count")),
                }
            )
        rows.append(row)
    missing_summary_ids = [
        str(row.get("summary_id") or "") for row in rows if not bool(row.get("present"))
    ]
    unsafe_summary_ids = [
        str(row.get("summary_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    linked_artifact_ids = _strings(command_artifact_link.get("linked_artifact_ids"))
    input_artifacts_consumed = all(
        artifact_id in linked_artifact_ids for artifact_id in required_input_artifact_ids
    )
    guard_artifact_produced = produced_guard_artifact_id in linked_artifact_ids
    guard_command_id = str(command_artifact_link.get("release_guard_command_id") or "")
    consumption_ok = (
        guard_command_id == "run_gpu_bubble_release_readiness_guard"
        and input_artifacts_consumed
        and guard_artifact_produced
        and bool(command_artifact_link.get("link_ok"))
        and bool(artifact_coverage.get("coverage_ok"))
        and bool(terminal_guard_dependency.get("dependency_ok"))
        and not missing_summary_ids
        and not unsafe_summary_ids
    )
    return {
        "consumption_ok": consumption_ok,
        "guard_command_id": guard_command_id,
        "required_input_artifact_count": len(required_input_artifact_ids),
        "required_input_artifact_ids": required_input_artifact_ids,
        "produced_guard_artifact_id": produced_guard_artifact_id,
        "input_artifacts_consumed": input_artifacts_consumed,
        "guard_artifact_produced": guard_artifact_produced,
        "required_consumed_summary_count": len(required_consumed_summary_ids),
        "required_consumed_summary_ids": required_consumed_summary_ids,
        "present_consumed_summary_count": sum(1 for row in rows if bool(row.get("present"))),
        "missing_consumed_summary_count": len(missing_summary_ids),
        "missing_consumed_summary_ids": missing_summary_ids,
        "unsafe_consumed_summary_count": len(unsafe_summary_ids),
        "unsafe_consumed_summary_ids": _unique(unsafe_summary_ids)[:50],
        "terminal_lineage_required": True,
        "terminal_lineage_summary_id": "roadmap_lineage_audit",
        "command_artifact_link_ok": bool(command_artifact_link.get("link_ok")),
        "artifact_coverage_ok": bool(artifact_coverage.get("coverage_ok")),
        "terminal_guard_dependency_ok": bool(terminal_guard_dependency.get("dependency_ok")),
        "blocked_until_input_acceptance": bool(
            command_artifact_link.get("blocked_until_input_acceptance")
        ),
        "rows": rows,
        "fail_closed": consumption_ok,
    }


def _release_gate_post_input_refresh_guard_report_acceptance_from_readiness(
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    guard_consumption = _release_gate_post_input_refresh_guard_consumption_from_readiness(readiness)
    required_guard_report_fields = [
        "roadmap",
        "artifact_role",
        "status",
        "ok",
        "failure_count",
        "input_artifact_summary",
        "not_release_evidence",
        "safe_to_auto_start",
        "release_claim_allowed",
        "blocked_actions",
    ]
    rows = [
        {
            "acceptance_id": "guard_report_identity",
            "required_field_ids": ["roadmap", "artifact_role"],
            "required": True,
            "expected_value_summary": "gpu_bubble_release_readiness_guard_report_on_active_roadmap",
            "present": True,
            "fail_closed": True,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": False,
        },
        {
            "acceptance_id": "guard_report_status",
            "required_field_ids": ["status", "ok", "failure_count"],
            "required": True,
            "expected_value_summary": "guard_passed_blocked_release_claim_ok_true_failures_zero",
            "present": True,
            "fail_closed": True,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": False,
        },
        {
            "acceptance_id": "guard_report_input_artifacts",
            "required_field_ids": ["input_artifact_summary"],
            "required": True,
            "expected_value_summary": "readiness_and_terminal_inputs_consumed",
            "present": bool(guard_consumption.get("input_artifacts_consumed")),
            "fail_closed": bool(guard_consumption.get("input_artifacts_consumed")),
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": not bool(guard_consumption.get("input_artifacts_consumed")),
        },
        {
            "acceptance_id": "guard_report_release_safety",
            "required_field_ids": [
                "not_release_evidence",
                "safe_to_auto_start",
                "release_claim_allowed",
                "blocked_actions",
            ],
            "required": True,
            "expected_value_summary": "not_release_no_auto_start_no_claim",
            "present": True,
            "fail_closed": True,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": False,
        },
        {
            "acceptance_id": "guard_consumption_linkage",
            "required_field_ids": [
                "guard_command_id",
                "produced_guard_artifact_id",
                "guard_artifact_produced",
            ],
            "required": True,
            "expected_value_summary": "run_gpu_bubble_release_readiness_guard_produces_report",
            "present": bool(guard_consumption.get("guard_artifact_produced")),
            "fail_closed": bool(guard_consumption.get("guard_artifact_produced")),
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "unsafe": not bool(guard_consumption.get("guard_artifact_produced")),
        },
    ]
    unsafe_acceptance_ids = [
        str(row.get("acceptance_id") or "") for row in rows if bool(row.get("unsafe"))
    ]
    guard_command_id = str(guard_consumption.get("guard_command_id") or "")
    guard_report_artifact_id = str(guard_consumption.get("produced_guard_artifact_id") or "")
    acceptance_ok = (
        guard_command_id == "run_gpu_bubble_release_readiness_guard"
        and guard_report_artifact_id == "gpu_bubble_release_readiness_guard_report"
        and bool(guard_consumption.get("consumption_ok"))
        and bool(guard_consumption.get("input_artifacts_consumed"))
        and bool(guard_consumption.get("guard_artifact_produced"))
        and not unsafe_acceptance_ids
    )
    return {
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_release_gate_post_input_refresh_guard_report_acceptance_summary",
        "acceptance_status": (
            "post_input_refresh_guard_report_acceptance_blocked_intact"
            if acceptance_ok
            else "post_input_refresh_guard_report_acceptance_drift_detected"
        ),
        "acceptance_ok": acceptance_ok,
        "guard_command_id": guard_command_id,
        "guard_report_artifact_id": guard_report_artifact_id,
        "expected_report_status": "guard_passed_blocked_release_claim",
        "expected_ok": True,
        "expected_failure_count": 0,
        "required_guard_report_field_count": len(required_guard_report_fields),
        "required_guard_report_fields": required_guard_report_fields,
        "requires_input_artifact_summary": True,
        "requires_not_release_evidence": True,
        "requires_safe_to_auto_start_false": True,
        "requires_release_claim_allowed_false": True,
        "requires_blocked_actions": True,
        "guard_consumption_ok": bool(guard_consumption.get("consumption_ok")),
        "input_artifacts_consumed": bool(guard_consumption.get("input_artifacts_consumed")),
        "guard_artifact_produced": bool(guard_consumption.get("guard_artifact_produced")),
        "blocked_until_input_acceptance": bool(guard_consumption.get("blocked_until_input_acceptance")),
        "acceptance_row_count": len(rows),
        "unsafe_acceptance_count": len(unsafe_acceptance_ids),
        "unsafe_acceptance_ids": _unique(unsafe_acceptance_ids)[:50],
        "rows": rows,
        "execution_policy": "post_input_refresh_guard_report_acceptance_only",
        "fail_closed": acceptance_ok,
    }


def _manual_protected_gpu_command_surface_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(readiness.get("manual_protected_gpu_command_surface_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_protected_gpu_command_surface_summary")
    )
    rows = [_mapping(row) for row in _list(summary.get("rows"))]
    unsafe_ids = _unique(
        [
            str(row.get("id") or "")
            for row in rows
            if bool(row.get("safe_to_auto_start"))
            or bool(row.get("release_claim_allowed_after_success"))
            or not bool(row.get("not_release_evidence"))
            or (
                bool(row.get("requires_gpu_if_executed"))
                and not bool(row.get("manual_start_required"))
            )
            or bool(row.get("unsafe"))
        ]
    )
    return {
        "source_artifact_count": len(_strings(summary.get("source_artifact_ids"))),
        "source_artifact_ids": _strings(summary.get("source_artifact_ids")),
        "command_surface_row_count": len(rows),
        "row_ids": _unique([str(row.get("id") or "") for row in rows]),
        "manual_gpu_command_count": sum(1 for row in rows if bool(row.get("requires_gpu_if_executed"))),
        "protected_gpu_command_count": sum(
            1
            for row in rows
            if bool(row.get("requires_gpu_if_executed")) and bool(row.get("manual_start_required"))
        ),
        "dry_run_command_count": sum(1 for row in rows if bool(row.get("dry_run_present"))),
        "template_command_count": sum(1 for row in rows if bool(row.get("template"))),
        "ready_command_count": sum(1 for row in rows if bool(row.get("ready"))),
        "blocked_command_count": sum(1 for row in rows if bool(row.get("blocked"))),
        "completed_existing_command_count": sum(1 for row in rows if bool(row.get("completed_existing"))),
        "rerun_blocked_without_new_axis_count": sum(
            1 for row in rows if bool(row.get("do_not_rerun_without_new_axis"))
        ),
        "requires_gpu_if_executed_count": sum(1 for row in rows if bool(row.get("requires_gpu_if_executed"))),
        "manual_start_required_count": sum(1 for row in rows if bool(row.get("manual_start_required"))),
        "release_relevant_command_count": sum(1 for row in rows if bool(row.get("release_relevant"))),
        "diagnostic_only_command_count": sum(1 for row in rows if bool(row.get("diagnostic_only"))),
        "release_claim_allowed_after_success_count": sum(
            1 for row in rows if bool(row.get("release_claim_allowed_after_success"))
        ),
        "unsafe_command_count": len(unsafe_ids),
        "unsafe_command_ids": unsafe_ids,
        "fail_closed": not unsafe_ids,
    }


def _manual_gpu_summary_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    return {
        "gpu_related_action_count": _safe_int(summary.get("gpu_related_action_count")),
        "current_gpu_heavy_action_count": _safe_int(summary.get("current_gpu_heavy_action_count")),
        "followup_gpu_required_action_count": _safe_int(summary.get("followup_gpu_required_action_count")),
        "protected_manual_gpu_ready_action_count": _safe_int(
            summary.get("protected_manual_gpu_ready_action_count")
        ),
        "blocked_missing_prerequisite_gpu_action_count": _safe_int(
            summary.get("blocked_missing_prerequisite_gpu_action_count")
        ),
        "waiting_manual_gpu_evidence_action_count": _safe_int(
            summary.get("waiting_manual_gpu_evidence_action_count")
        ),
    }


def _check_manual_gpu_summary_counts(
    failures: list[dict[str, Any]],
    *,
    check_id: str,
    reason: str,
    summary: Mapping[str, Any],
    expected: Mapping[str, int],
) -> None:
    observed = _manual_gpu_summary_counts(summary)
    if dict(observed) != dict(expected):
        _failure(
            failures,
            check_id,
            reason,
            value={"summary": observed, "computed": dict(expected)},
        )


def _input_artifact_summary(
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    remaining: Mapping[str, Any],
    terminal_audit: Mapping[str, Any],
    artifact_freshness: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    guard_report_acceptance = _mapping(
        terminal.get("release_gate_post_input_refresh_guard_report_acceptance_summary")
    )
    runner_manifest_summary = _mapping(
        terminal.get("external_input_json_refresh_runner_manifest_summary")
    )
    return {
        "readiness": {
            "report": str(readiness.get("report") or ""),
            "artifact_status": str(readiness.get("artifact_status") or ""),
            "release_readiness": str(readiness.get("release_readiness") or ""),
            "json_ready_action_count": _safe_int(remaining.get("json_ready_action_count")),
            "json_closed_action_count": _safe_int(remaining.get("json_closed_action_count")),
            "gpu_bubble_release_hard_gate_ids": list(readiness_hard_gates)[:20],
            "safe_to_auto_start": bool(readiness.get("safe_to_auto_start")),
            "release_claim_allowed": bool(readiness.get("release_claim_allowed")),
            "publishable": bool(readiness.get("publishable")),
        },
        "terminal": {
            "report": str(terminal.get("report") or ""),
            "terminal_status": str(terminal.get("terminal_status") or ""),
            "chain_integrity_status": str(terminal.get("chain_integrity_status") or ""),
            "source_path_count": _safe_int(terminal_audit.get("source_path_count")),
            "source_path_missing_count": _safe_int(terminal_audit.get("source_path_missing_count")),
            "source_path_load_error_count": _safe_int(terminal_audit.get("source_path_load_error_count")),
            "chain_complete": bool(terminal_audit.get("chain_complete")),
            "safe_to_auto_start": bool(terminal.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal.get("release_claim_allowed")),
        },
        "freshness": {
            "roadmap": str(artifact_freshness.get("roadmap") or ""),
            "artifact_role": str(artifact_freshness.get("artifact_role") or ""),
            "freshness_ok": bool(artifact_freshness.get("freshness_ok")),
            "required_artifact_missing_count": _safe_int(
                artifact_freshness.get("required_artifact_missing_count")
            ),
            "upstream_newer_than_readiness_count": _safe_int(
                artifact_freshness.get("upstream_newer_than_readiness_count")
            ),
            "terminal_observation_not_older_than_readiness": bool(
                artifact_freshness.get("terminal_observation_not_older_than_readiness")
            ),
            "fail_closed": bool(artifact_freshness.get("fail_closed")),
            "not_release_evidence": bool(artifact_freshness.get("not_release_evidence")),
            "safe_to_auto_start": bool(artifact_freshness.get("safe_to_auto_start")),
            "release_claim_allowed": bool(artifact_freshness.get("release_claim_allowed")),
            "publishable": bool(artifact_freshness.get("publishable")),
            "does_not_run_training": bool(artifact_freshness.get("does_not_run_training")),
            "does_not_run_cuda": bool(artifact_freshness.get("does_not_run_cuda")),
        },
        "release_gate_post_input_refresh_guard_report_acceptance_summary": {
            "roadmap": str(guard_report_acceptance.get("roadmap") or ""),
            "artifact_role": str(guard_report_acceptance.get("artifact_role") or ""),
            "acceptance_status": str(guard_report_acceptance.get("acceptance_status") or ""),
            "acceptance_ok": bool(guard_report_acceptance.get("acceptance_ok")),
            "guard_command_id": str(guard_report_acceptance.get("guard_command_id") or ""),
            "guard_report_artifact_id": str(
                guard_report_acceptance.get("guard_report_artifact_id") or ""
            ),
            "expected_report_status": str(
                guard_report_acceptance.get("expected_report_status") or ""
            ),
            "expected_ok": bool(guard_report_acceptance.get("expected_ok")),
            "expected_failure_count": _safe_int(
                guard_report_acceptance.get("expected_failure_count")
            ),
            "required_guard_report_field_count": _safe_int(
                guard_report_acceptance.get("required_guard_report_field_count")
            ),
            "acceptance_row_count": _safe_int(
                guard_report_acceptance.get("acceptance_row_count")
            ),
            "unsafe_acceptance_count": _safe_int(
                guard_report_acceptance.get("unsafe_acceptance_count")
            ),
            "guard_consumption_ok": bool(guard_report_acceptance.get("guard_consumption_ok")),
            "input_artifacts_consumed": bool(
                guard_report_acceptance.get("input_artifacts_consumed")
            ),
            "guard_artifact_produced": bool(
                guard_report_acceptance.get("guard_artifact_produced")
            ),
            "blocked_until_input_acceptance": bool(
                guard_report_acceptance.get("blocked_until_input_acceptance")
            ),
            "execution_policy": str(guard_report_acceptance.get("execution_policy") or ""),
            "fail_closed": bool(guard_report_acceptance.get("fail_closed")),
            "not_release_evidence": bool(guard_report_acceptance.get("not_release_evidence")),
            "safe_to_auto_start": bool(guard_report_acceptance.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                guard_report_acceptance.get("release_claim_allowed")
            ),
        },
        "external_input_json_refresh_runner_manifest_summary": {
            "roadmap": str(runner_manifest_summary.get("roadmap") or ""),
            "artifact_role": str(runner_manifest_summary.get("artifact_role") or ""),
            "status": str(runner_manifest_summary.get("status") or ""),
            "manifest_ok": bool(runner_manifest_summary.get("manifest_ok")),
            "runner_ready": bool(runner_manifest_summary.get("runner_ready")),
            "execution_ok": bool(runner_manifest_summary.get("execution_ok")),
            "row_execution_consistent": bool(
                runner_manifest_summary.get("row_execution_consistent")
            ),
            "expected_command_count": _safe_int(
                runner_manifest_summary.get("expected_command_count")
            ),
            "command_count": _safe_int(runner_manifest_summary.get("command_count")),
            "row_count": _safe_int(runner_manifest_summary.get("row_count")),
            "executed_count": _safe_int(runner_manifest_summary.get("executed_count")),
            "executed_row_count": _safe_int(
                runner_manifest_summary.get("executed_row_count")
            ),
            "failure_count": _safe_int(runner_manifest_summary.get("failure_count")),
            "failed_row_count": _safe_int(runner_manifest_summary.get("failed_row_count")),
            "output_missing_count": _safe_int(
                runner_manifest_summary.get("output_missing_count")
            ),
            "missing_output_row_count": _safe_int(
                runner_manifest_summary.get("missing_output_row_count")
            ),
            "forbidden_heavy_flag_count": _safe_int(
                runner_manifest_summary.get("forbidden_heavy_flag_count")
            ),
            "row_forbidden_heavy_flag_count": _safe_int(
                runner_manifest_summary.get("row_forbidden_heavy_flag_count")
            ),
            "unsafe_row_count": _safe_int(runner_manifest_summary.get("unsafe_row_count")),
            "validation_issue_count": _safe_int(
                runner_manifest_summary.get("validation_issue_count")
            ),
            "sequence_ok": bool(runner_manifest_summary.get("sequence_ok")),
            "canonical_stage_ids": _strings(
                runner_manifest_summary.get("canonical_stage_ids")
            )[:20],
            "observed_stage_ids": _strings(
                runner_manifest_summary.get("observed_stage_ids")
            )[:20],
            "row_stage_ids": _strings(
                runner_manifest_summary.get("row_stage_ids")
            )[:20],
            "stage_manifest_source": str(
                runner_manifest_summary.get("stage_manifest_source") or ""
            ),
            "stage_manifest_ok": bool(
                runner_manifest_summary.get("stage_manifest_ok")
            ),
            "stage_manifest_issue_count": _safe_int(
                runner_manifest_summary.get("stage_manifest_issue_count")
            ),
            "stage_manifest_issue_reasons": _strings(
                runner_manifest_summary.get("stage_manifest_issue_reasons")
            )[:20],
            "stage_count": _safe_int(runner_manifest_summary.get("stage_count")),
            "stage_ids": _strings(runner_manifest_summary.get("stage_ids"))[:20],
            "script_count": _safe_int(runner_manifest_summary.get("script_count")),
            "expected_output_count": _safe_int(
                runner_manifest_summary.get("expected_output_count")
            ),
            "stage_manifest_forbidden_heavy_flag_count": _safe_int(
                runner_manifest_summary.get("stage_manifest_forbidden_heavy_flag_count")
            ),
            "fail_closed": bool(runner_manifest_summary.get("fail_closed")),
            "not_release_evidence": bool(runner_manifest_summary.get("not_release_evidence")),
            "safe_to_auto_start": bool(runner_manifest_summary.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                runner_manifest_summary.get("release_claim_allowed")
            ),
        },
    }


def _downstream_artifact_rows(readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(readiness.get("downstream_artifacts"))]


def _downstream_artifact_summary(readiness: Mapping[str, Any]) -> dict[str, Any]:
    rows = _downstream_artifact_rows(readiness)
    return {
        "artifact_count": len(rows),
        "artifact_ids": [str(row.get("id") or "") for row in rows],
        "source_path_dependency_count": sum(1 for row in rows if bool(row.get("source_path_dependency"))),
        "not_release_evidence_count": sum(1 for row in rows if bool(row.get("not_release_evidence"))),
        "auto_startable_count": sum(1 for row in rows if bool(row.get("safe_to_auto_start"))),
        "release_claim_allowed_count": sum(1 for row in rows if bool(row.get("release_claim_allowed"))),
        "required_after_readiness_refresh_count": sum(
            1 for row in rows if bool(row.get("required_after_readiness_refresh"))
        ),
    }


def _check_downstream_artifacts(failures: list[dict[str, Any]], readiness: Mapping[str, Any]) -> None:
    rows = _downstream_artifact_rows(readiness)
    if not rows:
        _failure(failures, "downstream_artifacts", "readiness_downstream_artifacts_missing")
        return
    expected = {
        "gpu_bubble_readiness_terminal_self_check": (
            "refresh_gpu_bubble_terminal_self_check",
            "gpu_bubble_readiness_terminal_self_check.json",
        ),
        "gpu_bubble_release_readiness_guard_report": (
            "run_gpu_bubble_release_readiness_guard",
            "gpu_bubble_release_readiness_guard_report.json",
        ),
    }
    by_id = {str(row.get("id") or ""): row for row in rows}
    missing = [artifact_id for artifact_id in expected if artifact_id not in by_id]
    if missing:
        _failure(failures, "downstream_artifacts_required", "required_downstream_artifacts_missing", value=missing)
    for artifact_id, (command_id, filename) in expected.items():
        row = by_id.get(artifact_id)
        if not row:
            continue
        row_path = str(row.get("path") or "")
        row_name = row_path.replace("\\", "/").rsplit("/", 1)[-1]
        if not row_path or row_name != filename:
            _failure(
                failures,
                f"downstream_{artifact_id}_path",
                "downstream_artifact_path_missing_or_wrong",
                value={"expected_filename": filename, "observed": row_path},
            )
        if not bool(row.get("exists")):
            _failure(
                failures,
                f"downstream_{artifact_id}_exists",
                "downstream_artifact_not_marked_existing",
            )
        if str(row.get("command_id") or "") != command_id:
            _failure(
                failures,
                f"downstream_{artifact_id}_command",
                "downstream_artifact_command_id_mismatch",
                value={"expected": command_id, "observed": row.get("command_id")},
            )
        if bool(row.get("source_path_dependency")):
            _failure(
                failures,
                f"downstream_{artifact_id}_source_dependency",
                "downstream_artifact_marked_as_source_path_dependency",
            )
        if not bool(row.get("not_release_evidence")):
            _failure(
                failures,
                f"downstream_{artifact_id}_not_release_evidence",
                "downstream_artifact_not_marked_as_non_release_evidence",
            )
        if bool(row.get("safe_to_auto_start")) or bool(row.get("release_claim_allowed")):
            _failure(
                failures,
                f"downstream_{artifact_id}_fail_closed",
                "downstream_artifact_allows_release_or_auto_start",
                value={
                    "safe_to_auto_start": row.get("safe_to_auto_start"),
                    "release_claim_allowed": row.get("release_claim_allowed"),
                },
            )
        if not bool(row.get("required_after_readiness_refresh")):
            _failure(
                failures,
                f"downstream_{artifact_id}_required_after_readiness",
                "downstream_artifact_not_required_after_readiness_refresh",
            )
    source_names = {
        str(row.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        for row in (_mapping(item) for item in _list(readiness.get("source_paths")))
    }
    forbidden = {
        "gpu_bubble_readiness_terminal_self_check.json",
        "gpu_bubble_release_readiness_guard_report.json",
    }
    leaked = sorted(source_names.intersection(forbidden))
    if leaked:
        _failure(
            failures,
            "downstream_artifacts_source_paths",
            "downstream_artifacts_leaked_into_readiness_source_paths",
            value=leaked,
        )


def _source_and_downstream_contract_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = [_mapping(item) for item in _list(readiness.get("source_paths"))]
    downstream_rows = _downstream_artifact_rows(readiness)
    source_missing = [
        str(row.get("path") or "") for row in source_rows if not bool(row.get("exists"))
    ]
    source_load_errors = [
        str(row.get("path") or "") for row in source_rows if str(row.get("load_error") or "")
    ]
    source_roadmap_mismatch = [
        str(row.get("path") or "")
        for row in source_rows
        if str(row.get("roadmap") or "") and str(row.get("roadmap") or "") != ROADMAP
    ]
    source_release_unsafe = [
        str(row.get("path") or "")
        for row in source_rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed"))
        or bool(row.get("publishable"))
    ]
    downstream_missing = [
        str(row.get("id") or row.get("path") or "")
        for row in downstream_rows
        if not bool(row.get("exists"))
    ]
    downstream_source_dependency = [
        str(row.get("id") or row.get("path") or "")
        for row in downstream_rows
        if bool(row.get("source_path_dependency"))
    ]
    downstream_release_unsafe = [
        str(row.get("id") or row.get("path") or "")
        for row in downstream_rows
        if bool(row.get("safe_to_auto_start"))
        or bool(row.get("release_claim_allowed"))
        or not bool(row.get("not_release_evidence"))
    ]
    source_paths = {
        str(row.get("path") or "") for row in source_rows if str(row.get("path") or "")
    }
    downstream_paths = {
        str(row.get("path") or "") for row in downstream_rows if str(row.get("path") or "")
    }
    downstream_in_source_paths = sorted(source_paths.intersection(downstream_paths))
    unsafe_ids = _unique(
        [
            *source_missing,
            *source_load_errors,
            *source_roadmap_mismatch,
            *source_release_unsafe,
            *downstream_missing,
            *downstream_source_dependency,
            *downstream_release_unsafe,
            *downstream_in_source_paths,
        ]
    )
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_source_and_downstream_artifact_contract_summary",
        "source_path_count": len(source_rows),
        "source_path_missing_count": len(source_missing),
        "source_path_missing_paths": source_missing[:20],
        "source_path_load_error_count": len(source_load_errors),
        "source_path_load_error_paths": source_load_errors[:20],
        "source_path_roadmap_mismatch_count": len(source_roadmap_mismatch),
        "source_path_roadmap_mismatch_paths": source_roadmap_mismatch[:20],
        "source_path_release_unsafe_count": len(source_release_unsafe),
        "source_path_release_unsafe_paths": source_release_unsafe[:20],
        "downstream_artifact_count": len(downstream_rows),
        "downstream_artifact_missing_count": len(downstream_missing),
        "downstream_artifact_missing_ids": downstream_missing[:20],
        "downstream_source_path_dependency_count": len(downstream_source_dependency),
        "downstream_source_path_dependency_ids": downstream_source_dependency[:20],
        "downstream_release_unsafe_count": len(downstream_release_unsafe),
        "downstream_release_unsafe_ids": downstream_release_unsafe[:20],
        "downstream_in_source_path_count": len(downstream_in_source_paths),
        "downstream_in_source_paths": downstream_in_source_paths[:20],
        "unsafe_artifact_count": len(unsafe_ids),
        "unsafe_artifact_ids": unsafe_ids[:50],
        "fail_closed": len(unsafe_ids) == 0,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
    }


def _check_source_and_downstream_artifact_contract_summary(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_summary = _mapping(readiness.get("source_and_downstream_artifact_contract_summary"))
    evidence_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "source_and_downstream_artifact_contract_summary"
        )
    )
    terminal_summary = _mapping(terminal.get("source_and_downstream_artifact_contract_summary"))
    computed = _source_and_downstream_contract_from_readiness(readiness)
    targets = {
        "readiness_source_downstream_contract_summary": readiness_summary,
        "readiness_evidence_source_downstream_contract_summary": evidence_summary,
        "terminal_source_downstream_contract_summary": terminal_summary,
    }
    scalar_fields = [
        "summary_version",
        "source_path_count",
        "source_path_missing_count",
        "source_path_load_error_count",
        "source_path_roadmap_mismatch_count",
        "source_path_release_unsafe_count",
        "downstream_artifact_count",
        "downstream_artifact_missing_count",
        "downstream_source_path_dependency_count",
        "downstream_release_unsafe_count",
        "downstream_in_source_path_count",
        "unsafe_artifact_count",
    ]
    set_fields = [
        "source_path_missing_paths",
        "source_path_load_error_paths",
        "source_path_roadmap_mismatch_paths",
        "source_path_release_unsafe_paths",
        "downstream_artifact_missing_ids",
        "downstream_source_path_dependency_ids",
        "downstream_release_unsafe_ids",
        "downstream_in_source_paths",
        "unsafe_artifact_ids",
    ]
    for check_id, summary in targets.items():
        if not summary:
            _failure(failures, check_id, "source_downstream_contract_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "source_downstream_contract_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, "source_downstream_contract_summary_allows_release_or_auto_start")
        if str(summary.get("artifact_role") or "") != str(computed.get("artifact_role") or ""):
            _failure(
                failures,
                f"{check_id}_artifact_role_match",
                "source_downstream_contract_summary_artifact_role_does_not_match_computed",
                value={"summary": summary.get("artifact_role"), "computed": computed.get("artifact_role")},
            )
        for field in scalar_fields:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_downstream_contract_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed.get(field)),
            )
        for field in set_fields:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_downstream_contract_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed.get(field)),
            )
    if _safe_int(computed.get("unsafe_artifact_count")) != 0:
        _failure(
            failures,
            "source_downstream_contract_computed_unsafe_artifacts",
            "source_downstream_contract_recomputed_unsafe_artifacts",
            value=_strings(computed.get("unsafe_artifact_ids")),
        )
    return {
        **computed,
        "readiness_summary_available": bool(readiness_summary),
        "readiness_evidence_summary_available": bool(evidence_summary),
        "terminal_summary_available": bool(terminal_summary),
        "computed_unsafe_artifact_count": _safe_int(computed.get("unsafe_artifact_count")),
    }


def _source_artifact_inventory_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    rows = [_mapping(row) for row in _list(readiness.get("source_paths"))]
    status_counts: dict[str, int] = {}
    report_counts: dict[str, int] = {}
    missing_ids: list[str] = []
    load_error_ids: list[str] = []
    roadmap_mismatch_ids: list[str] = []
    release_unsafe_ids: list[str] = []
    publishable_ids: list[str] = []
    not_release_false_ids: list[str] = []
    no_training_false_ids: list[str] = []
    no_cuda_false_ids: list[str] = []
    for row in rows:
        path = str(row.get("path") or "")
        artifact_id = Path(path).stem if path else "unknown_source_artifact"
        status = str(row.get("status") or "status_missing")
        report = str(row.get("report") or "report_missing")
        status_counts[status] = status_counts.get(status, 0) + 1
        report_counts[report] = report_counts.get(report, 0) + 1
        if not bool(row.get("exists")):
            missing_ids.append(artifact_id)
        if str(row.get("load_error") or ""):
            load_error_ids.append(artifact_id)
        if str(row.get("roadmap") or "") != ROADMAP:
            roadmap_mismatch_ids.append(artifact_id)
        if bool(row.get("safe_to_auto_start")) or bool(row.get("release_claim_allowed")):
            release_unsafe_ids.append(artifact_id)
        if bool(row.get("publishable")):
            publishable_ids.append(artifact_id)
        if not bool(row.get("not_release_evidence")):
            not_release_false_ids.append(artifact_id)
        if not bool(row.get("does_not_run_training")):
            no_training_false_ids.append(artifact_id)
        if not bool(row.get("does_not_run_cuda")):
            no_cuda_false_ids.append(artifact_id)
    unsafe_ids = _unique(
        [
            *missing_ids,
            *load_error_ids,
            *roadmap_mismatch_ids,
            *release_unsafe_ids,
            *publishable_ids,
            *not_release_false_ids,
            *no_training_false_ids,
            *no_cuda_false_ids,
        ]
    )
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_source_artifact_inventory_summary",
        "status": (
            "source_artifacts_loaded_fail_closed"
            if rows and not unsafe_ids
            else "source_artifacts_incomplete_or_unsafe"
        ),
        "source_artifact_count": len(rows),
        "present_source_artifact_count": sum(1 for row in rows if bool(row.get("exists"))),
        "missing_source_artifact_count": len(missing_ids),
        "load_error_count": len(load_error_ids),
        "roadmap_mismatch_count": len(roadmap_mismatch_ids),
        "release_unsafe_count": len(release_unsafe_ids),
        "publishable_count": len(publishable_ids),
        "not_release_evidence_false_count": len(not_release_false_ids),
        "does_not_run_training_false_count": len(no_training_false_ids),
        "does_not_run_cuda_false_count": len(no_cuda_false_ids),
        "unique_report_count": len(report_counts),
        "unsafe_source_artifact_count": len(unsafe_ids),
        "missing_source_artifact_ids": _unique(missing_ids)[:20],
        "load_error_source_artifact_ids": _unique(load_error_ids)[:20],
        "roadmap_mismatch_source_artifact_ids": _unique(roadmap_mismatch_ids)[:20],
        "release_unsafe_source_artifact_ids": _unique(release_unsafe_ids)[:20],
        "unsafe_source_artifact_ids": unsafe_ids[:50],
        "status_counts": dict(sorted(status_counts.items())),
        "report_counts": dict(sorted(report_counts.items())),
        "execution_policy": "read_only_source_artifact_inventory",
        "fail_closed": bool(rows and not unsafe_ids),
        "not_release_evidence": True,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "publishable": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
    }


def _check_source_artifact_inventory_summary(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_summary = _mapping(readiness.get("source_artifact_inventory_summary"))
    evidence_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_artifact_inventory_summary")
    )
    terminal_summary = _mapping(terminal.get("source_artifact_inventory_summary"))
    computed = _source_artifact_inventory_from_readiness(readiness)
    summary = _check_json_only_summary_mirror(
        failures,
        summary_id="source_artifact_inventory_summary",
        artifact_role="gpu_bubble_source_artifact_inventory_summary",
        readiness_summary=readiness_summary,
        evidence_summary=evidence_summary,
        terminal_summary=terminal_summary,
        string_fields=["status", "execution_policy"],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "publishable",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "source_artifact_count",
            "present_source_artifact_count",
            "missing_source_artifact_count",
            "load_error_count",
            "roadmap_mismatch_count",
            "release_unsafe_count",
            "publishable_count",
            "not_release_evidence_false_count",
            "does_not_run_training_false_count",
            "does_not_run_cuda_false_count",
            "unique_report_count",
            "unsafe_source_artifact_count",
        ],
        list_fields=[
            "missing_source_artifact_ids",
            "load_error_source_artifact_ids",
            "roadmap_mismatch_source_artifact_ids",
            "release_unsafe_source_artifact_ids",
            "unsafe_source_artifact_ids",
        ],
    )
    for field in [
        "source_artifact_count",
        "present_source_artifact_count",
        "missing_source_artifact_count",
        "load_error_count",
        "roadmap_mismatch_count",
        "release_unsafe_count",
        "publishable_count",
        "not_release_evidence_false_count",
        "does_not_run_training_false_count",
        "does_not_run_cuda_false_count",
        "unique_report_count",
        "unsafe_source_artifact_count",
    ]:
        _compare_scalar_field(
            failures,
            check_id=f"source_artifact_inventory_summary_{field}_computed_match",
            reason=f"source_artifact_inventory_summary_{field}_does_not_match_source_paths",
            left=_safe_int(readiness_summary.get(field)),
            right=_safe_int(computed.get(field)),
        )
    for field in [
        "missing_source_artifact_ids",
        "load_error_source_artifact_ids",
        "roadmap_mismatch_source_artifact_ids",
        "release_unsafe_source_artifact_ids",
        "unsafe_source_artifact_ids",
    ]:
        _compare_set_field(
            failures,
            check_id=f"source_artifact_inventory_summary_{field}_computed_match",
            reason=f"source_artifact_inventory_summary_{field}_does_not_match_source_paths",
            left=_strings(readiness_summary.get(field)),
            right=_strings(computed.get(field)),
        )
    summary["computed_source_artifact_count"] = _safe_int(computed.get("source_artifact_count"))
    summary["computed_unsafe_source_artifact_count"] = _safe_int(
        computed.get("unsafe_source_artifact_count")
    )
    return summary


def _evidence_summary_inventory_from_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        (str(key), value)
        for key, value in _mapping(readiness.get("evidence_summary")).items()
        if str(key) != "evidence_summary_inventory_summary"
    ]
    type_counts: dict[str, int] = {}
    roadmap_mismatch_ids: list[str] = []
    publishable_ids: list[str] = []
    wording_allowed_ids: list[str] = []
    not_release_false_ids: list[str] = []
    safe_to_auto_start_ids: list[str] = []
    release_claim_allowed_ids: list[str] = []
    for key, value in rows:
        if isinstance(value, Mapping):
            entry_type = "mapping"
            entry = _mapping(value)
            roadmap = str(entry.get("roadmap") or entry.get("expected_roadmap") or "")
            if roadmap and roadmap != ROADMAP:
                roadmap_mismatch_ids.append(key)
            if bool(entry.get("publishable")):
                publishable_ids.append(key)
            if bool(entry.get("safe_to_auto_start")):
                safe_to_auto_start_ids.append(key)
            if bool(entry.get("release_claim_allowed")) or bool(
                entry.get("release_claim_allowed_after_success")
            ):
                release_claim_allowed_ids.append(key)
            if bool(entry.get("release_gain_claim_wording_allowed")):
                wording_allowed_ids.append(key)
            if "not_release_evidence" in entry and not bool(entry.get("not_release_evidence")):
                not_release_false_ids.append(key)
        elif isinstance(value, list):
            entry_type = "list"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            entry_type = "scalar"
            if key in {"release_gain_claim_wording_allowed", "publishable"} and bool(value):
                wording_allowed_ids.append(key)
            if key == "safe_to_auto_start" and bool(value):
                safe_to_auto_start_ids.append(key)
            if key in {"release_claim_allowed", "release_claim_allowed_after_success"} and bool(value):
                release_claim_allowed_ids.append(key)
        else:
            entry_type = "other"
        type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
    release_unsafe_ids = _unique(
        [
            *publishable_ids,
            *wording_allowed_ids,
            *not_release_false_ids,
            *safe_to_auto_start_ids,
            *release_claim_allowed_ids,
        ]
    )
    unsafe_ids = _unique([*roadmap_mismatch_ids, *release_unsafe_ids])
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_evidence_summary_inventory_summary",
        "status": (
            "evidence_summary_fail_closed_non_release"
            if rows and not unsafe_ids
            else "evidence_summary_incomplete_or_unsafe"
        ),
        "evidence_key_count": len(rows),
        "mapping_entry_count": type_counts.get("mapping", 0),
        "scalar_entry_count": type_counts.get("scalar", 0),
        "list_entry_count": type_counts.get("list", 0),
        "other_entry_count": type_counts.get("other", 0),
        "roadmap_mismatch_entry_count": len(roadmap_mismatch_ids),
        "release_unsafe_entry_count": len(release_unsafe_ids),
        "publishable_entry_count": len(publishable_ids),
        "release_gain_wording_allowed_count": len(wording_allowed_ids),
        "not_release_evidence_false_entry_count": len(not_release_false_ids),
        "safe_to_auto_start_true_entry_count": len(safe_to_auto_start_ids),
        "release_claim_allowed_true_entry_count": len(release_claim_allowed_ids),
        "unsafe_evidence_entry_count": len(unsafe_ids),
        "roadmap_mismatch_entry_ids": _unique(roadmap_mismatch_ids)[:20],
        "release_unsafe_entry_ids": release_unsafe_ids[:20],
        "unsafe_evidence_entry_ids": unsafe_ids[:50],
        "entry_type_counts": dict(sorted(type_counts.items())),
        "execution_policy": "read_only_evidence_summary_inventory",
        "fail_closed": bool(rows and not unsafe_ids),
        "not_release_evidence": True,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "publishable": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
    }


def _check_evidence_summary_inventory_summary(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_summary = _mapping(readiness.get("evidence_summary_inventory_summary"))
    evidence_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get("evidence_summary_inventory_summary")
    )
    terminal_summary = _mapping(terminal.get("evidence_summary_inventory_summary"))
    computed = _evidence_summary_inventory_from_readiness(readiness)
    summary = _check_json_only_summary_mirror(
        failures,
        summary_id="evidence_summary_inventory_summary",
        artifact_role="gpu_bubble_evidence_summary_inventory_summary",
        readiness_summary=readiness_summary,
        evidence_summary=evidence_summary,
        terminal_summary=terminal_summary,
        string_fields=["status", "execution_policy"],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "publishable",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "evidence_key_count",
            "mapping_entry_count",
            "scalar_entry_count",
            "list_entry_count",
            "other_entry_count",
            "roadmap_mismatch_entry_count",
            "release_unsafe_entry_count",
            "publishable_entry_count",
            "release_gain_wording_allowed_count",
            "not_release_evidence_false_entry_count",
            "safe_to_auto_start_true_entry_count",
            "release_claim_allowed_true_entry_count",
            "unsafe_evidence_entry_count",
        ],
        list_fields=[
            "roadmap_mismatch_entry_ids",
            "release_unsafe_entry_ids",
            "unsafe_evidence_entry_ids",
        ],
    )
    for field in [
        "evidence_key_count",
        "mapping_entry_count",
        "scalar_entry_count",
        "list_entry_count",
        "other_entry_count",
        "roadmap_mismatch_entry_count",
        "release_unsafe_entry_count",
        "publishable_entry_count",
        "release_gain_wording_allowed_count",
        "not_release_evidence_false_entry_count",
        "safe_to_auto_start_true_entry_count",
        "release_claim_allowed_true_entry_count",
        "unsafe_evidence_entry_count",
    ]:
        _compare_scalar_field(
            failures,
            check_id=f"evidence_summary_inventory_summary_{field}_computed_match",
            reason=f"evidence_summary_inventory_summary_{field}_does_not_match_evidence_summary",
            left=_safe_int(readiness_summary.get(field)),
            right=_safe_int(computed.get(field)),
        )
    for field in [
        "roadmap_mismatch_entry_ids",
        "release_unsafe_entry_ids",
        "unsafe_evidence_entry_ids",
    ]:
        _compare_set_field(
            failures,
            check_id=f"evidence_summary_inventory_summary_{field}_computed_match",
            reason=f"evidence_summary_inventory_summary_{field}_does_not_match_evidence_summary",
            left=_strings(readiness_summary.get(field)),
            right=_strings(computed.get(field)),
        )
    summary["computed_evidence_key_count"] = _safe_int(computed.get("evidence_key_count"))
    summary["computed_unsafe_evidence_entry_count"] = _safe_int(
        computed.get("unsafe_evidence_entry_count")
    )
    return summary


def _check_roadmap_lineage_audit(failures: list[dict[str, Any]], terminal: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = _mapping(terminal.get("roadmap_lineage_audit"))
    if not audit:
        _failure(failures, "roadmap_lineage_audit", "terminal_roadmap_lineage_audit_missing")
        return {}
    if bool(audit.get("safe_to_auto_start")) or bool(audit.get("release_claim_allowed")):
        _failure(
            failures,
            "roadmap_lineage_fail_closed",
            "roadmap_lineage_audit_allows_release_or_auto_start",
        )
    if str(audit.get("expected_roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "roadmap_lineage_expected_roadmap",
            "roadmap_lineage_expected_roadmap_wrong",
            value=audit.get("expected_roadmap"),
        )
    if _safe_int(audit.get("missing_required_artifact_count")) != 0:
        _failure(
            failures,
            "roadmap_lineage_missing_required",
            "roadmap_lineage_required_artifact_missing",
            value=_strings(audit.get("missing_required_artifact_ids")),
        )
    if _safe_int(audit.get("mismatched_artifact_count")) != 0:
        _failure(
            failures,
            "roadmap_lineage_mismatch",
            "roadmap_lineage_artifact_roadmap_mismatch",
            value=_strings(audit.get("mismatched_artifact_ids")),
        )
    if not bool(audit.get("lineage_ok")):
        _failure(
            failures,
            "roadmap_lineage_ok",
            "roadmap_lineage_audit_not_ok",
            value={
                "missing": _strings(audit.get("missing_required_artifact_ids")),
                "mismatched": _strings(audit.get("mismatched_artifact_ids")),
            },
        )
    return audit


def _source_cache_pipeline_stage_lineage_summary(terminal: Mapping[str, Any]) -> dict[str, Any]:
    pipeline = _mapping(terminal.get("source_cache_axis_pipeline"))
    lineage = _mapping(pipeline.get("stage_roadmap_lineage"))
    return {
        "available": bool(lineage),
        "expected_roadmap": str(lineage.get("expected_roadmap") or ""),
        "lineage_ok": bool(lineage.get("lineage_ok")),
        "stage_count": _safe_int(lineage.get("stage_count")),
        "pipeline_stage_count": _safe_int(pipeline.get("stage_count")),
        "roadmap_mismatch_count": _safe_int(lineage.get("roadmap_mismatch_count")),
        "roadmap_mismatch_stage_ids": _strings(lineage.get("roadmap_mismatch_stage_ids"))[:20],
        "safe_to_auto_start": bool(lineage.get("safe_to_auto_start")),
        "release_claim_allowed": bool(lineage.get("release_claim_allowed")),
        "not_release_evidence": bool(lineage.get("not_release_evidence")),
    }


def _check_source_cache_pipeline_stage_lineage(
    failures: list[dict[str, Any]],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _source_cache_pipeline_stage_lineage_summary(terminal)
    if not bool(summary.get("available")):
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage",
            "terminal_source_cache_pipeline_stage_lineage_missing",
        )
        return summary
    if bool(summary.get("safe_to_auto_start")) or bool(summary.get("release_claim_allowed")) or not bool(
        summary.get("not_release_evidence")
    ):
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_fail_closed",
            "source_cache_pipeline_stage_lineage_not_fail_closed",
            value={
                "safe_to_auto_start": summary.get("safe_to_auto_start"),
                "release_claim_allowed": summary.get("release_claim_allowed"),
                "not_release_evidence": summary.get("not_release_evidence"),
            },
        )
    if str(summary.get("expected_roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_expected_roadmap",
            "source_cache_pipeline_stage_lineage_expected_roadmap_wrong",
            value=summary.get("expected_roadmap"),
        )
    if _safe_int(summary.get("stage_count")) <= 0:
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_stage_count",
            "source_cache_pipeline_stage_lineage_stage_count_missing",
            value=summary.get("stage_count"),
        )
    if _safe_int(summary.get("stage_count")) != _safe_int(summary.get("pipeline_stage_count")):
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_stage_count_match",
            "source_cache_pipeline_stage_lineage_stage_count_does_not_match_pipeline",
            value={
                "lineage": summary.get("stage_count"),
                "pipeline": summary.get("pipeline_stage_count"),
            },
        )
    if _safe_int(summary.get("roadmap_mismatch_count")) != 0 or _strings(
        summary.get("roadmap_mismatch_stage_ids")
    ):
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_mismatch",
            "source_cache_pipeline_stage_lineage_roadmap_mismatch",
            value={
                "roadmap_mismatch_count": summary.get("roadmap_mismatch_count"),
                "roadmap_mismatch_stage_ids": _strings(summary.get("roadmap_mismatch_stage_ids")),
            },
        )
    if not bool(summary.get("lineage_ok")):
        _failure(
            failures,
            "source_cache_pipeline_stage_lineage_ok",
            "source_cache_pipeline_stage_lineage_not_ok",
            value={
                "roadmap_mismatch_count": summary.get("roadmap_mismatch_count"),
                "roadmap_mismatch_stage_ids": _strings(summary.get("roadmap_mismatch_stage_ids")),
            },
        )
    return summary


def _check_first_release_policy(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    first_release = _mapping(readiness.get("first_release_scope"))
    remaining = _mapping(readiness.get("remaining_work_summary"))
    terminal_unblocker = _mapping(terminal.get("release_unblocker_summary"))
    readiness_policy_summary = _mapping(readiness.get("first_release_policy_summary"))
    readiness_evidence_policy_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get("first_release_policy_summary")
    )
    terminal_policy_summary = _mapping(terminal.get("first_release_policy_summary"))
    first_policy = str(first_release.get("recommended_release_policy") or "")
    remaining_policy = str(remaining.get("recommended_release_policy") or "")
    terminal_policy = str(terminal.get("recommended_release_policy") or "")
    terminal_unblocker_policy = str(terminal_unblocker.get("recommended_release_policy") or "")
    first_gate_ids = _strings(first_release.get("gpu_bubble_release_hard_gate_ids"))
    expected_gates = _strings(readiness_hard_gates)
    first_release_policy_unsafe = (
        str(first_release.get("scope") or "") != "gpu_bubble_readiness_only"
        or bool(first_release.get("stable_first_release_blocked_by_this_artifact"))
        or bool(first_release.get("gpu_bubble_release_claim_allowed"))
        or (bool(expected_gates) and not bool(first_release.get("gpu_bubble_release_claim_blocked")))
        or (bool(expected_gates) and set(first_gate_ids) != set(expected_gates))
        or (bool(expected_gates) and first_policy != BLOCKED_RELEASE_POLICY)
        or str(first_release.get("claim_publication_scope") or "") != "non_release_benchmark_claims"
        or not bool(first_release.get("does_not_prove_global_product_release"))
    )
    summary = {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_first_release_policy_summary",
        "scope": str(first_release.get("scope") or ""),
        "stable_first_release_blocked_by_this_artifact": bool(
            first_release.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_allowed": bool(first_release.get("gpu_bubble_release_claim_allowed")),
        "gpu_bubble_release_claim_blocked": bool(first_release.get("gpu_bubble_release_claim_blocked")),
        "gpu_bubble_release_hard_gate_count": len(first_gate_ids),
        "gpu_bubble_release_hard_gate_ids": first_gate_ids[:20],
        "readiness_hard_gate_ids": expected_gates[:20],
        "recommended_release_policy": first_policy,
        "remaining_recommended_release_policy": remaining_policy,
        "terminal_recommended_release_policy": terminal_policy,
        "terminal_unblocker_recommended_release_policy": terminal_unblocker_policy,
        "claim_publication_scope": str(first_release.get("claim_publication_scope") or ""),
        "terminal_claim_publication_scope": str(terminal.get("claim_publication_scope") or ""),
        "does_not_prove_global_product_release": bool(
            first_release.get("does_not_prove_global_product_release")
        ),
        "unsafe_policy_count": 1 if first_release_policy_unsafe else 0,
        "unsafe_policy_ids": ["first_release_policy_scope_or_claim_drift"]
        if first_release_policy_unsafe
        else [],
        "fail_closed": not first_release_policy_unsafe,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }
    if str(first_release.get("scope") or "") != "gpu_bubble_readiness_only":
        _failure(
            failures,
            "first_release_policy_scope",
            "first_release_scope_is_not_gpu_bubble_readiness_only",
            value=first_release.get("scope"),
        )
    if bool(first_release.get("stable_first_release_blocked_by_this_artifact")) or bool(
        terminal.get("stable_first_release_blocked_by_this_artifact")
    ):
        _failure(
            failures,
            "first_release_policy_stable_baseline",
            "gpu_bubble_artifact_blocks_stable_first_release",
        )
    if bool(first_release.get("gpu_bubble_release_claim_allowed")):
        _failure(
            failures,
            "first_release_policy_claim_allowed",
            "first_release_scope_allows_gpu_bubble_release_claim",
        )
    if expected_gates and not bool(first_release.get("gpu_bubble_release_claim_blocked")):
        _failure(
            failures,
            "first_release_policy_claim_blocked",
            "gpu_bubble_claim_not_blocked_despite_hard_gates",
            value=expected_gates,
        )
    if expected_gates and set(first_gate_ids) != set(expected_gates):
        _failure(
            failures,
            "first_release_policy_hard_gates_match",
            "first_release_scope_hard_gates_do_not_match_readiness",
            value={"first_release_scope": first_gate_ids, "readiness": expected_gates},
        )
    if expected_gates and first_policy != BLOCKED_RELEASE_POLICY:
        _failure(
            failures,
            "first_release_policy_recommendation",
            "first_release_policy_does_not_ship_baseline_without_gpu_bubble_claim",
            value=first_policy,
        )
    for check_id, policy in (
        ("first_release_policy_remaining_match", remaining_policy),
        ("first_release_policy_terminal_match", terminal_policy),
        ("first_release_policy_terminal_unblocker_match", terminal_unblocker_policy),
    ):
        if policy and first_policy and policy != first_policy:
            _failure(
                failures,
                check_id,
                "first_release_policy_summary_drift",
                value={"first_release_scope": first_policy, "observed": policy},
            )
    if str(first_release.get("claim_publication_scope") or "") != "non_release_benchmark_claims":
        _failure(
            failures,
            "first_release_policy_claim_scope",
            "first_release_claim_scope_is_not_non_release_benchmark_claims",
            value=first_release.get("claim_publication_scope"),
        )
    if not bool(first_release.get("does_not_prove_global_product_release")):
        _failure(
            failures,
            "first_release_policy_global_release_scope",
            "first_release_scope_does_not_mark_global_release_as_unproven",
        )
    policy_targets = {
        "readiness_first_release_policy_summary": readiness_policy_summary,
        "readiness_evidence_first_release_policy_summary": readiness_evidence_policy_summary,
        "terminal_first_release_policy_summary": terminal_policy_summary,
    }
    for check_id, candidate in policy_targets.items():
        if not candidate:
            _failure(failures, check_id, "first_release_policy_summary_missing")
            continue
        if str(candidate.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "first_release_policy_summary_roadmap_missing_or_wrong",
                value=candidate.get("roadmap"),
            )
        if (
            bool(candidate.get("safe_to_auto_start"))
            or bool(candidate.get("release_claim_allowed"))
            or not bool(candidate.get("not_release_evidence"))
            or not bool(candidate.get("fail_closed"))
            or _safe_int(candidate.get("unsafe_policy_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "first_release_policy_summary_allows_release_or_auto_start",
            )
        for field in [
            "scope",
            "recommended_release_policy",
            "claim_publication_scope",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"first_release_policy_summary_{field}_does_not_match_computed",
                left=str(candidate.get(field) or ""),
                right=str(summary.get(field) or ""),
            )
        for field in [
            "stable_first_release_blocked_by_this_artifact",
            "gpu_bubble_release_claim_allowed",
            "gpu_bubble_release_claim_blocked",
            "does_not_prove_global_product_release",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"first_release_policy_summary_{field}_does_not_match_computed",
                left=bool(candidate.get(field)),
                right=bool(summary.get(field)),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_gpu_bubble_release_hard_gate_count_match",
            reason="first_release_policy_summary_hard_gate_count_does_not_match_computed",
            left=_safe_int(candidate.get("gpu_bubble_release_hard_gate_count")),
            right=len(first_gate_ids),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_gpu_bubble_release_hard_gate_ids_match",
            reason="first_release_policy_summary_hard_gate_ids_do_not_match_computed",
            left=_strings(candidate.get("gpu_bubble_release_hard_gate_ids")),
            right=first_gate_ids,
        )
    return summary


def _normalize_claim_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _claim_like_text_rows(artifact_id: str, artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in _list(artifact.get("proven_claims")):
        item = _mapping(raw)
        text = " ".join(
            part
            for part in (
                str(item.get("case_id") or ""),
                str(item.get("label") or ""),
                str(item.get("claim_scope") or ""),
            )
            if part
        )
        rows.append({"artifact": artifact_id, "source": "proven_claims", "id": str(item.get("case_id") or ""), "text": text})
    for raw in _list(artifact.get("non_release_supported_claims")):
        item = _mapping(raw)
        rows.append(
            {
                "artifact": artifact_id,
                "source": "non_release_supported_claims",
                "id": str(item.get("id") or ""),
                "text": str(item.get("claim") or ""),
            }
        )
    for index, note in enumerate(_strings(artifact.get("notes"))):
        rows.append({"artifact": artifact_id, "source": "notes", "id": str(index), "text": note})
    return rows


def _scan_forbidden_claim_wording(*, readiness: Mapping[str, Any], terminal: Mapping[str, Any]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for row in [
        *_claim_like_text_rows("readiness", readiness),
        *_claim_like_text_rows("terminal", terminal),
    ]:
        normalized = _normalize_claim_text(row["text"])
        for token in FORBIDDEN_CLAIM_WORDING_TOKENS:
            if _normalize_claim_text(token) in normalized:
                hits.append({**row, "token": token})
    return hits


def _check_claim_wording_policy(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_audit = _mapping(readiness.get("forbidden_claim_wording_audit"))
    terminal_audit = _mapping(terminal.get("forbidden_claim_wording_audit"))
    scan_hits = _scan_forbidden_claim_wording(readiness=readiness, terminal=terminal)
    summary = {
        "claim_wording_policy": str(readiness.get("claim_wording_policy") or ""),
        "terminal_claim_wording_policy": str(terminal.get("claim_wording_policy") or ""),
        "release_gain_claim_wording_allowed": bool(readiness.get("release_gain_claim_wording_allowed")),
        "terminal_release_gain_claim_wording_allowed": bool(
            terminal.get("release_gain_claim_wording_allowed")
        ),
        "forbidden_claim_wording_hit_count": _safe_int(
            readiness.get("forbidden_claim_wording_hit_count"),
            _safe_int(readiness_audit.get("forbidden_claim_wording_hit_count")),
        ),
        "terminal_forbidden_claim_wording_hit_count": _safe_int(
            terminal.get("forbidden_claim_wording_hit_count"),
            _safe_int(terminal_audit.get("forbidden_claim_wording_hit_count")),
        ),
        "guard_rescan_hit_count": len(scan_hits),
        "guard_rescan_hits": scan_hits[:20],
        "forbidden_claim_wording_tokens": list(FORBIDDEN_CLAIM_WORDING_TOKENS),
        "readiness_not_release_evidence": bool(readiness_audit.get("not_release_evidence")),
        "terminal_not_release_evidence": bool(terminal_audit.get("not_release_evidence")),
        "readiness_safe_to_auto_start": bool(readiness_audit.get("safe_to_auto_start")),
        "terminal_safe_to_auto_start": bool(terminal_audit.get("safe_to_auto_start")),
        "readiness_release_claim_allowed": bool(readiness_audit.get("release_claim_allowed")),
        "terminal_release_claim_allowed": bool(terminal_audit.get("release_claim_allowed")),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }
    if summary["claim_wording_policy"] != CLAIM_WORDING_POLICY:
        _failure(
            failures,
            "claim_wording_policy",
            "readiness_claim_wording_policy_wrong_or_missing",
            value=summary["claim_wording_policy"],
        )
    if summary["terminal_claim_wording_policy"] != CLAIM_WORDING_POLICY:
        _failure(
            failures,
            "terminal_claim_wording_policy",
            "terminal_claim_wording_policy_wrong_or_missing",
            value=summary["terminal_claim_wording_policy"],
        )
    if bool(summary["release_gain_claim_wording_allowed"]) or bool(
        summary["terminal_release_gain_claim_wording_allowed"]
    ):
        _failure(
            failures,
            "claim_wording_release_gain_allowed",
            "release_gain_claim_wording_allowed",
            value={
                "readiness": summary["release_gain_claim_wording_allowed"],
                "terminal": summary["terminal_release_gain_claim_wording_allowed"],
            },
        )
    if _safe_int(summary["forbidden_claim_wording_hit_count"]) != 0:
        _failure(
            failures,
            "claim_wording_readiness_hits",
            "readiness_forbidden_claim_wording_hits_present",
            value=_list(readiness_audit.get("forbidden_claim_wording_hits")),
        )
    if _safe_int(summary["terminal_forbidden_claim_wording_hit_count"]) != 0:
        _failure(
            failures,
            "claim_wording_terminal_hits",
            "terminal_forbidden_claim_wording_hits_present",
            value=_list(terminal_audit.get("forbidden_claim_wording_hits")),
        )
    if scan_hits:
        _failure(
            failures,
            "claim_wording_guard_rescan_hits",
            "guard_rescan_found_forbidden_claim_wording",
            value=scan_hits[:20],
        )
    if not bool(summary["readiness_not_release_evidence"]) or not bool(
        summary["terminal_not_release_evidence"]
    ):
        _failure(
            failures,
            "claim_wording_not_release_evidence",
            "forbidden_claim_wording_audit_not_marked_non_release",
            value={
                "readiness": summary["readiness_not_release_evidence"],
                "terminal": summary["terminal_not_release_evidence"],
            },
        )
    if (
        bool(summary["readiness_safe_to_auto_start"])
        or bool(summary["terminal_safe_to_auto_start"])
        or bool(summary["readiness_release_claim_allowed"])
        or bool(summary["terminal_release_claim_allowed"])
    ):
        _failure(
            failures,
            "claim_wording_audit_allows_release_or_auto_start",
            "forbidden_claim_wording_audit_allows_release_or_auto_start",
            value={
                "readiness_safe_to_auto_start": summary["readiness_safe_to_auto_start"],
                "terminal_safe_to_auto_start": summary["terminal_safe_to_auto_start"],
                "readiness_release_claim_allowed": summary["readiness_release_claim_allowed"],
                "terminal_release_claim_allowed": summary["terminal_release_claim_allowed"],
            },
        )
    return summary


def _compare_set_field(
    failures: list[dict[str, Any]],
    *,
    check_id: str,
    reason: str,
    left: Sequence[Any],
    right: Sequence[Any],
) -> None:
    left_values = _strings(left)
    right_values = _strings(right)
    if set(left_values) != set(right_values):
        _failure(failures, check_id, reason, value={"readiness": left_values, "terminal": right_values})


def _compare_scalar_field(
    failures: list[dict[str, Any]],
    *,
    check_id: str,
    reason: str,
    left: Any,
    right: Any,
) -> None:
    if left != right:
        _failure(failures, check_id, reason, value={"readiness": left, "terminal": right})


def _check_json_only_summary_mirror(
    failures: list[dict[str, Any]],
    *,
    summary_id: str,
    artifact_role: str,
    readiness_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    terminal_summary: Mapping[str, Any],
    string_fields: Sequence[str] = (),
    bool_fields: Sequence[str] = (),
    int_fields: Sequence[str] = (),
    list_fields: Sequence[str] = (),
) -> dict[str, Any]:
    computed = _mapping(readiness_summary) or _mapping(evidence_summary)
    summaries = {
        f"readiness_{summary_id}": _mapping(readiness_summary),
        f"readiness_evidence_{summary_id}": _mapping(evidence_summary),
        f"terminal_{summary_id}": _mapping(terminal_summary),
    }
    compact = {
        "available": bool(computed),
        "terminal_available": bool(terminal_summary),
        "roadmap": str(_mapping(terminal_summary).get("roadmap") or ""),
        "artifact_role": str(_mapping(terminal_summary).get("artifact_role") or ""),
        "status": str(_mapping(terminal_summary).get("status") or ""),
        "computed_status": str(computed.get("status") or ""),
        "fail_closed": bool(_mapping(terminal_summary).get("fail_closed")),
        "computed_fail_closed": bool(computed.get("fail_closed")),
        "not_release_evidence": bool(_mapping(terminal_summary).get("not_release_evidence")),
        "safe_to_auto_start": bool(_mapping(terminal_summary).get("safe_to_auto_start")),
        "release_claim_allowed": bool(_mapping(terminal_summary).get("release_claim_allowed")),
        "does_not_run_training": bool(_mapping(terminal_summary).get("does_not_run_training")),
        "does_not_run_cuda": bool(_mapping(terminal_summary).get("does_not_run_cuda")),
    }
    if "publishable" in computed or "publishable" in _mapping(terminal_summary):
        compact["publishable"] = bool(_mapping(terminal_summary).get("publishable"))
    terminal_compact = _mapping(terminal_summary)
    for field in string_fields:
        compact[field] = str(terminal_compact.get(field) or "")
        compact[f"computed_{field}"] = str(computed.get(field) or "")
    for field in bool_fields:
        compact[field] = bool(terminal_compact.get(field))
        compact[f"computed_{field}"] = bool(computed.get(field))
    for field in int_fields:
        compact[field] = _safe_int(terminal_compact.get(field))
        compact[f"computed_{field}"] = _safe_int(computed.get(field))
    for field in list_fields:
        compact[field] = _strings(terminal_compact.get(field))
        compact[f"computed_{field}"] = _strings(computed.get(field))
    for check_id, summary in summaries.items():
        if not summary:
            _failure(failures, check_id, f"{summary_id}_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(failures, check_id, f"{summary_id}_roadmap_missing_or_wrong", value=summary.get("roadmap"))
        if str(summary.get("artifact_role") or "") != artifact_role:
            _failure(failures, check_id, f"{summary_id}_artifact_role_wrong", value=summary.get("artifact_role"))
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or bool(summary.get("publishable"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("does_not_run_training"))
            or not bool(summary.get("does_not_run_cuda"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, f"{summary_id}_allows_release_or_auto_start")
        for field in string_fields:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"{summary_id}_{field}_does_not_match_readiness",
                left=str(summary.get(field) or ""),
                right=str(computed.get(field) or ""),
            )
        for field in bool_fields:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"{summary_id}_{field}_does_not_match_readiness",
                left=bool(summary.get(field)),
                right=bool(computed.get(field)),
            )
        for field in int_fields:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"{summary_id}_{field}_does_not_match_readiness",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed.get(field)),
            )
        for field in list_fields:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"{summary_id}_{field}_does_not_match_readiness",
                left=_strings(summary.get(field)),
                right=_strings(computed.get(field)),
            )
    return compact


def _check_roadmap_acceptance_gate_summary(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    summary = _mapping(readiness.get("roadmap_acceptance_gate_summary"))
    terminal_summary = _mapping(terminal.get("roadmap_acceptance_gate_summary"))
    compact = {
        "available": bool(summary),
        "terminal_available": bool(terminal_summary),
        "roadmap": str(summary.get("roadmap") or ""),
        "terminal_roadmap": str(terminal_summary.get("roadmap") or ""),
        "metric_contract_status": str(summary.get("metric_contract_status") or ""),
        "experiment_matrix_status": str(summary.get("experiment_matrix_status") or ""),
        "acceptance_gate_status": str(summary.get("acceptance_gate_status") or ""),
        "required_metric_count": _safe_int(summary.get("required_metric_count")),
        "required_metric_ids": _strings(summary.get("required_metric_ids"))[:30],
        "required_experiment_batch_count": _safe_int(
            summary.get("required_experiment_batch_count")
        ),
        "required_experiment_batch_ids": _strings(summary.get("required_experiment_batch_ids"))[:20],
        "required_acceptance_gate_count": _safe_int(
            summary.get("required_acceptance_gate_count")
        ),
        "required_acceptance_gate_ids": _strings(summary.get("required_acceptance_gate_ids"))[:30],
        "blocked_acceptance_gate_count": _safe_int(
            summary.get("blocked_acceptance_gate_count")
        ),
        "blocked_acceptance_gate_ids": _strings(summary.get("blocked_acceptance_gate_ids"))[:30],
        "gpu_bubble_release_hard_gate_ids": _strings(
            summary.get("gpu_bubble_release_hard_gate_ids")
        )[:20],
        "terminal_blocked_acceptance_gate_count": _safe_int(
            terminal_summary.get("blocked_acceptance_gate_count")
        ),
        "fail_closed": bool(summary.get("fail_closed")),
        "terminal_fail_closed": bool(terminal_summary.get("fail_closed")),
        "not_release_evidence": bool(summary.get("not_release_evidence")),
        "terminal_not_release_evidence": bool(terminal_summary.get("not_release_evidence")),
        "safe_to_auto_start": bool(summary.get("safe_to_auto_start")),
        "terminal_safe_to_auto_start": bool(terminal_summary.get("safe_to_auto_start")),
        "release_claim_allowed": bool(summary.get("release_claim_allowed")),
        "terminal_release_claim_allowed": bool(terminal_summary.get("release_claim_allowed")),
        "ready_for_recommended_sorting": bool(summary.get("ready_for_recommended_sorting")),
        "ready_for_ui_advisor_stable_strategy": bool(
            summary.get("ready_for_ui_advisor_stable_strategy")
        ),
        "next_allowed_stage": str(summary.get("next_allowed_stage") or ""),
    }
    if not summary:
        _failure(failures, "roadmap_acceptance_gate_summary", "readiness_acceptance_summary_missing")
        return compact
    if not terminal_summary:
        _failure(failures, "terminal_roadmap_acceptance_gate_summary", "terminal_acceptance_summary_missing")
        return compact

    if str(summary.get("roadmap") or "") != ROADMAP or str(terminal_summary.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "roadmap_acceptance_roadmap",
            "acceptance_summary_roadmap_missing_or_wrong",
            value={"readiness": summary.get("roadmap"), "terminal": terminal_summary.get("roadmap")},
        )
    if str(summary.get("artifact_role") or "") != "gpu_bubble_roadmap_acceptance_gate_summary":
        _failure(
            failures,
            "roadmap_acceptance_artifact_role",
            "readiness_acceptance_summary_role_wrong",
            value=summary.get("artifact_role"),
        )
    if not bool(summary.get("fail_closed")) or not bool(terminal_summary.get("fail_closed")):
        _failure(
            failures,
            "roadmap_acceptance_fail_closed",
            "acceptance_summary_not_fail_closed",
            value={"readiness": summary.get("fail_closed"), "terminal": terminal_summary.get("fail_closed")},
        )
    if (
        bool(summary.get("safe_to_auto_start"))
        or bool(terminal_summary.get("safe_to_auto_start"))
        or bool(summary.get("release_claim_allowed"))
        or bool(terminal_summary.get("release_claim_allowed"))
        or not bool(summary.get("not_release_evidence"))
        or not bool(terminal_summary.get("not_release_evidence"))
    ):
        _failure(
            failures,
            "roadmap_acceptance_fail_closed_flags",
            "acceptance_summary_allows_release_or_auto_start",
        )
    if bool(summary.get("ready_for_recommended_sorting")) or bool(
        terminal_summary.get("ready_for_recommended_sorting")
    ):
        _failure(
            failures,
            "roadmap_acceptance_recommended_sorting",
            "acceptance_summary_allows_recommended_sorting_before_gates_clear",
        )
    if bool(summary.get("ready_for_ui_advisor_stable_strategy")) or bool(
        terminal_summary.get("ready_for_ui_advisor_stable_strategy")
    ):
        _failure(
            failures,
            "roadmap_acceptance_ui_advisor",
            "acceptance_summary_allows_stable_ui_advisor_before_gates_clear",
        )
    expected_counts = (
        ("metric", REQUIRED_METRIC_IDS, "required_metric_count"),
        ("experiment_batch", REQUIRED_EXPERIMENT_BATCH_IDS, "required_experiment_batch_count"),
        ("acceptance_gate", REQUIRED_ACCEPTANCE_GATE_IDS, "required_acceptance_gate_count"),
    )
    for label, expected, count_key in expected_counts:
        expected_count = len(expected)
        if _safe_int(summary.get(count_key)) != expected_count or _safe_int(
            terminal_summary.get(count_key)
        ) != expected_count:
            _failure(
                failures,
                f"roadmap_acceptance_{label}_count",
                "acceptance_summary_required_count_does_not_match_contract",
                value={
                    "expected": expected_count,
                    "readiness": summary.get(count_key),
                    "terminal": terminal_summary.get(count_key),
                },
            )
    hard_gates_present = bool(readiness_hard_gates)
    if hard_gates_present and (
        str(summary.get("acceptance_gate_status") or "") != "blocked_pending_release_hard_gates"
        or str(terminal_summary.get("acceptance_gate_status") or "") != "blocked_pending_release_hard_gates"
    ):
        _failure(
            failures,
            "roadmap_acceptance_status",
            "acceptance_summary_status_not_blocked_while_hard_gates_present",
            value={
                "readiness": summary.get("acceptance_gate_status"),
                "terminal": terminal_summary.get("acceptance_gate_status"),
            },
        )
    if _strings(summary.get("missing_external_inputs")) and (
        str(summary.get("next_allowed_stage") or "") != "provide_external_inputs_or_refresh_json_admission"
        or str(terminal_summary.get("next_allowed_stage") or "") != "provide_external_inputs_or_refresh_json_admission"
    ):
        _failure(
            failures,
            "roadmap_acceptance_next_stage",
            "acceptance_summary_next_stage_does_not_match_missing_external_inputs",
            value={
                "readiness": summary.get("next_allowed_stage"),
                "terminal": terminal_summary.get("next_allowed_stage"),
            },
        )

    expected_sets = (
        ("metric", REQUIRED_METRIC_IDS, summary.get("required_metric_ids"), terminal_summary.get("required_metric_ids")),
        (
            "experiment_batch",
            REQUIRED_EXPERIMENT_BATCH_IDS,
            summary.get("required_experiment_batch_ids"),
            terminal_summary.get("required_experiment_batch_ids"),
        ),
        (
            "acceptance_gate",
            REQUIRED_ACCEPTANCE_GATE_IDS,
            summary.get("required_acceptance_gate_ids"),
            terminal_summary.get("required_acceptance_gate_ids"),
        ),
    )
    for label, expected, readiness_values, terminal_values in expected_sets:
        readiness_set = set(_strings(readiness_values))
        terminal_set = set(_strings(terminal_values))
        expected_set = set(expected)
        if readiness_set != expected_set or terminal_set != expected_set:
            _failure(
                failures,
                f"roadmap_acceptance_{label}_ids",
                "acceptance_summary_required_ids_do_not_match_roadmap_contract",
                value={
                    "expected": list(expected),
                    "readiness": sorted(readiness_set),
                    "terminal": sorted(terminal_set),
                },
            )

    _compare_set_field(
        failures,
        check_id="roadmap_acceptance_hard_gates_match",
        reason="acceptance_summary_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(summary.get("gpu_bubble_release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="terminal_roadmap_acceptance_hard_gates_match",
        reason="terminal_acceptance_summary_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(terminal_summary.get("gpu_bubble_release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="roadmap_acceptance_blocked_gates_match",
        reason="terminal_acceptance_blocked_gates_do_not_match_readiness",
        left=_strings(summary.get("blocked_acceptance_gate_ids")),
        right=_strings(terminal_summary.get("blocked_acceptance_gate_ids")),
    )
    blocked_ids = set(_strings(summary.get("blocked_acceptance_gate_ids")))
    if readiness_hard_gates and not set(_strings(readiness_hard_gates)).issubset(blocked_ids):
        _failure(
            failures,
            "roadmap_acceptance_blocked_hard_gates",
            "acceptance_summary_does_not_include_release_hard_gates_as_blockers",
            value={"hard_gates": _strings(readiness_hard_gates), "blocked": sorted(blocked_ids)},
        )
    if _safe_int(summary.get("json_ready_action_count")) != 0:
        _failure(
            failures,
            "roadmap_acceptance_json_ready",
            "acceptance_summary_reports_json_ready_actions",
            value=summary.get("json_ready_action_count"),
        )
    if _safe_int(summary.get("auto_startable_gpu_action_count")) != 0 or _safe_int(
        summary.get("release_claim_allowed_after_success_action_count")
    ) != 0:
        _failure(
            failures,
            "roadmap_acceptance_manual_gpu_policy",
            "acceptance_summary_allows_auto_gpu_or_release_after_success",
        )
    return compact


def _check_roadmap_execution_contract_summary(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    summary = _mapping(readiness.get("roadmap_execution_contract_summary"))
    terminal_summary = _mapping(terminal.get("roadmap_execution_contract_summary"))
    compact = {
        "available": bool(summary),
        "terminal_available": bool(terminal_summary),
        "roadmap": str(summary.get("roadmap") or ""),
        "terminal_roadmap": str(terminal_summary.get("roadmap") or ""),
        "attribution_rule_count": _safe_int(summary.get("attribution_rule_count")),
        "attribution_rule_ids": _strings(summary.get("attribution_rule_ids"))[:20],
        "family_strategy_count": _safe_int(summary.get("family_strategy_count")),
        "family_strategy_ids": _strings(summary.get("family_strategy_ids"))[:20],
        "progression_phase_count": _safe_int(summary.get("progression_phase_count")),
        "progression_phase_ids": _strings(summary.get("progression_phase_ids"))[:20],
        "current_progression_phase_id": str(summary.get("current_progression_phase_id") or ""),
        "progression_status": str(summary.get("progression_status") or ""),
        "next_allowed_stage": str(summary.get("next_allowed_stage") or ""),
        "gpu_bubble_release_hard_gate_ids": _strings(
            summary.get("gpu_bubble_release_hard_gate_ids")
        )[:20],
        "fail_closed": bool(summary.get("fail_closed")),
        "terminal_fail_closed": bool(terminal_summary.get("fail_closed")),
        "not_release_evidence": bool(summary.get("not_release_evidence")),
        "terminal_not_release_evidence": bool(terminal_summary.get("not_release_evidence")),
        "safe_to_auto_start": bool(summary.get("safe_to_auto_start")),
        "terminal_safe_to_auto_start": bool(terminal_summary.get("safe_to_auto_start")),
        "release_claim_allowed": bool(summary.get("release_claim_allowed")),
        "terminal_release_claim_allowed": bool(terminal_summary.get("release_claim_allowed")),
        "ready_for_combined_strategy": bool(summary.get("ready_for_combined_strategy")),
        "ready_for_long_training_validation": bool(summary.get("ready_for_long_training_validation")),
        "ready_for_ui_advisor_stable_strategy": bool(
            summary.get("ready_for_ui_advisor_stable_strategy")
        ),
    }
    if not summary:
        _failure(failures, "roadmap_execution_contract_summary", "readiness_execution_summary_missing")
        return compact
    if not terminal_summary:
        _failure(
            failures,
            "terminal_roadmap_execution_contract_summary",
            "terminal_execution_summary_missing",
        )
        return compact
    if str(summary.get("roadmap") or "") != ROADMAP or str(terminal_summary.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "roadmap_execution_roadmap",
            "execution_summary_roadmap_missing_or_wrong",
            value={"readiness": summary.get("roadmap"), "terminal": terminal_summary.get("roadmap")},
        )
    if str(summary.get("artifact_role") or "") != "gpu_bubble_roadmap_execution_contract_summary":
        _failure(
            failures,
            "roadmap_execution_artifact_role",
            "readiness_execution_summary_role_wrong",
            value=summary.get("artifact_role"),
        )
    if not bool(summary.get("fail_closed")) or not bool(terminal_summary.get("fail_closed")):
        _failure(
            failures,
            "roadmap_execution_fail_closed",
            "execution_summary_not_fail_closed",
            value={"readiness": summary.get("fail_closed"), "terminal": terminal_summary.get("fail_closed")},
        )
    if (
        bool(summary.get("safe_to_auto_start"))
        or bool(terminal_summary.get("safe_to_auto_start"))
        or bool(summary.get("release_claim_allowed"))
        or bool(terminal_summary.get("release_claim_allowed"))
        or not bool(summary.get("not_release_evidence"))
        or not bool(terminal_summary.get("not_release_evidence"))
    ):
        _failure(
            failures,
            "roadmap_execution_fail_closed_flags",
            "execution_summary_allows_release_or_auto_start",
        )
    if bool(summary.get("ready_for_combined_strategy")) or bool(
        terminal_summary.get("ready_for_combined_strategy")
    ):
        _failure(
            failures,
            "roadmap_execution_combined_strategy",
            "execution_summary_allows_combined_strategy_before_single_factor_evidence",
        )
    if bool(summary.get("ready_for_long_training_validation")) or bool(
        terminal_summary.get("ready_for_long_training_validation")
    ):
        _failure(
            failures,
            "roadmap_execution_long_validation",
            "execution_summary_allows_long_training_validation_before_short_benchmark_gates",
        )
    if bool(summary.get("ready_for_ui_advisor_stable_strategy")) or bool(
        terminal_summary.get("ready_for_ui_advisor_stable_strategy")
    ):
        _failure(
            failures,
            "roadmap_execution_ui_advisor",
            "execution_summary_allows_stable_ui_advisor_before_gates_clear",
        )

    expected_sets = (
        ("attribution_rule", REQUIRED_ATTRIBUTION_RULE_IDS, summary.get("attribution_rule_ids"), terminal_summary.get("attribution_rule_ids")),
        ("family_strategy", REQUIRED_FAMILY_STRATEGY_IDS, summary.get("family_strategy_ids"), terminal_summary.get("family_strategy_ids")),
        ("progression_phase", REQUIRED_PROGRESSION_PHASE_IDS, summary.get("progression_phase_ids"), terminal_summary.get("progression_phase_ids")),
    )
    for label, expected, readiness_values, terminal_values in expected_sets:
        readiness_set = set(_strings(readiness_values))
        terminal_set = set(_strings(terminal_values))
        expected_set = set(expected)
        if readiness_set != expected_set or terminal_set != expected_set:
            _failure(
                failures,
                f"roadmap_execution_{label}_ids",
                "execution_summary_required_ids_do_not_match_roadmap_contract",
                value={
                    "expected": list(expected),
                    "readiness": sorted(readiness_set),
                    "terminal": sorted(terminal_set),
                },
            )
    _compare_set_field(
        failures,
        check_id="roadmap_execution_hard_gates_match",
        reason="execution_summary_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(summary.get("gpu_bubble_release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="terminal_roadmap_execution_hard_gates_match",
        reason="terminal_execution_summary_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(terminal_summary.get("gpu_bubble_release_hard_gate_ids")),
    )
    return compact


def _check_experiment_matrix_readiness(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    summary = _mapping(readiness.get("experiment_matrix_readiness"))
    terminal_summary = _mapping(terminal.get("experiment_matrix_readiness"))
    compact = {
        "available": bool(summary),
        "terminal_available": bool(terminal_summary),
        "roadmap": str(summary.get("roadmap") or ""),
        "terminal_roadmap": str(terminal_summary.get("roadmap") or ""),
        "matrix_status": str(summary.get("matrix_status") or ""),
        "row_count": _safe_int(summary.get("row_count")),
        "terminal_row_count": _safe_int(terminal_summary.get("row_count")),
        "required_batch_count": _safe_int(summary.get("required_batch_count")),
        "required_batch_ids": _strings(summary.get("required_batch_ids"))[:20],
        "covered_batch_ids": _strings(summary.get("covered_batch_ids"))[:20],
        "missing_batch_ids": _strings(summary.get("missing_batch_ids"))[:20],
        "batch_row_counts": dict(_mapping(summary.get("batch_row_counts"))),
        "family_row_counts": dict(_mapping(summary.get("family_row_counts"))),
        "coverage_state_counts": dict(_mapping(summary.get("coverage_state_counts"))),
        "release_hard_gate_ids": _strings(summary.get("release_hard_gate_ids"))[:20],
        "current_progression_phase_id": str(summary.get("current_progression_phase_id") or ""),
        "unsafe_row_count": _safe_int(summary.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(summary.get("unsafe_row_ids"))[:20],
        "ready_for_release_claim": bool(summary.get("ready_for_release_claim")),
        "fail_closed": bool(summary.get("fail_closed")),
        "terminal_fail_closed": bool(terminal_summary.get("fail_closed")),
        "not_release_evidence": bool(summary.get("not_release_evidence")),
        "terminal_not_release_evidence": bool(terminal_summary.get("not_release_evidence")),
        "safe_to_auto_start": bool(summary.get("safe_to_auto_start")),
        "terminal_safe_to_auto_start": bool(terminal_summary.get("safe_to_auto_start")),
        "release_claim_allowed": bool(summary.get("release_claim_allowed")),
        "terminal_release_claim_allowed": bool(terminal_summary.get("release_claim_allowed")),
    }
    if not summary:
        _failure(failures, "experiment_matrix_readiness", "readiness_experiment_matrix_missing")
        return compact
    if not terminal_summary:
        _failure(
            failures,
            "terminal_experiment_matrix_readiness",
            "terminal_experiment_matrix_missing",
        )
        return compact
    if str(summary.get("roadmap") or "") != ROADMAP or str(terminal_summary.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "experiment_matrix_roadmap",
            "experiment_matrix_roadmap_missing_or_wrong",
            value={"readiness": summary.get("roadmap"), "terminal": terminal_summary.get("roadmap")},
        )
    if str(summary.get("artifact_role") or "") != "gpu_bubble_experiment_matrix_readiness_summary":
        _failure(
            failures,
            "experiment_matrix_artifact_role",
            "experiment_matrix_artifact_role_wrong",
            value=summary.get("artifact_role"),
        )
    if not bool(summary.get("fail_closed")) or not bool(terminal_summary.get("fail_closed")):
        _failure(
            failures,
            "experiment_matrix_fail_closed",
            "experiment_matrix_not_fail_closed",
            value={"readiness": summary.get("fail_closed"), "terminal": terminal_summary.get("fail_closed")},
        )
    if (
        bool(summary.get("safe_to_auto_start"))
        or bool(terminal_summary.get("safe_to_auto_start"))
        or bool(summary.get("release_claim_allowed"))
        or bool(terminal_summary.get("release_claim_allowed"))
        or bool(summary.get("ready_for_release_claim"))
        or not bool(summary.get("not_release_evidence"))
        or not bool(terminal_summary.get("not_release_evidence"))
    ):
        _failure(
            failures,
            "experiment_matrix_fail_closed_flags",
            "experiment_matrix_allows_release_or_auto_start",
        )
    expected_batches = set(REQUIRED_EXPERIMENT_BATCH_IDS)
    if set(_strings(summary.get("required_batch_ids"))) != expected_batches or set(
        _strings(terminal_summary.get("required_batch_ids"))
    ) != expected_batches:
        _failure(
            failures,
            "experiment_matrix_required_batches",
            "experiment_matrix_required_batches_do_not_match_roadmap",
            value={
                "expected": sorted(expected_batches),
                "readiness": sorted(set(_strings(summary.get("required_batch_ids")))),
                "terminal": sorted(set(_strings(terminal_summary.get("required_batch_ids")))),
            },
        )
    if _safe_int(summary.get("required_batch_count")) != len(REQUIRED_EXPERIMENT_BATCH_IDS) or _safe_int(
        terminal_summary.get("required_batch_count")
    ) != len(REQUIRED_EXPERIMENT_BATCH_IDS):
        _failure(
            failures,
            "experiment_matrix_required_batch_count",
            "experiment_matrix_required_batch_count_wrong",
            value={
                "expected": len(REQUIRED_EXPERIMENT_BATCH_IDS),
                "readiness": summary.get("required_batch_count"),
                "terminal": terminal_summary.get("required_batch_count"),
            },
        )
    if set(_strings(summary.get("covered_batch_ids"))) != expected_batches:
        _failure(
            failures,
            "experiment_matrix_batch_coverage",
            "experiment_matrix_does_not_cover_all_required_batches",
            value={
                "covered": _strings(summary.get("covered_batch_ids")),
                "missing": _strings(summary.get("missing_batch_ids")),
            },
        )
    if _strings(summary.get("missing_batch_ids")):
        _failure(
            failures,
            "experiment_matrix_missing_batches",
            "experiment_matrix_reports_missing_required_batches",
            value=_strings(summary.get("missing_batch_ids")),
        )
    expected_row_count = len(_list(readiness.get("next_actions")))
    if _safe_int(summary.get("row_count")) != expected_row_count or _safe_int(
        terminal_summary.get("row_count")
    ) != expected_row_count:
        _failure(
            failures,
            "experiment_matrix_row_count",
            "experiment_matrix_row_count_does_not_match_next_actions",
            value={
                "expected": expected_row_count,
                "readiness": summary.get("row_count"),
                "terminal": terminal_summary.get("row_count"),
            },
        )
    _compare_set_field(
        failures,
        check_id="experiment_matrix_hard_gates_match",
        reason="experiment_matrix_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(summary.get("release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="terminal_experiment_matrix_hard_gates_match",
        reason="terminal_experiment_matrix_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(terminal_summary.get("release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="experiment_matrix_covered_batches_match",
        reason="terminal_experiment_matrix_covered_batches_do_not_match_readiness",
        left=_strings(summary.get("covered_batch_ids")),
        right=_strings(terminal_summary.get("covered_batch_ids")),
    )
    if _safe_int(summary.get("unsafe_row_count")) != 0 or _safe_int(
        terminal_summary.get("unsafe_row_count")
    ) != 0:
        _failure(
            failures,
            "experiment_matrix_unsafe_rows",
            "experiment_matrix_reports_unsafe_rows",
            value={
                "readiness": _strings(summary.get("unsafe_row_ids")),
                "terminal": _strings(terminal_summary.get("unsafe_row_ids")),
            },
        )
    return compact


def _blocked_release_gates_from_rows(rows: Sequence[Any]) -> list[str]:
    blocked: list[str] = []
    for raw in rows:
        row = _mapping(raw)
        vector = _mapping(row.get("gate_vector"))
        for gate_id, raw_gate in vector.items():
            gate = _mapping(raw_gate)
            if str(gate.get("state") or "") == "blocked":
                blocked.append(str(gate_id))
    return _unique(blocked)


def _normalized_gate_explanation_from_mapping(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = [_mapping(item) for item in _list(summary.get("rows"))]
    gate_outcome_counts: dict[str, int] = {}
    row_outcome_counts: dict[str, int] = {}
    blocker_explanation_counts: dict[str, int] = {}
    missing_metric_counts: dict[str, int] = {}
    release_hard_gate_row_counts: dict[str, int] = {}
    release_hard_gate_ids = set(_strings(summary.get("release_hard_gate_ids")))
    release_gate_map = {
        "sd15_lora_512": "sd15_lora_512_release_coverage",
        "natural_load_canary_pending": "natural_load_canary_release_gate",
    }
    expected_release_gate_ids = {
        gate_id for source_id, gate_id in release_gate_map.items() if source_id in release_hard_gate_ids
    }
    unsafe_row_ids: list[str] = []

    for row in rows:
        row_id = str(row.get("row_id") or "")
        gate_vector = _mapping(row.get("gate_vector"))
        blocked_gate_ids = {
            str(gate_id)
            for gate_id, raw_gate in gate_vector.items()
            if str(_mapping(raw_gate).get("state") or "") == "blocked"
        } or set(_strings(row.get("blocked_gate_ids")))
        missing_gate_ids = {
            str(gate_id)
            for gate_id, raw_gate in gate_vector.items()
            if str(_mapping(raw_gate).get("state") or "") == "missing"
        } or set(_strings(row.get("missing_gate_ids")))
        if (
            bool(row.get("safe_to_auto_start"))
            or bool(row.get("release_claim_allowed"))
            or not bool(row.get("not_release_evidence"))
        ):
            unsafe_row_ids.append(row_id)
        if blocked_gate_ids.intersection(expected_release_gate_ids):
            row_outcome = "blocked_by_release_hard_gate"
        elif blocked_gate_ids:
            row_outcome = "blocked_by_non_release_gate"
        elif missing_gate_ids:
            row_outcome = "missing_required_metric_or_context"
        else:
            row_outcome = "all_gates_passed_but_non_release"
        row_outcome_counts[row_outcome] = row_outcome_counts.get(row_outcome, 0) + 1

        for gate_id, raw_gate in gate_vector.items():
            gate = _mapping(raw_gate)
            state = str(gate.get("state") or "unknown")
            gate_key = f"{gate_id}:{state}"
            gate_outcome_counts[gate_key] = gate_outcome_counts.get(gate_key, 0) + 1
            if state in {"blocked", "missing"}:
                for reason_id in _strings(gate.get("reason_ids")) or [f"{gate_id}_{state}"]:
                    blocker_explanation_counts[reason_id] = (
                        blocker_explanation_counts.get(reason_id, 0) + 1
                    )
                if state == "missing":
                    for metric_id in _strings(gate.get("metric_ids")):
                        missing_metric_counts[metric_id] = missing_metric_counts.get(metric_id, 0) + 1
        for source_id, gate_id in release_gate_map.items():
            if source_id in release_hard_gate_ids and gate_id in blocked_gate_ids:
                release_hard_gate_row_counts[source_id] = (
                    release_hard_gate_row_counts.get(source_id, 0) + 1
                )

    fail_closed = (
        bool(rows)
        and not unsafe_row_ids
        and bool(expected_release_gate_ids)
        and expected_release_gate_ids.issubset(set(_strings(summary.get("blocked_release_gate_ids"))))
        and not bool(summary.get("release_claim_allowed"))
    )
    return {
        "mapped_row_count": len(rows),
        "row_outcome_counts": dict(sorted(row_outcome_counts.items())),
        "gate_outcome_counts": dict(sorted(gate_outcome_counts.items())),
        "blocker_explanation_counts": dict(sorted(blocker_explanation_counts.items())),
        "missing_metric_counts": dict(sorted(missing_metric_counts.items())),
        "release_hard_gate_row_counts": dict(sorted(release_hard_gate_row_counts.items())),
        "unsafe_row_count": len(unsafe_row_ids),
        "unsafe_row_ids": unsafe_row_ids[:20],
        "fail_closed": fail_closed,
    }


def _check_normalized_evidence_gate_mapping(
    failures: list[dict[str, Any]],
    *,
    readiness: Mapping[str, Any],
    terminal: Mapping[str, Any],
    readiness_hard_gates: Sequence[str],
) -> dict[str, Any]:
    summary = _mapping(readiness.get("normalized_evidence_gate_mapping"))
    terminal_summary = _mapping(terminal.get("normalized_evidence_gate_mapping"))
    explanation = _mapping(readiness.get("normalized_evidence_gate_explanation_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("normalized_evidence_gate_explanation_summary")
    )
    terminal_explanation = _mapping(terminal.get("normalized_evidence_gate_explanation_summary"))
    rows = _list(summary.get("rows"))
    blocked_from_rows = _blocked_release_gates_from_rows(rows)
    computed_explanation = _normalized_gate_explanation_from_mapping(summary)
    compact = {
        "available": bool(summary),
        "terminal_available": bool(terminal_summary),
        "roadmap": str(summary.get("roadmap") or ""),
        "terminal_roadmap": str(terminal_summary.get("roadmap") or ""),
        "artifact_role": str(summary.get("artifact_role") or ""),
        "terminal_artifact_role": str(terminal_summary.get("artifact_role") or ""),
        "source_report": str(summary.get("source_report") or ""),
        "source_normalized_evidence_count": _safe_int(
            summary.get("source_normalized_evidence_count")
        ),
        "mapped_row_count": _safe_int(summary.get("mapped_row_count")),
        "terminal_mapped_row_count": _safe_int(terminal_summary.get("mapped_row_count")),
        "unmapped_row_count": _safe_int(summary.get("unmapped_row_count")),
        "unmapped_row_ids": _strings(summary.get("unmapped_row_ids"))[:20],
        "unsafe_row_count": _safe_int(summary.get("unsafe_row_count")),
        "unsafe_row_ids": _strings(summary.get("unsafe_row_ids"))[:20],
        "gate_ids": _strings(summary.get("gate_ids"))[:30],
        "gate_state_counts": dict(_mapping(summary.get("gate_state_counts"))),
        "release_claim_role_counts": dict(_mapping(summary.get("release_claim_role_counts"))),
        "family_row_counts": dict(_mapping(summary.get("family_row_counts"))),
        "source_kind_counts": dict(_mapping(summary.get("source_kind_counts"))),
        "release_hard_gate_ids": _strings(summary.get("release_hard_gate_ids"))[:20],
        "blocked_release_gate_ids": _strings(summary.get("blocked_release_gate_ids"))[:30],
        "blocked_release_gate_ids_from_rows": blocked_from_rows[:30],
        "missing_metric_ids": _strings(summary.get("missing_metric_ids"))[:30],
        "experiment_matrix_row_count": _safe_int(summary.get("experiment_matrix_row_count")),
        "ready_for_release_claim": bool(summary.get("ready_for_release_claim")),
        "fail_closed": bool(summary.get("fail_closed")),
        "terminal_fail_closed": bool(terminal_summary.get("fail_closed")),
        "not_release_evidence": bool(summary.get("not_release_evidence")),
        "terminal_not_release_evidence": bool(terminal_summary.get("not_release_evidence")),
        "safe_to_auto_start": bool(summary.get("safe_to_auto_start")),
        "terminal_safe_to_auto_start": bool(terminal_summary.get("safe_to_auto_start")),
        "release_claim_allowed": bool(summary.get("release_claim_allowed")),
        "terminal_release_claim_allowed": bool(terminal_summary.get("release_claim_allowed")),
        "explanation_available": bool(explanation),
        "terminal_explanation_available": bool(terminal_explanation),
        "explanation_artifact_role": str(explanation.get("artifact_role") or ""),
        "terminal_explanation_artifact_role": str(terminal_explanation.get("artifact_role") or ""),
        "explanation_mapped_row_count": _safe_int(explanation.get("mapped_row_count")),
        "terminal_explanation_mapped_row_count": _safe_int(
            terminal_explanation.get("mapped_row_count")
        ),
        "computed_explanation_mapped_row_count": computed_explanation["mapped_row_count"],
        "explanation_row_outcome_counts": dict(_mapping(explanation.get("row_outcome_counts"))),
        "computed_explanation_row_outcome_counts": computed_explanation["row_outcome_counts"],
        "explanation_gate_outcome_counts": dict(_mapping(explanation.get("gate_outcome_counts"))),
        "computed_explanation_gate_outcome_counts": computed_explanation["gate_outcome_counts"],
        "explanation_blocker_explanation_counts": dict(
            _mapping(explanation.get("blocker_explanation_counts"))
        ),
        "computed_explanation_blocker_explanation_counts": computed_explanation[
            "blocker_explanation_counts"
        ],
        "explanation_missing_metric_counts": dict(_mapping(explanation.get("missing_metric_counts"))),
        "computed_explanation_missing_metric_counts": computed_explanation["missing_metric_counts"],
        "explanation_release_hard_gate_row_counts": dict(
            _mapping(explanation.get("release_hard_gate_row_counts"))
        ),
        "computed_explanation_release_hard_gate_row_counts": computed_explanation[
            "release_hard_gate_row_counts"
        ],
        "explanation_unsafe_row_count": _safe_int(explanation.get("unsafe_row_count")),
        "terminal_explanation_unsafe_row_count": _safe_int(
            terminal_explanation.get("unsafe_row_count")
        ),
        "computed_explanation_unsafe_row_count": computed_explanation["unsafe_row_count"],
        "explanation_fail_closed": bool(explanation.get("fail_closed")),
        "terminal_explanation_fail_closed": bool(terminal_explanation.get("fail_closed")),
        "computed_explanation_fail_closed": computed_explanation["fail_closed"],
    }
    if not summary:
        _failure(failures, "normalized_evidence_gate_mapping", "readiness_normalized_gate_mapping_missing")
        return compact
    if not terminal_summary:
        _failure(
            failures,
            "terminal_normalized_evidence_gate_mapping",
            "terminal_normalized_gate_mapping_missing",
        )
        return compact
    if not explanation:
        _failure(
            failures,
            "normalized_evidence_gate_explanation_summary",
            "readiness_normalized_gate_explanation_summary_missing",
        )
    if not terminal_explanation:
        _failure(
            failures,
            "terminal_normalized_evidence_gate_explanation_summary",
            "terminal_normalized_gate_explanation_summary_missing",
        )
    if str(summary.get("roadmap") or "") != ROADMAP or str(terminal_summary.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "normalized_gate_mapping_roadmap",
            "normalized_gate_mapping_roadmap_missing_or_wrong",
            value={"readiness": summary.get("roadmap"), "terminal": terminal_summary.get("roadmap")},
        )
    if str(summary.get("artifact_role") or "") != "gpu_bubble_normalized_evidence_gate_mapping":
        _failure(
            failures,
            "normalized_gate_mapping_artifact_role",
            "normalized_gate_mapping_artifact_role_wrong",
            value=summary.get("artifact_role"),
        )
    if not bool(summary.get("fail_closed")) or not bool(terminal_summary.get("fail_closed")):
        _failure(
            failures,
            "normalized_gate_mapping_fail_closed",
            "normalized_gate_mapping_not_fail_closed",
            value={"readiness": summary.get("fail_closed"), "terminal": terminal_summary.get("fail_closed")},
        )
    if (
        bool(summary.get("safe_to_auto_start"))
        or bool(terminal_summary.get("safe_to_auto_start"))
        or bool(explanation.get("safe_to_auto_start"))
        or bool(terminal_explanation.get("safe_to_auto_start"))
        or bool(summary.get("release_claim_allowed"))
        or bool(terminal_summary.get("release_claim_allowed"))
        or bool(explanation.get("release_claim_allowed"))
        or bool(terminal_explanation.get("release_claim_allowed"))
        or bool(summary.get("ready_for_release_claim"))
        or bool(explanation.get("ready_for_release_claim"))
        or not bool(summary.get("not_release_evidence"))
        or not bool(terminal_summary.get("not_release_evidence"))
        or (bool(explanation) and not bool(explanation.get("not_release_evidence")))
        or (bool(terminal_explanation) and not bool(terminal_explanation.get("not_release_evidence")))
    ):
        _failure(
            failures,
            "normalized_gate_mapping_fail_closed_flags",
            "normalized_gate_mapping_allows_release_or_auto_start",
        )
    expected_gates = set(NORMALIZED_EVIDENCE_GATE_IDS)
    if set(_strings(summary.get("gate_ids"))) != expected_gates or set(
        _strings(terminal_summary.get("gate_ids"))
    ) != expected_gates:
        _failure(
            failures,
            "normalized_gate_mapping_gate_ids",
            "normalized_gate_mapping_gate_ids_do_not_match_contract",
            value={
                "expected": sorted(expected_gates),
                "readiness": sorted(set(_strings(summary.get("gate_ids")))),
                "terminal": sorted(set(_strings(terminal_summary.get("gate_ids")))),
            },
        )
    source_count = _safe_int(summary.get("source_normalized_evidence_count"))
    if source_count <= 0:
        _failure(
            failures,
            "normalized_gate_mapping_source_count",
            "normalized_gate_mapping_source_count_missing_or_zero",
            value=summary.get("source_normalized_evidence_count"),
        )
    if _safe_int(summary.get("mapped_row_count")) != source_count or len(rows) != source_count:
        _failure(
            failures,
            "normalized_gate_mapping_row_count",
            "normalized_gate_mapping_row_count_does_not_match_source",
            value={
                "source": source_count,
                "mapped": summary.get("mapped_row_count"),
                "rows": len(rows),
            },
        )
    if _safe_int(terminal_summary.get("mapped_row_count")) != source_count:
        _failure(
            failures,
            "terminal_normalized_gate_mapping_row_count",
            "terminal_normalized_gate_mapping_row_count_does_not_match_readiness",
            value={"source": source_count, "terminal": terminal_summary.get("mapped_row_count")},
        )
    if _safe_int(summary.get("unmapped_row_count")) != 0:
        _failure(
            failures,
            "normalized_gate_mapping_unmapped_rows",
            "normalized_gate_mapping_reports_unmapped_rows",
            value=_strings(summary.get("unmapped_row_ids")),
        )
    if _safe_int(summary.get("unsafe_row_count")) != 0 or _safe_int(
        terminal_summary.get("unsafe_row_count")
    ) != 0 or _safe_int(explanation.get("unsafe_row_count")) != 0 or _safe_int(
        terminal_explanation.get("unsafe_row_count")
    ) != 0 or computed_explanation["unsafe_row_count"] != 0:
        _failure(
            failures,
            "normalized_gate_mapping_unsafe_rows",
            "normalized_gate_mapping_reports_unsafe_rows",
            value={
                "readiness": _strings(summary.get("unsafe_row_ids")),
                "terminal": _strings(terminal_summary.get("unsafe_row_ids")),
                "explanation": _strings(explanation.get("unsafe_row_ids")),
                "computed_explanation": computed_explanation["unsafe_row_ids"],
            },
        )
    required_blocked_gates = {
        "sd15_lora_512_release_coverage" if "sd15_lora_512" in readiness_hard_gates else "",
        "natural_load_canary_release_gate" if "natural_load_canary_pending" in readiness_hard_gates else "",
    } - {""}
    if not required_blocked_gates.issubset(set(blocked_from_rows)):
        _failure(
            failures,
            "normalized_gate_mapping_hard_gates_from_rows",
            "normalized_gate_mapping_cannot_recompute_hard_gates_from_gate_vector",
            value={"required": sorted(required_blocked_gates), "from_rows": sorted(set(blocked_from_rows))},
        )
    _compare_set_field(
        failures,
        check_id="normalized_gate_mapping_hard_gates_match",
        reason="normalized_gate_mapping_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(summary.get("release_hard_gate_ids")),
    )
    _compare_set_field(
        failures,
        check_id="terminal_normalized_gate_mapping_hard_gates_match",
        reason="terminal_normalized_gate_mapping_hard_gates_do_not_match_readiness",
        left=readiness_hard_gates,
        right=_strings(terminal_summary.get("release_hard_gate_ids")),
    )
    if str(explanation.get("roadmap") or "") != ROADMAP or str(terminal_explanation.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "normalized_gate_explanation_roadmap",
            "normalized_gate_explanation_roadmap_missing_or_wrong",
            value={
                "readiness": explanation.get("roadmap"),
                "terminal": terminal_explanation.get("roadmap"),
            },
        )
    if (
        str(explanation.get("artifact_role") or "")
        != "gpu_bubble_normalized_evidence_gate_explanation_summary"
    ):
        _failure(
            failures,
            "normalized_gate_explanation_artifact_role",
            "normalized_gate_explanation_artifact_role_wrong",
            value=explanation.get("artifact_role"),
        )
    if (
        not bool(explanation.get("fail_closed"))
        or not bool(terminal_explanation.get("fail_closed"))
        or not computed_explanation["fail_closed"]
    ):
        _failure(
            failures,
            "normalized_gate_explanation_fail_closed",
            "normalized_gate_explanation_not_fail_closed",
            value={
                "readiness": explanation.get("fail_closed"),
                "terminal": terminal_explanation.get("fail_closed"),
                "computed": computed_explanation["fail_closed"],
            },
        )
    for field in (
        "mapped_row_count",
        "row_outcome_counts",
        "gate_outcome_counts",
        "blocker_explanation_counts",
        "missing_metric_counts",
        "release_hard_gate_row_counts",
    ):
        expected = computed_explanation[field]
        if explanation.get(field) != expected or terminal_explanation.get(field) != expected:
            _failure(
                failures,
                f"normalized_gate_explanation_{field}",
                f"normalized_gate_explanation_{field}_does_not_match_computed",
                value={
                    "readiness": explanation.get(field),
                    "terminal": terminal_explanation.get(field),
                    "computed": expected,
                },
            )
    return compact


def build_gpu_bubble_release_readiness_guard(
    *,
    readiness_next_actions: Mapping[str, Any] | None = None,
    terminal_self_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = _mapping(readiness_next_actions)
    terminal = _mapping(terminal_self_check)
    first_release = _mapping(readiness.get("first_release_scope"))
    remaining = _mapping(readiness.get("remaining_work_summary"))
    terminal_remaining_work = _mapping(terminal.get("remaining_work_summary"))
    computed_remaining_work = _remaining_work_summary_from_readiness(readiness)
    unblocker = _mapping(readiness.get("release_unblocker_summary"))
    manual_gpu_summary = _mapping(readiness.get("manual_gpu_execution_summary"))
    terminal_manual_gpu_summary = _mapping(terminal.get("manual_gpu_execution_summary"))
    readiness_next_action_machine = _mapping(readiness.get("next_action_machine_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("next_action_machine_summary")
    )
    readiness_next_action_contract = _mapping(readiness.get("next_action_contract_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("next_action_contract_summary")
    )
    terminal_next_action_machine = _mapping(terminal.get("next_action_machine_summary"))
    terminal_next_action_contract = _mapping(terminal.get("next_action_contract_summary"))
    computed_next_action_machine = _next_action_machine_summary_from_actions(readiness)
    computed_next_action_contract = _next_action_contract_summary_from_actions(readiness)
    readiness_manual_review_queue = _mapping(readiness.get("manual_review_queue_summary"))
    readiness_evidence_manual_review_queue = _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_review_queue_summary")
    )
    terminal_manual_review_queue = _mapping(terminal.get("manual_review_queue_summary"))
    computed_manual_review_queue = _manual_review_queue_summary_from_readiness(readiness)
    readiness_manual_review_artifact_chain = _mapping(
        readiness.get("manual_review_artifact_chain_summary")
    )
    readiness_evidence_manual_review_artifact_chain = _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_review_artifact_chain_summary")
    )
    terminal_manual_review_artifact_chain = _mapping(
        terminal.get("manual_review_artifact_chain_summary")
    )
    readiness_sdxl_diagnostic_artifact_chain = _mapping(
        readiness.get("sdxl_diagnostic_artifact_chain_summary")
    )
    readiness_evidence_sdxl_diagnostic_artifact_chain = _mapping(
        _mapping(readiness.get("evidence_summary")).get("sdxl_diagnostic_artifact_chain_summary")
    )
    terminal_sdxl_diagnostic_artifact_chain = _mapping(
        terminal.get("sdxl_diagnostic_artifact_chain_summary")
    )
    readiness_protected_run_plan_chain = _mapping(
        readiness.get("protected_followup_run_plan_artifact_chain_summary")
    )
    readiness_evidence_protected_run_plan_chain = _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "protected_followup_run_plan_artifact_chain_summary"
        )
    )
    terminal_protected_run_plan_chain = _mapping(
        terminal.get("protected_followup_run_plan_artifact_chain_summary")
    )
    readiness_followup_queue = _mapping(readiness.get("protected_followup_gpu_queue_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("protected_followup_gpu_queue_summary")
    )
    terminal_followup_queue = _mapping(terminal.get("protected_followup_gpu_queue_summary"))
    computed_followup_queue = _protected_followup_gpu_queue_from_actions(readiness)
    readiness_blocker_matrix = _mapping(readiness.get("remaining_release_blocker_matrix_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_release_blocker_matrix_summary")
    )
    terminal_blocker_matrix = _mapping(terminal.get("remaining_release_blocker_matrix_summary"))
    computed_blocker_matrix = _remaining_release_blocker_matrix_from_readiness(readiness)
    readiness_blocker_handoff = _mapping(readiness.get("remaining_blocker_resolution_handoff_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_blocker_resolution_handoff_summary")
    )
    terminal_blocker_handoff = _mapping(terminal.get("remaining_blocker_resolution_handoff_summary"))
    computed_blocker_handoff = _remaining_blocker_resolution_handoff_from_readiness(readiness)
    readiness_action_dependency_graph = _mapping(
        readiness.get("remaining_action_dependency_graph_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("remaining_action_dependency_graph_summary"))
    terminal_action_dependency_graph = _mapping(terminal.get("remaining_action_dependency_graph_summary"))
    computed_action_dependency_graph = _remaining_action_dependency_graph_from_readiness(readiness)
    readiness_action_unblock_sequence = _mapping(
        readiness.get("remaining_action_unblock_sequence_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("remaining_action_unblock_sequence_summary"))
    terminal_action_unblock_sequence = _mapping(terminal.get("remaining_action_unblock_sequence_summary"))
    computed_action_unblock_sequence = _remaining_action_unblock_sequence_from_readiness(readiness)
    readiness_blocker_presence = _mapping(readiness.get("remaining_blocker_artifact_presence_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("remaining_blocker_artifact_presence_summary")
    )
    terminal_blocker_presence = _mapping(terminal.get("remaining_blocker_artifact_presence_summary"))
    computed_blocker_presence = _remaining_blocker_artifact_presence_from_readiness(readiness)
    readiness_release_exit = _mapping(readiness.get("release_claim_exit_criteria_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("release_claim_exit_criteria_summary")
    )
    terminal_release_exit = _mapping(terminal.get("release_claim_exit_criteria_summary"))
    computed_release_exit = _release_claim_exit_criteria_from_readiness(readiness)
    readiness_release_gate_input_dependency = _mapping(
        readiness.get("release_gate_input_dependency_summary")
    ) or _mapping(_mapping(readiness.get("evidence_summary")).get("release_gate_input_dependency_summary"))
    terminal_release_gate_input_dependency = _mapping(
        terminal.get("release_gate_input_dependency_summary")
    )
    computed_release_gate_input_dependency = _release_gate_input_dependency_from_readiness(readiness)
    readiness_release_gate_post_input_refresh_plan = _mapping(
        readiness.get("release_gate_post_input_refresh_plan_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_plan_summary"
        )
    )
    terminal_release_gate_post_input_refresh_plan = _mapping(
        terminal.get("release_gate_post_input_refresh_plan_summary")
    )
    computed_release_gate_post_input_refresh_plan = (
        _release_gate_post_input_refresh_plan_from_readiness(readiness)
    )
    readiness_release_gate_input_detection_source = _mapping(
        readiness.get("release_gate_input_detection_source_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_detection_source_summary"
        )
    )
    terminal_release_gate_input_detection_source = _mapping(
        terminal.get("release_gate_input_detection_source_summary")
    )
    computed_release_gate_input_detection_source = (
        _release_gate_input_detection_source_from_readiness(readiness)
    )
    readiness_release_gate_input_acceptance_criteria = _mapping(
        readiness.get("release_gate_input_acceptance_criteria_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_acceptance_criteria_summary"
        )
    )
    terminal_release_gate_input_acceptance_criteria = _mapping(
        terminal.get("release_gate_input_acceptance_criteria_summary")
    )
    computed_release_gate_input_acceptance_criteria = (
        _release_gate_input_acceptance_criteria_from_readiness(readiness)
    )
    readiness_release_gate_input_refresh_readiness = _mapping(
        readiness.get("release_gate_input_refresh_readiness_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_refresh_readiness_summary"
        )
    )
    terminal_release_gate_input_refresh_readiness = _mapping(
        terminal.get("release_gate_input_refresh_readiness_summary")
    )
    computed_release_gate_input_refresh_readiness = (
        _release_gate_input_refresh_readiness_from_readiness(readiness)
    )
    readiness_release_gate_input_refresh_blocker = _mapping(
        readiness.get("release_gate_input_refresh_blocker_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_refresh_blocker_summary"
        )
    )
    terminal_release_gate_input_refresh_blocker = _mapping(
        terminal.get("release_gate_input_refresh_blocker_summary")
    )
    computed_release_gate_input_refresh_blocker = (
        _release_gate_input_refresh_blocker_from_readiness(readiness)
    )
    readiness_release_gate_input_lifecycle = _mapping(
        readiness.get("release_gate_input_lifecycle_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_input_lifecycle_summary"
        )
    )
    terminal_release_gate_input_lifecycle = _mapping(
        terminal.get("release_gate_input_lifecycle_summary")
    )
    computed_release_gate_input_lifecycle = (
        _release_gate_input_lifecycle_from_readiness(readiness)
    )
    readiness_external_input_release_gate_alignment = _mapping(
        readiness.get("external_input_release_gate_alignment_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "external_input_release_gate_alignment_summary"
        )
    )
    terminal_external_input_release_gate_alignment = _mapping(
        terminal.get("external_input_release_gate_alignment_summary")
    )
    computed_external_input_release_gate_alignment = (
        _external_input_release_gate_alignment_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_command_surface = _mapping(
        readiness.get("release_gate_post_input_refresh_command_surface_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_command_surface_summary"
        )
    )
    terminal_release_gate_post_input_refresh_command_surface = _mapping(
        terminal.get("release_gate_post_input_refresh_command_surface_summary")
    )
    computed_release_gate_post_input_refresh_command_surface = (
        _release_gate_post_input_refresh_command_surface_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_sequence_integrity = _mapping(
        readiness.get("release_gate_post_input_refresh_sequence_integrity_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_sequence_integrity_summary"
        )
    )
    terminal_release_gate_post_input_refresh_sequence_integrity = _mapping(
        terminal.get("release_gate_post_input_refresh_sequence_integrity_summary")
    )
    computed_release_gate_post_input_refresh_sequence_integrity = (
        _release_gate_post_input_refresh_sequence_integrity_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_terminal_guard_dependency = _mapping(
        readiness.get("release_gate_post_input_refresh_terminal_guard_dependency_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_terminal_guard_dependency_summary"
        )
    )
    terminal_release_gate_post_input_refresh_terminal_guard_dependency = _mapping(
        terminal.get("release_gate_post_input_refresh_terminal_guard_dependency_summary")
    )
    computed_release_gate_post_input_refresh_terminal_guard_dependency = (
        _release_gate_post_input_refresh_terminal_guard_dependency_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_artifact_coverage = _mapping(
        readiness.get("release_gate_post_input_refresh_artifact_coverage_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_artifact_coverage_summary"
        )
    )
    terminal_release_gate_post_input_refresh_artifact_coverage = _mapping(
        terminal.get("release_gate_post_input_refresh_artifact_coverage_summary")
    )
    computed_release_gate_post_input_refresh_artifact_coverage = (
        _release_gate_post_input_refresh_artifact_coverage_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_command_artifact_link = _mapping(
        readiness.get("release_gate_post_input_refresh_command_artifact_link_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_command_artifact_link_summary"
        )
    )
    terminal_release_gate_post_input_refresh_command_artifact_link = _mapping(
        terminal.get("release_gate_post_input_refresh_command_artifact_link_summary")
    )
    computed_release_gate_post_input_refresh_command_artifact_link = (
        _release_gate_post_input_refresh_command_artifact_link_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_guard_consumption = _mapping(
        readiness.get("release_gate_post_input_refresh_guard_consumption_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_guard_consumption_summary"
        )
    )
    terminal_release_gate_post_input_refresh_guard_consumption = _mapping(
        terminal.get("release_gate_post_input_refresh_guard_consumption_summary")
    )
    computed_release_gate_post_input_refresh_guard_consumption = (
        _release_gate_post_input_refresh_guard_consumption_from_readiness(readiness)
    )
    readiness_release_gate_post_input_refresh_guard_report_acceptance = _mapping(
        readiness.get("release_gate_post_input_refresh_guard_report_acceptance_summary")
    ) or _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "release_gate_post_input_refresh_guard_report_acceptance_summary"
        )
    )
    terminal_release_gate_post_input_refresh_guard_report_acceptance = _mapping(
        terminal.get("release_gate_post_input_refresh_guard_report_acceptance_summary")
    )
    computed_release_gate_post_input_refresh_guard_report_acceptance = (
        _release_gate_post_input_refresh_guard_report_acceptance_from_readiness(readiness)
    )
    readiness_command_surface = _mapping(readiness.get("manual_protected_gpu_command_surface_summary")) or _mapping(
        _mapping(readiness.get("evidence_summary")).get("manual_protected_gpu_command_surface_summary")
    )
    terminal_command_surface = _mapping(terminal.get("manual_protected_gpu_command_surface_summary"))
    computed_command_surface = _manual_protected_gpu_command_surface_from_readiness(readiness)
    terminal_audit = _mapping(terminal.get("json_only_progress_audit"))
    external_input_audit = _mapping(terminal.get("external_input_filesystem_audit"))
    readiness_source_axis_freshness = _mapping(readiness.get("source_axis_freshness_dedupe_audit"))
    readiness_evidence_source_axis_freshness = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_axis_freshness_dedupe_audit")
    )
    terminal_source_axis_freshness = _mapping(terminal.get("source_axis_freshness_dedupe_audit"))
    readiness_source_axis_requirement = _mapping(readiness.get("source_axis_requirement_summary"))
    readiness_evidence_source_axis_requirement = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_axis_requirement")
    )
    terminal_source_axis_requirement = _mapping(terminal.get("source_axis_requirement_summary"))
    readiness_source_cache_pipeline_summary = _mapping(
        readiness.get("source_cache_axis_pipeline_readiness_summary")
    )
    readiness_evidence_source_cache_pipeline_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_cache_axis_pipeline_readiness")
    )
    terminal_source_cache_pipeline_summary = _mapping(
        terminal.get("source_cache_axis_pipeline_readiness_summary")
    )
    readiness_external_input_admission = _mapping(readiness.get("external_input_admission_summary"))
    evidence_external_input_admission = _mapping(
        _mapping(readiness.get("evidence_summary")).get("external_input_admission")
    )
    terminal_external_input_admission = _mapping(terminal.get("external_input_admission_summary"))
    readiness_external_input_intake = _mapping(readiness.get("external_input_intake_registry_summary"))
    evidence_external_input_intake = _mapping(
        _mapping(readiness.get("evidence_summary")).get("external_input_intake_registry")
    )
    terminal_external_input_intake = _mapping(terminal.get("external_input_intake_registry_summary"))
    readiness_external_input_replay = _mapping(readiness.get("external_input_replay_plan_summary"))
    evidence_external_input_replay = _mapping(
        _mapping(readiness.get("evidence_summary")).get("external_input_replay_plan")
    )
    terminal_external_input_replay = _mapping(terminal.get("external_input_replay_plan_summary"))
    readiness_external_input_handoff = _mapping(readiness.get("external_input_handoff_packet_summary"))
    evidence_external_input_handoff = _mapping(
        _mapping(readiness.get("evidence_summary")).get("external_input_handoff_packet")
    )
    terminal_external_input_handoff = _mapping(terminal.get("external_input_handoff_packet_summary"))
    readiness_external_input_json_refresh_runner_manifest = _mapping(
        readiness.get("external_input_json_refresh_runner_manifest_summary")
    )
    evidence_external_input_json_refresh_runner_manifest = _mapping(
        _mapping(readiness.get("evidence_summary")).get(
            "external_input_json_refresh_runner_manifest_summary"
        )
    )
    terminal_external_input_json_refresh_runner_manifest = _mapping(
        terminal.get("external_input_json_refresh_runner_manifest_summary")
    )
    readiness_newbie_warm_cache = _mapping(readiness.get("newbie_warm_cache_inventory_summary"))
    evidence_newbie_warm_cache = _mapping(
        _mapping(readiness.get("evidence_summary")).get("newbie_warm_cache_inventory")
    )
    terminal_newbie_warm_cache = _mapping(terminal.get("newbie_warm_cache_inventory_summary"))
    readiness_source_cache_identity_summary = _mapping(
        readiness.get("source_cache_axis_identity_registry_summary")
    )
    evidence_source_cache_identity_summary = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_cache_axis_identity_registry_summary")
    )
    terminal_source_cache_identity_summary = _mapping(
        terminal.get("source_cache_axis_identity_registry_summary")
    )
    readiness_source_cache_preflight = _mapping(readiness.get("source_cache_axis_admission_preflight_summary"))
    evidence_source_cache_preflight = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_cache_axis_admission_preflight")
    )
    terminal_source_cache_preflight = _mapping(terminal.get("source_cache_axis_admission_preflight_summary"))
    readiness_source_cache_manual_plan = _mapping(readiness.get("source_cache_axis_manual_canary_plan_summary"))
    evidence_source_cache_manual_plan = _mapping(
        _mapping(readiness.get("evidence_summary")).get("source_cache_axis_manual_canary_plan")
    )
    terminal_source_cache_manual_plan = _mapping(terminal.get("source_cache_axis_manual_canary_plan_summary"))
    readiness_post_manual_rebuild = _mapping(readiness.get("post_manual_evidence_rebuild_plan_summary"))
    evidence_post_manual_rebuild = _mapping(
        _mapping(readiness.get("evidence_summary")).get("post_manual_evidence_rebuild_plan")
    )
    terminal_post_manual_rebuild = _mapping(terminal.get("post_manual_evidence_rebuild_plan_summary"))
    source_cache_negative = _mapping(terminal.get("source_cache_negative_evidence_summary"))
    source_cache_identity = _mapping(terminal.get("source_cache_axis_identity_registry"))
    artifact_freshness = _mapping(terminal.get("artifact_freshness_audit"))
    terminal_unblocker = _mapping(terminal.get("release_unblocker_summary"))
    readiness_input = _mapping(unblocker.get("input_resolution_summary"))
    terminal_input = _mapping(terminal.get("input_resolution_summary"))
    readiness_transition = _mapping(readiness.get("external_input_transition_table"))
    terminal_transition = _mapping(terminal.get("external_input_transition_table"))
    terminal_unblocker_transition = _mapping(terminal_unblocker.get("external_input_transition_table"))
    readiness_manual = _mapping(unblocker.get("manual_evidence_blocking_summary"))
    terminal_manual = _mapping(terminal.get("manual_evidence_blocking_summary"))

    failures: list[dict[str, Any]] = []
    roadmap_lineage = _check_roadmap_lineage_audit(failures, terminal)
    source_cache_stage_lineage = _check_source_cache_pipeline_stage_lineage(failures, terminal)
    if str(readiness.get("report") or "") != READINESS_REPORT:
        _failure(failures, "readiness_report", "unexpected_or_missing_readiness_report", value=readiness.get("report"))
    if str(terminal.get("report") or "") != TERMINAL_REPORT:
        _failure(failures, "terminal_report", "unexpected_or_missing_terminal_report", value=terminal.get("report"))
    if str(readiness.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "readiness_roadmap",
            "readiness_roadmap_missing_or_wrong",
            value=readiness.get("roadmap"),
        )
    if str(terminal.get("roadmap") or "") != ROADMAP:
        _failure(
            failures,
            "terminal_roadmap",
            "terminal_roadmap_missing_or_wrong",
            value=terminal.get("roadmap"),
        )

    if bool(readiness.get("safe_to_auto_start")) or bool(terminal.get("safe_to_auto_start")):
        _failure(failures, "safe_to_auto_start", "gpu_bubble_artifact_allows_auto_start")
    if bool(readiness.get("release_claim_allowed")) or bool(terminal.get("release_claim_allowed")):
        _failure(failures, "release_claim_allowed", "gpu_bubble_release_claim_is_not_fail_closed")
    if bool(readiness.get("publishable")):
        _failure(failures, "publishable", "gpu_bubble_readiness_marked_publishable")
    if not bool(readiness.get("does_not_run_training")) or not bool(readiness.get("does_not_run_cuda")):
        _failure(
            failures,
            "readiness_json_only_execution_contract",
            "readiness_artifact_not_marked_json_only_no_training_no_cuda",
            value={
                "does_not_run_training": readiness.get("does_not_run_training"),
                "does_not_run_cuda": readiness.get("does_not_run_cuda"),
            },
        )
    if not bool(terminal.get("does_not_run_training")) or not bool(terminal.get("does_not_run_cuda")):
        _failure(
            failures,
            "terminal_json_only_execution_contract",
            "terminal_artifact_not_marked_json_only_no_training_no_cuda",
            value={
                "does_not_run_training": terminal.get("does_not_run_training"),
                "does_not_run_cuda": terminal.get("does_not_run_cuda"),
            },
        )
    if bool(first_release.get("stable_first_release_blocked_by_this_artifact")) or bool(
        terminal.get("stable_first_release_blocked_by_this_artifact")
    ):
        _failure(failures, "stable_first_release_scope", "stable_first_release_blocked_by_gpu_bubble_artifact")
    _check_downstream_artifacts(failures, readiness)

    readiness_hard_gates = _strings(unblocker.get("gpu_bubble_release_hard_gate_ids")) or _strings(
        remaining.get("gpu_bubble_release_hard_gate_ids")
    )
    roadmap_acceptance = _check_roadmap_acceptance_gate_summary(
        failures,
        readiness=readiness,
        terminal=terminal,
        readiness_hard_gates=readiness_hard_gates,
    )
    roadmap_execution = _check_roadmap_execution_contract_summary(
        failures,
        readiness=readiness,
        terminal=terminal,
        readiness_hard_gates=readiness_hard_gates,
    )
    experiment_matrix = _check_experiment_matrix_readiness(
        failures,
        readiness=readiness,
        terminal=terminal,
        readiness_hard_gates=readiness_hard_gates,
    )
    normalized_gate_mapping = _check_normalized_evidence_gate_mapping(
        failures,
        readiness=readiness,
        terminal=terminal,
        readiness_hard_gates=readiness_hard_gates,
    )
    source_artifact_inventory = _check_source_artifact_inventory_summary(
        failures,
        readiness=readiness,
        terminal=terminal,
    )
    evidence_summary_inventory = _check_evidence_summary_inventory_summary(
        failures,
        readiness=readiness,
        terminal=terminal,
    )
    source_downstream_contract = _check_source_and_downstream_artifact_contract_summary(
        failures,
        readiness=readiness,
        terminal=terminal,
    )
    first_release_policy = _check_first_release_policy(
        failures,
        readiness=readiness,
        terminal=terminal,
        readiness_hard_gates=readiness_hard_gates,
    )
    claim_wording_policy = _check_claim_wording_policy(
        failures,
        readiness=readiness,
        terminal=terminal,
    )
    terminal_hard_gates = _strings(terminal.get("gpu_bubble_release_hard_gate_ids"))
    if not readiness_hard_gates:
        _failure(failures, "release_hard_gates", "readiness_hard_gates_missing")
    if readiness_hard_gates and terminal_hard_gates and set(readiness_hard_gates) != set(terminal_hard_gates):
        _failure(
            failures,
            "release_hard_gates_match",
            "terminal_hard_gates_do_not_match_readiness",
            value={"readiness": readiness_hard_gates, "terminal": terminal_hard_gates},
        )
    top_hard_gates = _strings(readiness.get("gpu_bubble_release_hard_gate_ids"))
    if not top_hard_gates:
        _failure(failures, "top_level_release_hard_gates", "top_level_readiness_hard_gates_missing")
    elif readiness_hard_gates and set(top_hard_gates) != set(readiness_hard_gates):
        _failure(
            failures,
            "top_level_release_hard_gates_match",
            "top_level_readiness_hard_gates_do_not_match_summary",
            value={"top_level": top_hard_gates, "summary": readiness_hard_gates},
        )

    readiness_json_ready = _safe_int(remaining.get("json_ready_action_count"))
    top_json_ready = readiness.get("json_ready_action_count")
    top_json_closed = readiness.get("json_closed_action_count")
    if top_json_ready is None:
        _failure(failures, "top_level_json_ready_actions", "top_level_json_ready_action_count_missing")
    elif _safe_int(top_json_ready) != readiness_json_ready:
        _failure(
            failures,
            "top_level_json_ready_actions_match",
            "top_level_json_ready_action_count_does_not_match_summary",
            value={"top_level": top_json_ready, "summary": readiness_json_ready},
        )
    if top_json_closed is None:
        _failure(failures, "top_level_json_closed_actions", "top_level_json_closed_action_count_missing")
    elif _safe_int(top_json_closed) != _safe_int(remaining.get("json_closed_action_count")):
        _failure(
            failures,
            "top_level_json_closed_actions_match",
            "top_level_json_closed_action_count_does_not_match_summary",
            value={"top_level": top_json_closed, "summary": remaining.get("json_closed_action_count")},
        )
    terminal_json_ready = _safe_int(terminal_audit.get("json_ready_action_count"))
    if readiness_json_ready != 0 or terminal_json_ready != 0:
        _failure(
            failures,
            "json_ready_actions",
            "json_ready_actions_remain",
            value={"readiness": readiness_json_ready, "terminal": terminal_json_ready},
        )
    if bool(terminal.get("json_only_substantive_progress_available")):
        _failure(failures, "json_only_progress", "terminal_reports_json_only_progress_available")
    for check_id, summary in {
        "readiness_remaining_work_summary": remaining,
        "terminal_remaining_work_summary": terminal_remaining_work,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_work_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_work_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_action_count")) != 0
        ):
            _failure(failures, check_id, "remaining_work_summary_allows_release_or_auto_start")
        if bool(summary.get("stable_first_release_blocked_by_this_artifact")):
            _failure(failures, check_id, "remaining_work_summary_blocks_stable_first_release")
        if not bool(summary.get("gpu_bubble_release_claim_blocked")):
            _failure(failures, check_id, "remaining_work_summary_unblocks_gpu_bubble_claim")
        if str(summary.get("recommended_release_policy") or "") != "ship_stable_baseline_without_gpu_bubble_gain_claim":
            _failure(
                failures,
                check_id,
                "remaining_work_summary_release_policy_wrong",
                value=summary.get("recommended_release_policy"),
            )
        for field in [
            "total_action_count",
            "gpu_bubble_release_hard_gate_count",
            "json_ready_action_count",
            "json_closed_action_count",
            "external_input_action_count",
            "missing_prerequisite_action_count",
            "manual_gpu_evidence_action_count",
            "protected_manual_gpu_ready_action_count",
            "followup_gpu_required_action_count",
            "current_gpu_heavy_action_count",
            "cache_axis_not_ready_action_count",
            "duplicate_or_stale_axis_action_count",
            "release_gate_related_action_count",
            "unsafe_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_work_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_remaining_work.get(field)),
            )
        for field in [
            "gpu_bubble_release_hard_gate_ids",
            "json_ready_action_ids",
            "external_input_action_ids",
            "missing_prerequisite_action_ids",
            "manual_gpu_evidence_action_ids",
            "protected_manual_gpu_ready_action_ids",
            "followup_gpu_required_action_ids",
            "current_gpu_heavy_action_ids",
            "cache_axis_not_ready_action_ids",
            "duplicate_or_stale_axis_action_ids",
            "release_gate_related_action_ids",
            "unsafe_action_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_work_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_remaining_work.get(field)),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_recommended_next_non_gpu_focus_match",
            reason="remaining_work_summary_recommended_next_non_gpu_focus_does_not_match_computed",
            left=str(summary.get("recommended_next_non_gpu_focus") or ""),
            right=str(computed_remaining_work.get("recommended_next_non_gpu_focus") or ""),
        )

    if not artifact_freshness:
        _failure(failures, "artifact_freshness_audit", "artifact_freshness_audit_missing")
    else:
        freshness_readiness = _mapping(artifact_freshness.get("readiness"))
        freshness_upstream = [_mapping(item) for item in _list(artifact_freshness.get("upstream_artifacts"))]
        required_upstream = [row for row in freshness_upstream if bool(row.get("required"))]
        if not freshness_readiness or str(freshness_readiness.get("id") or "") != "readiness_next_actions":
            _failure(
                failures,
                "artifact_freshness_readiness_row",
                "artifact_freshness_readiness_row_missing_or_wrong",
                value=freshness_readiness,
            )
        if not required_upstream:
            _failure(
                failures,
                "artifact_freshness_required_upstream_rows",
                "artifact_freshness_required_upstream_rows_missing",
            )
        elif any(not str(row.get("id") or "") or not bool(row.get("exists")) for row in required_upstream):
            _failure(
                failures,
                "artifact_freshness_required_upstream_integrity",
                "artifact_freshness_required_upstream_row_missing_id_or_exists",
                value=required_upstream[:20],
            )
        if bool(artifact_freshness.get("safe_to_auto_start")) or bool(
            artifact_freshness.get("release_claim_allowed")
        ):
            _failure(
                failures,
                "artifact_freshness_fail_closed",
                "artifact_freshness_audit_allows_release_or_auto_start",
            )
        if _safe_int(artifact_freshness.get("required_artifact_missing_count")) != 0:
            _failure(
                failures,
                "artifact_freshness_required_missing",
                "required_artifact_missing_in_freshness_audit",
                value=_strings(artifact_freshness.get("required_artifact_missing_ids")),
            )
        if _safe_int(artifact_freshness.get("upstream_newer_than_readiness_count")) != 0:
            _failure(
                failures,
                "artifact_freshness_readiness_stale",
                "readiness_older_than_upstream_artifact",
                value=_strings(artifact_freshness.get("upstream_newer_than_readiness_ids")),
            )
        if not bool(artifact_freshness.get("readiness_not_older_than_upstream")):
            _failure(
                failures,
                "artifact_freshness_readiness_order",
                "readiness_not_older_than_upstream_false",
                value=_strings(artifact_freshness.get("drift_reason_ids")),
            )
        if not bool(artifact_freshness.get("terminal_observation_not_older_than_readiness")):
            _failure(
                failures,
                "artifact_freshness_terminal_order",
                "terminal_observation_older_than_readiness",
                value=_mapping(artifact_freshness.get("readiness")),
            )
        if not bool(artifact_freshness.get("freshness_ok")):
            _failure(
                failures,
                "artifact_freshness_ok",
                "artifact_freshness_audit_not_ok",
                value=_strings(artifact_freshness.get("drift_reason_ids")),
            )

    if not terminal_unblocker:
        _failure(failures, "terminal_release_unblocker_summary", "terminal_release_unblocker_summary_missing")
    else:
        if bool(terminal_unblocker.get("safe_to_auto_start")) or bool(
            terminal_unblocker.get("release_claim_allowed")
        ) or bool(terminal_unblocker.get("gpu_bubble_release_claim_allowed")):
            _failure(
                failures,
                "terminal_release_unblocker_fail_closed",
                "terminal_release_unblocker_summary_allows_release_or_auto_start",
            )
        _compare_set_field(
            failures,
            check_id="terminal_unblocker_hard_gates_match",
            reason="terminal_release_unblocker_hard_gates_do_not_match_readiness",
            left=readiness_hard_gates,
            right=_strings(terminal_unblocker.get("gpu_bubble_release_hard_gate_ids")),
        )
        _compare_set_field(
            failures,
            check_id="terminal_unblocker_missing_inputs_match",
            reason="terminal_release_unblocker_missing_inputs_do_not_match_readiness",
            left=_strings(unblocker.get("missing_external_inputs")),
            right=_strings(terminal_unblocker.get("missing_external_inputs")),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_pipeline_status_match",
            reason="terminal_release_unblocker_pipeline_status_does_not_match_readiness",
            left=str(unblocker.get("source_cache_axis_pipeline_status") or ""),
            right=str(terminal_unblocker.get("source_cache_axis_pipeline_status") or ""),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_pipeline_stage_count_match",
            reason="terminal_release_unblocker_pipeline_stage_count_does_not_match_readiness",
            left=_safe_int(unblocker.get("source_cache_axis_stage_count")),
            right=_safe_int(terminal_unblocker.get("source_cache_axis_stage_count")),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_pipeline_stage_ok_match",
            reason="terminal_release_unblocker_pipeline_stage_ok_count_does_not_match_readiness",
            left=_safe_int(unblocker.get("source_cache_axis_stage_ok_count")),
            right=_safe_int(terminal_unblocker.get("source_cache_axis_stage_ok_count")),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_post_manual_status_match",
            reason="terminal_release_unblocker_post_manual_status_does_not_match_readiness",
            left=str(unblocker.get("post_manual_rebuild_status") or ""),
            right=str(terminal_unblocker.get("post_manual_rebuild_status") or ""),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_post_manual_next_stage_match",
            reason="terminal_release_unblocker_post_manual_next_stage_does_not_match_readiness",
            left=str(unblocker.get("post_manual_next_rebuild_stage_id") or ""),
            right=str(terminal_unblocker.get("post_manual_next_rebuild_stage_id") or ""),
        )
        _compare_scalar_field(
            failures,
            check_id="terminal_unblocker_sd15_status_match",
            reason="terminal_release_unblocker_sd15_status_does_not_match_readiness",
            left=str(unblocker.get("sd15_release_gap_status") or ""),
            right=str(terminal_unblocker.get("sd15_release_gap_status") or ""),
        )
        _compare_set_field(
            failures,
            check_id="terminal_unblocker_sd15_blockers_match",
            reason="terminal_release_unblocker_sd15_blockers_do_not_match_readiness",
            left=_strings(unblocker.get("sd15_release_gap_blockers")),
            right=_strings(terminal_unblocker.get("sd15_release_gap_blockers")),
        )
        _compare_set_field(
            failures,
            check_id="terminal_unblocker_refresh_sequence_match",
            reason="terminal_release_unblocker_refresh_sequence_does_not_match_readiness",
            left=_strings(readiness_input.get("next_json_refresh_sequence")),
            right=_strings(_mapping(terminal_unblocker.get("input_resolution_summary")).get("next_json_refresh_sequence")),
        )

    terminal_unblocker_input = _mapping(terminal_unblocker.get("input_resolution_summary"))
    input_resolution_targets = {
        "terminal_input_resolution": terminal_input,
        "terminal_unblocker_input_resolution": terminal_unblocker_input,
    }
    if not readiness_input:
        _failure(failures, "readiness_input_resolution_summary", "readiness_input_resolution_summary_missing")
    for check_id, candidate in input_resolution_targets.items():
        if not candidate:
            _failure(failures, check_id, "input_resolution_summary_missing")
            continue
        if str(candidate.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "input_resolution_summary_roadmap_missing_or_wrong",
                value=candidate.get("roadmap"),
            )
        if bool(candidate.get("safe_to_auto_start")) or bool(candidate.get("release_claim_allowed")):
            _failure(failures, check_id, "input_resolution_summary_allows_release_or_auto_start")
        if not bool(candidate.get("not_release_evidence")):
            _failure(failures, check_id, "input_resolution_summary_not_marked_non_release")
        _compare_set_field(
            failures,
            check_id=f"{check_id}_missing_inputs_match",
            reason="input_resolution_missing_external_inputs_do_not_match_readiness",
            left=_strings(readiness_input.get("missing_external_inputs")),
            right=_strings(candidate.get("missing_external_inputs")),
        )
        for field in [
            "sd15_checkpoint_exists",
            "sd15_checkpoint_required",
            "new_source_root_required",
            "source_or_cache_axis_required",
            "warm_cache_or_caption_repair_required",
            "external_input_detected",
            "json_replay_ready",
            "preflight_admitted",
            "manual_canary_plan_ready",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"input_resolution_{field}_does_not_match_readiness",
                left=bool(readiness_input.get(field)),
                right=bool(candidate.get(field)),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_new_source_root_count_match",
            reason="input_resolution_new_source_root_count_does_not_match_readiness",
            left=_safe_int(readiness_input.get("new_source_root_count")),
            right=_safe_int(candidate.get("new_source_root_count")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_refresh_sequence_match",
            reason="input_resolution_refresh_sequence_does_not_match_readiness",
            left=_strings(readiness_input.get("next_json_refresh_sequence")),
            right=_strings(candidate.get("next_json_refresh_sequence")),
        )

    transition_targets = {
        "terminal_external_input_transition_table": terminal_transition,
        "terminal_unblocker_external_input_transition_table": terminal_unblocker_transition,
    }
    if not readiness_transition:
        _failure(
            failures,
            "readiness_external_input_transition_table",
            "external_input_transition_table_missing",
        )
    else:
        readiness_transition_rows = [
            _mapping(item) for item in _list(readiness_transition.get("rows"))
        ]
        if str(readiness_transition.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                "readiness_external_input_transition_roadmap",
                "external_input_transition_table_roadmap_missing_or_wrong",
                value=readiness_transition.get("roadmap"),
            )
        if not bool(readiness_transition.get("not_release_evidence")):
            _failure(
                failures,
                "readiness_external_input_transition_not_release",
                "external_input_transition_table_not_marked_non_release",
            )
        if (
            bool(readiness_transition.get("safe_to_auto_start"))
            or bool(readiness_transition.get("release_claim_allowed"))
            or not bool(readiness_transition.get("fail_closed"))
        ):
            _failure(
                failures,
                "readiness_external_input_transition_fail_closed",
                "external_input_transition_table_allows_release_or_auto_start",
            )
        if _safe_int(readiness_transition.get("row_count")) != len(readiness_transition_rows):
            _failure(
                failures,
                "readiness_external_input_transition_row_count",
                "external_input_transition_row_count_does_not_match_rows",
                value={
                    "row_count": readiness_transition.get("row_count"),
                    "actual_row_count": len(readiness_transition_rows),
                },
            )
        if _safe_int(readiness_transition.get("unsafe_row_count")) != 0:
            _failure(
                failures,
                "readiness_external_input_transition_unsafe_rows",
                "external_input_transition_table_reports_unsafe_rows",
                value=_strings(readiness_transition.get("unsafe_row_ids")),
            )
        _compare_set_field(
            failures,
            check_id="readiness_external_input_transition_missing_inputs_match",
            reason="external_input_transition_missing_inputs_do_not_match_release_unblocker",
            left=_strings(unblocker.get("missing_external_inputs")),
            right=_strings(readiness_transition.get("missing_external_inputs")),
        )
        _compare_set_field(
            failures,
            check_id="readiness_external_input_transition_refresh_sequence_match",
            reason="external_input_transition_refresh_sequence_does_not_match_input_resolution",
            left=_strings(readiness_input.get("next_json_refresh_sequence")),
            right=_strings(readiness_transition.get("next_json_refresh_sequence")),
        )
        for row in readiness_transition_rows:
            row_id = str(row.get("input_id") or "")
            if not row_id:
                _failure(
                    failures,
                    "readiness_external_input_transition_row_id",
                    "external_input_transition_row_missing_input_id",
                )
            if (
                bool(row.get("safe_to_auto_start"))
                or bool(row.get("release_claim_allowed"))
                or bool(row.get("release_claim_allowed_after_success"))
                or not bool(row.get("not_release_evidence"))
            ):
                _failure(
                    failures,
                    f"readiness_external_input_transition_row_{row_id or 'unknown'}",
                    "external_input_transition_row_allows_release_or_auto_start",
                    value=row,
                )
            if bool(row.get("required")) and row_id not in _strings(
                readiness_transition.get("missing_external_inputs")
            ):
                _failure(
                    failures,
                    f"readiness_external_input_transition_required_{row_id}",
                    "external_input_transition_required_row_not_in_missing_inputs",
                    value=row,
                )
    for check_id, candidate in transition_targets.items():
        if not candidate:
            _failure(failures, check_id, "external_input_transition_table_missing")
            continue
        if str(candidate.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "external_input_transition_table_roadmap_missing_or_wrong",
                value=candidate.get("roadmap"),
            )
        if (
            bool(candidate.get("safe_to_auto_start"))
            or bool(candidate.get("release_claim_allowed"))
            or not bool(candidate.get("not_release_evidence"))
            or not bool(candidate.get("fail_closed"))
        ):
            _failure(failures, check_id, "external_input_transition_table_allows_release_or_auto_start")
        for field in [
            "transition_status",
            "row_count",
            "required_row_count",
            "blocked_row_count",
            "detected_row_count",
            "admitted_row_count",
            "manual_plan_ready_row_count",
            "unsafe_row_count",
            "replay_command_count",
            "replay_ready_command_count",
        ]:
            left_value = (
                _safe_int(readiness_transition.get(field))
                if field.endswith("_count")
                else str(readiness_transition.get(field) or "")
            )
            right_value = (
                _safe_int(candidate.get(field))
                if field.endswith("_count")
                else str(candidate.get(field) or "")
            )
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"external_input_transition_{field}_does_not_match_readiness",
                left=left_value,
                right=right_value,
            )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_missing_inputs_match",
            reason="external_input_transition_missing_inputs_do_not_match_readiness",
            left=_strings(readiness_transition.get("missing_external_inputs")),
            right=_strings(candidate.get("missing_external_inputs")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_refresh_sequence_match",
            reason="external_input_transition_refresh_sequence_do_not_match_readiness",
            left=_strings(readiness_transition.get("next_json_refresh_sequence")),
            right=_strings(candidate.get("next_json_refresh_sequence")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_unsafe_rows_match",
            reason="external_input_transition_unsafe_rows_do_not_match_readiness",
            left=_strings(readiness_transition.get("unsafe_row_ids")),
            right=_strings(candidate.get("unsafe_row_ids")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_row_ids_match",
            reason="external_input_transition_row_ids_do_not_match_readiness",
            left=[str(row.get("input_id") or "") for row in _list(readiness_transition.get("rows"))],
            right=[str(_mapping(row).get("input_id") or "") for row in _list(candidate.get("rows"))],
        )

    if not external_input_audit:
        _failure(failures, "external_input_filesystem_audit", "external_input_filesystem_audit_missing")
    else:
        unblocker_missing_inputs = _strings(unblocker.get("missing_external_inputs"))
        audit_expected_missing_inputs = _strings(external_input_audit.get("expected_missing_external_inputs"))
        if set(audit_expected_missing_inputs) != set(unblocker_missing_inputs):
            _failure(
                failures,
                "external_input_expected_missing_inputs_match",
                "external_input_expected_missing_inputs_do_not_match_release_unblocker",
                value={"expected": audit_expected_missing_inputs, "unblocker": unblocker_missing_inputs},
            )
        if not bool(external_input_audit.get("live_scan_available")):
            _failure(failures, "external_input_live_scan", "external_input_live_scan_missing")
        if not bool(external_input_audit.get("registry_available")):
            _failure(failures, "external_input_intake_registry", "external_input_intake_registry_missing")
        if not bool(external_input_audit.get("registry_matches_live_scan")):
            _failure(
                failures,
                "external_input_registry_live_match",
                "external_input_intake_registry_does_not_match_live_scan",
                value={
                    "registry": _strings(external_input_audit.get("registry_missing_external_inputs")),
                    "live": _strings(external_input_audit.get("live_missing_external_inputs")),
                },
            )
        if not bool(external_input_audit.get("live_or_registry_matches_expected_missing_inputs")):
            _failure(
                failures,
                "external_input_missing_inputs_match",
                "external_input_missing_inputs_do_not_match_release_unblocker",
                value={
                    "expected": _strings(external_input_audit.get("expected_missing_external_inputs")),
                    "live": _strings(external_input_audit.get("live_missing_external_inputs")),
                },
            )
        if bool(external_input_audit.get("filesystem_external_input_detected")):
            _failure(
                failures,
                "external_input_filesystem_detected",
                "filesystem_has_external_input_but_readiness_is_still_blocked_missing_input",
                value={
                    "sd15_checkpoint_exists": external_input_audit.get("sd15_checkpoint_exists"),
                    "new_source_root_count": external_input_audit.get("new_source_root_count"),
                    "new_source_roots": _strings(external_input_audit.get("new_source_roots")),
                },
            )
        expected_missing = set(_strings(external_input_audit.get("expected_missing_external_inputs")))
        if "sd15_checkpoint" in expected_missing and (
            bool(external_input_audit.get("sd15_checkpoint_exists"))
            or _safe_int(external_input_audit.get("sd15_checkpoint_count")) > 0
        ):
            _failure(failures, "external_input_sd15_drift", "sd15_checkpoint_present_but_marked_missing")
        if "new_source_root" in expected_missing and _safe_int(external_input_audit.get("new_source_root_count")) > 0:
            _failure(failures, "external_input_source_root_drift", "new_source_root_present_but_marked_missing")
        if not bool(external_input_audit.get("artifact_matches_filesystem_and_release_blockers")):
            _failure(
                failures,
                "external_input_artifact_filesystem_match",
                "external_input_artifacts_do_not_match_filesystem_or_release_blockers",
                value=_strings(external_input_audit.get("drift_reason_ids")),
            )

    computed_source_axis_freshness = (
        readiness_source_axis_freshness or readiness_evidence_source_axis_freshness
    )
    for check_id, summary in {
        "readiness_source_axis_freshness_dedupe_audit": readiness_source_axis_freshness,
        "readiness_evidence_source_axis_freshness_dedupe_audit": readiness_evidence_source_axis_freshness,
        "terminal_source_axis_freshness_dedupe_audit": terminal_source_axis_freshness,
    }.items():
        if not summary:
            _failure(failures, check_id, "source_axis_freshness_dedupe_audit_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "source_axis_freshness_dedupe_audit_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("publishable"))
            or bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("does_not_run_training"))
            or not bool(summary.get("does_not_run_cuda"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_audit_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "source_axis_freshness_dedupe_audit_allows_release_or_auto_start",
            )
        missing_external = set(_strings(unblocker.get("missing_external_inputs")))
        if "new_source_root" in missing_external:
            if _safe_int(summary.get("new_source_root_count")) != 0:
                _failure(
                    failures,
                    f"{check_id}_new_source_root_count",
                    "new_source_root_missing_but_freshness_reports_new_roots",
                    value=summary.get("new_source_root_count"),
                )
            if bool(summary.get("external_input_detected")):
                _failure(
                    failures,
                    f"{check_id}_external_input_detected",
                    "new_source_root_missing_but_freshness_reports_external_input",
                )
        if bool(summary.get("candidate_fresh")) and not bool(summary.get("preflight_admitted")):
            _failure(
                failures,
                f"{check_id}_candidate_preflight",
                "fresh_source_axis_candidate_without_preflight_admission",
            )
        if bool(summary.get("manual_canary_plan_ready")) and not bool(summary.get("preflight_admitted")):
            _failure(
                failures,
                f"{check_id}_manual_canary_preflight",
                "source_axis_manual_canary_ready_without_preflight_admission",
            )
        for field in [
            "report",
            "status",
            "axis_state",
            "candidate_status",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_freshness_dedupe_audit_{field}_does_not_match_readiness",
                left=str(summary.get(field) or ""),
                right=str(computed_source_axis_freshness.get(field) or ""),
            )
        for field in [
            "external_input_detected",
            "candidate_fresh",
            "candidate_duplicate_or_stale",
            "preflight_admitted",
            "manual_canary_plan_ready",
            "publishable",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_freshness_dedupe_audit_{field}_does_not_match_readiness",
                left=bool(summary.get(field)),
                right=bool(computed_source_axis_freshness.get(field)),
            )
        for field in [
            "new_source_root_count",
            "completed_axis_count",
            "completed_out_dir_count",
            "matching_axis_count",
            "blocker_count",
            "unsafe_audit_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_freshness_dedupe_audit_{field}_does_not_match_readiness",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_source_axis_freshness.get(field)),
            )
        for field in [
            "new_source_roots",
            "blockers",
            "acceptance_gates",
            "blocked_actions",
            "unsafe_audit_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_freshness_dedupe_audit_{field}_does_not_match_readiness",
                left=_strings(summary.get(field)),
                right=_strings(computed_source_axis_freshness.get(field)),
            )

    computed_source_axis_requirement = (
        readiness_source_axis_requirement or readiness_evidence_source_axis_requirement
    )
    computed_source_axis_requirement_action_count = len(
        [
            action
            for action in _list(readiness.get("next_actions"))
            if str(_mapping(action).get("action_type") or "") == "external_source_or_cache_axis"
            and bool(_mapping(action).get("requires_external_input"))
        ]
    )
    for check_id, summary in {
        "readiness_source_axis_requirement_summary": readiness_source_axis_requirement,
        "readiness_evidence_source_axis_requirement_summary": readiness_evidence_source_axis_requirement,
        "terminal_source_axis_requirement_summary": terminal_source_axis_requirement,
    }.items():
        if not summary:
            _failure(failures, check_id, "source_axis_requirement_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "source_axis_requirement_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if str(summary.get("artifact_role") or "") != "gpu_bubble_source_axis_requirement_summary":
            _failure(
                failures,
                check_id,
                "source_axis_requirement_summary_artifact_role_wrong",
                value=summary.get("artifact_role"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("does_not_run_training"))
            or not bool(summary.get("does_not_run_cuda"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(
                failures,
                check_id,
                "source_axis_requirement_summary_allows_release_or_auto_start",
            )
        for field in ["report", "status"]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_requirement_summary_{field}_does_not_match_readiness",
                left=str(summary.get(field) or ""),
                right=str(computed_source_axis_requirement.get(field) or ""),
            )
        for field in [
            "external_input_required",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_requirement_summary_{field}_does_not_match_readiness",
                left=bool(summary.get(field)),
                right=bool(computed_source_axis_requirement.get(field)),
            )
        for field in [
            "family_count",
            "external_input_required_count",
            "candidate_available_family_count",
            "exhausted_family_count",
            "no_ready_source_axis_family_count",
            "completed_existing_command_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_axis_requirement_summary_{field}_does_not_match_readiness",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_source_axis_requirement.get(field)),
            )
        if computed_source_axis_requirement_action_count > 0:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_external_input_action_count_match",
                reason="source_axis_requirement_external_input_count_does_not_match_next_actions",
                left=_safe_int(summary.get("external_input_required_count")),
                right=computed_source_axis_requirement_action_count,
            )

    computed_source_cache_pipeline_summary = (
        readiness_source_cache_pipeline_summary or readiness_evidence_source_cache_pipeline_summary
    )
    for check_id, summary in {
        "readiness_source_cache_axis_pipeline_readiness_summary": readiness_source_cache_pipeline_summary,
        "readiness_evidence_source_cache_axis_pipeline_readiness_summary": (
            readiness_evidence_source_cache_pipeline_summary
        ),
        "terminal_source_cache_axis_pipeline_readiness_summary": terminal_source_cache_pipeline_summary,
    }.items():
        if not summary:
            _failure(failures, check_id, "source_cache_axis_pipeline_readiness_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "source_cache_axis_pipeline_readiness_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if str(summary.get("artifact_role") or "") != "gpu_bubble_source_cache_axis_pipeline_readiness_summary":
            _failure(
                failures,
                check_id,
                "source_cache_axis_pipeline_readiness_summary_artifact_role_wrong",
                value=summary.get("artifact_role"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("does_not_run_training"))
            or not bool(summary.get("does_not_run_cuda"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(
                failures,
                check_id,
                "source_cache_axis_pipeline_readiness_summary_allows_release_or_auto_start",
            )
        if bool(summary.get("manual_canary_plan_ready")) and not bool(summary.get("preflight_admitted")):
            _failure(
                failures,
                f"{check_id}_manual_canary_preflight",
                "source_cache_axis_pipeline_manual_canary_ready_without_preflight",
            )
        if bool(summary.get("preflight_admitted")) and not bool(summary.get("external_input_required")):
            _failure(
                failures,
                f"{check_id}_preflight_requires_external_input",
                "source_cache_axis_pipeline_preflight_admitted_without_external_input_context",
            )
        for field in ["report", "status", "axis_readiness_status"]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_cache_axis_pipeline_readiness_summary_{field}_does_not_match_readiness",
                left=str(summary.get(field) or ""),
                right=str(computed_source_cache_pipeline_summary.get(field) or ""),
            )
        for field in [
            "pipeline_complete",
            "external_input_required",
            "preflight_admitted",
            "manual_canary_plan_ready",
            "waiting_external_input",
            "duplicate_or_stale_axis_blocked",
            "cache_axis_not_ready",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_cache_axis_pipeline_readiness_summary_{field}_does_not_match_readiness",
                left=bool(summary.get(field)),
                right=bool(computed_source_cache_pipeline_summary.get(field)),
            )
        for field in [
            "stage_count",
            "stage_ok_count",
            "blocker_count",
            "next_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"source_cache_axis_pipeline_readiness_summary_{field}_does_not_match_readiness",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_source_cache_pipeline_summary.get(field)),
            )

    external_input_admission_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="external_input_admission_summary",
        artifact_role="gpu_bubble_external_input_admission_summary",
        readiness_summary=readiness_external_input_admission,
        evidence_summary=evidence_external_input_admission,
        terminal_summary=terminal_external_input_admission,
        string_fields=["report", "status", "sd15_status", "source_axis_status"],
        bool_fields=[
            "external_input_required",
            "sd15_checkpoint_exists",
            "sd15_evidence_ready",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "source_axis_external_input_required_count",
            "source_axis_candidate_available_family_count",
        ],
    )
    external_input_intake_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="external_input_intake_registry_summary",
        artifact_role="gpu_bubble_external_input_intake_registry_summary",
        readiness_summary=readiness_external_input_intake,
        evidence_summary=evidence_external_input_intake,
        terminal_summary=terminal_external_input_intake,
        string_fields=["report", "status"],
        bool_fields=[
            "external_input_detected",
            "external_input_required",
            "publishable",
            "sd15_checkpoint_exists",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "source_new_root_count",
            "intake_item_count",
            "missing_external_input_count",
            "registration_slot_count",
            "rescan_request_count",
        ],
    )
    external_input_replay_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="external_input_replay_plan_summary",
        artifact_role="gpu_bubble_external_input_replay_plan_summary",
        readiness_summary=readiness_external_input_replay,
        evidence_summary=evidence_external_input_replay,
        terminal_summary=terminal_external_input_replay,
        string_fields=["report", "status"],
        bool_fields=[
            "external_input_detected",
            "sd15_checkpoint_exists",
            "publishable",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "new_source_root_count",
            "command_count",
            "ready_command_count",
            "template_command_count",
        ],
    )
    external_input_handoff_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="external_input_handoff_packet_summary",
        artifact_role="gpu_bubble_external_input_handoff_packet_summary",
        readiness_summary=readiness_external_input_handoff,
        evidence_summary=evidence_external_input_handoff,
        terminal_summary=terminal_external_input_handoff,
        string_fields=[
            "report",
            "status",
            "input_lifecycle_status",
            "replay_status",
            "next_manual_gpu_gate",
        ],
        bool_fields=[
            "external_input_detected",
            "external_input_required",
            "sd15_checkpoint_required",
            "source_or_cache_axis_required",
            "warm_cache_or_caption_repair_required",
            "json_replay_ready",
            "publishable",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "missing_external_input_count",
            "detected_input_count",
            "accepted_input_count",
            "detected_unaccepted_input_count",
            "pending_input_count",
            "handoff_step_count",
            "registration_slot_count",
            "command_count",
            "ready_command_count",
            "blocked_command_count",
            "unsafe_command_count",
            "replay_command_count",
            "replay_ready_command_count",
        ],
        list_fields=[
            "missing_external_inputs",
            "detected_input_ids",
            "accepted_input_ids",
            "detected_unaccepted_input_ids",
            "pending_input_ids",
            "release_gate_blockers",
        ],
    )
    external_input_json_refresh_runner_manifest_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="external_input_json_refresh_runner_manifest_summary",
        artifact_role="gpu_bubble_external_input_json_refresh_runner_manifest_summary",
        readiness_summary=readiness_external_input_json_refresh_runner_manifest,
        evidence_summary=evidence_external_input_json_refresh_runner_manifest,
        terminal_summary=terminal_external_input_json_refresh_runner_manifest,
        string_fields=[
            "status",
            "manifest_probe",
            "stage_manifest_source",
            "execution_policy",
        ],
        bool_fields=[
            "manifest_available",
            "manifest_ok",
            "runner_ready",
            "execution_ok",
            "row_execution_consistent",
            "sequence_ok",
            "stage_manifest_ok",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "expected_command_count",
            "command_count",
            "row_count",
            "executed_count",
            "executed_row_count",
            "failure_count",
            "failed_row_count",
            "output_missing_count",
            "missing_output_row_count",
            "forbidden_heavy_flag_count",
            "row_forbidden_heavy_flag_count",
            "unsafe_row_count",
            "validation_issue_count",
            "stage_manifest_issue_count",
            "stage_count",
            "script_count",
            "expected_output_count",
            "stage_manifest_forbidden_heavy_flag_count",
        ],
        list_fields=[
            "canonical_stage_ids",
            "observed_stage_ids",
            "row_stage_ids",
            "stage_manifest_issue_reasons",
            "stage_ids",
        ],
    )
    newbie_warm_cache_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="newbie_warm_cache_inventory_summary",
        artifact_role="gpu_bubble_newbie_warm_cache_inventory_summary",
        readiness_summary=readiness_newbie_warm_cache,
        evidence_summary=evidence_newbie_warm_cache,
        terminal_summary=terminal_newbie_warm_cache,
        string_fields=[
            "report",
            "status",
            "selected_axis_kind",
            "selected_axis_caption_coverage",
        ],
        bool_fields=[
            "selected_axis_cache_ready",
            "evidence_pack_indexed",
            "release_claim_allowed",
            "claimable",
            "selected_axis_supersedes_cache_missing_blockers",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
        ],
        int_fields=[
            "axis_count",
            "ready_axis_count",
            "completed_canary_axis_count",
            "selected_axis_completed_canary_command_count",
            "selected_axis_sample_count",
            "selected_axis_manifest_sample_count",
            "selected_axis_metadata_sample_count",
            "historical_cache_readiness_blocker_count",
        ],
    )
    source_cache_identity_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="source_cache_axis_identity_registry_summary",
        artifact_role="gpu_bubble_source_cache_axis_identity_registry",
        readiness_summary=readiness_source_cache_identity_summary,
        evidence_summary=evidence_source_cache_identity_summary,
        terminal_summary=terminal_source_cache_identity_summary,
        string_fields=["report", "status", "axis_state"],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "publishable",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "identity_schema_version",
            "row_count",
            "root_identity_row_count",
            "full_axis_identity_row_count",
            "current_source_root_count",
            "new_source_root_count",
            "duplicate_or_stale_axis_count",
            "fresh_axis_candidate_count",
            "unsafe_row_count",
        ],
        list_fields=["unsafe_row_ids"],
    )
    source_cache_preflight_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="source_cache_axis_admission_preflight_summary",
        artifact_role="gpu_bubble_source_cache_axis_admission_preflight_summary",
        readiness_summary=readiness_source_cache_preflight,
        evidence_summary=evidence_source_cache_preflight,
        terminal_summary=terminal_source_cache_preflight,
        string_fields=["report", "status", "candidate_family", "candidate_root"],
        bool_fields=[
            "admission_allows_protected_manual_gpu_plan",
            "matched_axis_found",
            "matched_axis_cache_ready",
            "matched_axis_quality_ok",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=["blocker_count"],
    )
    source_cache_manual_plan_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="source_cache_axis_manual_canary_plan_summary",
        artifact_role="gpu_bubble_source_cache_axis_manual_canary_plan_summary",
        readiness_summary=readiness_source_cache_manual_plan,
        evidence_summary=evidence_source_cache_manual_plan,
        terminal_summary=terminal_source_cache_manual_plan,
        string_fields=["report", "status", "preflight_status"],
        bool_fields=[
            "preflight_admitted",
            "requires_gpu_if_executed",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=["command_count", "blocked_command_count", "blocker_count"],
    )
    post_manual_rebuild_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="post_manual_evidence_rebuild_plan_summary",
        artifact_role="gpu_bubble_post_manual_evidence_rebuild_plan_summary",
        readiness_summary=readiness_post_manual_rebuild,
        evidence_summary=evidence_post_manual_rebuild,
        terminal_summary=terminal_post_manual_rebuild,
        string_fields=[
            "report",
            "status",
            "sd15_status",
            "next_rebuild_stage_id",
            "release_readiness",
            "natural_load_status",
        ],
        bool_fields=[
            "manual_canary_plan_ready",
            "manual_gpu_evidence_ready",
            "manual_gpu_evidence_required",
            "source_cache_axis_manual_canary_plan_required",
            "sd15_checkpoint_required",
            "natural_load_canary_pending",
            "release_claims_rebuild_required",
            "publishable",
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "manual_canary_command_count",
            "command_count",
            "ready_command_count",
            "stage_count",
            "ready_stage_count",
            "blocked_stage_count",
            "expected_output_count",
            "existing_expected_output_count",
            "missing_expected_output_count",
            "blocked_expected_output_count",
            "pending_expected_output_count",
            "evidence_gap_count",
            "natural_load_ready_family_count",
            "natural_load_family_count",
            "blocker_count",
        ],
        list_fields=["release_gate_blockers", "next_required_inputs", "first_blocked_stage_ids"],
    )
    manual_review_artifact_chain_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="manual_review_artifact_chain_summary",
        artifact_role="gpu_bubble_manual_review_artifact_chain_summary",
        readiness_summary=readiness_manual_review_artifact_chain,
        evidence_summary=readiness_evidence_manual_review_artifact_chain,
        terminal_summary=terminal_manual_review_artifact_chain,
        string_fields=["status"],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "expected_artifact_count",
            "present_artifact_count",
            "missing_artifact_count",
            "manual_review_ready_count",
            "closed_blocked_or_regression_count",
            "closed_diagnostic_or_promotion_count",
            "followup_gpu_required_action_count",
            "unsafe_artifact_count",
        ],
        list_fields=[
            "present_artifact_ids",
            "missing_artifact_ids",
            "unsafe_artifact_ids",
            "blocked_actions",
        ],
    )
    sdxl_diagnostic_artifact_chain_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="sdxl_diagnostic_artifact_chain_summary",
        artifact_role="gpu_bubble_sdxl_diagnostic_artifact_chain_summary",
        readiness_summary=readiness_sdxl_diagnostic_artifact_chain,
        evidence_summary=readiness_evidence_sdxl_diagnostic_artifact_chain,
        terminal_summary=terminal_sdxl_diagnostic_artifact_chain,
        string_fields=[
            "status",
            "probe_status",
            "debug_repeat_status",
            "manual_gpu_queue_status",
        ],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "expected_artifact_count",
            "present_artifact_count",
            "missing_artifact_count",
            "probe_group_count",
            "probe_rollback_group_count",
            "probe_pending_group_count",
            "debug_repeat_candidate_pass_count",
            "debug_repeat_fully_repeated_candidate_count",
            "debug_repeat_missing_report_count",
            "debug_repeat_missing_summary_count",
            "debug_repeat_execution_failure_count",
            "manual_gpu_queue_item_count",
            "manual_gpu_completed_group_count",
            "manual_gpu_pending_ready_command_count",
            "manual_gpu_completed_summary_count",
            "manual_gpu_missing_summary_count",
            "unsafe_artifact_count",
        ],
        list_fields=[
            "present_artifact_ids",
            "missing_artifact_ids",
            "unsafe_artifact_ids",
            "blocked_actions",
        ],
    )
    protected_followup_run_plan_artifact_chain_summary = _check_json_only_summary_mirror(
        failures,
        summary_id="protected_followup_run_plan_artifact_chain_summary",
        artifact_role="gpu_bubble_protected_followup_run_plan_artifact_chain_summary",
        readiness_summary=readiness_protected_run_plan_chain,
        evidence_summary=readiness_evidence_protected_run_plan_chain,
        terminal_summary=terminal_protected_run_plan_chain,
        string_fields=["status", "execution_policy"],
        bool_fields=[
            "fail_closed",
            "not_release_evidence",
            "does_not_run_training",
            "does_not_run_cuda",
            "safe_to_auto_start",
            "release_claim_allowed",
        ],
        int_fields=[
            "expected_artifact_count",
            "present_artifact_count",
            "missing_artifact_count",
            "total_command_count",
            "manual_start_required_command_count",
            "release_claim_allowed_after_success_command_count",
            "unsafe_command_count",
            "unsafe_scaffold_count",
            "contract_ok_artifact_count",
            "unsafe_artifact_count",
        ],
        list_fields=[
            "present_artifact_ids",
            "missing_artifact_ids",
            "unsafe_artifact_ids",
        ],
    )
    if (
        not bool(terminal_source_cache_manual_plan.get("preflight_admitted"))
        and _safe_int(terminal_source_cache_manual_plan.get("command_count")) > 0
    ):
        _failure(
            failures,
            "source_cache_manual_plan_preflight_gate",
            "source_cache_manual_plan_has_commands_without_preflight_admission",
        )
    if (
        not bool(terminal_source_cache_manual_plan.get("preflight_admitted"))
        and bool(terminal_post_manual_rebuild.get("manual_canary_plan_ready"))
    ):
        _failure(
            failures,
            "post_manual_rebuild_manual_canary_preflight_gate",
            "post_manual_rebuild_reports_manual_canary_ready_without_preflight",
        )

    if not source_cache_negative:
        _failure(
            failures,
            "source_cache_negative_evidence_summary",
            "source_cache_negative_evidence_summary_missing",
        )
    else:
        if str(source_cache_negative.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                "source_cache_negative_roadmap",
                "source_cache_negative_evidence_roadmap_missing_or_wrong",
                value=source_cache_negative.get("roadmap"),
            )
        if not bool(source_cache_negative.get("not_release_evidence")):
            _failure(
                failures,
                "source_cache_negative_not_release_evidence",
                "source_cache_negative_evidence_not_marked_non_release",
            )
        if bool(source_cache_negative.get("safe_to_auto_start")) or bool(
            source_cache_negative.get("release_claim_allowed")
        ):
            _failure(
                failures,
                "source_cache_negative_fail_closed",
                "source_cache_negative_evidence_allows_release_or_auto_start",
            )
        missing_external = set(_strings(unblocker.get("missing_external_inputs")))
        if "new_source_root" in missing_external:
            if _safe_int(source_cache_negative.get("new_source_root_count")) != 0:
                _failure(
                    failures,
                    "source_cache_negative_new_root_count",
                    "new_source_root_missing_but_negative_summary_reports_new_roots",
                    value=source_cache_negative.get("new_source_root_count"),
                )
            if not bool(source_cache_negative.get("cannot_clear_new_source_root_blocker_from_current_axis")):
                _failure(
                    failures,
                    "source_cache_negative_current_axis_duplicate",
                    "current_source_axis_not_marked_as_unable_to_clear_new_root_blocker",
                    value={
                        "current_source_root_duplicate_count": source_cache_negative.get(
                            "current_source_root_duplicate_count"
                        ),
                        "current_source_roots": _strings(source_cache_negative.get("current_source_roots")),
                    },
                )
        if "warm_cache_axis" in missing_external and not bool(
            source_cache_negative.get("cannot_clear_warm_cache_axis_from_inventory")
        ):
            _failure(
                failures,
                "source_cache_negative_warm_cache",
                "warm_cache_axis_missing_but_inventory_marked_as_clearing_blocker",
                value={
                    "newbie_warm_cache_status": source_cache_negative.get("newbie_warm_cache_status"),
                    "claimable_cache_axis_count": source_cache_negative.get("claimable_cache_axis_count"),
                },
            )
        if "warm_cache_axis" in missing_external:
            warm_cache_status = str(source_cache_negative.get("newbie_warm_cache_status") or "")
            if _safe_int(source_cache_negative.get("claimable_cache_axis_count")) != 0 or warm_cache_status in {
                "release_ready",
                "claimable",
                "warm_cache_axis_release_ready",
            }:
                _failure(
                    failures,
                    "source_cache_negative_warm_cache_claimable",
                    "warm_cache_axis_missing_but_negative_summary_reports_claimable_cache",
                    value={
                        "newbie_warm_cache_status": warm_cache_status,
                        "claimable_cache_axis_count": source_cache_negative.get("claimable_cache_axis_count"),
                    },
                )
        if _safe_int(source_cache_negative.get("external_input_required_family_count")) <= 0 and bool(
            missing_external.intersection({"new_source_root", "warm_cache_axis", "caption_repair_axis"})
        ):
            _failure(
                failures,
                "source_cache_negative_external_family_count",
                "source_cache_external_inputs_missing_but_required_family_count_zero",
            )

    if not source_cache_identity:
        _failure(
            failures,
            "source_cache_axis_identity_registry",
            "source_cache_axis_identity_registry_missing",
        )
    else:
        if str(source_cache_identity.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                "source_cache_axis_identity_registry_roadmap",
                "source_cache_axis_identity_registry_roadmap_missing_or_wrong",
                value=source_cache_identity.get("roadmap"),
            )
        if not bool(source_cache_identity.get("fail_closed")):
            _failure(
                failures,
                "source_cache_axis_identity_registry_fail_closed",
                "source_cache_axis_identity_registry_not_fail_closed",
                value=source_cache_identity.get("fail_closed"),
            )
        if _safe_int(source_cache_identity.get("row_count")) <= 0:
            _failure(
                failures,
                "source_cache_axis_identity_registry_row_count",
                "source_cache_axis_identity_registry_row_count_missing_or_zero",
                value=source_cache_identity.get("row_count"),
            )
        if _safe_int(source_cache_identity.get("unsafe_row_count")) != 0:
            _failure(
                failures,
                "source_cache_axis_identity_registry_unsafe_rows",
                "source_cache_axis_identity_registry_reports_unsafe_rows",
                value=_strings(source_cache_identity.get("unsafe_row_ids")),
            )
        if (
            bool(source_cache_identity.get("safe_to_auto_start"))
            or bool(source_cache_identity.get("release_claim_allowed"))
            or not bool(source_cache_identity.get("not_release_evidence"))
        ):
            _failure(
                failures,
                "source_cache_axis_identity_registry_fail_closed_flags",
                "source_cache_axis_identity_registry_allows_release_or_auto_start",
            )

    if not manual_gpu_summary:
        _failure(failures, "manual_gpu_execution_summary", "manual_gpu_execution_summary_missing")
    if not terminal_manual_gpu_summary:
        _failure(
            failures,
            "terminal_manual_gpu_execution_summary",
            "terminal_manual_gpu_execution_summary_missing",
        )
    if str(manual_gpu_summary.get("execution_policy") or "") != "manual_or_external_input_only":
        _failure(
            failures,
            "manual_gpu_execution_policy",
            "manual_gpu_execution_policy_is_not_manual_or_external_input_only",
            value=manual_gpu_summary.get("execution_policy"),
        )
    if str(terminal_manual_gpu_summary.get("execution_policy") or "") != "manual_or_external_input_only":
        _failure(
            failures,
            "terminal_manual_gpu_execution_policy",
            "terminal_manual_gpu_execution_policy_is_not_manual_or_external_input_only",
            value=terminal_manual_gpu_summary.get("execution_policy"),
        )
    if bool(manual_gpu_summary.get("safe_to_auto_start")) or _safe_int(
        manual_gpu_summary.get("auto_startable_gpu_action_count")
    ) != 0:
        _failure(
            failures,
            "manual_gpu_auto_start",
            "manual_gpu_execution_summary_allows_auto_start",
            value={
                "safe_to_auto_start": manual_gpu_summary.get("safe_to_auto_start"),
                "auto_startable_gpu_action_ids": _strings(
                    manual_gpu_summary.get("auto_startable_gpu_action_ids")
                ),
            },
        )
    if bool(terminal_manual_gpu_summary.get("safe_to_auto_start")) or _safe_int(
        terminal_manual_gpu_summary.get("auto_startable_gpu_action_count")
    ) != 0:
        _failure(
            failures,
            "terminal_manual_gpu_auto_start",
            "terminal_manual_gpu_execution_summary_allows_auto_start",
            value={
                "safe_to_auto_start": terminal_manual_gpu_summary.get("safe_to_auto_start"),
                "auto_startable_gpu_action_ids": _strings(
                    terminal_manual_gpu_summary.get("auto_startable_gpu_action_ids")
                ),
            },
        )
    if bool(manual_gpu_summary.get("release_claim_allowed")) or _safe_int(
        manual_gpu_summary.get("release_claim_allowed_after_success_action_count")
    ) != 0:
        _failure(
            failures,
            "manual_gpu_release_claim",
            "manual_gpu_execution_summary_allows_release_claim",
            value={
                "release_claim_allowed": manual_gpu_summary.get("release_claim_allowed"),
                "release_claim_allowed_after_success_action_ids": _strings(
                    manual_gpu_summary.get("release_claim_allowed_after_success_action_ids")
                ),
            },
        )
    if bool(terminal_manual_gpu_summary.get("release_claim_allowed")) or _safe_int(
        terminal_manual_gpu_summary.get("release_claim_allowed_after_success_action_count")
    ) != 0:
        _failure(
            failures,
            "terminal_manual_gpu_release_claim",
            "terminal_manual_gpu_execution_summary_allows_release_claim",
            value={
                "release_claim_allowed": terminal_manual_gpu_summary.get("release_claim_allowed"),
                "release_claim_allowed_after_success_action_ids": _strings(
                    terminal_manual_gpu_summary.get("release_claim_allowed_after_success_action_ids")
                ),
            },
        )
    readiness_gpu_related = _safe_int(manual_gpu_summary.get("gpu_related_action_count"))
    terminal_gpu_related = _safe_int(terminal_manual_gpu_summary.get("gpu_related_action_count"))
    if readiness_gpu_related and terminal_gpu_related and readiness_gpu_related != terminal_gpu_related:
        _failure(
            failures,
            "manual_gpu_execution_summary_match",
            "terminal_manual_gpu_execution_summary_does_not_match_readiness",
            value={"readiness": readiness_gpu_related, "terminal": terminal_gpu_related},
        )
    computed_manual_gpu_counts = _manual_gpu_counts_from_actions(readiness)
    _check_manual_gpu_summary_counts(
        failures,
        check_id="manual_gpu_execution_summary_counts_match",
        reason="manual_gpu_execution_summary_counts_do_not_match_next_actions",
        summary=manual_gpu_summary,
        expected=computed_manual_gpu_counts,
    )
    _check_manual_gpu_summary_counts(
        failures,
        check_id="terminal_manual_gpu_execution_summary_counts_match",
        reason="terminal_manual_gpu_execution_summary_counts_do_not_match_next_actions",
        summary=terminal_manual_gpu_summary,
        expected=computed_manual_gpu_counts,
    )
    auto_start_action_ids: list[str] = []
    release_claim_after_success_ids: list[str] = []
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        is_gpu_related = (
            bool(action.get("requires_gpu_heavy_run"))
            or bool(action.get("followup_requires_gpu_heavy_run"))
            or str(action.get("readiness_state") or "")
            in {
                "protected_manual_gpu_ready",
                "followup_axis_preparation_ready",
                "blocked_missing_prerequisite",
                "waiting_manual_gpu_evidence",
            }
        )
        if not is_gpu_related:
            continue
        action_id = str(action.get("id") or "")
        if bool(action.get("safe_to_auto_start")):
            auto_start_action_ids.append(action_id)
        if bool(action.get("release_claim_allowed_after_success")):
            release_claim_after_success_ids.append(action_id)
    if auto_start_action_ids:
        _failure(
            failures,
            "manual_gpu_action_auto_start",
            "gpu_related_next_action_allows_auto_start",
            value=_unique(auto_start_action_ids),
        )
    if release_claim_after_success_ids:
        _failure(
            failures,
            "manual_gpu_action_release_claim",
            "gpu_related_next_action_allows_release_claim_after_success",
            value=_unique(release_claim_after_success_ids),
        )
    for check_id, summary in {
        "readiness_next_action_machine_summary": readiness_next_action_machine,
        "terminal_next_action_machine_summary": terminal_next_action_machine,
    }.items():
        if not summary:
            _failure(failures, check_id, "next_action_machine_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "next_action_machine_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, "next_action_machine_summary_allows_release_or_auto_start")
        for field in [
            "unique_action_count",
            "json_ready_action_count",
            "json_closed_action_count",
            "unsafe_action_count",
            "missing_machine_field_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"next_action_machine_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_next_action_machine.get(field)),
            )
        if dict(_mapping(summary.get("readiness_state_counts"))) != dict(
            computed_next_action_machine.get("readiness_state_counts")
        ):
            _failure(
                failures,
                f"{check_id}_state_counts_match",
                "next_action_machine_summary_state_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("readiness_state_counts"))),
                    "computed": computed_next_action_machine.get("readiness_state_counts"),
                },
            )
        if dict(_mapping(summary.get("readiness_blocker_kind_counts"))) != dict(
            computed_next_action_machine.get("readiness_blocker_kind_counts")
        ):
            _failure(
                failures,
                f"{check_id}_blocker_counts_match",
                "next_action_machine_summary_blocker_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("readiness_blocker_kind_counts"))),
                    "computed": computed_next_action_machine.get("readiness_blocker_kind_counts"),
                },
            )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_unsafe_action_ids_match",
            reason="next_action_machine_summary_unsafe_action_ids_do_not_match_computed",
            left=_strings(summary.get("unsafe_action_ids")),
            right=_strings(computed_next_action_machine.get("unsafe_action_ids")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_missing_machine_fields_match",
            reason="next_action_machine_summary_missing_machine_fields_do_not_match_computed",
            left=_strings(summary.get("missing_machine_field_action_ids")),
            right=_strings(computed_next_action_machine.get("missing_machine_field_action_ids")),
        )
    for check_id, summary in {
        "readiness_next_action_contract_summary": readiness_next_action_contract,
        "terminal_next_action_contract_summary": terminal_next_action_contract,
    }.items():
        if not summary:
            _failure(failures, check_id, "next_action_contract_summary_missing")
            continue
        if str(summary.get("expected_roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "next_action_contract_summary_roadmap_missing_or_wrong",
                value=summary.get("expected_roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("contract_ok"))
        ):
            _failure(failures, check_id, "next_action_contract_summary_allows_release_or_auto_start")
        if check_id.startswith("terminal_") and not bool(summary.get("fail_closed")):
            _failure(failures, check_id, "next_action_contract_summary_not_fail_closed")
        for field in [
            "action_count",
            "contract_complete_action_count",
            "missing_contract_action_count",
            "release_or_auto_start_unsafe_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"next_action_contract_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_next_action_contract.get(field)),
            )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_missing_contract_actions_match",
            reason="next_action_contract_summary_missing_actions_do_not_match_computed",
            left=_strings(summary.get("missing_contract_action_ids")),
            right=_strings(computed_next_action_contract.get("missing_contract_action_ids")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_unsafe_actions_match",
            reason="next_action_contract_summary_unsafe_actions_do_not_match_computed",
            left=_strings(summary.get("release_or_auto_start_unsafe_action_ids")),
            right=_strings(computed_next_action_contract.get("release_or_auto_start_unsafe_action_ids")),
        )
        computed_missing = {
            str(row.get("id") or ""): set(_strings(row.get("missing_keys")))
            for row in _list(computed_next_action_contract.get("missing_contract_fields_by_action"))
        }
        summary_missing = {
            str(_mapping(row).get("id") or ""): set(_strings(_mapping(row).get("missing_keys")))
            for row in _list(summary.get("missing_contract_fields_by_action"))
        }
        if summary_missing != computed_missing:
            _failure(
                failures,
                f"{check_id}_missing_contract_fields_match",
                "next_action_contract_summary_missing_fields_do_not_match_computed",
                value={
                    "summary": {
                        key: sorted(value) for key, value in sorted(summary_missing.items())
                    },
                    "computed": {
                        key: sorted(value) for key, value in sorted(computed_missing.items())
                    },
                },
            )
    for check_id, summary in {
        "readiness_manual_review_queue_summary": readiness_manual_review_queue,
        "readiness_evidence_manual_review_queue_summary": readiness_evidence_manual_review_queue,
        "terminal_manual_review_queue_summary": terminal_manual_review_queue,
    }.items():
        if not summary:
            _failure(failures, check_id, "manual_review_queue_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "manual_review_queue_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, "manual_review_queue_summary_allows_release_or_auto_start")
        for field in [
            "manual_review_ready_count",
            "closed_blocked_or_regression_count",
            "closed_diagnostic_or_promotion_count",
            "review_only_action_count",
            "followup_gpu_action_count",
            "current_gpu_heavy_action_count",
            "unsafe_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"manual_review_queue_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_manual_review_queue.get(field)),
            )
        for field in [
            "closed_blocked_or_regression_action_ids",
            "closed_diagnostic_or_promotion_action_ids",
            "review_only_action_ids",
            "followup_gpu_action_ids",
            "current_gpu_heavy_action_ids",
            "unsafe_action_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"manual_review_queue_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_manual_review_queue.get(field)),
            )
        if dict(_mapping(summary.get("review_outcome_counts"))) != dict(
            _mapping(computed_manual_review_queue.get("review_outcome_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_review_outcome_counts_match",
                "manual_review_queue_summary_review_outcome_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("review_outcome_counts"))),
                    "computed": dict(_mapping(computed_manual_review_queue.get("review_outcome_counts"))),
                },
            )
        summary_outcomes = {
            str(kind): set(_strings(ids))
            for kind, ids in _mapping(summary.get("review_outcome_action_ids")).items()
        }
        computed_outcomes = {
            str(kind): set(_strings(ids))
            for kind, ids in _mapping(computed_manual_review_queue.get("review_outcome_action_ids")).items()
        }
        if summary_outcomes != computed_outcomes:
            _failure(
                failures,
                f"{check_id}_review_outcome_action_ids_match",
                "manual_review_queue_summary_review_outcome_action_ids_do_not_match_computed",
                value={
                    "summary": {key: sorted(value) for key, value in sorted(summary_outcomes.items())},
                    "computed": {key: sorted(value) for key, value in sorted(computed_outcomes.items())},
                },
            )
    for check_id, summary in {
        "readiness_protected_followup_gpu_queue_summary": readiness_followup_queue,
        "terminal_protected_followup_gpu_queue_summary": terminal_followup_queue,
    }.items():
        if not summary:
            _failure(failures, check_id, "protected_followup_gpu_queue_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "protected_followup_gpu_queue_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, "protected_followup_gpu_queue_summary_allows_release_or_auto_start")
        if str(summary.get("execution_policy") or "") != "manual_protected_followup_only":
            _failure(
                failures,
                check_id,
                "protected_followup_gpu_queue_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "followup_gpu_required_action_count",
            "current_action_gpu_count",
            "current_action_manual_start_count",
            "followup_manual_start_required_count",
            "requires_external_input_count",
            "unsafe_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"protected_followup_gpu_queue_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_followup_queue.get(field)),
            )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_action_ids_match",
            reason="protected_followup_gpu_queue_summary_action_ids_do_not_match_computed",
            left=_strings(summary.get("followup_gpu_required_action_ids")),
            right=_strings(computed_followup_queue.get("followup_gpu_required_action_ids")),
        )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_unsafe_action_ids_match",
            reason="protected_followup_gpu_queue_summary_unsafe_action_ids_do_not_match_computed",
            left=_strings(summary.get("unsafe_action_ids")),
            right=_strings(computed_followup_queue.get("unsafe_action_ids")),
        )
        if dict(_mapping(summary.get("family_counts"))) != dict(
            _mapping(computed_followup_queue.get("family_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_family_counts_match",
                "protected_followup_gpu_queue_summary_family_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("family_counts"))),
                    "computed": dict(_mapping(computed_followup_queue.get("family_counts"))),
                },
            )
        if dict(_mapping(summary.get("readiness_state_counts"))) != dict(
            _mapping(computed_followup_queue.get("readiness_state_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_state_counts_match",
                "protected_followup_gpu_queue_summary_state_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("readiness_state_counts"))),
                    "computed": dict(_mapping(computed_followup_queue.get("readiness_state_counts"))),
                },
            )
    for check_id, summary in {
        "readiness_remaining_release_blocker_matrix_summary": readiness_blocker_matrix,
        "terminal_remaining_release_blocker_matrix_summary": terminal_blocker_matrix,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_release_blocker_matrix_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_release_blocker_matrix_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(failures, check_id, "remaining_release_blocker_matrix_summary_allows_release_or_auto_start")
        if str(summary.get("execution_policy") or "") != "manual_or_external_input_only":
            _failure(
                failures,
                check_id,
                "remaining_release_blocker_matrix_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "total_unclosed_action_count",
            "json_ready_action_count",
            "external_input_required_action_count",
            "manual_gpu_required_action_count",
            "protected_followup_gpu_required_action_count",
            "source_cache_blocked_action_count",
            "sd15_checkpoint_action_count",
            "duplicate_or_stale_source_axis_action_count",
            "unsafe_action_count",
            "family_count",
            "unsafe_family_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_release_blocker_matrix_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_blocker_matrix.get(field)),
            )
        for field in [
            "unclosed_action_ids",
            "release_hard_gate_ids",
            "external_input_required_action_ids",
            "missing_external_inputs",
            "manual_gpu_required_action_ids",
            "protected_followup_gpu_required_action_ids",
            "source_cache_blocked_action_ids",
            "sd15_checkpoint_action_ids",
            "duplicate_or_stale_source_axis_action_ids",
            "next_unlock_inputs",
            "unsafe_action_ids",
            "family_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_release_blocker_matrix_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_blocker_matrix.get(field)),
            )
        if dict(_mapping(summary.get("readiness_state_counts"))) != dict(
            _mapping(computed_blocker_matrix.get("readiness_state_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_state_counts_match",
                "remaining_release_blocker_matrix_summary_state_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("readiness_state_counts"))),
                    "computed": dict(_mapping(computed_blocker_matrix.get("readiness_state_counts"))),
                },
            )
        if dict(_mapping(summary.get("blocked_by_kind_counts"))) != dict(
            _mapping(computed_blocker_matrix.get("blocked_by_kind_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_blocker_counts_match",
                "remaining_release_blocker_matrix_summary_blocker_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("blocked_by_kind_counts"))),
                    "computed": dict(_mapping(computed_blocker_matrix.get("blocked_by_kind_counts"))),
                },
            )
        for field in [
            "family_action_counts",
            "family_external_input_required_counts",
            "family_manual_gpu_required_counts",
            "family_protected_followup_gpu_required_counts",
            "family_source_cache_blocked_counts",
            "family_unsafe_action_counts",
        ]:
            if dict(_mapping(summary.get(field))) != dict(_mapping(computed_blocker_matrix.get(field))):
                _failure(
                    failures,
                    f"{check_id}_{field}_match",
                    f"remaining_release_blocker_matrix_summary_{field}_does_not_match_computed",
                    value={
                        "summary": dict(_mapping(summary.get(field))),
                        "computed": dict(_mapping(computed_blocker_matrix.get(field))),
                    },
                )
    for check_id, summary in {
        "readiness_remaining_blocker_resolution_handoff_summary": readiness_blocker_handoff,
        "terminal_remaining_blocker_resolution_handoff_summary": terminal_blocker_handoff,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_blocker_resolution_handoff_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_blocker_resolution_handoff_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or bool(summary.get("release_claim_after_resolution_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("resolution_contract_ok"))
        ):
            _failure(
                failures,
                check_id,
                "remaining_blocker_resolution_handoff_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "manual_or_external_input_handoff_only":
            _failure(
                failures,
                check_id,
                "remaining_blocker_resolution_handoff_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "row_count",
            "external_input_row_count",
            "current_gpu_row_count",
            "protected_followup_gpu_row_count",
            "resolution_contract_bad_count",
            "json_only_resolution_available_count",
            "external_input_required_count",
            "manual_gpu_required_count",
            "protected_runner_required_count",
            "unsafe_row_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_blocker_resolution_handoff_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_blocker_handoff.get(field)),
            )
        for field in [
            "row_ids",
            "next_unlock_input_ids",
            "required_refresh_command_ids",
            "resolution_contract_bad_ids",
            "unsafe_row_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_blocker_resolution_handoff_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_blocker_handoff.get(field)),
            )
        if dict(_mapping(summary.get("blocker_bucket_counts"))) != dict(
            _mapping(computed_blocker_handoff.get("blocker_bucket_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_bucket_counts_match",
                "remaining_blocker_resolution_handoff_summary_bucket_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("blocker_bucket_counts"))),
                    "computed": dict(_mapping(computed_blocker_handoff.get("blocker_bucket_counts"))),
                },
            )
        if dict(_mapping(summary.get("resolution_bucket_counts"))) != dict(
            _mapping(computed_blocker_handoff.get("resolution_bucket_counts"))
        ):
            _failure(
                failures,
                f"{check_id}_resolution_bucket_counts_match",
                "remaining_blocker_resolution_handoff_summary_resolution_bucket_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("resolution_bucket_counts"))),
                    "computed": dict(_mapping(computed_blocker_handoff.get("resolution_bucket_counts"))),
                },
            )
        computed_rows = {
            str(row.get("id") or ""): _mapping(row.get("resolution_contract"))
            for row in _list(computed_blocker_handoff.get("rows"))
        }
        mismatched_contract_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            row_id = str(row.get("id") or "")
            observed = _mapping(row.get("resolution_contract"))
            expected = computed_rows.get(row_id, {})
            if not expected:
                mismatched_contract_ids.append(row_id)
                continue
            scalar_fields = [
                "resolution_kind",
                "can_resolve_json_only_now",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_protected_runner",
                "terminal_guard_required",
                "release_claim_after_resolution_allowed",
                "safe_to_auto_start_after_resolution",
                "not_release_evidence",
            ]
            list_fields = [
                "required_input_ids",
                "missing_input_ids",
                "post_unlock_refresh_command_ids",
                "post_unlock_required_artifact_ids",
            ]
            scalar_mismatch = any(observed.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(observed.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_contract_ids.append(row_id)
        if mismatched_contract_ids:
            _failure(
                failures,
                f"{check_id}_resolution_contract_rows_match",
                "remaining_blocker_resolution_handoff_summary_row_contracts_do_not_match_computed",
                value=_unique(mismatched_contract_ids),
            )
    for check_id, summary in {
        "readiness_remaining_action_dependency_graph_summary": readiness_action_dependency_graph,
        "terminal_remaining_action_dependency_graph_summary": terminal_action_dependency_graph,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_action_dependency_graph_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_action_dependency_graph_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or bool(summary.get("ready_for_release_claim"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(
                failures,
                check_id,
                "remaining_action_dependency_graph_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "dependency_graph_only_manual_or_external_input":
            _failure(
                failures,
                check_id,
                "remaining_action_dependency_graph_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        if not bool(summary.get("refresh_sequence_terminal_guard_ok")):
            _failure(
                failures,
                check_id,
                "remaining_action_dependency_graph_summary_refresh_sequence_missing_terminal_guard",
            )
        for field in [
            "action_node_count",
            "dependency_node_count",
            "edge_count",
            "unsafe_action_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_action_dependency_graph_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_action_dependency_graph.get(field)),
            )
        for field in [
            "action_node_ids",
            "dependency_node_ids",
            "missing_external_inputs",
            "release_hard_gate_ids",
            "required_refresh_command_ids",
            "unsafe_action_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_action_dependency_graph_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_action_dependency_graph.get(field)),
            )
        for field in [
            "action_state_counts",
            "blocker_kind_counts",
            "dependency_kind_counts",
        ]:
            if dict(_mapping(summary.get(field))) != dict(
                _mapping(computed_action_dependency_graph.get(field))
            ):
                _failure(
                    failures,
                    f"{check_id}_{field}_match",
                    f"remaining_action_dependency_graph_summary_{field}_does_not_match_computed",
                    value={
                        "summary": dict(_mapping(summary.get(field))),
                        "computed": dict(_mapping(computed_action_dependency_graph.get(field))),
                    },
                )
        computed_rows = {
            str(row.get("action_id") or ""): row
            for row in _list(computed_action_dependency_graph.get("rows"))
        }
        mismatched_row_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            row_id = str(row.get("action_id") or "")
            expected = _mapping(computed_rows.get(row_id))
            if not expected:
                mismatched_row_ids.append(row_id)
                continue
            scalar_fields = [
                "readiness_state",
                "readiness_blocker_kind",
                "requires_external_input",
                "requires_current_gpu",
                "followup_requires_gpu_heavy_run",
                "safe_to_auto_start",
                "release_claim_allowed_after_success",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = ["dependency_ids", "dependency_kinds"]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_row_ids.append(row_id)
        if mismatched_row_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "remaining_action_dependency_graph_summary_rows_do_not_match_computed",
                value=_unique(mismatched_row_ids),
            )
    for check_id, summary in {
        "readiness_remaining_action_unblock_sequence_summary": readiness_action_unblock_sequence,
        "terminal_remaining_action_unblock_sequence_summary": terminal_action_unblock_sequence,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_action_unblock_sequence_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_action_unblock_sequence_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or bool(summary.get("ready_for_release_claim"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(
                failures,
                check_id,
                "remaining_action_unblock_sequence_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "ordered_handoff_only_manual_or_external_input":
            _failure(
                failures,
                check_id,
                "remaining_action_unblock_sequence_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        if not bool(summary.get("refresh_sequence_terminal_guard_ok")) or not bool(
            summary.get("terminal_guard_required")
        ):
            _failure(
                failures,
                check_id,
                "remaining_action_unblock_sequence_summary_refresh_sequence_missing_terminal_guard",
            )
        for field in [
            "stage_count",
            "manual_gpu_stage_count",
            "external_input_stage_count",
            "protected_runner_stage_count",
            "unsafe_stage_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_action_unblock_sequence_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_action_unblock_sequence.get(field)),
            )
        for field in [
            "stage_ids",
            "next_required_input_ids",
            "release_hard_gate_ids",
            "required_refresh_command_ids",
            "unsafe_stage_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_action_unblock_sequence_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_action_unblock_sequence.get(field)),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_current_stage_id_match",
            reason="remaining_action_unblock_sequence_summary_current_stage_id_does_not_match_computed",
            left=str(summary.get("current_stage_id") or ""),
            right=str(computed_action_unblock_sequence.get("current_stage_id") or ""),
        )
        computed_rows = {
            str(row.get("stage_id") or ""): row
            for row in _list(computed_action_unblock_sequence.get("rows"))
        }
        mismatched_stage_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            stage_id = str(row.get("stage_id") or "")
            expected = _mapping(computed_rows.get(stage_id))
            if not expected:
                mismatched_stage_ids.append(stage_id)
                continue
            scalar_fields = [
                "stage_kind",
                "status",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_protected_runner",
                "terminal_guard_required_after_stage",
                "safe_to_auto_start",
                "release_claim_allowed_after_stage",
                "not_release_evidence",
            ]
            list_fields = ["required_input_ids", "related_action_ids", "required_refresh_command_ids"]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_stage_ids.append(stage_id)
        if mismatched_stage_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "remaining_action_unblock_sequence_summary_rows_do_not_match_computed",
                value=_unique(mismatched_stage_ids),
            )
    for check_id, summary in {
        "readiness_remaining_blocker_artifact_presence_summary": readiness_blocker_presence,
        "terminal_remaining_blocker_artifact_presence_summary": terminal_blocker_presence,
    }.items():
        if not summary:
            _failure(failures, check_id, "remaining_blocker_artifact_presence_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "remaining_blocker_artifact_presence_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
        ):
            _failure(
                failures,
                check_id,
                "remaining_blocker_artifact_presence_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "read_only_artifact_presence_audit":
            _failure(
                failures,
                check_id,
                "remaining_blocker_artifact_presence_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "row_count",
            "expected_output_action_count",
            "expected_output_missing_action_count",
            "evidence_path_action_count",
            "evidence_path_missing_action_count",
            "unsafe_row_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_blocker_artifact_presence_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_blocker_presence.get(field)),
            )
        for field in [
            "row_ids",
            "expected_output_missing_action_ids",
            "evidence_path_missing_action_ids",
            "unsafe_row_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"remaining_blocker_artifact_presence_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_blocker_presence.get(field)),
            )
    for check_id, summary in {
        "readiness_release_claim_exit_criteria_summary": readiness_release_exit,
        "terminal_release_claim_exit_criteria_summary": terminal_release_exit,
    }.items():
        if not summary:
            _failure(failures, check_id, "release_claim_exit_criteria_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_claim_exit_criteria_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_gate_count")) != 0
            or _safe_int(summary.get("json_only_exit_available_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_claim_exit_criteria_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_claim_exit_criteria_only":
            _failure(
                failures,
                check_id,
                "release_claim_exit_criteria_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "release_hard_gate_count",
            "gate_row_count",
            "json_only_exit_available_count",
            "manual_gpu_required_gate_count",
            "protected_runner_required_gate_count",
            "missing_declared_output_gate_count",
            "unsafe_gate_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_claim_exit_criteria_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_exit.get(field)),
            )
        for field in ["release_hard_gate_ids", "unsafe_gate_ids"]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_claim_exit_criteria_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_exit.get(field)),
            )
        computed_rows = {
            str(row.get("gate_id") or ""): row
            for row in (_mapping(item) for item in _list(computed_release_exit.get("rows")))
        }
        mismatched_gate_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            gate_id = str(row.get("gate_id") or "")
            expected = _mapping(computed_rows.get(gate_id))
            if not expected:
                mismatched_gate_ids.append(gate_id)
                continue
            scalar_fields = [
                "gate_status",
                "manual_gpu_required",
                "protected_runner_required",
                "json_only_exit_available",
                "terminal_guard_required",
                "release_claim_allowed_after_exit",
                "safe_to_auto_start",
                "not_release_evidence",
                "related_external_input_action_count",
                "related_manual_gpu_action_count",
                "related_protected_followup_action_count",
            ]
            list_fields = [
                "related_action_ids",
                "related_family_ids",
                "required_input_ids",
                "required_output_ids",
                "missing_declared_output_action_ids",
            ]
            mapping_fields = [
                "related_family_counts",
                "related_readiness_state_counts",
                "related_blocker_kind_counts",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            mapping_mismatch = any(
                dict(_mapping(row.get(field))) != dict(_mapping(expected.get(field)))
                for field in mapping_fields
            )
            if scalar_mismatch or list_mismatch or mapping_mismatch:
                mismatched_gate_ids.append(gate_id)
        if mismatched_gate_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_claim_exit_criteria_summary_rows_do_not_match_computed",
                value=_unique(mismatched_gate_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_dependency_summary": readiness_release_gate_input_dependency,
        "terminal_release_gate_input_dependency_summary": terminal_release_gate_input_dependency,
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_dependency_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_dependency_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_input_count")) != 0
            or _safe_int(summary.get("json_only_resolution_available_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_dependency_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_dependency_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_dependency_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "release_hard_gate_count",
            "dependency_row_count",
            "missing_input_count",
            "external_input_dependency_count",
            "manual_gpu_dependency_count",
            "source_cache_refresh_dependency_count",
            "json_only_resolution_available_count",
            "unsafe_input_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_dependency_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_dependency.get(field)),
            )
        for field in [
            "release_hard_gate_ids",
            "required_input_ids",
            "missing_input_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_dependency_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_dependency.get(field)),
            )
        computed_input_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_dependency.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_input_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "dependency_status",
                "missing",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "json_only_resolution_available",
                "terminal_guard_required_after_input",
                "release_claim_allowed_after_input",
                "safe_to_auto_start",
                "not_release_evidence",
            ]
            list_fields = [
                "related_gate_ids",
                "related_action_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_dependency_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_plan_summary": (
            readiness_release_gate_post_input_refresh_plan
        ),
        "terminal_release_gate_post_input_refresh_plan_summary": (
            terminal_release_gate_post_input_refresh_plan
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_plan_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_plan_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_plan_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_plan_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_json_refresh_plan_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_plan_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "plan_row_count",
            "blocked_input_count",
            "external_input_plan_count",
            "manual_gpu_evidence_plan_count",
            "source_cache_refresh_plan_count",
            "unsafe_plan_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_plan_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_plan.get(field)),
            )
        for field in [
            "input_ids",
            "blocked_input_ids",
            "required_refresh_command_ids",
            "terminal_guard_command_ids",
            "post_refresh_required_artifact_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_plan_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_plan.get(field)),
            )
        computed_plan_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_post_input_refresh_plan.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_plan_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "plan_status",
                "input_missing",
                "external_input_required_before_refresh",
                "manual_gpu_evidence_required_before_refresh",
                "source_cache_refresh_input",
                "terminal_guard_required_after_refresh",
                "safe_to_auto_start_refresh",
                "release_claim_allowed_after_refresh",
                "not_release_evidence",
            ]
            list_fields = [
                "related_gate_ids",
                "related_action_ids",
                "affected_family_ids",
                "required_refresh_command_ids",
                "terminal_guard_command_ids",
                "post_refresh_required_artifact_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_plan_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_detection_source_summary": (
            readiness_release_gate_input_detection_source
        ),
        "terminal_release_gate_input_detection_source_summary": (
            terminal_release_gate_input_detection_source
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_detection_source_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_detection_source_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_detector_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_detection_source_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_detection_source_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_detection_source_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "detection_row_count",
            "missing_or_unverified_input_count",
            "detected_input_count",
            "external_input_detector_count",
            "manual_gpu_detector_count",
            "source_cache_refresh_detector_count",
            "unsafe_detector_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_detection_source_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_detection_source.get(field)),
            )
        for field in ["input_ids", "missing_or_unverified_input_ids", "unsafe_input_ids"]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_detection_source_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_detection_source.get(field)),
            )
        computed_detection_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_detection_source.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_detection_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "detection_status",
                "input_missing",
                "detected",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "terminal_guard_required_after_detection",
                "safe_to_auto_start",
                "release_claim_allowed_after_detection",
                "not_release_evidence",
            ]
            list_fields = [
                "detector_artifact_ids",
                "required_refresh_command_ids",
                "related_gate_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_detection_source_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_acceptance_criteria_summary": (
            readiness_release_gate_input_acceptance_criteria
        ),
        "terminal_release_gate_input_acceptance_criteria_summary": (
            terminal_release_gate_input_acceptance_criteria
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_acceptance_criteria_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_acceptance_criteria_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_acceptance_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_acceptance_criteria_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_acceptance_criteria_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_acceptance_criteria_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "acceptance_row_count",
            "accepted_input_count",
            "unsatisfied_input_count",
            "external_input_acceptance_count",
            "manual_gpu_acceptance_count",
            "source_cache_refresh_acceptance_count",
            "unsafe_acceptance_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_acceptance_criteria_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_acceptance_criteria.get(field)),
            )
        for field in ["input_ids", "accepted_input_ids", "unsatisfied_input_ids", "unsafe_input_ids"]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_acceptance_criteria_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_acceptance_criteria.get(field)),
            )
        computed_acceptance_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_acceptance_criteria.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_acceptance_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "acceptance_status",
                "input_missing",
                "detected",
                "accepted",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "terminal_guard_required_after_acceptance",
                "release_claim_allowed_after_acceptance",
                "safe_to_auto_start",
                "not_release_evidence",
            ]
            list_fields = [
                "acceptance_criteria_ids",
                "required_evidence_artifact_ids",
                "detector_artifact_ids",
                "required_refresh_command_ids",
                "related_gate_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_acceptance_criteria_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_refresh_readiness_summary": (
            readiness_release_gate_input_refresh_readiness
        ),
        "terminal_release_gate_input_refresh_readiness_summary": (
            terminal_release_gate_input_refresh_readiness
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_refresh_readiness_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_readiness_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_refresh_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_readiness_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_refresh_readiness_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_readiness_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "refresh_row_count",
            "accepted_input_count",
            "refresh_ready_input_count",
            "blocked_refresh_input_count",
            "external_input_refresh_count",
            "manual_gpu_refresh_count",
            "source_cache_refresh_count",
            "unsafe_refresh_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_refresh_readiness_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_refresh_readiness.get(field)),
            )
        for field in [
            "input_ids",
            "accepted_input_ids",
            "refresh_ready_input_ids",
            "blocked_refresh_input_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_refresh_readiness_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_refresh_readiness.get(field)),
            )
        computed_refresh_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_refresh_readiness.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_refresh_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "refresh_readiness_status",
                "accepted",
                "input_missing",
                "detected",
                "refresh_ready",
                "blocked_refresh",
                "acceptance_status",
                "plan_status",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "terminal_guard_required_after_refresh",
                "safe_to_auto_start_refresh",
                "release_claim_allowed_after_refresh",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = [
                "acceptance_criteria_ids",
                "required_evidence_artifact_ids",
                "required_refresh_command_ids",
                "terminal_guard_command_ids",
                "post_refresh_required_artifact_ids",
                "related_gate_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_refresh_readiness_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_refresh_blocker_summary": (
            readiness_release_gate_input_refresh_blocker
        ),
        "terminal_release_gate_input_refresh_blocker_summary": (
            terminal_release_gate_input_refresh_blocker
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_refresh_blocker_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_blocker_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_blocker_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_blocker_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_refresh_blocker_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_refresh_blocker_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "blocker_row_count",
            "blocked_input_count",
            "refresh_ready_input_count",
            "missing_input_blocker_count",
            "undetected_input_blocker_count",
            "unaccepted_input_blocker_count",
            "external_input_blocker_count",
            "manual_gpu_blocker_count",
            "source_cache_refresh_blocker_count",
            "terminal_guard_required_count",
            "unsafe_blocker_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_refresh_blocker_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_refresh_blocker.get(field)),
            )
        for field in [
            "input_ids",
            "blocked_input_ids",
            "refresh_ready_input_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_refresh_blocker_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_refresh_blocker.get(field)),
            )
        computed_blocker_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_refresh_blocker.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_blocker_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "blocker_status",
                "blocked_refresh",
                "refresh_ready",
                "accepted",
                "detected",
                "input_missing",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "terminal_guard_required_after_refresh",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = [
                "blocked_reason_ids",
                "required_refresh_command_ids",
                "terminal_guard_command_ids",
                "related_gate_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_refresh_blocker_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_input_lifecycle_summary": (
            readiness_release_gate_input_lifecycle
        ),
        "terminal_release_gate_input_lifecycle_summary": (
            terminal_release_gate_input_lifecycle
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_input_lifecycle_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_input_lifecycle_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_input_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_input_lifecycle_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "release_gate_input_lifecycle_summary_only":
            _failure(
                failures,
                check_id,
                "release_gate_input_lifecycle_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_lifecycle_status_match",
            reason="release_gate_input_lifecycle_summary_lifecycle_status_does_not_match_computed",
            left=str(summary.get("lifecycle_status") or ""),
            right=str(computed_release_gate_input_lifecycle.get("lifecycle_status") or ""),
        )
        for field in [
            "input_count",
            "detected_input_count",
            "detected_unaccepted_input_count",
            "accepted_input_count",
            "accepted_pending_refresh_input_count",
            "refresh_ready_input_count",
            "blocked_input_count",
            "external_input_count",
            "manual_gpu_input_count",
            "source_cache_refresh_input_count",
            "unsafe_input_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_lifecycle_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_input_lifecycle.get(field)),
            )
        for field in [
            "input_ids",
            "detected_input_ids",
            "detected_unaccepted_input_ids",
            "accepted_input_ids",
            "accepted_pending_refresh_input_ids",
            "refresh_ready_input_ids",
            "blocked_input_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_input_lifecycle_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_input_lifecycle.get(field)),
            )
        if (
            dict(_mapping(summary.get("lifecycle_stage_counts")))
            != dict(_mapping(computed_release_gate_input_lifecycle.get("lifecycle_stage_counts")))
        ):
            _failure(
                failures,
                f"{check_id}_lifecycle_stage_counts_match",
                "release_gate_input_lifecycle_summary_lifecycle_stage_counts_do_not_match_computed",
                value={
                    "summary": dict(_mapping(summary.get("lifecycle_stage_counts"))),
                    "computed": dict(
                        _mapping(
                            computed_release_gate_input_lifecycle.get(
                                "lifecycle_stage_counts"
                            )
                        )
                    ),
                },
            )
        computed_lifecycle_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_input_lifecycle.get("rows"))
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_lifecycle_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "lifecycle_stage",
                "dependency_status",
                "detection_status",
                "acceptance_status",
                "refresh_readiness_status",
                "blocker_status",
                "missing",
                "detected",
                "accepted",
                "refresh_ready",
                "blocked_refresh",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = [
                "blocked_reason_ids",
                "required_refresh_command_ids",
                "terminal_guard_command_ids",
                "related_gate_ids",
                "affected_family_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_input_lifecycle_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_external_input_release_gate_alignment_summary": (
            readiness_external_input_release_gate_alignment
        ),
        "terminal_external_input_release_gate_alignment_summary": (
            terminal_external_input_release_gate_alignment
        ),
    }.items():
        if not summary:
            _failure(
                failures,
                check_id,
                "external_input_release_gate_alignment_summary_missing",
            )
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "external_input_release_gate_alignment_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("alignment_ok"))
            or _safe_int(summary.get("unsafe_alignment_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "external_input_release_gate_alignment_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "external_input_release_gate_alignment_only":
            _failure(
                failures,
                check_id,
                "external_input_release_gate_alignment_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in ["alignment_status"]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"external_input_release_gate_alignment_summary_{field}_does_not_match_computed",
                left=str(summary.get(field) or ""),
                right=str(computed_external_input_release_gate_alignment.get(field) or ""),
            )
        for field in ["alignment_ok"]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"external_input_release_gate_alignment_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_external_input_release_gate_alignment.get(field)),
            )
        for field in [
            "external_input_count",
            "release_gate_input_count",
            "external_release_gate_input_count",
            "manual_gpu_release_gate_input_count",
            "source_cache_refresh_release_gate_input_count",
            "non_external_release_gate_input_count",
            "external_missing_from_release_gate_count",
            "release_external_missing_from_transition_count",
            "blocked_input_count",
            "unsafe_alignment_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"external_input_release_gate_alignment_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_external_input_release_gate_alignment.get(field)),
            )
        for field in [
            "external_input_ids",
            "release_gate_input_ids",
            "non_external_release_gate_input_ids",
            "external_missing_from_release_gate_ids",
            "release_external_missing_from_transition_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"external_input_release_gate_alignment_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_external_input_release_gate_alignment.get(field)),
            )
        computed_alignment_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_external_input_release_gate_alignment.get("rows")
                )
            )
        }
        mismatched_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_alignment_rows.get(input_id))
            if not expected:
                mismatched_input_ids.append(input_id)
                continue
            scalar_fields = [
                "input_kind",
                "alignment_kind",
                "requires_external_input",
                "requires_manual_gpu",
                "requires_source_cache_refresh",
                "in_external_transition_table",
                "expected_in_external_transition_table",
                "external_input_missing",
                "release_gate_input_present",
                "lifecycle_stage",
                "transition_state",
                "missing",
                "detected",
                "accepted",
                "blocked_refresh",
                "handoff_step_id",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = [
                "related_gate_ids",
                "blocked_reason_ids",
                "replay_command_ids",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_input_ids.append(input_id)
        if mismatched_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "external_input_release_gate_alignment_summary_rows_do_not_match_computed",
                value=_unique(mismatched_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_command_surface_summary": (
            readiness_release_gate_post_input_refresh_command_surface
        ),
        "terminal_release_gate_post_input_refresh_command_surface_summary": (
            terminal_release_gate_post_input_refresh_command_surface
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_command_surface_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_surface_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_command_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_surface_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_command_surface_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_surface_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "command_row_count",
            "required_command_count",
            "json_refresh_command_count",
            "terminal_guard_command_count",
            "blocked_command_count",
            "ready_command_count",
            "blocked_input_count",
            "unsafe_command_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_surface_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_command_surface.get(field)),
            )
        for field in [
            "required_command_ids",
            "blocked_command_ids",
            "ready_command_ids",
            "blocked_input_ids",
            "unsafe_command_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_surface_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_command_surface.get(field)),
            )
        computed_command_rows = {
            str(row.get("command_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(computed_release_gate_post_input_refresh_command_surface.get("rows"))
            )
        }
        mismatched_command_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            command_id = str(row.get("command_id") or "")
            expected = _mapping(computed_command_rows.get(command_id))
            if not expected:
                mismatched_command_ids.append(command_id)
                continue
            scalar_fields = [
                "command_order",
                "command_kind",
                "command_status",
                "blocked_input_count",
                "refresh_ready_input_count",
                "terminal_guard_command",
                "required_after_input_acceptance",
                "blocked_until_input_acceptance",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = ["related_input_ids", "blocked_input_ids"]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_command_ids.append(command_id)
        if mismatched_command_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_command_surface_summary_rows_do_not_match_computed",
                value=_unique(mismatched_command_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_sequence_integrity_summary": (
            readiness_release_gate_post_input_refresh_sequence_integrity
        ),
        "terminal_release_gate_post_input_refresh_sequence_integrity_summary": (
            terminal_release_gate_post_input_refresh_sequence_integrity
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_sequence_integrity_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_sequence_integrity_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("sequence_ok"))
            or _safe_int(summary.get("unsafe_sequence_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_sequence_integrity_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_sequence_integrity_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_sequence_integrity_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "expected_command_count",
            "observed_command_count",
            "missing_command_count",
            "unexpected_command_count",
            "duplicate_command_count",
            "blocked_command_count",
            "ready_command_count",
            "unsafe_sequence_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_sequence_integrity_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_sequence_integrity.get(field)),
            )
        for field in [
            "sequence_ok",
            "order_matches_expected",
            "terminal_guard_tail_ok",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_sequence_integrity_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_sequence_integrity.get(field)),
            )
        for field in [
            "expected_command_ids",
            "observed_command_ids",
            "missing_command_ids",
            "unexpected_command_ids",
            "duplicate_command_ids",
            "terminal_guard_command_ids",
            "unsafe_command_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_sequence_integrity_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_sequence_integrity.get(field)),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_terminal_guard_dependency_summary": (
            readiness_release_gate_post_input_refresh_terminal_guard_dependency
        ),
        "terminal_release_gate_post_input_refresh_terminal_guard_dependency_summary": (
            terminal_release_gate_post_input_refresh_terminal_guard_dependency
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_terminal_guard_dependency_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_terminal_guard_dependency_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("dependency_ok"))
            or _safe_int(summary.get("unsafe_dependency_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_terminal_guard_dependency_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_terminal_guard_dependency_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_terminal_guard_dependency_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "terminal_guard_command_count",
            "expected_terminal_guard_command_count",
            "json_refresh_command_count",
            "blocked_command_count",
            "ready_command_count",
            "unsafe_dependency_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_terminal_guard_dependency_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_terminal_guard_dependency.get(field)),
            )
        for field in [
            "dependency_ok",
            "terminal_guard_required",
            "terminal_self_check_required",
            "release_guard_required",
            "terminal_guard_tail_ok",
            "all_json_refresh_commands_before_terminal_guard",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_terminal_guard_dependency_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_terminal_guard_dependency.get(field)),
            )
        for field in [
            "terminal_guard_command_ids",
            "expected_terminal_guard_command_ids",
            "unsafe_command_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_terminal_guard_dependency_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_terminal_guard_dependency.get(field)),
            )
        _compare_scalar_field(
            failures,
            check_id=f"{check_id}_terminal_guard_command_orders_match",
            reason="release_gate_post_input_refresh_terminal_guard_dependency_summary_terminal_guard_command_orders_do_not_match_computed",
            left=[_safe_int(item) for item in _list(summary.get("terminal_guard_command_orders"))],
            right=[
                _safe_int(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                        "terminal_guard_command_orders"
                    )
                )
            ],
        )
        computed_dependency_rows = {
            str(row.get("command_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_terminal_guard_dependency.get("rows")
                )
            )
        }
        mismatched_dependency_command_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            command_id = str(row.get("command_id") or "")
            expected = _mapping(computed_dependency_rows.get(command_id))
            if not expected:
                mismatched_dependency_command_ids.append(command_id)
                continue
            for field in [
                "dependency_order",
                "command_order",
                "guard_kind",
                "depends_on_json_refresh_sequence",
                "required_after_json_refresh",
                "blocked_until_input_acceptance",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]:
                if row.get(field) != expected.get(field):
                    mismatched_dependency_command_ids.append(command_id)
                    break
        if mismatched_dependency_command_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_terminal_guard_dependency_summary_rows_do_not_match_computed",
                value=_unique(mismatched_dependency_command_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_artifact_coverage_summary": (
            readiness_release_gate_post_input_refresh_artifact_coverage
        ),
        "terminal_release_gate_post_input_refresh_artifact_coverage_summary": (
            terminal_release_gate_post_input_refresh_artifact_coverage
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_artifact_coverage_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_artifact_coverage_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("coverage_ok"))
            or _safe_int(summary.get("unsafe_artifact_coverage_count")) != 0
            or _safe_int(summary.get("missing_coverage_input_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_artifact_coverage_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_artifact_coverage_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_artifact_coverage_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "required_artifact_count",
            "input_row_count",
            "covered_input_count",
            "missing_coverage_input_count",
            "unsafe_artifact_coverage_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_artifact_coverage_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_artifact_coverage.get(field)),
            )
        for field in [
            "coverage_ok",
            "readiness_artifact_required",
            "terminal_artifact_required",
            "release_guard_artifact_required",
            "terminal_guard_dependency_ok",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_artifact_coverage_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_artifact_coverage.get(field)),
            )
        for field in [
            "required_artifact_ids",
            "covered_input_ids",
            "missing_coverage_input_ids",
            "unsafe_input_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_artifact_coverage_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_artifact_coverage.get(field)),
            )
        computed_artifact_rows = {
            str(row.get("input_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_artifact_coverage.get("rows")
                )
            )
        }
        mismatched_artifact_input_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            input_id = str(row.get("input_id") or "")
            expected = _mapping(computed_artifact_rows.get(input_id))
            if not expected:
                mismatched_artifact_input_ids.append(input_id)
                continue
            scalar_fields = [
                "artifact_count",
                "readiness_artifact_required",
                "terminal_artifact_required",
                "release_guard_artifact_required",
                "covered",
                "blocked_until_input_acceptance",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            list_fields = ["artifact_ids", "missing_artifact_ids"]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = any(
                set(_strings(row.get(field))) != set(_strings(expected.get(field)))
                for field in list_fields
            )
            if scalar_mismatch or list_mismatch:
                mismatched_artifact_input_ids.append(input_id)
        if mismatched_artifact_input_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_artifact_coverage_summary_rows_do_not_match_computed",
                value=_unique(mismatched_artifact_input_ids),
            )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_command_artifact_link_summary": (
            readiness_release_gate_post_input_refresh_command_artifact_link
        ),
        "terminal_release_gate_post_input_refresh_command_artifact_link_summary": (
            terminal_release_gate_post_input_refresh_command_artifact_link
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_command_artifact_link_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_artifact_link_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("link_ok"))
            or not bool(summary.get("artifact_coverage_ok"))
            or not bool(summary.get("blocked_until_input_acceptance"))
            or _safe_int(summary.get("unsafe_link_count")) != 0
            or _safe_int(summary.get("missing_link_artifact_count")) != 0
            or _safe_int(summary.get("extra_link_artifact_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_artifact_link_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_command_artifact_link_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_command_artifact_link_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "command_row_count",
            "required_artifact_count",
            "linked_artifact_count",
            "missing_link_artifact_count",
            "extra_link_artifact_count",
            "command_artifact_link_count",
            "unsafe_link_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_artifact_link_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_command_artifact_link.get(field)),
            )
        for field in [
            "link_ok",
            "artifact_coverage_ok",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_artifact_link_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_command_artifact_link.get(field)),
            )
        for field in [
            "required_artifact_ids",
            "linked_artifact_ids",
            "missing_link_artifact_ids",
            "extra_link_artifact_ids",
            "command_ids_with_artifacts",
            "command_ids_without_artifacts",
            "unsafe_command_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_artifact_link_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_command_artifact_link.get(field)),
            )
        for field in [
            "readiness_command_id",
            "terminal_command_id",
            "release_guard_command_id",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_command_artifact_link_summary_{field}_does_not_match_computed",
                left=str(summary.get(field) or ""),
                right=str(computed_release_gate_post_input_refresh_command_artifact_link.get(field) or ""),
            )
        computed_link_rows = {
            str(row.get("command_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_command_artifact_link.get("rows")
                )
            )
        }
        mismatched_link_command_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            command_id = str(row.get("command_id") or "")
            expected = _mapping(computed_link_rows.get(command_id))
            if not expected:
                mismatched_link_command_ids.append(command_id)
                continue
            scalar_fields = [
                "command_order",
                "command_kind",
                "output_artifact_count",
                "produces_required_post_refresh_artifact",
                "blocked_until_input_acceptance",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]
            scalar_mismatch = any(row.get(field) != expected.get(field) for field in scalar_fields)
            list_mismatch = set(_strings(row.get("output_artifact_ids"))) != set(
                _strings(expected.get("output_artifact_ids"))
            )
            if scalar_mismatch or list_mismatch:
                mismatched_link_command_ids.append(command_id)
        if mismatched_link_command_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_command_artifact_link_summary_rows_do_not_match_computed",
                value=_unique(mismatched_link_command_ids),
            )
    terminal_roadmap_lineage = _mapping(terminal.get("roadmap_lineage_audit"))
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_guard_consumption_summary": (
            readiness_release_gate_post_input_refresh_guard_consumption
        ),
        "terminal_release_gate_post_input_refresh_guard_consumption_summary": (
            terminal_release_gate_post_input_refresh_guard_consumption
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_guard_consumption_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_consumption_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("consumption_ok"))
            or not bool(summary.get("input_artifacts_consumed"))
            or not bool(summary.get("guard_artifact_produced"))
            or not bool(summary.get("command_artifact_link_ok"))
            or not bool(summary.get("artifact_coverage_ok"))
            or not bool(summary.get("terminal_guard_dependency_ok"))
            or not bool(summary.get("terminal_lineage_required"))
            or not bool(summary.get("blocked_until_input_acceptance"))
            or _safe_int(summary.get("missing_consumed_summary_count")) != 0
            or _safe_int(summary.get("unsafe_consumed_summary_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_consumption_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_guard_consumption_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_consumption_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "required_input_artifact_count",
            "required_consumed_summary_count",
            "present_consumed_summary_count",
            "missing_consumed_summary_count",
            "unsafe_consumed_summary_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_consumption_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_guard_consumption.get(field)),
            )
        for field in [
            "consumption_ok",
            "input_artifacts_consumed",
            "guard_artifact_produced",
            "terminal_lineage_required",
            "command_artifact_link_ok",
            "artifact_coverage_ok",
            "terminal_guard_dependency_ok",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_consumption_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_guard_consumption.get(field)),
            )
        for field in [
            "required_input_artifact_ids",
            "required_consumed_summary_ids",
            "missing_consumed_summary_ids",
            "unsafe_consumed_summary_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_consumption_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_guard_consumption.get(field)),
            )
        for field in [
            "guard_command_id",
            "produced_guard_artifact_id",
            "terminal_lineage_summary_id",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_consumption_summary_{field}_does_not_match_computed",
                left=str(summary.get(field) or ""),
                right=str(computed_release_gate_post_input_refresh_guard_consumption.get(field) or ""),
            )
        computed_consumption_rows = {
            str(row.get("summary_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_guard_consumption.get("rows")
                )
            )
        }
        mismatched_consumption_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            summary_id = str(row.get("summary_id") or "")
            expected = _mapping(computed_consumption_rows.get(summary_id))
            if not expected:
                mismatched_consumption_ids.append(summary_id)
                continue
            for field in [
                "consumption_stage",
                "required_for_guard",
                "present",
                "fail_closed",
                "terminal_only",
                "safe_to_auto_start",
                "release_claim_allowed",
                "not_release_evidence",
                "unsafe",
            ]:
                if row.get(field) != expected.get(field):
                    mismatched_consumption_ids.append(summary_id)
                    break
            if summary_id == "external_input_json_refresh_runner_manifest_summary":
                for field in [
                    "manifest_ok",
                    "runner_ready",
                    "execution_ok",
                    "row_execution_consistent",
                ]:
                    if bool(row.get(field)) != bool(expected.get(field)):
                        mismatched_consumption_ids.append(summary_id)
                        break
                else:
                    for field in [
                        "expected_command_count",
                        "row_count",
                        "executed_row_count",
                        "failed_row_count",
                        "missing_output_row_count",
                        "row_forbidden_heavy_flag_count",
                        "unsafe_row_count",
                    ]:
                        if _safe_int(row.get(field)) != _safe_int(expected.get(field)):
                            mismatched_consumption_ids.append(summary_id)
                            break
        if mismatched_consumption_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_guard_consumption_summary_rows_do_not_match_computed",
                value=_unique(mismatched_consumption_ids),
            )
    if (
        not terminal_roadmap_lineage
        or not bool(terminal_roadmap_lineage.get("lineage_ok"))
        or str(
            terminal_roadmap_lineage.get("roadmap")
            or terminal_roadmap_lineage.get("expected_roadmap")
            or ""
        )
        != ROADMAP
    ):
        _failure(
            failures,
            "terminal_release_gate_post_input_refresh_guard_consumption_summary_roadmap_lineage_audit",
            "release_gate_post_input_refresh_guard_consumption_summary_terminal_roadmap_lineage_missing_or_drifted",
            value=terminal_roadmap_lineage,
        )
    for check_id, summary in {
        "readiness_release_gate_post_input_refresh_guard_report_acceptance_summary": (
            readiness_release_gate_post_input_refresh_guard_report_acceptance
        ),
        "terminal_release_gate_post_input_refresh_guard_report_acceptance_summary": (
            terminal_release_gate_post_input_refresh_guard_report_acceptance
        ),
    }.items():
        if not summary:
            _failure(failures, check_id, "release_gate_post_input_refresh_guard_report_acceptance_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_report_acceptance_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or not bool(summary.get("acceptance_ok"))
            or str(summary.get("guard_command_id") or "") != "run_gpu_bubble_release_readiness_guard"
            or str(summary.get("guard_report_artifact_id") or "")
            != "gpu_bubble_release_readiness_guard_report"
            or str(summary.get("expected_report_status") or "")
            != "guard_passed_blocked_release_claim"
            or not bool(summary.get("expected_ok"))
            or _safe_int(summary.get("expected_failure_count")) != 0
            or not bool(summary.get("guard_consumption_ok"))
            or not bool(summary.get("input_artifacts_consumed"))
            or not bool(summary.get("guard_artifact_produced"))
            or not bool(summary.get("blocked_until_input_acceptance"))
            or _safe_int(summary.get("unsafe_acceptance_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_report_acceptance_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "post_input_refresh_guard_report_acceptance_only":
            _failure(
                failures,
                check_id,
                "release_gate_post_input_refresh_guard_report_acceptance_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "expected_failure_count",
            "required_guard_report_field_count",
            "acceptance_row_count",
            "unsafe_acceptance_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_report_acceptance_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_release_gate_post_input_refresh_guard_report_acceptance.get(field)),
            )
        for field in [
            "acceptance_ok",
            "expected_ok",
            "requires_input_artifact_summary",
            "requires_not_release_evidence",
            "requires_safe_to_auto_start_false",
            "requires_release_claim_allowed_false",
            "requires_blocked_actions",
            "guard_consumption_ok",
            "input_artifacts_consumed",
            "guard_artifact_produced",
            "blocked_until_input_acceptance",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_report_acceptance_summary_{field}_does_not_match_computed",
                left=bool(summary.get(field)),
                right=bool(computed_release_gate_post_input_refresh_guard_report_acceptance.get(field)),
            )
        for field in [
            "required_guard_report_fields",
            "unsafe_acceptance_ids",
        ]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_report_acceptance_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_release_gate_post_input_refresh_guard_report_acceptance.get(field)),
            )
        for field in [
            "artifact_role",
            "acceptance_status",
            "guard_command_id",
            "guard_report_artifact_id",
            "expected_report_status",
            "execution_policy",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"release_gate_post_input_refresh_guard_report_acceptance_summary_{field}_does_not_match_computed",
                left=str(summary.get(field) or ""),
                right=str(computed_release_gate_post_input_refresh_guard_report_acceptance.get(field) or ""),
            )
        computed_acceptance_rows = {
            str(row.get("acceptance_id") or ""): row
            for row in (
                _mapping(item)
                for item in _list(
                    computed_release_gate_post_input_refresh_guard_report_acceptance.get("rows")
                )
            )
        }
        mismatched_acceptance_ids: list[str] = []
        for raw_row in _list(summary.get("rows")):
            row = _mapping(raw_row)
            acceptance_id = str(row.get("acceptance_id") or "")
            expected = _mapping(computed_acceptance_rows.get(acceptance_id))
            if not expected:
                mismatched_acceptance_ids.append(acceptance_id)
                continue
            scalar_mismatch = any(
                row.get(field) != expected.get(field)
                for field in [
                    "required",
                    "expected_value_summary",
                    "present",
                    "fail_closed",
                    "safe_to_auto_start",
                    "release_claim_allowed",
                    "not_release_evidence",
                    "unsafe",
                ]
            )
            list_mismatch = set(_strings(row.get("required_field_ids"))) != set(
                _strings(expected.get("required_field_ids"))
            )
            if scalar_mismatch or list_mismatch:
                mismatched_acceptance_ids.append(acceptance_id)
        if mismatched_acceptance_ids:
            _failure(
                failures,
                f"{check_id}_rows_match",
                "release_gate_post_input_refresh_guard_report_acceptance_summary_rows_do_not_match_computed",
                value=_unique(mismatched_acceptance_ids),
            )
    for check_id, summary in {
        "readiness_manual_protected_gpu_command_surface_summary": readiness_command_surface,
        "terminal_manual_protected_gpu_command_surface_summary": terminal_command_surface,
    }.items():
        if not summary:
            _failure(failures, check_id, "manual_protected_gpu_command_surface_summary_missing")
            continue
        if str(summary.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "manual_protected_gpu_command_surface_summary_roadmap_missing_or_wrong",
                value=summary.get("roadmap"),
            )
        if (
            bool(summary.get("safe_to_auto_start"))
            or bool(summary.get("release_claim_allowed"))
            or not bool(summary.get("not_release_evidence"))
            or not bool(summary.get("fail_closed"))
            or _safe_int(summary.get("unsafe_command_count")) != 0
            or _safe_int(summary.get("release_claim_allowed_after_success_count")) != 0
        ):
            _failure(
                failures,
                check_id,
                "manual_protected_gpu_command_surface_summary_allows_release_or_auto_start",
            )
        if str(summary.get("execution_policy") or "") != "manual_protected_or_external_input_only":
            _failure(
                failures,
                check_id,
                "manual_protected_gpu_command_surface_summary_execution_policy_wrong",
                value=summary.get("execution_policy"),
            )
        for field in [
            "source_artifact_count",
            "command_surface_row_count",
            "manual_gpu_command_count",
            "protected_gpu_command_count",
            "dry_run_command_count",
            "template_command_count",
            "ready_command_count",
            "blocked_command_count",
            "completed_existing_command_count",
            "rerun_blocked_without_new_axis_count",
            "requires_gpu_if_executed_count",
            "manual_start_required_count",
            "release_relevant_command_count",
            "diagnostic_only_command_count",
            "release_claim_allowed_after_success_count",
            "unsafe_command_count",
        ]:
            _compare_scalar_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"manual_protected_gpu_command_surface_summary_{field}_does_not_match_computed",
                left=_safe_int(summary.get(field)),
                right=_safe_int(computed_command_surface.get(field)),
            )
        for field in ["source_artifact_ids", "unsafe_command_ids"]:
            _compare_set_field(
                failures,
                check_id=f"{check_id}_{field}_match",
                reason=f"manual_protected_gpu_command_surface_summary_{field}_does_not_match_computed",
                left=_strings(summary.get(field)),
                right=_strings(computed_command_surface.get(field)),
            )
        _compare_set_field(
            failures,
            check_id=f"{check_id}_row_ids_match",
            reason="manual_protected_gpu_command_surface_summary_row_ids_do_not_match_computed",
            left=[str(_mapping(row).get("id") or "") for row in _list(summary.get("rows"))],
            right=_strings(computed_command_surface.get("row_ids")),
        )
        unsafe_row_ids = [
            str(row.get("id") or "")
            for row in (_mapping(item) for item in _list(summary.get("rows")))
            if bool(row.get("safe_to_auto_start"))
            or bool(row.get("release_claim_allowed_after_success"))
            or not bool(row.get("not_release_evidence"))
            or (
                bool(row.get("requires_gpu_if_executed"))
                and not bool(row.get("manual_start_required"))
            )
            or bool(row.get("unsafe"))
        ]
        if unsafe_row_ids:
            _failure(
                failures,
                f"{check_id}_rows_fail_closed",
                "manual_protected_gpu_command_surface_summary_rows_allow_release_or_auto_start",
                value=_unique(unsafe_row_ids),
            )
    if _safe_int(terminal_audit.get("source_path_missing_count")) != 0:
        _failure(failures, "source_paths_missing", "terminal_source_path_missing_count_nonzero")
    if _safe_int(terminal_audit.get("source_path_load_error_count")) != 0:
        _failure(failures, "source_paths_load_error", "terminal_source_path_load_error_count_nonzero")
    if not bool(terminal_audit.get("chain_complete")):
        _failure(failures, "json_chain_complete", "terminal_json_chain_not_complete")

    if str(terminal.get("terminal_status") or "") != "external_input_and_manual_gpu_blocked":
        _failure(failures, "terminal_status", "terminal_status_not_external_or_manual_blocked", value=terminal.get("terminal_status"))
    if str(terminal.get("chain_integrity_status") or "") != "complete_waiting_external_input":
        _failure(failures, "chain_integrity_status", "terminal_chain_integrity_not_complete_waiting_external_input", value=terminal.get("chain_integrity_status"))

    required_inputs = _unique(
        [
            *_strings(readiness_manual.get("next_required_inputs")),
            *_strings(terminal_manual.get("next_required_inputs")),
        ]
    )
    for required in ("sd15_checkpoint", "source_cache_axis_manual_canary_evidence", "manual_gpu_evidence"):
        if required not in required_inputs:
            _failure(failures, f"required_input_{required}", "required_manual_or_external_input_missing", value=required_inputs)

    if not bool(readiness_input.get("sd15_checkpoint_required")) or not bool(
        terminal_input.get("sd15_checkpoint_required")
    ):
        _failure(failures, "sd15_checkpoint_required", "sd15_checkpoint_required_not_reflected_in_summaries")
    if not bool(readiness_manual.get("manual_gpu_evidence_required")) or not bool(
        terminal_manual.get("manual_gpu_evidence_required")
    ):
        _failure(failures, "manual_gpu_evidence_required", "manual_gpu_evidence_required_not_reflected_in_summaries")
    for check_id, candidate in {
        "readiness_manual_evidence_blocking_summary": readiness_manual,
        "terminal_manual_evidence_blocking_summary": terminal_manual,
    }.items():
        if not candidate:
            _failure(failures, check_id, "manual_evidence_blocking_summary_missing")
            continue
        if str(candidate.get("roadmap") or "") != ROADMAP:
            _failure(
                failures,
                check_id,
                "manual_evidence_blocking_summary_roadmap_missing_or_wrong",
                value=candidate.get("roadmap"),
            )
        if bool(candidate.get("safe_to_auto_start")) or bool(candidate.get("release_claim_allowed")):
            _failure(failures, check_id, "manual_evidence_blocking_summary_allows_release_or_auto_start")
        if not bool(candidate.get("not_release_evidence")):
            _failure(failures, check_id, "manual_evidence_blocking_summary_not_marked_non_release")
    for field in [
        "manual_gpu_evidence_ready",
        "manual_gpu_evidence_required",
        "source_cache_axis_manual_canary_plan_ready",
        "source_cache_axis_manual_canary_plan_required",
        "sd15_checkpoint_required",
        "natural_load_canary_pending",
        "release_claims_rebuild_required",
    ]:
        _compare_scalar_field(
            failures,
            check_id=f"manual_evidence_{field}_match",
            reason=f"manual_evidence_{field}_does_not_match_readiness",
            left=bool(readiness_manual.get(field)),
            right=bool(terminal_manual.get(field)),
        )
    _compare_set_field(
        failures,
        check_id="manual_evidence_required_inputs_match",
        reason="manual_evidence_required_inputs_do_not_match_readiness",
        left=_strings(readiness_manual.get("next_required_inputs")),
        right=_strings(terminal_manual.get("next_required_inputs")),
    )
    _compare_set_field(
        failures,
        check_id="manual_evidence_release_gate_blockers_match",
        reason="manual_evidence_release_gate_blockers_do_not_match_readiness",
        left=_strings(readiness_manual.get("release_gate_blockers")),
        right=_strings(terminal_manual.get("release_gate_blockers")),
    )
    _compare_scalar_field(
        failures,
        check_id="manual_evidence_next_rebuild_stage_match",
        reason="manual_evidence_next_rebuild_stage_does_not_match_readiness",
        left=str(readiness_manual.get("next_json_rebuild_stage_id") or ""),
        right=str(terminal_manual.get("next_json_rebuild_stage_id") or ""),
    )

    prep_action = _action_by_id(readiness, "prepare_sdxl_cuda_debug_repeat_protected_followup_axis_after_review")
    if prep_action and str(prep_action.get("readiness_state") or "") != "json_closed":
        _failure(
            failures,
            "protected_followup_axis_preparation",
            "prepared_followup_axis_action_not_closed",
            value=prep_action.get("readiness_state"),
        )

    ok = not failures
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_release_readiness_guard_report",
        "not_release_evidence": True,
        "status": "guard_passed_blocked_release_claim" if ok else "guard_failed",
        "ok": ok,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "claim_wording_policy": CLAIM_WORDING_POLICY,
        "release_gain_claim_wording_allowed": False,
        "forbidden_claim_wording_hit_count": _safe_int(
            claim_wording_policy.get("forbidden_claim_wording_hit_count")
        )
        + _safe_int(claim_wording_policy.get("terminal_forbidden_claim_wording_hit_count"))
        + _safe_int(claim_wording_policy.get("guard_rescan_hit_count")),
        "stable_first_release_blocked_by_this_artifact": bool(
            first_release.get("stable_first_release_blocked_by_this_artifact")
        ),
        "gpu_bubble_release_claim_blocked": bool(readiness_hard_gates),
        "gpu_bubble_release_hard_gate_ids": readiness_hard_gates[:20],
        "input_artifact_summary": _input_artifact_summary(
            readiness=readiness,
            terminal=terminal,
            remaining=remaining,
            terminal_audit=terminal_audit,
            artifact_freshness=artifact_freshness,
            readiness_hard_gates=readiness_hard_gates,
        ),
        "release_gate_post_input_refresh_guard_report_acceptance_summary": {
            "roadmap": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get("roadmap") or ""
            ),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get("artifact_role") or ""
            ),
            "acceptance_status": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_status"
                )
                or ""
            ),
            "acceptance_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_ok"
                )
            ),
            "computed_acceptance_ok": bool(
                computed_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_ok"
                )
            ),
            "guard_command_id": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_command_id"
                )
                or ""
            ),
            "guard_report_artifact_id": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_report_artifact_id"
                )
                or ""
            ),
            "expected_report_status": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_report_status"
                )
                or ""
            ),
            "expected_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_ok"
                )
            ),
            "expected_failure_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_failure_count"
                )
            ),
            "required_guard_report_field_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "required_guard_report_field_count"
                )
            ),
            "acceptance_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_row_count"
                )
            ),
            "unsafe_acceptance_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "unsafe_acceptance_count"
                )
            ),
            "guard_consumption_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_consumption_ok"
                )
            ),
            "input_artifacts_consumed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "input_artifacts_consumed"
                )
            ),
            "guard_artifact_produced": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_artifact_produced"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "fail_closed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "not_release_evidence"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "release_claim_allowed"
                )
            ),
        },
        "external_input_json_refresh_runner_manifest_summary": external_input_json_refresh_runner_manifest_summary,
        "downstream_artifacts": _downstream_artifact_summary(readiness),
        "roadmap_lineage_audit": {
            "expected_roadmap": str(roadmap_lineage.get("expected_roadmap") or ""),
            "lineage_ok": bool(roadmap_lineage.get("lineage_ok")),
            "audited_artifact_count": _safe_int(roadmap_lineage.get("audited_artifact_count")),
            "missing_required_artifact_count": _safe_int(
                roadmap_lineage.get("missing_required_artifact_count")
            ),
            "missing_required_artifact_ids": _strings(roadmap_lineage.get("missing_required_artifact_ids"))[:20],
            "mismatched_artifact_count": _safe_int(roadmap_lineage.get("mismatched_artifact_count")),
            "mismatched_artifact_ids": _strings(roadmap_lineage.get("mismatched_artifact_ids"))[:20],
        },
        "source_cache_axis_pipeline_stage_lineage": source_cache_stage_lineage,
        "roadmap_acceptance_gate_summary": roadmap_acceptance,
        "roadmap_execution_contract_summary": roadmap_execution,
        "experiment_matrix_readiness": experiment_matrix,
        "normalized_evidence_gate_mapping": normalized_gate_mapping,
        "normalized_evidence_gate_explanation_summary": {
            "available": bool(normalized_gate_mapping.get("explanation_available")),
            "terminal_available": bool(normalized_gate_mapping.get("terminal_explanation_available")),
            "artifact_role": str(normalized_gate_mapping.get("explanation_artifact_role") or ""),
            "terminal_artifact_role": str(
                normalized_gate_mapping.get("terminal_explanation_artifact_role") or ""
            ),
            "mapped_row_count": _safe_int(
                normalized_gate_mapping.get("explanation_mapped_row_count")
            ),
            "terminal_mapped_row_count": _safe_int(
                normalized_gate_mapping.get("terminal_explanation_mapped_row_count")
            ),
            "computed_mapped_row_count": _safe_int(
                normalized_gate_mapping.get("computed_explanation_mapped_row_count")
            ),
            "row_outcome_counts": dict(
                _mapping(normalized_gate_mapping.get("explanation_row_outcome_counts"))
            ),
            "computed_row_outcome_counts": dict(
                _mapping(normalized_gate_mapping.get("computed_explanation_row_outcome_counts"))
            ),
            "gate_outcome_counts": dict(
                _mapping(normalized_gate_mapping.get("explanation_gate_outcome_counts"))
            ),
            "computed_gate_outcome_counts": dict(
                _mapping(normalized_gate_mapping.get("computed_explanation_gate_outcome_counts"))
            ),
            "blocker_explanation_counts": dict(
                _mapping(normalized_gate_mapping.get("explanation_blocker_explanation_counts"))
            ),
            "computed_blocker_explanation_counts": dict(
                _mapping(
                    normalized_gate_mapping.get("computed_explanation_blocker_explanation_counts")
                )
            ),
            "missing_metric_counts": dict(
                _mapping(normalized_gate_mapping.get("explanation_missing_metric_counts"))
            ),
            "computed_missing_metric_counts": dict(
                _mapping(normalized_gate_mapping.get("computed_explanation_missing_metric_counts"))
            ),
            "release_hard_gate_row_counts": dict(
                _mapping(normalized_gate_mapping.get("explanation_release_hard_gate_row_counts"))
            ),
            "computed_release_hard_gate_row_counts": dict(
                _mapping(
                    normalized_gate_mapping.get(
                        "computed_explanation_release_hard_gate_row_counts"
                    )
                )
            ),
            "unsafe_row_count": _safe_int(
                normalized_gate_mapping.get("explanation_unsafe_row_count")
            ),
            "terminal_unsafe_row_count": _safe_int(
                normalized_gate_mapping.get("terminal_explanation_unsafe_row_count")
            ),
            "computed_unsafe_row_count": _safe_int(
                normalized_gate_mapping.get("computed_explanation_unsafe_row_count")
            ),
            "fail_closed": bool(normalized_gate_mapping.get("explanation_fail_closed")),
            "terminal_fail_closed": bool(
                normalized_gate_mapping.get("terminal_explanation_fail_closed")
            ),
            "computed_fail_closed": bool(
                normalized_gate_mapping.get("computed_explanation_fail_closed")
            ),
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
        },
        "source_artifact_inventory_summary": source_artifact_inventory,
        "evidence_summary_inventory_summary": evidence_summary_inventory,
        "source_and_downstream_artifact_contract_summary": source_downstream_contract,
        "first_release_policy_summary": first_release_policy,
        "claim_wording_policy_summary": claim_wording_policy,
        "forbidden_claim_wording_audit": {
            "claim_wording_policy": str(claim_wording_policy.get("claim_wording_policy") or ""),
            "terminal_claim_wording_policy": str(
                claim_wording_policy.get("terminal_claim_wording_policy") or ""
            ),
            "release_gain_claim_wording_allowed": bool(
                claim_wording_policy.get("release_gain_claim_wording_allowed")
            ),
            "terminal_release_gain_claim_wording_allowed": bool(
                claim_wording_policy.get("terminal_release_gain_claim_wording_allowed")
            ),
            "forbidden_claim_wording_hit_count": _safe_int(
                claim_wording_policy.get("forbidden_claim_wording_hit_count")
            ),
            "terminal_forbidden_claim_wording_hit_count": _safe_int(
                claim_wording_policy.get("terminal_forbidden_claim_wording_hit_count")
            ),
            "guard_rescan_hit_count": _safe_int(
                claim_wording_policy.get("guard_rescan_hit_count")
            ),
            "forbidden_claim_wording_tokens": _strings(
                claim_wording_policy.get("forbidden_claim_wording_tokens")
            )[:20],
            "readiness_not_release_evidence": bool(
                claim_wording_policy.get("readiness_not_release_evidence")
            ),
            "terminal_not_release_evidence": bool(
                claim_wording_policy.get("terminal_not_release_evidence")
            ),
            "readiness_safe_to_auto_start": bool(
                claim_wording_policy.get("readiness_safe_to_auto_start")
            ),
            "terminal_safe_to_auto_start": bool(
                claim_wording_policy.get("terminal_safe_to_auto_start")
            ),
            "readiness_release_claim_allowed": bool(
                claim_wording_policy.get("readiness_release_claim_allowed")
            ),
            "terminal_release_claim_allowed": bool(
                claim_wording_policy.get("terminal_release_claim_allowed")
            ),
            "not_release_evidence": bool(claim_wording_policy.get("not_release_evidence")),
            "safe_to_auto_start": bool(claim_wording_policy.get("safe_to_auto_start")),
            "release_claim_allowed": bool(claim_wording_policy.get("release_claim_allowed")),
        },
        "json_ready_action_count": readiness_json_ready,
        "json_closed_action_count": _safe_int(remaining.get("json_closed_action_count")),
        "remaining_work_summary": {
            "roadmap": str(terminal_remaining_work.get("roadmap") or ""),
            "artifact_role": str(terminal_remaining_work.get("artifact_role") or ""),
            "total_action_count": _safe_int(terminal_remaining_work.get("total_action_count")),
            "computed_total_action_count": _safe_int(
                computed_remaining_work.get("total_action_count")
            ),
            "stable_first_release_blocked_by_this_artifact": bool(
                terminal_remaining_work.get("stable_first_release_blocked_by_this_artifact")
            ),
            "gpu_bubble_release_claim_blocked": bool(
                terminal_remaining_work.get("gpu_bubble_release_claim_blocked")
            ),
            "gpu_bubble_release_hard_gate_count": _safe_int(
                terminal_remaining_work.get("gpu_bubble_release_hard_gate_count")
            ),
            "computed_gpu_bubble_release_hard_gate_count": _safe_int(
                computed_remaining_work.get("gpu_bubble_release_hard_gate_count")
            ),
            "gpu_bubble_release_hard_gate_ids": _strings(
                terminal_remaining_work.get("gpu_bubble_release_hard_gate_ids")
            )[:20],
            "json_ready_action_count": _safe_int(
                terminal_remaining_work.get("json_ready_action_count")
            ),
            "computed_json_ready_action_count": _safe_int(
                computed_remaining_work.get("json_ready_action_count")
            ),
            "json_closed_action_count": _safe_int(
                terminal_remaining_work.get("json_closed_action_count")
            ),
            "computed_json_closed_action_count": _safe_int(
                computed_remaining_work.get("json_closed_action_count")
            ),
            "external_input_action_count": _safe_int(
                terminal_remaining_work.get("external_input_action_count")
            ),
            "computed_external_input_action_count": _safe_int(
                computed_remaining_work.get("external_input_action_count")
            ),
            "manual_gpu_evidence_action_count": _safe_int(
                terminal_remaining_work.get("manual_gpu_evidence_action_count")
            ),
            "computed_manual_gpu_evidence_action_count": _safe_int(
                computed_remaining_work.get("manual_gpu_evidence_action_count")
            ),
            "followup_gpu_required_action_count": _safe_int(
                terminal_remaining_work.get("followup_gpu_required_action_count")
            ),
            "computed_followup_gpu_required_action_count": _safe_int(
                computed_remaining_work.get("followup_gpu_required_action_count")
            ),
            "current_gpu_heavy_action_count": _safe_int(
                terminal_remaining_work.get("current_gpu_heavy_action_count")
            ),
            "computed_current_gpu_heavy_action_count": _safe_int(
                computed_remaining_work.get("current_gpu_heavy_action_count")
            ),
            "cache_axis_not_ready_action_count": _safe_int(
                terminal_remaining_work.get("cache_axis_not_ready_action_count")
            ),
            "duplicate_or_stale_axis_action_count": _safe_int(
                terminal_remaining_work.get("duplicate_or_stale_axis_action_count")
            ),
            "release_gate_related_action_count": _safe_int(
                terminal_remaining_work.get("release_gate_related_action_count")
            ),
            "computed_release_gate_related_action_count": _safe_int(
                computed_remaining_work.get("release_gate_related_action_count")
            ),
            "recommended_release_policy": str(
                terminal_remaining_work.get("recommended_release_policy") or ""
            ),
            "recommended_next_non_gpu_focus": str(
                terminal_remaining_work.get("recommended_next_non_gpu_focus") or ""
            ),
            "unsafe_action_count": _safe_int(terminal_remaining_work.get("unsafe_action_count")),
            "computed_unsafe_action_count": _safe_int(
                computed_remaining_work.get("unsafe_action_count")
            ),
            "unsafe_action_ids": _strings(terminal_remaining_work.get("unsafe_action_ids"))[:50],
            "fail_closed": bool(terminal_remaining_work.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_remaining_work.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_remaining_work.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_remaining_work.get("not_release_evidence")),
        },
        "manual_review_queue_summary": {
            "roadmap": str(terminal_manual_review_queue.get("roadmap") or ""),
            "artifact_role": str(terminal_manual_review_queue.get("artifact_role") or ""),
            "manual_review_ready_count": _safe_int(
                terminal_manual_review_queue.get("manual_review_ready_count")
            ),
            "computed_manual_review_ready_count": _safe_int(
                computed_manual_review_queue.get("manual_review_ready_count")
            ),
            "closed_blocked_or_regression_count": _safe_int(
                terminal_manual_review_queue.get("closed_blocked_or_regression_count")
            ),
            "computed_closed_blocked_or_regression_count": _safe_int(
                computed_manual_review_queue.get("closed_blocked_or_regression_count")
            ),
            "closed_diagnostic_or_promotion_count": _safe_int(
                terminal_manual_review_queue.get("closed_diagnostic_or_promotion_count")
            ),
            "computed_closed_diagnostic_or_promotion_count": _safe_int(
                computed_manual_review_queue.get("closed_diagnostic_or_promotion_count")
            ),
            "review_only_action_count": _safe_int(
                terminal_manual_review_queue.get("review_only_action_count")
            ),
            "computed_review_only_action_count": _safe_int(
                computed_manual_review_queue.get("review_only_action_count")
            ),
            "followup_gpu_action_count": _safe_int(
                terminal_manual_review_queue.get("followup_gpu_action_count")
            ),
            "computed_followup_gpu_action_count": _safe_int(
                computed_manual_review_queue.get("followup_gpu_action_count")
            ),
            "current_gpu_heavy_action_count": _safe_int(
                terminal_manual_review_queue.get("current_gpu_heavy_action_count")
            ),
            "computed_current_gpu_heavy_action_count": _safe_int(
                computed_manual_review_queue.get("current_gpu_heavy_action_count")
            ),
            "closed_blocked_or_regression_action_ids": _strings(
                terminal_manual_review_queue.get("closed_blocked_or_regression_action_ids")
            )[:20],
            "closed_diagnostic_or_promotion_action_ids": _strings(
                terminal_manual_review_queue.get("closed_diagnostic_or_promotion_action_ids")
            )[:20],
            "review_only_action_ids": _strings(
                terminal_manual_review_queue.get("review_only_action_ids")
            )[:20],
            "followup_gpu_action_ids": _strings(
                terminal_manual_review_queue.get("followup_gpu_action_ids")
            )[:20],
            "current_gpu_heavy_action_ids": _strings(
                terminal_manual_review_queue.get("current_gpu_heavy_action_ids")
            )[:20],
            "review_outcome_counts": dict(
                _mapping(terminal_manual_review_queue.get("review_outcome_counts"))
            ),
            "computed_review_outcome_counts": dict(
                _mapping(computed_manual_review_queue.get("review_outcome_counts"))
            ),
            "review_outcome_action_ids": {
                str(kind): _strings(ids)[:20]
                for kind, ids in _mapping(
                    terminal_manual_review_queue.get("review_outcome_action_ids")
                ).items()
            },
            "computed_review_outcome_action_ids": {
                str(kind): _strings(ids)[:20]
                for kind, ids in _mapping(
                    computed_manual_review_queue.get("review_outcome_action_ids")
                ).items()
            },
            "unsafe_action_count": _safe_int(terminal_manual_review_queue.get("unsafe_action_count")),
            "computed_unsafe_action_count": _safe_int(
                computed_manual_review_queue.get("unsafe_action_count")
            ),
            "unsafe_action_ids": _strings(terminal_manual_review_queue.get("unsafe_action_ids"))[:20],
            "computed_unsafe_action_ids": _strings(
                computed_manual_review_queue.get("unsafe_action_ids")
            )[:20],
            "fail_closed": bool(terminal_manual_review_queue.get("fail_closed")),
            "computed_fail_closed": bool(computed_manual_review_queue.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_manual_review_queue.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                terminal_manual_review_queue.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_manual_review_queue.get("not_release_evidence")
            ),
        },
        "manual_gpu_execution_summary": {
            "gpu_related_action_count": _safe_int(manual_gpu_summary.get("gpu_related_action_count")),
            "terminal_gpu_related_action_count": _safe_int(
                terminal_manual_gpu_summary.get("gpu_related_action_count")
            ),
            "auto_startable_gpu_action_count": _safe_int(
                manual_gpu_summary.get("auto_startable_gpu_action_count")
            ),
            "terminal_auto_startable_gpu_action_count": _safe_int(
                terminal_manual_gpu_summary.get("auto_startable_gpu_action_count")
            ),
            "release_claim_allowed_after_success_action_count": _safe_int(
                manual_gpu_summary.get("release_claim_allowed_after_success_action_count")
            ),
            "terminal_release_claim_allowed_after_success_action_count": _safe_int(
                terminal_manual_gpu_summary.get("release_claim_allowed_after_success_action_count")
            ),
            "execution_policy": str(manual_gpu_summary.get("execution_policy") or ""),
        },
        "protected_followup_gpu_queue_summary": {
            "roadmap": str(terminal_followup_queue.get("roadmap") or ""),
            "artifact_role": str(terminal_followup_queue.get("artifact_role") or ""),
            "queue_status": str(terminal_followup_queue.get("queue_status") or ""),
            "followup_gpu_required_action_count": _safe_int(
                terminal_followup_queue.get("followup_gpu_required_action_count")
            ),
            "computed_followup_gpu_required_action_count": _safe_int(
                computed_followup_queue.get("followup_gpu_required_action_count")
            ),
            "followup_gpu_required_action_ids": _strings(
                terminal_followup_queue.get("followup_gpu_required_action_ids")
            )[:20],
            "family_counts": dict(_mapping(terminal_followup_queue.get("family_counts"))),
            "readiness_state_counts": dict(
                _mapping(terminal_followup_queue.get("readiness_state_counts"))
            ),
            "readiness_blocker_kind_counts": dict(
                _mapping(terminal_followup_queue.get("readiness_blocker_kind_counts"))
            ),
            "current_action_gpu_count": _safe_int(
                terminal_followup_queue.get("current_action_gpu_count")
            ),
            "current_action_manual_start_count": _safe_int(
                terminal_followup_queue.get("current_action_manual_start_count")
            ),
            "followup_manual_start_required_count": _safe_int(
                terminal_followup_queue.get("followup_manual_start_required_count")
            ),
            "requires_external_input_count": _safe_int(
                terminal_followup_queue.get("requires_external_input_count")
            ),
            "unsafe_action_count": _safe_int(terminal_followup_queue.get("unsafe_action_count")),
            "unsafe_action_ids": _strings(terminal_followup_queue.get("unsafe_action_ids"))[:50],
            "execution_policy": str(terminal_followup_queue.get("execution_policy") or ""),
            "fail_closed": bool(terminal_followup_queue.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_followup_queue.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_followup_queue.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_followup_queue.get("not_release_evidence")),
        },
        "remaining_release_blocker_matrix_summary": {
            "roadmap": str(terminal_blocker_matrix.get("roadmap") or ""),
            "artifact_role": str(terminal_blocker_matrix.get("artifact_role") or ""),
            "matrix_status": str(terminal_blocker_matrix.get("matrix_status") or ""),
            "total_unclosed_action_count": _safe_int(
                terminal_blocker_matrix.get("total_unclosed_action_count")
            ),
            "computed_total_unclosed_action_count": _safe_int(
                computed_blocker_matrix.get("total_unclosed_action_count")
            ),
            "unclosed_action_ids": _strings(terminal_blocker_matrix.get("unclosed_action_ids"))[:50],
            "json_ready_action_count": _safe_int(terminal_blocker_matrix.get("json_ready_action_count")),
            "readiness_state_counts": dict(
                _mapping(terminal_blocker_matrix.get("readiness_state_counts"))
            ),
            "blocked_by_kind_counts": dict(_mapping(terminal_blocker_matrix.get("blocked_by_kind_counts"))),
            "family_count": _safe_int(terminal_blocker_matrix.get("family_count")),
            "computed_family_count": _safe_int(computed_blocker_matrix.get("family_count")),
            "family_ids": _strings(terminal_blocker_matrix.get("family_ids"))[:20],
            "family_action_counts": dict(_mapping(terminal_blocker_matrix.get("family_action_counts"))),
            "family_external_input_required_counts": dict(
                _mapping(terminal_blocker_matrix.get("family_external_input_required_counts"))
            ),
            "family_manual_gpu_required_counts": dict(
                _mapping(terminal_blocker_matrix.get("family_manual_gpu_required_counts"))
            ),
            "family_protected_followup_gpu_required_counts": dict(
                _mapping(terminal_blocker_matrix.get("family_protected_followup_gpu_required_counts"))
            ),
            "family_source_cache_blocked_counts": dict(
                _mapping(terminal_blocker_matrix.get("family_source_cache_blocked_counts"))
            ),
            "unsafe_family_count": _safe_int(terminal_blocker_matrix.get("unsafe_family_count")),
            "family_unsafe_action_counts": dict(
                _mapping(terminal_blocker_matrix.get("family_unsafe_action_counts"))
            ),
            "release_hard_gate_ids": _strings(terminal_blocker_matrix.get("release_hard_gate_ids"))[:20],
            "external_input_required_action_count": _safe_int(
                terminal_blocker_matrix.get("external_input_required_action_count")
            ),
            "external_input_required_action_ids": _strings(
                terminal_blocker_matrix.get("external_input_required_action_ids")
            )[:50],
            "missing_external_inputs": _strings(terminal_blocker_matrix.get("missing_external_inputs"))[:20],
            "manual_gpu_required_action_count": _safe_int(
                terminal_blocker_matrix.get("manual_gpu_required_action_count")
            ),
            "manual_gpu_required_action_ids": _strings(
                terminal_blocker_matrix.get("manual_gpu_required_action_ids")
            )[:50],
            "protected_followup_gpu_required_action_count": _safe_int(
                terminal_blocker_matrix.get("protected_followup_gpu_required_action_count")
            ),
            "protected_followup_gpu_required_action_ids": _strings(
                terminal_blocker_matrix.get("protected_followup_gpu_required_action_ids")
            )[:50],
            "source_cache_blocked_action_count": _safe_int(
                terminal_blocker_matrix.get("source_cache_blocked_action_count")
            ),
            "source_cache_blocked_action_ids": _strings(
                terminal_blocker_matrix.get("source_cache_blocked_action_ids")
            )[:50],
            "sd15_checkpoint_action_count": _safe_int(
                terminal_blocker_matrix.get("sd15_checkpoint_action_count")
            ),
            "sd15_checkpoint_action_ids": _strings(
                terminal_blocker_matrix.get("sd15_checkpoint_action_ids")
            )[:50],
            "duplicate_or_stale_source_axis_action_count": _safe_int(
                terminal_blocker_matrix.get("duplicate_or_stale_source_axis_action_count")
            ),
            "duplicate_or_stale_source_axis_action_ids": _strings(
                terminal_blocker_matrix.get("duplicate_or_stale_source_axis_action_ids")
            )[:50],
            "next_unlock_inputs": _strings(terminal_blocker_matrix.get("next_unlock_inputs"))[:50],
            "unsafe_action_count": _safe_int(terminal_blocker_matrix.get("unsafe_action_count")),
            "unsafe_action_ids": _strings(terminal_blocker_matrix.get("unsafe_action_ids"))[:50],
            "execution_policy": str(terminal_blocker_matrix.get("execution_policy") or ""),
            "fail_closed": bool(terminal_blocker_matrix.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_blocker_matrix.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_blocker_matrix.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_blocker_matrix.get("not_release_evidence")),
        },
        "remaining_blocker_resolution_handoff_summary": {
            "roadmap": str(terminal_blocker_handoff.get("roadmap") or ""),
            "artifact_role": str(terminal_blocker_handoff.get("artifact_role") or ""),
            "handoff_status": str(terminal_blocker_handoff.get("handoff_status") or ""),
            "row_count": _safe_int(terminal_blocker_handoff.get("row_count")),
            "computed_row_count": _safe_int(computed_blocker_handoff.get("row_count")),
            "row_ids": _strings(terminal_blocker_handoff.get("row_ids"))[:50],
            "resolution_contract_version": _safe_int(
                terminal_blocker_handoff.get("resolution_contract_version"),
                1,
            ),
            "resolution_contract_ok": bool(
                terminal_blocker_handoff.get("resolution_contract_ok")
            ),
            "resolution_contract_bad_count": _safe_int(
                terminal_blocker_handoff.get("resolution_contract_bad_count")
            ),
            "resolution_contract_bad_ids": _strings(
                terminal_blocker_handoff.get("resolution_contract_bad_ids")
            )[:50],
            "resolution_bucket_counts": dict(
                _mapping(terminal_blocker_handoff.get("resolution_bucket_counts"))
            ),
            "json_only_resolution_available_count": _safe_int(
                terminal_blocker_handoff.get("json_only_resolution_available_count")
            ),
            "external_input_required_count": _safe_int(
                terminal_blocker_handoff.get("external_input_required_count")
            ),
            "manual_gpu_required_count": _safe_int(
                terminal_blocker_handoff.get("manual_gpu_required_count")
            ),
            "protected_runner_required_count": _safe_int(
                terminal_blocker_handoff.get("protected_runner_required_count")
            ),
            "release_claim_after_resolution_allowed": bool(
                terminal_blocker_handoff.get("release_claim_after_resolution_allowed")
            ),
            "blocker_bucket_counts": dict(
                _mapping(terminal_blocker_handoff.get("blocker_bucket_counts"))
            ),
            "next_unlock_input_ids": _strings(
                terminal_blocker_handoff.get("next_unlock_input_ids")
            )[:50],
            "required_refresh_command_ids": _strings(
                terminal_blocker_handoff.get("required_refresh_command_ids")
            )[:20],
            "external_input_row_count": _safe_int(
                terminal_blocker_handoff.get("external_input_row_count")
            ),
            "current_gpu_row_count": _safe_int(terminal_blocker_handoff.get("current_gpu_row_count")),
            "protected_followup_gpu_row_count": _safe_int(
                terminal_blocker_handoff.get("protected_followup_gpu_row_count")
            ),
            "unsafe_row_count": _safe_int(terminal_blocker_handoff.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(terminal_blocker_handoff.get("unsafe_row_ids"))[:50],
            "execution_policy": str(terminal_blocker_handoff.get("execution_policy") or ""),
            "fail_closed": bool(terminal_blocker_handoff.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_blocker_handoff.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_blocker_handoff.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_blocker_handoff.get("not_release_evidence")),
        },
        "remaining_action_dependency_graph_summary": {
            "roadmap": str(terminal_action_dependency_graph.get("roadmap") or ""),
            "artifact_role": str(terminal_action_dependency_graph.get("artifact_role") or ""),
            "graph_status": str(terminal_action_dependency_graph.get("graph_status") or ""),
            "action_node_count": _safe_int(terminal_action_dependency_graph.get("action_node_count")),
            "computed_action_node_count": _safe_int(
                computed_action_dependency_graph.get("action_node_count")
            ),
            "action_node_ids": _strings(terminal_action_dependency_graph.get("action_node_ids"))[:50],
            "dependency_node_count": _safe_int(
                terminal_action_dependency_graph.get("dependency_node_count")
            ),
            "computed_dependency_node_count": _safe_int(
                computed_action_dependency_graph.get("dependency_node_count")
            ),
            "dependency_node_ids": _strings(
                terminal_action_dependency_graph.get("dependency_node_ids")
            )[:80],
            "edge_count": _safe_int(terminal_action_dependency_graph.get("edge_count")),
            "computed_edge_count": _safe_int(computed_action_dependency_graph.get("edge_count")),
            "action_state_counts": dict(
                _mapping(terminal_action_dependency_graph.get("action_state_counts"))
            ),
            "computed_action_state_counts": dict(
                _mapping(computed_action_dependency_graph.get("action_state_counts"))
            ),
            "blocker_kind_counts": dict(
                _mapping(terminal_action_dependency_graph.get("blocker_kind_counts"))
            ),
            "computed_blocker_kind_counts": dict(
                _mapping(computed_action_dependency_graph.get("blocker_kind_counts"))
            ),
            "dependency_kind_counts": dict(
                _mapping(terminal_action_dependency_graph.get("dependency_kind_counts"))
            ),
            "computed_dependency_kind_counts": dict(
                _mapping(computed_action_dependency_graph.get("dependency_kind_counts"))
            ),
            "missing_external_inputs": _strings(
                terminal_action_dependency_graph.get("missing_external_inputs")
            )[:20],
            "release_hard_gate_ids": _strings(
                terminal_action_dependency_graph.get("release_hard_gate_ids")
            )[:20],
            "required_refresh_command_ids": _strings(
                terminal_action_dependency_graph.get("required_refresh_command_ids")
            )[:20],
            "refresh_sequence_terminal_guard_ok": bool(
                terminal_action_dependency_graph.get("refresh_sequence_terminal_guard_ok")
            ),
            "unsafe_action_count": _safe_int(
                terminal_action_dependency_graph.get("unsafe_action_count")
            ),
            "computed_unsafe_action_count": _safe_int(
                computed_action_dependency_graph.get("unsafe_action_count")
            ),
            "unsafe_action_ids": _strings(
                terminal_action_dependency_graph.get("unsafe_action_ids")
            )[:50],
            "execution_policy": str(terminal_action_dependency_graph.get("execution_policy") or ""),
            "fail_closed": bool(terminal_action_dependency_graph.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_action_dependency_graph.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                terminal_action_dependency_graph.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_action_dependency_graph.get("not_release_evidence")
            ),
        },
        "remaining_action_unblock_sequence_summary": {
            "roadmap": str(terminal_action_unblock_sequence.get("roadmap") or ""),
            "artifact_role": str(terminal_action_unblock_sequence.get("artifact_role") or ""),
            "sequence_status": str(terminal_action_unblock_sequence.get("sequence_status") or ""),
            "stage_count": _safe_int(terminal_action_unblock_sequence.get("stage_count")),
            "computed_stage_count": _safe_int(computed_action_unblock_sequence.get("stage_count")),
            "stage_ids": _strings(terminal_action_unblock_sequence.get("stage_ids"))[:20],
            "current_stage_id": str(terminal_action_unblock_sequence.get("current_stage_id") or ""),
            "computed_current_stage_id": str(
                computed_action_unblock_sequence.get("current_stage_id") or ""
            ),
            "next_required_input_ids": _strings(
                terminal_action_unblock_sequence.get("next_required_input_ids")
            )[:50],
            "manual_gpu_stage_count": _safe_int(
                terminal_action_unblock_sequence.get("manual_gpu_stage_count")
            ),
            "computed_manual_gpu_stage_count": _safe_int(
                computed_action_unblock_sequence.get("manual_gpu_stage_count")
            ),
            "external_input_stage_count": _safe_int(
                terminal_action_unblock_sequence.get("external_input_stage_count")
            ),
            "computed_external_input_stage_count": _safe_int(
                computed_action_unblock_sequence.get("external_input_stage_count")
            ),
            "protected_runner_stage_count": _safe_int(
                terminal_action_unblock_sequence.get("protected_runner_stage_count")
            ),
            "computed_protected_runner_stage_count": _safe_int(
                computed_action_unblock_sequence.get("protected_runner_stage_count")
            ),
            "release_hard_gate_ids": _strings(
                terminal_action_unblock_sequence.get("release_hard_gate_ids")
            )[:20],
            "terminal_guard_required": bool(
                terminal_action_unblock_sequence.get("terminal_guard_required")
            ),
            "required_refresh_command_ids": _strings(
                terminal_action_unblock_sequence.get("required_refresh_command_ids")
            )[:20],
            "refresh_sequence_terminal_guard_ok": bool(
                terminal_action_unblock_sequence.get("refresh_sequence_terminal_guard_ok")
            ),
            "unsafe_stage_count": _safe_int(
                terminal_action_unblock_sequence.get("unsafe_stage_count")
            ),
            "computed_unsafe_stage_count": _safe_int(
                computed_action_unblock_sequence.get("unsafe_stage_count")
            ),
            "unsafe_stage_ids": _strings(
                terminal_action_unblock_sequence.get("unsafe_stage_ids")
            )[:20],
            "execution_policy": str(terminal_action_unblock_sequence.get("execution_policy") or ""),
            "fail_closed": bool(terminal_action_unblock_sequence.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_action_unblock_sequence.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                terminal_action_unblock_sequence.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_action_unblock_sequence.get("not_release_evidence")
            ),
        },
        "remaining_blocker_artifact_presence_summary": {
            "roadmap": str(terminal_blocker_presence.get("roadmap") or ""),
            "artifact_role": str(terminal_blocker_presence.get("artifact_role") or ""),
            "presence_status": str(terminal_blocker_presence.get("presence_status") or ""),
            "row_count": _safe_int(terminal_blocker_presence.get("row_count")),
            "computed_row_count": _safe_int(computed_blocker_presence.get("row_count")),
            "row_ids": _strings(terminal_blocker_presence.get("row_ids"))[:50],
            "expected_output_action_count": _safe_int(
                terminal_blocker_presence.get("expected_output_action_count")
            ),
            "expected_output_missing_action_count": _safe_int(
                terminal_blocker_presence.get("expected_output_missing_action_count")
            ),
            "expected_output_missing_action_ids": _strings(
                terminal_blocker_presence.get("expected_output_missing_action_ids")
            )[:50],
            "evidence_path_action_count": _safe_int(
                terminal_blocker_presence.get("evidence_path_action_count")
            ),
            "evidence_path_missing_action_count": _safe_int(
                terminal_blocker_presence.get("evidence_path_missing_action_count")
            ),
            "evidence_path_missing_action_ids": _strings(
                terminal_blocker_presence.get("evidence_path_missing_action_ids")
            )[:50],
            "unsafe_row_count": _safe_int(terminal_blocker_presence.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(terminal_blocker_presence.get("unsafe_row_ids"))[:50],
            "execution_policy": str(terminal_blocker_presence.get("execution_policy") or ""),
            "fail_closed": bool(terminal_blocker_presence.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_blocker_presence.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_blocker_presence.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_blocker_presence.get("not_release_evidence")),
        },
        "release_claim_exit_criteria_summary": {
            "roadmap": str(terminal_release_exit.get("roadmap") or ""),
            "artifact_role": str(terminal_release_exit.get("artifact_role") or ""),
            "exit_status": str(terminal_release_exit.get("exit_status") or ""),
            "release_hard_gate_count": _safe_int(
                terminal_release_exit.get("release_hard_gate_count")
            ),
            "computed_release_hard_gate_count": _safe_int(
                computed_release_exit.get("release_hard_gate_count")
            ),
            "release_hard_gate_ids": _strings(
                terminal_release_exit.get("release_hard_gate_ids")
            )[:20],
            "gate_row_count": _safe_int(terminal_release_exit.get("gate_row_count")),
            "computed_gate_row_count": _safe_int(computed_release_exit.get("gate_row_count")),
            "json_only_exit_available_count": _safe_int(
                terminal_release_exit.get("json_only_exit_available_count")
            ),
            "manual_gpu_required_gate_count": _safe_int(
                terminal_release_exit.get("manual_gpu_required_gate_count")
            ),
            "protected_runner_required_gate_count": _safe_int(
                terminal_release_exit.get("protected_runner_required_gate_count")
            ),
            "missing_declared_output_gate_count": _safe_int(
                terminal_release_exit.get("missing_declared_output_gate_count")
            ),
            "unsafe_gate_count": _safe_int(terminal_release_exit.get("unsafe_gate_count")),
            "unsafe_gate_ids": _strings(terminal_release_exit.get("unsafe_gate_ids"))[:50],
            "rows": [
                {
                    "gate_id": str(row.get("gate_id") or ""),
                    "gate_status": str(row.get("gate_status") or ""),
                    "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                    "related_family_ids": _strings(row.get("related_family_ids"))[:20],
                    "related_family_counts": dict(_mapping(row.get("related_family_counts"))),
                    "related_readiness_state_counts": dict(
                        _mapping(row.get("related_readiness_state_counts"))
                    ),
                    "related_blocker_kind_counts": dict(
                        _mapping(row.get("related_blocker_kind_counts"))
                    ),
                    "related_external_input_action_count": _safe_int(
                        row.get("related_external_input_action_count")
                    ),
                    "related_manual_gpu_action_count": _safe_int(
                        row.get("related_manual_gpu_action_count")
                    ),
                    "related_protected_followup_action_count": _safe_int(
                        row.get("related_protected_followup_action_count")
                    ),
                    "required_input_ids": _strings(row.get("required_input_ids"))[:20],
                    "required_output_ids": _strings(row.get("required_output_ids"))[:20],
                    "missing_declared_output_action_ids": _strings(
                        row.get("missing_declared_output_action_ids")
                    )[:50],
                    "manual_gpu_required": bool(row.get("manual_gpu_required")),
                    "protected_runner_required": bool(row.get("protected_runner_required")),
                    "json_only_exit_available": bool(row.get("json_only_exit_available")),
                    "terminal_guard_required": bool(row.get("terminal_guard_required")),
                    "release_claim_allowed_after_exit": bool(
                        row.get("release_claim_allowed_after_exit")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                }
                for row in (_mapping(item) for item in _list(terminal_release_exit.get("rows")))
            ],
            "execution_policy": str(terminal_release_exit.get("execution_policy") or ""),
            "fail_closed": bool(terminal_release_exit.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_release_exit.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_release_exit.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_release_exit.get("not_release_evidence")),
        },
        "release_gate_input_dependency_summary": {
            "roadmap": str(terminal_release_gate_input_dependency.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_dependency.get("artifact_role") or ""
            ),
            "dependency_status": str(
                terminal_release_gate_input_dependency.get("dependency_status") or ""
            ),
            "release_hard_gate_count": _safe_int(
                terminal_release_gate_input_dependency.get("release_hard_gate_count")
            ),
            "computed_release_hard_gate_count": _safe_int(
                computed_release_gate_input_dependency.get("release_hard_gate_count")
            ),
            "release_hard_gate_ids": _strings(
                terminal_release_gate_input_dependency.get("release_hard_gate_ids")
            )[:20],
            "dependency_row_count": _safe_int(
                terminal_release_gate_input_dependency.get("dependency_row_count")
            ),
            "computed_dependency_row_count": _safe_int(
                computed_release_gate_input_dependency.get("dependency_row_count")
            ),
            "required_input_ids": _strings(
                terminal_release_gate_input_dependency.get("required_input_ids")
            )[:50],
            "missing_input_count": _safe_int(
                terminal_release_gate_input_dependency.get("missing_input_count")
            ),
            "computed_missing_input_count": _safe_int(
                computed_release_gate_input_dependency.get("missing_input_count")
            ),
            "missing_input_ids": _strings(
                terminal_release_gate_input_dependency.get("missing_input_ids")
            )[:50],
            "external_input_dependency_count": _safe_int(
                terminal_release_gate_input_dependency.get("external_input_dependency_count")
            ),
            "manual_gpu_dependency_count": _safe_int(
                terminal_release_gate_input_dependency.get("manual_gpu_dependency_count")
            ),
            "source_cache_refresh_dependency_count": _safe_int(
                terminal_release_gate_input_dependency.get(
                    "source_cache_refresh_dependency_count"
                )
            ),
            "json_only_resolution_available_count": _safe_int(
                terminal_release_gate_input_dependency.get(
                    "json_only_resolution_available_count"
                )
            ),
            "unsafe_input_count": _safe_int(
                terminal_release_gate_input_dependency.get("unsafe_input_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_dependency.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "dependency_status": str(row.get("dependency_status") or ""),
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "missing": bool(row.get("missing")),
                    "requires_external_input": bool(row.get("requires_external_input")),
                    "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                    "requires_source_cache_refresh": bool(
                        row.get("requires_source_cache_refresh")
                    ),
                    "json_only_resolution_available": bool(
                        row.get("json_only_resolution_available")
                    ),
                    "terminal_guard_required_after_input": bool(
                        row.get("terminal_guard_required_after_input")
                    ),
                    "release_claim_allowed_after_input": bool(
                        row.get("release_claim_allowed_after_input")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(terminal_release_gate_input_dependency.get("rows"))
                )
            ],
            "execution_policy": str(
                terminal_release_gate_input_dependency.get("execution_policy") or ""
            ),
            "fail_closed": bool(terminal_release_gate_input_dependency.get("fail_closed")),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_dependency.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_dependency.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_dependency.get("not_release_evidence")
            ),
        },
        "release_gate_post_input_refresh_plan_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_plan.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_plan.get("artifact_role") or ""
            ),
            "plan_status": str(
                terminal_release_gate_post_input_refresh_plan.get("plan_status") or ""
            ),
            "plan_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get("plan_row_count")
            ),
            "computed_plan_row_count": _safe_int(
                computed_release_gate_post_input_refresh_plan.get("plan_row_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get("input_ids")
            )[:50],
            "blocked_input_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get("blocked_input_count")
            ),
            "computed_blocked_input_count": _safe_int(
                computed_release_gate_post_input_refresh_plan.get("blocked_input_count")
            ),
            "blocked_input_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get("blocked_input_ids")
            )[:50],
            "external_input_plan_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get("external_input_plan_count")
            ),
            "manual_gpu_evidence_plan_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get(
                    "manual_gpu_evidence_plan_count"
                )
            ),
            "source_cache_refresh_plan_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get(
                    "source_cache_refresh_plan_count"
                )
            ),
            "required_refresh_command_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get(
                    "required_refresh_command_ids"
                )
            )[:20],
            "terminal_guard_command_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get(
                    "terminal_guard_command_ids"
                )
            )[:10],
            "post_refresh_required_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get(
                    "post_refresh_required_artifact_ids"
                )
            )[:20],
            "unsafe_plan_count": _safe_int(
                terminal_release_gate_post_input_refresh_plan.get("unsafe_plan_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_post_input_refresh_plan.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "plan_status": str(row.get("plan_status") or ""),
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "related_action_ids": _strings(row.get("related_action_ids"))[:50],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "input_missing": bool(row.get("input_missing")),
                    "external_input_required_before_refresh": bool(
                        row.get("external_input_required_before_refresh")
                    ),
                    "manual_gpu_evidence_required_before_refresh": bool(
                        row.get("manual_gpu_evidence_required_before_refresh")
                    ),
                    "source_cache_refresh_input": bool(row.get("source_cache_refresh_input")),
                    "required_refresh_command_ids": _strings(
                        row.get("required_refresh_command_ids")
                    )[:20],
                    "terminal_guard_command_ids": _strings(
                        row.get("terminal_guard_command_ids")
                    )[:10],
                    "post_refresh_required_artifact_ids": _strings(
                        row.get("post_refresh_required_artifact_ids")
                    )[:20],
                    "terminal_guard_required_after_refresh": bool(
                        row.get("terminal_guard_required_after_refresh")
                    ),
                    "safe_to_auto_start_refresh": bool(
                        row.get("safe_to_auto_start_refresh")
                    ),
                    "release_claim_allowed_after_refresh": bool(
                        row.get("release_claim_allowed_after_refresh")
                    ),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(terminal_release_gate_post_input_refresh_plan.get("rows"))
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_plan.get("execution_policy") or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_plan.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_plan.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_plan.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_plan.get("not_release_evidence")
            ),
        },
        "release_gate_input_detection_source_summary": {
            "roadmap": str(terminal_release_gate_input_detection_source.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_detection_source.get("artifact_role") or ""
            ),
            "detection_status": str(
                terminal_release_gate_input_detection_source.get("detection_status") or ""
            ),
            "detection_row_count": _safe_int(
                terminal_release_gate_input_detection_source.get("detection_row_count")
            ),
            "computed_detection_row_count": _safe_int(
                computed_release_gate_input_detection_source.get("detection_row_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_input_detection_source.get("input_ids")
            )[:50],
            "missing_or_unverified_input_count": _safe_int(
                terminal_release_gate_input_detection_source.get(
                    "missing_or_unverified_input_count"
                )
            ),
            "computed_missing_or_unverified_input_count": _safe_int(
                computed_release_gate_input_detection_source.get(
                    "missing_or_unverified_input_count"
                )
            ),
            "missing_or_unverified_input_ids": _strings(
                terminal_release_gate_input_detection_source.get(
                    "missing_or_unverified_input_ids"
                )
            )[:50],
            "detected_input_count": _safe_int(
                terminal_release_gate_input_detection_source.get("detected_input_count")
            ),
            "external_input_detector_count": _safe_int(
                terminal_release_gate_input_detection_source.get(
                    "external_input_detector_count"
                )
            ),
            "manual_gpu_detector_count": _safe_int(
                terminal_release_gate_input_detection_source.get("manual_gpu_detector_count")
            ),
            "source_cache_refresh_detector_count": _safe_int(
                terminal_release_gate_input_detection_source.get(
                    "source_cache_refresh_detector_count"
                )
            ),
            "unsafe_detector_count": _safe_int(
                terminal_release_gate_input_detection_source.get("unsafe_detector_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_detection_source.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "detection_status": str(row.get("detection_status") or ""),
                    "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                    "required_refresh_command_ids": _strings(
                        row.get("required_refresh_command_ids")
                    )[:20],
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "input_missing": bool(row.get("input_missing")),
                    "detected": bool(row.get("detected")),
                    "requires_external_input": bool(row.get("requires_external_input")),
                    "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                    "requires_source_cache_refresh": bool(
                        row.get("requires_source_cache_refresh")
                    ),
                    "terminal_guard_required_after_detection": bool(
                        row.get("terminal_guard_required_after_detection")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed_after_detection": bool(
                        row.get("release_claim_allowed_after_detection")
                    ),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(terminal_release_gate_input_detection_source.get("rows"))
                )
            ],
            "execution_policy": str(
                terminal_release_gate_input_detection_source.get("execution_policy") or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_input_detection_source.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_detection_source.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_detection_source.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_detection_source.get("not_release_evidence")
            ),
        },
        "release_gate_input_acceptance_criteria_summary": {
            "roadmap": str(terminal_release_gate_input_acceptance_criteria.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_acceptance_criteria.get("artifact_role") or ""
            ),
            "acceptance_status": str(
                terminal_release_gate_input_acceptance_criteria.get("acceptance_status") or ""
            ),
            "acceptance_row_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get("acceptance_row_count")
            ),
            "computed_acceptance_row_count": _safe_int(
                computed_release_gate_input_acceptance_criteria.get("acceptance_row_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_input_acceptance_criteria.get("input_ids")
            )[:50],
            "accepted_input_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get("accepted_input_count")
            ),
            "computed_accepted_input_count": _safe_int(
                computed_release_gate_input_acceptance_criteria.get("accepted_input_count")
            ),
            "accepted_input_ids": _strings(
                terminal_release_gate_input_acceptance_criteria.get("accepted_input_ids")
            )[:50],
            "unsatisfied_input_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get("unsatisfied_input_count")
            ),
            "computed_unsatisfied_input_count": _safe_int(
                computed_release_gate_input_acceptance_criteria.get("unsatisfied_input_count")
            ),
            "unsatisfied_input_ids": _strings(
                terminal_release_gate_input_acceptance_criteria.get("unsatisfied_input_ids")
            )[:50],
            "external_input_acceptance_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get(
                    "external_input_acceptance_count"
                )
            ),
            "manual_gpu_acceptance_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get(
                    "manual_gpu_acceptance_count"
                )
            ),
            "source_cache_refresh_acceptance_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get(
                    "source_cache_refresh_acceptance_count"
                )
            ),
            "unsafe_acceptance_count": _safe_int(
                terminal_release_gate_input_acceptance_criteria.get("unsafe_acceptance_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_acceptance_criteria.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "acceptance_status": str(row.get("acceptance_status") or ""),
                    "acceptance_criteria_ids": _strings(
                        row.get("acceptance_criteria_ids")
                    )[:20],
                    "required_evidence_artifact_ids": _strings(
                        row.get("required_evidence_artifact_ids")
                    )[:20],
                    "detector_artifact_ids": _strings(row.get("detector_artifact_ids"))[:20],
                    "required_refresh_command_ids": _strings(
                        row.get("required_refresh_command_ids")
                    )[:20],
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "input_missing": bool(row.get("input_missing")),
                    "detected": bool(row.get("detected")),
                    "accepted": bool(row.get("accepted")),
                    "requires_external_input": bool(row.get("requires_external_input")),
                    "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                    "requires_source_cache_refresh": bool(
                        row.get("requires_source_cache_refresh")
                    ),
                    "terminal_guard_required_after_acceptance": bool(
                        row.get("terminal_guard_required_after_acceptance")
                    ),
                    "release_claim_allowed_after_acceptance": bool(
                        row.get("release_claim_allowed_after_acceptance")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_input_acceptance_criteria.get("rows")
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_input_acceptance_criteria.get("execution_policy")
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_input_acceptance_criteria.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_acceptance_criteria.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_acceptance_criteria.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_acceptance_criteria.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_input_refresh_readiness_summary": {
            "roadmap": str(terminal_release_gate_input_refresh_readiness.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_refresh_readiness.get("artifact_role") or ""
            ),
            "refresh_readiness_status": str(
                terminal_release_gate_input_refresh_readiness.get("refresh_readiness_status")
                or ""
            ),
            "refresh_row_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get("refresh_row_count")
            ),
            "computed_refresh_row_count": _safe_int(
                computed_release_gate_input_refresh_readiness.get("refresh_row_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_input_refresh_readiness.get("input_ids")
            )[:50],
            "accepted_input_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get("accepted_input_count")
            ),
            "computed_accepted_input_count": _safe_int(
                computed_release_gate_input_refresh_readiness.get("accepted_input_count")
            ),
            "accepted_input_ids": _strings(
                terminal_release_gate_input_refresh_readiness.get("accepted_input_ids")
            )[:50],
            "refresh_ready_input_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get(
                    "refresh_ready_input_count"
                )
            ),
            "computed_refresh_ready_input_count": _safe_int(
                computed_release_gate_input_refresh_readiness.get(
                    "refresh_ready_input_count"
                )
            ),
            "refresh_ready_input_ids": _strings(
                terminal_release_gate_input_refresh_readiness.get(
                    "refresh_ready_input_ids"
                )
            )[:50],
            "blocked_refresh_input_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get(
                    "blocked_refresh_input_count"
                )
            ),
            "computed_blocked_refresh_input_count": _safe_int(
                computed_release_gate_input_refresh_readiness.get(
                    "blocked_refresh_input_count"
                )
            ),
            "blocked_refresh_input_ids": _strings(
                terminal_release_gate_input_refresh_readiness.get(
                    "blocked_refresh_input_ids"
                )
            )[:50],
            "external_input_refresh_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get(
                    "external_input_refresh_count"
                )
            ),
            "manual_gpu_refresh_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get("manual_gpu_refresh_count")
            ),
            "source_cache_refresh_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get(
                    "source_cache_refresh_count"
                )
            ),
            "unsafe_refresh_count": _safe_int(
                terminal_release_gate_input_refresh_readiness.get("unsafe_refresh_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_refresh_readiness.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "refresh_readiness_status": str(
                        row.get("refresh_readiness_status") or ""
                    ),
                    "accepted": bool(row.get("accepted")),
                    "input_missing": bool(row.get("input_missing")),
                    "detected": bool(row.get("detected")),
                    "refresh_ready": bool(row.get("refresh_ready")),
                    "blocked_refresh": bool(row.get("blocked_refresh")),
                    "acceptance_status": str(row.get("acceptance_status") or ""),
                    "plan_status": str(row.get("plan_status") or ""),
                    "acceptance_criteria_ids": _strings(
                        row.get("acceptance_criteria_ids")
                    )[:20],
                    "required_evidence_artifact_ids": _strings(
                        row.get("required_evidence_artifact_ids")
                    )[:20],
                    "required_refresh_command_ids": _strings(
                        row.get("required_refresh_command_ids")
                    )[:20],
                    "terminal_guard_command_ids": _strings(
                        row.get("terminal_guard_command_ids")
                    )[:10],
                    "post_refresh_required_artifact_ids": _strings(
                        row.get("post_refresh_required_artifact_ids")
                    )[:20],
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "requires_external_input": bool(row.get("requires_external_input")),
                    "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                    "requires_source_cache_refresh": bool(
                        row.get("requires_source_cache_refresh")
                    ),
                    "terminal_guard_required_after_refresh": bool(
                        row.get("terminal_guard_required_after_refresh")
                    ),
                    "safe_to_auto_start_refresh": bool(
                        row.get("safe_to_auto_start_refresh")
                    ),
                    "release_claim_allowed_after_refresh": bool(
                        row.get("release_claim_allowed_after_refresh")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_input_refresh_readiness.get("rows")
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_input_refresh_readiness.get("execution_policy")
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_input_refresh_readiness.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_refresh_readiness.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_refresh_readiness.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_refresh_readiness.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_input_refresh_blocker_summary": {
            "roadmap": str(terminal_release_gate_input_refresh_blocker.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_refresh_blocker.get("artifact_role") or ""
            ),
            "blocker_status": str(
                terminal_release_gate_input_refresh_blocker.get("blocker_status") or ""
            ),
            "blocker_row_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get("blocker_row_count")
            ),
            "computed_blocker_row_count": _safe_int(
                computed_release_gate_input_refresh_blocker.get("blocker_row_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_input_refresh_blocker.get("input_ids")
            )[:50],
            "blocked_input_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get("blocked_input_count")
            ),
            "computed_blocked_input_count": _safe_int(
                computed_release_gate_input_refresh_blocker.get("blocked_input_count")
            ),
            "blocked_input_ids": _strings(
                terminal_release_gate_input_refresh_blocker.get("blocked_input_ids")
            )[:50],
            "refresh_ready_input_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get("refresh_ready_input_count")
            ),
            "computed_refresh_ready_input_count": _safe_int(
                computed_release_gate_input_refresh_blocker.get("refresh_ready_input_count")
            ),
            "refresh_ready_input_ids": _strings(
                terminal_release_gate_input_refresh_blocker.get("refresh_ready_input_ids")
            )[:50],
            "missing_input_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "missing_input_blocker_count"
                )
            ),
            "undetected_input_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "undetected_input_blocker_count"
                )
            ),
            "unaccepted_input_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "unaccepted_input_blocker_count"
                )
            ),
            "external_input_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "external_input_blocker_count"
                )
            ),
            "manual_gpu_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get("manual_gpu_blocker_count")
            ),
            "source_cache_refresh_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "source_cache_refresh_blocker_count"
                )
            ),
            "terminal_guard_required_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get(
                    "terminal_guard_required_count"
                )
            ),
            "unsafe_blocker_count": _safe_int(
                terminal_release_gate_input_refresh_blocker.get("unsafe_blocker_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_refresh_blocker.get("unsafe_input_ids")
            )[:50],
            "rows": [
                {
                    "input_id": str(row.get("input_id") or ""),
                    "input_kind": str(row.get("input_kind") or ""),
                    "blocker_status": str(row.get("blocker_status") or ""),
                    "blocked_reason_ids": _strings(row.get("blocked_reason_ids"))[:20],
                    "blocked_refresh": bool(row.get("blocked_refresh")),
                    "refresh_ready": bool(row.get("refresh_ready")),
                    "accepted": bool(row.get("accepted")),
                    "detected": bool(row.get("detected")),
                    "input_missing": bool(row.get("input_missing")),
                    "requires_external_input": bool(row.get("requires_external_input")),
                    "requires_manual_gpu": bool(row.get("requires_manual_gpu")),
                    "requires_source_cache_refresh": bool(
                        row.get("requires_source_cache_refresh")
                    ),
                    "required_refresh_command_ids": _strings(
                        row.get("required_refresh_command_ids")
                    )[:20],
                    "terminal_guard_command_ids": _strings(
                        row.get("terminal_guard_command_ids")
                    )[:10],
                    "related_gate_ids": _strings(row.get("related_gate_ids"))[:20],
                    "affected_family_ids": _strings(row.get("affected_family_ids"))[:20],
                    "terminal_guard_required_after_refresh": bool(
                        row.get("terminal_guard_required_after_refresh")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_input_refresh_blocker.get("rows")
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_input_refresh_blocker.get("execution_policy") or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_input_refresh_blocker.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_refresh_blocker.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_refresh_blocker.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_refresh_blocker.get("not_release_evidence")
            ),
        },
        "release_gate_input_lifecycle_summary": {
            "roadmap": str(terminal_release_gate_input_lifecycle.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_input_lifecycle.get("artifact_role") or ""
            ),
            "lifecycle_status": str(
                terminal_release_gate_input_lifecycle.get("lifecycle_status") or ""
            ),
            "computed_lifecycle_status": str(
                computed_release_gate_input_lifecycle.get("lifecycle_status") or ""
            ),
            "input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("input_count")
            ),
            "computed_input_count": _safe_int(
                computed_release_gate_input_lifecycle.get("input_count")
            ),
            "input_ids": _strings(
                terminal_release_gate_input_lifecycle.get("input_ids")
            )[:50],
            "lifecycle_stage_counts": dict(
                _mapping(
                    terminal_release_gate_input_lifecycle.get(
                        "lifecycle_stage_counts"
                    )
                )
            ),
            "computed_lifecycle_stage_counts": dict(
                _mapping(
                    computed_release_gate_input_lifecycle.get(
                        "lifecycle_stage_counts"
                    )
                )
            ),
            "detected_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("detected_input_count")
            ),
            "computed_detected_input_count": _safe_int(
                computed_release_gate_input_lifecycle.get("detected_input_count")
            ),
            "accepted_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("accepted_input_count")
            ),
            "computed_accepted_input_count": _safe_int(
                computed_release_gate_input_lifecycle.get("accepted_input_count")
            ),
            "refresh_ready_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("refresh_ready_input_count")
            ),
            "computed_refresh_ready_input_count": _safe_int(
                computed_release_gate_input_lifecycle.get("refresh_ready_input_count")
            ),
            "blocked_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("blocked_input_count")
            ),
            "computed_blocked_input_count": _safe_int(
                computed_release_gate_input_lifecycle.get("blocked_input_count")
            ),
            "blocked_input_ids": _strings(
                terminal_release_gate_input_lifecycle.get("blocked_input_ids")
            )[:50],
            "external_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("external_input_count")
            ),
            "manual_gpu_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("manual_gpu_input_count")
            ),
            "source_cache_refresh_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get(
                    "source_cache_refresh_input_count"
                )
            ),
            "unsafe_input_count": _safe_int(
                terminal_release_gate_input_lifecycle.get("unsafe_input_count")
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_input_lifecycle.get("unsafe_input_ids")
            )[:50],
            "execution_policy": str(
                terminal_release_gate_input_lifecycle.get("execution_policy") or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_input_lifecycle.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_input_lifecycle.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_input_lifecycle.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_release_gate_input_lifecycle.get("not_release_evidence")
            ),
        },
        "external_input_release_gate_alignment_summary": {
            "roadmap": str(
                terminal_external_input_release_gate_alignment.get("roadmap") or ""
            ),
            "artifact_role": str(
                terminal_external_input_release_gate_alignment.get("artifact_role")
                or ""
            ),
            "alignment_status": str(
                terminal_external_input_release_gate_alignment.get("alignment_status")
                or ""
            ),
            "computed_alignment_status": str(
                computed_external_input_release_gate_alignment.get("alignment_status")
                or ""
            ),
            "alignment_ok": bool(
                terminal_external_input_release_gate_alignment.get("alignment_ok")
            ),
            "computed_alignment_ok": bool(
                computed_external_input_release_gate_alignment.get("alignment_ok")
            ),
            "external_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "external_input_count"
                )
            ),
            "computed_external_input_count": _safe_int(
                computed_external_input_release_gate_alignment.get(
                    "external_input_count"
                )
            ),
            "external_input_ids": _strings(
                terminal_external_input_release_gate_alignment.get("external_input_ids")
            )[:50],
            "release_gate_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "release_gate_input_count"
                )
            ),
            "computed_release_gate_input_count": _safe_int(
                computed_external_input_release_gate_alignment.get(
                    "release_gate_input_count"
                )
            ),
            "release_gate_input_ids": _strings(
                terminal_external_input_release_gate_alignment.get(
                    "release_gate_input_ids"
                )
            )[:50],
            "external_release_gate_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "external_release_gate_input_count"
                )
            ),
            "manual_gpu_release_gate_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "manual_gpu_release_gate_input_count"
                )
            ),
            "source_cache_refresh_release_gate_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "source_cache_refresh_release_gate_input_count"
                )
            ),
            "non_external_release_gate_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "non_external_release_gate_input_count"
                )
            ),
            "non_external_release_gate_input_ids": _strings(
                terminal_external_input_release_gate_alignment.get(
                    "non_external_release_gate_input_ids"
                )
            )[:50],
            "external_missing_from_release_gate_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "external_missing_from_release_gate_count"
                )
            ),
            "release_external_missing_from_transition_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "release_external_missing_from_transition_count"
                )
            ),
            "blocked_input_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "blocked_input_count"
                )
            ),
            "unsafe_alignment_count": _safe_int(
                terminal_external_input_release_gate_alignment.get(
                    "unsafe_alignment_count"
                )
            ),
            "unsafe_input_ids": _strings(
                terminal_external_input_release_gate_alignment.get("unsafe_input_ids")
            )[:50],
            "execution_policy": str(
                terminal_external_input_release_gate_alignment.get("execution_policy")
                or ""
            ),
            "fail_closed": bool(
                terminal_external_input_release_gate_alignment.get("fail_closed")
            ),
            "safe_to_auto_start": bool(
                terminal_external_input_release_gate_alignment.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_external_input_release_gate_alignment.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_external_input_release_gate_alignment.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_command_surface_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_command_surface.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_command_surface.get("artifact_role") or ""
            ),
            "command_surface_status": str(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "command_surface_status"
                )
                or ""
            ),
            "command_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "command_row_count"
                )
            ),
            "computed_command_row_count": _safe_int(
                computed_release_gate_post_input_refresh_command_surface.get(
                    "command_row_count"
                )
            ),
            "required_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "required_command_count"
                )
            ),
            "computed_required_command_count": _safe_int(
                computed_release_gate_post_input_refresh_command_surface.get(
                    "required_command_count"
                )
            ),
            "required_command_ids": _strings(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "required_command_ids"
                )
            )[:50],
            "json_refresh_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "json_refresh_command_count"
                )
            ),
            "terminal_guard_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "terminal_guard_command_count"
                )
            ),
            "blocked_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "blocked_command_count"
                )
            ),
            "computed_blocked_command_count": _safe_int(
                computed_release_gate_post_input_refresh_command_surface.get(
                    "blocked_command_count"
                )
            ),
            "blocked_command_ids": _strings(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "blocked_command_ids"
                )
            )[:50],
            "ready_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "ready_command_count"
                )
            ),
            "computed_ready_command_count": _safe_int(
                computed_release_gate_post_input_refresh_command_surface.get(
                    "ready_command_count"
                )
            ),
            "ready_command_ids": _strings(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "ready_command_ids"
                )
            )[:50],
            "blocked_input_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "blocked_input_count"
                )
            ),
            "blocked_input_ids": _strings(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "blocked_input_ids"
                )
            )[:50],
            "unsafe_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "unsafe_command_count"
                )
            ),
            "unsafe_command_ids": _strings(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "unsafe_command_ids"
                )
            )[:50],
            "rows": [
                {
                    "command_id": str(row.get("command_id") or ""),
                    "command_order": _safe_int(row.get("command_order")),
                    "command_kind": str(row.get("command_kind") or ""),
                    "command_status": str(row.get("command_status") or ""),
                    "related_input_ids": _strings(row.get("related_input_ids"))[:50],
                    "blocked_input_ids": _strings(row.get("blocked_input_ids"))[:50],
                    "blocked_input_count": _safe_int(row.get("blocked_input_count")),
                    "refresh_ready_input_count": _safe_int(
                        row.get("refresh_ready_input_count")
                    ),
                    "terminal_guard_command": bool(row.get("terminal_guard_command")),
                    "required_after_input_acceptance": bool(
                        row.get("required_after_input_acceptance")
                    ),
                    "blocked_until_input_acceptance": bool(
                        row.get("blocked_until_input_acceptance")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_post_input_refresh_command_surface.get(
                            "rows"
                        )
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_command_surface.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_sequence_integrity_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_sequence_integrity.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_sequence_integrity.get("artifact_role") or ""
            ),
            "sequence_status": str(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "sequence_status"
                )
                or ""
            ),
            "sequence_ok": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "sequence_ok"
                )
            ),
            "computed_sequence_ok": bool(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "sequence_ok"
                )
            ),
            "expected_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "expected_command_count"
                )
            ),
            "computed_expected_command_count": _safe_int(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "expected_command_count"
                )
            ),
            "observed_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "observed_command_count"
                )
            ),
            "computed_observed_command_count": _safe_int(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "observed_command_count"
                )
            ),
            "expected_command_ids": _strings(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "expected_command_ids"
                )
            )[:50],
            "observed_command_ids": _strings(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "observed_command_ids"
                )
            )[:50],
            "missing_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "missing_command_count"
                )
            ),
            "unexpected_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "unexpected_command_count"
                )
            ),
            "duplicate_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "duplicate_command_count"
                )
            ),
            "order_matches_expected": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "order_matches_expected"
                )
            ),
            "terminal_guard_tail_ok": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "terminal_guard_tail_ok"
                )
            ),
            "terminal_guard_command_ids": _strings(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "terminal_guard_command_ids"
                )
            )[:50],
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "blocked_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "blocked_command_count"
                )
            ),
            "computed_blocked_command_count": _safe_int(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "blocked_command_count"
                )
            ),
            "ready_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "ready_command_count"
                )
            ),
            "computed_ready_command_count": _safe_int(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "ready_command_count"
                )
            ),
            "unsafe_sequence_count": _safe_int(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "unsafe_sequence_count"
                )
            ),
            "computed_unsafe_sequence_count": _safe_int(
                computed_release_gate_post_input_refresh_sequence_integrity.get(
                    "unsafe_sequence_count"
                )
            ),
            "unsafe_command_ids": _strings(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "unsafe_command_ids"
                )
            )[:50],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_sequence_integrity.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_terminal_guard_dependency_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_terminal_guard_dependency.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get("artifact_role") or ""
            ),
            "dependency_status": str(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "dependency_status"
                )
                or ""
            ),
            "dependency_ok": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "dependency_ok"
                )
            ),
            "computed_dependency_ok": bool(
                computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "dependency_ok"
                )
            ),
            "terminal_guard_required": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_required"
                )
            ),
            "terminal_guard_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_command_count"
                )
            ),
            "computed_terminal_guard_command_count": _safe_int(
                computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_command_count"
                )
            ),
            "expected_terminal_guard_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "expected_terminal_guard_command_count"
                )
            ),
            "terminal_guard_command_ids": _strings(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_command_ids"
                )
            )[:50],
            "expected_terminal_guard_command_ids": _strings(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "expected_terminal_guard_command_ids"
                )
            )[:50],
            "terminal_guard_command_orders": [
                _safe_int(item)
                for item in _list(
                    terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                        "terminal_guard_command_orders"
                    )
                )
            ][:10],
            "terminal_self_check_required": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_self_check_required"
                )
            ),
            "release_guard_required": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "release_guard_required"
                )
            ),
            "terminal_guard_tail_ok": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "terminal_guard_tail_ok"
                )
            ),
            "all_json_refresh_commands_before_terminal_guard": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "all_json_refresh_commands_before_terminal_guard"
                )
            ),
            "json_refresh_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "json_refresh_command_count"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "blocked_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "blocked_command_count"
                )
            ),
            "computed_blocked_command_count": _safe_int(
                computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "blocked_command_count"
                )
            ),
            "ready_command_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "ready_command_count"
                )
            ),
            "computed_ready_command_count": _safe_int(
                computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "ready_command_count"
                )
            ),
            "unsafe_dependency_count": _safe_int(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "unsafe_dependency_count"
                )
            ),
            "computed_unsafe_dependency_count": _safe_int(
                computed_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "unsafe_dependency_count"
                )
            ),
            "unsafe_command_ids": _strings(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "unsafe_command_ids"
                )
            )[:50],
            "rows": [
                {
                    "command_id": str(row.get("command_id") or ""),
                    "dependency_order": _safe_int(row.get("dependency_order")),
                    "command_order": _safe_int(row.get("command_order")),
                    "guard_kind": str(row.get("guard_kind") or ""),
                    "depends_on_json_refresh_sequence": bool(
                        row.get("depends_on_json_refresh_sequence")
                    ),
                    "required_after_json_refresh": bool(
                        row.get("required_after_json_refresh")
                    ),
                    "blocked_until_input_acceptance": bool(
                        row.get("blocked_until_input_acceptance")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                            "rows"
                        )
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_terminal_guard_dependency.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_artifact_coverage_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_artifact_coverage.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_artifact_coverage.get("artifact_role") or ""
            ),
            "coverage_status": str(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "coverage_status"
                )
                or ""
            ),
            "coverage_ok": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "coverage_ok"
                )
            ),
            "computed_coverage_ok": bool(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "coverage_ok"
                )
            ),
            "required_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "required_artifact_count"
                )
            ),
            "computed_required_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "required_artifact_count"
                )
            ),
            "required_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "required_artifact_ids"
                )
            )[:50],
            "input_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "input_row_count"
                )
            ),
            "computed_input_row_count": _safe_int(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "input_row_count"
                )
            ),
            "covered_input_count": _safe_int(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "covered_input_count"
                )
            ),
            "computed_covered_input_count": _safe_int(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "covered_input_count"
                )
            ),
            "covered_input_ids": _strings(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "covered_input_ids"
                )
            )[:50],
            "missing_coverage_input_count": _safe_int(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "missing_coverage_input_count"
                )
            ),
            "computed_missing_coverage_input_count": _safe_int(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "missing_coverage_input_count"
                )
            ),
            "missing_coverage_input_ids": _strings(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "missing_coverage_input_ids"
                )
            )[:50],
            "readiness_artifact_required": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "readiness_artifact_required"
                )
            ),
            "terminal_artifact_required": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "terminal_artifact_required"
                )
            ),
            "release_guard_artifact_required": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "release_guard_artifact_required"
                )
            ),
            "terminal_guard_dependency_ok": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "terminal_guard_dependency_ok"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "unsafe_artifact_coverage_count": _safe_int(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "unsafe_artifact_coverage_count"
                )
            ),
            "computed_unsafe_artifact_coverage_count": _safe_int(
                computed_release_gate_post_input_refresh_artifact_coverage.get(
                    "unsafe_artifact_coverage_count"
                )
            ),
            "unsafe_input_ids": _strings(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "unsafe_input_ids"
                )
            )[:50],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_artifact_coverage.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_command_artifact_link_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_command_artifact_link.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get("artifact_role") or ""
            ),
            "link_status": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "link_status"
                )
                or ""
            ),
            "link_ok": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "link_ok"
                )
            ),
            "computed_link_ok": bool(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "link_ok"
                )
            ),
            "command_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_row_count"
                )
            ),
            "computed_command_row_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_row_count"
                )
            ),
            "required_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "required_artifact_count"
                )
            ),
            "computed_required_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "required_artifact_count"
                )
            ),
            "required_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "required_artifact_ids"
                )
            )[:50],
            "linked_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "linked_artifact_count"
                )
            ),
            "computed_linked_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "linked_artifact_count"
                )
            ),
            "linked_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "linked_artifact_ids"
                )
            )[:50],
            "missing_link_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "missing_link_artifact_count"
                )
            ),
            "computed_missing_link_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "missing_link_artifact_count"
                )
            ),
            "missing_link_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "missing_link_artifact_ids"
                )
            )[:50],
            "extra_link_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "extra_link_artifact_count"
                )
            ),
            "computed_extra_link_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "extra_link_artifact_count"
                )
            ),
            "extra_link_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "extra_link_artifact_ids"
                )
            )[:50],
            "command_artifact_link_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_artifact_link_count"
                )
            ),
            "computed_command_artifact_link_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_artifact_link_count"
                )
            ),
            "command_ids_with_artifacts": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_ids_with_artifacts"
                )
            )[:50],
            "command_ids_without_artifacts": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "command_ids_without_artifacts"
                )
            )[:50],
            "readiness_command_id": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "readiness_command_id"
                )
                or ""
            ),
            "terminal_command_id": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "terminal_command_id"
                )
                or ""
            ),
            "release_guard_command_id": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "release_guard_command_id"
                )
                or ""
            ),
            "artifact_coverage_ok": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "artifact_coverage_ok"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "unsafe_link_count": _safe_int(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "unsafe_link_count"
                )
            ),
            "computed_unsafe_link_count": _safe_int(
                computed_release_gate_post_input_refresh_command_artifact_link.get(
                    "unsafe_link_count"
                )
            ),
            "unsafe_command_ids": _strings(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "unsafe_command_ids"
                )
            )[:50],
            "rows": [
                {
                    "command_id": str(row.get("command_id") or ""),
                    "command_order": _safe_int(row.get("command_order")),
                    "command_kind": str(row.get("command_kind") or ""),
                    "output_artifact_ids": _strings(row.get("output_artifact_ids"))[:20],
                    "output_artifact_count": _safe_int(row.get("output_artifact_count")),
                    "produces_required_post_refresh_artifact": bool(
                        row.get("produces_required_post_refresh_artifact")
                    ),
                    "blocked_until_input_acceptance": bool(
                        row.get("blocked_until_input_acceptance")
                    ),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_post_input_refresh_command_artifact_link.get(
                            "rows"
                        )
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_command_artifact_link.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_guard_consumption_summary": {
            "roadmap": str(terminal_release_gate_post_input_refresh_guard_consumption.get("roadmap") or ""),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get("artifact_role") or ""
            ),
            "consumption_status": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "consumption_status"
                )
                or ""
            ),
            "consumption_ok": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "consumption_ok"
                )
            ),
            "computed_consumption_ok": bool(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "consumption_ok"
                )
            ),
            "guard_command_id": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "guard_command_id"
                )
                or ""
            ),
            "required_input_artifact_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "required_input_artifact_count"
                )
            ),
            "computed_required_input_artifact_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "required_input_artifact_count"
                )
            ),
            "required_input_artifact_ids": _strings(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "required_input_artifact_ids"
                )
            )[:20],
            "produced_guard_artifact_id": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "produced_guard_artifact_id"
                )
                or ""
            ),
            "input_artifacts_consumed": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "input_artifacts_consumed"
                )
            ),
            "guard_artifact_produced": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "guard_artifact_produced"
                )
            ),
            "required_consumed_summary_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "required_consumed_summary_count"
                )
            ),
            "computed_required_consumed_summary_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "required_consumed_summary_count"
                )
            ),
            "required_consumed_summary_ids": _strings(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "required_consumed_summary_ids"
                )
            )[:50],
            "present_consumed_summary_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "present_consumed_summary_count"
                )
            ),
            "computed_present_consumed_summary_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "present_consumed_summary_count"
                )
            ),
            "missing_consumed_summary_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "missing_consumed_summary_count"
                )
            ),
            "computed_missing_consumed_summary_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "missing_consumed_summary_count"
                )
            ),
            "missing_consumed_summary_ids": _strings(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "missing_consumed_summary_ids"
                )
            )[:50],
            "unsafe_consumed_summary_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "unsafe_consumed_summary_count"
                )
            ),
            "computed_unsafe_consumed_summary_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_consumption.get(
                    "unsafe_consumed_summary_count"
                )
            ),
            "unsafe_consumed_summary_ids": _strings(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "unsafe_consumed_summary_ids"
                )
            )[:50],
            "terminal_lineage_required": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "terminal_lineage_required"
                )
            ),
            "terminal_lineage_summary_id": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "terminal_lineage_summary_id"
                )
                or ""
            ),
            "terminal_lineage_ok": bool(terminal_roadmap_lineage.get("lineage_ok")),
            "command_artifact_link_ok": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "command_artifact_link_ok"
                )
            ),
            "artifact_coverage_ok": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "artifact_coverage_ok"
                )
            ),
            "terminal_guard_dependency_ok": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "terminal_guard_dependency_ok"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "rows": [
                {
                    "summary_id": str(row.get("summary_id") or ""),
                    "consumption_stage": str(row.get("consumption_stage") or ""),
                    "required_for_guard": bool(row.get("required_for_guard")),
                    "present": bool(row.get("present")),
                    "fail_closed": bool(row.get("fail_closed")),
                    "terminal_only": bool(row.get("terminal_only")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "unsafe": bool(row.get("unsafe")),
                    **(
                        {
                            "manifest_ok": bool(row.get("manifest_ok")),
                            "runner_ready": bool(row.get("runner_ready")),
                            "execution_ok": bool(row.get("execution_ok")),
                            "row_execution_consistent": bool(
                                row.get("row_execution_consistent")
                            ),
                            "expected_command_count": _safe_int(
                                row.get("expected_command_count")
                            ),
                            "row_count": _safe_int(row.get("row_count")),
                            "executed_row_count": _safe_int(
                                row.get("executed_row_count")
                            ),
                            "failed_row_count": _safe_int(row.get("failed_row_count")),
                            "missing_output_row_count": _safe_int(
                                row.get("missing_output_row_count")
                            ),
                            "row_forbidden_heavy_flag_count": _safe_int(
                                row.get("row_forbidden_heavy_flag_count")
                            ),
                            "unsafe_row_count": _safe_int(row.get("unsafe_row_count")),
                        }
                        if str(row.get("summary_id") or "")
                        == "external_input_json_refresh_runner_manifest_summary"
                        else {}
                    ),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_post_input_refresh_guard_consumption.get(
                            "rows"
                        )
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_guard_consumption.get(
                    "not_release_evidence"
                )
            ),
        },
        "release_gate_post_input_refresh_guard_report_acceptance_summary": {
            "roadmap": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get("roadmap") or ""
            ),
            "artifact_role": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get("artifact_role") or ""
            ),
            "acceptance_status": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_status"
                )
                or ""
            ),
            "acceptance_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_ok"
                )
            ),
            "computed_acceptance_ok": bool(
                computed_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_ok"
                )
            ),
            "guard_command_id": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_command_id"
                )
                or ""
            ),
            "guard_report_artifact_id": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_report_artifact_id"
                )
                or ""
            ),
            "expected_report_status": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_report_status"
                )
                or ""
            ),
            "expected_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_ok"
                )
            ),
            "expected_failure_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "expected_failure_count"
                )
            ),
            "required_guard_report_field_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "required_guard_report_field_count"
                )
            ),
            "computed_required_guard_report_field_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "required_guard_report_field_count"
                )
            ),
            "required_guard_report_fields": _strings(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "required_guard_report_fields"
                )
            )[:50],
            "guard_consumption_ok": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_consumption_ok"
                )
            ),
            "input_artifacts_consumed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "input_artifacts_consumed"
                )
            ),
            "guard_artifact_produced": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "guard_artifact_produced"
                )
            ),
            "blocked_until_input_acceptance": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "blocked_until_input_acceptance"
                )
            ),
            "acceptance_row_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_row_count"
                )
            ),
            "computed_acceptance_row_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "acceptance_row_count"
                )
            ),
            "unsafe_acceptance_count": _safe_int(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "unsafe_acceptance_count"
                )
            ),
            "computed_unsafe_acceptance_count": _safe_int(
                computed_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "unsafe_acceptance_count"
                )
            ),
            "unsafe_acceptance_ids": _strings(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "unsafe_acceptance_ids"
                )
            )[:50],
            "rows": [
                {
                    "acceptance_id": str(row.get("acceptance_id") or ""),
                    "required_field_ids": _strings(row.get("required_field_ids"))[:20],
                    "required": bool(row.get("required")),
                    "expected_value_summary": str(row.get("expected_value_summary") or ""),
                    "present": bool(row.get("present")),
                    "fail_closed": bool(row.get("fail_closed")),
                    "not_release_evidence": bool(row.get("not_release_evidence")),
                    "safe_to_auto_start": bool(row.get("safe_to_auto_start")),
                    "release_claim_allowed": bool(row.get("release_claim_allowed")),
                    "unsafe": bool(row.get("unsafe")),
                }
                for row in (
                    _mapping(item)
                    for item in _list(
                        terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                            "rows"
                        )
                    )
                )
            ],
            "execution_policy": str(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "execution_policy"
                )
                or ""
            ),
            "fail_closed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "fail_closed"
                )
            ),
            "safe_to_auto_start": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "safe_to_auto_start"
                )
            ),
            "release_claim_allowed": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "release_claim_allowed"
                )
            ),
            "not_release_evidence": bool(
                terminal_release_gate_post_input_refresh_guard_report_acceptance.get(
                    "not_release_evidence"
                )
            ),
        },
        "manual_protected_gpu_command_surface_summary": {
            "roadmap": str(terminal_command_surface.get("roadmap") or ""),
            "artifact_role": str(terminal_command_surface.get("artifact_role") or ""),
            "surface_status": str(terminal_command_surface.get("surface_status") or ""),
            "source_artifact_count": _safe_int(terminal_command_surface.get("source_artifact_count")),
            "computed_source_artifact_count": _safe_int(
                computed_command_surface.get("source_artifact_count")
            ),
            "source_artifact_ids": _strings(
                terminal_command_surface.get("source_artifact_ids")
            )[:20],
            "command_surface_row_count": _safe_int(
                terminal_command_surface.get("command_surface_row_count")
            ),
            "computed_command_surface_row_count": _safe_int(
                computed_command_surface.get("command_surface_row_count")
            ),
            "manual_gpu_command_count": _safe_int(
                terminal_command_surface.get("manual_gpu_command_count")
            ),
            "protected_gpu_command_count": _safe_int(
                terminal_command_surface.get("protected_gpu_command_count")
            ),
            "dry_run_command_count": _safe_int(terminal_command_surface.get("dry_run_command_count")),
            "template_command_count": _safe_int(terminal_command_surface.get("template_command_count")),
            "ready_command_count": _safe_int(terminal_command_surface.get("ready_command_count")),
            "blocked_command_count": _safe_int(terminal_command_surface.get("blocked_command_count")),
            "completed_existing_command_count": _safe_int(
                terminal_command_surface.get("completed_existing_command_count")
            ),
            "rerun_blocked_without_new_axis_count": _safe_int(
                terminal_command_surface.get("rerun_blocked_without_new_axis_count")
            ),
            "requires_gpu_if_executed_count": _safe_int(
                terminal_command_surface.get("requires_gpu_if_executed_count")
            ),
            "manual_start_required_count": _safe_int(
                terminal_command_surface.get("manual_start_required_count")
            ),
            "release_relevant_command_count": _safe_int(
                terminal_command_surface.get("release_relevant_command_count")
            ),
            "diagnostic_only_command_count": _safe_int(
                terminal_command_surface.get("diagnostic_only_command_count")
            ),
            "release_claim_allowed_after_success_count": _safe_int(
                terminal_command_surface.get("release_claim_allowed_after_success_count")
            ),
            "unsafe_command_count": _safe_int(terminal_command_surface.get("unsafe_command_count")),
            "computed_unsafe_command_count": _safe_int(
                computed_command_surface.get("unsafe_command_count")
            ),
            "unsafe_command_ids": _strings(terminal_command_surface.get("unsafe_command_ids"))[:50],
            "execution_policy": str(terminal_command_surface.get("execution_policy") or ""),
            "fail_closed": bool(terminal_command_surface.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_command_surface.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_command_surface.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_command_surface.get("not_release_evidence")),
        },
        "protected_followup_run_plan_artifact_chain_summary": protected_followup_run_plan_artifact_chain_summary,
        "next_action_machine_summary": {
            "roadmap": str(terminal_next_action_machine.get("roadmap") or ""),
            "artifact_role": str(terminal_next_action_machine.get("artifact_role") or ""),
            "unique_action_count": _safe_int(terminal_next_action_machine.get("unique_action_count")),
            "computed_unique_action_count": _safe_int(
                computed_next_action_machine.get("unique_action_count")
            ),
            "json_ready_action_count": _safe_int(
                terminal_next_action_machine.get("json_ready_action_count")
            ),
            "json_closed_action_count": _safe_int(
                terminal_next_action_machine.get("json_closed_action_count")
            ),
            "readiness_state_counts": dict(
                _mapping(terminal_next_action_machine.get("readiness_state_counts"))
            ),
            "readiness_blocker_kind_counts": dict(
                _mapping(terminal_next_action_machine.get("readiness_blocker_kind_counts"))
            ),
            "unsafe_action_count": _safe_int(terminal_next_action_machine.get("unsafe_action_count")),
            "unsafe_action_ids": _strings(terminal_next_action_machine.get("unsafe_action_ids"))[:50],
            "missing_machine_field_action_count": _safe_int(
                terminal_next_action_machine.get("missing_machine_field_action_count")
            ),
            "missing_machine_field_action_ids": _strings(
                terminal_next_action_machine.get("missing_machine_field_action_ids")
            )[:50],
            "fail_closed": bool(terminal_next_action_machine.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_next_action_machine.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_next_action_machine.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_next_action_machine.get("not_release_evidence")),
        },
        "next_action_contract_summary": {
            "expected_roadmap": str(terminal_next_action_contract.get("expected_roadmap") or ""),
            "artifact_role": str(terminal_next_action_contract.get("artifact_role") or ""),
            "action_count": _safe_int(terminal_next_action_contract.get("action_count")),
            "computed_action_count": _safe_int(computed_next_action_contract.get("action_count")),
            "contract_complete_action_count": _safe_int(
                terminal_next_action_contract.get("contract_complete_action_count")
            ),
            "computed_contract_complete_action_count": _safe_int(
                computed_next_action_contract.get("contract_complete_action_count")
            ),
            "missing_contract_action_count": _safe_int(
                terminal_next_action_contract.get("missing_contract_action_count")
            ),
            "computed_missing_contract_action_count": _safe_int(
                computed_next_action_contract.get("missing_contract_action_count")
            ),
            "missing_contract_action_ids": _strings(
                terminal_next_action_contract.get("missing_contract_action_ids")
            )[:50],
            "release_or_auto_start_unsafe_action_count": _safe_int(
                terminal_next_action_contract.get("release_or_auto_start_unsafe_action_count")
            ),
            "computed_release_or_auto_start_unsafe_action_count": _safe_int(
                computed_next_action_contract.get("release_or_auto_start_unsafe_action_count")
            ),
            "release_or_auto_start_unsafe_action_ids": _strings(
                terminal_next_action_contract.get("release_or_auto_start_unsafe_action_ids")
            )[:50],
            "contract_ok": bool(terminal_next_action_contract.get("contract_ok")),
            "fail_closed": bool(terminal_next_action_contract.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_next_action_contract.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                terminal_next_action_contract.get("release_claim_allowed")
            ),
            "not_release_evidence": bool(
                terminal_next_action_contract.get("not_release_evidence")
            ),
        },
        "terminal_status": str(terminal.get("terminal_status") or ""),
        "chain_integrity_status": str(terminal.get("chain_integrity_status") or ""),
        "missing_external_inputs": _strings(unblocker.get("missing_external_inputs"))[:20],
        "release_unblocker_summary": {
            "terminal_present": bool(terminal_unblocker),
            "recommended_next_non_gpu_focus": str(terminal_unblocker.get("recommended_next_non_gpu_focus") or ""),
            "gpu_bubble_release_claim_blocked": bool(
                terminal_unblocker.get("gpu_bubble_release_claim_blocked")
            ),
            "missing_external_inputs": _strings(terminal_unblocker.get("missing_external_inputs"))[:20],
            "post_manual_rebuild_status": str(terminal_unblocker.get("post_manual_rebuild_status") or ""),
            "sd15_release_gap_status": str(terminal_unblocker.get("sd15_release_gap_status") or ""),
        },
        "input_resolution_summary": {
            "roadmap": str(terminal_input.get("roadmap") or ""),
            "missing_external_inputs": _strings(terminal_input.get("missing_external_inputs"))[:20],
            "sd15_checkpoint_exists": bool(terminal_input.get("sd15_checkpoint_exists")),
            "sd15_checkpoint_required": bool(terminal_input.get("sd15_checkpoint_required")),
            "new_source_root_count": _safe_int(terminal_input.get("new_source_root_count")),
            "new_source_root_required": bool(terminal_input.get("new_source_root_required")),
            "source_or_cache_axis_required": bool(terminal_input.get("source_or_cache_axis_required")),
            "warm_cache_or_caption_repair_required": bool(
                terminal_input.get("warm_cache_or_caption_repair_required")
            ),
            "external_input_detected": bool(terminal_input.get("external_input_detected")),
            "safe_to_auto_start": bool(terminal_input.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_input.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_input.get("not_release_evidence")),
        },
        "external_input_transition_table": {
            "roadmap": str(terminal_transition.get("roadmap") or ""),
            "artifact_role": str(terminal_transition.get("artifact_role") or ""),
            "transition_status": str(terminal_transition.get("transition_status") or ""),
            "external_input_required": bool(terminal_transition.get("external_input_required")),
            "missing_external_inputs": _strings(terminal_transition.get("missing_external_inputs"))[:20],
            "row_count": _safe_int(terminal_transition.get("row_count")),
            "required_row_count": _safe_int(terminal_transition.get("required_row_count")),
            "blocked_row_count": _safe_int(terminal_transition.get("blocked_row_count")),
            "detected_row_count": _safe_int(terminal_transition.get("detected_row_count")),
            "admitted_row_count": _safe_int(terminal_transition.get("admitted_row_count")),
            "manual_plan_ready_row_count": _safe_int(
                terminal_transition.get("manual_plan_ready_row_count")
            ),
            "unsafe_row_count": _safe_int(terminal_transition.get("unsafe_row_count")),
            "unsafe_row_ids": _strings(terminal_transition.get("unsafe_row_ids"))[:20],
            "transition_state_counts": dict(_mapping(terminal_transition.get("transition_state_counts"))),
            "next_json_refresh_sequence": _strings(terminal_transition.get("next_json_refresh_sequence"))[:20],
            "replay_command_count": _safe_int(terminal_transition.get("replay_command_count")),
            "replay_ready_command_count": _safe_int(terminal_transition.get("replay_ready_command_count")),
            "handoff_step_ids": _strings(terminal_transition.get("handoff_step_ids"))[:20],
            "row_ids": [
                str(_mapping(row).get("input_id") or "")
                for row in _list(terminal_transition.get("rows"))
            ],
            "fail_closed": bool(terminal_transition.get("fail_closed")),
            "safe_to_auto_start": bool(terminal_transition.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_transition.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_transition.get("not_release_evidence")),
        },
        "manual_evidence_blocking_summary": {
            "roadmap": str(terminal_manual.get("roadmap") or ""),
            "manual_gpu_evidence_ready": bool(terminal_manual.get("manual_gpu_evidence_ready")),
            "manual_gpu_evidence_required": bool(terminal_manual.get("manual_gpu_evidence_required")),
            "source_cache_axis_manual_canary_plan_ready": bool(
                terminal_manual.get("source_cache_axis_manual_canary_plan_ready")
            ),
            "source_cache_axis_manual_canary_plan_required": bool(
                terminal_manual.get("source_cache_axis_manual_canary_plan_required")
            ),
            "sd15_checkpoint_required": bool(terminal_manual.get("sd15_checkpoint_required")),
            "natural_load_canary_pending": bool(terminal_manual.get("natural_load_canary_pending")),
            "release_claims_rebuild_required": bool(terminal_manual.get("release_claims_rebuild_required")),
            "release_gate_blockers": _strings(terminal_manual.get("release_gate_blockers"))[:20],
            "next_required_inputs": _strings(terminal_manual.get("next_required_inputs"))[:20],
            "next_json_rebuild_stage_id": str(terminal_manual.get("next_json_rebuild_stage_id") or ""),
            "safe_to_auto_start": bool(terminal_manual.get("safe_to_auto_start")),
            "release_claim_allowed": bool(terminal_manual.get("release_claim_allowed")),
            "not_release_evidence": bool(terminal_manual.get("not_release_evidence")),
        },
        "external_input_filesystem_audit": {
            "registry_status": str(external_input_audit.get("registry_status") or ""),
            "live_status": str(external_input_audit.get("live_status") or ""),
            "filesystem_external_input_detected": bool(
                external_input_audit.get("filesystem_external_input_detected")
            ),
            "sd15_checkpoint_exists": bool(external_input_audit.get("sd15_checkpoint_exists")),
            "sd15_checkpoint_count": _safe_int(external_input_audit.get("sd15_checkpoint_count")),
            "new_source_root_count": _safe_int(external_input_audit.get("new_source_root_count")),
            "artifact_matches_filesystem_and_release_blockers": bool(
                external_input_audit.get("artifact_matches_filesystem_and_release_blockers")
            ),
            "drift_reason_ids": _strings(external_input_audit.get("drift_reason_ids"))[:20],
        },
        "source_axis_freshness_dedupe_audit": {
            "roadmap": str(terminal_source_axis_freshness.get("roadmap") or ""),
            "artifact_role": str(terminal_source_axis_freshness.get("artifact_role") or ""),
            "report": str(terminal_source_axis_freshness.get("report") or ""),
            "status": str(terminal_source_axis_freshness.get("status") or ""),
            "axis_state": str(terminal_source_axis_freshness.get("axis_state") or ""),
            "external_input_detected": bool(
                terminal_source_axis_freshness.get("external_input_detected")
            ),
            "computed_external_input_detected": bool(
                computed_source_axis_freshness.get("external_input_detected")
            ),
            "new_source_root_count": _safe_int(
                terminal_source_axis_freshness.get("new_source_root_count")
            ),
            "computed_new_source_root_count": _safe_int(
                computed_source_axis_freshness.get("new_source_root_count")
            ),
            "completed_axis_count": _safe_int(
                terminal_source_axis_freshness.get("completed_axis_count")
            ),
            "completed_out_dir_count": _safe_int(
                terminal_source_axis_freshness.get("completed_out_dir_count")
            ),
            "candidate_status": str(terminal_source_axis_freshness.get("candidate_status") or ""),
            "candidate_fresh": bool(terminal_source_axis_freshness.get("candidate_fresh")),
            "candidate_duplicate_or_stale": bool(
                terminal_source_axis_freshness.get("candidate_duplicate_or_stale")
            ),
            "matching_axis_count": _safe_int(
                terminal_source_axis_freshness.get("matching_axis_count")
            ),
            "preflight_admitted": bool(
                terminal_source_axis_freshness.get("preflight_admitted")
            ),
            "manual_canary_plan_ready": bool(
                terminal_source_axis_freshness.get("manual_canary_plan_ready")
            ),
            "publishable": bool(terminal_source_axis_freshness.get("publishable")),
            "blocker_count": _safe_int(terminal_source_axis_freshness.get("blocker_count")),
            "blockers": _strings(terminal_source_axis_freshness.get("blockers"))[:20],
            "unsafe_audit_count": _safe_int(
                terminal_source_axis_freshness.get("unsafe_audit_count")
            ),
            "computed_unsafe_audit_count": _safe_int(
                computed_source_axis_freshness.get("unsafe_audit_count")
            ),
            "unsafe_audit_ids": _strings(
                terminal_source_axis_freshness.get("unsafe_audit_ids")
            )[:20],
            "fail_closed": bool(terminal_source_axis_freshness.get("fail_closed")),
            "computed_fail_closed": bool(computed_source_axis_freshness.get("fail_closed")),
            "not_release_evidence": bool(
                terminal_source_axis_freshness.get("not_release_evidence")
            ),
            "does_not_run_training": bool(
                terminal_source_axis_freshness.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(
                terminal_source_axis_freshness.get("does_not_run_cuda")
            ),
            "safe_to_auto_start": bool(
                terminal_source_axis_freshness.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_source_axis_freshness.get("release_claim_allowed")
            ),
        },
        "source_axis_requirement_summary": {
            "roadmap": str(terminal_source_axis_requirement.get("roadmap") or ""),
            "artifact_role": str(terminal_source_axis_requirement.get("artifact_role") or ""),
            "report": str(terminal_source_axis_requirement.get("report") or ""),
            "computed_report": str(computed_source_axis_requirement.get("report") or ""),
            "status": str(terminal_source_axis_requirement.get("status") or ""),
            "computed_status": str(computed_source_axis_requirement.get("status") or ""),
            "family_count": _safe_int(terminal_source_axis_requirement.get("family_count")),
            "computed_family_count": _safe_int(computed_source_axis_requirement.get("family_count")),
            "external_input_required_count": _safe_int(
                terminal_source_axis_requirement.get("external_input_required_count")
            ),
            "computed_external_input_required_count": _safe_int(
                computed_source_axis_requirement.get("external_input_required_count")
            ),
            "computed_external_input_required_action_count": computed_source_axis_requirement_action_count,
            "candidate_available_family_count": _safe_int(
                terminal_source_axis_requirement.get("candidate_available_family_count")
            ),
            "exhausted_family_count": _safe_int(
                terminal_source_axis_requirement.get("exhausted_family_count")
            ),
            "no_ready_source_axis_family_count": _safe_int(
                terminal_source_axis_requirement.get("no_ready_source_axis_family_count")
            ),
            "completed_existing_command_count": _safe_int(
                terminal_source_axis_requirement.get("completed_existing_command_count")
            ),
            "external_input_required": bool(
                terminal_source_axis_requirement.get("external_input_required")
            ),
            "fail_closed": bool(terminal_source_axis_requirement.get("fail_closed")),
            "not_release_evidence": bool(
                terminal_source_axis_requirement.get("not_release_evidence")
            ),
            "does_not_run_training": bool(
                terminal_source_axis_requirement.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(terminal_source_axis_requirement.get("does_not_run_cuda")),
            "safe_to_auto_start": bool(terminal_source_axis_requirement.get("safe_to_auto_start")),
            "release_claim_allowed": bool(
                terminal_source_axis_requirement.get("release_claim_allowed")
            ),
        },
        "source_cache_axis_pipeline_readiness_summary": {
            "roadmap": str(terminal_source_cache_pipeline_summary.get("roadmap") or ""),
            "artifact_role": str(terminal_source_cache_pipeline_summary.get("artifact_role") or ""),
            "report": str(terminal_source_cache_pipeline_summary.get("report") or ""),
            "computed_report": str(computed_source_cache_pipeline_summary.get("report") or ""),
            "status": str(terminal_source_cache_pipeline_summary.get("status") or ""),
            "computed_status": str(computed_source_cache_pipeline_summary.get("status") or ""),
            "axis_readiness_status": str(
                terminal_source_cache_pipeline_summary.get("axis_readiness_status") or ""
            ),
            "computed_axis_readiness_status": str(
                computed_source_cache_pipeline_summary.get("axis_readiness_status") or ""
            ),
            "pipeline_complete": bool(
                terminal_source_cache_pipeline_summary.get("pipeline_complete")
            ),
            "external_input_required": bool(
                terminal_source_cache_pipeline_summary.get("external_input_required")
            ),
            "preflight_admitted": bool(
                terminal_source_cache_pipeline_summary.get("preflight_admitted")
            ),
            "manual_canary_plan_ready": bool(
                terminal_source_cache_pipeline_summary.get("manual_canary_plan_ready")
            ),
            "waiting_external_input": bool(
                terminal_source_cache_pipeline_summary.get("waiting_external_input")
            ),
            "duplicate_or_stale_axis_blocked": bool(
                terminal_source_cache_pipeline_summary.get("duplicate_or_stale_axis_blocked")
            ),
            "cache_axis_not_ready": bool(
                terminal_source_cache_pipeline_summary.get("cache_axis_not_ready")
            ),
            "stage_count": _safe_int(terminal_source_cache_pipeline_summary.get("stage_count")),
            "computed_stage_count": _safe_int(computed_source_cache_pipeline_summary.get("stage_count")),
            "stage_ok_count": _safe_int(
                terminal_source_cache_pipeline_summary.get("stage_ok_count")
            ),
            "computed_stage_ok_count": _safe_int(
                computed_source_cache_pipeline_summary.get("stage_ok_count")
            ),
            "blocker_count": _safe_int(terminal_source_cache_pipeline_summary.get("blocker_count")),
            "next_action_count": _safe_int(
                terminal_source_cache_pipeline_summary.get("next_action_count")
            ),
            "fail_closed": bool(terminal_source_cache_pipeline_summary.get("fail_closed")),
            "not_release_evidence": bool(
                terminal_source_cache_pipeline_summary.get("not_release_evidence")
            ),
            "does_not_run_training": bool(
                terminal_source_cache_pipeline_summary.get("does_not_run_training")
            ),
            "does_not_run_cuda": bool(
                terminal_source_cache_pipeline_summary.get("does_not_run_cuda")
            ),
            "safe_to_auto_start": bool(
                terminal_source_cache_pipeline_summary.get("safe_to_auto_start")
            ),
            "release_claim_allowed": bool(
                terminal_source_cache_pipeline_summary.get("release_claim_allowed")
            ),
        },
        "external_input_admission_summary": external_input_admission_summary,
        "external_input_intake_registry_summary": external_input_intake_summary,
        "external_input_replay_plan_summary": external_input_replay_summary,
        "external_input_handoff_packet_summary": external_input_handoff_summary,
        "newbie_warm_cache_inventory_summary": newbie_warm_cache_summary,
        "source_cache_axis_admission_preflight_summary": source_cache_preflight_summary,
        "source_cache_axis_manual_canary_plan_summary": source_cache_manual_plan_summary,
        "post_manual_evidence_rebuild_plan_summary": post_manual_rebuild_summary,
        "manual_review_artifact_chain_summary": manual_review_artifact_chain_summary,
        "sdxl_diagnostic_artifact_chain_summary": sdxl_diagnostic_artifact_chain_summary,
        "source_cache_negative_evidence_summary": {
            "roadmap": str(source_cache_negative.get("roadmap") or ""),
            "new_source_root_count": _safe_int(source_cache_negative.get("new_source_root_count")),
            "current_source_root_duplicate_count": _safe_int(
                source_cache_negative.get("current_source_root_duplicate_count")
            ),
            "external_input_required_family_count": _safe_int(
                source_cache_negative.get("external_input_required_family_count")
            ),
            "newbie_warm_cache_status": str(source_cache_negative.get("newbie_warm_cache_status") or ""),
            "claimable_cache_axis_count": _safe_int(
                source_cache_negative.get("claimable_cache_axis_count")
            ),
            "cannot_clear_new_source_root_blocker_from_current_axis": bool(
                source_cache_negative.get("cannot_clear_new_source_root_blocker_from_current_axis")
            ),
            "cannot_clear_warm_cache_axis_from_inventory": bool(
                source_cache_negative.get("cannot_clear_warm_cache_axis_from_inventory")
            ),
            "negative_evidence_reason_ids": _strings(
                source_cache_negative.get("negative_evidence_reason_ids")
            )[:20],
            "not_release_evidence": bool(source_cache_negative.get("not_release_evidence")),
            "safe_to_auto_start": bool(source_cache_negative.get("safe_to_auto_start")),
            "release_claim_allowed": bool(source_cache_negative.get("release_claim_allowed")),
        },
        "source_cache_axis_identity_registry": {
            "report": str(source_cache_identity.get("report") or ""),
            "roadmap": str(source_cache_identity.get("roadmap") or ""),
            "status": str(source_cache_identity.get("status") or ""),
            "axis_state": str(source_cache_identity.get("axis_state") or ""),
            "row_count": _safe_int(source_cache_identity.get("row_count")),
            "full_axis_identity_row_count": _safe_int(
                source_cache_identity.get("full_axis_identity_row_count")
            ),
            "duplicate_or_stale_axis_count": _safe_int(
                source_cache_identity.get("duplicate_or_stale_axis_count")
            ),
            "fresh_axis_candidate_count": _safe_int(
                source_cache_identity.get("fresh_axis_candidate_count")
            ),
            "unsafe_row_count": _safe_int(source_cache_identity.get("unsafe_row_count")),
            "fail_closed": bool(source_cache_identity.get("fail_closed")),
            "not_release_evidence": bool(source_cache_identity.get("not_release_evidence")),
            "safe_to_auto_start": bool(source_cache_identity.get("safe_to_auto_start")),
            "release_claim_allowed": bool(source_cache_identity.get("release_claim_allowed")),
        },
        "source_cache_axis_identity_registry_summary": source_cache_identity_summary,
        "artifact_freshness_audit": {
            "freshness_ok": bool(artifact_freshness.get("freshness_ok")),
            "required_artifact_missing_count": _safe_int(
                artifact_freshness.get("required_artifact_missing_count")
            ),
            "upstream_newer_than_readiness_count": _safe_int(
                artifact_freshness.get("upstream_newer_than_readiness_count")
            ),
            "readiness_not_older_than_upstream": bool(
                artifact_freshness.get("readiness_not_older_than_upstream")
            ),
            "terminal_observation_not_older_than_readiness": bool(
                artifact_freshness.get("terminal_observation_not_older_than_readiness")
            ),
            "drift_reason_ids": _strings(artifact_freshness.get("drift_reason_ids"))[:20],
        },
        "next_required_inputs": required_inputs[:20],
        "failure_count": len(failures),
        "failures": failures,
        "blocked_actions": [
            "do_not_publish_gpu_bubble_release_claim_when_guard_fails",
            "do_not_auto_start_gpu_heavy_from_guard",
            "do_not_treat_guard_pass_as_release_evidence",
        ],
    }


__all__ = [
    "REPORT",
    "READINESS_REPORT",
    "ROADMAP",
    "TERMINAL_REPORT",
    "build_gpu_bubble_release_readiness_guard",
]
