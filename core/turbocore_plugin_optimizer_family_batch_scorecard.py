"""Batch scorecard for selected plugin optimizer native-family evidence.

Plugin optimizer selectors expose many optimizer formulas behind one enum.  This
report keeps selector classification, selected-family ABI gates, and native
dispatch policy in one default-off evidence package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_factored_custom_optimizer_state_layout_scorecard import (
    build_factored_custom_optimizer_state_layout_scorecard,
)
from core.turbocore_plugin_factored_memory_state_layout_scorecard import (
    build_plugin_factored_memory_state_layout_scorecard,
)
from core.turbocore_plugin_adamlike_selected_optimizer_scorecard import (
    build_plugin_adamlike_selected_optimizer_scorecard,
)
from core.turbocore_plugin_adamlike_family_batch_scorecard import (
    build_plugin_adamlike_family_batch_scorecard,
)
from core.turbocore_plugin_adamlike_owner_release_hold_scorecard import (
    build_plugin_adamlike_owner_release_hold_scorecard,
)
from core.turbocore_plugin_adamlike_request_schema_ui_non_exposure_scorecard import (
    build_plugin_adamlike_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_adaptivelr_family_batch_scorecard import (
    build_plugin_adaptivelr_family_batch_scorecard,
)
from core.turbocore_plugin_adaptivelr_owner_release_hold_scorecard import (
    build_plugin_adaptivelr_owner_release_hold_scorecard,
)
from core.turbocore_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard import (
    build_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_closure_second_order_family_batch_scorecard import (
    build_plugin_closure_second_order_family_batch_scorecard,
)
from core.turbocore_plugin_closure_second_order_owner_release_hold_scorecard import (
    build_plugin_closure_second_order_owner_release_hold_scorecard,
)
from core.turbocore_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard import (
    build_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_custom_formula_family_batch_scorecard import (
    build_plugin_custom_formula_family_batch_scorecard,
)
from core.turbocore_plugin_custom_formula_owner_release_hold_scorecard import (
    build_plugin_custom_formula_owner_release_hold_scorecard,
)
from core.turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard import (
    build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_factored_memory_family_batch_scorecard import (
    build_plugin_factored_memory_family_batch_scorecard,
)
from core.turbocore_plugin_factored_memory_owner_release_hold_scorecard import (
    build_plugin_factored_memory_owner_release_hold_scorecard,
)
from core.turbocore_plugin_factored_memory_request_schema_ui_non_exposure_scorecard import (
    build_plugin_factored_memory_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_fused_backward_family_batch_scorecard import (
    build_plugin_fused_backward_family_batch_scorecard,
)
from core.turbocore_plugin_fused_backward_owner_release_hold_scorecard import (
    build_plugin_fused_backward_owner_release_hold_scorecard,
)
from core.turbocore_plugin_fused_backward_request_schema_ui_non_exposure_scorecard import (
    build_plugin_fused_backward_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_model_shape_aware_family_batch_scorecard import (
    build_plugin_model_shape_aware_family_batch_scorecard,
)
from core.turbocore_plugin_model_shape_aware_owner_release_hold_scorecard import (
    build_plugin_model_shape_aware_owner_release_hold_scorecard,
)
from core.turbocore_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard import (
    build_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_schedulefree_family_batch_scorecard import (
    build_plugin_schedulefree_family_batch_scorecard,
)
from core.turbocore_plugin_schedulefree_owner_release_hold_scorecard import (
    build_plugin_schedulefree_owner_release_hold_scorecard,
)
from core.turbocore_plugin_schedulefree_request_schema_ui_non_exposure_scorecard import (
    build_plugin_schedulefree_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_schedulefree_selected_optimizer_scorecard import (
    build_plugin_schedulefree_selected_optimizer_scorecard,
)
from core.turbocore_plugin_simple_formula_family_batch_scorecard import (
    build_plugin_simple_formula_family_batch_scorecard,
)
from core.turbocore_plugin_simple_formula_e2e_shadow_matrix_scorecard import (
    build_plugin_simple_formula_e2e_shadow_matrix_scorecard,
)
from core.turbocore_plugin_simple_formula_canary_rollout_policy_scorecard import (
    build_plugin_simple_formula_canary_rollout_policy_scorecard,
)
from core.turbocore_plugin_simple_formula_dispatch_integration_review_scorecard import (
    build_plugin_simple_formula_dispatch_integration_review_scorecard,
)
from core.turbocore_plugin_simple_formula_owner_release_hold_scorecard import (
    build_plugin_simple_formula_owner_release_hold_scorecard,
)
from core.turbocore_plugin_simple_formula_request_schema_ui_non_exposure_scorecard import (
    build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_state_adapter_special_family_batch_scorecard import (
    build_plugin_state_adapter_special_family_batch_scorecard,
)
from core.turbocore_plugin_state_adapter_special_owner_release_hold_scorecard import (
    build_plugin_state_adapter_special_owner_release_hold_scorecard,
)
from core.turbocore_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard import (
    build_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard,
)


_SELECTED_GATE_BY_FAMILY = {
    "adam_like_formula": "selected_adamlike_abi_ready",
    "adaptive_lr_state_machine": "selected_adaptivelr_family_batch_ready",
    "closure_or_second_order": "selected_closure_second_order_family_batch_ready",
    "custom_formula": "selected_custom_formula_family_batch_ready",
    "factored_memory_layout": "selected_factored_memory_family_batch_ready",
    "fused_backward": "selected_fused_backward_family_batch_ready",
    "model_or_shape_aware": "selected_model_shape_aware_family_batch_ready",
    "schedule_free_state_machine": "selected_schedulefree_abi_ready",
    "simple_formula": "selected_simple_formula_family_batch_ready",
    "state_adapter_special": "selected_state_adapter_special_family_batch_ready",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_optimizer_family_batch_scorecard(
    *,
    write_artifact: bool = False,
    refresh_family_artifacts: bool = False,
) -> dict[str, Any]:
    """Aggregate plugin-selector family evidence without enabling dispatch."""

    if refresh_family_artifacts:
        _refresh_family_artifacts()
    selector = build_plugin_optimizer_selector_scorecard()
    adamlike = build_plugin_adamlike_selected_optimizer_scorecard()
    adamlike_batch = _adamlike_family_batch_report()
    adamlike_owner_hold = _adamlike_owner_release_hold_report(adamlike_batch)
    adamlike_request_schema_ui = _adamlike_request_schema_ui_report(adamlike_owner_hold)
    schedulefree = build_plugin_schedulefree_selected_optimizer_scorecard()
    schedulefree_batch = _schedulefree_family_batch_report()
    schedulefree_owner_hold = _schedulefree_owner_release_hold_report(schedulefree_batch)
    schedulefree_request_schema_ui = _schedulefree_request_schema_ui_report(schedulefree_owner_hold)
    adaptivelr_batch = _adaptivelr_family_batch_report()
    adaptivelr_owner_hold = _adaptivelr_owner_release_hold_report(adaptivelr_batch)
    adaptivelr_request_schema_ui = _adaptivelr_request_schema_ui_report(adaptivelr_owner_hold)
    simple_formula_batch = _simple_formula_family_batch_report()
    simple_formula_e2e = _simple_formula_e2e_shadow_matrix_report(simple_formula_batch)
    simple_formula_rollout = _simple_formula_canary_rollout_policy_report(simple_formula_e2e)
    simple_formula_dispatch = _simple_formula_dispatch_integration_review_report(simple_formula_rollout)
    simple_formula_owner_hold = _simple_formula_owner_release_hold_report(simple_formula_dispatch)
    simple_formula_request_schema_ui = _simple_formula_request_schema_ui_report(simple_formula_owner_hold)
    closure_second_order_batch = _closure_second_order_family_batch_report()
    closure_second_order_owner_hold = _closure_second_order_owner_release_hold_report(closure_second_order_batch)
    closure_second_order_request_schema_ui = _closure_second_order_request_schema_ui_report(
        closure_second_order_owner_hold
    )
    custom_formula_batch = _custom_formula_family_batch_report()
    custom_formula_owner_hold = _custom_formula_owner_release_hold_report(custom_formula_batch)
    custom_formula_request_schema_ui = _custom_formula_request_schema_ui_report(custom_formula_owner_hold)
    factored_memory_batch = _factored_memory_family_batch_report()
    factored_memory_owner_hold = _factored_memory_owner_release_hold_report(factored_memory_batch)
    factored_memory_request_schema_ui = _factored_memory_request_schema_ui_report(factored_memory_owner_hold)
    fused_backward_batch = _fused_backward_family_batch_report()
    fused_backward_owner_hold = _fused_backward_owner_release_hold_report(fused_backward_batch)
    fused_backward_request_schema_ui = _fused_backward_request_schema_ui_report(fused_backward_owner_hold)
    model_shape_batch = _model_shape_family_batch_report()
    model_shape_owner_hold = _model_shape_owner_release_hold_report(model_shape_batch)
    model_shape_request_schema_ui = _model_shape_request_schema_ui_report(model_shape_owner_hold)
    state_adapter_special_batch = _state_adapter_special_family_batch_report()
    state_adapter_special_owner_hold = _state_adapter_special_owner_release_hold_report(state_adapter_special_batch)
    state_adapter_special_request_schema_ui = _state_adapter_special_request_schema_ui_report(
        state_adapter_special_owner_hold
    )
    factored = build_factored_custom_optimizer_state_layout_scorecard(write_artifact=write_artifact)
    plugin_factored = build_plugin_factored_memory_state_layout_scorecard()
    family_rows = _family_rows(
        selector,
        adamlike,
        adamlike_batch,
        adamlike_owner_hold,
        adamlike_request_schema_ui,
        schedulefree,
        schedulefree_batch,
        schedulefree_owner_hold,
        schedulefree_request_schema_ui,
        adaptivelr_batch,
        adaptivelr_owner_hold,
        adaptivelr_request_schema_ui,
        simple_formula_batch,
        simple_formula_owner_hold,
        simple_formula_request_schema_ui,
        closure_second_order_batch,
        closure_second_order_owner_hold,
        closure_second_order_request_schema_ui,
        custom_formula_batch,
        custom_formula_owner_hold,
        custom_formula_request_schema_ui,
        factored_memory_batch,
        factored_memory_owner_hold,
        factored_memory_request_schema_ui,
        fused_backward_batch,
        fused_backward_owner_hold,
        fused_backward_request_schema_ui,
        model_shape_batch,
        model_shape_owner_hold,
        model_shape_request_schema_ui,
        state_adapter_special_batch,
        state_adapter_special_owner_hold,
        state_adapter_special_request_schema_ui,
        factored,
        plugin_factored,
    )
    failed_sources = _failed_sources(
        selector,
        adamlike,
        adamlike_owner_hold,
        adamlike_request_schema_ui,
        schedulefree,
        schedulefree_batch,
        schedulefree_owner_hold,
        schedulefree_request_schema_ui,
        adaptivelr_batch,
        adaptivelr_owner_hold,
        adaptivelr_request_schema_ui,
        simple_formula_batch,
        simple_formula_e2e,
        simple_formula_rollout,
        simple_formula_dispatch,
        simple_formula_owner_hold,
        simple_formula_request_schema_ui,
        closure_second_order_batch,
        closure_second_order_owner_hold,
        closure_second_order_request_schema_ui,
        custom_formula_batch,
        custom_formula_owner_hold,
        custom_formula_request_schema_ui,
        factored_memory_batch,
        factored_memory_owner_hold,
        factored_memory_request_schema_ui,
        fused_backward_batch,
        fused_backward_owner_hold,
        fused_backward_request_schema_ui,
        model_shape_batch,
        model_shape_owner_hold,
        model_shape_request_schema_ui,
        state_adapter_special_batch,
        state_adapter_special_owner_hold,
        state_adapter_special_request_schema_ui,
        factored,
        plugin_factored,
    )
    unsafe_claims = _unsafe_claims(
        selector,
        adamlike,
        adamlike_owner_hold,
        adamlike_request_schema_ui,
        schedulefree,
        schedulefree_batch,
        schedulefree_owner_hold,
        schedulefree_request_schema_ui,
        adaptivelr_batch,
        adaptivelr_owner_hold,
        adaptivelr_request_schema_ui,
        simple_formula_batch,
        simple_formula_e2e,
        simple_formula_rollout,
        simple_formula_dispatch,
        simple_formula_owner_hold,
        simple_formula_request_schema_ui,
        closure_second_order_batch,
        closure_second_order_owner_hold,
        closure_second_order_request_schema_ui,
        custom_formula_batch,
        custom_formula_owner_hold,
        custom_formula_request_schema_ui,
        factored_memory_batch,
        factored_memory_owner_hold,
        factored_memory_request_schema_ui,
        fused_backward_batch,
        fused_backward_owner_hold,
        fused_backward_request_schema_ui,
        model_shape_batch,
        model_shape_owner_hold,
        model_shape_request_schema_ui,
        state_adapter_special_batch,
        state_adapter_special_owner_hold,
        state_adapter_special_request_schema_ui,
        factored,
        plugin_factored,
    )
    selected_ready = [row for row in family_rows if row["selected_optimizer_gate_ready"] is True]
    pending = [row for row in family_rows if row["selected_optimizer_gate_ready"] is False]
    blockers = failed_sources + unsafe_claims
    ready = bool(family_rows) and not blockers

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_optimizer_family_batch_scorecard_v0",
        "gate": "plugin_optimizer_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "plugin_optimizer_family_batch_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "selector_scorecard": _compact_selector(selector),
        "selected_gate_scorecards": {
            "adam_like_formula": _compact_selected_gate(adamlike),
            "adam_like_family_batch": _compact_adamlike_batch(adamlike_batch),
            "adam_like_owner_release_hold": _compact_adamlike_owner_release_hold(adamlike_owner_hold),
            "adam_like_request_schema_ui_non_exposure": _compact_adamlike_request_schema_ui(
                adamlike_request_schema_ui
            ),
            "adaptive_lr_state_machine_family_batch": _compact_adaptivelr_batch(adaptivelr_batch),
            "adaptive_lr_state_machine_owner_release_hold": _compact_adaptivelr_owner_release_hold(
                adaptivelr_owner_hold
            ),
            "adaptive_lr_state_machine_request_schema_ui_non_exposure": (
                _compact_adaptivelr_request_schema_ui(adaptivelr_request_schema_ui)
            ),
            "closure_second_order_family_batch": _compact_closure_second_order_batch(closure_second_order_batch),
            "closure_second_order_owner_release_hold": _compact_closure_second_order_owner_release_hold(
                closure_second_order_owner_hold
            ),
            "closure_second_order_request_schema_ui_non_exposure": (
                _compact_closure_second_order_request_schema_ui(closure_second_order_request_schema_ui)
            ),
            "custom_formula_family_batch": _compact_custom_formula_batch(custom_formula_batch),
            "custom_formula_owner_release_hold": _compact_custom_formula_owner_release_hold(
                custom_formula_owner_hold
            ),
            "custom_formula_request_schema_ui_non_exposure": _compact_custom_formula_request_schema_ui(
                custom_formula_request_schema_ui
            ),
            "factored_memory_family_batch": _compact_factored_memory_batch(factored_memory_batch),
            "factored_memory_owner_release_hold": _compact_factored_memory_owner_release_hold(
                factored_memory_owner_hold
            ),
            "factored_memory_request_schema_ui_non_exposure": _compact_factored_memory_request_schema_ui(
                factored_memory_request_schema_ui
            ),
            "fused_backward_family_batch": _compact_fused_backward_batch(fused_backward_batch),
            "fused_backward_owner_release_hold": _compact_fused_backward_owner_release_hold(
                fused_backward_owner_hold
            ),
            "fused_backward_request_schema_ui_non_exposure": _compact_fused_backward_request_schema_ui(
                fused_backward_request_schema_ui
            ),
            "model_shape_aware_family_batch": _compact_model_shape_batch(model_shape_batch),
            "model_shape_aware_owner_release_hold": _compact_model_shape_owner_release_hold(
                model_shape_owner_hold
            ),
            "model_shape_aware_request_schema_ui_non_exposure": _compact_model_shape_request_schema_ui(
                model_shape_request_schema_ui
            ),
            "state_adapter_special_family_batch": _compact_state_adapter_special_batch(state_adapter_special_batch),
            "state_adapter_special_owner_release_hold": _compact_state_adapter_special_owner_release_hold(
                state_adapter_special_owner_hold
            ),
            "state_adapter_special_request_schema_ui_non_exposure": (
                _compact_state_adapter_special_request_schema_ui(state_adapter_special_request_schema_ui)
            ),
            "schedule_free_state_machine": _compact_selected_gate(schedulefree),
            "schedule_free_family_batch": _compact_schedulefree_batch(schedulefree_batch),
            "schedule_free_owner_release_hold": _compact_schedulefree_owner_release_hold(
                schedulefree_owner_hold
            ),
            "schedule_free_request_schema_ui_non_exposure": _compact_schedulefree_request_schema_ui(
                schedulefree_request_schema_ui
            ),
            "simple_formula_family_batch": _compact_simple_formula_batch(simple_formula_batch),
            "simple_formula_e2e_shadow_matrix": _compact_simple_formula_e2e_shadow_matrix(
                simple_formula_e2e
            ),
            "simple_formula_canary_rollout_policy": _compact_simple_formula_rollout_policy(
                simple_formula_rollout
            ),
            "simple_formula_dispatch_integration_review": _compact_simple_formula_dispatch_review(
                simple_formula_dispatch
            ),
            "simple_formula_owner_release_hold": _compact_simple_formula_owner_release_hold(
                simple_formula_owner_hold
            ),
            "simple_formula_request_schema_ui_non_exposure": _compact_simple_formula_request_schema_ui(
                simple_formula_request_schema_ui
            ),
            "factored_custom_builtin_layout": _compact_factored_gate(factored),
            "plugin_factored_memory_state_layout": _compact_plugin_factored_gate(plugin_factored),
        },
        "family_rows": family_rows,
        "summary": {
            "plugin_optimizer_count": int(_summary(selector).get("plugin_optimizer_count", 0) or 0),
            "route_family_counts": dict(_as_dict(_summary(selector).get("route_family_counts"))),
            "selected_optimizer_gate_ready_count": len(selected_ready),
            "selected_optimizer_gate_pending_count": len(pending),
            "selected_adamlike_case_count": int(_summary(adamlike).get("case_count", 0) or 0),
            "selected_adamlike_native_canary_ready_count": int(
                _summary(adamlike_batch).get("selected_native_canary_ready_count", 0) or 0
            ),
            "selected_adamlike_pending_count": int(
                _summary(adamlike_batch).get("pending_selected_optimizer_count", 0) or 0
            ),
            "selected_adamlike_e2e_shadow_matrix_ready": _summary(adamlike_batch).get(
                "e2e_shadow_matrix_ready"
            )
            is True,
            "selected_adamlike_canary_rollout_policy_ready": _summary(adamlike_batch).get(
                "canary_rollout_policy_ready"
            )
            is True,
            "selected_adamlike_owner_release_hold_ready": adamlike_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_adamlike_owner_release_hold_optimizer_count": int(
                _summary(adamlike_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_adamlike_owner_release_hold_product_native_ready_count": int(
                _summary(adamlike_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_adamlike_request_schema_ui_non_exposure_ready": (
                adamlike_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_adamlike_request_schema_ui_optimizer_count": int(
                _summary(adamlike_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_adamlike_request_schema_ui_forbidden_token_hit_count": int(
                _summary(adamlike_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_adamlike_request_schema_ui_product_native_ready_count": int(
                _summary(adamlike_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_schedulefree_case_count": int(_summary(schedulefree).get("case_count", 0) or 0),
            "selected_schedulefree_family_batch_ready": schedulefree_batch.get(
                "selected_schedulefree_family_batch_ready"
            )
            is True,
            "selected_schedulefree_e2e_shadow_case_count": int(
                _summary(schedulefree_batch).get("e2e_shadow_case_count", 0) or 0
            ),
            "selected_schedulefree_native_canary_ready_count": int(
                _summary(schedulefree_batch).get("selected_native_canary_ready_count", 0) or 0
            ),
            "selected_schedulefree_dispatch_review_gate_ready": _summary(schedulefree_batch).get(
                "dispatch_review_gate_ready"
            )
            is True,
            "selected_schedulefree_owner_release_hold_ready": schedulefree_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_schedulefree_owner_release_hold_optimizer_count": int(
                _summary(schedulefree_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_schedulefree_owner_release_hold_product_native_ready_count": int(
                _summary(schedulefree_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_schedulefree_request_schema_ui_non_exposure_ready": (
                schedulefree_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_schedulefree_request_schema_ui_optimizer_count": int(
                _summary(schedulefree_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_schedulefree_request_schema_ui_forbidden_token_hit_count": int(
                _summary(schedulefree_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_schedulefree_request_schema_ui_product_native_ready_count": int(
                _summary(schedulefree_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_adaptivelr_family_batch_ready": adaptivelr_batch.get(
                "selected_adaptivelr_family_batch_ready"
            )
            is True,
            "selected_adaptivelr_reference_ready_count": int(
                _summary(adaptivelr_batch).get("selected_state_machine_reference_ready_count", 0) or 0
            ),
            "selected_adaptivelr_state_machine_abi_spec_ready_count": int(
                _summary(adaptivelr_batch).get("selected_state_machine_abi_spec_ready_count", 0) or 0
            ),
            "selected_adaptivelr_state_machine_abi_implementation_ready_count": int(
                _summary(adaptivelr_batch).get("selected_state_machine_abi_implementation_ready_count", 0) or 0
            ),
            "selected_adaptivelr_native_kernel_preconditions_spec_ready_count": int(
                _summary(adaptivelr_batch).get("selected_native_kernel_preconditions_spec_ready_count", 0) or 0
            ),
            "selected_adaptivelr_native_kernel_preconditions_implementation_ready_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_native_kernel_preconditions_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_state_machine_replay_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_state_machine_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_state_machine_replay_case_planned_count": int(
                _summary(adaptivelr_batch).get("selected_state_machine_replay_case_planned_count", 0) or 0
            ),
            "selected_adaptivelr_state_machine_replay_case_implementation_ready_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_state_machine_replay_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_state_machine_replay_resume_case_planned_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_state_machine_replay_resume_case_planned_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count": int(
                _summary(adaptivelr_batch).get(
                    "selected_state_machine_replay_resume_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_adaptivelr_owner_release_hold_ready": adaptivelr_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_adaptivelr_owner_release_hold_optimizer_count": int(
                _summary(adaptivelr_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_adaptivelr_owner_release_hold_product_native_ready_count": int(
                _summary(adaptivelr_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_adaptivelr_request_schema_ui_non_exposure_ready": (
                adaptivelr_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_adaptivelr_request_schema_ui_optimizer_count": int(
                _summary(adaptivelr_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_adaptivelr_request_schema_ui_forbidden_token_hit_count": int(
                _summary(adaptivelr_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_adaptivelr_request_schema_ui_product_native_ready_count": int(
                _summary(adaptivelr_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_simple_formula_family_batch_ready": simple_formula_batch.get(
                "selected_simple_formula_family_batch_ready"
            )
            is True,
            "selected_simple_formula_optimizer_count": int(
                _summary(simple_formula_batch).get("selected_simple_formula_optimizer_count", 0) or 0
            ),
            "selected_simple_formula_reference_canary_ready_count": int(
                _summary(simple_formula_batch).get("reference_canary_ready_count", 0) or 0
            ),
            "selected_simple_formula_native_canary_ready_count": int(
                _summary(simple_formula_batch).get("selected_plugin_native_canary_ready_count", 0) or 0
            ),
            "selected_simple_formula_e2e_shadow_matrix_ready": simple_formula_e2e.get(
                "e2e_shadow_matrix_ready"
            )
            is True,
            "selected_simple_formula_e2e_shadow_case_count": int(
                _summary(simple_formula_e2e).get("case_count", 0) or 0
            ),
            "selected_simple_formula_canary_rollout_policy_ready": simple_formula_rollout.get(
                "canary_rollout_policy_ready"
            )
            is True,
            "selected_simple_formula_canary_rollout_policy_ready_count": int(
                _summary(simple_formula_rollout).get("canary_rollout_policy_ready_count", 0) or 0
            ),
            "selected_simple_formula_dispatch_review_gate_ready": simple_formula_dispatch.get(
                "review_gate_ready"
            )
            is True,
            "selected_simple_formula_dispatch_review_ready_count": int(
                _summary(simple_formula_dispatch).get("optimizer_count", 0) or 0
            )
            if simple_formula_dispatch.get("review_gate_ready") is True
            else 0,
            "selected_simple_formula_owner_release_hold_ready": simple_formula_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_simple_formula_owner_release_hold_optimizer_count": int(
                _summary(simple_formula_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_simple_formula_owner_release_hold_product_native_ready_count": int(
                _summary(simple_formula_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_simple_formula_request_schema_ui_non_exposure_ready": (
                simple_formula_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_simple_formula_request_schema_ui_optimizer_count": int(
                _summary(simple_formula_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_simple_formula_request_schema_ui_forbidden_token_hit_count": int(
                _summary(simple_formula_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_simple_formula_request_schema_ui_product_native_ready_count": int(
                _summary(simple_formula_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_closure_second_order_family_batch_ready": closure_second_order_batch.get(
                "selected_closure_second_order_family_batch_ready"
            )
            is True,
            "selected_closure_second_order_optimizer_count": int(
                _summary(closure_second_order_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_closure_second_order_higher_order_abi_required_count": int(
                _summary(closure_second_order_batch).get("higher_order_training_loop_abi_required_count", 0) or 0
            ),
            "selected_closure_second_order_training_loop_abi_spec_ready_count": int(
                _summary(closure_second_order_batch).get("training_loop_abi_spec_ready_count", 0) or 0
            ),
            "selected_closure_second_order_training_loop_abi_implementation_ready_count": int(
                _summary(closure_second_order_batch).get("training_loop_abi_implementation_ready_count", 0) or 0
            ),
            "selected_closure_second_order_resume_parity_matrix_spec_ready_count": int(
                _summary(closure_second_order_batch).get("resume_parity_matrix_spec_ready_count", 0) or 0
            ),
            "selected_closure_second_order_resume_parity_matrix_implementation_ready_count": int(
                _summary(closure_second_order_batch).get(
                    "resume_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_closure_second_order_closure_replay_case_planned_count": int(
                _summary(closure_second_order_batch).get("closure_replay_case_planned_count", 0) or 0
            ),
            "selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count": int(
                _summary(closure_second_order_batch).get("create_graph_hvp_lifetime_case_planned_count", 0) or 0
            ),
            "selected_closure_second_order_closure_resume_replay_artifact_ready_count": int(
                _summary(closure_second_order_batch).get("closure_resume_replay_artifact_ready_count", 0) or 0
            ),
            "selected_closure_second_order_closure_resume_replay_artifact_row_count": int(
                _summary(closure_second_order_batch).get("closure_resume_replay_artifact_row_count", 0) or 0
            ),
            "selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count": int(
                _summary(closure_second_order_batch).get(
                    "closure_resume_replay_artifact_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_closure_second_order_closure_resume_replay_row_implementation_ready_count": int(
                _summary(closure_second_order_batch).get("closure_resume_replay_row_implementation_ready_count", 0)
                or 0
            ),
            "selected_closure_second_order_native_kernel_precondition_plan_ready_count": int(
                _summary(closure_second_order_batch).get("native_kernel_precondition_plan_ready_count", 0) or 0
            ),
            "selected_closure_second_order_native_kernel_preconditions_implementation_ready_count": int(
                _summary(closure_second_order_batch).get(
                    "native_kernel_preconditions_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_closure_second_order_owner_release_hold_ready": closure_second_order_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_closure_second_order_owner_release_hold_optimizer_count": int(
                _summary(closure_second_order_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_closure_second_order_owner_release_hold_product_native_ready_count": int(
                _summary(closure_second_order_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_closure_second_order_request_schema_ui_non_exposure_ready": (
                closure_second_order_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_closure_second_order_request_schema_ui_optimizer_count": int(
                _summary(closure_second_order_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_closure_second_order_request_schema_ui_forbidden_token_hit_count": int(
                _summary(closure_second_order_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_closure_second_order_request_schema_ui_product_native_ready_count": int(
                _summary(closure_second_order_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_custom_formula_family_batch_ready": custom_formula_batch.get(
                "selected_custom_formula_family_batch_ready"
            )
            is True,
            "selected_custom_formula_optimizer_count": int(
                _summary(custom_formula_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_custom_formula_parity_required_count": int(
                _summary(custom_formula_batch).get("formula_parity_required_count", 0) or 0
            ),
            "selected_custom_formula_backlog_ready_count": int(
                _summary(custom_formula_batch).get("backlog_ready_count", 0) or 0
            ),
            "selected_custom_formula_evidence_artifact_planned_count": int(
                _summary(custom_formula_batch).get("evidence_artifact_planned_count", 0) or 0
            ),
            "selected_custom_formula_evidence_status_pending_total": int(
                _summary(custom_formula_batch).get("evidence_status_pending_total", 0) or 0
            ),
            "selected_custom_formula_formula_spec_artifact_ready_count": int(
                _summary(custom_formula_batch).get("formula_spec_artifact_ready_count", 0) or 0
            ),
            "selected_custom_formula_formula_spec_artifact_pending_count": int(
                _summary(custom_formula_batch).get("formula_spec_artifact_pending_count", 0) or 0
            ),
            "selected_custom_formula_state_inventory_skeleton_count": int(
                _summary(custom_formula_batch).get("formula_state_inventory_skeleton_count", 0) or 0
            ),
            "selected_custom_formula_state_inventory_artifact_ready_count": int(
                _summary(custom_formula_batch).get("state_inventory_artifact_ready_count", 0) or 0
            ),
            "selected_custom_formula_state_inventory_artifact_pending_count": int(
                _summary(custom_formula_batch).get("state_inventory_artifact_pending_count", 0) or 0
            ),
            "selected_custom_formula_quality_guard_matrix_artifact_ready_count": int(
                _summary(custom_formula_batch).get("quality_guard_matrix_artifact_ready_count", 0) or 0
            ),
            "selected_custom_formula_quality_guard_matrix_artifact_pending_count": int(
                _summary(custom_formula_batch).get("quality_guard_matrix_artifact_pending_count", 0) or 0
            ),
            "selected_custom_formula_quality_guard_matrix_case_planned_count": int(
                _summary(custom_formula_batch).get("quality_guard_matrix_case_planned_count", 0) or 0
            ),
            "selected_custom_formula_formula_parity_matrix_artifact_planned_count": int(
                _summary(custom_formula_batch).get("formula_parity_matrix_artifact_planned_count", 0) or 0
            ),
            "selected_custom_formula_formula_parity_matrix_implementation_ready_count": int(
                _summary(custom_formula_batch).get("formula_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "selected_custom_formula_formula_parity_case_planned_count": int(
                _summary(custom_formula_batch).get("formula_parity_case_planned_count", 0) or 0
            ),
            "selected_custom_formula_resume_parity_matrix_artifact_planned_count": int(
                _summary(custom_formula_batch).get("resume_parity_matrix_artifact_planned_count", 0) or 0
            ),
            "selected_custom_formula_resume_parity_matrix_implementation_ready_count": int(
                _summary(custom_formula_batch).get("resume_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "selected_custom_formula_resume_parity_case_planned_count": int(
                _summary(custom_formula_batch).get("resume_parity_case_planned_count", 0) or 0
            ),
            "selected_custom_formula_execution_matrix_ready": _summary(custom_formula_batch).get(
                "execution_matrix_ready"
            )
            is True,
            "selected_custom_formula_step_execution_ready_count": int(
                _summary(custom_formula_batch).get("formula_step_execution_ready_count", 0) or 0
            ),
            "selected_custom_formula_resume_next_step_replay_ready_count": int(
                _summary(custom_formula_batch).get("resume_next_step_replay_ready_count", 0) or 0
            ),
            "selected_custom_formula_execution_failed_count": int(
                _summary(custom_formula_batch).get("execution_failed_count", 0) or 0
            ),
            "selected_custom_formula_owner_release_hold_ready": custom_formula_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_custom_formula_owner_release_hold_optimizer_count": int(
                _summary(custom_formula_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_custom_formula_owner_release_hold_product_native_ready_count": int(
                _summary(custom_formula_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_custom_formula_request_schema_ui_non_exposure_ready": custom_formula_request_schema_ui.get(
                "request_schema_ui_non_exposure_ready"
            )
            is True,
            "selected_custom_formula_request_schema_ui_optimizer_count": int(
                _summary(custom_formula_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_custom_formula_request_schema_ui_forbidden_token_hit_count": int(
                _summary(custom_formula_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_custom_formula_request_schema_ui_product_native_ready_count": int(
                _summary(custom_formula_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_factored_memory_family_batch_ready": factored_memory_batch.get(
                "selected_factored_memory_family_batch_ready"
            )
            is True,
            "selected_factored_memory_optimizer_count": int(
                _summary(factored_memory_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_factored_memory_observed_layout_count": int(
                _summary(factored_memory_batch).get("observed_resume_layout_count", 0) or 0
            ),
            "selected_factored_memory_native_layout_abi_ready_count": int(
                _summary(factored_memory_batch).get("native_layout_abi_ready_count", 0) or 0
            ),
            "selected_factored_memory_quality_matrix_ready_count": int(
                _summary(factored_memory_batch).get("quality_matrix_ready_count", 0) or 0
            ),
            "selected_factored_memory_native_kernel_entry_condition_ready_count": int(
                _summary(factored_memory_batch).get("native_kernel_entry_condition_ready_count", 0) or 0
            ),
            "selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count": int(
                _summary(factored_memory_batch).get("formula_tensor_binding_matrix_artifact_ready_count", 0) or 0
            ),
            "selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count": int(
                _summary(factored_memory_batch).get(
                    "formula_tensor_binding_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_factored_memory_formula_step_execution_ready_count": int(
                _summary(factored_memory_batch).get("formula_step_execution_ready_count", 0) or 0
            ),
            "selected_factored_memory_resume_next_step_replay_ready_count": int(
                _summary(factored_memory_batch).get("resume_next_step_replay_ready_count", 0) or 0
            ),
            "selected_factored_memory_tensor_binding_ready_count": int(
                _summary(factored_memory_batch).get("tensor_binding_ready_count", 0) or 0
            ),
            "selected_factored_memory_dispatch_review_gate_ready": _summary(factored_memory_batch).get(
                "dispatch_review_gate_ready"
            )
            is True,
            "selected_factored_memory_dispatch_review_ready_count": int(
                _summary(factored_memory_batch).get("dispatch_review_ready_count", 0) or 0
            ),
            "selected_factored_memory_formula_parity_case_planned_count": int(
                _summary(factored_memory_batch).get("formula_parity_case_planned_count", 0) or 0
            ),
            "selected_factored_memory_tensor_binding_case_planned_count": int(
                _summary(factored_memory_batch).get("tensor_binding_case_planned_count", 0) or 0
            ),
            "selected_factored_memory_owner_release_hold_ready": factored_memory_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_factored_memory_owner_release_hold_optimizer_count": int(
                _summary(factored_memory_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_factored_memory_owner_release_hold_product_native_ready_count": int(
                _summary(factored_memory_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_factored_memory_request_schema_ui_non_exposure_ready": factored_memory_request_schema_ui.get(
                "request_schema_ui_non_exposure_ready"
            )
            is True,
            "selected_factored_memory_request_schema_ui_optimizer_count": int(
                _summary(factored_memory_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_factored_memory_request_schema_ui_forbidden_token_hit_count": int(
                _summary(factored_memory_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_factored_memory_request_schema_ui_product_native_ready_count": int(
                _summary(factored_memory_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_fused_backward_family_batch_ready": fused_backward_batch.get(
                "selected_fused_backward_family_batch_ready"
            )
            is True,
            "selected_fused_backward_optimizer_count": int(
                _summary(fused_backward_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_fused_backward_gradient_ownership_abi_required_count": int(
                _summary(fused_backward_batch).get("gradient_ownership_abi_required_count", 0) or 0
            ),
            "selected_fused_backward_per_optimizer_abi_spec_ready_count": int(
                _summary(fused_backward_batch).get("per_optimizer_abi_spec_ready_count", 0) or 0
            ),
            "selected_fused_backward_abi_implementation_ready_count": int(
                _summary(fused_backward_batch).get("fused_backward_abi_implementation_ready_count", 0) or 0
            ),
            "selected_fused_backward_native_kernel_preconditions_spec_ready_count": int(
                _summary(fused_backward_batch).get("native_kernel_preconditions_spec_ready_count", 0) or 0
            ),
            "selected_fused_backward_resume_parity_matrix_spec_ready_count": int(
                _summary(fused_backward_batch).get("resume_parity_matrix_spec_ready_count", 0) or 0
            ),
            "selected_fused_backward_resume_parity_matrix_implementation_ready_count": int(
                _summary(fused_backward_batch).get("resume_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "selected_fused_backward_replay_case_planned_count": int(
                _summary(fused_backward_batch).get("fused_backward_replay_case_planned_count", 0) or 0
            ),
            "selected_fused_backward_replay_case_implementation_ready_count": int(
                _summary(fused_backward_batch).get("fused_backward_replay_case_implementation_ready_count", 0) or 0
            ),
            "selected_fused_backward_loss_scale_boundary_case_planned_count": int(
                _summary(fused_backward_batch).get("loss_scale_boundary_case_planned_count", 0) or 0
            ),
            "selected_fused_backward_owner_release_hold_ready": fused_backward_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_fused_backward_owner_release_hold_optimizer_count": int(
                _summary(fused_backward_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_fused_backward_owner_release_hold_product_native_ready_count": int(
                _summary(fused_backward_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_fused_backward_request_schema_ui_non_exposure_ready": fused_backward_request_schema_ui.get(
                "request_schema_ui_non_exposure_ready"
            )
            is True,
            "selected_fused_backward_request_schema_ui_optimizer_count": int(
                _summary(fused_backward_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_fused_backward_request_schema_ui_forbidden_token_hit_count": int(
                _summary(fused_backward_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_fused_backward_request_schema_ui_product_native_ready_count": int(
                _summary(fused_backward_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_family_batch_ready": model_shape_batch.get(
                "selected_model_shape_aware_family_batch_ready"
            )
            is True,
            "selected_model_shape_aware_optimizer_count": int(
                _summary(model_shape_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_contract_count": int(
                _summary(model_shape_batch).get("param_group_semantics_dependent_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_abi_spec_ready_count": int(
                _summary(model_shape_batch).get("param_group_abi_spec_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_abi_implementation_ready_count": int(
                _summary(model_shape_batch).get("param_group_abi_implementation_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count": int(
                _summary(model_shape_batch).get("param_group_resume_replay_matrix_artifact_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_resume_replay_matrix_row_count": int(
                _summary(model_shape_batch).get("param_group_resume_replay_matrix_row_count", 0) or 0
            ),
            "selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count": int(
                _summary(model_shape_batch).get(
                    "param_group_resume_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count": int(
                _summary(model_shape_batch).get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_owner_release_hold_ready": model_shape_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_model_shape_aware_owner_release_hold_optimizer_count": int(
                _summary(model_shape_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_model_shape_aware_owner_release_hold_product_native_ready_count": int(
                _summary(model_shape_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_model_shape_aware_request_schema_ui_non_exposure_ready": model_shape_request_schema_ui.get(
                "request_schema_ui_non_exposure_ready"
            )
            is True,
            "selected_model_shape_aware_request_schema_ui_optimizer_count": int(
                _summary(model_shape_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_model_shape_aware_request_schema_ui_forbidden_token_hit_count": int(
                _summary(model_shape_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_model_shape_aware_request_schema_ui_product_native_ready_count": int(
                _summary(model_shape_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_family_batch_ready": state_adapter_special_batch.get(
                "selected_state_adapter_special_family_batch_ready"
            )
            is True,
            "selected_state_adapter_special_optimizer_count": int(
                _summary(state_adapter_special_batch).get("selected_optimizer_count", 0) or 0
            ),
            "selected_state_adapter_special_param_ownership_abi_required_count": int(
                _summary(state_adapter_special_batch).get("param_ownership_abi_required_count", 0) or 0
            ),
            "selected_state_adapter_special_adapter_abi_spec_ready_count": int(
                _summary(state_adapter_special_batch).get("adapter_abi_spec_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_adapter_abi_implementation_ready_count": int(
                _summary(state_adapter_special_batch).get("adapter_abi_implementation_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_native_kernel_precondition_spec_ready_count": int(
                _summary(state_adapter_special_batch).get("native_kernel_precondition_spec_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_native_kernel_precondition_implementation_ready_count": int(
                _summary(state_adapter_special_batch).get(
                    "native_kernel_precondition_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_state_adapter_special_resume_matrix_artifact_ready_count": int(
                _summary(state_adapter_special_batch).get("adapter_resume_matrix_artifact_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_resume_matrix_implementation_ready_count": int(
                _summary(state_adapter_special_batch).get("adapter_resume_matrix_implementation_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_resume_replay_case_planned_count": int(
                _summary(state_adapter_special_batch).get("adapter_resume_replay_case_planned_count", 0) or 0
            ),
            "selected_state_adapter_special_resume_replay_case_implementation_ready_count": int(
                _summary(state_adapter_special_batch).get(
                    "adapter_resume_replay_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_state_adapter_special_resume_translation_case_planned_count": int(
                _summary(state_adapter_special_batch).get("adapter_resume_translation_case_planned_count", 0) or 0
            ),
            "selected_state_adapter_special_resume_translation_case_implementation_ready_count": int(
                _summary(state_adapter_special_batch).get(
                    "adapter_resume_translation_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_state_adapter_special_owner_release_hold_ready": state_adapter_special_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "selected_state_adapter_special_owner_release_hold_optimizer_count": int(
                _summary(state_adapter_special_owner_hold).get("optimizer_count", 0) or 0
            ),
            "selected_state_adapter_special_owner_release_hold_product_native_ready_count": int(
                _summary(state_adapter_special_owner_hold).get("product_native_ready_count", 0) or 0
            ),
            "selected_state_adapter_special_request_schema_ui_non_exposure_ready": (
                state_adapter_special_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            ),
            "selected_state_adapter_special_request_schema_ui_optimizer_count": int(
                _summary(state_adapter_special_request_schema_ui).get("optimizer_count", 0) or 0
            ),
            "selected_state_adapter_special_request_schema_ui_forbidden_token_hit_count": int(
                _summary(state_adapter_special_request_schema_ui).get("forbidden_token_hit_count", 0) or 0
            ),
            "selected_state_adapter_special_request_schema_ui_product_native_ready_count": int(
                _summary(state_adapter_special_request_schema_ui).get("product_native_ready_count", 0) or 0
            ),
            "builtin_factored_layout_reference_count": int(_summary(factored).get("optimizer_count", 0) or 0),
            "plugin_factored_memory_layout_observed_count": int(
                _summary(plugin_factored).get("observed_resume_layout_count", 0) or 0
            ),
            "plugin_factored_memory_manual_pending_count": int(
                _summary(plugin_factored).get("manual_contract_pending_count", 0) or 0
            ),
            "plugin_selected_native_ready_count": 0,
            "plugin_selected_runtime_dispatch_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "selected_plugin_family_native_kernel_matrix_missing",
            "selected_plugin_family_runtime_dispatch_missing",
            "owner_release_hold_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(family_rows) if ready else (
            "fix plugin optimizer family batch source failures or unsafe dispatch claims"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Selected AdamW plugin compatibility is not a blanket AdamW-like native approval.",
            "Factored/custom layout evidence is built-in reference evidence, not a plugin selected-kernel claim.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _family_rows(
    selector: Mapping[str, Any],
    adamlike: Mapping[str, Any],
    adamlike_batch: Mapping[str, Any],
    adamlike_owner_hold: Mapping[str, Any],
    adamlike_request_schema_ui: Mapping[str, Any],
    schedulefree: Mapping[str, Any],
    schedulefree_batch: Mapping[str, Any],
    schedulefree_owner_hold: Mapping[str, Any],
    schedulefree_request_schema_ui: Mapping[str, Any],
    adaptivelr_batch: Mapping[str, Any],
    adaptivelr_owner_hold: Mapping[str, Any],
    adaptivelr_request_schema_ui: Mapping[str, Any],
    simple_formula_batch: Mapping[str, Any],
    simple_formula_owner_hold: Mapping[str, Any],
    simple_formula_request_schema_ui: Mapping[str, Any],
    closure_second_order_batch: Mapping[str, Any],
    closure_second_order_owner_hold: Mapping[str, Any],
    closure_second_order_request_schema_ui: Mapping[str, Any],
    custom_formula_batch: Mapping[str, Any],
    custom_formula_owner_hold: Mapping[str, Any],
    custom_formula_request_schema_ui: Mapping[str, Any],
    factored_memory_batch: Mapping[str, Any],
    factored_memory_owner_hold: Mapping[str, Any],
    factored_memory_request_schema_ui: Mapping[str, Any],
    fused_backward_batch: Mapping[str, Any],
    fused_backward_owner_hold: Mapping[str, Any],
    fused_backward_request_schema_ui: Mapping[str, Any],
    model_shape_batch: Mapping[str, Any],
    model_shape_owner_hold: Mapping[str, Any],
    model_shape_request_schema_ui: Mapping[str, Any],
    state_adapter_special_batch: Mapping[str, Any],
    state_adapter_special_owner_hold: Mapping[str, Any],
    state_adapter_special_request_schema_ui: Mapping[str, Any],
    factored: Mapping[str, Any],
    plugin_factored: Mapping[str, Any],
) -> list[dict[str, Any]]:
    counts = dict(_as_dict(_summary(selector).get("route_family_counts")))
    rows: list[dict[str, Any]] = []
    for family, count in sorted(counts.items()):
        gate = _SELECTED_GATE_BY_FAMILY.get(str(family), "selected_gate_pending")
        ready = _selected_ready(
            str(family),
            adamlike,
            schedulefree,
            schedulefree_batch,
            adaptivelr_batch,
            simple_formula_batch,
            closure_second_order_batch,
            custom_formula_batch,
            factored_memory_batch,
            fused_backward_batch,
            model_shape_batch,
            state_adapter_special_batch,
        )
        layout_reference = str(family) == "factored_memory_layout" and factored.get("state_layout_reference_ready") is True
        plugin_layout_observed = _plugin_factored_layout_observed_count(str(family), plugin_factored)
        rows.append(
            {
                "native_route_family": str(family),
                "plugin_optimizer_count": int(count or 0),
                "selector_classified": selector.get("plugin_selector_classification_ready") is True,
                "selected_optimizer_gate": gate,
                "selected_optimizer_gate_ready": ready,
                "selected_native_canary_ready_count": _selected_native_canary_ready_count(
                    str(family),
                    adamlike_batch,
                    schedulefree_batch,
                    simple_formula_batch,
                ),
                "builtin_layout_reference_available": layout_reference,
                "plugin_state_layout_observed_count": plugin_layout_observed,
                "native_dispatch_allowed": False,
                "runtime_dispatch_ready": False,
                "training_path_enabled": False,
                "default_behavior_changed": False,
                "next_gate": _next_gate(
                    str(family),
                    ready,
                    layout_reference,
                    adamlike_owner_hold,
                    adamlike_request_schema_ui,
                    adaptivelr_batch,
                    adaptivelr_owner_hold,
                    adaptivelr_request_schema_ui,
                    custom_formula_batch,
                    factored_memory_batch,
                    fused_backward_batch,
                    model_shape_batch,
                    state_adapter_special_batch,
                    schedulefree_owner_hold,
                    schedulefree_request_schema_ui,
                    simple_formula_owner_hold,
                    simple_formula_request_schema_ui,
                    closure_second_order_batch,
                    closure_second_order_owner_hold,
                    closure_second_order_request_schema_ui,
                    custom_formula_owner_hold,
                    custom_formula_request_schema_ui,
                    factored_memory_owner_hold,
                    factored_memory_request_schema_ui,
                    fused_backward_owner_hold,
                    fused_backward_request_schema_ui,
                    model_shape_owner_hold,
                    model_shape_request_schema_ui,
                    state_adapter_special_owner_hold,
                    state_adapter_special_request_schema_ui,
                ),
            }
        )
    return rows


def _recommended_next_step(family_rows: list[Mapping[str, Any]]) -> str:
    if family_rows and all(row.get("selected_optimizer_gate_ready") is True for row in family_rows):
        return "prepare plugin selected-family owner/release hold with product dispatch still default-off"
    for family in (
        "custom_formula",
        "factored_memory_layout",
        "model_or_shape_aware",
        "closure_or_second_order",
        "fused_backward",
        "state_adapter_special",
        "adaptive_lr_state_machine",
        "simple_formula",
        "adam_like_formula",
        "schedule_free_state_machine",
    ):
        for row in family_rows:
            if row.get("native_route_family") == family:
                next_gate = str(row.get("next_gate"))
                if "manual dispatch" in next_gate.lower():
                    continue
                return f"continue {family} with {next_gate}"
    return "continue selected plugin implementation/replay matrices while keeping dispatch default-off"


def _selected_native_canary_ready_count(
    family: str,
    adamlike_batch: Mapping[str, Any],
    schedulefree_batch: Mapping[str, Any],
    simple_formula_batch: Mapping[str, Any],
) -> int:
    if family == "adam_like_formula":
        return int(_summary(adamlike_batch).get("selected_native_canary_ready_count", 0) or 0)
    if family == "schedule_free_state_machine":
        return int(_summary(schedulefree_batch).get("selected_native_canary_ready_count", 0) or 0)
    if family == "simple_formula":
        return int(_summary(simple_formula_batch).get("selected_plugin_native_canary_ready_count", 0) or 0)
    return 0


def _plugin_factored_layout_observed_count(family: str, plugin_factored: Mapping[str, Any]) -> int:
    if family != "factored_memory_layout":
        return 0
    return int(_summary(plugin_factored).get("observed_resume_layout_count", 0) or 0)


def _selected_ready(
    family: str,
    adamlike: Mapping[str, Any],
    schedulefree: Mapping[str, Any],
    schedulefree_batch: Mapping[str, Any],
    adaptivelr_batch: Mapping[str, Any],
    simple_formula_batch: Mapping[str, Any],
    closure_second_order_batch: Mapping[str, Any],
    custom_formula_batch: Mapping[str, Any],
    factored_memory_batch: Mapping[str, Any],
    fused_backward_batch: Mapping[str, Any],
    model_shape_batch: Mapping[str, Any],
    state_adapter_special_batch: Mapping[str, Any],
) -> bool:
    if family == "adam_like_formula":
        return adamlike.get("selected_optimizer_abi_ready") is True
    if family == "adaptive_lr_state_machine":
        return adaptivelr_batch.get("selected_adaptivelr_family_batch_ready") is True
    if family == "schedule_free_state_machine":
        return (
            schedulefree.get("selected_optimizer_abi_ready") is True
            and schedulefree_batch.get("selected_schedulefree_family_batch_ready") is True
        )
    if family == "simple_formula":
        return simple_formula_batch.get("selected_simple_formula_family_batch_ready") is True
    if family == "closure_or_second_order":
        return closure_second_order_batch.get("selected_closure_second_order_family_batch_ready") is True
    if family == "custom_formula":
        return custom_formula_batch.get("selected_custom_formula_family_batch_ready") is True
    if family == "factored_memory_layout":
        return factored_memory_batch.get("selected_factored_memory_family_batch_ready") is True
    if family == "fused_backward":
        return fused_backward_batch.get("selected_fused_backward_family_batch_ready") is True
    if family == "model_or_shape_aware":
        return model_shape_batch.get("selected_model_shape_aware_family_batch_ready") is True
    if family == "state_adapter_special":
        return state_adapter_special_batch.get("selected_state_adapter_special_family_batch_ready") is True
    return False


def _next_gate(
    family: str,
    ready: bool,
    layout_reference: bool,
    adamlike_owner_hold: Mapping[str, Any],
    adamlike_request_schema_ui: Mapping[str, Any],
    adaptivelr_batch: Mapping[str, Any],
    adaptivelr_owner_hold: Mapping[str, Any],
    adaptivelr_request_schema_ui: Mapping[str, Any],
    custom_formula_batch: Mapping[str, Any],
    factored_memory_batch: Mapping[str, Any],
    fused_backward_batch: Mapping[str, Any],
    model_shape_batch: Mapping[str, Any],
    state_adapter_special_batch: Mapping[str, Any],
    schedulefree_owner_hold: Mapping[str, Any],
    schedulefree_request_schema_ui: Mapping[str, Any],
    simple_formula_owner_hold: Mapping[str, Any],
    simple_formula_request_schema_ui: Mapping[str, Any],
    closure_second_order_batch: Mapping[str, Any],
    closure_second_order_owner_hold: Mapping[str, Any],
    closure_second_order_request_schema_ui: Mapping[str, Any],
    custom_formula_owner_hold: Mapping[str, Any],
    custom_formula_request_schema_ui: Mapping[str, Any],
    factored_memory_owner_hold: Mapping[str, Any],
    factored_memory_request_schema_ui: Mapping[str, Any],
    fused_backward_owner_hold: Mapping[str, Any],
    fused_backward_request_schema_ui: Mapping[str, Any],
    model_shape_owner_hold: Mapping[str, Any],
    model_shape_request_schema_ui: Mapping[str, Any],
    state_adapter_special_owner_hold: Mapping[str, Any],
    state_adapter_special_request_schema_ui: Mapping[str, Any],
) -> str:
    if ready:
        if family == "adam_like_formula" and adamlike_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep Adam-like native dispatch unwired until explicit owner/release approval is recorded"
        if family == "adam_like_formula" and adamlike_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for Adam-like with dispatch still default-off"
        if family == "adaptive_lr_state_machine" and adaptivelr_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep adaptive-LR native dispatch unwired until explicit owner/release approval is recorded"
        if family == "adaptive_lr_state_machine" and adaptivelr_owner_hold.get(
            "owner_release_hold_ready"
        ) is True:
            return "request/schema/UI non-exposure gate for adaptive-LR with dispatch still default-off"
        if family == "schedule_free_state_machine" and schedulefree_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep schedule-free native dispatch unwired until explicit owner/release approval is recorded"
        if family == "schedule_free_state_machine" and schedulefree_owner_hold.get(
            "owner_release_hold_ready"
        ) is True:
            return "request/schema/UI non-exposure gate for schedule-free with dispatch still default-off"
        if family == "simple_formula" and simple_formula_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep simple-formula native dispatch unwired until explicit owner/release approval is recorded"
        if family == "simple_formula" and simple_formula_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for simple-formula with dispatch still default-off"
        if family == "state_adapter_special" and state_adapter_special_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep state-adapter-special native dispatch unwired until explicit owner/release approval is recorded"
        if family == "state_adapter_special" and state_adapter_special_owner_hold.get(
            "owner_release_hold_ready"
        ) is True:
            return "request/schema/UI non-exposure gate for state-adapter-special with dispatch still default-off"
        if family == "model_or_shape_aware" and model_shape_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep model/shape-aware native dispatch unwired until explicit owner/release approval is recorded"
        if family == "model_or_shape_aware" and model_shape_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for model/shape-aware with dispatch still default-off"
        if family == "fused_backward" and fused_backward_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep fused-backward native dispatch unwired until explicit owner/release approval is recorded"
        if family == "fused_backward" and fused_backward_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for fused-backward with dispatch still default-off"
        if family == "factored_memory_layout" and factored_memory_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep factored-memory native dispatch unwired until explicit owner/release approval is recorded"
        if family == "factored_memory_layout" and factored_memory_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for factored-memory with dispatch still default-off"
        if family == "custom_formula" and custom_formula_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep custom-formula native dispatch unwired until explicit owner/release approval is recorded"
        if family == "custom_formula" and custom_formula_owner_hold.get("owner_release_hold_ready") is True:
            return "request/schema/UI non-exposure gate for custom-formula with dispatch still default-off"
        if family == "closure_or_second_order" and closure_second_order_request_schema_ui.get(
            "request_schema_ui_non_exposure_ready"
        ) is True:
            return "keep closure/second-order native dispatch unwired until explicit owner/release approval is recorded"
        if family == "closure_or_second_order" and closure_second_order_owner_hold.get(
            "owner_release_hold_ready"
        ) is True:
            return "request/schema/UI non-exposure gate for closure/second-order with dispatch still default-off"
        if family == "adaptive_lr_state_machine" and _adaptivelr_implementation_ready(adaptivelr_batch):
            return "owner/release hold for implementation-ready adaptive-LR matrices with dispatch default-off"
        if family == "custom_formula" and _custom_formula_implementation_ready(custom_formula_batch):
            return "owner/release hold for implementation-ready parity/resume matrices with dispatch default-off"
        if family == "factored_memory_layout" and _factored_memory_implementation_ready(factored_memory_batch):
            return "owner/release hold for implementation-ready formula/tensor-binding matrices with dispatch default-off"
        if family == "model_or_shape_aware" and _model_shape_implementation_ready(model_shape_batch):
            return "owner/release hold for implementation-ready param-group ABI matrices with dispatch default-off"
        if family == "closure_or_second_order" and _closure_second_order_implementation_ready(
            closure_second_order_batch
        ):
            return "owner/release hold for implementation-ready closure/second-order ABI matrices with dispatch default-off"
        if family == "fused_backward" and _fused_backward_implementation_ready(fused_backward_batch):
            return "owner/release hold for implementation-ready fused-backward ABI matrices with dispatch default-off"
        if family == "state_adapter_special" and _state_adapter_special_implementation_ready(
            state_adapter_special_batch
        ):
            return "owner/release hold for implementation-ready state-adapter ABI matrices with dispatch default-off"
        mapping = {
            "custom_formula": "formula and resume replay implementation matrix",
            "adaptive_lr_state_machine": "state-machine ABI implementation and replay matrix",
            "factored_memory_layout": "formula/tensor-binding implementation matrix",
            "fused_backward": "fused-backward ABI implementation and replay matrix",
            "model_or_shape_aware": "param-group ABI implementation and replay matrix",
            "state_adapter_special": "adapter ABI implementation and resume translation matrix",
            "closure_or_second_order": "closure replay implementation matrix",
            "simple_formula": "selected-plugin owner/release hold with dispatch default-off",
            "adam_like_formula": "selected-plugin owner/release hold for ready canaries with dispatch default-off",
            "schedule_free_state_machine": "selected-plugin owner/release hold for ready canaries with dispatch default-off",
        }
        return mapping.get(family, "selected_family_training_tensor_binding_or_kernel_canary")
    if layout_reference:
        return "selected_plugin_factored_memory_abi_gate"
    mapping = {
        "adaptive_lr_state_machine": "selected_plugin_adaptive_lr_state_machine_abi_gate",
        "simple_formula": "selected_plugin_simple_formula_parity_gate",
        "factored_memory_layout": "selected_plugin_factored_memory_layout_gate",
        "closure_or_second_order": "closure_create_graph_training_loop_abi",
        "fused_backward": "fused_backward_ownership_abi",
        "model_or_shape_aware": "model_shape_aware_param_group_abi",
        "state_adapter_special": "special_state_adapter_resume_abi",
    }
    return mapping.get(family, "selected_plugin_formula_state_reference_gate")


def _custom_formula_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    formula = int(summary.get("formula_parity_matrix_implementation_ready_count", 0) or 0)
    resume = int(summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0)
    return count > 0 and formula == count and resume == count


def _adaptivelr_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    abi = int(summary.get("selected_state_machine_abi_implementation_ready_count", 0) or 0)
    replay = int(summary.get("selected_state_machine_replay_matrix_implementation_ready_count", 0) or 0)
    replay_cases = int(summary.get("selected_state_machine_replay_case_implementation_ready_count", 0) or 0)
    replay_planned = int(summary.get("selected_state_machine_replay_case_planned_count", 0) or 0)
    resume_cases = int(summary.get("selected_state_machine_replay_resume_case_implementation_ready_count", 0) or 0)
    resume_planned = int(summary.get("selected_state_machine_replay_resume_case_planned_count", 0) or 0)
    preconditions = int(summary.get("selected_native_kernel_preconditions_implementation_ready_count", 0) or 0)
    return (
        count > 0
        and abi == count
        and replay == count
        and preconditions == count
        and replay_planned > 0
        and replay_cases == replay_planned
        and resume_planned > 0
        and resume_cases == resume_planned
    )


def _factored_memory_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    ready = int(summary.get("formula_tensor_binding_matrix_implementation_ready_count", 0) or 0)
    return count > 0 and ready == count


def _model_shape_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    abi = int(summary.get("param_group_abi_implementation_ready_count", 0) or 0)
    replay = int(summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0)
    return count > 0 and abi == count and replay == count


def _closure_second_order_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    abi = int(summary.get("training_loop_abi_implementation_ready_count", 0) or 0)
    replay = int(summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0)
    return count > 0 and abi == count and replay == count


def _fused_backward_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    abi = int(summary.get("fused_backward_abi_implementation_ready_count", 0) or 0)
    resume = int(summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0)
    cases = int(summary.get("fused_backward_replay_case_implementation_ready_count", 0) or 0)
    planned = int(summary.get("fused_backward_replay_case_planned_count", 0) or 0)
    preconditions = int(summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0)
    return count > 0 and abi == count and resume == count and preconditions == count and planned > 0 and cases == planned


def _state_adapter_special_implementation_ready(report: Mapping[str, Any]) -> bool:
    summary = _summary(report)
    count = int(summary.get("selected_optimizer_count", 0) or 0)
    abi = int(summary.get("adapter_abi_implementation_ready_count", 0) or 0)
    resume = int(summary.get("adapter_resume_matrix_implementation_ready_count", 0) or 0)
    replay_cases = int(summary.get("adapter_resume_replay_case_implementation_ready_count", 0) or 0)
    replay_planned = int(summary.get("adapter_resume_replay_case_planned_count", 0) or 0)
    translation_cases = int(summary.get("adapter_resume_translation_case_implementation_ready_count", 0) or 0)
    translation_planned = int(summary.get("adapter_resume_translation_case_planned_count", 0) or 0)
    preconditions = int(summary.get("native_kernel_precondition_implementation_ready_count", 0) or 0)
    return (
        count > 0
        and abi == count
        and resume == count
        and preconditions == count
        and replay_planned > 0
        and replay_cases == replay_planned
        and translation_planned > 0
        and translation_cases == translation_planned
    )


def _failed_sources(*reports: Mapping[str, Any]) -> list[str]:
    names = (
        "plugin_selector_scorecard_not_ok",
        "plugin_adamlike_selected_scorecard_not_ok",
        "plugin_adamlike_owner_release_hold_scorecard_not_ok",
        "plugin_adamlike_request_schema_ui_scorecard_not_ok",
        "plugin_schedulefree_selected_scorecard_not_ok",
        "plugin_schedulefree_family_batch_scorecard_not_ok",
        "plugin_schedulefree_owner_release_hold_scorecard_not_ok",
        "plugin_schedulefree_request_schema_ui_scorecard_not_ok",
        "plugin_adaptivelr_family_batch_scorecard_not_ok",
        "plugin_adaptivelr_owner_release_hold_scorecard_not_ok",
        "plugin_adaptivelr_request_schema_ui_scorecard_not_ok",
        "plugin_simple_formula_family_batch_scorecard_not_ok",
        "plugin_simple_formula_e2e_shadow_matrix_scorecard_not_ok",
        "plugin_simple_formula_canary_rollout_policy_scorecard_not_ok",
        "plugin_simple_formula_dispatch_integration_review_scorecard_not_ok",
        "plugin_simple_formula_owner_release_hold_scorecard_not_ok",
        "plugin_simple_formula_request_schema_ui_scorecard_not_ok",
        "plugin_closure_second_order_family_batch_scorecard_not_ok",
        "plugin_closure_second_order_owner_release_hold_scorecard_not_ok",
        "plugin_closure_second_order_request_schema_ui_scorecard_not_ok",
        "plugin_custom_formula_family_batch_scorecard_not_ok",
        "plugin_custom_formula_owner_release_hold_scorecard_not_ok",
        "plugin_custom_formula_request_schema_ui_scorecard_not_ok",
        "plugin_factored_memory_family_batch_scorecard_not_ok",
        "plugin_factored_memory_owner_release_hold_scorecard_not_ok",
        "plugin_factored_memory_request_schema_ui_scorecard_not_ok",
        "plugin_fused_backward_family_batch_scorecard_not_ok",
        "plugin_fused_backward_owner_release_hold_scorecard_not_ok",
        "plugin_fused_backward_request_schema_ui_scorecard_not_ok",
        "plugin_model_shape_aware_family_batch_scorecard_not_ok",
        "plugin_model_shape_aware_owner_release_hold_scorecard_not_ok",
        "plugin_model_shape_aware_request_schema_ui_scorecard_not_ok",
        "plugin_state_adapter_special_family_batch_scorecard_not_ok",
        "plugin_state_adapter_special_owner_release_hold_scorecard_not_ok",
        "plugin_state_adapter_special_request_schema_ui_scorecard_not_ok",
        "factored_custom_layout_scorecard_not_ok",
        "plugin_factored_memory_layout_scorecard_not_ok",
    )
    return [name for name, report in zip(names, reports) if report.get("ok") is not True]


def _refresh_family_artifacts() -> None:
    adamlike = build_plugin_adamlike_family_batch_scorecard(include_live_canaries=True, write_artifact=True)
    adamlike_hold = build_plugin_adamlike_owner_release_hold_scorecard(
        family_batch_report=adamlike,
        write_artifact=True,
    )
    build_plugin_adamlike_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=adamlike_hold,
        write_artifact=True,
    )
    schedulefree = build_plugin_schedulefree_family_batch_scorecard(write_artifact=True)
    schedulefree_hold = build_plugin_schedulefree_owner_release_hold_scorecard(
        family_batch_report=schedulefree,
        write_artifact=True,
    )
    build_plugin_schedulefree_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=schedulefree_hold,
        write_artifact=True,
    )
    adaptivelr = build_plugin_adaptivelr_family_batch_scorecard(write_artifact=True)
    adaptivelr_hold = build_plugin_adaptivelr_owner_release_hold_scorecard(
        family_batch_report=adaptivelr,
        write_artifact=True,
    )
    build_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=adaptivelr_hold,
        write_artifact=True,
    )
    simple_formula_batch = build_plugin_simple_formula_family_batch_scorecard(
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    simple_formula_e2e = build_plugin_simple_formula_e2e_shadow_matrix_scorecard(
        simple_formula_batch_report=simple_formula_batch,
        write_artifact=True,
    )
    simple_formula_rollout = build_plugin_simple_formula_canary_rollout_policy_scorecard(
        shadow_matrix_report=simple_formula_e2e,
        write_artifact=True,
    )
    simple_formula_dispatch = build_plugin_simple_formula_dispatch_integration_review_scorecard(
        rollout_policy_report=simple_formula_rollout,
        write_artifact=True,
    )
    simple_formula_hold = build_plugin_simple_formula_owner_release_hold_scorecard(
        dispatch_review_report=simple_formula_dispatch,
        write_artifact=True,
    )
    build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=simple_formula_hold,
        write_artifact=True,
    )
    closure_second_order = build_plugin_closure_second_order_family_batch_scorecard(write_artifact=True)
    closure_hold = build_plugin_closure_second_order_owner_release_hold_scorecard(
        family_batch_report=closure_second_order,
        write_artifact=True,
    )
    build_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=closure_hold,
        write_artifact=True,
    )
    custom_formula = build_plugin_custom_formula_family_batch_scorecard(write_artifact=True)
    custom_hold = build_plugin_custom_formula_owner_release_hold_scorecard(
        family_batch_report=custom_formula,
        write_artifact=True,
    )
    build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=custom_hold,
        write_artifact=True,
    )
    factored_memory = build_plugin_factored_memory_family_batch_scorecard(write_artifact=True)
    factored_hold = build_plugin_factored_memory_owner_release_hold_scorecard(
        family_batch_report=factored_memory,
        write_artifact=True,
    )
    build_plugin_factored_memory_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=factored_hold,
        write_artifact=True,
    )
    fused_backward = build_plugin_fused_backward_family_batch_scorecard(write_artifact=True)
    fused_hold = build_plugin_fused_backward_owner_release_hold_scorecard(
        family_batch_report=fused_backward,
        write_artifact=True,
    )
    build_plugin_fused_backward_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=fused_hold,
        write_artifact=True,
    )
    model_shape = build_plugin_model_shape_aware_family_batch_scorecard(write_artifact=True)
    model_shape_hold = build_plugin_model_shape_aware_owner_release_hold_scorecard(
        family_batch_report=model_shape,
        write_artifact=True,
    )
    build_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=model_shape_hold,
        write_artifact=True,
    )
    state_adapter = build_plugin_state_adapter_special_family_batch_scorecard(write_artifact=True)
    state_adapter_hold = build_plugin_state_adapter_special_owner_release_hold_scorecard(
        family_batch_report=state_adapter,
        write_artifact=True,
    )
    build_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=state_adapter_hold,
        write_artifact=True,
    )


def _adamlike_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_adamlike_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _adamlike_owner_release_hold_report(adamlike_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_adamlike_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_adamlike_owner_release_hold_scorecard(
        family_batch_report=adamlike_batch
    )


def _adamlike_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_adamlike_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_adamlike_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _schedulefree_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_schedulefree_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _schedulefree_owner_release_hold_report(schedulefree_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_schedulefree_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_schedulefree_owner_release_hold_scorecard(
        family_batch_report=schedulefree_batch
    )


def _schedulefree_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_schedulefree_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_schedulefree_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _adaptivelr_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_adaptivelr_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _adaptivelr_owner_release_hold_report(adaptivelr_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_adaptivelr_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_adaptivelr_owner_release_hold_scorecard(
        family_batch_report=adaptivelr_batch
    )


def _adaptivelr_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_adaptivelr_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _simple_formula_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_simple_formula_family_batch_scorecard.json"
    if not source.exists():
        return build_plugin_simple_formula_family_batch_scorecard(workspace_root=REPO_ROOT)
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return build_plugin_simple_formula_family_batch_scorecard(workspace_root=REPO_ROOT)


def _simple_formula_e2e_shadow_matrix_report(simple_formula_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_simple_formula_e2e_shadow_matrix_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_simple_formula_e2e_shadow_matrix_scorecard(
        simple_formula_batch_report=simple_formula_batch
    )


def _simple_formula_canary_rollout_policy_report(simple_formula_e2e: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_simple_formula_canary_rollout_policy_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_simple_formula_canary_rollout_policy_scorecard(
        shadow_matrix_report=simple_formula_e2e
    )


def _simple_formula_dispatch_integration_review_report(simple_formula_rollout: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_simple_formula_dispatch_integration_review_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_simple_formula_dispatch_integration_review_scorecard(
        rollout_policy_report=simple_formula_rollout
    )


def _simple_formula_owner_release_hold_report(simple_formula_dispatch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_simple_formula_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_simple_formula_owner_release_hold_scorecard(
        dispatch_review_report=simple_formula_dispatch
    )


def _simple_formula_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_simple_formula_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_simple_formula_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _closure_second_order_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_closure_second_order_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _closure_second_order_owner_release_hold_report(closure_second_order_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_closure_second_order_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_closure_second_order_owner_release_hold_scorecard(
        family_batch_report=closure_second_order_batch
    )


def _closure_second_order_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_closure_second_order_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _custom_formula_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_custom_formula_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _custom_formula_owner_release_hold_report(custom_formula_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_custom_formula_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_custom_formula_owner_release_hold_scorecard(
        family_batch_report=custom_formula_batch
    )


def _custom_formula_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _factored_memory_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_factored_memory_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _factored_memory_owner_release_hold_report(factored_memory_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_factored_memory_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_factored_memory_owner_release_hold_scorecard(
        family_batch_report=factored_memory_batch
    )


def _factored_memory_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_factored_memory_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_factored_memory_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _fused_backward_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_fused_backward_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _fused_backward_owner_release_hold_report(fused_backward_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_fused_backward_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_fused_backward_owner_release_hold_scorecard(
        family_batch_report=fused_backward_batch
    )


def _fused_backward_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_fused_backward_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_fused_backward_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _model_shape_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_model_shape_aware_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _model_shape_owner_release_hold_report(model_shape_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_model_shape_aware_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_model_shape_aware_owner_release_hold_scorecard(
        family_batch_report=model_shape_batch
    )


def _model_shape_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_model_shape_aware_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _state_adapter_special_family_batch_report() -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_state_adapter_special_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        return _as_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _state_adapter_special_owner_release_hold_report(state_adapter_batch: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_state_adapter_special_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_state_adapter_special_owner_release_hold_scorecard(
        family_batch_report=state_adapter_batch
    )


def _state_adapter_special_request_schema_ui_report(owner_hold: Mapping[str, Any]) -> dict[str, Any]:
    source = REPO_ROOT / "temp" / "turbocore_optimizer"
    source = source / "turbocore_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard.json"
    if source.exists():
        try:
            return _as_dict(json.loads(source.read_text(encoding="utf-8")))
        except Exception:
            pass
    return build_plugin_state_adapter_special_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        workspace_root=REPO_ROOT,
    )


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_optimizer_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unsafe_claims(*reports: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for report in reports:
        scorecard = str(report.get("scorecard", "unknown_scorecard"))
        if report.get("training_path_enabled") is True:
            out.append(f"{scorecard}:training_path_enabled")
        if report.get("default_behavior_changed") is True:
            out.append(f"{scorecard}:default_behavior_changed")
        if report.get("runtime_dispatch_ready") is True:
            out.append(f"{scorecard}:runtime_dispatch_ready")
        if report.get("native_dispatch_allowed") is True:
            out.append(f"{scorecard}:native_dispatch_allowed")
    return out


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "missing_classification_count": int(report.get("missing_classification_count", 0) or 0),
        "route_family_counts": dict(_as_dict(summary.get("route_family_counts"))),
    }


def _compact_selected_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "selected_optimizer_abi_ready": report.get("selected_optimizer_abi_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "passed_case_count": int(summary.get("passed_case_count", 0) or 0),
    }


def _compact_adamlike_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_adamlike_family_batch_ready": report.get("selected_adamlike_family_batch_ready") is True,
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "pending_selected_optimizer_count": int(summary.get("pending_selected_optimizer_count", 0) or 0),
        "e2e_shadow_matrix_ready": summary.get("e2e_shadow_matrix_ready") is True,
        "canary_rollout_policy_ready": summary.get("canary_rollout_policy_ready") is True,
        "plugin_selected_native_ready_count": int(summary.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_adamlike_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_adamlike_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_schedulefree_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_schedulefree_family_batch_ready": report.get("selected_schedulefree_family_batch_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "e2e_shadow_case_count": int(summary.get("e2e_shadow_case_count", 0) or 0),
        "dispatch_review_gate_ready": summary.get("dispatch_review_gate_ready") is True,
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_schedulefree_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "selected_native_canary_ready_count": int(summary.get("selected_native_canary_ready_count", 0) or 0),
        "e2e_shadow_case_count": int(summary.get("e2e_shadow_case_count", 0) or 0),
        "dispatch_review_gate_ready": summary.get("dispatch_review_gate_ready") is True,
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_schedulefree_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_adaptivelr_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_adaptivelr_family_batch_ready": report.get("selected_adaptivelr_family_batch_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "selected_state_machine_reference_ready_count": int(
            summary.get("selected_state_machine_reference_ready_count", 0) or 0
        ),
        "selected_state_machine_abi_spec_ready_count": int(
            summary.get("selected_state_machine_abi_spec_ready_count", 0) or 0
        ),
        "selected_state_machine_abi_implementation_ready_count": int(
            summary.get("selected_state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "selected_native_kernel_preconditions_spec_ready_count": int(
            summary.get("selected_native_kernel_preconditions_spec_ready_count", 0) or 0
        ),
        "selected_native_kernel_preconditions_implementation_ready_count": int(
            summary.get("selected_native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "selected_state_machine_replay_matrix_artifact_ready_count": int(
            summary.get("selected_state_machine_replay_matrix_artifact_ready_count", 0) or 0
        ),
        "selected_state_machine_replay_matrix_implementation_ready_count": int(
            summary.get("selected_state_machine_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_state_machine_replay_case_planned_count": int(
            summary.get("selected_state_machine_replay_case_planned_count", 0) or 0
        ),
        "selected_state_machine_replay_case_implementation_ready_count": int(
            summary.get("selected_state_machine_replay_case_implementation_ready_count", 0) or 0
        ),
        "selected_state_machine_replay_resume_case_planned_count": int(
            summary.get("selected_state_machine_replay_resume_case_planned_count", 0) or 0
        ),
        "selected_state_machine_replay_resume_case_implementation_ready_count": int(
            summary.get("selected_state_machine_replay_resume_case_implementation_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_adaptivelr_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "selected_state_machine_reference_ready_count": int(
            summary.get("selected_state_machine_reference_ready_count", 0) or 0
        ),
        "selected_state_machine_abi_implementation_ready_count": int(
            summary.get("selected_state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_adaptivelr_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_simple_formula_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_simple_formula_family_batch_ready": report.get("selected_simple_formula_family_batch_ready") is True,
        "selected_simple_formula_optimizer_count": int(
            summary.get("selected_simple_formula_optimizer_count", 0) or 0
        ),
        "reference_canary_ready_count": int(summary.get("reference_canary_ready_count", 0) or 0),
        "selected_plugin_native_canary_ready_count": int(
            summary.get("selected_plugin_native_canary_ready_count", 0) or 0
        ),
        "selected_plugin_native_ready_count": int(summary.get("selected_plugin_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_simple_formula_e2e_shadow_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "e2e_shadow_matrix_ready": report.get("e2e_shadow_matrix_ready") is True,
        "report_only_matrix_scaffold_ready": report.get("report_only_matrix_scaffold_ready") is True,
        "live_shadow_matrix_executed": report.get("live_shadow_matrix_executed") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "ready_case_count": int(summary.get("ready_case_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_simple_formula_rollout_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "canary_auto_enabled": report.get("canary_auto_enabled") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "canary_rollout_policy_ready_count": int(
            summary.get("canary_rollout_policy_ready_count", 0) or 0
        ),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_simple_formula_dispatch_review(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "review_gate_ready": report.get("review_gate_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
    }


def _compact_simple_formula_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "dispatch_review_gate_ready": report.get("dispatch_review_gate_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_simple_formula_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "dispatch_review_gate_ready": report.get("dispatch_review_gate_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
    }


def _compact_closure_second_order_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_closure_second_order_family_batch_ready": report.get(
            "selected_closure_second_order_family_batch_ready"
        )
        is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "higher_order_training_loop_abi_required_count": int(
            summary.get("higher_order_training_loop_abi_required_count", 0) or 0
        ),
        "training_loop_abi_spec_ready_count": int(summary.get("training_loop_abi_spec_ready_count", 0) or 0),
        "training_loop_abi_implementation_ready_count": int(
            summary.get("training_loop_abi_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_spec_ready_count": int(
            summary.get("resume_parity_matrix_spec_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "closure_replay_case_planned_count": int(summary.get("closure_replay_case_planned_count", 0) or 0),
        "create_graph_hvp_lifetime_case_planned_count": int(
            summary.get("create_graph_hvp_lifetime_case_planned_count", 0) or 0
        ),
        "closure_resume_replay_artifact_ready_count": int(
            summary.get("closure_resume_replay_artifact_ready_count", 0) or 0
        ),
        "closure_resume_replay_artifact_row_count": int(
            summary.get("closure_resume_replay_artifact_row_count", 0) or 0
        ),
        "closure_resume_replay_artifact_implementation_ready_count": int(
            summary.get("closure_resume_replay_artifact_implementation_ready_count", 0) or 0
        ),
        "native_kernel_precondition_plan_ready_count": int(
            summary.get("native_kernel_precondition_plan_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_closure_second_order_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_closure_second_order_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_custom_formula_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_custom_formula_family_batch_ready": report.get("selected_custom_formula_family_batch_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "formula_parity_required_count": int(summary.get("formula_parity_required_count", 0) or 0),
        "backlog_ready_count": int(summary.get("backlog_ready_count", 0) or 0),
        "evidence_artifact_planned_count": int(summary.get("evidence_artifact_planned_count", 0) or 0),
        "evidence_status_pending_total": int(summary.get("evidence_status_pending_total", 0) or 0),
        "formula_spec_artifact_ready_count": int(summary.get("formula_spec_artifact_ready_count", 0) or 0),
        "formula_spec_artifact_pending_count": int(summary.get("formula_spec_artifact_pending_count", 0) or 0),
        "formula_state_inventory_skeleton_count": int(
            summary.get("formula_state_inventory_skeleton_count", 0) or 0
        ),
        "execution_matrix_ready": summary.get("execution_matrix_ready") is True,
        "formula_step_execution_ready_count": int(summary.get("formula_step_execution_ready_count", 0) or 0),
        "resume_next_step_replay_ready_count": int(summary.get("resume_next_step_replay_ready_count", 0) or 0),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_custom_formula_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_custom_formula_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_factored_memory_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_factored_memory_family_batch_ready": report.get(
            "selected_factored_memory_family_batch_ready"
        )
        is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "observed_resume_layout_count": int(summary.get("observed_resume_layout_count", 0) or 0),
        "native_layout_abi_ready_count": int(summary.get("native_layout_abi_ready_count", 0) or 0),
        "quality_matrix_ready_count": int(summary.get("quality_matrix_ready_count", 0) or 0),
        "native_kernel_entry_condition_ready_count": int(
            summary.get("native_kernel_entry_condition_ready_count", 0) or 0
        ),
        "formula_tensor_binding_matrix_artifact_ready_count": int(
            summary.get("formula_tensor_binding_matrix_artifact_ready_count", 0) or 0
        ),
        "formula_tensor_binding_matrix_implementation_ready_count": int(
            summary.get("formula_tensor_binding_matrix_implementation_ready_count", 0) or 0
        ),
        "formula_parity_case_planned_count": int(summary.get("formula_parity_case_planned_count", 0) or 0),
        "tensor_binding_case_planned_count": int(summary.get("tensor_binding_case_planned_count", 0) or 0),
        "dispatch_review_gate_ready": summary.get("dispatch_review_gate_ready") is True,
        "dispatch_review_ready_count": int(summary.get("dispatch_review_ready_count", 0) or 0),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_factored_memory_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_factored_memory_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_fused_backward_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_fused_backward_family_batch_ready": report.get("selected_fused_backward_family_batch_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "gradient_ownership_abi_required_count": int(
            summary.get("gradient_ownership_abi_required_count", 0) or 0
        ),
        "per_optimizer_abi_spec_ready_count": int(summary.get("per_optimizer_abi_spec_ready_count", 0) or 0),
        "fused_backward_abi_implementation_ready_count": int(
            summary.get("fused_backward_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_spec_ready_count": int(
            summary.get("native_kernel_preconditions_spec_ready_count", 0) or 0
        ),
        "resume_parity_matrix_spec_ready_count": int(
            summary.get("resume_parity_matrix_spec_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "fused_backward_replay_case_planned_count": int(
            summary.get("fused_backward_replay_case_planned_count", 0) or 0
        ),
        "fused_backward_replay_case_implementation_ready_count": int(
            summary.get("fused_backward_replay_case_implementation_ready_count", 0) or 0
        ),
        "loss_scale_boundary_case_planned_count": int(
            summary.get("loss_scale_boundary_case_planned_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_fused_backward_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_fused_backward_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_model_shape_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_model_shape_aware_family_batch_ready": report.get(
            "selected_model_shape_aware_family_batch_ready"
        )
        is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "param_group_semantics_dependent_count": int(
            summary.get("param_group_semantics_dependent_count", 0) or 0
        ),
        "param_group_abi_spec_ready_count": int(summary.get("param_group_abi_spec_ready_count", 0) or 0),
        "param_group_abi_implementation_ready_count": int(
            summary.get("param_group_abi_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_artifact_ready_count": int(
            summary.get("param_group_resume_replay_matrix_artifact_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_row_count": int(
            summary.get("param_group_resume_replay_matrix_row_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_implementation_ready_count": int(
            summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("selected_plugin_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_model_shape_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_model_shape_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_state_adapter_special_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_state_adapter_special_family_batch_ready": report.get(
            "selected_state_adapter_special_family_batch_ready"
        )
        is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "param_ownership_abi_required_count": int(summary.get("param_ownership_abi_required_count", 0) or 0),
        "adapter_abi_spec_ready_count": int(summary.get("adapter_abi_spec_ready_count", 0) or 0),
        "adapter_abi_implementation_ready_count": int(
            summary.get("adapter_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_precondition_spec_ready_count": int(
            summary.get("native_kernel_precondition_spec_ready_count", 0) or 0
        ),
        "native_kernel_precondition_implementation_ready_count": int(
            summary.get("native_kernel_precondition_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_matrix_artifact_ready_count": int(
            summary.get("adapter_resume_matrix_artifact_ready_count", 0) or 0
        ),
        "adapter_resume_matrix_implementation_ready_count": int(
            summary.get("adapter_resume_matrix_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_replay_case_planned_count": int(
            summary.get("adapter_resume_replay_case_planned_count", 0) or 0
        ),
        "adapter_resume_replay_case_implementation_ready_count": int(
            summary.get("adapter_resume_replay_case_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_translation_case_planned_count": int(
            summary.get("adapter_resume_translation_case_planned_count", 0) or 0
        ),
        "adapter_resume_translation_case_implementation_ready_count": int(
            summary.get("adapter_resume_translation_case_implementation_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "plugin_selected_native_ready_count": int(report.get("plugin_selected_native_ready_count", 0) or 0),
    }


def _compact_state_adapter_special_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_state_adapter_special_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "request_fields_emitted": report.get("request_fields_emitted") is True,
        "schema_exposure_allowed": report.get("schema_exposure_allowed") is True,
        "ui_exposure_allowed": report.get("ui_exposure_allowed") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_factored_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "state_layout_reference_ready": report.get("state_layout_reference_ready") is True,
        "adamw_kernel_reuse_blocked": report.get("adamw_kernel_reuse_blocked") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
    }


def _compact_plugin_factored_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "state_layout_audit_complete": report.get("state_layout_audit_complete") is True,
        "state_layout_reference_ready": report.get("state_layout_reference_ready") is True,
        "selected_optimizer_abi_ready": report.get("selected_optimizer_abi_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "observed_resume_layout_count": int(summary.get("observed_resume_layout_count", 0) or 0),
        "manual_contract_pending_count": int(summary.get("manual_contract_pending_count", 0) or 0),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
    }


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_plugin_optimizer_family_batch_scorecard"]
