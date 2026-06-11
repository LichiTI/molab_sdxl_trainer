"""Report-only native ABI sketch for selected schedule-free plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_plugin_schedulefree_selected_optimizer_scorecard import (
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_schedulefree_selected_optimizer_scorecard,
)


def build_plugin_schedulefree_native_abi_sketch_scorecard() -> dict[str, Any]:
    """Build a machine-readable ABI sketch without enabling native dispatch."""

    selected = build_plugin_schedulefree_selected_optimizer_scorecard()
    cases = [case for case in selected.get("cases", []) if isinstance(case, Mapping)]
    case_failures = [str(case.get("optimizer_name", "")) for case in cases if not bool(case.get("ok", False))]
    case_map = {str(case.get("optimizer_name", "")): dict(case) for case in cases}
    launch_plan = _launch_plan(case_map)
    checkpoint_adapter = _checkpoint_adapter(case_map)
    fallback = _fallback_authority()
    dispatch_policy = _dispatch_policy()
    ready = (
        bool(selected.get("selected_optimizer_abi_ready", False))
        and not case_failures
        and bool(launch_plan.get("launch_plan_contract_ready", False))
        and bool(checkpoint_adapter.get("checkpoint_adapter_contract_ready", False))
        and bool(fallback.get("fallback_authority_ready", False))
        and not bool(dispatch_policy.get("native_dispatch_allowed", True))
    )
    blockers = []
    if case_failures:
        blockers.extend(f"selected_schedulefree_case_failed:{name}" for name in case_failures)

    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_native_abi_sketch_scorecard_v0",
        "gate": "plugin_schedulefree_native_abi_sketch",
        "ok": ready,
        "promotion_ready": False,
        "native_abi_sketch_ready": ready,
        "launch_plan_contract_ready": bool(launch_plan.get("launch_plan_contract_ready", False)),
        "checkpoint_adapter_contract_ready": bool(checkpoint_adapter.get("checkpoint_adapter_contract_ready", False)),
        "fallback_authority_ready": bool(fallback.get("fallback_authority_ready", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "selected_optimizer_family": "schedule_free_state_machine",
        "selected_optimizer_abi_source": selected.get("scorecard", ""),
        "launch_plan": launch_plan,
        "checkpoint_adapter_contract": checkpoint_adapter,
        "fallback_authority": fallback,
        "dispatch_policy": dispatch_policy,
        "summary": {
            "selected_optimizer_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "case_count": len(cases),
            "case_failure_count": len(case_failures),
            "checkpoint_adapter_required_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "blocked_mode_count": len(dispatch_policy["blocked_modes_until_review"]),
        },
        "promotion_blockers": blockers
        + [
            "selected_schedulefree_native_kernel_missing",
            "selected_schedulefree_checkpoint_adapter_runtime_missing",
            "selected_schedulefree_tensor_binding_missing",
            "owner_release_hold_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build report-only checkpoint adapter proof for selected schedule-free plugin optimizers"
            if ready
            else "fix schedule-free selected optimizer ABI sketch blockers"
        ),
        "notes": [
            "This sketch records the native ABI boundary only; it does not call native code.",
            "The active update authority remains the selected pytorch_optimizer plugin route.",
            "Canary and auto modes stay blocked until a real native kernel, checkpoint adapter, and tensor binding exist.",
        ],
    }


def _launch_plan(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    state_models = []
    for name in TARGET_PLUGIN_OPTIMIZERS:
        case = dict(cases.get(name, {}))
        after_step = dict(case.get("after_step", {}) or {})
        state_models.append(
            {
                "optimizer_name": name,
                "optimizer_class": str(case.get("optimizer_class", "")),
                "param_state_keys": list(after_step.get("param_state_keys", []) or []),
                "param_group_keys": list(after_step.get("param_group_keys", []) or []),
                "requires_train_eval": True,
                "requires_constant_scheduler": True,
                "requires_group_train_mode": "train_mode" in set(after_step.get("param_group_keys", []) or []),
            }
        )
    required_group_fields = sorted(
        {
            str(key)
            for item in state_models
            for key in item.get("param_group_keys", [])
            if str(key) not in {"params"}
        }
    )
    required_param_fields = sorted({str(key) for item in state_models for key in item.get("param_state_keys", [])})
    return {
        "schema_version": 1,
        "plan_kind": "plugin_schedulefree_selected_optimizer_launch_plan_v0",
        "launch_plan_contract_ready": bool(state_models),
        "launch_allowed": False,
        "launch_attempted": False,
        "training_dispatch": False,
        "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
        "entrypoint": "future_turbocore_schedulefree_update",
        "required_request_fields": [
            "optimizer_type",
            "optimizer_args.name",
            "learning_rate",
            "weight_decay",
            "scheduler_policy",
            "param_groups",
            "state_handles",
            "mode",
        ],
        "required_group_state_fields": required_group_fields,
        "required_param_state_fields": required_param_fields,
        "state_models": state_models,
        "mode_contract": {
            "train": "native update may run only while optimizer group train_mode is true",
            "eval": "native update must refuse and fall back to plugin optimizer",
            "external_scheduler": "constant_required",
        },
        "blocked_reasons": [
            "native_kernel_missing",
            "checkpoint_adapter_runtime_missing",
            "training_tensor_binding_missing",
            "owner_release_hold_missing",
        ],
    }


def _checkpoint_adapter(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    adapters = []
    for name in TARGET_PLUGIN_OPTIMIZERS:
        case = dict(cases.get(name, {}))
        after_step = dict(case.get("after_step", {}) or {})
        adapters.append(
            {
                "optimizer_name": name,
                "pack_source": "selected_plugin_optimizer.state_dict",
                "unpack_target": "selected_plugin_optimizer.load_state_dict",
                "required_param_state_keys": list(after_step.get("param_state_keys", []) or []),
                "required_group_state_keys": list(after_step.get("param_group_keys", []) or []),
                "requires_train_mode_restore": True,
                "resume_parity_proven_by": "plugin_schedulefree_selected_optimizer_abi",
                "runtime_adapter_enabled": False,
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_adapter_kind": "plugin_schedulefree_state_dict_adapter_v0",
        "checkpoint_adapter_contract_ready": bool(adapters),
        "training_checkpoint_integration_enabled": False,
        "runtime_adapter_enabled": False,
        "adapters": adapters,
        "required_artifact_fields": [
            "selected_optimizer_name",
            "optimizer_class",
            "param_groups",
            "state",
            "train_mode",
        ],
        "blocked_reasons": [
            "runtime_pack_unpack_probe_missing",
            "training_checkpoint_integration_missing",
            "owner_release_hold_missing",
        ],
    }


def _fallback_authority() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fallback_authority_ready": True,
        "training_update_authority": "selected_pytorch_optimizer_plugin",
        "native_update_authority": "none_until_review",
        "fallback_backend": "selected_pytorch_optimizer_plugin",
        "rollback_triggers": [
            "non_finite_update",
            "missing_train_mode",
            "checkpoint_adapter_failure",
            "resume_mismatch",
            "tensor_binding_failure",
            "unsupported_selected_optimizer",
        ],
    }


def _dispatch_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "observe_mode_behavior": "record selected optimizer ABI decision and fallback reason only",
        "canary_mode_behavior": "blocked_before_native_dispatch",
        "auto_mode_behavior": "blocked_before_native_dispatch",
    }


__all__ = ["build_plugin_schedulefree_native_abi_sketch_scorecard"]
