"""Gate policy for TurboCore native update experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from core.turbocore_native_update_dispatch_contract import build_native_update_dispatch_contract
from core.turbocore_native_update_dispatch_request import build_native_update_dispatch_request
from core.turbocore_native_update_kernel_launcher import build_native_update_kernel_launch_plan
from core.turbocore_native_update_preflight import build_native_update_dispatch_preflight
from core.turbocore_native_update_fallback import build_native_update_fallback_policy


@dataclass(frozen=True)
class TurboCoreNativeUpdateGateConfig:
    mode: str = "off"
    required_shadow_passes: int = 3
    max_abs_diff: float = 5e-5
    max_mean_abs_diff: float = 1e-6
    allow_missing_native_kernel: bool = False
    strict: bool = False
    dispatch_enabled: bool = False

    @property
    def requested(self) -> bool:
        return self.mode in {"profile", "native_experimental"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurboCoreNativeUpdateGateState:
    consecutive_shadow_passes: int = 0
    last_shadow_ok: bool = False
    last_blocked_reasons: list[str] = field(default_factory=list)


class TurboCoreNativeUpdateGate:
    """Evaluate whether a future native update path would be allowed."""

    def __init__(self, config: TurboCoreNativeUpdateGateConfig) -> None:
        self.config = config
        self.state = TurboCoreNativeUpdateGateState()

    @property
    def requested(self) -> bool:
        return self.config.requested

    def update(
        self,
        *,
        shadow_report: Mapping[str, Any] | None,
        optimizer: Any,
        trainable_param_count: int,
        runtime_context: Mapping[str, Any] | None = None,
        readiness_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = dict(runtime_context or {})
        reasons = self._static_blocked_reasons(optimizer, trainable_param_count, context)
        readiness = self._readiness_status(readiness_report)
        reasons.extend(readiness["blocked_reasons"])
        shadow = self._shadow_status(shadow_report)
        if not shadow["ok"]:
            reasons.extend(shadow["blocked_reasons"])
            self.state.consecutive_shadow_passes = 0
            self.state.last_shadow_ok = False
        else:
            self.state.consecutive_shadow_passes += 1
            self.state.last_shadow_ok = True
        if self.state.consecutive_shadow_passes < int(self.config.required_shadow_passes):
            reasons.append("shadow_warmup_not_satisfied")
        if not bool(self.config.allow_missing_native_kernel):
            reasons.append("native_kernel_promotion_not_enabled")
        unique_reasons = _dedupe(reasons)
        would_enable = bool(self.requested and not unique_reasons)
        fallback_policy = build_native_update_fallback_policy(
            mode=self.config.mode,
            strict=bool(self.config.strict),
            readiness_report=readiness_report,
            shadow_report=shadow_report,
            runtime_context=context,
        )
        preflight_shadow_report = dict(shadow_report or {})
        preflight_shadow_report["fallback_policy"] = fallback_policy
        dispatch_preflight = build_native_update_dispatch_preflight(
            mode=self.config.mode,
            requested=self.requested,
            readiness_report=readiness_report,
            shadow_report=preflight_shadow_report,
            gate_blocked_reasons=unique_reasons,
            consecutive_shadow_passes=self.state.consecutive_shadow_passes,
            required_shadow_passes=self.config.required_shadow_passes,
            allow_missing_native_kernel=self.config.allow_missing_native_kernel,
            runtime_context=context,
        )
        dispatch_contract = build_native_update_dispatch_contract(
            mode=self.config.mode,
            requested=self.requested,
            readiness_report=readiness_report,
            shadow_report=preflight_shadow_report,
            dispatch_preflight=dispatch_preflight,
            fallback_policy=fallback_policy,
            gate_blocked_reasons=unique_reasons,
            runtime_context=context,
        )
        report = {
            "schema_version": 1,
            "gate": "turbocore_native_update_gate_v0",
            "mode": self.config.mode,
            "requested": bool(self.requested),
            "would_enable_native_update": would_enable,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "required_shadow_passes": int(self.config.required_shadow_passes),
            "consecutive_shadow_passes": int(self.state.consecutive_shadow_passes),
            "readiness": readiness,
            "shadow": shadow,
            "fallback_policy": fallback_policy,
            "dispatch_preflight": dispatch_preflight,
            "dispatch_contract": dispatch_contract,
            "blocked_reasons": unique_reasons,
            "config": self.config.as_dict(),
        }
        report["dispatch_request"] = build_native_update_dispatch_request(
            mode=self.config.mode,
            dispatch_enabled=self.config.dispatch_enabled,
            gate_report=report,
            dispatch_contract=dispatch_contract,
            runtime_context=context,
        )
        report["kernel_launch_plan"] = build_native_update_kernel_launch_plan(
            dispatch_request=report["dispatch_request"],
            dispatch_contract=dispatch_contract,
            owner_native_launch_probe=preflight_shadow_report.get("owner_native_launch_probe"),
        )
        self.state.last_blocked_reasons = unique_reasons
        return report

    def _static_blocked_reasons(self, optimizer: Any, trainable_param_count: int, context: Mapping[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not self.requested:
            reasons.append("gate_not_requested")
        name = type(optimizer).__name__.lower() if optimizer is not None else ""
        if "adamw" not in name:
            reasons.append("optimizer_not_adamw")
        if int(trainable_param_count or 0) <= 0:
            reasons.append("no_trainable_params")
        if (
            bool(context.get("multi_gpu", False))
            or int(context.get("num_processes", 1) or 1) > 1
            or int(context.get("num_machines", 1) or 1) > 1
        ):
            reasons.append("distributed_not_supported")
        if bool(context.get("deepspeed", False)):
            reasons.append("deepspeed_not_supported")
        if bool(context.get("gradient_release_active", False)):
            reasons.append("gradient_release_not_supported")
        return reasons

    def _shadow_status(self, shadow_report: Mapping[str, Any] | None) -> dict[str, Any]:
        report = dict(shadow_report or {})
        after = report.get("after_optimizer") if isinstance(report.get("after_optimizer"), Mapping) else {}
        blocked: list[str] = []
        if not report:
            blocked.append("shadow_report_missing")
        if report.get("error"):
            blocked.append("shadow_prepare_error")
        if after.get("error"):
            blocked.append("shadow_compare_error")
        if after and not bool(after.get("compared", False)):
            blocked.append("shadow_not_compared")
        parity_ok = bool(after.get("parity_ok_loose", False)) if after else False
        if not parity_ok:
            blocked.append("shadow_parity_not_ok")
        max_abs = _float(after.get("max_abs_param_diff"), default=float("inf")) if after else float("inf")
        mean_abs = _float(after.get("mean_abs_param_diff"), default=float("inf")) if after else float("inf")
        if max_abs > float(self.config.max_abs_diff):
            blocked.append("shadow_max_abs_diff_too_high")
        if mean_abs > float(self.config.max_mean_abs_diff):
            blocked.append("shadow_mean_abs_diff_too_high")
        copyback = report.get("copyback_probe") if isinstance(report.get("copyback_probe"), Mapping) else {}
        copyback_requested = bool(copyback)
        copyback_validated = bool(copyback.get("scratch_copyback_validated", False)) if copyback_requested else None
        copyback_mutated_real_params = bool(copyback.get("real_parameters_mutated", False)) if copyback_requested else None
        dispatch = report.get("copyback_dispatch_probe") if isinstance(report.get("copyback_dispatch_probe"), Mapping) else {}
        dispatch_present = bool(dispatch)
        dispatch_enabled = bool(dispatch.get("copyback_dispatch_enabled", False)) if dispatch_present else None
        dispatch_validated = bool(dispatch.get("copyback_dispatch_validated", False)) if dispatch_present else None
        dispatch_mutated_real_params = bool(dispatch.get("real_parameters_mutated", False)) if dispatch_present else None
        if copyback_requested and not copyback_validated:
            blocked.append("copyback_scratch_validation_failed")
        if copyback_mutated_real_params:
            blocked.append("copyback_probe_mutated_training_parameters")
        if dispatch_present and not dispatch_validated:
            blocked.append("copyback_dispatch_validation_failed")
        if dispatch_mutated_real_params and not bool(dispatch.get("real_parameters_restored", False)):
            blocked.append("copyback_dispatch_left_training_parameters_mutated")
        native_binding = report.get("native_binding_probe") if isinstance(report.get("native_binding_probe"), Mapping) else {}
        native_binding_present = bool(native_binding)
        stream_contract = native_binding.get("stream_contract") if isinstance(native_binding.get("stream_contract"), Mapping) else {}
        owner_native = report.get("owner_native_launch_probe") if isinstance(report.get("owner_native_launch_probe"), Mapping) else {}
        owner_native_present = bool(owner_native)
        owner_native_ok = bool(owner_native.get("ok", False)) if owner_native_present else None
        owner_native_attempted = bool(owner_native.get("attempted", False)) if owner_native_present else None
        if owner_native_present and owner_native_attempted and not owner_native_ok:
            blocked.append("owner_native_launch_probe_failed")
        return {
            "ok": not blocked,
            "parity_ok_loose": parity_ok,
            "max_abs_param_diff": max_abs if max_abs != float("inf") else None,
            "mean_abs_param_diff": mean_abs if mean_abs != float("inf") else None,
            "copyback_probe_present": copyback_requested,
            "copyback_scratch_validated": copyback_validated,
            "copyback_real_parameters_mutated": copyback_mutated_real_params,
            "copyback_elapsed_ms": _float(copyback.get("elapsed_ms"), default=0.0) if copyback_requested else None,
            "copyback_dispatch_probe_present": dispatch_present,
            "copyback_dispatch_enabled": dispatch_enabled,
            "copyback_dispatch_validated": dispatch_validated,
            "copyback_dispatch_target": str(dispatch.get("copyback_dispatch_target", "") or "") if dispatch_present else None,
            "copyback_dispatch_real_parameters_mutated": dispatch_mutated_real_params,
            "copyback_dispatch_real_parameters_restored": bool(dispatch.get("real_parameters_restored", False)) if dispatch_present else None,
            "copyback_dispatch_elapsed_ms": _float(dispatch.get("elapsed_ms"), default=0.0) if dispatch_present else None,
            "native_binding_probe_present": native_binding_present,
            "native_binding_request_shape_ready": bool(native_binding.get("request_shape_ready", False)) if native_binding_present else None,
            "native_binding_tensor_object_ready": bool(native_binding.get("tensor_object_binding_ready", False)) if native_binding_present else None,
            "native_binding_launch_plan_ready": bool(native_binding.get("launch_plan_ready", False)) if native_binding_present else None,
            "native_binding_stream_lifetime_bound": bool(native_binding.get("stream_lifetime_bound", False)) if native_binding_present else None,
            "native_binding_stream_contract_present": bool(stream_contract) if native_binding_present else None,
            "native_binding_stream_kind": str(stream_contract.get("stream_kind", "") or "") if stream_contract else None,
            "native_binding_stream_lease_id": int(native_binding.get("stream_lease_id", 0) or 0) if native_binding_present else None,
            "native_binding_stream_guard_present": bool(native_binding.get("stream_guard_present", False)) if native_binding_present else None,
            "native_binding_stream_guard_ready": bool(native_binding.get("stream_guard_ready", False)) if native_binding_present else None,
            "native_binding_stream_identity_ready": bool(native_binding.get("stream_identity_ready", False)) if native_binding_present else None,
            "native_binding_stream_guard_level": str(native_binding.get("stream_guard_level", "") or "") if native_binding_present else None,
            "native_binding_stream_handle_kind": str(native_binding.get("stream_handle_kind", "") or "") if native_binding_present else None,
            "native_binding_stream_handle_reported": bool(native_binding.get("stream_handle_reported", False)) if native_binding_present else None,
            "native_binding_stream_handle_nonzero": bool(native_binding.get("stream_handle_nonzero", False)) if native_binding_present else None,
            "native_binding_synchronization_guard_ready": bool(native_binding.get("synchronization_guard_ready", False)) if native_binding_present else None,
            "native_binding_synchronization_strategy": str(native_binding.get("synchronization_strategy", "") or "") if native_binding_present else None,
            "native_binding_event_chain_contract": str(native_binding.get("event_chain_contract", "") or "") if native_binding_present else None,
            "native_binding_event_chain_state": str(native_binding.get("event_chain_state", "") or "") if native_binding_present else None,
            "native_binding_event_chain_probe_requested": bool(native_binding.get("event_chain_probe_requested", False)) if native_binding_present else None,
            "native_binding_event_chain_probe_attempted": bool(native_binding.get("event_chain_probe_attempted", False)) if native_binding_present else None,
            "native_binding_event_chain_verified": bool(native_binding.get("event_chain_verified", False)) if native_binding_present else None,
            "native_binding_pre_launch_ordering_verified": bool(native_binding.get("pre_launch_ordering_verified", False)) if native_binding_present else None,
            "native_binding_post_launch_ordering_verified": bool(native_binding.get("post_launch_ordering_verified", False)) if native_binding_present else None,
            "native_binding_stream_wait_event_verified": bool(native_binding.get("stream_wait_event_verified", False)) if native_binding_present else None,
            "native_binding_native_launch_candidate": bool(native_binding.get("native_launch_candidate", False)) if native_binding_present else None,
            "native_binding_borrowed_external_stream": bool(native_binding.get("borrowed_external_stream", False)) if native_binding_present else None,
            "native_binding_stream_device_match": bool(native_binding.get("stream_device_match", False)) if native_binding_present else None,
            "native_binding_elapsed_ms": _float(native_binding.get("elapsed_ms"), default=0.0) if native_binding_present else None,
            "owner_native_launch_probe_present": owner_native_present,
            "owner_native_launch_attempted": owner_native_attempted,
            "owner_native_launch_ok": owner_native_ok,
            "owner_native_launch_kernel_executed": bool(owner_native.get("kernel_executed", False)) if owner_native_present else None,
            "owner_native_launch_parity_ok": bool(owner_native.get("parity_ok", False)) if owner_native_present else None,
            "owner_native_launch_persistent_owner_mutated": bool(owner_native.get("persistent_owner_mutated", False)) if owner_native_present else None,
            "owner_native_launch_event_chain_requested": bool(owner_native.get("event_chain_probe_requested", False)) if owner_native_present else None,
            "owner_native_launch_event_chain_attempted": bool(owner_native.get("event_chain_probe_attempted", False)) if owner_native_present else None,
            "owner_native_launch_event_chain_verified": bool(owner_native.get("event_chain_verified", False)) if owner_native_present else None,
            "owner_native_launch_pre_launch_ordering_verified": bool(owner_native.get("pre_launch_ordering_verified", False)) if owner_native_present else None,
            "owner_native_launch_post_launch_ordering_verified": bool(owner_native.get("post_launch_ordering_verified", False)) if owner_native_present else None,
            "owner_native_launch_max_abs_diff": _float(owner_native.get("max_abs_diff"), default=0.0) if owner_native_present else None,
            "owner_native_launch_max_rel_diff": _float(owner_native.get("max_rel_diff"), default=0.0) if owner_native_present else None,
            "owner_native_launch_elapsed_ms": _float(owner_native.get("elapsed_ms"), default=0.0) if owner_native_present else None,
            "blocked_reasons": _dedupe(blocked),
        }

    def _readiness_status(self, readiness_report: Mapping[str, Any] | None) -> dict[str, Any]:
        report = dict(readiness_report or {})
        blocked = [str(item) for item in report.get("blocked_reasons", [])] if isinstance(report.get("blocked_reasons"), list) else []
        if self.requested and not report:
            blocked.append("readiness_report_missing")
        ok = bool(report.get("ok", False)) if report else False
        if report and not ok and not blocked:
            blocked.append("readiness_not_ok")
        native_checks = report.get("native_checks") if isinstance(report.get("native_checks"), Mapping) else {}
        owner_checks = report.get("owner_checks") if isinstance(report.get("owner_checks"), Mapping) else {}
        static_checks = report.get("static_checks") if isinstance(report.get("static_checks"), Mapping) else {}
        return {
            "ok": bool(ok and not blocked),
            "present": bool(report),
            "native_kernel_present": bool(report.get("native_kernel_present", False)),
            "performance_test_ready": bool(report.get("performance_test_ready", False)),
            "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
            "stream_lifetime_ownership_bound": bool(
                report.get("stream_lifetime_ownership_bound", report.get("stream_lifetime_bound", False))
            ),
            "stream_ordering_verified": bool(report.get("stream_ordering_verified", False)),
            "event_chain_verified": bool(report.get("event_chain_verified", False)),
            "static_checks": dict(static_checks),
            "owner_checks": dict(owner_checks),
            "native_checks": dict(native_checks),
            "blocked_reasons": _dedupe(blocked),
        }


def build_native_update_gate_config(
    mode: str = "off",
    *,
    required_shadow_passes: int = 3,
    max_abs_diff: float = 5e-5,
    max_mean_abs_diff: float = 1e-6,
    allow_missing_native_kernel: bool = False,
    strict: bool = False,
    dispatch_enabled: bool = False,
) -> TurboCoreNativeUpdateGateConfig:
    normalized = str(mode or "off").strip().lower().replace("-", "_")
    if normalized not in {"off", "profile", "native_experimental"}:
        normalized = "off"
    return TurboCoreNativeUpdateGateConfig(
        mode=normalized,
        required_shadow_passes=max(int(required_shadow_passes or 3), 1),
        max_abs_diff=max(float(max_abs_diff or 0.0), 0.0),
        max_mean_abs_diff=max(float(max_mean_abs_diff or 0.0), 0.0),
        allow_missing_native_kernel=bool(allow_missing_native_kernel),
        strict=bool(strict),
        dispatch_enabled=bool(dispatch_enabled),
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["TurboCoreNativeUpdateGate", "TurboCoreNativeUpdateGateConfig", "build_native_update_gate_config"]
