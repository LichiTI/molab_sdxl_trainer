"""Shadow/profile wrapper for TurboCore update-path experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from core.turbocore_copyback_dispatch_probe import build_copyback_dispatch_probe
from core.turbocore_owner_native_launch_probe import (
    DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
    TurboCoreOwnerNativeLaunchProbe,
    build_owner_native_launch_probe_skip,
)
from core.turbocore_update_executor import TurboCoreUpdateExecutor, TurboCoreUpdateExecutorConfig
from core.turbocore_update_checkpoint_contract import (
    build_flat_adamw_checkpoint_contract,
    sync_flat_owner_state_from_optimizer,
)
from core.turbocore_update_native_binding_probe import build_update_native_binding_probe


@dataclass(frozen=True)
class TurboCoreUpdateShadowConfig:
    mode: str = "off"
    max_params: int = 0
    compare_interval: int = 1
    direct_grad: bool = False
    prefer_triton: bool = False
    compare_sample_params: int = 0
    stop_after_consecutive_passes: int = 0
    checkpoint_contract: bool = False
    copyback_probe: bool = False
    copyback_dispatch_experimental: bool = False
    native_binding_probe: bool = False
    owner_native_launch_probe: bool = False
    owner_native_launch_max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL
    owner_native_event_chain_probe: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode in {"profile", "shadow"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TurboCoreUpdateShadow:
    """Profile a TurboCore update beside the real PyTorch optimizer step."""

    def __init__(self, config: TurboCoreUpdateShadowConfig) -> None:
        self.config = config
        self.executor: TurboCoreUpdateExecutor | None = None
        self.params: list[torch.nn.Parameter] = []
        self._before_flat: torch.Tensor | None = None
        self._consecutive_passes = 0
        self._auto_stopped = False
        self._last_direct_grad_lifecycle_report: dict[str, Any] = {}
        self._pending_owner_state: dict[str, Any] | None = None
        self._owner_native_launch_probe: TurboCoreOwnerNativeLaunchProbe | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def prepare_before_optimizer(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        step: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        started = time.perf_counter()
        if self._auto_stopped:
            return self._skip("auto_stopped_after_consecutive_passes", started)
        selected = self._select_params(params)
        if not selected:
            return self._skip("no_trainable_params", started)
        if self.executor is None or self.params != selected:
            self.close()
            self.params = selected
            self.executor = TurboCoreUpdateExecutor(
                selected,
                self._executor_config(optimizer, max_grad_norm=max_grad_norm),
            )
            if self.executor.direct_grad_binding is not None:
                self.executor.direct_grad_binding.zero_owner_grad()
                self.executor.direct_grad_binding.set_active(True)
        assert self.executor is not None
        self.executor.owner.param_flat.copy_(_flatten_params(selected))
        self._before_flat = self.executor.owner.param_flat.detach().clone()
        state_sync = sync_flat_owner_state_from_optimizer(self.executor.owner, optimizer, selected)
        direct_grad_audit = self._direct_grad_audit(selected)
        grads = [param.grad for param in selected]
        self.executor.owner.set_grads(grads)
        owner_native_launch_prepare = self._prepare_owner_native_launch_probe() if self.config.owner_native_launch_probe else {}
        report = self.executor.shadow_step_from_grads(
            grads,
            sync_grads=False,
        )
        copyback_probe = self._copyback_probe(selected) if self.config.copyback_probe else {}
        copyback_dispatch_probe = (
            build_copyback_dispatch_probe(self.executor.owner, selected, scratch_probe=copyback_probe)
            if self.config.copyback_dispatch_experimental
            else {}
        )
        native_binding_probe = (
            build_update_native_binding_probe(self.executor.owner)
            if self.config.native_binding_probe
            else {}
        )
        owner_native_launch_probe = (
            self._finish_owner_native_launch_probe(owner_native_launch_prepare, report.owner_step)
            if self.config.owner_native_launch_probe
            else {}
        )
        checkpoint_contract = (
            build_flat_adamw_checkpoint_contract(
                self.executor.owner,
                optimizer=optimizer,
                params=selected,
                run_roundtrip=True,
            )
            if self.config.checkpoint_contract
            else {}
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "schema_version": 1,
            "mode": self.config.mode,
            "stage": "before_optimizer",
            "step": int(step),
            "parameter_tensors": len(selected),
            "parameter_numel": int(sum(param.numel() for param in selected)),
            "elapsed_ms": round(elapsed_ms, 4),
            "optimizer_state_sync": state_sync,
            "direct_grad_audit": direct_grad_audit,
            "copyback_probe": copyback_probe,
            "copyback_dispatch_probe": copyback_dispatch_probe,
            "native_binding_probe": native_binding_probe,
            "owner_native_launch_probe": owner_native_launch_probe,
            "checkpoint_contract": checkpoint_contract,
            "executor": report.as_dict(),
            "training_path_enabled": False,
        }
        return payload

    def prepare_before_backward(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        step: int,
        reset_owner_grad: bool = False,
    ) -> dict[str, Any]:
        """Install direct-grad hooks before backward for audit-only lifecycle proof."""

        started = time.perf_counter()
        if not self.enabled or not bool(self.config.direct_grad):
            return {}
        if self._auto_stopped:
            return self._direct_grad_lifecycle_skip("auto_stopped_after_consecutive_passes", started)
        selected = self._select_params(params)
        if not selected:
            return self._direct_grad_lifecycle_skip("no_trainable_params", started)
        if self.executor is None or self.params != selected:
            self.close()
            self.params = selected
            self.executor = TurboCoreUpdateExecutor(
                selected,
                self._executor_config(optimizer, max_grad_norm=max_grad_norm),
            )
            self._apply_pending_owner_state_if_possible()
        assert self.executor is not None
        if self.executor.direct_grad_binding is None:
            return self._direct_grad_lifecycle_skip("direct_grad_binding_missing", started)
        self.executor.direct_grad_binding.set_active(True)
        if reset_owner_grad:
            self.executor.direct_grad_binding.zero_owner_grad()
        report = {
            "schema_version": 1,
            "stage": "before_backward",
            "mode": self.config.mode,
            "step": int(step),
            "reset_owner_grad": bool(reset_owner_grad),
            "parameter_tensors": len(selected),
            "parameter_numel": int(sum(param.numel() for param in selected)),
            "snapshot": self.executor.direct_grad_binding.snapshot(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "training_path_enabled": False,
        }
        self._last_direct_grad_lifecycle_report = report
        return report

    def compare_after_optimizer(self, *, step: int) -> dict[str, Any]:
        if not self.enabled or self.executor is None or self._before_flat is None:
            return {}
        if self._auto_stopped:
            return {
                "schema_version": 1,
                "mode": self.config.mode,
                "stage": "after_optimizer",
                "compared": False,
                "skipped": True,
                "reason": "auto_stopped_after_consecutive_passes",
                "consecutive_passes": int(self._consecutive_passes),
                "training_path_enabled": False,
            }
        interval = max(int(self.config.compare_interval or 1), 1)
        if int(step) % interval != 0:
            return {"schema_version": 1, "mode": self.config.mode, "stage": "after_optimizer", "compared": False}
        started = time.perf_counter()
        compare_params, compare_ranges = self._comparison_selection(self.params)
        actual_flat = _flatten_params(compare_params)
        predicted_flat = _flatten_owner_ranges_like_params(self.executor.owner.param_flat.detach(), compare_params, compare_ranges)
        before = _flatten_ranges_like_params(self._before_flat.detach(), compare_params, compare_ranges)
        diff = (actual_flat.float() - predicted_flat.float()).abs()
        actual_delta = (actual_flat.float() - before.float()).abs()
        predicted_delta = (predicted_flat.float() - before.float()).abs()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        max_abs = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().detach().cpu().item()) if diff.numel() else 0.0
        max_actual_delta = float(actual_delta.max().detach().cpu().item()) if actual_delta.numel() else 0.0
        max_pred_delta = float(predicted_delta.max().detach().cpu().item()) if predicted_delta.numel() else 0.0
        parity_ok = bool(max_abs <= 5e-5 or max_abs <= max(max_actual_delta, max_pred_delta, 1e-12) * 0.02)
        if parity_ok:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0
        stop_after = max(int(self.config.stop_after_consecutive_passes or 0), 0)
        auto_stop_now = bool(stop_after > 0 and self._consecutive_passes >= stop_after)
        if auto_stop_now:
            self._auto_stopped = True
            if self.executor.direct_grad_binding is not None:
                self.executor.direct_grad_binding.set_active(False)
                self.executor.direct_grad_binding.remove()
        elif self.executor.direct_grad_binding is not None:
            self.executor.direct_grad_binding.zero_owner_grad()
        return {
            "schema_version": 1,
            "mode": self.config.mode,
            "stage": "after_optimizer",
            "step": int(step),
            "compared": True,
            "sampled": len(compare_params) != len(self.params),
            "sample_parameter_tensors": len(compare_params),
            "total_parameter_tensors": len(self.params),
            "elapsed_ms": round(elapsed_ms, 4),
            "max_abs_param_diff": max_abs,
            "mean_abs_param_diff": mean_abs,
            "max_actual_delta": max_actual_delta,
            "max_predicted_delta": max_pred_delta,
            "parity_ok_loose": parity_ok,
            "consecutive_passes": int(self._consecutive_passes),
            "auto_stopped_after_this_step": auto_stop_now,
            "training_path_enabled": False,
        }

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()
        if self._owner_native_launch_probe is not None:
            self._owner_native_launch_probe.close()
            self._owner_native_launch_probe = None
        self.executor = None
        self.params = []
        self._before_flat = None
        self._consecutive_passes = 0
        self._auto_stopped = False
        self._last_direct_grad_lifecycle_report = {}

    def _select_params(self, params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        selected = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if int(self.config.max_params or 0) > 0:
            selected = selected[: int(self.config.max_params)]
        return selected

    def _executor_config(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_grad_norm: float,
    ) -> TurboCoreUpdateExecutorConfig:
        group = optimizer.param_groups[0] if optimizer.param_groups else {}
        betas = group.get("betas", (0.9, 0.999))
        return TurboCoreUpdateExecutorConfig(
            optimizer="AdamW",
            lr=float(group.get("lr", 1e-4)),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(group.get("eps", 1e-8)),
            weight_decay=float(group.get("weight_decay", 0.0)),
            max_grad_norm=0.0,
            finite_check=True,
            direct_grad=bool(self.config.direct_grad),
            copy_params_back=False,
            zero_owner_grad=False,
            prefer_triton=bool(self.config.prefer_triton),
        )

    def _comparison_selection(self, params: list[torch.nn.Parameter]) -> tuple[list[torch.nn.Parameter], list[tuple[int, int]]]:
        limit = max(int(self.config.compare_sample_params or 0), 0)
        selected: list[torch.nn.Parameter] = []
        ranges: list[tuple[int, int]] = []
        offset = 0
        for index, param in enumerate(params):
            count = int(param.numel())
            if limit <= 0 or index < limit:
                selected.append(param)
                ranges.append((offset, count))
            offset += count
        return selected, ranges

    def _direct_grad_audit(self, params: list[torch.nn.Parameter]) -> dict[str, Any]:
        if self.executor is None or self.executor.direct_grad_binding is None:
            return {"enabled": False, "training_path_enabled": False}
        expected = _flatten_grads(params, device=self.executor.owner.grad_flat.device)
        actual = self.executor.owner.grad_flat.detach().float()
        diff = (actual - expected).abs()
        max_abs = float(diff.max().cpu().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().cpu().item()) if diff.numel() else 0.0
        snapshot = self.executor.direct_grad_binding.snapshot()
        warmup_no_writes = int(snapshot.get("writes", 0) or 0) <= 0
        parity_ok = None if warmup_no_writes else bool(max_abs <= 1e-6)
        mismatch_reason = ""
        if parity_ok is False:
            mismatch_reason = "param_grad_changed_after_backward_possible_clip_or_postprocess"
        return {
            "enabled": True,
            "max_abs_grad_diff": max_abs,
            "mean_abs_grad_diff": mean_abs,
            "parity_ok": parity_ok,
            "mismatch_reason": mismatch_reason,
            "warmup_no_writes": warmup_no_writes,
            "snapshot": snapshot,
            "training_path_enabled": False,
        }

    def _copyback_probe(self, params: list[torch.nn.Parameter]) -> dict[str, Any]:
        if self.executor is None:
            return {}
        started = time.perf_counter()
        real_before = _flatten_params(params)
        try:
            scratch = [torch.empty_like(param.detach()) for param in params]
            self.executor.owner.copy_params_to_(scratch)
            expected = _flatten_owner_like_params(self.executor.owner.param_flat.detach(), params)
            scratch_flat = _flatten_params(scratch)
            real_after = _flatten_params(params)
            scratch_diff = (scratch_flat.float() - expected.float()).abs()
            real_diff = (real_after.float() - real_before.float()).abs()
            scratch_max_abs = float(scratch_diff.max().detach().cpu().item()) if scratch_diff.numel() else 0.0
            scratch_mean_abs = float(scratch_diff.mean().detach().cpu().item()) if scratch_diff.numel() else 0.0
            real_max_abs = float(real_diff.max().detach().cpu().item()) if real_diff.numel() else 0.0
            shape_validated = all(tuple(target.shape) == tuple(source.shape) for target, source in zip(scratch, params))
            stride_preserved = all(tuple(target.stride()) == tuple(source.stride()) for target, source in zip(scratch, params))
            dtype_cast_validated = bool(scratch_max_abs == 0.0)
            real_parameters_mutated = bool(real_max_abs != 0.0)
            validated = bool(shape_validated and dtype_cast_validated and not real_parameters_mutated)
            error = ""
        except Exception as exc:  # pragma: no cover - defensive report for research probes
            expected = torch.empty(0, dtype=torch.float32)
            scratch_flat = torch.empty(0, dtype=torch.float32)
            scratch_max_abs = float("inf")
            scratch_mean_abs = float("inf")
            real_max_abs = float("inf")
            shape_validated = False
            stride_preserved = False
            dtype_cast_validated = False
            real_parameters_mutated = False
            validated = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "schema_version": 1,
            "probe": "turbocore_copyback_scratch_probe_v1",
            "parameter_tensors": len(params),
            "parameter_numel": int(sum(param.numel() for param in params)),
            "elapsed_ms": round(elapsed_ms, 4),
            "copyback_target": "scratch_tensors",
            "scratch_copyback_validated": validated,
            "shape_validated": shape_validated,
            "stride_preserved": stride_preserved,
            "dtype_cast_validated": dtype_cast_validated,
            "scratch_flat_numel": int(scratch_flat.numel()),
            "expected_flat_numel": int(expected.numel()),
            "scratch_max_abs_diff": scratch_max_abs,
            "scratch_mean_abs_diff": scratch_mean_abs,
            "real_parameter_max_abs_diff": real_max_abs,
            "real_parameters_mutated": real_parameters_mutated,
            "parameters_mutated": real_parameters_mutated,
            "target_dtypes": sorted({str(param.dtype).replace("torch.", "") for param in params}),
            "target_devices": sorted({str(param.device) for param in params}),
            "owner_flat_dtype": str(self.executor.owner.param_flat.dtype).replace("torch.", ""),
            "training_path_enabled": False,
        }
        if error:
            payload["error"] = error
        return payload

    def _prepare_owner_native_launch_probe(self) -> dict[str, Any]:
        if self.executor is None:
            return build_owner_native_launch_probe_skip(
                reason="executor_missing",
                max_numel=self.config.owner_native_launch_max_numel,
            )
        if self._owner_native_launch_probe is None:
            self._owner_native_launch_probe = TurboCoreOwnerNativeLaunchProbe(
                event_chain_probe=self.config.owner_native_event_chain_probe
            )
        else:
            self._owner_native_launch_probe.event_chain_probe = bool(self.config.owner_native_event_chain_probe)
        return self._owner_native_launch_probe.prepare_from_owner(self.executor.owner, max_numel=self.config.owner_native_launch_max_numel)

    def _finish_owner_native_launch_probe(self, prepare_report: dict[str, Any], owner_step_report: dict[str, Any] | None) -> dict[str, Any]:
        if self.executor is None or self._owner_native_launch_probe is None or not bool(prepare_report.get("prepared", False)):
            return prepare_report
        return self._owner_native_launch_probe.run_prepared(
            expected_owner_after=self.executor.owner,
            owner_step_report=owner_step_report,
            max_numel=self.config.owner_native_launch_max_numel,
        )

    def checkpoint_state(self, *, include_owner_state: bool = False) -> dict[str, Any]:
        owner_state: dict[str, Any] | None = None
        checkpoint_contract: dict[str, Any] = {}
        if self.executor is not None:
            checkpoint_contract = build_flat_adamw_checkpoint_contract(
                self.executor.owner,
                params=self.params,
                run_roundtrip=bool(self.config.checkpoint_contract),
            )
            if include_owner_state:
                owner_state = self.executor.owner.state_dict()
        return {
            "schema_version": 1,
            "state": "turbocore_update_shadow_checkpoint_v0",
            "mode": self.config.mode,
            "config": self.config.as_dict(),
            "training_path_enabled": False,
            "native_kernel_present": False,
            "checkpoint_metadata_integrated": True,
            "trainer_state_metadata_integrated": True,
            "owner_state_included": owner_state is not None,
            "parameter_tensors": len(self.params),
            "parameter_numel": int(sum(param.numel() for param in self.params)),
            "direct_grad_lifecycle": dict(self._last_direct_grad_lifecycle_report),
            "checkpoint_contract": checkpoint_contract,
            "owner_state_dict": owner_state,
        }

    def load_checkpoint_state(self, state: dict[str, Any], params: Iterable[torch.nn.Parameter] | None = None) -> dict[str, Any]:
        payload = dict(state or {})
        selected = self._select_params(params or [])
        owner_state = payload.get("owner_state_dict") if isinstance(payload.get("owner_state_dict"), dict) else None
        expected_numel = int(sum(param.numel() for param in selected)) if selected else 0
        state_numel = int(((owner_state or {}).get("layout") or {}).get("total_numel", 0) or 0)
        compatible = bool(not owner_state or not expected_numel or state_numel == expected_numel)
        if owner_state and compatible:
            self._pending_owner_state = dict(owner_state)
        report = {
            "schema_version": 1,
            "state": "turbocore_update_shadow_checkpoint_v0",
            "loaded": bool(payload),
            "compatible": compatible,
            "owner_state_pending": bool(owner_state and compatible),
            "expected_numel": expected_numel,
            "state_numel": state_numel,
            "training_path_enabled": False,
        }
        return report

    def _skip(self, reason: str, started: float) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.config.mode,
            "stage": "before_optimizer",
            "skipped": True,
            "reason": reason,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "training_path_enabled": False,
        }

    def _direct_grad_lifecycle_skip(self, reason: str, started: float) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "stage": "before_backward",
            "mode": self.config.mode,
            "skipped": True,
            "reason": reason,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "training_path_enabled": False,
        }
        self._last_direct_grad_lifecycle_report = report
        return report

    def _apply_pending_owner_state_if_possible(self) -> None:
        if self.executor is None or self._pending_owner_state is None:
            return
        try:
            self.executor.owner.load_state_dict(self._pending_owner_state)
        finally:
            self._pending_owner_state = None


def build_update_shadow_config(
    mode: str = "off",
    *,
    max_params: int = 0,
    compare_interval: int = 1,
    direct_grad: bool = False,
    prefer_triton: bool = False,
    compare_sample_params: int = 0,
    stop_after_consecutive_passes: int = 0,
    checkpoint_contract: bool = False,
    copyback_probe: bool = False,
    copyback_dispatch_experimental: bool = False,
    native_binding_probe: bool = False,
    owner_native_launch_probe: bool = False,
    owner_native_launch_max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
    owner_native_event_chain_probe: bool = False,
) -> TurboCoreUpdateShadowConfig:
    normalized = str(mode or "off").strip().lower().replace("-", "_")
    if normalized not in {"off", "profile", "shadow"}:
        normalized = "off"
    return TurboCoreUpdateShadowConfig(
        mode=normalized,
        max_params=max(int(max_params or 0), 0),
        compare_interval=max(int(compare_interval or 1), 1),
        direct_grad=bool(direct_grad),
        prefer_triton=bool(prefer_triton),
        compare_sample_params=max(int(compare_sample_params or 0), 0),
        stop_after_consecutive_passes=max(int(stop_after_consecutive_passes or 0), 0),
        checkpoint_contract=bool(checkpoint_contract),
        copyback_probe=bool(copyback_probe),
        copyback_dispatch_experimental=bool(copyback_dispatch_experimental),
        native_binding_probe=bool(native_binding_probe),
        owner_native_launch_probe=bool(owner_native_launch_probe),
        owner_native_launch_max_numel=max(int(owner_native_launch_max_numel or 0), 1),
        owner_native_event_chain_probe=bool(owner_native_event_chain_probe),
    )


def _flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [param.detach().float().reshape(-1) for param in params if isinstance(param, torch.Tensor)]
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors).contiguous()


def _flatten_grads(params: Iterable[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param in params:
        if not isinstance(param, torch.Tensor):
            continue
        grad = getattr(param, "grad", None)
        if grad is None:
            parts.append(torch.zeros(int(param.numel()), device=device, dtype=torch.float32))
        else:
            parts.append(grad.detach().float().reshape(-1).to(device=device))
    if not parts:
        return torch.empty(0, device=device, dtype=torch.float32)
    return torch.cat(parts).contiguous()


def _flatten_owner_like_params(flat: torch.Tensor, params: Iterable[torch.Tensor]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    offset = 0
    for param in params:
        if not isinstance(param, torch.Tensor):
            continue
        count = int(param.numel())
        part = flat.narrow(0, offset, count).view_as(param).to(dtype=param.dtype).detach().float().reshape(-1)
        parts.append(part)
        offset += count
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=flat.device)
    return torch.cat(parts).contiguous()


def _flatten_owner_ranges_like_params(flat: torch.Tensor, params: Iterable[torch.Tensor], ranges: Iterable[tuple[int, int]]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param, (offset, count) in zip(params, ranges):
        part = flat.narrow(0, int(offset), int(count)).view_as(param).to(dtype=param.dtype).detach().float().reshape(-1)
        parts.append(part)
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=flat.device)
    return torch.cat(parts).contiguous()


def _flatten_ranges_like_params(flat: torch.Tensor, params: Iterable[torch.Tensor], ranges: Iterable[tuple[int, int]]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param, (offset, count) in zip(params, ranges):
        part = flat.narrow(0, int(offset), int(count)).view_as(param).detach().float().reshape(-1)
        parts.append(part)
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=flat.device)
    return torch.cat(parts).contiguous()


__all__ = ["TurboCoreUpdateShadow", "TurboCoreUpdateShadowConfig", "build_update_shadow_config"]
