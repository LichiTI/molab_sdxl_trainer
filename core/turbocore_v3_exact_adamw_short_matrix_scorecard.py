"""Short real-training matrix for V3 exact AdamW native canary."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping
from unittest.mock import patch

import torch

import core.lulynx_trainer.training_loop as training_loop_module
from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_v3_exact_adamw_rollout_manifest_scorecard import (
    build_v3_exact_adamw_rollout_manifest_scorecard,
)


OPTIMIZER_KIND = "exact_adamw"
NATIVE_BACKEND = "rust_cuda_adamw_v0"
PARITY_MAX_ABS_DIFF = 5e-5


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


class _QuietProgress:
    def __init__(self, iterable: Iterable[Any], *_args: Any, **_kwargs: Any) -> None:
        self._iterable = iterable

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def build_v3_exact_adamw_short_matrix_scorecard(
    *,
    rollout_manifest: Mapping[str, Any] | None = None,
    steps: int = 4,
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Run a tiny baseline/canary matrix through the real TrainingLoop boundary."""

    manifest = dict(
        rollout_manifest
        or build_v3_exact_adamw_rollout_manifest_scorecard(
            native_training_mode="canary",
            run_live_training=run_live_training,
        )
    )
    if not run_live_training:
        matrix = _skipped_matrix("live_training_matrix_disabled")
    elif not torch.cuda.is_available():
        matrix = _skipped_matrix("cuda_required_for_v3_exact_adamw_short_matrix")
    else:
        matrix = _run_matrix(max(int(steps or 1), 2))
    comparison = _compare_cases(matrix)
    progress_gates = _progress_gates(manifest, matrix, comparison)
    ready = all(progress_gates.values())
    blockers = _blockers(progress_gates, matrix, comparison)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_short_matrix_scorecard_v0",
        "gate": "v3_exact_adamw_short_real_training_matrix",
        "ok": bool(manifest.get("ok", False)) and bool(matrix.get("ok", False)),
        "milestone_completed": ready,
        "short_matrix_ready": ready,
        "optimizer_kind": OPTIMIZER_KIND,
        "native_backend": NATIVE_BACKEND,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "rollout_manifest_summary": dict(manifest.get("manifest_summary") or {}),
        "matrix": matrix,
        "comparison": comparison,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "harden V3 runtime recovery around exact AdamW native canary"
            if ready
            else "complete V3-P2 exact AdamW short real-training matrix blockers"
        ),
        "notes": [
            "This matrix uses the real TrainingLoop optimizer-step boundary with a tiny synthetic parameter.",
            "Performance numbers are recorded for trend visibility only; this gate does not require speedup.",
            "Default training remains off even when the explicit canary case executes native steps.",
        ],
    }


def _run_matrix(steps: int) -> dict[str, Any]:
    initial = torch.tensor([1.0, -2.0, 0.5, -0.25], dtype=torch.float32, device="cuda")
    baseline = _run_case("baseline_off", native_enabled=False, initial=initial, steps=steps)
    canary = _run_case("explicit_canary", native_enabled=True, initial=initial, steps=steps)
    return {
        "schema_version": 1,
        "matrix": "v3_exact_adamw_short_real_training_matrix_v0",
        "ok": bool(baseline.get("ok", False)) and bool(canary.get("ok", False)),
        "steps": steps,
        "cases": [baseline, canary],
    }


def _run_case(
    case: str,
    *,
    native_enabled: bool,
    initial: torch.Tensor,
    steps: int,
) -> dict[str, Any]:
    loop, param = _make_loop(native_enabled=native_enabled, initial=initial, steps=steps)
    captured: list[dict[str, Any]] = []
    step_times: list[float] = []
    last_step_end = time.perf_counter()

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        params = self._get_trainable_params()
        for item in params:
            if item.grad is not None:
                item.grad = None
        loss = sum((item * item).sum() for item in params) / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    def _on_step_end(_step: int, _loss: float, info: dict[str, Any]) -> None:
        nonlocal last_step_end
        now = time.perf_counter()
        step_times.append(max(now - last_step_end, 0.0))
        last_step_end = now
        captured.append(dict(info))

    loop.on_step_end = _on_step_end
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step), patch.object(
        training_loop_module,
        "tqdm",
        new=_QuietProgress,
    ):
        result = loop.train_epoch([{} for _ in range(steps)], 0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory = 0
    elapsed = max(time.perf_counter() - started, 0.0)
    step_reports = [_step_report(info) for info in captured]
    native_steps = sum(1 for report in step_reports if report["native_step_executed"])
    pytorch_steps = sum(1 for report in step_reports if report["should_call_pytorch_optimizer_step"])
    state_sync_steps = sum(1 for report in step_reports if report["pytorch_optimizer_state_synced"])
    owner_backends = _dedupe([report["owner_backend"] for report in step_reports])
    return {
        "schema_version": 1,
        "case": case,
        "ok": int(result.get("steps", 0)) == steps and len(captured) == steps,
        "native_enabled": native_enabled,
        "result": result,
        "captured_step_count": len(captured),
        "elapsed_seconds": round(elapsed, 6),
        "mean_step_ms": round((elapsed / max(steps, 1)) * 1000.0, 4),
        "step_times_ms": [round(value * 1000.0, 4) for value in step_times],
        "peak_cuda_memory_bytes": peak_memory,
        "native_steps": native_steps,
        "pytorch_steps": pytorch_steps,
        "state_sync_steps": state_sync_steps,
        "owner_backends": owner_backends,
        "step_reports": step_reports,
        "final_param": [float(value) for value in param.detach().cpu().flatten().tolist()],
        "blocked_reasons": [] if int(result.get("steps", 0)) == steps else ["training_loop_step_count_mismatch"],
    }


def _make_loop(
    *,
    native_enabled: bool,
    initial: torch.Tensor,
    steps: int,
) -> tuple[TrainingLoop, torch.nn.Parameter]:
    param = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW([param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector([param]),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cuda",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=100.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode="shadow" if native_enabled else "off",
        turbocore_update_shadow_compare_interval=1,
        turbocore_update_shadow_direct_grad=False,
        turbocore_update_shadow_checkpoint_contract=native_enabled,
        turbocore_update_shadow_copyback_probe=native_enabled,
        turbocore_update_shadow_copyback_dispatch_experimental=native_enabled,
        turbocore_update_shadow_native_binding_probe=native_enabled,
        turbocore_update_shadow_owner_native_launch_probe=native_enabled,
        turbocore_update_shadow_owner_native_launch_max_numel=1024,
        turbocore_update_shadow_owner_native_event_chain_probe=native_enabled,
        turbocore_update_shadow_save_owner_state=native_enabled,
        turbocore_native_update_mode="native_experimental" if native_enabled else "off",
        turbocore_native_update_required_shadow_passes=1,
        turbocore_native_update_allow_missing_kernel=True,
        turbocore_native_update_dispatch_enabled=native_enabled,
        turbocore_native_update_training_path_enabled=native_enabled,
        turbocore_native_update_require_native_cuda=native_enabled,
    )
    loop.total_steps = steps
    return loop, param


def _step_report(info: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _as_dict(info.get("turbocore_native_update_dispatch_runtime"))
    executor = _as_dict(runtime.get("training_executor"))
    executor_result = _as_dict(executor.get("result"))
    update_report = _as_dict(executor_result.get("update_report"))
    return {
        "native_step_executed": bool(runtime.get("native_step_executed", False)),
        "native_kernel_launched": bool(runtime.get("native_kernel_launched", False)),
        "should_call_pytorch_optimizer_step": bool(
            runtime.get("should_call_pytorch_optimizer_step", True)
        ),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(executor.get("called", False)),
        "pytorch_optimizer_state_synced": bool(
            executor_result.get("pytorch_optimizer_state_synced", False)
        ),
        "owner_backend": str(update_report.get("owner_backend") or ""),
    }


def _compare_cases(matrix: Mapping[str, Any]) -> dict[str, Any]:
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), list) else []
    by_name = {
        str(case.get("case")): case
        for case in cases
        if isinstance(case, Mapping)
    }
    baseline = _as_dict(by_name.get("baseline_off"))
    canary = _as_dict(by_name.get("explicit_canary"))
    baseline_param = baseline.get("final_param") if isinstance(baseline.get("final_param"), list) else []
    canary_param = canary.get("final_param") if isinstance(canary.get("final_param"), list) else []
    diffs = [
        abs(float(left) - float(right))
        for left, right in zip(baseline_param, canary_param)
    ]
    baseline_ms = float(baseline.get("mean_step_ms") or 0.0)
    canary_ms = float(canary.get("mean_step_ms") or 0.0)
    return {
        "schema_version": 1,
        "final_param_max_abs_diff": max(diffs) if diffs else None,
        "parity_ok": bool(diffs) and max(diffs) <= PARITY_MAX_ABS_DIFF,
        "parity_threshold": PARITY_MAX_ABS_DIFF,
        "baseline_mean_step_ms": baseline_ms,
        "canary_mean_step_ms": canary_ms,
        "canary_over_baseline_step_time_ratio": (
            round(canary_ms / baseline_ms, 6) if baseline_ms > 0.0 else None
        ),
        "peak_memory_delta_bytes": int(canary.get("peak_cuda_memory_bytes") or 0)
        - int(baseline.get("peak_cuda_memory_bytes") or 0),
    }


def _progress_gates(
    manifest: Mapping[str, Any],
    matrix: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, bool]:
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), list) else []
    by_name = {
        str(case.get("case")): case
        for case in cases
        if isinstance(case, Mapping)
    }
    baseline = _as_dict(by_name.get("baseline_off"))
    canary = _as_dict(by_name.get("explicit_canary"))
    steps = int(matrix.get("steps") or 0)
    return {
        "rollout_manifest_ready": bool(manifest.get("rollout_manifest_ready", False)),
        "baseline_default_off": bool(baseline.get("ok", False))
        and int(baseline.get("native_steps") or 0) == 0
        and int(baseline.get("pytorch_steps") or 0) == steps,
        "canary_native_steps": bool(canary.get("ok", False))
        and int(canary.get("native_steps") or 0) >= max(steps - 1, 1)
        and NATIVE_BACKEND in set(canary.get("owner_backends") or []),
        "fallback_preserved": bool(canary.get("step_reports", [{}])[0].get("should_call_pytorch_optimizer_step", False))
        if canary.get("step_reports")
        else False,
        "state_sync_ready": int(canary.get("state_sync_steps") or 0)
        == int(canary.get("native_steps") or -1),
        "final_param_parity": bool(comparison.get("parity_ok", False)),
        "metrics_recorded": bool(comparison.get("baseline_mean_step_ms", 0.0))
        and bool(comparison.get("canary_mean_step_ms", 0.0)),
        "default_behavior_unchanged": (
            not bool(manifest.get("default_behavior_changed", True))
            and not bool(manifest.get("default_training_path_enabled", True))
        ),
    }


def _blockers(
    progress_gates: Mapping[str, bool],
    matrix: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    blockers = [f"v3_p2_{name}_missing" for name, ready in progress_gates.items() if not ready]
    if not bool(matrix.get("ok", False)):
        blockers.append("v3_p2_training_matrix_failed")
    if comparison.get("final_param_max_abs_diff") is None:
        blockers.append("v3_p2_final_param_diff_missing")
    return _dedupe(blockers)


def _skipped_matrix(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "matrix": "v3_exact_adamw_short_real_training_matrix_v0",
        "ok": False,
        "steps": 0,
        "cases": [],
        "skipped": True,
        "blocked_reasons": [str(reason)],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v3_exact_adamw_short_matrix_scorecard"]
