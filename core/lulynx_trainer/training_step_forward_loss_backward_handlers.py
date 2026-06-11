"""Forward input, loss-plan, and backward-plan handlers for train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .multi_batch_diagnostic_forward import run_lulynx_diagnostic_microbatch_forward
from .multi_batch_forward_strategy_switch import build_lulynx_multi_batch_forward_strategy_switch
from .training_pipeline_trace import LulynxTrainingPipelineTrace
from .training_step_backward_stage import LulynxTrainingStepBackwardStagePlan, build_lulynx_training_step_backward_stage_plan
from .training_step_forward_stage import LulynxTrainingStepForwardStagePlan, build_lulynx_training_step_forward_stage_plan
from .training_step_loss_stage import LulynxTrainingStepLossStagePlan, build_lulynx_training_step_loss_stage_plan
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxForwardInputStageExecution:
    forward_stage_plan: LulynxTrainingStepForwardStagePlan
    noisy_latents: Any
    prompt_embeds: dict[str, Any]
    unet_kwargs: dict[str, Any]
    control_residual_applied: bool
    ip_adapter_applied: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxForwardExecutionStageExecution:
    noise_pred: Any
    vram_diag_step: bool
    entropy_probe_step: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxLossPlanStageExecution:
    loss_stage_plan: LulynxTrainingStepLossStagePlan
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxBackwardPlanStageExecution:
    backward_stage_plan: LulynxTrainingStepBackwardStagePlan
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxBackwardExecutionStageExecution:
    backward_called: bool
    gradient_release_context_exited: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_forward_execution_stage_handler(
    *,
    owner: Any,
    unet_kwargs: Mapping[str, Any],
    hook_context: Mapping[str, Any],
    do_backward: bool,
    should_try_cudagraph: bool,
    multi_batch_execution_strategy: Mapping[str, Any] | None = None,
    multi_batch_execution_strategy_gate: Mapping[str, Any] | None = None,
    logger: Any,
) -> LulynxForwardExecutionStageExecution:
    """Execute the existing UNet forward path through the stage handler."""

    forward_strategy_switch = build_lulynx_multi_batch_forward_strategy_switch(
        execution_strategy=multi_batch_execution_strategy,
        execution_strategy_gate=multi_batch_execution_strategy_gate,
        implemented_runtime_routes=("existing_eager_forward_path", "diagnostic_microbatch_forward_path"),
    )
    noise_pred = None
    vram_diag_step = False
    entropy_probe_step = False
    try:
        device_type = torch.device(owner.device).type
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=owner.dtype)
            if device_type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            vram_diag_step = (
                owner._advanced_monitoring
                and torch.cuda.is_available()
                and owner.global_step % owner._peak_vram_diag_interval == 0
            )
            if vram_diag_step:
                torch.cuda.reset_peak_memory_stats()

            entropy_probe_step = (
                owner._advanced_monitoring
                and owner.global_step % owner._attn_entropy_interval == 0
            )
            if entropy_probe_step:
                from .attn_entropy import arm_probe

                arm_probe()

            act_drift_step = (
                owner._advanced_monitoring
                and owner.global_step % owner._act_drift_interval == 0
            )
            if act_drift_step and owner._act_drift_tracker is None:
                from .act_drift import ActivationDriftTracker

                anchors = [item.strip() for item in owner._act_drift_anchor_layers.split(",") if item.strip()] or None
                owner._act_drift_tracker = ActivationDriftTracker(owner.unet, anchor_layers=anchors)
                owner._act_drift_tracker.install()

            if owner.lulynx_wrapper and not owner.lulynx_wrapper._hooks:
                owner.lulynx_wrapper.register_feature_hooks(owner.unet)
            if do_backward:
                try:
                    from core.services.training_hooks import emit_before_forward_event
                except Exception:  # pragma: no cover - optional launcher/plugin surface
                    emit_before_forward_event = None
                if emit_before_forward_event is not None:
                    emit_before_forward_event(**dict(hook_context))

            with owner._activation_compression_ctx.context():
                selected_forward_route = str(forward_strategy_switch.get("selected_forward_route") or "")
                route_ready = str(forward_strategy_switch.get("status") or "") == "ready_to_use_selected_forward_route"
                if route_ready and selected_forward_route == "diagnostic_microbatch_forward_path":
                    diagnostic = run_lulynx_diagnostic_microbatch_forward(
                        unet=owner.unet,
                        unet_kwargs=dict(unet_kwargs),
                        microbatch_size=1,
                    )
                    noise_pred = diagnostic.output
                    forward_strategy_switch = dict(forward_strategy_switch)
                    forward_strategy_switch["diagnostic_forward_report"] = diagnostic.report
                elif owner._cudagraph_active and should_try_cudagraph:
                    replay_out = owner._cudagraph_replay(dict(unet_kwargs))
                    if replay_out is not None:
                        noise_pred = replay_out.sample if hasattr(replay_out, "sample") else replay_out
                    else:
                        owner._cudagraph_active = False
                        logger.warning("[CUDAGraph] Replay returned None - falling back to eager")
                        noise_pred = owner.unet(**dict(unet_kwargs)).sample
                elif owner.cpu_offload_checkpointing:
                    noise_pred = _run_offloaded_forward(owner, dict(unet_kwargs))
                else:
                    noise_pred = owner.unet(**dict(unet_kwargs)).sample

            if (
                do_backward
                and should_try_cudagraph
                and selected_forward_route != "diagnostic_microbatch_forward_path"
                and not owner._cudagraph_active
                and owner._cudagraph_capture is None
                and owner.global_step == 0
            ):
                owner._try_init_cudagraph(dict(unet_kwargs))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and owner.safe_fallback:
            if torch.cuda.is_available():
                owner._maybe_release_cuda_cache("oom_recovery", owner.global_step, force=True)
            owner._cudagraph_active = False
            owner._cudagraph_capture = None
            logger.error(
                "[SafeFallback] CUDA OOM during UNet forward. CPU step fallback is disabled "
                "because it can mix CPU/CUDA tensors; lower batch/resolution or enable cache/checkpointing."
            )
        raise

    return LulynxForwardExecutionStageExecution(
        noise_pred=noise_pred,
        vram_diag_step=bool(vram_diag_step),
        entropy_probe_step=bool(entropy_probe_step),
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward"),
            status="forward_execution_stage_handler_executed",
            handler_source="existing_training_loop_forward_execution_path",
            extra={"multi_batch_forward_strategy_switch": forward_strategy_switch},
        ),
    )


def _run_offloaded_forward(owner: Any, unet_kwargs: dict[str, Any]) -> Any:
    if owner._offloaded_checkpoint_ctx is not None:
        from .offloaded_checkpointing import offloaded_checkpoint_forward

        def _unet_forward_offloaded(**kwargs: Any) -> Any:
            return owner.unet(**kwargs)

        return offloaded_checkpoint_forward(
            _unet_forward_offloaded,
            ctx=owner._offloaded_checkpoint_ctx,
            **unet_kwargs,
        ).sample
    from .memory_optimizations import cpu_offload_checkpoint

    def _unet_forward(**kwargs: Any) -> Any:
        return owner.unet(**kwargs)

    return cpu_offload_checkpoint(_unet_forward, **unet_kwargs).sample


def run_lulynx_forward_input_stage_handler(
    *,
    batch: Mapping[str, Any],
    noisy_latents: Any,
    timesteps: Any,
    prompt_embeds: dict[str, Any],
    padding_mask: Any,
    batch_size: int,
    model_arch: str,
    cached_native: bool,
    trace: LulynxTrainingPipelineTrace,
    device: Any,
    target_dtype: Any,
    easy_control: Any,
    ip_adapter: Any,
    cudagraph_active: bool,
    cudagraph_requested: bool,
    cpu_offload_checkpointing: bool,
    offloaded_checkpoint_context_available: bool,
    profiled_to: Callable[..., Any],
    get_timestep_embedding: Callable[..., Any],
) -> LulynxForwardInputStageExecution:
    control_residual_applied = False
    if easy_control is not None and isinstance(batch.get("control_images"), torch.Tensor):
        control_images = profiled_to(batch["control_images"], label="control_images", dtype=target_dtype)
        control_residual = easy_control(control_images, target_size=noisy_latents.shape[-2:])
        noisy_latents = noisy_latents + control_residual.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        control_residual_applied = True

    ip_adapter_applied = False
    ip_features = batch.get("ip_adapter_image_features")
    if ip_adapter is not None and (isinstance(batch.get("ip_adapter_images"), torch.Tensor) or isinstance(ip_features, torch.Tensor)):
        if isinstance(ip_features, torch.Tensor):
            image_tokens = ip_adapter.projector(profiled_to(ip_features, label="ip_adapter_image_features", dtype=target_dtype)) * ip_adapter.config.scale
        else:
            image_tokens = ip_adapter(profiled_to(batch["ip_adapter_images"], label="ip_adapter_images", dtype=target_dtype))
        merged_tokens, merged_mask = ip_adapter.merge_with_text_cond(
            image_tokens,
            prompt_embeds.get("encoder_hidden_states"),
            prompt_embeds.get("attention_mask"),
        )
        prompt_embeds["encoder_hidden_states"] = merged_tokens.to(device, dtype=target_dtype)
        if merged_mask is not None:
            prompt_embeds["attention_mask"] = merged_mask.to(device)
        ip_adapter_applied = True

    time_embeds = get_timestep_embedding(
        batch_size,
        batch.get("original_sizes", [(1024, 1024)] * batch_size),
        batch.get("target_sizes", [(1024, 1024)] * batch_size),
        batch.get("crop_coords", [(0, 0, 1024, 1024)] * batch_size),
    )
    unet_kwargs = {
        "sample": noisy_latents,
        "timestep": timesteps,
        "encoder_hidden_states": prompt_embeds["encoder_hidden_states"],
    }
    pooled = prompt_embeds.get("pooled_prompt_embeds")
    time_cond = time_embeds.get("added_cond_kwargs", {}) if time_embeds else {}
    if time_cond or pooled is not None:
        cond = dict(time_cond)
        if pooled is not None:
            cond["text_embeds"] = pooled
        unet_kwargs["added_cond_kwargs"] = cond
    if str(model_arch or "").strip().lower() == "anima" and padding_mask is not None:
        unet_kwargs["padding_mask"] = padding_mask
    qwen3_hs = prompt_embeds.get("qwen3_hidden_states")
    if qwen3_hs is not None:
        unet_kwargs["qwen3_hidden_states"] = qwen3_hs
        qwen3_mask = prompt_embeds.get("qwen3_attention_mask")
        if qwen3_mask is not None:
            unet_kwargs["qwen3_attention_mask"] = qwen3_mask

    plan = build_lulynx_training_step_forward_stage_plan(
        unet_kwargs=unet_kwargs,
        model_arch=model_arch,
        cudagraph_active=cudagraph_active,
        cudagraph_requested=cudagraph_requested,
        cpu_offload_checkpointing=cpu_offload_checkpointing,
        offloaded_checkpoint_context_available=offloaded_checkpoint_context_available,
        control_residual_applied=control_residual_applied,
        ip_adapter_applied=ip_adapter_applied,
    )
    trace.mark("forward", model_arch=str(model_arch), cached_native=bool(cached_native), forward_stage_plan=plan.as_dict())
    return LulynxForwardInputStageExecution(
        forward_stage_plan=plan,
        noisy_latents=noisy_latents,
        prompt_embeds=prompt_embeds,
        unet_kwargs=unet_kwargs,
        control_residual_applied=control_residual_applied,
        ip_adapter_applied=ip_adapter_applied,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward"),
            status="forward_input_stage_handler_executed",
            handler_source="existing_training_loop_forward_input_path",
            stage_plans={"forward_stage_plan": plan.as_dict()},
        ),
    )


def run_lulynx_loss_plan_stage_handler(*, trace: LulynxTrainingPipelineTrace, **kwargs: Any) -> LulynxLossPlanStageExecution:
    plan = build_lulynx_training_step_loss_stage_plan(**kwargs)
    trace.mark("loss", loss_stage_plan=plan.as_dict())
    return LulynxLossPlanStageExecution(
        loss_stage_plan=plan,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss"),
            status="loss_plan_stage_handler_executed",
            handler_source="existing_training_loop_loss_plan_path",
            stage_plans={"loss_stage_plan": plan.as_dict()},
        ),
    )


def run_lulynx_backward_execution_stage_handler(
    *,
    loss: Any,
    gradient_release_context: Any,
    optimizer_used_fused_backward: bool,
    optimizer_deferred_step_closure: bool,
    create_graph_backward: bool,
) -> LulynxBackwardExecutionStageExecution:
    """Execute the current backward call and gradient-release exit unchanged."""

    backward_called = False
    if not optimizer_used_fused_backward and not optimizer_deferred_step_closure:
        loss.backward(create_graph=create_graph_backward)
        backward_called = True
    context_exited = False
    if gradient_release_context is not None:
        gradient_release_context.__exit__(None, None, None)
        context_exited = True
    return LulynxBackwardExecutionStageExecution(
        backward_called=backward_called,
        gradient_release_context_exited=context_exited,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss", "backward"),
            status="backward_execution_stage_handler_executed",
            handler_source="existing_training_loop_backward_execution_path",
        ),
    )


def run_lulynx_backward_plan_stage_handler(
    *,
    do_backward: bool,
    sync_gradients: bool,
    accumulation_steps: int,
    uses_step_closure: bool,
    uses_fused_backward: bool,
    gradient_release_mode: str,
    create_graph_backward: bool,
    trace: LulynxTrainingPipelineTrace,
) -> LulynxBackwardPlanStageExecution:
    plan = build_lulynx_training_step_backward_stage_plan(
        do_backward=do_backward,
        sync_gradients=sync_gradients,
        gradient_accumulation_steps=accumulation_steps,
        uses_step_closure=uses_step_closure,
        uses_fused_backward=uses_fused_backward,
        gradient_release_mode=gradient_release_mode,
        create_graph_backward=create_graph_backward,
    )
    trace.mark("backward", sync_gradients=bool(sync_gradients), backward_stage_plan=plan.as_dict())
    return LulynxBackwardPlanStageExecution(
        backward_stage_plan=plan,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss", "backward"),
            status="backward_plan_stage_handler_executed",
            handler_source="existing_training_loop_backward_plan_path",
            stage_plans={"backward_stage_plan": plan.as_dict()},
        ),
    )


__all__ = [
    "LulynxBackwardExecutionStageExecution",
    "LulynxBackwardPlanStageExecution",
    "LulynxForwardExecutionStageExecution",
    "LulynxForwardInputStageExecution",
    "LulynxLossPlanStageExecution",
    "run_lulynx_backward_plan_stage_handler",
    "run_lulynx_backward_execution_stage_handler",
    "run_lulynx_forward_execution_stage_handler",
    "run_lulynx_forward_input_stage_handler",
    "run_lulynx_loss_plan_stage_handler",
]
