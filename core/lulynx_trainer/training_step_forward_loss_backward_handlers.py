"""Forward input, loss-plan, and backward-plan handlers for train steps."""

from __future__ import annotations

import logging
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
from .newbie_backward_op_profile import profile_backward_autograd_call

logger = logging.getLogger(__name__)


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

            # saved_tensors_hooks do not nest (innermost wins): when the pinned
            # CPU offload is enabled it takes precedence over compression.
            _offload_ctx = getattr(owner, "_activation_offload_ctx", None)
            _saved_tensor_ctx = (
                _offload_ctx.context()
                if _offload_ctx is not None and getattr(_offload_ctx, "enabled", False)
                else owner._activation_compression_ctx.context()
            )
            with _saved_tensor_ctx:
                # JLT-style EMA feature alignment (default-off): when enabled,
                # capture student block features and force an eager forward so
                # the python ``_run_blocks`` loop runs (cudagraph replay and the
                # offloaded-checkpoint path bypass it, leaving capture empty).
                _ema_capture_on = (
                    do_backward
                    and getattr(owner, "anima_ema_feat_align_enabled", False)
                    and float(getattr(owner, "anima_ema_feat_align_weight", 0.0) or 0.0) > 0.0
                    and str(getattr(owner, "_model_arch", "") or "").strip().lower() == "anima"
                )
                if _ema_capture_on:
                    from .anima_feature_capture import feature_capture_scope, parse_layer_list
                    from .anima_ema_feature_align import EmaLoraShadow

                    # Refresh the EMA-of-LoRA teacher shadow from the current
                    # (post previous optimizer.step) weights, then capture the
                    # student features under an eager forward.
                    if getattr(owner, "_ema_lora_shadow", None) is None:
                        owner._ema_lora_shadow = EmaLoraShadow()
                    owner._ema_lora_shadow.update(
                        list(owner.unet.named_parameters()),
                        float(getattr(owner, "anima_ema_feat_align_decay", 0.9999) or 0.9999),
                    )
                    _student_layers = parse_layer_list(
                        getattr(owner, "anima_ema_feat_align_student_layers", "")
                    )
                    with feature_capture_scope(_student_layers) as _cap:
                        noise_pred = owner.unet(**dict(unet_kwargs)).sample
                    owner._ema_feat_student_features = dict(_cap.features)
                else:
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
                and not _ema_capture_on
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
    anima_faithful_forward: bool = False,
    anima_llm_adapter: Any = None,
    easycontrol_v2_adapter: Any = None,
    easycontrol_v2_clean_latents: Any = None,
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

    # EasyControl v2 two-stream: set the condition the patched blocks read during
    # this step's forward. Production path uses paired sidecar cond_latents; when
    # absent (e.g. the cache-first anima path), derive a coarse reference by
    # adaptive-pooling the clean target latent to a fixed 16x16 grid so the
    # mechanism A/B is runnable without a sidecar pipeline (this leaks the target
    # -> a usable condition that proves the stream is wired; it is NOT a control-
    # quality signal). No condition -> clear (patched forward is bitwise-original).
    easycontrol_v2_applied = False
    if easycontrol_v2_adapter is not None:
        cond_src = batch.get("cond_latents")
        if isinstance(cond_src, torch.Tensor):
            cond_lat = profiled_to(cond_src, label="cond_latents", dtype=target_dtype).to(device)
            easycontrol_v2_adapter.set_cond(cond_lat)
            easycontrol_v2_applied = True
            _cond_source = "sidecar"
        elif isinstance(easycontrol_v2_clean_latents, torch.Tensor):
            ref = torch.nn.functional.adaptive_avg_pool2d(
                easycontrol_v2_clean_latents.to(device=device, dtype=target_dtype), (16, 16)
            )
            easycontrol_v2_adapter.set_cond(ref)
            easycontrol_v2_applied = True
            _cond_source = "derived_from_target"
        else:
            easycontrol_v2_adapter.clear_cond()
            _cond_source = "none"
        # One-time observability: which condition source the colorize/v2 stream is
        # consuming. "sidecar" = a real control-image cond latent (production);
        # "derived_from_target" = the cache-first fallback (mechanism A/B only, leaks
        # target). Logged once per adapter so the hot path stays quiet.
        if easycontrol_v2_applied and not getattr(easycontrol_v2_adapter, "_cond_source_logged", False):
            logger.info("[easycontrol-v2] condition source = %s", _cond_source)
            try:
                easycontrol_v2_adapter._cond_source_logged = True
            except Exception:
                pass

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

    # #147 fix ③: faithful native forward conditions on the *frozen* llm_adapter
    # output (Qwen3 hidden + T5 ids -> cross-attn context), not raw Qwen3 hidden.
    # The faithful executable forward signature is (sample, timestep,
    # encoder_hidden_states, padding_mask) -> override encoder_hidden_states and
    # strip the stub-only qwen3_* kwargs it does not accept. Default off -> this
    # block is skipped and the legacy (#132) raw-Qwen3 context is bitwise-unchanged.
    anima_faithful_context_applied = False
    if anima_faithful_forward and anima_llm_adapter is not None and str(model_arch or "").strip().lower() == "anima":
        from .anima_faithful_train_context import resolve_anima_faithful_context

        context = resolve_anima_faithful_context(
            prompt_embeds, batch, anima_llm_adapter, device, target_dtype
        )
        unet_kwargs["encoder_hidden_states"] = context
        unet_kwargs.pop("qwen3_hidden_states", None)
        unet_kwargs.pop("qwen3_attention_mask", None)
        anima_faithful_context_applied = True

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
    trace.mark("forward", model_arch=str(model_arch), cached_native=bool(cached_native), anima_faithful_context=anima_faithful_context_applied, forward_stage_plan=plan.as_dict())
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
    step_phase_profiler: Any = None,
    substage_label_prefix: str = "",
    backward_op_profile_enabled: bool = False,
    backward_op_profile_top_k: int = 12,
    backward_op_profile_record_shapes: bool = False,
    backward_op_profile_sink: Callable[[dict[str, Any]], None] | None = None,
) -> LulynxBackwardExecutionStageExecution:
    """Execute the current backward call and gradient-release exit unchanged."""

    label_prefix = str(substage_label_prefix or "").strip().rstrip(".")

    def _start_profile() -> Any:
        if step_phase_profiler is None or not label_prefix:
            return None
        start = getattr(step_phase_profiler, "start", None)
        if not callable(start):
            return None
        return start()

    def _record_profile(label: str, started: Any) -> None:
        if started is None or step_phase_profiler is None or not label_prefix:
            return
        record = getattr(step_phase_profiler, "record", None)
        if callable(record):
            record(f"{label_prefix}.{label}", started)

    backward_called = False
    if not optimizer_used_fused_backward and not optimizer_deferred_step_closure:
        backward_started = _start_profile()
        if bool(backward_op_profile_enabled):
            profile = profile_backward_autograd_call(
                lambda: loss.backward(create_graph=create_graph_backward),
                top_k=max(int(backward_op_profile_top_k or 0), 1),
                record_shapes=bool(backward_op_profile_record_shapes),
            )
            if callable(backward_op_profile_sink):
                backward_op_profile_sink(profile)
        else:
            loss.backward(create_graph=create_graph_backward)
        _record_profile("backward_autograd_call", backward_started)
        backward_called = True
    context_exited = False
    if gradient_release_context is not None:
        release_started = _start_profile()
        gradient_release_context.__exit__(None, None, None)
        _record_profile("backward_gradient_release_exit", release_started)
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
