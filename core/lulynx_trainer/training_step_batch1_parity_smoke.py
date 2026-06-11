"""CPU-only batch1 handler parity smoke for the staged train-step shell."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from .training_pipeline_contract import lulynx_training_stage_ids
from .training_pipeline_trace import LulynxTrainingPipelineTrace, compact_lulynx_pipeline_trace
from .training_step_orchestrator import run_lulynx_training_step_orchestrator_slice
from .training_step_orchestrator_handlers import (
    run_lulynx_backward_execution_stage_handler,
    run_lulynx_backward_plan_stage_handler,
    run_lulynx_batch_contract_stage_handler,
    run_lulynx_forward_execution_stage_handler,
    run_lulynx_forward_input_stage_handler,
    run_lulynx_loss_execution_stage_handler,
    run_lulynx_loss_plan_stage_handler,
    run_lulynx_noise_timestep_stage_handler,
    run_lulynx_optimizer_execution_stage_handler,
    run_lulynx_telemetry_execution_stage_handler,
    run_lulynx_transfer_conditioning_stage_handler,
)


LULYNX_BATCH1_HANDLER_PARITY_SMOKE = "lulynx_batch1_handler_parity_smoke_v0"


class _ToyScheduler:
    config = SimpleNamespace(num_train_timesteps=4)
    alphas_cumprod = torch.linspace(0.1, 0.9, 4)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return latents + noise * 0.0


class _ToyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, **kwargs: Any) -> Any:
        return SimpleNamespace(sample=kwargs["sample"] * self.weight)


class _ScalarCache:
    def get(self, tensor: torch.Tensor) -> float:
        return float(tensor.detach().float().item())


def run_lulynx_batch1_handler_parity_smoke() -> dict[str, Any]:
    """Compare direct handler execution with the behavior-equivalent shell."""

    direct = _run_toy_batch1_step(through_orchestrator=False)
    orchestrated = _run_toy_batch1_step(through_orchestrator=True)
    checks = {
        "stage_order_match": orchestrated["stage_ids"] == lulynx_training_stage_ids()
        and direct["stage_ids"] == lulynx_training_stage_ids(),
        "loss_match": _close(orchestrated["loss"], direct["loss"]),
        "weight_match": _close(orchestrated["weight"], direct["weight"]),
        "forward_route_existing_eager": orchestrated["forward_route"] == "existing_eager_forward_path",
        "trace_completed_tail": orchestrated["compact_trace"].get("stage_ids", [])[-2:]
        == ["optimizer_step", "telemetry"],
        "release_claim_closed": not bool(orchestrated.get("release_claim_allowed"))
        and not bool(orchestrated["compact_trace"].get("multi_batch_execution_strategy", {}).get("release_claim_allowed")),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    passed = not blockers
    return {
        "schema_version": 1,
        "smoke": LULYNX_BATCH1_HANDLER_PARITY_SMOKE,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "release_claim_allowed": False,
        "does_not_start_gpu_work": True,
        "does_not_add_training_entrypoint": True,
        "behavior_equivalent_only": True,
        "required_before_internal_orchestrator_gate": True,
        "direct": _summary(direct),
        "orchestrated": _summary(orchestrated),
        "checks": checks,
        "blockers": blockers,
    }


def _run_toy_batch1_step(*, through_orchestrator: bool) -> dict[str, Any]:
    torch.manual_seed(1337)
    trace = LulynxTrainingPipelineTrace()
    owner = _toy_owner()
    optimizer = torch.optim.SGD(owner.unet.parameters(), lr=0.1)
    context: dict[str, Any] = {
        "batch": {
            "latents": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
            "encoder_hidden_states": torch.tensor([[1.0, -1.0]], dtype=torch.float32),
            "captions": ["toy"],
        },
        "trace": trace,
        "owner": owner,
        "optimizer": optimizer,
    }
    handlers = _stage_handlers()
    if through_orchestrator:
        report = run_lulynx_training_step_orchestrator_slice(
            stage_handlers=handlers,
            execution_readiness=_ready_execution_gate(),
            internal_gate_enabled=True,
            initial_context=context,
        )
        context = report["context"]
    else:
        report = {"status": "direct_handler_sequence", "stage_ids": []}
        for stage_id in lulynx_training_stage_ids():
            handlers[stage_id](context)
            report["stage_ids"].append(stage_id)

    compact_trace = compact_lulynx_pipeline_trace(context["telemetry_execution"].completed_trace)
    return {
        "status": report["status"],
        "stage_ids": report["stage_ids"],
        "loss": float(context["loss_execution"].loss.detach().item()),
        "weight": float(context["owner"].unet.weight.detach().item()),
        "compact_trace": compact_trace,
        "forward_route": context["forward_execution"].orchestrator_runtime["multi_batch_forward_strategy_switch"][
            "selected_forward_route"
        ],
        "release_claim_allowed": bool(report.get("release_claim_allowed", False)),
    }


def _stage_handlers() -> dict[str, Any]:
    return {
        "dataset_scan": _dataset_scan,
        "bucket_plan": _bucket_plan,
        "batch_collate": _batch_collate,
        "batch_contract": _batch_contract,
        "host_to_device": _transfer,
        "conditioning": lambda ctx: ctx.get("transfer_execution").orchestrator_runtime,
        "noise_timestep": _noise,
        "forward": _forward,
        "loss": _loss,
        "backward": _backward,
        "optimizer_step": _optimizer_step,
        "telemetry": _telemetry,
    }


def _toy_owner() -> SimpleNamespace:
    unet = _ToyUnet()
    scheduler = _ToyScheduler()
    return SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
        _advanced_monitoring=False,
        global_step=1,
        _peak_vram_diag_interval=10,
        _attn_entropy_interval=10,
        _act_drift_interval=10,
        _act_drift_tracker=None,
        _act_drift_anchor_layers="",
        lulynx_wrapper=None,
        lora_injector=None,
        _activation_compression_ctx=SimpleNamespace(context=lambda: nullcontext()),
        _cudagraph_active=False,
        cpu_offload_checkpointing=False,
        _offloaded_checkpoint_ctx=None,
        _cudagraph_capture=None,
        safe_fallback=False,
        unet=unet,
        noise_scheduler=scheduler,
        _try_init_cudagraph=lambda _kwargs: False,
        _cudagraph_replay=lambda _kwargs: None,
        _maybe_release_cuda_cache=lambda *_, **__: None,
        _model_arch="sd15",
        _sdxl_flow_sigmas=None,
        _sdxl_flow_weighting="none",
        flow_logit_mean=0.0,
        flow_logit_std=1.0,
        snr_gamma=None,
        debiased_estimation=False,
        v_parameterization=False,
        adaptive_loss_weighter=None,
        wavelet_loss_enabled=False,
        pattern_loss_enabled=False,
        prior_loss_weight=0.0,
        reg_dataloader=None,
        _reg_iter=None,
        dop=None,
        b_tier_runtime=None,
        _loss_tracker=None,
        _loss_operands=lambda noise_pred, target: (noise_pred, target),
        _compute_diffusion_loss=lambda pred, target, reduction="none", timesteps=None: torch.nn.functional.mse_loss(
            pred,
            target,
            reduction=reduction,
        ),
        _loss_to_per_sample=lambda loss, _batch: loss.view(loss.shape[0], -1).mean(dim=1),
        _weighted_mean_loss=lambda loss, _batch: loss.mean(),
        _compute_repa_loss=lambda _batch, _prompt_embeds: None,
    )


def _dataset_scan(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["dataset_scan_observed"] = True
    return {"observed": True}


def _bucket_plan(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["bucket_plan_observed"] = True
    return {"observed": True}


def _batch_collate(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["batch_collate_observed"] = True
    return {"observed": True}


def _batch_contract(ctx: dict[str, Any]) -> Any:
    execution = run_lulynx_batch_contract_stage_handler(
        batch=ctx["batch"],
        model_arch="sd15",
        trace=ctx["trace"],
        previous_training_data_pipeline_report={
            "report": "lulynx_training_data_pipeline_report_v0",
            "batch_collate_stage_plan": {"observed": True},
            "missing_runtime_evidence": ["batch_collate_not_observed_without_dataloader_iteration"],
        },
        accumulation_steps=1,
        do_backward=True,
    )
    ctx["batch_contract_execution"] = execution
    return execution.orchestrator_runtime


def _transfer(ctx: dict[str, Any]) -> Any:
    execution = run_lulynx_transfer_conditioning_stage_handler(
        batch=ctx["batch"],
        batch_stage_plan=ctx["batch_contract_execution"].batch_stage_plan,
        model_arch="sd15",
        trace=ctx["trace"],
        target_dtype=torch.float32,
        images=None,
        captions=ctx["batch"]["captions"],
        qwen3_encoder_available=False,
        do_backward=True,
        te_dropout=0.0,
        clip_l_dropout_rate=0.0,
        clip_g_dropout_rate=0.0,
        t5_dropout_rate=0.0,
        profiled_to=lambda value, **_: value,
        encode_latents_with_vae=lambda images: images,
        encode_prompt=lambda captions: {},
        encode_qwen3=lambda captions: {},
        apply_text_encoder_dropout=lambda prompt_embeds, **_: prompt_embeds,
    )
    ctx["transfer_execution"] = execution
    return execution.orchestrator_runtime


def _noise(ctx: dict[str, Any]) -> Any:
    execution = run_lulynx_noise_timestep_stage_handler(
        latents=ctx["transfer_execution"].latents,
        model_arch="sd15",
        trace=ctx["trace"],
        device="cpu",
        flow_model="",
        noise_scheduler=ctx["owner"].noise_scheduler,
        v_parameterization=False,
        optimal_noise_enabled=False,
        optimal_noise_candidates=1,
        multires_noise_iterations=0,
        multires_noise_discount=0.0,
        spectral_noise_blend=0.0,
        spectral_noise_sigma=0.0,
        noise_offset=0.0,
        adaptive_noise_scale=0.0,
        noise_offset_random_strength=False,
        perlin_noise_offset_enabled=False,
        perlin_noise_offset_strength=0.0,
        perlin_noise_offset_scale=1.0,
        flow_use_ot=False,
        ddpm_timestep_sampling="uniform",
        anima_timestep_sampling="uniform",
        anima_sigmoid_scale=1.0,
        anima_discrete_flow_shift=1.0,
        anima_weighting_scheme="none",
        anima_model_prediction_type="velocity",
        sdxl_timestep_sampling="uniform",
        sdxl_sigmoid_scale=1.0,
        sdxl_flow_shift=1.0,
        sdxl_flow_weighting_scheme="none",
        sdxl_model_prediction_type="epsilon",
        flow_logit_mean=0.0,
        flow_logit_std=1.0,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
        sample_strength=lambda value, *_: value,
        velocity_target=lambda latents, noise, timesteps: noise,
    )
    ctx["noise_execution"] = execution
    return execution.orchestrator_runtime


def _forward(ctx: dict[str, Any]) -> Any:
    forward_input = run_lulynx_forward_input_stage_handler(
        batch=ctx["batch"],
        noisy_latents=ctx["noise_execution"].noisy_latents,
        timesteps=ctx["noise_execution"].timesteps,
        prompt_embeds=ctx["transfer_execution"].prompt_embeds,
        padding_mask=ctx["transfer_execution"].padding_mask,
        batch_size=ctx["noise_execution"].batch_size,
        model_arch="sd15",
        cached_native=True,
        trace=ctx["trace"],
        device="cpu",
        target_dtype=torch.float32,
        easy_control=None,
        ip_adapter=None,
        cudagraph_active=False,
        cudagraph_requested=False,
        cpu_offload_checkpointing=False,
        offloaded_checkpoint_context_available=False,
        profiled_to=lambda value, **_: value,
        get_timestep_embedding=lambda *_, **__: {},
    )
    forward_execution = run_lulynx_forward_execution_stage_handler(
        owner=ctx["owner"],
        unet_kwargs=forward_input.unet_kwargs,
        hook_context={
            "training_type": "sd15_lora",
            "global_step": 1,
            "micro_batch_index": 1,
            "micro_batch_count": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sync_gradients": True,
        },
        do_backward=True,
        should_try_cudagraph=False,
        multi_batch_execution_strategy=ctx["batch_contract_execution"].execution_strategy,
        multi_batch_execution_strategy_gate=ctx["batch_contract_execution"].execution_strategy_gate,
        logger=SimpleNamespace(warning=lambda *_: None, error=lambda *_: None),
    )
    ctx["forward_input"] = forward_input
    ctx["forward_execution"] = forward_execution
    return forward_execution.orchestrator_runtime


def _loss(ctx: dict[str, Any]) -> Any:
    loss_plan = run_lulynx_loss_plan_stage_handler(
        batch=ctx["batch"],
        model_arch="sd15",
        batch_size=1,
        loss_type="mse",
        uses_flow_matching=False,
        uses_sdxl_flow=False,
        sdxl_flow_sigmas_available=False,
        v_parameterization=False,
        masked_loss=False,
        alpha_mask=False,
        strict_masked_loss=False,
        debiased_estimation=False,
        snr_gamma=0.0,
        adaptive_loss_weighter_available=False,
        wavelet_loss_enabled=False,
        pattern_loss_enabled=False,
        prior_loss_weight=0.0,
        reg_dataloader_available=False,
        lulynx_wrapper_available=False,
        repa_active=False,
        dop_active=False,
        b_tier_runtime_available=False,
        do_backward=True,
        trace=ctx["trace"],
    )
    loss_execution = run_lulynx_loss_execution_stage_handler(
        owner=ctx["owner"],
        batch=ctx["batch"],
        prompt_embeds=ctx["transfer_execution"].prompt_embeds,
        noise_pred=ctx["forward_execution"].noise_pred,
        target=ctx["noise_execution"].target,
        timesteps=ctx["noise_execution"].timesteps,
        padding_mask=ctx["transfer_execution"].padding_mask,
        noisy_latents=ctx["noise_execution"].noisy_latents,
        uses_flow_matching=False,
        uses_sdxl_flow=False,
        do_backward=True,
        loss_scalars=_ScalarCache(),
        logger=SimpleNamespace(warning=lambda *_: None),
    )
    ctx["loss_plan"] = loss_plan
    ctx["loss_execution"] = loss_execution
    return loss_execution.orchestrator_runtime


def _backward(ctx: dict[str, Any]) -> Any:
    backward_plan = run_lulynx_backward_plan_stage_handler(
        do_backward=True,
        sync_gradients=True,
        accumulation_steps=1,
        uses_step_closure=False,
        uses_fused_backward=False,
        gradient_release_mode="",
        create_graph_backward=False,
        trace=ctx["trace"],
    )
    backward_execution = run_lulynx_backward_execution_stage_handler(
        loss=ctx["loss_execution"].loss,
        gradient_release_context=None,
        optimizer_used_fused_backward=False,
        optimizer_deferred_step_closure=False,
        create_graph_backward=False,
    )
    ctx["backward_plan"] = backward_plan
    ctx["backward_execution"] = backward_execution
    return backward_execution.orchestrator_runtime


def _optimizer_step(ctx: dict[str, Any]) -> Any:
    ctx["optimizer"].step()
    ctx["optimizer"].zero_grad(set_to_none=True)
    execution = run_lulynx_optimizer_execution_stage_handler(
        optimizer=ctx["optimizer"],
        trace=ctx["trace"],
        gradient_accumulation_steps=1,
        optimizer_step_executed=True,
        scheduler_step_executed=False,
        zero_grad_called=True,
        uses_step_closure=False,
        uses_fused_backward=False,
        native_update_runtime={},
    )
    ctx["optimizer_execution"] = execution
    return execution.orchestrator_runtime


def _telemetry(ctx: dict[str, Any]) -> Any:
    execution = run_lulynx_telemetry_execution_stage_handler(
        trace=ctx["trace"],
        step_info={"training_loop_runtime": {"step_phase_bubble_profile": {"kind": "toy"}}},
        step_wall_seconds=0.01,
    )
    ctx["telemetry_execution"] = execution
    return execution.orchestrator_runtime


def _ready_execution_gate() -> dict[str, Any]:
    return {
        "gate": "lulynx_training_pipeline_execution_readiness_v0",
        "status": "ready_for_behavior_equivalent_orchestrator_slice",
        "ready_for_behavior_equivalent_orchestrator_slice": True,
        "blockers": [],
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "stage_ids": list(result["stage_ids"]),
        "loss": result["loss"],
        "weight": result["weight"],
        "forward_route": result["forward_route"],
        "release_claim_allowed": result["release_claim_allowed"],
        "trace_stage_tail": list(result["compact_trace"].get("stage_ids", [])[-3:]),
    }


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-7


__all__ = ["LULYNX_BATCH1_HANDLER_PARITY_SMOKE", "run_lulynx_batch1_handler_parity_smoke"]
