"""Batch, transfer, and conditioning handlers for Lulynx train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .multi_batch_contract import inspect_batch_contract, recommend_execution_strategy
from .multi_batch_execution_strategy_gate import build_lulynx_multi_batch_execution_strategy_gate
from .training_data_pipeline_stage import observe_lulynx_data_pipeline_batch_collate
from .training_pipeline_trace import LulynxTrainingPipelineTrace
from .training_step_batch_stage import (
    LulynxTrainingStepBatchStagePlan,
    build_lulynx_training_step_batch_stage_plan,
)
from .training_step_conditioning_stage import (
    LulynxTrainingStepConditioningStagePlan,
    build_lulynx_training_step_conditioning_stage_plan,
)
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime
from .training_step_transfer_stage import (
    LulynxTrainingStepTransferStagePlan,
    build_lulynx_training_step_transfer_stage_plan,
)


@dataclass(frozen=True)
class LulynxBatchContractStageExecution:
    batch_stage_plan: LulynxTrainingStepBatchStagePlan
    training_data_pipeline_report: dict[str, Any]
    execution_strategy: dict[str, Any]
    execution_strategy_gate: dict[str, Any]
    orchestrator_runtime: dict[str, Any]

    @property
    def cached_native(self) -> bool:
        return bool(self.batch_stage_plan.cached_native)


@dataclass(frozen=True)
class LulynxTransferConditioningStageExecution:
    transfer_stage_plan: LulynxTrainingStepTransferStagePlan
    conditioning_stage_plan: LulynxTrainingStepConditioningStagePlan
    latents: Any
    prompt_embeds: dict[str, Any]
    padding_mask: Any
    orchestrator_runtime: dict[str, Any]


def run_lulynx_batch_contract_stage_handler(
    *,
    batch: Mapping[str, Any],
    model_arch: str,
    trace: LulynxTrainingPipelineTrace,
    previous_training_data_pipeline_report: Mapping[str, Any] | None = None,
    accumulation_steps: int = 1,
    do_backward: bool = True,
) -> LulynxBatchContractStageExecution:
    batch_stage_plan = build_lulynx_training_step_batch_stage_plan(batch=batch, model_arch=model_arch)
    required_fields = (
        ("latents", "encoder_hidden_states", "captions")
        if batch_stage_plan.cached_native
        else ("images", "captions")
    )
    data_pipeline_report = observe_lulynx_data_pipeline_batch_collate(
        previous_training_data_pipeline_report,
        batch=batch,
        expected_physical_batch_size=batch_stage_plan.expected_physical_batch_size,
        required_fields=required_fields,
    )
    batch_contract = inspect_batch_contract(
        batch,
        expected_physical_batch_size=batch_stage_plan.expected_physical_batch_size,
    )
    execution_strategy = recommend_execution_strategy(
        batch_contract=batch_contract,
    )
    execution_strategy_gate = build_lulynx_multi_batch_execution_strategy_gate(
        execution_strategy=execution_strategy,
        internal_gate_enabled=False,
        allow_diagnostic_strategy=False,
    )
    trace.start(
        batch=batch,
        expected_physical_batch_size=batch_stage_plan.expected_physical_batch_size,
    )
    trace.mark(
        "batch_contract",
        cached_native=bool(batch_stage_plan.cached_native),
        do_backward=bool(do_backward),
        accumulation_steps=int(accumulation_steps),
        batch_stage_plan=batch_stage_plan.as_dict(),
        multi_batch_execution_strategy=execution_strategy,
        multi_batch_execution_strategy_gate=execution_strategy_gate,
    )
    return LulynxBatchContractStageExecution(
        batch_stage_plan=batch_stage_plan,
        training_data_pipeline_report=data_pipeline_report,
        execution_strategy=execution_strategy,
        execution_strategy_gate=execution_strategy_gate,
        orchestrator_runtime=build_lulynx_batch_contract_orchestrator_runtime(
            batch_stage_plan=batch_stage_plan,
            execution_strategy=execution_strategy,
            execution_strategy_gate=execution_strategy_gate,
            accumulation_steps=accumulation_steps,
            do_backward=do_backward,
        ),
    )


def run_lulynx_transfer_conditioning_stage_handler(
    *,
    batch: Mapping[str, Any],
    batch_stage_plan: LulynxTrainingStepBatchStagePlan,
    model_arch: str,
    trace: LulynxTrainingPipelineTrace,
    target_dtype: Any,
    images: Any,
    captions: Any,
    qwen3_encoder_available: bool,
    do_backward: bool,
    te_dropout: float,
    clip_l_dropout_rate: float,
    clip_g_dropout_rate: float,
    t5_dropout_rate: float,
    profiled_to: Callable[..., Any],
    encode_latents_with_vae: Callable[[Any], Any],
    encode_prompt: Callable[[Any], dict[str, Any]],
    encode_qwen3: Callable[[Any], dict[str, Any]],
    apply_text_encoder_dropout: Callable[..., dict[str, Any]],
) -> LulynxTransferConditioningStageExecution:
    cached_native = bool(batch_stage_plan.cached_native)
    transfer_stage_plan = build_lulynx_training_step_transfer_stage_plan(
        batch=batch,
        route=batch_stage_plan.host_to_device_route,
        model_arch=model_arch,
        cached_native=cached_native,
        target_dtype=target_dtype,
    )
    trace.mark(
        "host_to_device",
        route=batch_stage_plan.host_to_device_route,
        transfer_stage_plan=transfer_stage_plan.as_dict(),
    )
    if cached_native:
        latents = profiled_to(batch["latents"], label="latents", dtype=target_dtype)
        padding_mask = batch.get("padding_mask")
        padding_mask = profiled_to(padding_mask, label="padding_mask", dtype=torch.bool) if isinstance(padding_mask, torch.Tensor) else None
        prompt_embeds = {
            "encoder_hidden_states": profiled_to(
                batch["encoder_hidden_states"],
                label="encoder_hidden_states",
                dtype=target_dtype,
            ),
        }
        if batch.get("pooled_prompt_embeds") is not None:
            prompt_embeds["pooled_prompt_embeds"] = profiled_to(batch["pooled_prompt_embeds"], label="pooled_prompt_embeds", dtype=target_dtype)
        if batch.get("attention_mask") is not None:
            prompt_embeds["attention_mask"] = profiled_to(batch["attention_mask"], label="attention_mask")
        if batch.get("qwen3_hidden_states") is not None:
            prompt_embeds["qwen3_hidden_states"] = profiled_to(batch["qwen3_hidden_states"], label="qwen3_hidden_states", dtype=target_dtype)
        if batch.get("qwen3_attention_mask") is not None:
            prompt_embeds["qwen3_attention_mask"] = profiled_to(batch["qwen3_attention_mask"], label="qwen3_attention_mask")
    else:
        padding_mask = None
        latents = encode_latents_with_vae(images)
        prompt_embeds = encode_prompt(captions)
        if str(model_arch or "").strip().lower() == "anima" and qwen3_encoder_available:
            prompt_embeds.update(encode_qwen3(captions))

    prompt_embeds = apply_text_encoder_dropout(prompt_embeds, do_backward=do_backward)
    conditioning_stage_plan = build_lulynx_training_step_conditioning_stage_plan(
        prompt_embeds=prompt_embeds,
        model_arch=model_arch,
        cached_native=cached_native,
        do_backward=do_backward,
        qwen3_encoder_available=qwen3_encoder_available,
        te_dropout=te_dropout,
        clip_l_dropout_rate=clip_l_dropout_rate,
        clip_g_dropout_rate=clip_g_dropout_rate,
        t5_dropout_rate=t5_dropout_rate,
    )
    trace.mark(
        "conditioning",
        has_pooled_prompt_embeds=bool(prompt_embeds.get("pooled_prompt_embeds") is not None),
        has_attention_mask=bool(prompt_embeds.get("attention_mask") is not None),
        conditioning_stage_plan=conditioning_stage_plan.as_dict(),
    )
    return LulynxTransferConditioningStageExecution(
        transfer_stage_plan=transfer_stage_plan,
        conditioning_stage_plan=conditioning_stage_plan,
        latents=latents,
        prompt_embeds=prompt_embeds,
        padding_mask=padding_mask,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning"),
            status="transfer_conditioning_stage_handlers_executed",
            handler_source="existing_training_loop_transfer_conditioning_path",
            stage_plans={
                "transfer_stage_plan": transfer_stage_plan.as_dict(),
                "conditioning_stage_plan": conditioning_stage_plan.as_dict(),
            },
        ),
    )


def build_lulynx_batch_contract_orchestrator_runtime(
    *,
    batch_stage_plan: LulynxTrainingStepBatchStagePlan,
    execution_strategy: Mapping[str, Any] | None = None,
    execution_strategy_gate: Mapping[str, Any] | None = None,
    accumulation_steps: int,
    do_backward: bool,
) -> dict[str, Any]:
    return build_lulynx_stage_orchestrator_runtime(
        executed_stage_ids=("batch_contract",),
        status="batch_contract_stage_handler_executed",
        handler_source="existing_training_loop_batch_contract_preamble",
        stage_plans={"batch_stage_plan": batch_stage_plan.as_dict()},
        extra={
            "batch_stage_plan": batch_stage_plan.as_dict(),
            "multi_batch_execution_strategy": dict(execution_strategy or {}),
            "multi_batch_execution_strategy_gate": dict(execution_strategy_gate or {}),
            "accumulation_steps": int(accumulation_steps),
            "do_backward": bool(do_backward),
        },
    )


__all__ = [
    "LulynxBatchContractStageExecution",
    "LulynxTransferConditioningStageExecution",
    "build_lulynx_batch_contract_orchestrator_runtime",
    "run_lulynx_batch_contract_stage_handler",
    "run_lulynx_transfer_conditioning_stage_handler",
]
