"""Canonical synthetic probe rows for the Newbie internal phase diagnosis."""

from __future__ import annotations

from typing import Any


def _probe(
    case_id: str,
    *,
    target_scope: str = "",
    data_wait_share: float = 0.013325,
    train_step_share: float = 0.95,
    forward_share: float = 0.35,
    backward_share: float = 0.60,
    substage: str = "",
    op_key: str = "",
    shape: str = "",
    module_group: str = "",
    triton_available: bool = False,
    triton_status: str = "",
    patched_lora_layers: int = 0,
) -> dict[str, Any]:
    phase_mean_ms = {
        "data_wait": round(data_wait_share * 1000.0, 4),
        "train_step_total": round(train_step_share * 1000.0, 4),
        "forward_total": round(forward_share * 1000.0, 4),
        "backward_total": round(backward_share * 1000.0, 4),
    }
    phase_share = {
        "data_wait": round(data_wait_share, 6),
        "train_step_total": round(train_step_share, 6),
        "forward_total": round(forward_share, 6),
        "backward_total": round(backward_share, 6),
    }
    if substage:
        phase_share[f"train_step_compute_substage.{substage}"] = 0.58 if substage == "newbie.backward_autograd_execution" else 0.35
        phase_mean_ms[f"train_step_compute_substage.{substage}"] = (
            phase_share[f"train_step_compute_substage.{substage}"] * 1000.0
        )
    train_step_breakdown = {
        "profile": "newbie_train_step_compute_substage_projection_v0",
        "source": "steady_bubble_profile.phase_mean_ms",
        "label_prefix": "train_step_compute_substage.",
        "available": bool(substage),
        "profiled_substage_count": 1 if substage else 0,
        "profiled_substage_labels": [substage] if substage else [],
        "profiled_substage_ms": (
            {substage: round(phase_share[f"train_step_compute_substage.{substage}"] * 1000.0, 4)}
            if substage
            else {}
        ),
        "profiled_substage_share": (
            {substage: round(phase_share[f"train_step_compute_substage.{substage}"], 6)}
            if substage
            else {}
        ),
        "profiled_substage_total_ms": (
            round(phase_share[f"train_step_compute_substage.{substage}"] * 1000.0, 4)
            if substage
            else 0.0
        ),
        "dominant_profiled_substage": substage,
        "dominant_profiled_substage_share": (
            round(phase_share[f"train_step_compute_substage.{substage}"], 6) if substage else 0.0
        ),
        "runtime_default_change": False,
    }
    backward_op_profile = {
        "profile": "newbie_backward_op_profile_projection_v0",
        "available": bool(op_key),
        "shape_profile_available": bool(shape),
        "sample_count": 1 if op_key else 0,
        "latest_status": "profiled" if op_key else "",
        "sort_key": "self_cuda_ms" if op_key else "",
        "top_op_key": op_key,
        "shape_group_count": 1 if shape else 0,
        "top_matmul_shape_key": "aten::mm" if shape else "",
        "top_matmul_shape": shape,
        "top_ops": (
            [
                {
                    "key": op_key,
                    "count": 1,
                    "self_cuda_ms": 1.0,
                    "cuda_ms": 1.0,
                    "self_cpu_ms": 1.0,
                    "cpu_ms": 1.0,
                }
            ]
            if op_key
            else []
        ),
        "top_matmul_shape_groups": (
            [
                {
                    "key": "aten::mm",
                    "input_shapes": shape,
                    "count": 1,
                    "self_cuda_ms": 1.0,
                    "cuda_ms": 1.0,
                    "self_cpu_ms": 1.0,
                    "cpu_ms": 1.0,
                }
            ]
            if shape
            else []
        ),
        "runtime_default_change": False,
    }
    module_timing_profile = {
        "profile": "newbie_module_timing_profile_projection_v0",
        "available": bool(module_group),
        "sample_count": 1 if module_group else 0,
        "latest_status": "profiled" if module_group else "",
        "tracked_module_count": 252 if module_group else 0,
        "group_count": 1 if module_group else 0,
        "top_group": module_group,
        "top_group_backward_cuda_ms": 1.0 if module_group else 0.0,
        "top_group_forward_cuda_ms": 1.0 if module_group else 0.0,
        "top_groups": (
            [
                {
                    "group": module_group,
                    "module_count": 80,
                    "forward_count": 80,
                    "backward_count": 80,
                    "forward_cuda_ms": 1.0,
                    "backward_cuda_ms": 1.0,
                    "forward_cpu_ms": 1.0,
                    "backward_cpu_ms": 1.0,
                    "module_name_examples": ["blocks.0.feed_forward.w1"],
                }
            ]
            if module_group
            else []
        ),
        "runtime_default_change": False,
    }
    triton_ops_runtime = {
        "profile": "triton_ops_runtime_projection_v0",
        "available": bool(triton_available),
        "enabled": triton_status == "patched" and bool(triton_available),
        "requested": bool(triton_available),
        "status": triton_status,
        "dtype": "bfloat16" if triton_status else "",
        "patched_lora_layers": patched_lora_layers,
        "patched_qkv_blocks": 0,
        "patched_adaln_blocks": 0,
        "inject_lora": bool(triton_status),
        "inject_qkv": False,
        "inject_adaln": False,
        "fp32_backward": False,
        "runtime_default_change": False,
    }
    return {
        "case_id": case_id,
        "family": "newbie",
        "summary_path": f"canonical::{case_id}::summary",
        "natural_evidence_path": f"canonical::{case_id}::natural_data_wait_evidence",
        "status": "no_natural_data_wait",
        "steps_completed": 48,
        "train_batch_size": 1,
        "native_cache_mode": "cache_first",
        "source_fixture": "heavy_raw_decode_mixed_sidecars_v0",
        "has_phase_profile": True,
        "dominant_bottleneck": "compute_bound",
        "data_wait_share": round(data_wait_share, 6),
        "data_wait_below_threshold": True,
        "mean_step_ms": 871.3,
        "steady_mean_step_ms": round((train_step_share + data_wait_share) * 1000.0, 4),
        "step_count": 12,
        "train_step_share": round(train_step_share, 6),
        "forward_total_share": round(forward_share, 6),
        "backward_total_share": round(backward_share, 6),
        "forward_backward_share": round(forward_share + backward_share, 6),
        "optimizer_share": 0.01,
        "h2d_transfer_share": 0.001,
        "host_gap_share": 0.001,
        "data_wait_mean_ms": round(data_wait_share * 1000.0, 4),
        "forward_total_mean_ms": round(forward_share * 1000.0, 4),
        "backward_total_mean_ms": round(backward_share * 1000.0, 4),
        "train_step_total_mean_ms": round(train_step_share * 1000.0, 4),
        "compute_dominates": True,
        "train_step_compute_substage_profile_available": bool(substage),
        "train_step_compute_breakdown": train_step_breakdown,
        "newbie_backward_op_profile_available": bool(op_key),
        "newbie_backward_op_profile": backward_op_profile,
        "newbie_module_timing_profile_available": bool(module_group),
        "newbie_module_timing_profile": module_timing_profile,
        "newbie_target_scope": target_scope,
        "newbie_target_module_count": 16 if target_scope == "tail8_attention" else (5 if target_scope == "balanced" else (2 if target_scope == "layer0_attention" else 0)),
        "newbie_injected_layer_count": 16 if target_scope == "tail8_attention" else (5 if target_scope == "balanced" else (2 if target_scope == "layer0_attention" else 0)),
        "triton_ops_runtime_available": bool(triton_available),
        "triton_ops_runtime": triton_ops_runtime,
        "dataloader_rebuild_observed": False,
        "natural_candidate": False,
        "top_phases": [
            {"label": "train_step_total", "mean_ms": round(train_step_share * 1000.0, 4), "share": round(train_step_share, 6)},
            {"label": "data_wait", "mean_ms": round(data_wait_share * 1000.0, 4), "share": round(data_wait_share, 6)},
        ],
    }


def canonical_newbie_internal_phase_probes() -> list[dict[str, Any]]:
    mm_shape = "[[16, 9216], [9216, 2304]]"
    return [
        _probe("newbie_cache_first_full_latent_batch1_checkpoint_off_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_lora_recompute_off_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_block_checkpoint_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_tread_keep50_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_diffcr_compress50_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_blockskip_skip25_compute_probe", target_scope="layer0_attention", substage="newbie.backward_autograd_execution", op_key="aten::mm", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_checkpoint_off_long_window_compute_probe", target_scope="layer0_attention", substage="newbie.forward_model_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_blockskip_skip25_long_window_compute_probe", target_scope="layer0_attention", substage="newbie.forward_model_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_checkpoint_off_long_window_seed2027_compute_probe", target_scope="layer0_attention", substage="newbie.forward_model_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_blockskip_skip25_long_window_seed2027_compute_probe", target_scope="layer0_attention", substage="newbie.forward_model_execution", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_balanced_targets_compute_probe", target_scope="balanced", substage="newbie.backward_autograd_execution", op_key="aten::mm", shape=mm_shape, triton_available=True, triton_status="patched", patched_lora_layers=1),
        _probe("newbie_cache_first_full_latent_batch1_backward_op_profile_probe", target_scope="unknown", substage="newbie.backward_autograd_execution", op_key="aten::mm", triton_available=True, triton_status="disabled"),
        _probe("newbie_cache_first_full_latent_batch1_backward_shape_profile_probe", target_scope="unknown", substage="newbie.backward_autograd_execution", op_key="aten::mm"),
        _probe("newbie_cache_first_full_latent_batch1_module_timing_profile_probe", target_scope="unknown", substage="newbie.backward_autograd_execution", module_group="newbie.ffn.expand_4x_linear"),
        _probe("newbie_cache_first_full_latent_batch1_triton_lora_compute_probe", target_scope="unknown"),
        _probe("newbie_cache_first_full_latent_batch1_tail8_attention_compute_probe", target_scope="unknown", substage="newbie.forward_model_execution"),
        _probe("newbie_cache_first_full_latent_batch1_tail8_attention_long_window_compute_probe", target_scope="unknown", substage="newbie.forward_model_execution"),
        _probe("newbie_cache_first_full_latent_batch1_tail8_attention_long_window_seed2027_compute_probe", target_scope="unknown", substage="newbie.forward_model_execution"),
        _probe("newbie_cache_first_full_latent_batch1_compile_per_block_compute_probe", target_scope="unknown"),
        _probe("newbie_cache_first_full_latent_batch1_tail8_forward_anomaly_probe", target_scope="unknown"),
    ]
