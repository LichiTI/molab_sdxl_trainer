"""Case definitions for bubble closed-loop matrix runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BubbleClosedLoopCase:
    case_id: str
    family: str
    description: str
    benchmark_args: tuple[str, ...]
    expected_evidence_statuses: tuple[str, ...] = ()
    source_fixture: str = ""
    source_fixture_samples: int = 4
    source_fixture_size: int = 4096
    source_fixture_seed: int = 1337
    source_fixture_source: str = ""
    source_fixture_sample_offset: int = 0
    build_natural_data_wait_evidence: bool = False
    expected_natural_data_wait_statuses: tuple[str, ...] = ()


CLOSED_LOOP_PROBE_STATUSES = ("keep_observed", "rollback_observed", "cooldown", "no_action")
NATURAL_DATA_WAIT_PROBE_STATUSES = ("natural_dataloader_rebuild_observed", "natural_data_wait_candidate", "no_natural_data_wait")


def _base_args(
    family: str,
    *,
    steps: int,
    warmup: int,
    tune_interval: int,
    samples: int,
    resolution: int,
    batch: int,
    workers: int,
    seed: int = 1337,
    checkpoint_policy: str = "",
    dataloader_prefetch_factor: int = 2,
    pin_memory: bool = True,
    phase_profile: bool = True,
    data_transfer_profile: bool = True,
    data_transfer_profile_mode: str = "sync",
    allow_dataloader_rebuild_current_run: bool = False,
    controlled_data_wait_share: float = 0.0,
    controlled_data_wait_mean_ms: float = 100.0,
    benchmark_data_wait_stall_ms: float = 0.0,
    benchmark_data_wait_direct_action: bool = False,
    max_actions_per_run: int = 1,
    controlled_rollback_slowdown_ratio: float = 1.0,
    controlled_rollback_after_apply_steps: int = 1,
    native_cache_mode: str = "",
    newbie_latent_crop_size: int | None = None,
    newbie_block_checkpointing: bool = False,
    newbie_block_checkpointing_mode: str = "block",
    lora_activation_recompute: str = "",
    torch_compile: bool = False,
    torch_compile_scope: str = "per_block",
    torch_compile_mode: str = "default",
    newbie_disable_safe_fallback: bool = False,
    newbie_backward_op_profile: bool = False,
    newbie_backward_op_profile_top_k: int = 12,
    newbie_backward_op_profile_max_samples: int = 1,
    newbie_backward_op_profile_shapes: bool = False,
    newbie_module_timing_profile: bool = False,
    newbie_module_timing_profile_top_k: int = 12,
    newbie_module_timing_profile_max_samples: int = 1,
    newbie_target_scope: str = "",
    dit_compute_reducer_strategy: str = "",
    dit_compute_reducer_keep_ratio: float = 1.0,
    dit_compute_reducer_min_keep_tokens: int = 1,
    dit_compute_reducer_compression_ratio: float = 1.0,
    dit_compute_reducer_min_tokens: int = 1,
    dit_compute_reducer_skip_ratio: float = 0.0,
    dit_compute_reducer_skip_every: int = 0,
    dit_compute_reducer_warmup_steps: int = 0,
    dit_compute_reducer_min_block: int = 0,
    dit_compute_reducer_score_mode: str = "l2",
    triton_ops: bool = False,
    triton_ops_inject_lora: bool = True,
    triton_ops_inject_qkv: bool = True,
    triton_ops_inject_adaln: bool = True,
    triton_ops_fp32_backward: bool = False,
) -> tuple[str, ...]:
    args = [
        "--family",
        family,
        "--profiles",
        "standard",
        "--steps",
        str(steps),
        "--steady-warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--resolution",
        str(resolution),
        "--network-dim",
        "1",
        "--train-batch-size",
        str(batch),
        "--dataloader-workers",
        str(workers),
        "--dataloader-prefetch-factor",
        str(max(int(dataloader_prefetch_factor), 1)),
        "--bubble-controller-enabled",
        "--bubble-controller-mode",
        "auto_apply",
        "--bubble-controller-warmup-steps",
        str(warmup),
        "--bubble-controller-tune-interval-steps",
        str(tune_interval),
        "--bubble-controller-max-actions-per-run",
        str(max(int(max_actions_per_run), 0)),
        "--bubble-controller-min-throughput-gain",
        "0.0",
    ]
    if int(seed) != 1337:
        args.extend(["--seed", str(max(int(seed), 0))])
    checkpoint_policy_value = str(checkpoint_policy or "").strip().lower().replace("-", "_")
    if checkpoint_policy_value in {"auto", "off", "full", "offloaded", "selective"}:
        args.extend(["--checkpoint-policy", checkpoint_policy_value])
    if phase_profile:
        args.append("--phase-profile")
    if not pin_memory:
        args.append("--no-pin-memory")
    if data_transfer_profile:
        args.extend(
            [
                "--data-transfer-profile",
                "--data-transfer-profile-mode",
                str(data_transfer_profile_mode or "event"),
            ]
        )
    if allow_dataloader_rebuild_current_run:
        args.append("--bubble-controller-allow-dataloader-rebuild-current-run")
    if native_cache_mode:
        args.extend(["--native-cache-mode", str(native_cache_mode)])
    if family == "newbie" and newbie_latent_crop_size is not None:
        args.extend(["--newbie-latent-crop-size", str(max(int(newbie_latent_crop_size), 0))])
    if family == "newbie" and bool(newbie_block_checkpointing):
        args.append("--newbie-block-checkpointing")
        checkpoint_mode = str(newbie_block_checkpointing_mode or "block").strip().lower().replace("-", "_")
        if checkpoint_mode != "block":
            checkpoint_mode = "block"
        args.extend(["--newbie-block-checkpointing-mode", checkpoint_mode])
    if family == "newbie" and bool(newbie_disable_safe_fallback):
        args.append("--newbie-disable-safe-fallback")
    if family == "newbie" and bool(newbie_backward_op_profile):
        args.extend(
            [
                "--newbie-backward-op-profile",
                "--newbie-backward-op-profile-top-k",
                str(max(int(newbie_backward_op_profile_top_k), 1)),
                "--newbie-backward-op-profile-max-samples",
                str(max(int(newbie_backward_op_profile_max_samples), 1)),
            ]
        )
        if bool(newbie_backward_op_profile_shapes):
            args.append("--newbie-backward-op-profile-shapes")
    if family == "newbie" and bool(newbie_module_timing_profile):
        args.extend(
            [
                "--newbie-module-timing-profile",
                "--newbie-module-timing-profile-top-k",
                str(max(int(newbie_module_timing_profile_top_k), 1)),
                "--newbie-module-timing-profile-max-samples",
                str(max(int(newbie_module_timing_profile_max_samples), 1)),
            ]
        )
    if family == "newbie" and str(newbie_target_scope or "").strip():
        args.extend(["--newbie-target-scope", str(newbie_target_scope).strip()])
    reducer_strategy = str(dit_compute_reducer_strategy or "").strip().lower().replace("-", "")
    if family == "newbie" and reducer_strategy in {"tread", "diffcr", "blockskip"}:
        args.extend(["--dit-compute-reducer-strategy", reducer_strategy])
        if reducer_strategy == "tread":
            args.extend(
                [
                    "--dit-compute-reducer-keep-ratio",
                    str(min(max(float(dit_compute_reducer_keep_ratio), 0.0), 1.0)),
                    "--dit-compute-reducer-min-keep-tokens",
                    str(max(int(dit_compute_reducer_min_keep_tokens), 1)),
                ]
            )
        elif reducer_strategy == "diffcr":
            args.extend(
                [
                    "--dit-compute-reducer-compression-ratio",
                    str(min(max(float(dit_compute_reducer_compression_ratio), 0.0), 1.0)),
                    "--dit-compute-reducer-min-tokens",
                    str(max(int(dit_compute_reducer_min_tokens), 1)),
                ]
            )
        elif reducer_strategy == "blockskip":
            args.extend(
                [
                    "--dit-compute-reducer-skip-ratio",
                    str(min(max(float(dit_compute_reducer_skip_ratio), 0.0), 0.95)),
                    "--dit-compute-reducer-skip-every",
                    str(max(int(dit_compute_reducer_skip_every), 0)),
                    "--dit-compute-reducer-warmup-steps",
                    str(max(int(dit_compute_reducer_warmup_steps), 0)),
                    "--dit-compute-reducer-min-block",
                    str(max(int(dit_compute_reducer_min_block), 0)),
                ]
            )
        args.extend(["--dit-compute-reducer-score-mode", str(dit_compute_reducer_score_mode or "l2")])
    if bool(triton_ops):
        args.append("--triton-ops")
        if not bool(triton_ops_inject_lora):
            args.append("--triton-ops-no-lora")
        if not bool(triton_ops_inject_qkv):
            args.append("--triton-ops-no-qkv")
        if not bool(triton_ops_inject_adaln):
            args.append("--triton-ops-no-adaln")
        if bool(triton_ops_fp32_backward):
            args.append("--triton-ops-fp32-backward")
    recompute_mode = str(lora_activation_recompute or "").strip().lower()
    if recompute_mode in {"auto", "on", "off"}:
        args.extend(["--lora-activation-recompute", recompute_mode])
    if torch_compile:
        compile_scope = str(torch_compile_scope or "per_block").strip().lower().replace("-", "_")
        if compile_scope not in {"per_block", "full", "full_core"}:
            compile_scope = "per_block"
        compile_mode = str(torch_compile_mode or "default").strip().lower()
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            compile_mode = "default"
        args.extend(
            [
                "--torch-compile",
                "--torch-compile-scope",
                compile_scope,
                "--torch-compile-mode",
                compile_mode,
            ]
        )
    if float(controlled_data_wait_share) > 0.0:
        args.extend(
            [
                "--bubble-controller-controlled-data-wait-share",
                str(controlled_data_wait_share),
                "--bubble-controller-controlled-data-wait-mean-ms",
                str(max(float(controlled_data_wait_mean_ms), 1.0)),
            ]
        )
    if float(benchmark_data_wait_stall_ms) > 0.0:
        args.extend(
            [
                "--bubble-controller-benchmark-data-wait-stall-ms",
                str(benchmark_data_wait_stall_ms),
            ]
        )
        if benchmark_data_wait_direct_action:
            args.append("--bubble-controller-benchmark-data-wait-direct-action")
    if float(controlled_rollback_slowdown_ratio) > 1.0:
        args.extend(
            [
                "--bubble-controller-controlled-rollback-slowdown-ratio",
                str(controlled_rollback_slowdown_ratio),
                "--bubble-controller-controlled-rollback-after-apply-steps",
                str(max(int(controlled_rollback_after_apply_steps), 0)),
            ]
        )
    return tuple(args)


def _sdxl_heavy_raw_case(
    case_id: str,
    description: str,
    *,
    steps: int = 16,
    warmup: int = 2,
    tune_interval: int = 3,
    samples: int = 4,
    resolution: int = 128,
    workers: int = 0,
    dataloader_prefetch_factor: int = 2,
    pin_memory: bool = True,
    source_fixture: str = "heavy_raw_decode_png_v0",
    source_fixture_samples: int = 4,
    source_fixture_size: int = 4096,
    strict_evidence: bool = False,
) -> BubbleClosedLoopCase:
    return BubbleClosedLoopCase(
        case_id=case_id,
        family="sdxl",
        description=description,
        expected_evidence_statuses=(
            ("keep_observed", "rollback_observed")
            if strict_evidence
            else CLOSED_LOOP_PROBE_STATUSES
        ),
        source_fixture=source_fixture,
        source_fixture_samples=source_fixture_samples,
        source_fixture_size=source_fixture_size,
        build_natural_data_wait_evidence=True,
        expected_natural_data_wait_statuses=(
            ("natural_dataloader_rebuild_observed", "natural_data_wait_candidate")
            if strict_evidence
            else NATURAL_DATA_WAIT_PROBE_STATUSES
        ),
        benchmark_args=_base_args(
            "sdxl",
            steps=steps,
            warmup=warmup,
            tune_interval=tune_interval,
            samples=samples,
            resolution=resolution,
            batch=1,
            workers=workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            pin_memory=pin_memory,
            phase_profile=True,
            data_transfer_profile=True,
            data_transfer_profile_mode="event",
            allow_dataloader_rebuild_current_run=True,
            max_actions_per_run=2,
        ),
    )


def _sdxl_heavy_raw_cases() -> list[BubbleClosedLoopCase]:
    mixed_fixture = "heavy_raw_decode_mixed_sidecars_v0"
    return [
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_natural_data_wait_closed_loop",
            "SDXL non-injected heavy raw PNG decode loop with mixed caption sidecars.",
            source_fixture=mixed_fixture,
            strict_evidence=True,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_pin_off_natural_data_wait_probe",
            "SDXL heavy raw decode pin_memory axis probe with the same non-injected fixture.",
            source_fixture=mixed_fixture,
            pin_memory=False,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_resolution512_natural_data_wait_probe",
            "SDXL heavy raw decode resolution axis probe at 512 training resolution.",
            resolution=512,
            steps=12,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_workers2_prefetch4_natural_data_wait_probe",
            "SDXL heavy raw decode workers2/prefetch4 next-run prewarm probe.",
            source_fixture=mixed_fixture,
            workers=2,
            dataloader_prefetch_factor=4,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_long_window_natural_data_wait_probe",
            "SDXL heavy raw decode longer-window probe with more source samples and steps.",
            steps=32,
            warmup=4,
            tune_interval=6,
            samples=8,
            source_fixture_samples=8,
            source_fixture_size=6144,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_heavy_raw_decode_mixed_sidecars_long_window_natural_data_wait_probe",
            "SDXL heavy raw decode mixed-sidecar longer-window probe with more source samples and steps.",
            steps=32,
            warmup=4,
            tune_interval=6,
            samples=8,
            source_fixture=mixed_fixture,
            source_fixture_samples=8,
            source_fixture_size=6144,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_lora1024_heavy_raw_decode_natural_data_wait_probe",
            "SDXL LoRA 1024 heavy raw decode natural data-wait probe with mixed sidecars.",
            steps=8,
            warmup=2,
            tune_interval=3,
            resolution=1024,
            source_fixture=mixed_fixture,
            source_fixture_size=4096,
        ),
        _sdxl_heavy_raw_case(
            "sdxl_lora1024_heavy_raw_workers2_prefetch4_natural_data_wait_probe",
            "SDXL LoRA 1024 heavy raw decode workers2/prefetch4 next-run data-wait probe.",
            steps=8,
            warmup=2,
            tune_interval=3,
            resolution=1024,
            workers=2,
            dataloader_prefetch_factor=4,
            source_fixture=mixed_fixture,
            source_fixture_size=4096,
        ),
    ]


def _cache_miss_case(
    case_id: str,
    family: str,
    description: str,
    *,
    native_cache_mode: str,
    steps: int,
    warmup: int,
    tune_interval: int,
    samples: int,
    resolution: int,
    batch: int,
    workers: int = 0,
    source_fixture_samples: int = 4,
    source_fixture_size: int = 1024,
) -> BubbleClosedLoopCase:
    return BubbleClosedLoopCase(
        case_id=case_id,
        family=family,
        description=description,
        expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
        source_fixture="heavy_raw_decode_cache_miss_mixed_sidecars_v0",
        source_fixture_samples=source_fixture_samples,
        source_fixture_size=source_fixture_size,
        build_natural_data_wait_evidence=True,
        expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
        benchmark_args=_base_args(
            family,
            steps=steps,
            warmup=warmup,
            tune_interval=tune_interval,
            samples=samples,
            resolution=resolution,
            batch=batch,
            workers=workers,
            phase_profile=True,
            data_transfer_profile=True,
            data_transfer_profile_mode="event",
            allow_dataloader_rebuild_current_run=True,
            max_actions_per_run=2,
            native_cache_mode=native_cache_mode,
        ),
    )


def _cache_miss_guard_cases() -> list[BubbleClosedLoopCase]:
    return [
        _cache_miss_case(
            "anima_online_cache_miss_natural_data_wait_probe",
            "anima",
            "Anima online_cache missing-cache natural data-wait guard probe.",
            native_cache_mode="online_cache",
            steps=24,
            warmup=4,
            tune_interval=8,
            samples=8,
            resolution=64,
            batch=8,
            source_fixture_samples=8,
            source_fixture_size=1024,
        ),
        _cache_miss_case(
            "anima_online_cache_miss_long_window_guard_probe",
            "anima",
            "Anima online_cache longer-window guard probe for cold cache miss transients.",
            native_cache_mode="online_cache",
            steps=48,
            warmup=8,
            tune_interval=12,
            samples=16,
            resolution=64,
            batch=8,
            source_fixture_samples=16,
            source_fixture_size=1024,
        ),
        _cache_miss_case(
            "newbie_cache_miss_prepare_guard_probe",
            "newbie",
            "Newbie rebuild_cache guard probe: cache building is pre-step and must not become a data_wait claim.",
            native_cache_mode="rebuild_cache",
            steps=8,
            warmup=2,
            tune_interval=4,
            samples=8,
            resolution=64,
            batch=4,
            source_fixture_samples=8,
            source_fixture_size=1024,
        ),
    ]


def _newbie_heavy_raw_cases() -> list[BubbleClosedLoopCase]:
    return [
        BubbleClosedLoopCase(
            case_id="newbie_heavy_raw_decode_mixed_sidecars_long_window_natural_data_wait_probe",
            family="newbie",
            description="Newbie non-cache heavy raw decode mixed-sidecar longer-window natural data-wait probe.",
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=32,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=4,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_heavy_raw_decode_mixed_sidecars_batch1_long_queue_natural_data_wait_probe",
            family="newbie",
            description=(
                "Newbie non-cache heavy raw decode mixed-sidecar batch1 probe; "
                "keeps fixture size stable while increasing observable DataLoader batches."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=6,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_long_queue_natural_data_wait_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 probe; stresses the cached DataLoader "
                "payload instead of relying on pre-step raw PNG cache materialization."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=6,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_lora_recompute_off_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 compute probe with LoRA activation "
                "recompute forced off; checks whether backward autograd is dominated by recompute."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=6,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                lora_activation_recompute="off",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_block_checkpoint_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 compute probe with existing native "
                "DiT block checkpointing enabled; validates whether recompute changes the "
                "FFN/backward-bound gate without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_block_checkpointing=True,
                newbie_block_checkpointing_mode="block",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_checkpoint_off_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 paired compute baseline with "
                "checkpoint_policy=off and no reducer; isolates checkpoint-off effects "
                "from TREAD token-routing effects."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_tread_keep50_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that enables "
                "the default-off DiT compute reducer seam with TREAD keep_ratio=0.5; "
                "checks whether token routing can reduce the FFN/backward-bound gate "
                "without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="tread",
                dit_compute_reducer_keep_ratio=0.5,
                dit_compute_reducer_min_keep_tokens=4,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_diffcr_compress50_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that enables "
                "the default-off DiT compute reducer seam with DiffCR compression_ratio=0.5; "
                "compares token compression against the checkpoint-off/no-reducer baseline "
                "without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="diffcr",
                dit_compute_reducer_compression_ratio=0.5,
                dit_compute_reducer_min_tokens=4,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_skip25_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that enables "
                "the default-off DiT compute reducer seam with BlockSkip skip_ratio=0.25; "
                "checks whether sparse block execution can reduce the base DiT FFN/attention "
                "backward-bound gate without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.25,
                dit_compute_reducer_skip_every=4,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_checkpoint_off_long_window_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window paired compute baseline "
                "with checkpoint_policy=off and no reducer; keeps the BlockSkip follow-up "
                "comparison on the same source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_skip25_long_window_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window protected compute probe "
                "that repeats BlockSkip skip_ratio=0.25 against a longer checkpoint-off "
                "baseline without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.25,
                dit_compute_reducer_skip_every=4,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_checkpoint_off_long_window_seed2027_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window seed2027 paired compute "
                "baseline with checkpoint_policy=off and no reducer; starts BlockSkip "
                "multi-seed stability evidence without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                seed=2027,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_skip25_long_window_seed2027_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window seed2027 protected "
                "compute probe that repeats BlockSkip skip_ratio=0.25 for multi-seed "
                "stability review without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                seed=2027,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.25,
                dit_compute_reducer_skip_every=4,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_skip10_long_window_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window protected compute probe "
                "that uses a milder BlockSkip skip_ratio=0.10 to test whether the loss gate "
                "can improve while preserving some checkpoint-off throughput gain."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.10,
                dit_compute_reducer_skip_every=10,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_skip10_long_window_seed2027_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window seed2027 protected "
                "compute probe that repeats the milder BlockSkip skip_ratio=0.10 quality "
                "gate candidate without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                seed=2027,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.10,
                dit_compute_reducer_skip_every=10,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_late24_skip25_long_window_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window protected compute probe "
                "that keeps BlockSkip skip_ratio=0.25 but limits skipping to late blocks "
                "with min_block=24 to test a more quality-preserving schedule."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.25,
                dit_compute_reducer_skip_every=4,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=24,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_blockskip_late24_skip25_long_window_seed2027_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window seed2027 protected "
                "compute probe that repeats the late-block-only BlockSkip skip_ratio=0.25 "
                "candidate without changing source/cache/target scope."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                seed=2027,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                dit_compute_reducer_strategy="blockskip",
                dit_compute_reducer_skip_ratio=0.25,
                dit_compute_reducer_skip_every=4,
                dit_compute_reducer_warmup_steps=4,
                dit_compute_reducer_min_block=24,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_compile_per_block_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 compute probe with torch.compile "
                "per-block enabled through the normal trainer runtime contract."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=6,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                torch_compile=True,
                torch_compile_scope="per_block",
                torch_compile_mode="default",
                newbie_disable_safe_fallback=True,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_backward_op_profile_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 diagnostic probe with one torch.profiler "
                "sample around loss.backward() to identify dominant autograd ops."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_backward_op_profile=True,
                newbie_backward_op_profile_top_k=12,
                newbie_backward_op_profile_max_samples=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_triton_lora_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that enables only "
                "the existing Triton standard-LoRA fused path, while also sampling one backward op profile."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_backward_op_profile=True,
                newbie_backward_op_profile_top_k=12,
                newbie_backward_op_profile_max_samples=1,
                triton_ops=True,
                triton_ops_inject_lora=True,
                triton_ops_inject_qkv=False,
                triton_ops_inject_adaln=False,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_balanced_targets_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that expands "
                "LoRA targets from the historical layer0 attention axis to the balanced "
                "Newbie target preset, while sampling one backward op profile."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_backward_op_profile=True,
                newbie_backward_op_profile_top_k=12,
                newbie_backward_op_profile_max_samples=1,
                newbie_target_scope="balanced",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_tail8_attention_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 protected compute probe that moves "
                "LoRA targets from the historical layer0 attention axis to the last 8 "
                "attention blocks. This tests whether shortening frozen-backbone autograd "
                "depth reduces the FFN/backward matmul bottleneck without using BlockSkip."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_backward_op_profile=True,
                newbie_backward_op_profile_top_k=12,
                newbie_backward_op_profile_max_samples=1,
                newbie_module_timing_profile=True,
                newbie_module_timing_profile_top_k=12,
                newbie_module_timing_profile_max_samples=1,
                newbie_target_scope="tail8_attention",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_tail8_attention_long_window_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window protected compute probe "
                "that keeps checkpoint_policy=off and moves LoRA targets to the last 8 "
                "attention blocks without profiler overhead, pairing against the existing "
                "checkpoint-off long-window baseline."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_target_scope="tail8_attention",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_tail8_attention_long_window_seed2027_compute_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 longer-window seed2027 protected "
                "compute probe that repeats the tail8 attention target-depth candidate "
                "without profiler overhead, reducer, Triton, compile, or target-scope "
                "changes beyond the late attention target placement."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=48,
                warmup=8,
                tune_interval=8,
                samples=8,
                resolution=64,
                batch=1,
                seed=2027,
                workers=0,
                checkpoint_policy="off",
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_target_scope="tail8_attention",
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_backward_shape_profile_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 diagnostic probe that records compact "
                "input-shape groups for backward matmul attribution."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_backward_op_profile=True,
                newbie_backward_op_profile_top_k=12,
                newbie_backward_op_profile_max_samples=1,
                newbie_backward_op_profile_shapes=True,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="newbie_cache_first_full_latent_batch1_module_timing_profile_probe",
            family="newbie",
            description=(
                "Newbie cache-first full-latent batch1 diagnostic probe that attaches temporary "
                "module hooks to attribute FFN/attention forward and backward timing groups."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            source_fixture="heavy_raw_decode_mixed_sidecars_v0",
            source_fixture_samples=8,
            source_fixture_size=6144,
            build_natural_data_wait_evidence=True,
            expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
            benchmark_args=_base_args(
                "newbie",
                steps=24,
                warmup=4,
                tune_interval=6,
                samples=8,
                resolution=64,
                batch=1,
                workers=0,
                dataloader_prefetch_factor=2,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                max_actions_per_run=2,
                native_cache_mode="cache_first",
                newbie_latent_crop_size=0,
                newbie_module_timing_profile=True,
                newbie_module_timing_profile_top_k=12,
                newbie_module_timing_profile_max_samples=1,
            ),
        ),
    ]


def _real_material_base_args(
    family: str,
    *,
    workers: int,
    prefetch: int,
    spec_override: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    specs = {
        "sd15": {"steps": 8, "warmup": 2, "tune_interval": 3, "samples": 8, "resolution": 512, "batch": 1},
        "sdxl": {"steps": 8, "warmup": 2, "tune_interval": 3, "samples": 8, "resolution": 1024, "batch": 1},
        "anima": {"steps": 24, "warmup": 4, "tune_interval": 8, "samples": 8, "resolution": 64, "batch": 8},
        "newbie": {"steps": 8, "warmup": 2, "tune_interval": 4, "samples": 8, "resolution": 64, "batch": 4},
    }
    spec = specs.get(family)
    if spec is None:
        raise ValueError(f"unsupported real material canary family: {family!r}")
    if spec_override:
        spec = {**spec}
        for key in ("steps", "warmup", "tune_interval", "samples", "resolution", "batch"):
            if key in spec_override and spec_override[key] is not None:
                spec[key] = int(spec_override[key])
    return _base_args(
        family,
        **spec,
        workers=workers,
        dataloader_prefetch_factor=prefetch,
        data_transfer_profile_mode="event",
        allow_dataloader_rebuild_current_run=True,
        max_actions_per_run=2,
        native_cache_mode="cache_first",
    )


def build_bubble_real_material_canary_cases(
    *,
    source_data: str = "sucai/6_lulu",
    families: Sequence[str] = ("sdxl", "anima", "newbie"),
    sample_offset: int = 0,
    family_specs: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, BubbleClosedLoopCase]:
    """Return warm-cache real-material canary pairs for P60."""

    cases: list[BubbleClosedLoopCase] = []
    for family in families:
        normalized = str(family or "").strip().lower().replace("-", "_")
        if normalized == "dit":
            normalized = "newbie"
        spec_override = (family_specs or {}).get(normalized)
        fixture_samples = max(int((spec_override or {}).get("samples", 8)), 1)
        label_family = "SD15 LoRA 512" if normalized == "sd15" else normalized.upper()
        for suffix, workers, prefetch in (
            ("workers0", 0, 2),
            ("workers2_prefetch4", 2, 4),
        ):
            case_prefix = "sd15_lora512" if normalized == "sd15" else normalized
            cases.append(
                BubbleClosedLoopCase(
                    case_id=f"{case_prefix}_real_material_cache_first_{suffix}_canary",
                    family=normalized,
                    description=(
                        f"{label_family} real-material warm-cache canary "
                        f"with DataLoader workers={workers}, prefetch={prefetch}."
                    ),
                    expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
                    source_fixture="real_material_canary_v0",
                    source_fixture_samples=fixture_samples,
                    source_fixture_size=0,
                    source_fixture_source=str(source_data or "sucai/6_lulu"),
                    source_fixture_sample_offset=max(int(sample_offset), 0),
                    build_natural_data_wait_evidence=True,
                    expected_natural_data_wait_statuses=NATURAL_DATA_WAIT_PROBE_STATUSES,
                    benchmark_args=_real_material_base_args(
                        normalized,
                        workers=workers,
                        prefetch=prefetch,
                        spec_override=spec_override,
                    ),
                )
            )
    return {case.case_id: case for case in cases}


def _anima_saturation_boundary_case() -> BubbleClosedLoopCase:
    return BubbleClosedLoopCase(
        case_id="anima_saturation_boundary_cache_first_compute_case",
        family="anima",
        description=(
            "Anima saturation boundary cache-first compute case with a longer steady window; "
            "uses real-material warm cache and the protected closed-loop runner."
        ),
        expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
        source_fixture="real_material_canary_v0",
        source_fixture_samples=16,
        source_fixture_size=0,
        source_fixture_source="sucai/6_lulu",
        benchmark_args=_base_args(
            "anima",
            steps=64,
            warmup=8,
            tune_interval=16,
            samples=16,
            resolution=64,
            batch=32,
            workers=2,
            dataloader_prefetch_factor=4,
            phase_profile=True,
            data_transfer_profile=True,
            data_transfer_profile_mode="event",
            allow_dataloader_rebuild_current_run=True,
            max_actions_per_run=2,
            native_cache_mode="cache_first",
        ),
    )


def default_bubble_closed_loop_cases() -> dict[str, BubbleClosedLoopCase]:
    """Return conservative real-training closed-loop cases."""

    cases = [
        _anima_saturation_boundary_case(),
        BubbleClosedLoopCase(
            case_id="anima_sync_profiler_closed_loop_smoke",
            family="anima",
            description="Anima cache-first P7 host-scheduling loop: disable sync profiling, cooldown, evaluate.",
            expected_evidence_statuses=("keep_observed",),
            benchmark_args=_base_args(
                "anima",
                steps=24,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=2,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="anima_sync_profiler_closed_loop_rollback_control",
            family="anima",
            description=(
                "Anima P7 controlled rollback loop: benchmark-only post-apply slowdown exercises rollback "
                "and manifest evidence without changing product defaults."
            ),
            expected_evidence_statuses=("rollback_observed",),
            benchmark_args=_base_args(
                "anima",
                steps=24,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=2,
                controlled_rollback_slowdown_ratio=3.0,
                controlled_rollback_after_apply_steps=1,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="anima_dataloader_rebuild_current_run_gate_probe",
            family="anima",
            description=(
                "Anima DataLoader epoch-boundary rebuild gate probe: passes the explicit current-run gate "
                "and leaves sync profiling off so data-bound evidence can drive the action when present."
            ),
            expected_evidence_statuses=CLOSED_LOOP_PROBE_STATUSES,
            benchmark_args=_base_args(
                "anima",
                steps=24,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=0,
                phase_profile=False,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="anima_dataloader_rebuild_closed_loop_controlled_data_wait",
            family="anima",
            description=(
                "Anima controlled data-wait closed loop: benchmark-only data_wait evidence drives "
                "DataLoader epoch-boundary rebuild through the real trainer apply/evaluate path."
            ),
            expected_evidence_statuses=("keep_observed", "rollback_observed", "cooldown"),
            benchmark_args=_base_args(
                "anima",
                steps=24,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=0,
                phase_profile=False,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                controlled_data_wait_share=0.14,
                controlled_data_wait_mean_ms=100.0,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="anima_dataloader_rebuild_closed_loop_real_data_wait_stall",
            family="anima",
            description=(
                "Anima real data-wait stall loop: benchmark-only Dataset item sleep is measured by "
                "step_phase_profile and drives DataLoader epoch-boundary rebuild through real profiler evidence."
            ),
            expected_evidence_statuses=("keep_observed", "rollback_observed", "cooldown"),
            benchmark_args=_base_args(
                "anima",
                steps=24,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=0,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                benchmark_data_wait_stall_ms=30.0,
                benchmark_data_wait_direct_action=True,
            ),
        ),
        BubbleClosedLoopCase(
            case_id="anima_dataloader_rebuild_closed_loop_profiler_handoff_stall",
            family="anima",
            description=(
                "Anima profiler handoff loop: real benchmark data_wait stall first closes sync profiling, "
                "then reuses the measured data_wait handoff to arm DataLoader epoch-boundary rebuild."
            ),
            expected_evidence_statuses=("keep_observed", "rollback_observed", "cooldown"),
            benchmark_args=_base_args(
                "anima",
                steps=32,
                warmup=4,
                tune_interval=8,
                samples=32,
                resolution=64,
                batch=16,
                workers=0,
                phase_profile=True,
                data_transfer_profile=True,
                data_transfer_profile_mode="event",
                allow_dataloader_rebuild_current_run=True,
                benchmark_data_wait_stall_ms=30.0,
                max_actions_per_run=2,
                controlled_rollback_after_apply_steps=1,
            ),
        ),
        *_sdxl_heavy_raw_cases(),
        *_newbie_heavy_raw_cases(),
        *_cache_miss_guard_cases(),
    ]
    return {case.case_id: case for case in cases}


__all__ = [
    "BubbleClosedLoopCase",
    "CLOSED_LOOP_PROBE_STATUSES",
    "NATURAL_DATA_WAIT_PROBE_STATUSES",
    "build_bubble_real_material_canary_cases",
    "default_bubble_closed_loop_cases",
]
