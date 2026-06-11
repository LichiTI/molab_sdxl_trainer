"""Mini training-step A/B probe for the experimental LXCS replacement loader."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import shutil
import subprocess
import threading
import time
from typing import Any

try:
    from .lossless_cache_replacement_policy import (
        LosslessCacheReplacementPolicyConfig,
        evaluate_lossless_cache_replacement_policy,
    )
    from .lossless_cache_replacement_loader import (
        LosslessCacheReplacementLoaderConfig,
        iter_newbie_lossless_cache_replacement_batches,
    )
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
    from .newbie_cached_dataset import NewbieCachedDataset, create_newbie_cached_dataloader
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_replacement_policy import (
        LosslessCacheReplacementPolicyConfig,
        evaluate_lossless_cache_replacement_policy,
    )
    from lossless_cache_replacement_loader import (
        LosslessCacheReplacementLoaderConfig,
        iter_newbie_lossless_cache_replacement_batches,
    )
    from lossless_cache_prefetch_queue import (
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
    from newbie_cached_dataset import NewbieCachedDataset, create_newbie_cached_dataloader


@dataclass(frozen=True)
class LosslessCacheTrainingAbConfig:
    batch_size: int = 1
    max_batches: int = 4
    prefetch_depth: int = 2
    sidecar_dir: str | None = None
    sidecar_suffix: str = ".lxcs"
    sidecar_strict: bool = False
    fallback_to_raw: bool = True
    prepare_sidecars: bool = True
    chunk_size: int = 1 << 20
    min_saving: float = 0.02
    device: str = "auto"
    compute_repeat: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    replacement_policy: str = "always"
    policy_min_raw_bytes: int = 64 * 1024
    policy_light_compute_min_raw_bytes: int = 512 * 1024
    policy_light_compute_repeat: int = 4
    policy_min_saved_bytes: int = 64 * 1024
    policy_max_compression_ratio: float = 0.85
    sample_gpu_metrics: bool = False
    gpu_sample_interval_ms: int = 250


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _round(value: float) -> float:
    return round(float(value), 4)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    items = sorted(float(item) for item in values)
    if len(items) == 1:
        return items[0]
    rank = (len(items) - 1) * min(max(float(q), 0.0), 1.0)
    low = int(rank)
    high = min(low + 1, len(items) - 1)
    frac = rank - low
    return items[low] * (1.0 - frac) + items[high] * frac


def _timings(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "total_ms": _round(sum(values)),
        "p50_ms": _round(_percentile(values, 0.50)),
        "p95_ms": _round(_percentile(values, 0.95)),
        "max_ms": _round(max(values)),
    }


class _GpuMetricSampler:
    def __init__(self, *, enabled: bool, device: str, interval_ms: int):
        self.enabled = bool(enabled) and str(device).startswith("cuda")
        self.interval_s = max(float(interval_ms), 50.0) / 1000.0
        self.samples: list[dict[str, float]] = []
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi_not_found"
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, name="lxcs-gpu-metric-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(self.interval_s * 2.0, 0.5))
        if self.enabled and not self.error:
            self._sample()
        return self.report()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if result.returncode != 0:
                self.error = (result.stderr or "nvidia-smi_failed").strip()[:200]
                return
            line = (result.stdout or "").splitlines()[0]
            gpu_util, memory_used, memory_total = [float(item.strip()) for item in line.split(",")[:3]]
            self.samples.append(
                {
                    "gpu_util_percent": gpu_util,
                    "memory_used_mb": memory_used,
                    "memory_total_mb": memory_total,
                }
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def report(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "available": False, "sample_count": 0}
        if not self.samples:
            return {"enabled": True, "available": False, "sample_count": 0, "error": self.error}
        gpu_values = [sample["gpu_util_percent"] for sample in self.samples]
        memory_values = [sample["memory_used_mb"] for sample in self.samples]
        total_values = [sample["memory_total_mb"] for sample in self.samples]
        return {
            "enabled": True,
            "available": True,
            "sample_count": len(self.samples),
            "gpu_util_avg_percent": _round(sum(gpu_values) / max(float(len(gpu_values)), 1.0)),
            "gpu_util_max_percent": _round(max(gpu_values)),
            "memory_used_max_mb": _round(max(memory_values)),
            "memory_total_mb": _round(max(total_values)),
            "error": self.error,
        }


def _resolve_device(requested: str) -> str:
    import torch

    value = str(requested or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return value


def _move_batch(batch: dict[str, object], device: str) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class _ScalarProbeModel:
    def __init__(self, device: str):
        import torch

        self.model = torch.nn.Parameter(torch.tensor(0.125, device=device))
        self.optimizer = torch.optim.SGD([self.model], lr=1.0e-3)

    def step(self, batch: dict[str, Any], *, repeat: int) -> float:
        import torch

        latents = batch["latents"].float()
        hidden = batch["encoder_hidden_states"].float()
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            hidden = hidden * mask.to(dtype=hidden.dtype).unsqueeze(-1)
        loss = torch.zeros((), device=latents.device, dtype=torch.float32)
        loops = max(int(repeat), 1)
        scale = self.model
        for _ in range(loops):
            latent_term = (latents * scale).square().mean()
            hidden_term = (hidden * scale).square().mean()
            loss = loss + latent_term + hidden_term
        loss = loss / float(loops)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if latents.device.type == "cuda":
            torch.cuda.synchronize(latents.device)
        return float(loss.detach().cpu())


def _warm_training_runtime(device: str) -> None:
    import torch

    model = _ScalarProbeModel(device)
    batch = {
        "latents": torch.zeros((1, 16, 4, 4), device=device),
        "encoder_hidden_states": torch.zeros((1, 4, 8), device=device),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool, device=device),
    }
    model.step(batch, repeat=1)


def _run_baseline_loop(dataset: NewbieCachedDataset, cfg: LosslessCacheTrainingAbConfig, device: str) -> dict[str, Any]:
    loader = create_newbie_cached_dataloader(
        dataset,
        batch_size=max(int(cfg.batch_size), 1),
        shuffle=False,
        num_workers=max(int(cfg.num_workers), 0),
        persistent_workers=bool(cfg.persistent_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=max(int(cfg.prefetch_factor), 1),
        drop_last=False,
    )
    return _run_training_loop(iter(loader), max_batches=max(int(cfg.max_batches), 1), device=device, cfg=cfg)


def _prepare_replacement_sidecars(
    dataset: NewbieCachedDataset,
    cfg: LosslessCacheTrainingAbConfig,
    codecs: Iterable[str],
) -> dict[str, Any]:
    if not cfg.prepare_sidecars:
        return {"ok": True, "skipped": True}
    limit = max(int(cfg.batch_size), 1) * max(int(cfg.max_batches), 1)
    paths = [sample.cache_path for sample in list(dataset.samples)[:limit]]
    return prepare_lossless_cache_prefetch_sidecars(
        paths,
        config=LosslessCachePrefetchQueueConfig(
            prefetch_depth=max(int(cfg.prefetch_depth), 1),
            sidecar_dir=cfg.sidecar_dir,
            sidecar_suffix=cfg.sidecar_suffix,
            sidecar_strict=cfg.sidecar_strict,
            fallback_to_raw=cfg.fallback_to_raw,
        ),
        chunk_size=max(int(cfg.chunk_size), 1),
        codecs=codecs,
        min_saving=float(cfg.min_saving),
    )


def _policy_config(cfg: LosslessCacheTrainingAbConfig) -> LosslessCacheReplacementPolicyConfig:
    return LosslessCacheReplacementPolicyConfig(
        mode=str(cfg.replacement_policy or "always"),
        min_raw_bytes=max(int(cfg.policy_min_raw_bytes), 0),
        light_compute_min_raw_bytes=max(int(cfg.policy_light_compute_min_raw_bytes), 0),
        light_compute_repeat=max(int(cfg.policy_light_compute_repeat), 1),
        min_saved_bytes=max(int(cfg.policy_min_saved_bytes), 0),
        max_compression_ratio=float(cfg.policy_max_compression_ratio),
    )


def _policy_workload(dataset: NewbieCachedDataset, cfg: LosslessCacheTrainingAbConfig, device: str) -> dict[str, Any]:
    return {
        "family": "newbie",
        "sample_count": len(dataset.samples),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "compute_repeat": max(int(cfg.compute_repeat), 1),
        "device": str(device or cfg.device or "auto"),
    }


def _bypassed_replacement_report(
    baseline: dict[str, Any],
    *,
    policy: dict[str, Any],
    sidecar_prepare: dict[str, Any],
) -> dict[str, Any]:
    report = dict(baseline)
    report.update(
        {
            "selected_path": "raw_dataloader",
            "bypassed": True,
            "bypass_reason": str(policy.get("reason") or "policy_bypass"),
            "replacement_policy": policy,
            "sidecar_prepare": sidecar_prepare,
            "training_path_enabled": False,
        }
    )
    return report


def _run_replacement_loop(
    dataset: NewbieCachedDataset,
    cfg: LosslessCacheTrainingAbConfig,
    device: str,
    codecs: Iterable[str],
    sidecar_prepare: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sidecar_prepare is None:
        sidecar_prepare = _prepare_replacement_sidecars(dataset, cfg, codecs)
    iterator = iter_newbie_lossless_cache_replacement_batches(
        dataset,
        config=LosslessCacheReplacementLoaderConfig(
            batch_size=max(int(cfg.batch_size), 1),
            max_batches=max(int(cfg.max_batches), 1),
            prefetch_depth=max(int(cfg.prefetch_depth), 1),
            sidecar_dir=cfg.sidecar_dir,
            sidecar_suffix=cfg.sidecar_suffix,
            sidecar_strict=cfg.sidecar_strict,
            fallback_to_raw=cfg.fallback_to_raw,
            prepare_sidecars=False,
            chunk_size=max(int(cfg.chunk_size), 1),
            min_saving=float(cfg.min_saving),
        ),
        codecs=codecs,
    )
    report = _run_training_loop(iterator, max_batches=max(int(cfg.max_batches), 1), device=device, cfg=cfg)
    report["sidecar_prepare"] = sidecar_prepare
    return report


def _run_training_loop(iterator: Any, *, max_batches: int, device: str, cfg: LosslessCacheTrainingAbConfig) -> dict[str, Any]:
    model = _ScalarProbeModel(device)
    loader_wait: list[float] = []
    h2d_ms: list[float] = []
    step_ms: list[float] = []
    losses: list[float] = []
    batch_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    started = _now()
    gpu_sampler = _GpuMetricSampler(
        enabled=bool(cfg.sample_gpu_metrics),
        device=device,
        interval_ms=max(int(cfg.gpu_sample_interval_ms), 1),
    )
    gpu_sampler.start()
    try:
        for _ in range(max(int(max_batches), 1)):
            wait_started = _now()
            try:
                item = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                errors.append(f"loader:{type(exc).__name__}: {exc}")
                break
            loader_wait.append(_elapsed_ms(wait_started))
            if isinstance(item, tuple) and len(item) == 2:
                batch, batch_report = item
                batch_reports.append(dict(batch_report))
            else:
                batch = item
            h2d_started = _now()
            moved = _move_batch(batch, device)
            if str(device).startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            h2d_ms.append(_elapsed_ms(h2d_started))
            step_started = _now()
            try:
                losses.append(model.step(moved, repeat=max(int(cfg.compute_repeat), 1)))
            except Exception as exc:
                errors.append(f"step:{type(exc).__name__}: {exc}")
                break
            step_ms.append(_elapsed_ms(step_started))
    finally:
        gpu_metrics = gpu_sampler.stop()
    return {
        "ok": not errors,
        "batch_count": len(step_ms),
        "wall_ms": _round(_elapsed_ms(started)),
        "loader_wait": _timings(loader_wait),
        "h2d": _timings(h2d_ms),
        "step": _timings(step_ms),
        "loss_first": _round(losses[0]) if losses else 0.0,
        "loss_last": _round(losses[-1]) if losses else 0.0,
        "batch_reports": batch_reports,
        "gpu_metrics": gpu_metrics,
        "errors": errors,
    }


def run_newbie_lossless_cache_training_ab(
    baseline_dataset: NewbieCachedDataset,
    replacement_dataset: NewbieCachedDataset,
    *,
    config: LosslessCacheTrainingAbConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> dict[str, Any]:
    cfg = config or LosslessCacheTrainingAbConfig()
    device = _resolve_device(cfg.device)
    _warm_training_runtime(device)
    baseline = _run_baseline_loop(baseline_dataset, cfg, device)
    policy_mode = str(cfg.replacement_policy or "always").lower()
    sidecar_prepare = {"ok": True, "skipped": True}
    policy = evaluate_lossless_cache_replacement_policy(
        sidecar_prepare,
        workload=_policy_workload(replacement_dataset, cfg, device),
        config=_policy_config(cfg),
    )
    if policy_mode == "adaptive":
        sidecar_prepare = _prepare_replacement_sidecars(replacement_dataset, cfg, codecs)
        policy = evaluate_lossless_cache_replacement_policy(
            sidecar_prepare,
            workload=_policy_workload(replacement_dataset, cfg, device),
            config=_policy_config(cfg),
        )
    if not bool(policy.get("enabled")):
        replacement = _bypassed_replacement_report(baseline, policy=policy, sidecar_prepare=sidecar_prepare)
    else:
        prepared = sidecar_prepare if policy_mode == "adaptive" else None
        replacement = _run_replacement_loop(replacement_dataset, cfg, device, codecs, sidecar_prepare=prepared)
        replacement["selected_path"] = "lxcs_replacement"
        replacement["bypassed"] = False
        replacement["replacement_policy"] = policy
    baseline_wall = float(baseline.get("wall_ms") or 0.0)
    replacement_wall = float(replacement.get("wall_ms") or 0.0)
    return {
        "provider": "lxcs_newbie_training_step_ab_v1",
        "ok": bool(baseline.get("ok")) and bool(replacement.get("ok")),
        "device": device,
        "sample_count": min(len(baseline_dataset.samples), len(replacement_dataset.samples)),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "compute_repeat": max(int(cfg.compute_repeat), 1),
        "baseline": baseline,
        "replacement": replacement,
        "replacement_vs_baseline_wall_ratio": _round(replacement_wall / baseline_wall) if baseline_wall > 0 else 0.0,
        "replacement_policy": policy,
        "selected_path": str(policy.get("selected_path") or "raw_dataloader"),
        "policy_enabled": bool(policy.get("enabled")),
        "policy_bypassed": not bool(policy.get("enabled")),
        "mini_training_step_ab": True,
        "real_training_model": False,
        "training_path_enabled": False,
        "p3_training_step_ab_probe_ready": bool(baseline.get("ok")) and bool(replacement.get("ok")),
        "readiness_blockers": [
            "mini_probe_not_real_trainer_model",
            "no full optimizer_forward_backward_wall_clock_ab",
            "gpu_idle_and_step_p95_not_verified",
            "replacement_path_not_enabled_in_runtime_request",
        ],
    }


__all__ = [
    "LosslessCacheTrainingAbConfig",
    "run_newbie_lossless_cache_training_ab",
]
