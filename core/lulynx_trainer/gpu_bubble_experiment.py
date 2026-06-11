"""GPU telemetry and benchmark evidence aggregation for bubble experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BASIC_GPU_FIELDS: tuple[str, ...] = (
    "timestamp",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text.upper() in {"N/A", "[N/A]", "NA"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return default


def _round(value: float, digits: int = 6) -> float:
    return round(float(value or 0.0), digits)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return sum(items) / len(items) if items else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    items = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    rank = (len(items) - 1) * min(max(float(percentile), 0.0), 1.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return items[lo]
    frac = rank - lo
    return items[lo] * (1.0 - frac) + items[hi] * frac


def parse_csv_row(text: str, fields: Sequence[str]) -> dict[str, str]:
    rows = list(csv.reader([text]))
    if not rows:
        return {}
    values = [item.strip() for item in rows[0]]
    return {str(field): values[index] if index < len(values) else "" for index, field in enumerate(fields)}


def normalize_basic_sample(row: Mapping[str, Any], *, monotonic_seconds: float | None = None) -> dict[str, Any]:
    return {
        "timestamp": str(row.get("timestamp", "") or ""),
        "monotonic_seconds": _round(float(monotonic_seconds if monotonic_seconds is not None else time.monotonic()), 6),
        "name": str(row.get("name", "") or ""),
        "gpu_util_pct": _safe_float(row.get("utilization.gpu")),
        "memory_util_pct": _safe_float(row.get("utilization.memory")),
        "memory_used_mb": _safe_float(row.get("memory.used")),
        "memory_total_mb": _safe_float(row.get("memory.total")),
        "power_draw_w": _safe_float(row.get("power.draw")),
        "temperature_gpu_c": _safe_float(row.get("temperature.gpu")),
        "pcie_link_gen": _safe_int(row.get("pcie.link.gen.current")),
        "pcie_link_width": _safe_int(row.get("pcie.link.width.current")),
    }


def parse_dmon_pcie_output(text: str) -> dict[str, float]:
    for raw_line in reversed(str(text or "").splitlines()):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        return {
            "pcie_rx_mib_s": _safe_float(parts[1]),
            "pcie_tx_mib_s": _safe_float(parts[2]),
        }
    return {}


def query_nvidia_smi_sample(
    *,
    gpu_index: int = 0,
    nvidia_smi: str = "nvidia-smi",
    include_pcie_dmon: bool = True,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    fields = ",".join(BASIC_GPU_FIELDS)
    command = [
        nvidia_smi,
        "-i",
        str(max(int(gpu_index), 0)),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=max(float(timeout_seconds), 0.1),
    )
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    sample = normalize_basic_sample(parse_csv_row(first_line, BASIC_GPU_FIELDS))
    if include_pcie_dmon:
        dmon = subprocess.run(
            [nvidia_smi, "dmon", "-i", str(max(int(gpu_index), 0)), "-s", "t", "-c", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 0.1),
        )
        sample.update(parse_dmon_pcie_output(dmon.stdout))
    return sample


class GpuTelemetrySampler:
    """Background nvidia-smi sampler for short benchmark runs."""

    def __init__(
        self,
        *,
        gpu_index: int = 0,
        interval_seconds: float = 0.5,
        nvidia_smi: str = "nvidia-smi",
        include_pcie_dmon: bool = True,
    ) -> None:
        self.gpu_index = max(int(gpu_index), 0)
        self.interval_seconds = max(float(interval_seconds), 0.1)
        self.nvidia_smi = str(nvidia_smi or "nvidia-smi")
        self.include_pcie_dmon = bool(include_pcie_dmon)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2.0, 1.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                sample = query_nvidia_smi_sample(
                    gpu_index=self.gpu_index,
                    nvidia_smi=self.nvidia_smi,
                    include_pcie_dmon=self.include_pcie_dmon,
                )
                self.samples.append(sample)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                if len(self.errors) >= 3:
                    return
            elapsed = time.monotonic() - started
            self._stop.wait(max(self.interval_seconds - elapsed, 0.05))


def summarize_gpu_telemetry(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in samples if isinstance(item, Mapping)]
    if not rows:
        return {
            "sample_count": 0,
            "available": False,
            "reason": "no_samples",
        }

    gpu_util = [_safe_float(row.get("gpu_util_pct")) for row in rows]
    mem_util = [_safe_float(row.get("memory_util_pct")) for row in rows]
    memory_used = [_safe_float(row.get("memory_used_mb")) for row in rows]
    memory_total = [_safe_float(row.get("memory_total_mb")) for row in rows]
    power = [_safe_float(row.get("power_draw_w")) for row in rows]
    temp = [_safe_float(row.get("temperature_gpu_c")) for row in rows]
    pcie_rx = [_safe_float(row.get("pcie_rx_mib_s")) for row in rows if "pcie_rx_mib_s" in row]
    pcie_tx = [_safe_float(row.get("pcie_tx_mib_s")) for row in rows if "pcie_tx_mib_s" in row]
    start = _safe_float(rows[0].get("monotonic_seconds"))
    end = _safe_float(rows[-1].get("monotonic_seconds"), start)
    active = [value for value in gpu_util if value >= 70.0]
    saturated = [value for value in gpu_util if value >= 90.0]
    idle = [value for value in gpu_util if value < 20.0]

    return {
        "sample_count": len(rows),
        "available": True,
        "duration_seconds": _round(max(end - start, 0.0), 4),
        "gpu_name": str(rows[-1].get("name", "") or ""),
        "gpu_util_pct_mean": _round(_mean(gpu_util), 4),
        "gpu_util_pct_p50": _round(_percentile(gpu_util, 0.50), 4),
        "gpu_util_pct_p95": _round(_percentile(gpu_util, 0.95), 4),
        "gpu_util_pct_max": _round(max(gpu_util) if gpu_util else 0.0, 4),
        "gpu_active_sample_ratio": _round(len(active) / max(len(gpu_util), 1), 6),
        "gpu_saturated_sample_ratio": _round(len(saturated) / max(len(gpu_util), 1), 6),
        "gpu_idle_sample_ratio": _round(len(idle) / max(len(gpu_util), 1), 6),
        "memory_util_pct_mean": _round(_mean(mem_util), 4),
        "memory_used_mb_max": _round(max(memory_used) if memory_used else 0.0, 4),
        "memory_total_mb": _round(max(memory_total) if memory_total else 0.0, 4),
        "power_draw_w_mean": _round(_mean(power), 4),
        "power_draw_w_max": _round(max(power) if power else 0.0, 4),
        "temperature_gpu_c_max": _round(max(temp) if temp else 0.0, 4),
        "pcie_rx_mib_s_mean": _round(_mean(pcie_rx), 4),
        "pcie_rx_mib_s_max": _round(max(pcie_rx) if pcie_rx else 0.0, 4),
        "pcie_tx_mib_s_mean": _round(_mean(pcie_tx), 4),
        "pcie_tx_mib_s_max": _round(max(pcie_tx) if pcie_tx else 0.0, 4),
        "pcie_link_gen": _safe_int(rows[-1].get("pcie_link_gen")),
        "pcie_link_width": _safe_int(rows[-1].get("pcie_link_width")),
    }


def _summarize_scoped_gpu_telemetry(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    threshold: float,
    contiguous: bool,
) -> dict[str, Any]:
    summary = summarize_gpu_telemetry(rows)
    summary["scope"] = scope
    summary["threshold_gpu_util_pct"] = _round(threshold, 4)
    summary["contiguous"] = bool(contiguous)
    return summary


def _contiguous_gpu_window(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[Mapping[str, Any]]:
    indexes = [
        index for index, row in enumerate(rows)
        if _safe_float(row.get("gpu_util_pct")) >= threshold
    ]
    if not indexes:
        return []
    return list(rows[indexes[0]: indexes[-1] + 1])


def summarize_gpu_telemetry_windows(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in samples if isinstance(item, Mapping)]
    windows: dict[str, Any] = {}
    if not rows:
        return windows

    active20 = _contiguous_gpu_window(rows, 20.0)
    if active20:
        windows["active_window_gpu20"] = _summarize_scoped_gpu_telemetry(
            active20,
            scope="first_to_last_sample_with_gpu_util_ge_20",
            threshold=20.0,
            contiguous=True,
        )

    compute50 = _contiguous_gpu_window(rows, 50.0)
    if compute50:
        windows["compute_window_gpu50"] = _summarize_scoped_gpu_telemetry(
            compute50,
            scope="first_to_last_sample_with_gpu_util_ge_50",
            threshold=50.0,
            contiguous=True,
        )

    hot70 = [row for row in rows if _safe_float(row.get("gpu_util_pct")) >= 70.0]
    if hot70:
        windows["hot_samples_gpu70"] = _summarize_scoped_gpu_telemetry(
            hot70,
            scope="noncontiguous_samples_with_gpu_util_ge_70",
            threshold=70.0,
            contiguous=False,
        )

    saturated90 = [row for row in rows if _safe_float(row.get("gpu_util_pct")) >= 90.0]
    if saturated90:
        windows["saturated_samples_gpu90"] = _summarize_scoped_gpu_telemetry(
            saturated90,
            scope="noncontiguous_samples_with_gpu_util_ge_90",
            threshold=90.0,
            contiguous=False,
        )

    return windows


def _run_summaries(benchmark_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = benchmark_payload.get("runs")
    if not isinstance(runs, Mapping):
        return []
    summaries: list[dict[str, Any]] = []
    for label, run in runs.items():
        if not isinstance(run, Mapping):
            continue
        bubble = run.get("steady_bubble_profile")
        evidence = bubble.get("evidence") if isinstance(bubble, Mapping) else {}
        summary = {
            "label": str(label),
            "success": bool(run.get("success", False)),
            "steps_completed": _safe_int(run.get("steps_completed")),
            "steady_mean_step_ms": _round(_safe_float(run.get("steady_mean_step_ms")), 4),
            "steady_samples_per_second": _round(_safe_float(run.get("steady_samples_per_second")), 6),
            "peak_vram_mb": _round(_safe_float(run.get("peak_vram_mb")), 4),
            "final_loss": _round(_safe_float(run.get("final_loss")), 6),
            "dominant_bottleneck": str(bubble.get("dominant_bottleneck", "unknown") if isinstance(bubble, Mapping) else "unknown"),
            "bubble_ratio_estimate": _round(_safe_float(bubble.get("bubble_ratio_estimate") if isinstance(bubble, Mapping) else 0.0), 6),
            "data_wait_share": _round(_safe_float(evidence.get("data_wait_share") if isinstance(evidence, Mapping) else 0.0), 6),
            "h2d_transfer_share": _round(_safe_float(evidence.get("h2d_transfer_share") if isinstance(evidence, Mapping) else 0.0), 6),
            "optimizer_share": _round(_safe_float(evidence.get("optimizer_share") if isinstance(evidence, Mapping) else 0.0), 6),
            "host_gap_share": _round(_safe_float(evidence.get("host_gap_share") if isinstance(evidence, Mapping) else 0.0), 6),
        }
        runtime_features = run.get("runtime_feature_summary")
        if isinstance(runtime_features, Mapping):
            summary["runtime_feature_summary"] = dict(runtime_features)
        summaries.append(summary)
    return summaries


def _iter_residency_summaries(run: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    runtime_features = run.get("runtime_feature_summary")
    if not isinstance(runtime_features, Mapping):
        return
    for key in ("anima_block_residency", "newbie_block_residency"):
        residency = runtime_features.get(key)
        if isinstance(residency, Mapping):
            yield residency


def _classify_experiment(
    gpu: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    active_gpu: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_mean = _safe_float(gpu.get("gpu_util_pct_mean"))
    saturated_ratio = _safe_float(gpu.get("gpu_saturated_sample_ratio"))
    idle_ratio = _safe_float(gpu.get("gpu_idle_sample_ratio"))
    active_source = active_gpu if isinstance(active_gpu, Mapping) and active_gpu.get("available") else gpu
    active_gpu_mean = _safe_float(active_source.get("gpu_util_pct_mean"))
    active_saturated_ratio = _safe_float(active_source.get("gpu_saturated_sample_ratio"))
    active_idle_ratio = _safe_float(active_source.get("gpu_idle_sample_ratio"))
    active_scope = str(active_source.get("scope", "full_run") or "full_run")
    bottlenecks = [str(run.get("dominant_bottleneck", "unknown")) for run in runs]
    transfer_share = max((_safe_float(run.get("h2d_transfer_share")) for run in runs), default=0.0)
    data_share = max((_safe_float(run.get("data_wait_share")) for run in runs), default=0.0)
    residency_profiles = [profile for run in runs for profile in _iter_residency_summaries(run)]
    offload_active = [
        profile for profile in residency_profiles
        if bool((profile.get("proof_flags") if isinstance(profile.get("proof_flags"), Mapping) else {}).get("streaming_offload_active"))
    ]
    prefetch_missed = max(
        (
            _safe_int((profile.get("prefetch") if isinstance(profile.get("prefetch"), Mapping) else {}).get("missed"))
            for profile in residency_profiles
        ),
        default=0,
    )
    residency_h2d_mb = max((_safe_float(profile.get("transfer_h2d_mb")) for profile in residency_profiles), default=0.0)

    if active_gpu_mean >= 85.0 or active_saturated_ratio >= 0.60:
        status = "gpu_saturated"
    elif transfer_share >= 0.05:
        status = "transfer_bound"
    elif data_share >= 0.08:
        status = "data_bound"
    elif active_idle_ratio >= 0.35 or active_gpu_mean < 60.0:
        status = "gpu_underfed_or_host_bound"
    elif "compute_bound" in bottlenecks:
        status = "compute_bound_not_fully_saturated"
    else:
        status = "mixed"

    return {
        "status": status,
        "gpu_util_pct_mean": _round(gpu_mean, 4),
        "gpu_saturated_sample_ratio": _round(saturated_ratio, 6),
        "gpu_idle_sample_ratio": _round(idle_ratio, 6),
        "active_gpu_scope": active_scope,
        "active_gpu_util_pct_mean": _round(active_gpu_mean, 4),
        "active_gpu_saturated_sample_ratio": _round(active_saturated_ratio, 6),
        "active_gpu_idle_sample_ratio": _round(active_idle_ratio, 6),
        "max_h2d_transfer_share": _round(transfer_share, 6),
        "max_data_wait_share": _round(data_share, 6),
        "offload_active_run_count": len(offload_active),
        "max_residency_transfer_h2d_mb": _round(residency_h2d_mb, 3),
        "max_prefetch_missed": prefetch_missed,
        "run_bottlenecks": bottlenecks,
    }


def build_experiment_report(
    *,
    benchmark_payload: Mapping[str, Any],
    telemetry_samples: Iterable[Mapping[str, Any]],
    command: Sequence[str] | None = None,
    return_code: int | None = None,
    sampler_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    gpu_summary = summarize_gpu_telemetry(telemetry_samples)
    gpu_windows = summarize_gpu_telemetry_windows(telemetry_samples)
    active_gpu = gpu_windows.get("active_window_gpu20") if isinstance(gpu_windows, Mapping) else None
    runs = _run_summaries(benchmark_payload)
    return {
        "schema_version": 1,
        "report": "gpu_bubble_experiment_report_v0",
        "command": list(command or []),
        "return_code": return_code,
        "benchmark": dict(benchmark_payload.get("benchmark", {}) if isinstance(benchmark_payload, Mapping) else {}),
        "gpu_telemetry": gpu_summary,
        "gpu_telemetry_windows": gpu_windows,
        "run_summaries": runs,
        "classification": _classify_experiment(gpu_summary, runs, active_gpu=active_gpu),
        "comparison": dict(benchmark_payload.get("comparison", {}) if isinstance(benchmark_payload.get("comparison"), Mapping) else {}),
        "sampler_errors": list(sampler_errors or []),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_summary(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _strip_command_separator(command: Sequence[str]) -> list[str]:
    items = list(command)
    if items and items[0] == "--":
        return items[1:]
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", type=Path, default=None)
    parser.add_argument("--telemetry-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("temp/gpu_bubble_experiment"))
    parser.add_argument("--summary-glob", default="**/*_summary.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--no-pcie-dmon", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _strip_command_separator(args.command)
    return_code: int | None = None
    sampler_errors: list[str] = []
    samples: list[dict[str, Any]] = []

    if command:
        sampler = GpuTelemetrySampler(
            gpu_index=args.gpu_index,
            interval_seconds=args.sample_interval,
            nvidia_smi=args.nvidia_smi,
            include_pcie_dmon=not bool(args.no_pcie_dmon),
        )
        sampler.start()
        try:
            completed = subprocess.run(command, check=False)
            return_code = int(completed.returncode)
        finally:
            sampler.stop()
        samples = list(sampler.samples)
        sampler_errors = list(sampler.errors)
    elif args.telemetry_json is not None:
        telemetry_payload = _load_json(args.telemetry_json)
        samples = list(telemetry_payload.get("samples", telemetry_payload) if isinstance(telemetry_payload, Mapping) else telemetry_payload)
    else:
        samples = [query_nvidia_smi_sample(gpu_index=args.gpu_index, nvidia_smi=args.nvidia_smi)]

    telemetry_path = out_dir / "gpu_telemetry_samples.json"
    telemetry_path.write_text(json.dumps({"samples": samples, "errors": sampler_errors}, indent=2), encoding="utf-8")

    summary_path = args.benchmark_summary
    if summary_path is None:
        summary_path = _find_summary(out_dir, str(args.summary_glob))
    if summary_path is None:
        gpu_summary = summarize_gpu_telemetry(samples)
        gpu_windows = summarize_gpu_telemetry_windows(samples)
        report = {
            "schema_version": 1,
            "report": "gpu_bubble_experiment_report_v0",
            "command": command,
            "return_code": return_code,
            "gpu_telemetry": gpu_summary,
            "gpu_telemetry_windows": gpu_windows,
            "run_summaries": [],
            "classification": _classify_experiment(gpu_summary, [], active_gpu=gpu_windows.get("active_window_gpu20")),
            "sampler_errors": sampler_errors,
            "benchmark_summary_missing": True,
        }
    else:
        report = build_experiment_report(
            benchmark_payload=_load_json(summary_path),
            telemetry_samples=samples,
            command=command,
            return_code=return_code,
            sampler_errors=sampler_errors,
        )
        report["benchmark_summary_path"] = str(summary_path)
    report["telemetry_path"] = str(telemetry_path)

    out_path = args.out or out_dir / "gpu_bubble_experiment_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[gpu-bubble] report={out_path}", flush=True)
    print(f"[gpu-bubble] classification={json.dumps(report.get('classification', {}), sort_keys=True)}", flush=True)
    return 0 if return_code in (None, 0) else return_code


if __name__ == "__main__":
    raise SystemExit(main())
