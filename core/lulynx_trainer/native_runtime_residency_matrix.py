"""Run DiT residency benchmark matrices in isolated subprocesses.

This wraps ``native_runtime_profile_benchmark.py`` so larger residency tests can
continue after a single OOM or runtime failure.  It is intended for Anima/Newbie
batch-size and Streaming Offload comparisons.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESIDENCY_CHOICES = ("resident", "streaming_offload", "block_cpu_pinned")
_PIPE_DONE = object()


@dataclass
class MatrixCase:
    family: str
    batch_size: int
    residency: str
    checkpointing: bool
    prefetch: bool
    prefetch_depth: int
    out_dir: Path
    command: list[str]
    returncode: int
    wall_seconds: float
    timed_out: bool = False
    summary_path: str = ""
    summary: dict[str, Any] | None = None
    stdout_tail: list[str] | None = None
    stderr_tail: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "batch_size": int(self.batch_size),
            "residency": self.residency,
            "checkpointing": bool(self.checkpointing),
            "prefetch": bool(self.prefetch),
            "prefetch_depth": int(self.prefetch_depth),
            "out_dir": str(self.out_dir),
            "command": list(self.command),
            "returncode": int(self.returncode),
            "success": self.returncode == 0,
            "wall_seconds": round(float(self.wall_seconds), 3),
            "timed_out": bool(self.timed_out),
            "summary_path": self.summary_path,
            "summary": self.summary or {},
            "stdout_tail": list(self.stdout_tail or []),
            "stderr_tail": list(self.stderr_tail or []),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _split_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for raw in str(value or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(max(int(item), 1))
    return out or [1]


def _split_csv_modes(value: str) -> list[str]:
    modes: list[str] = []
    for raw in str(value or "").replace(";", ",").split(","):
        mode = raw.strip().lower().replace("-", "_")
        if mode and mode in RESIDENCY_CHOICES and mode not in modes:
            modes.append(mode)
    return modes or ["resident", "streaming_offload", "block_cpu_pinned"]


def _tail_lines(text: str, limit: int = 40) -> list[str]:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    return lines[-max(int(limit), 1):]


def _append_tail(lines: list[str], line: str, *, limit: int = 120) -> None:
    lines.append(line.rstrip())
    del lines[: max(len(lines) - max(int(limit), 1), 0)]


def _pipe_reader(pipe: Any, output: "queue.Queue[str | object]") -> None:
    try:
        for line in pipe:
            output.put(str(line))
    finally:
        output.put(_PIPE_DONE)


def _summary_brief(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs", {}) if isinstance(summary, dict) else {}
    first_run = next(iter(runs.values()), {}) if isinstance(runs, dict) and runs else {}
    if not isinstance(first_run, dict):
        first_run = {}
    benchmark = summary.get("benchmark", {}) if isinstance(summary, dict) else {}
    return {
        "benchmark": benchmark,
        "success": bool(first_run.get("success", False)),
        "steps_completed": int(first_run.get("steps_completed", 0) or 0),
        "mean_step_ms": float(first_run.get("mean_step_ms", 0.0) or 0.0),
        "steady_mean_step_ms": float(first_run.get("steady_mean_step_ms", 0.0) or 0.0),
        "steady_samples_per_second": float(first_run.get("steady_samples_per_second", 0.0) or 0.0),
        "peak_vram_mb": float(first_run.get("peak_vram_mb", 0.0) or 0.0),
        "setup_peak_vram_mb": float(first_run.get("setup_peak_vram_mb", 0.0) or 0.0),
        "training_peak_vram_mb": float(first_run.get("training_peak_vram_mb", 0.0) or 0.0),
    }


def _build_command(
    *,
    python_exe: str,
    benchmark_script: Path,
    family: str,
    batch_size: int,
    residency: str,
    checkpointing: bool,
    prefetch: bool,
    prefetch_depth: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_exe,
        str(benchmark_script),
        "--family",
        family,
        "--profiles",
        str(args.profiles),
        "--steps",
        str(max(int(args.steps), 1)),
        "--steady-warmup",
        str(max(int(args.steady_warmup), 0)),
        "--samples",
        str(max(int(args.samples), 1)),
        "--resolution",
        str(max(int(args.resolution), 1)),
        "--network-dim",
        str(max(int(args.network_dim), 1)),
        "--train-batch-size",
        str(max(int(batch_size), 1)),
        "--out",
        str(out_dir),
        "--lora-activation-recompute",
        str(args.lora_activation_recompute),
    ]
    if args.source_data:
        command.extend(["--source-data", str(args.source_data)])
    if args.fused_adamw:
        command.append("--fused-adamw")

    if family == "anima":
        command.extend([
            "--anima-latent-crop-size",
            str(max(int(args.anima_latent_crop_size), 0)),
            "--anima-fixed-text-tokens",
            str(max(int(args.anima_fixed_text_tokens), 0)),
            "--anima-fixed-visual-tokens",
            str(max(int(args.anima_fixed_visual_tokens), 0)),
            "--anima-block-residency",
            residency,
            "--anima-block-residency-min-params",
            str(max(int(args.residency_min_params), 0)),
        ])
        if checkpointing:
            command.append("--anima-block-checkpointing")
        if prefetch:
            command.extend([
                "--anima-block-prefetch",
                "--anima-block-prefetch-depth",
                str(max(int(prefetch_depth), 0)),
            ])
    else:
        command.extend([
            "--newbie-latent-crop-size",
            str(max(int(args.newbie_latent_crop_size), 0)),
            "--newbie-fixed-text-tokens",
            str(max(int(args.newbie_fixed_text_tokens), 0)),
            "--newbie-fixed-visual-tokens",
            str(max(int(args.newbie_fixed_visual_tokens), 0)),
            "--newbie-block-residency",
            residency,
            "--newbie-block-residency-min-params",
            str(max(int(args.residency_min_params), 0)),
        ])
        if checkpointing:
            command.append("--newbie-block-checkpointing")
        if prefetch:
            command.extend([
                "--newbie-block-prefetch",
                "--newbie-block-prefetch-depth",
                str(max(int(prefetch_depth), 0)),
            ])
    return command


def _run_case(
    command: list[str],
    *,
    family: str,
    batch_size: int,
    residency: str,
    checkpointing: bool,
    prefetch: bool,
    prefetch_depth: int,
    out_dir: Path,
    timeout_seconds: int,
    stream_output: bool,
) -> MatrixCase:
    start = time.perf_counter()
    timed_out = False
    stdout_tail: list[str] = []
    stderr_tail: list[str] = []
    returncode = 0
    if stream_output:
        process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        deadline = time.perf_counter() + timeout_seconds if timeout_seconds > 0 else None
        assert process.stdout is not None
        output_queue: "queue.Queue[str | object]" = queue.Queue()
        reader = threading.Thread(target=_pipe_reader, args=(process.stdout, output_queue), daemon=True)
        reader.start()
        stream_done = False
        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = None
            if line is _PIPE_DONE:
                stream_done = True
            elif line is not None:
                text_line = str(line)
                _append_tail(stdout_tail, text_line)
                print(f"[case:{family}/bs{batch_size}/{residency}] {text_line}", end="", flush=True)
            if deadline is not None and time.perf_counter() > deadline:
                timed_out = True
                process.kill()
                process.wait()
                _append_tail(stderr_tail, f"Timed out after {timeout_seconds} seconds")
                break
            if process.poll() is not None and stream_done:
                break
        while not output_queue.empty():
            rest = output_queue.get_nowait()
            if rest is not _PIPE_DONE:
                text_rest = str(rest)
                _append_tail(stdout_tail, text_rest)
                print(f"[case:{family}/bs{batch_size}/{residency}] {text_rest}", end="", flush=True)
        returncode = int(process.returncode if process.returncode is not None else -9)
    else:
        try:
            completed = subprocess.run(
                command,
                cwd=str(_repo_root()),
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
            )
            returncode = int(completed.returncode)
            stdout_tail = _tail_lines(completed.stdout, limit=120)
            stderr_tail = _tail_lines(completed.stderr, limit=120)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -9
            stdout_tail = _tail_lines(exc.stdout or "", limit=120)
            stderr_tail = _tail_lines(exc.stderr or "", limit=120)
            stderr_tail.append(f"Timed out after {timeout_seconds} seconds")
    wall_seconds = time.perf_counter() - start
    summary_path = out_dir / f"{family}_summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None
    return MatrixCase(
        family=family,
        batch_size=batch_size,
        residency=residency,
        checkpointing=checkpointing,
        prefetch=prefetch,
        prefetch_depth=prefetch_depth,
        out_dir=out_dir,
        command=command,
        returncode=returncode,
        wall_seconds=wall_seconds,
        timed_out=timed_out,
        summary_path=str(summary_path) if summary_path.exists() else "",
        summary=_summary_brief(summary or {}) if summary else None,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("anima", "newbie"), required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--residency-modes", default="resident,streaming_offload,block_cpu_pinned")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--steady-warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--network-dim", type=int, default=1)
    parser.add_argument("--profiles", choices=("standard", "aggressive"), default="standard")
    parser.add_argument("--source-data", type=Path, default=None)
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--checkpoint-block-cpu-pinned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DiT block checkpointing for streaming_offload cases. Use --no-checkpoint-streaming only for unsafe diagnostics.",
    )
    parser.add_argument("--residency-min-params", type=int, default=0)
    parser.add_argument("--anima-latent-crop-size", type=int, default=0)
    parser.add_argument("--anima-fixed-text-tokens", type=int, default=0)
    parser.add_argument("--anima-fixed-visual-tokens", type=int, default=0)
    parser.add_argument("--newbie-latent-crop-size", type=int, default=0)
    parser.add_argument("--newbie-fixed-text-tokens", type=int, default=0)
    parser.add_argument("--newbie-fixed-visual-tokens", type=int, default=0)
    parser.add_argument("--lora-activation-recompute", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--block-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable async block prefetch for streaming_offload cases.",
    )
    parser.add_argument("--block-prefetch-depth", type=int, default=1)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=0,
        help="Abort a single matrix case after this many seconds. 0 disables the timeout.",
    )
    parser.add_argument("--stream-output", action="store_true", help="Print each child benchmark's output as it runs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = _repo_root()
    benchmark_script = Path(__file__).resolve().with_name("native_runtime_profile_benchmark.py")
    run_root = args.out or repo / "temp" / "native_runtime_residency_matrix" / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    batch_sizes = _split_csv_ints(args.batch_sizes)
    residency_modes = _split_csv_modes(args.residency_modes)

    results: list[MatrixCase] = []
    for batch_size in batch_sizes:
        for residency in residency_modes:
            checkpointing = (
                (residency == "block_cpu_pinned" and bool(args.checkpoint_block_cpu_pinned))
                or (residency == "streaming_offload" and bool(args.checkpoint_streaming))
            )
            prefetch = residency == "streaming_offload" and bool(args.block_prefetch)
            prefetch_depth = max(int(args.block_prefetch_depth), 0)
            case_name = f"{args.family}_bs{batch_size}_{residency}"
            if checkpointing:
                case_name += "_ckpt"
            if prefetch:
                case_name += f"_prefetch{prefetch_depth}"
            out_dir = run_root / case_name
            command = _build_command(
                python_exe=str(args.python),
                benchmark_script=benchmark_script,
                family=str(args.family),
                batch_size=batch_size,
                residency=residency,
                checkpointing=checkpointing,
                prefetch=prefetch,
                prefetch_depth=prefetch_depth,
                out_dir=out_dir,
                args=args,
            )
            print(
                f"[matrix] start family={args.family} batch={batch_size} residency={residency} "
                f"checkpointing={checkpointing} prefetch={prefetch} depth={prefetch_depth}",
                flush=True,
            )
            result = _run_case(
                command,
                family=str(args.family),
                batch_size=batch_size,
                residency=residency,
                checkpointing=checkpointing,
                prefetch=prefetch,
                prefetch_depth=prefetch_depth,
                out_dir=out_dir,
                timeout_seconds=max(int(args.case_timeout_seconds), 0),
                stream_output=bool(args.stream_output),
            )
            results.append(result)
            brief = result.summary or {}
            print(
                f"[matrix] done returncode={result.returncode} success={result.returncode == 0} "
                f"timed_out={result.timed_out} "
                f"peak_vram={brief.get('peak_vram_mb', 0):.1f}MB "
                f"steady_ms={brief.get('steady_mean_step_ms', 0):.2f} "
                f"samples_per_sec={brief.get('steady_samples_per_second', 0):.3f}",
                flush=True,
            )
            if result.returncode != 0 and args.stop_on_failure:
                break
        if results and results[-1].returncode != 0 and args.stop_on_failure:
            break

    payload = {
        "matrix": {
            "family": args.family,
            "batch_sizes": batch_sizes,
            "residency_modes": residency_modes,
            "steps": max(int(args.steps), 1),
            "samples": max(int(args.samples), 1),
            "resolution": max(int(args.resolution), 1),
            "profiles": args.profiles,
            "checkpoint_block_cpu_pinned": bool(args.checkpoint_block_cpu_pinned),
            "checkpoint_streaming": bool(args.checkpoint_streaming),
            "block_prefetch": bool(args.block_prefetch),
            "block_prefetch_depth": max(int(args.block_prefetch_depth), 0),
            "case_timeout_seconds": max(int(args.case_timeout_seconds), 0),
            "stream_output": bool(args.stream_output),
        },
        "results": [case.to_dict() for case in results],
    }
    summary_path = run_root / f"{args.family}_residency_matrix.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[matrix] summary={summary_path}", flush=True)
    return 0 if all(case.returncode == 0 for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
