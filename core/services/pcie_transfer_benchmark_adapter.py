"""Compatibility adapter for the PCIe transfer-format benchmark route."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


class PcieTransferBenchmarkError(RuntimeError):
    """Legacy route error with a stable API error code."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class PcieTransferBenchmarkRequest:
    rows: int = 4096
    cols: int = 4096
    shapes: list[str] = field(default_factory=list)
    formats: list[str] = field(
        default_factory=lambda: ["raw_fp16", "raw_bf16", "fp8_e4m3", "int8_rowwise", "uint4_rowwise"]
    )
    batch: int = 16
    compute_dtype: str = "fp16"
    iters: int = 30
    warmup: int = 5
    pack_iters: int = 1
    seed: int = 1234
    no_matmul: bool = False
    timeout_sec: int = 900
    include_tensorcore_decode: bool = False
    tensorcore_decode_iters: int = 3
    tensorcore_decode_warmup: int = 1
    tensorcore_decode_pack_iters: int = 1
    tensorcore_decode_shape_preset: str = ""


CompletedProcessLike = Any
SubprocessRunner = Callable[..., CompletedProcessLike]
ExperimentPlanBuilder = Callable[[Any], Any]
KernelRoadmapBuilder = Callable[[], Any]


def hidden_subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
    return kwargs


def resolve_tool_python_executable(backend_root: Path, *, fallback_executable: str | None = None) -> Path:
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            [
                backend_root / "env" / "python-flashattention" / "python.exe",
                backend_root / "env" / "python_flashattention" / "python.exe",
                backend_root / "env" / "python_launcher" / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                backend_root / "env" / "python-flashattention" / "bin" / "python",
                backend_root / "env" / "python_flashattention" / "bin" / "python",
                backend_root / "env" / "python_launcher" / "bin" / "python",
            ]
        )
    executable = fallback_executable or sys.executable
    try:
        candidates.append(Path(executable).resolve())
    except Exception:
        if executable:
            candidates.append(Path(executable))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower() if os.name == "nt" else str(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return Path(executable)


def normalize_pcie_transfer_benchmark_request(payload: Mapping[str, Any] | Any) -> PcieTransferBenchmarkRequest:
    """Normalize a Pydantic model, dict, or simple object into service params."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        data = {
            name: getattr(payload, name)
            for name in PcieTransferBenchmarkRequest.__dataclass_fields__
            if hasattr(payload, name)
        }

    return PcieTransferBenchmarkRequest(
        rows=int(data.get("rows", 4096)),
        cols=int(data.get("cols", 4096)),
        shapes=_string_list(data.get("shapes", [])),
        formats=_string_list(
            data.get("formats", ["raw_fp16", "raw_bf16", "fp8_e4m3", "int8_rowwise", "uint4_rowwise"])
        ),
        batch=int(data.get("batch", 16)),
        compute_dtype=str(data.get("compute_dtype", "fp16") or "fp16").strip().lower(),
        iters=int(data.get("iters", 30)),
        warmup=int(data.get("warmup", 5)),
        pack_iters=int(data.get("pack_iters", 1)),
        seed=int(data.get("seed", 1234)),
        no_matmul=bool(data.get("no_matmul", False)),
        timeout_sec=int(data.get("timeout_sec", 900)),
        include_tensorcore_decode=bool(data.get("include_tensorcore_decode", False)),
        tensorcore_decode_iters=int(data.get("tensorcore_decode_iters", 3)),
        tensorcore_decode_warmup=int(data.get("tensorcore_decode_warmup", 1)),
        tensorcore_decode_pack_iters=int(data.get("tensorcore_decode_pack_iters", 1)),
        tensorcore_decode_shape_preset=str(data.get("tensorcore_decode_shape_preset", "") or "").strip(),
    )


def run_pcie_transfer_benchmark_payload(
    request: PcieTransferBenchmarkRequest | Mapping[str, Any] | Any,
    *,
    backend_root: Path,
    project_root: Path,
    python_exe: Path,
    hidden_subprocess_kwargs: Mapping[str, Any] | None = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
    experiment_plan_builder: ExperimentPlanBuilder | None = None,
    kernel_roadmap_builder: KernelRoadmapBuilder | None = None,
) -> dict[str, Any]:
    """Run the benchmark and return the legacy ``response.data`` payload."""
    req = request if isinstance(request, PcieTransferBenchmarkRequest) else normalize_pcie_transfer_benchmark_request(request)
    if req.compute_dtype not in {"fp16", "bf16", "fp32"}:
        raise PcieTransferBenchmarkError(
            "compute_dtype must be one of fp16/bf16/fp32",
            "invalid_compute_dtype",
        )

    script_path = backend_root / "core" / "lulynx_trainer" / "pcie_transfer_format_benchmark.py"
    tensorcore_decode_script_path = backend_root / "core" / "lulynx_trainer" / "tensorcore_transfer_kernel_benchmark.py"
    if not script_path.is_file():
        raise PcieTransferBenchmarkError(
            f"PCIe 传输格式 benchmark 脚本不存在: {script_path}",
            "pcie_transfer_benchmark_missing",
        )
    if not python_exe.is_file():
        raise PcieTransferBenchmarkError(
            f"找不到可用的 Python 运行时: {python_exe}",
            "pcie_transfer_benchmark_python_missing",
        )

    if experiment_plan_builder is None:
        from backend.core.lulynx_trainer.transfer_format import transfer_format_experiment_plan

        experiment_plan_builder = transfer_format_experiment_plan
    if kernel_roadmap_builder is None:
        from backend.core.lulynx_trainer.tensorcore_transfer_kernel import tensorcore_kernel_roadmap

        kernel_roadmap_builder = tensorcore_kernel_roadmap

    fd, raw_temp_path = tempfile.mkstemp(prefix="lulynx-pcie-transfer-benchmark-", suffix=".json")
    os.close(fd)
    json_path = Path(raw_temp_path)
    started = time.perf_counter()
    hidden_kwargs = dict(hidden_subprocess_kwargs or {})
    shapes = csv_cli_arg(req.shapes)
    formats = csv_cli_arg(req.formats)

    try:
        cmd = [
            str(python_exe),
            str(script_path),
            "--rows",
            str(req.rows),
            "--cols",
            str(req.cols),
            "--batch",
            str(req.batch),
            "--compute-dtype",
            req.compute_dtype,
            "--iters",
            str(req.iters),
            "--warmup",
            str(req.warmup),
            "--pack-iters",
            str(req.pack_iters),
            "--seed",
            str(req.seed),
            "--json-out",
            str(json_path),
        ]
        if shapes:
            cmd.extend(["--shapes", shapes])
        if formats:
            cmd.extend(["--formats", formats])
        if req.no_matmul:
            cmd.append("--no-matmul")

        completed = _run_subprocess(
            subprocess_runner,
            cmd,
            timeout=max(req.timeout_sec, 30),
            cwd=project_root,
            hidden_kwargs=hidden_kwargs,
            timeout_message=f"PCIe 传输格式 benchmark 超时（>{req.timeout_sec} 秒）",
            timeout_code="pcie_transfer_benchmark_timeout",
            launch_code="pcie_transfer_benchmark_launch_failed",
            launch_prefix="启动 PCIe 传输格式 benchmark 失败",
        )
        if int(getattr(completed, "returncode", 1)) != 0:
            returncode = getattr(completed, "returncode", 1)
            detail = (getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "").strip()
            raise PcieTransferBenchmarkError(
                f"PCIe 传输格式 benchmark 失败: {detail or f'exit code {returncode}'}",
                "pcie_transfer_benchmark_failed",
            )

        try:
            benchmark_payload = load_json_output(stdout=getattr(completed, "stdout", "") or "", json_path=json_path)
        except Exception as exc:
            logger.exception("failed to parse pcie transfer benchmark payload")
            raise PcieTransferBenchmarkError(
                f"PCIe 传输格式 benchmark 结果解析失败: {exc}",
                "pcie_transfer_benchmark_parse_failed",
            ) from exc

        try:
            experiment = experiment_plan_builder(benchmark_payload)
        except Exception as exc:
            logger.exception("failed to parse transfer format experiment plan")
            raise PcieTransferBenchmarkError(
                f"PCIe benchmark 推荐计划生成失败: {exc}",
                "pcie_transfer_benchmark_plan_failed",
            ) from exc

        tensorcore_decode_benchmark, tensorcore_decode_benchmark_error = _run_tensorcore_decode_benchmark(
            req,
            tensorcore_decode_script_path=tensorcore_decode_script_path,
            python_exe=python_exe,
            project_root=project_root,
            shapes=shapes,
            hidden_kwargs=hidden_kwargs,
            subprocess_runner=subprocess_runner,
        )

        return {
            "request": {
                "rows": req.rows,
                "cols": req.cols,
                "shapes": req.shapes,
                "formats": req.formats,
                "batch": req.batch,
                "compute_dtype": req.compute_dtype,
                "iters": req.iters,
                "warmup": req.warmup,
                "pack_iters": req.pack_iters,
                "seed": req.seed,
                "include_matmul": not req.no_matmul,
                "include_tensorcore_decode": req.include_tensorcore_decode,
                "tensorcore_decode_shape_preset": req.tensorcore_decode_shape_preset,
            },
            "benchmark": benchmark_payload,
            "experiment": experiment,
            "tensorcore_transfer_kernel": kernel_roadmap_builder(),
            "tensorcore_decode_benchmark": tensorcore_decode_benchmark,
            "tensorcore_decode_benchmark_error": tensorcore_decode_benchmark_error,
            "runtime": {
                "python": str(python_exe),
                "script": str(script_path),
                "tensorcore_decode_script": str(tensorcore_decode_script_path),
                "timeout_sec": req.timeout_sec,
            },
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    finally:
        _unlink_quietly(json_path)


def csv_cli_arg(value: Sequence[str] | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return ",".join(item for item in _string_list(value) if item)


def load_json_output(*, stdout: str = "", json_path: Path | None = None) -> Any:
    candidates: list[str] = []
    if json_path is not None and json_path.is_file():
        candidates.append(json_path.read_text(encoding="utf-8-sig"))
    text = str(stdout or "").strip()
    if text:
        candidates.append(text)

    last_error: Exception | None = None
    for raw in candidates:
        body = raw.strip()
        if not body:
            continue
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            last_error = exc
            start = body.find("{")
            end = body.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(body[start : end + 1])
                except json.JSONDecodeError as inner_exc:
                    last_error = inner_exc
    if last_error is not None:
        raise ValueError(f"invalid JSON output: {last_error}") from last_error
    raise ValueError("missing JSON output")


def _run_tensorcore_decode_benchmark(
    req: PcieTransferBenchmarkRequest,
    *,
    tensorcore_decode_script_path: Path,
    python_exe: Path,
    project_root: Path,
    shapes: str,
    hidden_kwargs: Mapping[str, Any],
    subprocess_runner: SubprocessRunner,
) -> tuple[Any | None, str]:
    if not req.include_tensorcore_decode:
        return None, ""
    if not tensorcore_decode_script_path.is_file():
        return None, f"TensorCore decode benchmark 脚本不存在: {tensorcore_decode_script_path}"

    fd, raw_tensorcore_temp_path = tempfile.mkstemp(prefix="lulynx-tc-fp8-decode-benchmark-", suffix=".json")
    os.close(fd)
    tensorcore_json_path = Path(raw_tensorcore_temp_path)
    try:
        tensorcore_cmd = [
            str(python_exe),
            str(tensorcore_decode_script_path),
            "--rows",
            str(req.rows),
            "--cols",
            str(req.cols),
            "--compute-dtype",
            req.compute_dtype,
            "--iters",
            str(req.tensorcore_decode_iters),
            "--warmup",
            str(req.tensorcore_decode_warmup),
            "--pack-iters",
            str(req.tensorcore_decode_pack_iters),
            "--seed",
            str(req.seed),
            "--json-out",
            str(tensorcore_json_path),
        ]
        if shapes:
            tensorcore_cmd.extend(["--shapes", shapes])
        elif req.tensorcore_decode_shape_preset:
            tensorcore_cmd.extend(["--shape-preset", req.tensorcore_decode_shape_preset])
        tensorcore_completed = _run_subprocess(
            subprocess_runner,
            tensorcore_cmd,
            timeout=max(30, min(req.timeout_sec, 600)),
            cwd=project_root,
            hidden_kwargs=hidden_kwargs,
            timeout_message="TC FP8 decode 短测超时",
            timeout_code="tensorcore_decode_timeout",
            launch_code="tensorcore_decode_launch_failed",
            launch_prefix="TC FP8 decode 短测失败",
        )
        if int(getattr(tensorcore_completed, "returncode", 1)) != 0:
            returncode = getattr(tensorcore_completed, "returncode", 1)
            detail = (
                getattr(tensorcore_completed, "stderr", "")
                or getattr(tensorcore_completed, "stdout", "")
                or ""
            ).strip()
            return None, f"TC FP8 decode 短测失败: {detail or f'exit code {returncode}'}"
        return load_json_output(stdout=getattr(tensorcore_completed, "stdout", "") or "", json_path=tensorcore_json_path), ""
    except subprocess.TimeoutExpired:
        return None, "TC FP8 decode 短测超时"
    except Exception as exc:
        logger.exception("tensorcore decode benchmark failed")
        return None, f"TC FP8 decode 短测失败: {exc}"
    finally:
        _unlink_quietly(tensorcore_json_path)


def _run_subprocess(
    runner: SubprocessRunner,
    cmd: list[str],
    *,
    timeout: int,
    cwd: Path,
    hidden_kwargs: Mapping[str, Any],
    timeout_message: str,
    timeout_code: str,
    launch_code: str,
    launch_prefix: str,
) -> CompletedProcessLike:
    try:
        return runner(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            cwd=str(cwd),
            **dict(hidden_kwargs),
        )
    except subprocess.TimeoutExpired as exc:
        raise PcieTransferBenchmarkError(timeout_message, timeout_code) from exc
    except PcieTransferBenchmarkError:
        raise
    except Exception as exc:
        logger.exception("failed to launch pcie transfer benchmark")
        raise PcieTransferBenchmarkError(f"{launch_prefix}: {exc}", launch_code) from exc


def _string_list(value: Iterable[Any] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()] if "," in value else ([value.strip()] if value.strip() else [])
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _unlink_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.debug("failed to clean temp file: %s", path)
