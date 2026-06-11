"""P6 scorecard for Rust-backed checkpoint artifact streaming I/O."""

from __future__ import annotations

import shutil
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any

from core.services.native_module_loader import (
    ensure_lulynx_native_artifact_path,
    probe_lulynx_native_loader,
)


DEFAULT_SIZE_BYTES = 32 * 1024 * 1024
DEFAULT_BUFFER_BYTES = 1024 * 1024
DEFAULT_REPEATS = 3
MIN_NATIVE_COPY_SPEEDUP = 1.0


def build_checkpoint_artifact_streaming_scorecard(
    *,
    size_bytes: int = DEFAULT_SIZE_BYTES,
    buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Compare a native Rust stream-copy primitive against Python copy I/O."""

    native = _load_native()
    capabilities = _capabilities(native)
    blockers: list[str] = []
    if native is None:
        blockers.append("lulynx_native_not_importable")
    elif not capabilities["native_entrypoint"]:
        blockers.append("checkpoint_artifact_streaming_entrypoint_missing")

    benchmark: dict[str, Any]
    parity: dict[str, Any]
    if blockers:
        benchmark = _blocked_case("native_checkpoint_artifact_streaming_unavailable")
        parity = _blocked_case("native_checkpoint_artifact_streaming_unavailable")
    else:
        try:
            benchmark, parity = _run_streaming_case(
                native,
                size_bytes=max(int(size_bytes), 1024),
                buffer_bytes=max(int(buffer_bytes), 64 * 1024),
                repeats=max(int(repeats), 1),
            )
        except Exception as exc:
            benchmark = _case_error("checkpoint_artifact_streaming_case", exc)
            parity = _case_error("checkpoint_artifact_streaming_case", exc, parity_ok=False)

    if not bool(parity.get("parity_ok", False)):
        blockers.append("checkpoint_artifact_streaming_parity_failed")
    if not bool(benchmark.get("ok", False)):
        blockers.append("checkpoint_artifact_streaming_benchmark_failed")
    speedup = _float_or_none(benchmark.get("native_speedup_vs_python_copy"))
    if speedup is not None and int(size_bytes) >= DEFAULT_SIZE_BYTES and speedup < MIN_NATIVE_COPY_SPEEDUP:
        blockers.append("checkpoint_artifact_streaming_speedup_below_1x")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_checkpoint_artifact_streaming_scorecard_v0",
        "gate": "p6_checkpoint_artifact_streaming_io",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": ready,
        "experimental_only": True,
        "default_behavior_changed": False,
        "size_bytes": int(size_bytes),
        "buffer_bytes": int(buffer_bytes),
        "repeats": int(repeats),
        "capabilities": capabilities,
        "parity": parity,
        "benchmark": benchmark,
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _run_streaming_case(
    native: Any,
    *,
    size_bytes: int,
    buffer_bytes: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temp_root = _temp_root()
    with tempfile.TemporaryDirectory(prefix="lulynx_checkpoint_stream_", dir=str(temp_root)) as tmp:
        root = Path(tmp)
        source = root / "source-checkpoint-artifact.bin"
        _write_fixture(source, size_bytes=size_bytes)
        source_checksum = _file_digest(source, buffer_bytes=buffer_bytes)

        native_runs, python_runs = _benchmark_copy_routes(
            native,
            source=source,
            root=root,
            buffer_bytes=buffer_bytes,
            repeats=repeats,
        )
        python_out = root / "python-copy-verify.bin"
        native_out = root / "native-copy-verify.bin"
        native_checked_out = root / "native-copy-checked.bin"
        python_copy = _python_copy(source, python_out, buffer_bytes=buffer_bytes)
        native_copy = _native_copy(native, source, native_out, buffer_bytes=buffer_bytes, compute_checksum=False)
        native_checked = _native_copy(native, source, native_checked_out, buffer_bytes=buffer_bytes, compute_checksum=True)

        native_checksum = _file_digest(native_out, buffer_bytes=buffer_bytes)
        native_checked_checksum = _file_digest(native_checked_out, buffer_bytes=buffer_bytes)
        python_checksum = _file_digest(python_out, buffer_bytes=buffer_bytes)

        native_elapsed = _median([float(item.get("elapsed_ms", 0.0) or 0.0) for item in native_runs])
        python_elapsed = _median([float(item.get("elapsed_ms", 0.0) or 0.0) for item in python_runs])
        parity_ok = bool(
            native_copy.get("ok", False)
            and python_copy.get("ok", False)
            and native_checked.get("ok", False)
            and int(native_copy.get("bytes_copied", 0) or 0) == size_bytes
            and int(native_checked.get("bytes_copied", 0) or 0) == size_bytes
            and int(python_copy.get("bytes_copied", 0) or 0) == size_bytes
            and bool(native_checked.get("checksum_computed", False))
            and str(native_checked.get("checksum_fnv1a64", ""))
            and native_checksum == source_checksum
            and native_checked_checksum == source_checksum
            and python_checksum == source_checksum
        )
        benchmark = {
            "schema_version": 1,
            "benchmark": "checkpoint_artifact_streaming_copy_v0",
            "ok": bool(native_copy.get("ok", False) and python_copy.get("ok", False)),
            "size_bytes": int(size_bytes),
            "buffer_bytes": int(buffer_bytes),
            "repeats": int(repeats),
            "native_stream_copy_ms": round(native_elapsed, 4),
            "python_copy_ms": round(python_elapsed, 4),
            "native_speedup_vs_python_copy": round(python_elapsed / max(native_elapsed, 1e-6), 4),
            "native": _compact_copy(native_copy),
            "python": _compact_copy(python_copy),
            "native_runs_ms": [round(float(item.get("elapsed_ms", 0.0) or 0.0), 4) for item in native_runs],
            "python_runs_ms": [round(float(item.get("elapsed_ms", 0.0) or 0.0), 4) for item in python_runs],
        }
        parity = {
            "schema_version": 1,
            "case": "checkpoint_artifact_streaming_copy_parity",
            "ok": True,
            "parity_ok": parity_ok,
            "source_checksum_blake2b": source_checksum,
            "native_output_checksum_blake2b": native_checksum,
            "native_checked_output_checksum_blake2b": native_checked_checksum,
            "python_output_checksum_blake2b": python_checksum,
            "native_report_checksum_fnv1a64": str(native_checked.get("checksum_fnv1a64", "")),
            "size_bytes": int(size_bytes),
            "blocked_reasons": [] if parity_ok else ["checkpoint_artifact_streaming_checksum_or_size_mismatch"],
        }
        return benchmark, parity


def _benchmark_copy_routes(
    native: Any,
    *,
    source: Path,
    root: Path,
    buffer_bytes: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    native_runs: list[dict[str, Any]] = []
    python_runs: list[dict[str, Any]] = []
    for index in range(max(int(repeats), 1)):
        native_target = root / f"native-copy-bench-{index}.bin"
        python_target = root / f"python-copy-bench-{index}.bin"
        if index % 2 == 0:
            native_runs.append(_native_copy(native, source, native_target, buffer_bytes=buffer_bytes, compute_checksum=False))
            python_runs.append(_python_copy(source, python_target, buffer_bytes=buffer_bytes))
        else:
            python_runs.append(_python_copy(source, python_target, buffer_bytes=buffer_bytes))
            native_runs.append(_native_copy(native, source, native_target, buffer_bytes=buffer_bytes, compute_checksum=False))
    return native_runs, python_runs


def _native_copy(
    native: Any,
    source: Path,
    target: Path,
    *,
    buffer_bytes: int,
    compute_checksum: bool,
) -> dict[str, Any]:
    return native.stream_copy_checkpoint_artifact_py(
        str(source),
        str(target),
        int(buffer_bytes),
        False,
        bool(compute_checksum),
    )


def _load_native() -> Any | None:
    ensure_lulynx_native_artifact_path()
    try:
        import importlib

        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _capabilities(native: Any | None) -> dict[str, Any]:
    loader = probe_lulynx_native_loader()
    return {
        "native_importable": native is not None,
        "native_entrypoint": bool(native is not None and hasattr(native, "stream_copy_checkpoint_artifact_py")),
        "loader": loader,
    }


def _temp_root() -> Path:
    root = Path("H:/tmp")
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_fixture(path: Path, *, size_bytes: int) -> None:
    pattern = bytes((index * 37 + 17) & 0xFF for index in range(64 * 1024))
    remaining = int(size_bytes)
    with path.open("wb") as handle:
        while remaining > 0:
            chunk = pattern[: min(len(pattern), remaining)]
            handle.write(chunk)
            remaining -= len(chunk)


def _python_copy(source: Path, target: Path, *, buffer_bytes: int) -> dict[str, Any]:
    started = time.perf_counter()
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=int(buffer_bytes))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ok": target.is_file() and target.stat().st_size == source.stat().st_size,
        "provider": "python.shutil.copyfileobj",
        "bytes_copied": target.stat().st_size if target.is_file() else 0,
        "elapsed_ms": elapsed_ms,
        "buffer_bytes": int(buffer_bytes),
    }


def _file_digest(path: Path, *, buffer_bytes: int) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(int(buffer_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compact_copy(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(payload.get("ok", False)),
        "provider": str(payload.get("provider", "")),
        "bytes_copied": int(payload.get("bytes_copied", 0) or 0),
        "elapsed_ms": round(float(payload.get("elapsed_ms", 0.0) or 0.0), 4),
        "buffer_bytes": int(payload.get("buffer_bytes", 0) or 0),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _blocked_case(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "parity_ok": False,
        "blocked_reasons": [reason],
    }


def _case_error(case: str, exc: Exception, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "ok": False,
        "case": case,
        "error": f"{type(exc).__name__}: {exc}",
        "blocked_reasons": [f"{case}_failed:{type(exc).__name__}"],
    }
    payload.update(extra)
    return payload


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_checkpoint_artifact_streaming_scorecard"]
