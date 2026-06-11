"""Small async checkpoint artifact writer used by report-only scorecards."""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


NativeCopyFn = Callable[[Path, Path, int, bool, bool], dict[str, Any]]


class AsyncCheckpointWriter:
    """Single-purpose async writer with atomic target replacement."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        buffer_bytes: int = 1024 * 1024,
        native_copy_fn: NativeCopyFn | None = None,
    ) -> None:
        self.buffer_bytes = max(int(buffer_bytes), 64 * 1024)
        self.native_copy_fn = native_copy_fn
        self._executor = ThreadPoolExecutor(max_workers=max(int(max_workers), 1), thread_name_prefix="lulynx-ckpt")
        self._closed = False
        self._lock = threading.Lock()

    def submit_bytes(self, target: str | Path, payload: bytes, *, fsync: bool = False) -> Future[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            return self._executor.submit(_write_bytes_atomic, Path(target), bytes(payload), bool(fsync))

    def submit_copy(
        self,
        source: str | Path,
        target: str | Path,
        *,
        fsync: bool = False,
        compute_checksum: bool = False,
    ) -> Future[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            return self._executor.submit(
                _copy_atomic,
                Path(source),
                Path(target),
                self.buffer_bytes,
                bool(fsync),
                bool(compute_checksum),
                self.native_copy_fn,
            )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=bool(wait), cancel_futures=False)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AsyncCheckpointWriter is closed")

    def __enter__(self) -> "AsyncCheckpointWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close(wait=True)


def _write_bytes_atomic(target: Path, payload: bytes, fsync: bool) -> dict[str, Any]:
    started = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(target)
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    os.replace(temp, target)
    return {
        "schema_version": 1,
        "ok": target.is_file() and target.stat().st_size == len(payload),
        "provider": "python.async_checkpoint_writer.bytes_atomic",
        "target": _slash(target),
        "bytes_written": target.stat().st_size if target.is_file() else 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
        "fsync": bool(fsync),
        "atomic_temp_path": _slash(temp),
    }


def _copy_atomic(
    source: Path,
    target: Path,
    buffer_bytes: int,
    fsync: bool,
    compute_checksum: bool,
    native_copy_fn: NativeCopyFn | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    if native_copy_fn is not None:
        result = dict(native_copy_fn(source, target, buffer_bytes, fsync, compute_checksum))
        result.setdefault("provider", "native.async_checkpoint_writer.copy_atomic")
        result["async_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
        return result
    temp = _temp_path(target)
    with source.open("rb") as src, temp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=int(buffer_bytes))
        dst.flush()
        if fsync:
            os.fsync(dst.fileno())
    os.replace(temp, target)
    return {
        "schema_version": 1,
        "ok": target.is_file() and target.stat().st_size == source.stat().st_size,
        "provider": "python.async_checkpoint_writer.copy_atomic",
        "input_path": _slash(source),
        "output_path": _slash(target),
        "bytes_copied": target.stat().st_size if target.is_file() else 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
        "buffer_bytes": int(buffer_bytes),
        "fsync": bool(fsync),
        "checksum_computed": False,
        "atomic_temp_path": _slash(temp),
    }


def _temp_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp.{threading.get_ident()}")


def _slash(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = ["AsyncCheckpointWriter", "NativeCopyFn"]
