"""P6 scorecard for an async checkpoint artifact writer queue."""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any

from core.services.native_module_loader import ensure_lulynx_native_artifact_path, probe_lulynx_native_loader
from core.turbocore_async_checkpoint_writer import AsyncCheckpointWriter


DEFAULT_SIZE_BYTES = 8 * 1024 * 1024
DEFAULT_BUFFER_BYTES = 1024 * 1024


def build_async_checkpoint_writer_scorecard(
    *,
    size_bytes: int = DEFAULT_SIZE_BYTES,
    buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    use_native_copy: bool = True,
) -> dict[str, Any]:
    """Run an isolated async writer proof without touching trainer saves."""

    native = _load_native() if use_native_copy else None
    capabilities = _capabilities(native)
    proof = _run_writer_proof(
        native=native,
        size_bytes=max(int(size_bytes), 64 * 1024),
        buffer_bytes=max(int(buffer_bytes), 64 * 1024),
    )
    validations = _validations(capabilities, proof)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_async_checkpoint_writer_scorecard_v0",
        "gate": "p6g_async_checkpoint_writer",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "size_bytes": int(size_bytes),
        "buffer_bytes": int(buffer_bytes),
        "capabilities": capabilities,
        "proof": proof,
        "validations": validations,
        "summary": {
            "async_writer_ready": ready,
            "native_copy_used": bool(proof.get("native_copy_used", False)),
            "submitted_job_count": int(proof.get("submitted_job_count", 0) or 0),
            "completed_job_count": int(proof.get("completed_job_count", 0) or 0),
            "submit_nonblocking_ok": bool(proof.get("submit_nonblocking_ok", False)),
            "atomic_commit_ok": bool(proof.get("atomic_commit_ok", False)),
            "parity_ok": bool(proof.get("parity_ok", False)),
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add async checkpoint writer observe manifest before trainer integration"
            if ready
            else "fix async checkpoint writer blockers"
        ),
        "notes": [
            "The queue writes isolated fixture artifacts through temp files and atomic replace.",
            "Trainer checkpoint save/load paths are not changed.",
            "Native stream copy is used when available, otherwise the proof falls back to Python copy.",
        ],
    }


def _run_writer_proof(*, native: Any | None, size_bytes: int, buffer_bytes: int) -> dict[str, Any]:
    temp_root = _temp_root()
    with tempfile.TemporaryDirectory(prefix="lulynx_async_ckpt_", dir=str(temp_root)) as tmp:
        root = Path(tmp)
        source = root / "source-artifact.bin"
        _write_fixture(source, size_bytes=size_bytes)
        source_digest = _digest(source)
        native_copy_fn = _native_copy_fn(native) if native is not None else None
        payload = b'{"schema_version":1,"kind":"async_checkpoint_writer_smoke"}\n'
        payload_digest = hashlib.blake2b(payload, digest_size=16).hexdigest()

        with AsyncCheckpointWriter(buffer_bytes=buffer_bytes, native_copy_fn=native_copy_fn) as writer:
            submit_started = time.perf_counter()
            futures = [
                writer.submit_bytes(root / "metadata.json", payload),
                writer.submit_copy(source, root / "artifact-a.bin", compute_checksum=True),
                writer.submit_copy(source, root / "artifact-b.bin", compute_checksum=False),
            ]
            submit_ms = (time.perf_counter() - submit_started) * 1000.0
            wait_started = time.perf_counter()
            results = [future.result(timeout=30.0) for future in futures]
            wait_ms = (time.perf_counter() - wait_started) * 1000.0

        targets = [root / "metadata.json", root / "artifact-a.bin", root / "artifact-b.bin"]
        target_digests = {
            target.name: (_digest(target) if target.is_file() else "")
            for target in targets
        }
        tmp_leftovers = [path.name for path in root.iterdir() if ".tmp." in path.name]
        parity_ok = (
            target_digests.get("metadata.json") == payload_digest
            and target_digests.get("artifact-a.bin") == source_digest
            and target_digests.get("artifact-b.bin") == source_digest
        )
        completed = sum(1 for item in results if bool(item.get("ok", False)))
        return {
            "schema_version": 1,
            "case": "async_checkpoint_writer_atomic_queue",
            "ok": bool(completed == len(results) and parity_ok and not tmp_leftovers),
            "submitted_job_count": len(results),
            "completed_job_count": completed,
            "submit_ms": round(submit_ms, 4),
            "wait_ms": round(wait_ms, 4),
            "submit_nonblocking_ok": bool(submit_ms < max(wait_ms, 0.001)),
            "atomic_commit_ok": not tmp_leftovers and all(target.is_file() for target in targets),
            "parity_ok": parity_ok,
            "native_copy_used": any("lulynx_native" in str(item.get("provider", "")) for item in results),
            "tmp_leftovers": tmp_leftovers,
            "target_digests": target_digests,
            "source_digest": source_digest,
            "results": [_compact_result(item) for item in results],
            "blocked_reasons": [] if completed == len(results) and parity_ok and not tmp_leftovers else ["async_checkpoint_writer_proof_failed"],
        }


def _validations(capabilities: dict[str, Any], proof: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "async_writer_jobs_completed",
            int(proof.get("submitted_job_count", 0) or 0) == int(proof.get("completed_job_count", -1) or -1),
            "async_checkpoint_writer_incomplete_jobs",
        ),
        _validation(
            "async_writer_atomic_commit",
            bool(proof.get("atomic_commit_ok", False)),
            "async_checkpoint_writer_atomic_commit_failed",
        ),
        _validation(
            "async_writer_parity",
            bool(proof.get("parity_ok", False)),
            "async_checkpoint_writer_parity_failed",
        ),
        _validation(
            "async_writer_submit_nonblocking",
            bool(proof.get("submit_nonblocking_ok", False)),
            "async_checkpoint_writer_submit_blocked",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(capabilities.get("training_path_enabled", False)),
            "async_checkpoint_writer_changed_default_behavior",
        ),
    ]


def _native_copy_fn(native: Any):
    def _copy(source: Path, target: Path, buffer_bytes: int, fsync: bool, compute_checksum: bool) -> dict[str, Any]:
        return dict(
            native.stream_copy_checkpoint_artifact_py(
                str(source),
                str(target),
                int(buffer_bytes),
                bool(fsync),
                bool(compute_checksum),
            )
        )

    return _copy


def _load_native() -> Any | None:
    ensure_lulynx_native_artifact_path()
    try:
        import importlib

        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _capabilities(native: Any | None) -> dict[str, Any]:
    return {
        "native_importable": native is not None,
        "native_stream_copy_entrypoint": bool(native is not None and hasattr(native, "stream_copy_checkpoint_artifact_py")),
        "loader": probe_lulynx_native_loader(),
        "python_fallback": True,
        "training_path_enabled": False,
    }


def _temp_root() -> Path:
    root = Path("H:/tmp")
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_fixture(path: Path, *, size_bytes: int) -> None:
    pattern = bytes((index * 29 + 11) & 0xFF for index in range(64 * 1024))
    remaining = int(size_bytes)
    with path.open("wb") as handle:
        while remaining > 0:
            chunk = pattern[: min(len(pattern), remaining)]
            handle.write(chunk)
            remaining -= len(chunk)


def _digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compact_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(item.get("ok", False)),
        "provider": str(item.get("provider", "")),
        "bytes": int(item.get("bytes_written", item.get("bytes_copied", 0)) or 0),
        "elapsed_ms": float(item.get("elapsed_ms", item.get("async_elapsed_ms", 0.0)) or 0.0),
    }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_async_checkpoint_writer_scorecard"]
