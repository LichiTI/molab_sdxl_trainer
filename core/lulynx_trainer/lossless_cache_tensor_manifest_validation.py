# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Manifest index validation helpers for research .lynx tensor cache shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import zlib


def _crc32(data: bytes | bytearray | memoryview) -> int:
    return int(zlib.crc32(data) & 0xFFFFFFFF)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def manifest_sample_index_crc32(manifest: dict[str, Any]) -> int:
    return _crc32(
        _json_bytes(
            {
                "shards": [
                    {
                        "shard_index": int(shard.get("shard_index") or 0),
                        "path": str(shard.get("path") or ""),
                        "sample_ids": [str(item) for item in shard.get("sample_ids") or []],
                    }
                    for shard in manifest.get("shards") or []
                ]
            }
        )
    )


def _issue(code: str, *, shard_index: int | None = None, detail: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"code": code}
    if shard_index is not None:
        row["shard_index"] = int(shard_index)
    if detail:
        row["detail"] = detail
    return row


def validate_manifest_index(
    manifest_path: str | Path,
    *,
    manifest_format_name: str,
    inspect_container: Callable[[bytes], dict[str, Any]],
    strict: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    if manifest.get("format") != manifest_format_name or int(manifest.get("version") or 0) != 1:
        issues.append(_issue("unsupported_manifest_version"))

    expected_crc = manifest.get("sample_index_crc32")
    actual_crc = manifest_sample_index_crc32(manifest)
    checksum_present = expected_crc is not None
    if checksum_present and int(expected_crc or 0) != actual_crc:
        issues.append(_issue("sample_index_checksum_mismatch"))

    seen_samples: dict[str, int] = {}
    shard_rows = manifest.get("shards") or []
    for fallback_index, shard in enumerate(shard_rows):
        shard_index = int(shard.get("shard_index") if shard.get("shard_index") is not None else fallback_index)
        shard_path = Path(str(shard.get("path") or ""))
        if not shard_path.is_absolute():
            candidate = path.parent / shard_path
            shard_path = candidate if candidate.exists() else shard_path
        manifest_sample_ids = [str(item) for item in shard.get("sample_ids") or []]
        for sample_id in manifest_sample_ids:
            previous = seen_samples.get(sample_id)
            if previous is not None:
                issues.append(
                    _issue(
                        "duplicate_manifest_sample_id",
                        shard_index=shard_index,
                        detail=f"{sample_id} already mapped to shard {previous}",
                    )
                )
            seen_samples[sample_id] = shard_index
        if not shard_path.is_file():
            issues.append(_issue("shard_file_missing", shard_index=shard_index, detail=str(shard_path)))
            continue
        try:
            header = inspect_container(shard_path.read_bytes())
        except Exception as exc:
            issues.append(
                _issue("shard_header_invalid", shard_index=shard_index, detail=f"{type(exc).__name__}: {exc}")
            )
            continue
        header_sample_ids = [
            str(sample.get("sample_id") or "")
            for sample in header.get("samples") or []
            if sample.get("sample_id")
        ]
        if header_sample_ids != manifest_sample_ids:
            issues.append(_issue("sample_key_parity_mismatch", shard_index=shard_index))
        for field in ("sample_count", "tensor_count", "total_raw_size", "total_encoded_size"):
            if int(header.get(field) or 0) != int(shard.get(field) or 0):
                issues.append(_issue(f"{field}_mismatch", shard_index=shard_index))

    report = {
        "ok": not issues,
        "manifest_path": str(path),
        "manifest_index_validation_ready": not issues,
        "checksum_present": checksum_present,
        "sample_index_crc32": actual_crc,
        "declared_sample_index_crc32": int(expected_crc or 0) if checksum_present else 0,
        "shard_count": len(shard_rows),
        "sample_count": len(seen_samples),
        "issue_count": len(issues),
        "issues": issues,
        "training_path_enabled": False,
        "resource_center_allowed": False,
    }
    if strict and issues:
        codes = ", ".join(str(issue.get("code") or "") for issue in issues)
        raise ValueError(f"LYNX_TENSOR_CACHE manifest index validation failed: {codes}")
    return report


__all__ = ["manifest_sample_index_crc32", "validate_manifest_index"]
