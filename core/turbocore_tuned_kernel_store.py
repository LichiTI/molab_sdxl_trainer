# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Registry for tuned turbocore CUDA kernel sources.

Layout (all under ``backend/data/turbocore_tuned_kernels/``)::

    registry.json                                  # index of tuned records
    <gpu_tag>/<kernel>/<numel_bucket>/<source>.cu  # winning variant source
    <gpu_tag>/<kernel>/<numel_bucket>/meta.json    # tune-time evidence
    history/<run_id>.json                          # experiment timeline (chart)

The tuned ``.cu`` keeps the *original* in-tree filename so the directory can
be handed directly to the native NVRTC override env
(``LULYNX_TURBOCORE_KERNEL_SOURCE_OVERRIDE_DIR``).

Matching rule at training time: exact ``gpu_tag`` + exact ``kernel`` +
nearest numel bucket within a 4x ratio, ``enabled`` records only.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_SCHEMA_VERSION = 1
DEFAULT_STORE_RELATIVE = Path("backend") / "data" / "turbocore_tuned_kernels"
NUMEL_MATCH_MAX_RATIO = 4.0


def default_store_root(workspace_root: str | Path | None = None) -> Path:
    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parents[2]
    return root / DEFAULT_STORE_RELATIVE


def gpu_tag_for(device_name: str, sm: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(device_name or "unknown_gpu").lower()).strip("_")
    sm_clean = re.sub(r"[^0-9]+", "", str(sm or ""))
    return f"{name}_sm{sm_clean or 'xx'}"


def numel_bucket(numel: int) -> str:
    """Power-of-two bucket label, e.g. 25M params -> 'n2e25'."""
    n = max(1, int(numel))
    return f"n2e{round(math.log2(n))}"


class TunedKernelStore:
    def __init__(self, store_root: str | Path | None = None, *, workspace_root: str | Path | None = None) -> None:
        self.root = Path(store_root) if store_root else default_store_root(workspace_root)
        self.registry_path = self.root / "registry.json"
        self.history_dir = self.root / "history"

    # -- registry ----------------------------------------------------------

    def load_registry(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "records": []}
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "records": []}
        return payload

    def _save_registry(self, payload: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    def records(self) -> List[Dict[str, Any]]:
        return list(self.load_registry().get("records", []))

    # -- write path (autotune harness) --------------------------------------

    def record_dir(self, gpu_tag: str, kernel: str, numel: int) -> Path:
        return self.root / gpu_tag / kernel / numel_bucket(numel)

    def save_tuned_kernel(
        self,
        *,
        kernel: str,
        source_filename: str,
        source_text: str,
        gpu_tag: str,
        gpu_name: str,
        sm: str,
        numel: int,
        block_size: int,
        elements_per_thread: int,
        baseline_gbps: float,
        best_gbps: float,
        variant_label: str,
        run_id: str,
        created_at: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist the winning source + meta, upsert the registry record."""
        target_dir = self.record_dir(gpu_tag, kernel, numel)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / source_filename).write_text(source_text, encoding="utf-8")
        record = {
            "record_id": f"{gpu_tag}/{kernel}/{numel_bucket(numel)}",
            "kernel": kernel,
            "source_filename": source_filename,
            "gpu_tag": gpu_tag,
            "gpu_name": gpu_name,
            "sm": sm,
            "numel": int(numel),
            "numel_bucket": numel_bucket(numel),
            "block_size": int(block_size),
            "elements_per_thread": int(elements_per_thread),
            "baseline_gbps": float(baseline_gbps),
            "best_gbps": float(best_gbps),
            "speedup": float(best_gbps) / float(baseline_gbps) if baseline_gbps else 0.0,
            "variant_label": variant_label,
            "parity": "bitwise",
            "enabled": True,
            "run_id": run_id,
            "created_at": created_at,
            "dir": str(target_dir),
        }
        meta = dict(record)
        if extra_meta:
            meta["extra"] = extra_meta
        (target_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        registry = self.load_registry()
        records = [r for r in registry.get("records", []) if r.get("record_id") != record["record_id"]]
        records.append(record)
        registry["schema_version"] = REGISTRY_SCHEMA_VERSION
        registry["records"] = records
        self._save_registry(registry)
        return record

    def save_history(self, run_id: str, payload: Dict[str, Any]) -> Path:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        path = self.history_dir / f"{run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_history(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads((self.history_dir / f"{run_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- management (launcher) ----------------------------------------------

    def set_enabled(self, record_id: str, enabled: bool) -> bool:
        registry = self.load_registry()
        hit = False
        for record in registry.get("records", []):
            if record.get("record_id") == record_id:
                record["enabled"] = bool(enabled)
                hit = True
        if hit:
            self._save_registry(registry)
        return hit

    def delete_record(self, record_id: str, *, delete_root: str | Path | None = None) -> bool:
        """Drop the registry entry; move artifacts to .delete/ (keep structure)."""
        registry = self.load_registry()
        kept, dropped = [], []
        for record in registry.get("records", []):
            (dropped if record.get("record_id") == record_id else kept).append(record)
        if not dropped:
            return False
        registry["records"] = kept
        self._save_registry(registry)
        trash_root = Path(delete_root) if delete_root else self.root.parents[2] / ".delete"
        for record in dropped:
            source_dir = Path(record.get("dir", ""))
            if not source_dir.is_dir():
                continue
            try:
                target = trash_root / source_dir.relative_to(self.root.parents[2])
            except ValueError:
                target = trash_root / source_dir.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(source_dir), str(target))
        return True

    # -- read path (training-time resolver) ---------------------------------

    def match_record(self, *, kernel: str, gpu_tag: str, numel: int) -> Optional[Dict[str, Any]]:
        """Best enabled record: exact gpu_tag+kernel, nearest numel within 4x."""
        n = max(1, int(numel))
        best, best_distance = None, None
        for record in self.records():
            if not record.get("enabled", False):
                continue
            if record.get("kernel") != kernel or record.get("gpu_tag") != gpu_tag:
                continue
            recorded = max(1, int(record.get("numel", 0) or 0))
            ratio = max(n, recorded) / min(n, recorded)
            if ratio > NUMEL_MATCH_MAX_RATIO:
                continue
            distance = abs(math.log2(n / recorded))
            if best_distance is None or distance < best_distance:
                best, best_distance = record, distance
        return best


__all__ = [
    "TunedKernelStore",
    "default_store_root",
    "gpu_tag_for",
    "numel_bucket",
    "NUMEL_MATCH_MAX_RATIO",
]
