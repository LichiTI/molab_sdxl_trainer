# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Persistent job-result store for tag analysis and suggestion payloads."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.services.native_module_loader import load_lulynx_native, native_with_entrypoints
from core.services.tag_intelligence_contracts import ANALYSIS_VERSION


def load_native_tag_job_results_api() -> Any:
    return load_lulynx_native()


load_native_tag_job_results_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_tag_job_results_api() -> Any:
    return native_with_entrypoints("scan_tag_job_result_records")


def native_tag_job_result_loader_api() -> Any:
    return native_with_entrypoints("load_tag_job_result_envelope")


def _sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda entry: str(entry.get("finished_at", "")), reverse=True)


class TagJobStore:
    """Persist compact tag analysis/suggestion results under backend-managed data."""

    def __init__(self, base_dir: Optional[Path] = None, keep_per_dataset: int = 5):
        root = Path(__file__).resolve().parents[2]
        self.base_dir = base_dir or (root / "data" / "tag_job_results")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self.keep_per_dataset = keep_per_dataset

    def _dataset_key(self, dataset_path: str) -> str:
        normalized = os.path.normcase(os.path.abspath(str(dataset_path or "")))
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.is_file():
            return {"jobs": {}, "datasets": {}}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"jobs": {}, "datasets": {}}
        except Exception:
            return {"jobs": {}, "datasets": {}}

    def _write_index(self, payload: Dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_result(
        self,
        *,
        kind: str,
        job_id: str,
        dataset_path: str,
        route_family: str,
        submitted_config: Dict[str, Any],
        payload: Dict[str, Any],
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        dataset_key = self._dataset_key(dataset_path)
        dataset_dir = self.base_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        envelope = {
            "version": ANALYSIS_VERSION,
            "job_id": job_id,
            "kind": kind,
            "dataset_path": dataset_path,
            "dataset_key": dataset_key,
            "route_family": route_family,
            "submitted_config": submitted_config,
            "started_at": started_at,
            "finished_at": finished_at or datetime.now().isoformat(),
            "payload_summary": self._payload_summary(payload),
            "payload": payload,
        }
        artifact_path = dataset_dir / f"{kind}_{job_id}.json"
        artifact_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_path = dataset_dir / f"latest_{kind}.json"
        latest_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        self._prune_dataset(dataset_dir, kind)
        index = self._read_index()
        jobs = dict(index.get("jobs", {}))
        jobs[job_id] = str(artifact_path)
        index["jobs"] = jobs
        datasets = dict(index.get("datasets", {}))
        dataset_bucket = dict(datasets.get(dataset_key, {}))
        kind_bucket = list(dataset_bucket.get(kind, []))
        record = {
            "job_id": job_id,
            "kind": kind,
            "route_family": route_family,
            "dataset_path": dataset_path,
            "artifact_path": str(artifact_path),
            "finished_at": envelope["finished_at"],
            "payload_summary": self._payload_summary(payload),
        }
        kind_bucket = [entry for entry in kind_bucket if entry.get("job_id") != job_id]
        kind_bucket.insert(0, record)
        dataset_bucket[kind] = kind_bucket[: self.keep_per_dataset]
        datasets[dataset_key] = dataset_bucket
        index["datasets"] = datasets
        self._write_index(index)
        return envelope

    def _prune_dataset(self, dataset_dir: Path, kind: str) -> None:
        artifacts = sorted(dataset_dir.glob(f"{kind}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in artifacts[self.keep_per_dataset :]:
            stale.unlink(missing_ok=True)

    def load_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        index = self._read_index()
        artifact = Path(str(index.get("jobs", {}).get(job_id, "") or ""))
        if not artifact.is_file():
            return self._load_job_result_native(job_id)
        try:
            return json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            return self._load_job_result_native(job_id)

    def _load_job_result_native(self, job_id: str) -> Optional[Dict[str, Any]]:
        native = native_tag_job_result_loader_api()
        if native is None:
            return None
        try:
            payload = native.load_tag_job_result_envelope(str(self.base_dir), str(job_id or ""))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    def load_latest(self, *, kind: str, dataset_path: str) -> Optional[Dict[str, Any]]:
        dataset_dir = self.base_dir / self._dataset_key(dataset_path)
        latest_path = dataset_dir / f"latest_{kind}.json"
        if not latest_path.is_file():
            return None
        try:
            return json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_dataset_results(self, *, dataset_path: str, kind: str = "") -> List[Dict[str, Any]]:
        index = self._read_index()
        dataset_bucket = dict(index.get("datasets", {}).get(self._dataset_key(dataset_path), {}) or {})
        if kind:
            indexed = list(dataset_bucket.get(kind, []))
            if indexed:
                return _sort_records(indexed)[: self.keep_per_dataset]
            return self._list_dataset_results_native(dataset_path=dataset_path, kind=kind)
        results: List[Dict[str, Any]] = []
        for entries in dataset_bucket.values():
            results.extend(list(entries or []))
        if results:
            return _sort_records(results)
        return self._list_dataset_results_native(dataset_path=dataset_path, kind="")

    def _list_dataset_results_native(self, *, dataset_path: str, kind: str = "") -> List[Dict[str, Any]]:
        native = native_tag_job_results_api()
        if native is None:
            return []
        dataset_dir = self.base_dir / self._dataset_key(dataset_path)
        try:
            payload = native.scan_tag_job_result_records(str(dataset_dir), str(kind or ""), int(self.keep_per_dataset))
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        records: List[Dict[str, Any]] = []
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("job_id", "") or "")
            if not job_id:
                continue
            raw_summary = item.get("payload_summary", {}) or {}
            records.append(
                {
                    "job_id": job_id,
                    "kind": str(item.get("kind", "") or kind or ""),
                    "route_family": str(item.get("route_family", "") or ""),
                    "dataset_path": str(item.get("dataset_path", "") or dataset_path),
                    "artifact_path": str(item.get("artifact_path", "") or ""),
                    "finished_at": str(item.get("finished_at", "") or ""),
                    "payload_summary": dict(raw_summary) if isinstance(raw_summary, dict) else {},
                }
            )
        return records

    def invalidate_dataset(self, dataset_path: str) -> None:
        dataset_dir = self.base_dir / self._dataset_key(dataset_path)
        if not dataset_dir.is_dir():
            return
        for latest in dataset_dir.glob("latest_*.json"):
            latest.unlink(missing_ok=True)
        index = self._read_index()
        dataset_key = self._dataset_key(dataset_path)
        datasets = dict(index.get("datasets", {}))
        if dataset_key in datasets:
            index["datasets"] = datasets
            self._write_index(index)

    def _payload_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "findings" in payload:
            return {
                "finding_count": len(payload.get("findings", []) or []),
                "image_count": int(payload.get("summary", {}).get("image_count", 0) or 0),
            }
        if "suggestions" in payload:
            return {
                "suggestion_count": len(payload.get("suggestions", []) or []),
                "selected_count": int(payload.get("summary", {}).get("selected_count", 0) or 0),
            }
        return {"keys": sorted(payload.keys())[:10]}
