"""Debug-only native descriptor shadow probes for Python dataset boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.turbocore_dataset_staging import (
    _python_plan_dataset_staging,
    _load_native_dataset_staging_handle_api,
)
from core.turbocore_dataset_staging_session import (
    create_native_dataset_descriptor_session,
    destroy_native_dataset_descriptor_session,
    native_dataset_descriptor_session_stats,
    validate_native_dataset_descriptor_session_parity,
)


_SAMPLER_REFERENCE_CACHE: dict[tuple[int, int, bool, bool, int, int, int], Dict[str, Any]] = {}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_signature(value: Any) -> str:
    return hashlib.blake2b(_stable_json(value).encode("utf-8"), digest_size=16).hexdigest()


def _manifest_sample_count(manifest: Dict[str, Any]) -> int:
    samples = manifest.get("samples")
    if isinstance(samples, list):
        return len(samples)
    synthetic = manifest.get("synthetic") if isinstance(manifest.get("synthetic"), dict) else manifest
    return max(int(synthetic.get("sample_count", 0) or 0), 0) if isinstance(synthetic, dict) else 0


def _size_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return max(int(value[0] or 0), 0), max(int(value[1] or 0), 0)
    return 0, 0


def _descriptor_from_sample(sample: Any, index: int) -> Dict[str, Any]:
    image_path = str(getattr(sample, "image_path", "") or "")
    caption_path = str(getattr(sample, "caption_path", "") or "")
    width, height = _size_pair(getattr(sample, "original_size", None))
    bucket_width, bucket_height = _size_pair(getattr(sample, "target_size", None))
    bucket = f"{bucket_width}x{bucket_height}" if bucket_width and bucket_height else "0x0"
    return {
        "id": Path(image_path).stem or f"sample_{index:08}",
        "path": image_path,
        "caption_path": caption_path,
        "width": width,
        "height": height,
        "bucket": bucket,
    }


def build_caption_dataset_descriptor_manifest(
    dataset: Any,
    *,
    max_samples: int | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the descriptor manifest that native probes should match."""

    samples: Iterable[Any] = getattr(dataset, "samples", []) or []
    limit = None if max_samples is None else max(int(max_samples), 0)
    descriptors: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        descriptors.append(_descriptor_from_sample(sample, index))
    return {"samples": descriptors}


def run_caption_dataset_descriptor_shadow_probe(
    dataset: Any,
    *,
    batch_size: int,
    drop_last: bool = False,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    max_samples: int | None = None,
    max_mismatches: int = 8,
) -> Dict[str, Any]:
    """Shadow a Python dataset descriptor list through native session parity."""

    manifest = build_caption_dataset_descriptor_manifest(dataset, max_samples=max_samples)
    session_id = create_native_dataset_descriptor_session(
        manifest,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    try:
        stats = native_dataset_descriptor_session_stats(session_id)
        parity = validate_native_dataset_descriptor_session_parity(
            session_id,
            manifest,
            max_mismatches=max_mismatches,
        )
        return {
            "schema_version": 1,
            "probe": "turbocore_caption_dataset_descriptor_shadow",
            "provider": "native_dataset_descriptor_shadow_probe",
            "native_runtime": True,
            "ok": bool(parity.get("ok", False)),
            "dataset_class": type(dataset).__name__,
            "dataset_length": len(getattr(dataset, "samples", []) or []),
            "descriptor_count": int(stats.get("descriptor_count", 0) or 0),
            "batch_count": int(stats.get("batch_count", 0) or 0),
            "bucket_counts": stats.get("bucket_counts", {}),
            "descriptor_preview": stats.get("descriptor_preview", []),
            "parity_probe": parity,
            "sample_descriptors_owned": True,
            "debug_only": True,
            "shadow_run": True,
            "training_path_enabled": False,
        }
    finally:
        destroy_native_dataset_descriptor_session(session_id)


def _sampler_reference_cache_key(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    prefetch_depth: int,
    chunk_size: int,
) -> tuple[int, int, bool, bool, int, int, int]:
    return (
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        bool(shuffle),
        max(int(seed), 0),
        max(int(prefetch_depth), 1),
        max(int(chunk_size), 1),
    )


def build_dataset_sampler_order_reference(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Build or reuse the Python debug sampler-order reference payload."""

    key = _sampler_reference_cache_key(
        sample_count=sample_count,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    if use_cache and key in _SAMPLER_REFERENCE_CACHE:
        return {**_SAMPLER_REFERENCE_CACHE[key], "reference_cache_hit": True}
    reference = _python_plan_dataset_staging(
        sample_count=key[0],
        batch_size=key[1],
        drop_last=key[2],
        shuffle=key[3],
        seed=key[4],
        prefetch_depth=key[5],
        chunk_size=key[6],
    )
    reference["reference_cache_hit"] = False
    if use_cache:
        _SAMPLER_REFERENCE_CACHE[key] = dict(reference)
    return reference


def validate_native_dataset_sampler_order_shadow(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    use_reference_cache: bool = True,
) -> Dict[str, Any]:
    """Validate native sampler order metadata against the Python debug reference."""

    reference = build_dataset_sampler_order_reference(
        sample_count=sample_count,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
        use_cache=use_reference_cache,
    )
    native = _load_native_dataset_staging_handle_api()
    payload = native.validate_dataset_sampler_order_parity(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        bool(shuffle),
        max(int(seed), 0),
        int(reference.get("index_checksum", 0) or 0),
        json.dumps(reference.get("index_preview", []), separators=(",", ":")),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("native_dataset_sampler_order_parity_failed")
    return {
        **payload,
        "reference_provider": reference.get("provider", "python_dataset_staging"),
        "reference_checksum_kind": reference.get("checksum_kind", ""),
        "reference_cache_hit": bool(reference.get("reference_cache_hit", False)),
    }


def validate_native_dataset_sampler_order_from_reference(
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Run native sampler-order parity against a prebuilt reference payload."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.validate_dataset_sampler_order_parity(
        max(int(reference.get("sample_count", 0) or 0), 0),
        max(int(reference.get("batch_size", 1) or 1), 1),
        bool(reference.get("drop_last", False)),
        bool(reference.get("shuffle", False)),
        max(int(reference.get("seed", 0) or 0), 0),
        int(reference.get("index_checksum", 0) or 0),
        json.dumps(reference.get("index_preview", []), separators=(",", ":")),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("native_dataset_sampler_order_parity_failed")
    return {
        **payload,
        "reference_provider": reference.get("provider", "python_dataset_staging"),
        "reference_checksum_kind": reference.get("checksum_kind", ""),
        "reference_cache_hit": bool(reference.get("reference_cache_hit", False)),
    }


def validate_native_dataset_staging_plan_sampler_order_from_reference(
    plan_id: int,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Run sampler parity against an existing native staging plan handle."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.validate_dataset_staging_plan_sampler_order_parity(
        max(int(plan_id), 0),
        int(reference.get("index_checksum", 0) or 0),
        json.dumps(reference.get("index_preview", []), separators=(",", ":")),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("native_dataset_staging_plan_sampler_order_parity_failed")
    return {
        **payload,
        "reference_provider": reference.get("provider", "python_dataset_staging"),
        "reference_checksum_kind": reference.get("checksum_kind", ""),
        "reference_cache_hit": bool(reference.get("reference_cache_hit", False)),
    }


def run_native_dataset_shadow_lifecycle_from_handles(
    *,
    session_id: int,
    plan_id: int,
    reference_manifest: Dict[str, Any],
    sampler_reference: Dict[str, Any],
    worker_count: int = 2,
    queue_depth: int = 512,
    max_batches_per_submit: int = 256,
) -> Dict[str, Any]:
    """Run the unified descriptor -> sampler -> worker shadow gate in native."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_shadow_lifecycle_probe(
        max(int(session_id), 0),
        max(int(plan_id), 0),
        json.dumps(reference_manifest, ensure_ascii=False, separators=(",", ":")),
        int(sampler_reference.get("index_checksum", 0) or 0),
        json.dumps(sampler_reference.get("index_preview", []), separators=(",", ":")),
        max(int(worker_count), 1),
        max(int(queue_depth), 1),
        max(int(max_batches_per_submit), 1),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("native_dataset_shadow_lifecycle_probe_failed")
    return {
        **payload,
        "reference_provider": sampler_reference.get("provider", "python_dataset_staging"),
        "reference_checksum_kind": sampler_reference.get("checksum_kind", ""),
        "reference_cache_hit": bool(sampler_reference.get("reference_cache_hit", False)),
    }


class NativeDatasetShadowSession:
    """Debug-only owner for reusable native descriptor/session shadow handles."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        *,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        prefetch_depth: int = 512,
        chunk_size: int = 256,
        use_reference_cache: bool = True,
    ) -> None:
        self.manifest = manifest
        self.batch_size = max(int(batch_size), 1)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = max(int(seed), 0)
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.chunk_size = max(int(chunk_size), 1)
        self.sample_count = _manifest_sample_count(manifest)
        self.manifest_signature = _payload_signature(manifest)
        self.sampler_reference = build_dataset_sampler_order_reference(
            sample_count=self.sample_count,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            seed=self.seed,
            prefetch_depth=self.prefetch_depth,
            chunk_size=self.chunk_size,
            use_cache=use_reference_cache,
        )
        self.sampler_signature = _payload_signature(
            {
                "sample_count": self.sample_count,
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "index_checksum": int(self.sampler_reference.get("index_checksum", 0) or 0),
                "index_preview": self.sampler_reference.get("index_preview", []),
            }
        )
        self._native = _load_native_dataset_staging_handle_api()
        self.session_id = create_native_dataset_descriptor_session(
            manifest,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            prefetch_depth=self.prefetch_depth,
            chunk_size=self.chunk_size,
        )
        self.plan_id = int(
            self._native.create_dataset_staging_plan(
                self.sample_count,
                self.batch_size,
                self.drop_last,
                self.shuffle,
                self.seed,
                self.prefetch_depth,
                self.chunk_size,
            )
        )
        if self.plan_id <= 0:
            destroy_native_dataset_descriptor_session(self.session_id)
            self.session_id = 0
            raise RuntimeError("native_dataset_staging_plan_create_failed")
        self.closed = False
        self.run_count = 0

    def __enter__(self) -> "NativeDatasetShadowSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.plan_id:
            self._native.destroy_dataset_staging_plan(self.plan_id)
            self.plan_id = 0
        if self.session_id:
            destroy_native_dataset_descriptor_session(self.session_id)
            self.session_id = 0
        self.closed = True

    def stats(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_shadow_session",
            "provider": "native_dataset_shadow_session_manager",
            "native_runtime": True,
            "ok": not self.closed,
            "session_id": self.session_id,
            "plan_id": self.plan_id,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "prefetch_depth": self.prefetch_depth,
            "chunk_size": self.chunk_size,
            "manifest_signature": self.manifest_signature,
            "sampler_signature": self.sampler_signature,
            "run_count": self.run_count,
            "closed": self.closed,
            "debug_only": True,
            "shadow_run": True,
            "training_path_enabled": False,
        }

    def validate_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        signature = _payload_signature(manifest)
        match = signature == self.manifest_signature
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_shadow_session_manifest_guard",
            "provider": "native_dataset_shadow_session_manager",
            "ok": match,
            "manifest_signature": self.manifest_signature,
            "reference_manifest_signature": signature,
            "requires_rebuild": not match,
            "debug_only": True,
            "shadow_run": True,
            "training_path_enabled": False,
        }

    def run_lifecycle(
        self,
        *,
        reference_manifest: Dict[str, Any] | None = None,
        worker_count: int = 2,
    ) -> Dict[str, Any]:
        if self.closed:
            raise RuntimeError("native_dataset_shadow_session_closed")
        manifest = reference_manifest if reference_manifest is not None else self.manifest
        manifest_guard = self.validate_manifest(manifest)
        if not bool(manifest_guard.get("ok", False)):
            return {
                **self.stats(),
                "ok": False,
                "reason": "reference_manifest_changed",
                "manifest_guard": manifest_guard,
                "requires_rebuild": True,
            }
        payload = run_native_dataset_shadow_lifecycle_from_handles(
            session_id=self.session_id,
            plan_id=self.plan_id,
            reference_manifest=manifest,
            sampler_reference=self.sampler_reference,
            worker_count=worker_count,
            queue_depth=self.prefetch_depth,
            max_batches_per_submit=self.chunk_size,
        )
        self.run_count += 1
        return {
            **payload,
            "shadow_session": self.stats(),
            "manifest_guard": manifest_guard,
            "requires_rebuild": False,
            "session_reused": self.run_count > 1,
        }


def create_caption_dataset_shadow_session(
    dataset: Any,
    *,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    max_samples: int | None = None,
    use_reference_cache: bool = True,
) -> NativeDatasetShadowSession:
    """Create a reusable debug shadow session for a CaptionDataset boundary."""

    return NativeDatasetShadowSession(
        build_caption_dataset_descriptor_manifest(dataset, max_samples=max_samples),
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
        use_reference_cache=use_reference_cache,
    )


def run_caption_dataset_shadow_lifecycle_probe(
    dataset: Any,
    *,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    max_samples: int | None = None,
    worker_count: int = 2,
    use_reference_cache: bool = True,
) -> Dict[str, Any]:
    """Shadow CaptionDataset through descriptor session, sampler plan, and worker preview."""

    manifest = build_caption_dataset_descriptor_manifest(dataset, max_samples=max_samples)
    sample_count = _manifest_sample_count(manifest)
    reference = build_dataset_sampler_order_reference(
        sample_count=sample_count,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
        use_cache=use_reference_cache,
    )
    native = _load_native_dataset_staging_handle_api()
    session_id = create_native_dataset_descriptor_session(
        manifest,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    plan_id = 0
    try:
        plan_id = int(
            native.create_dataset_staging_plan(
                sample_count,
                max(int(batch_size), 1),
                bool(drop_last),
                bool(shuffle),
                max(int(seed), 0),
                max(int(prefetch_depth), 1),
                max(int(chunk_size), 1),
            )
        )
        if plan_id <= 0:
            raise RuntimeError("native_dataset_staging_plan_create_failed")
        payload = run_native_dataset_shadow_lifecycle_from_handles(
            session_id=session_id,
            plan_id=plan_id,
            reference_manifest=manifest,
            sampler_reference=reference,
            worker_count=worker_count,
            queue_depth=prefetch_depth,
            max_batches_per_submit=chunk_size,
        )
        return {
            **payload,
            "dataset_class": type(dataset).__name__,
            "dataset_length": len(getattr(dataset, "samples", []) or []),
            "descriptor_count": sample_count,
            "shuffle": bool(shuffle),
        }
    finally:
        if plan_id:
            native.destroy_dataset_staging_plan(plan_id)
        destroy_native_dataset_descriptor_session(session_id)


__all__ = [
    "build_caption_dataset_descriptor_manifest",
    "build_dataset_sampler_order_reference",
    "create_caption_dataset_shadow_session",
    "NativeDatasetShadowSession",
    "run_caption_dataset_descriptor_shadow_probe",
    "run_caption_dataset_shadow_lifecycle_probe",
    "run_native_dataset_shadow_lifecycle_from_handles",
    "validate_native_dataset_sampler_order_from_reference",
    "validate_native_dataset_sampler_order_shadow",
    "validate_native_dataset_staging_plan_sampler_order_from_reference",
]
