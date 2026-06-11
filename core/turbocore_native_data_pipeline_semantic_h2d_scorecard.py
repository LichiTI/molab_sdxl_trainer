"""Semantic parity and H2D ownership matrix for native TurboCore data pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.services.native_module_loader import ensure_lulynx_native_artifact_path
from core.turbocore_dataset_staging import plan_dataset_staging
from core.turbocore_dataset_staging_session import run_native_dataset_descriptor_session_probe
from core.turbocore_native_data_pipeline_adapter_shadow_scorecard import (
    build_native_data_pipeline_adapter_shadow_scorecard,
)


FEATURE = "native_data_pipeline"
MATRIX_KIND = "native_data_pipeline_semantic_h2d_matrix_v0"


def build_native_data_pipeline_semantic_h2d_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build report-only semantic and transfer-ownership evidence."""

    ensure_lulynx_native_artifact_path()
    adapter = dict(
        adapter_report
        or build_native_data_pipeline_adapter_shadow_scorecard(
            native_training_mode=native_training_mode,
        )
    )
    semantic_matrix = _semantic_parity_matrix()
    descriptor_parity = _descriptor_semantic_parity()
    h2d_contract = _h2d_ownership_contract()
    validations = _validations(adapter, semantic_matrix, descriptor_parity, h2d_contract)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_semantic_h2d_scorecard_v0",
        "gate": "p6j_native_data_pipeline_semantic_h2d",
        "ok": ready,
        "promotion_ready": ready,
        "semantic_h2d_matrix_ready": ready,
        "semantic_parity_matrix_ready": bool(semantic_matrix.get("ok", False))
        and bool(descriptor_parity.get("ok", False)),
        "h2d_ownership_contract_ready": bool(h2d_contract.get("ok", False)),
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
        "feature": FEATURE,
        "matrix_kind": MATRIX_KIND,
        "native_training_mode": str(adapter.get("native_training_mode") or native_training_mode),
        "adapter_summary": dict(adapter.get("summary") or {}),
        "semantic_matrix": semantic_matrix,
        "descriptor_parity": descriptor_parity,
        "h2d_ownership_contract": h2d_contract,
        "validations": validations,
        "summary": {
            "semantic_h2d_matrix_ready": ready,
            "sampler_case_count": int(semantic_matrix.get("case_count", 0) or 0),
            "sampler_failed_case_count": int(semantic_matrix.get("failed_case_count", 0) or 0),
            "descriptor_parity_ok": bool(descriptor_parity.get("ok", False)),
            "h2d_provider": str(h2d_contract.get("provider", "")),
            "h2d_copy_independent": bool(h2d_contract.get("copy_independent", False)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add native data pipeline end-to-end shadow before canary dispatch"
            if ready
            else "fix native data pipeline semantic/H2D blockers"
        ),
        "notes": [
            "This matrix proves sampler metadata and descriptor semantics only.",
            "H2D ownership is validated as a contract; no trainer data transfer is replaced.",
            "StandardCore remains authoritative until end-to-end shadow and rollback gates pass.",
        ],
    }


def _semantic_parity_matrix() -> dict[str, Any]:
    cases = [
        {
            "name": "sequential_keep_remainder",
            "sample_count": 10,
            "batch_size": 4,
            "drop_last": False,
            "shuffle": False,
            "seed": 7,
            "chunk_size": 2,
        },
        {
            "name": "sequential_drop_last",
            "sample_count": 10,
            "batch_size": 4,
            "drop_last": True,
            "shuffle": False,
            "seed": 7,
            "chunk_size": 8,
        },
        {
            "name": "shuffle_drop_last_native",
            "sample_count": 33,
            "batch_size": 5,
            "drop_last": True,
            "shuffle": True,
            "seed": 77,
            "chunk_size": 3,
        },
        {
            "name": "shuffle_keep_remainder_native",
            "sample_count": 35,
            "batch_size": 6,
            "drop_last": False,
            "shuffle": True,
            "seed": 91,
            "chunk_size": 4,
        },
    ]
    rows = [_sampler_case(case) for case in cases]
    failed = [row for row in rows if not bool(row.get("ok", False))]
    return {
        "schema_version": 1,
        "matrix": MATRIX_KIND,
        "case_count": len(rows),
        "failed_case_count": len(failed),
        "ok": not failed,
        "training_path_enabled": False,
        "cases": rows,
        "blocked_reasons": _dedupe(
            [reason for row in failed for reason in row.get("blocked_reasons", []) or []]
        ),
    }


def _sampler_case(case: Mapping[str, Any]) -> dict[str, Any]:
    name = str(case["name"])
    common = {
        "sample_count": int(case["sample_count"]),
        "batch_size": int(case["batch_size"]),
        "drop_last": bool(case["drop_last"]),
        "shuffle": bool(case["shuffle"]),
        "seed": int(case["seed"]),
        "prefetch_depth": 8,
        "chunk_size": int(case["chunk_size"]),
    }
    reference = plan_dataset_staging(prefer_native=False, **common)
    candidate = plan_dataset_staging(prefer_native=True, **common)
    compared_fields = [
        "batch_count",
        "covered_samples",
        "dropped_samples",
        "chunk_count",
        "index_checksum",
        "index_preview",
    ]
    mismatches = [
        field
        for field in compared_fields
        if reference.get(field) != candidate.get(field)
    ]
    chunk_shape_ok = _chunk_shapes(reference.get("chunks")) == _chunk_shapes(candidate.get("chunks"))
    native_expected = bool(common["shuffle"])
    native_ok = (not native_expected) or bool(candidate.get("native_runtime", False))
    ok = bool(reference.get("ok", False)) and bool(candidate.get("ok", False)) and not mismatches and chunk_shape_ok and native_ok
    blockers: list[str] = []
    if mismatches:
        blockers.append(f"{name}_sampler_field_mismatch")
    if not chunk_shape_ok:
        blockers.append(f"{name}_chunk_shape_mismatch")
    if not native_ok:
        blockers.append(f"{name}_native_shuffle_plan_missing")
    return {
        "schema_version": 1,
        "case": name,
        "ok": ok,
        "provider": str(candidate.get("provider", "")),
        "native_runtime": bool(candidate.get("native_runtime", False)),
        "native_required": native_expected,
        "reference_provider": str(reference.get("provider", "")),
        "batch_count": int(candidate.get("batch_count", 0) or 0),
        "covered_samples": int(candidate.get("covered_samples", 0) or 0),
        "dropped_samples": int(candidate.get("dropped_samples", 0) or 0),
        "index_checksum": int(candidate.get("index_checksum", 0) or 0),
        "index_preview": list(candidate.get("index_preview", []) or [])[:8],
        "mismatched_fields": mismatches,
        "chunk_shape_ok": chunk_shape_ok,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
    }


def _descriptor_semantic_parity() -> dict[str, Any]:
    manifest = _descriptor_manifest()
    probe = run_native_dataset_descriptor_session_probe(
        manifest,
        batch_size=2,
        drop_last=False,
        prefetch_depth=4,
        chunk_size=2,
        epochs=2,
    )
    stats = probe.get("initial_stats") if isinstance(probe.get("initial_stats"), Mapping) else {}
    preview = stats.get("descriptor_preview") if isinstance(stats.get("descriptor_preview"), list) else []
    first = preview[0] if preview and isinstance(preview[0], Mapping) else {}
    bucket_counts = stats.get("bucket_counts") if isinstance(stats.get("bucket_counts"), Mapping) else {}
    ok = bool(probe.get("ok", False)) and bool(probe.get("descriptor_parity_ok", False)) and first.get("id") == "sample_0001" and dict(bucket_counts) == {"512x768": 2, "768x512": 1}
    return {
        "schema_version": 1,
        "case": "descriptor_semantic_parity",
        "ok": ok,
        "provider": str(probe.get("provider", "")),
        "native_runtime": bool(probe.get("native_runtime", False)),
        "descriptor_count": int(probe.get("descriptor_count", 0) or 0),
        "batch_count": int(probe.get("batch_count", 0) or 0),
        "sample_descriptors_owned": bool(probe.get("sample_descriptors_owned", False)),
        "worker_results_owned": bool(probe.get("worker_results_owned", False)),
        "descriptor_parity_ok": bool(probe.get("descriptor_parity_ok", False)),
        "bucket_counts": dict(bucket_counts),
        "first_descriptor_id": str(first.get("id", "")),
        "training_path_enabled": False,
        "blocked_reasons": [] if ok else ["native_data_pipeline_descriptor_semantic_parity_failed"],
    }


def _h2d_ownership_contract() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is expected in trainer env
        return {
            "schema_version": 1,
            "case": "h2d_ownership_contract",
            "ok": False,
            "provider": "torch_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": ["native_data_pipeline_h2d_torch_unavailable"],
            "training_path_enabled": False,
        }

    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    original = source.clone()
    provider = "cpu_clone_reference"
    pinned = False
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"
        try:
            source_for_copy = source.pin_memory()
            pinned = True
        except RuntimeError:
            source_for_copy = source
        copied = source_for_copy.to("cuda", non_blocking=pinned)
        torch.cuda.synchronize()
        provider = "torch_h2d_copy_contract"
    else:
        source_for_copy = source
        copied = source_for_copy.clone()

    source.add_(1000.0)
    if device_type == "cuda":
        copied_back = copied.detach().cpu()
    else:
        copied_back = copied
    copy_independent = bool(torch.equal(copied_back, original))
    source_mutated_after_copy = bool(not torch.equal(source, original))
    ok = copy_independent and source_mutated_after_copy
    return {
        "schema_version": 1,
        "case": "h2d_ownership_contract",
        "ok": ok,
        "provider": provider,
        "device_type": device_type,
        "pinned_memory_used": pinned,
        "source_owner": "standardcore_python_data_path",
        "device_copy_owner": "training_step_after_explicit_handoff",
        "native_pipeline_owns_device_tensor": False,
        "copy_independent": copy_independent,
        "source_mutated_after_copy": source_mutated_after_copy,
        "requires_explicit_release": True,
        "async_copy_requires_stream_sync": device_type == "cuda",
        "training_path_enabled": False,
        "blocked_reasons": [] if ok else ["native_data_pipeline_h2d_ownership_contract_failed"],
    }


def _validations(
    adapter: Mapping[str, Any],
    semantic_matrix: Mapping[str, Any],
    descriptor_parity: Mapping[str, Any],
    h2d_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p6i_adapter_shadow_ready",
            bool(adapter.get("adapter_shadow_ready", False)),
            "native_data_pipeline_adapter_shadow_missing",
        ),
        _validation(
            "sampler_semantic_parity_matrix",
            bool(semantic_matrix.get("ok", False)),
            "native_data_pipeline_sampler_semantic_parity_failed",
        ),
        _validation(
            "descriptor_semantic_parity",
            bool(descriptor_parity.get("ok", False)),
            "native_data_pipeline_descriptor_semantic_parity_failed",
        ),
        _validation(
            "h2d_ownership_contract",
            bool(h2d_contract.get("ok", False))
            and not bool(h2d_contract.get("native_pipeline_owns_device_tensor", True)),
            "native_data_pipeline_h2d_ownership_contract_failed",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(adapter.get("runtime_dispatch_ready", True))
            and not bool(adapter.get("native_dispatch_allowed", True))
            and not bool(adapter.get("training_path_enabled", True)),
            "native_data_pipeline_semantic_h2d_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(adapter.get("training_path_enabled", True))
            and not bool(adapter.get("default_behavior_changed", True)),
            "native_data_pipeline_semantic_h2d_changed_default_behavior",
        ),
    ]


def _descriptor_manifest() -> dict[str, Any]:
    return {
        "samples": [
            {
                "id": "sample_0001",
                "path": "samples/sample_0001.png",
                "caption_path": "samples/sample_0001.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
            {
                "id": "sample_0002",
                "path": "samples/sample_0002.png",
                "caption_path": "samples/sample_0002.txt",
                "width": 768,
                "height": 512,
                "bucket": "768x512",
            },
            {
                "id": "sample_0003",
                "path": "samples/sample_0003.png",
                "caption_path": "samples/sample_0003.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
        ]
    }


def _chunk_shapes(value: Any) -> list[tuple[int, int, int]]:
    chunks = value if isinstance(value, list) else []
    rows: list[tuple[int, int, int]] = []
    for item in chunks:
        if not isinstance(item, Mapping):
            continue
        rows.append((
            int(item.get("start_batch", 0) or 0),
            int(item.get("batch_count", 0) or 0),
            int(item.get("sample_count", 0) or 0),
        ))
    return rows


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_data_pipeline_semantic_h2d_scorecard"]
