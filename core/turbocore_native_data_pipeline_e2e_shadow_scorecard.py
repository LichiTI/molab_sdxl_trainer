"""End-to-end shadow matrix for the native TurboCore data pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_dataset_staging_session import run_native_dataset_descriptor_session_probe
from core.turbocore_native_data_pipeline_semantic_h2d_scorecard import (
    build_native_data_pipeline_semantic_h2d_scorecard,
)


FEATURE = "native_data_pipeline"
SHADOW_KIND = "native_data_pipeline_e2e_shadow_v0"


def build_native_data_pipeline_e2e_shadow_scorecard(
    *,
    semantic_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Run a fallback-authoritative synthetic data-path shadow."""

    semantic = dict(
        semantic_report
        or build_native_data_pipeline_semantic_h2d_scorecard(
            native_training_mode=native_training_mode,
        )
    )
    shadow = _run_shadow_case()
    validations = _validations(semantic, shadow)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_e2e_shadow_scorecard_v0",
        "gate": "p6k_native_data_pipeline_e2e_shadow",
        "ok": ready,
        "promotion_ready": ready,
        "e2e_shadow_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_shadow_updates_original": False,
        "fallback_backend_authoritative": True,
        "feature": FEATURE,
        "shadow_kind": SHADOW_KIND,
        "native_training_mode": str(semantic.get("native_training_mode") or native_training_mode),
        "semantic_summary": dict(semantic.get("summary") or {}),
        "shadow_case": shadow,
        "validations": validations,
        "summary": {
            "e2e_shadow_ready": ready,
            "shadow_case_status": shadow.get("status"),
            "batch_descriptor_parity_ok": bool(shadow.get("batch_descriptor_parity_ok", False)),
            "loss_parity_ok": bool(shadow.get("loss_parity_ok", False)),
            "native_shadow_updates_original": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add native data pipeline explicit canary rollout policy before dispatch review"
            if ready
            else "fix native data pipeline end-to-end shadow blockers"
        ),
        "notes": [
            "The shadow path uses cloned tensors and never becomes the training data authority.",
            "The synthetic loss proves descriptor-to-batch parity only, not quality or throughput.",
            "Real trainer data-path wiring still requires explicit review and rollback policy.",
        ],
    }


def _run_shadow_case() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - trainer env includes torch
        return {
            "schema_version": 1,
            "case": "native_data_pipeline_e2e_shadow",
            "ok": False,
            "status": "torch_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": ["native_data_pipeline_e2e_shadow_torch_unavailable"],
            "training_path_enabled": False,
        }

    manifest = _descriptor_manifest()
    reference_descriptors = list(manifest["samples"][:2])
    native_probe = run_native_dataset_descriptor_session_probe(
        manifest,
        batch_size=2,
        drop_last=False,
        prefetch_depth=4,
        chunk_size=2,
        epochs=1,
    )
    first_chunk = native_probe.get("first_chunk") if isinstance(native_probe.get("first_chunk"), Mapping) else {}
    native_descriptors = first_chunk.get("descriptor_preview") if isinstance(first_chunk.get("descriptor_preview"), list) else []
    native_descriptors = [dict(item) for item in native_descriptors[:2] if isinstance(item, Mapping)]

    reference_batch = _batch_tensor(reference_descriptors, torch=torch)
    native_shadow_batch = _batch_tensor(native_descriptors, torch=torch).clone()
    original_reference = reference_batch.clone()
    original_shadow = native_shadow_batch.clone()
    loss_reference = _synthetic_loss(reference_batch, torch=torch)
    loss_shadow = _synthetic_loss(native_shadow_batch, torch=torch)

    native_shadow_batch.add_(1000.0)
    reference_unchanged = bool(torch.equal(reference_batch, original_reference))
    shadow_mutated = bool(not torch.equal(native_shadow_batch, original_shadow))
    descriptor_ids_match = [item.get("id") for item in reference_descriptors] == [
        item.get("id") for item in native_descriptors
    ]
    max_batch_diff = float((original_reference - original_shadow).abs().max().item()) if original_reference.numel() else 0.0
    loss_diff = abs(float(loss_reference.item()) - float(loss_shadow.item()))
    ok = (
        bool(native_probe.get("ok", False))
        and descriptor_ids_match
        and max_batch_diff == 0.0
        and loss_diff <= 1e-7
        and reference_unchanged
        and shadow_mutated
    )
    return {
        "schema_version": 1,
        "case": "native_data_pipeline_e2e_shadow",
        "ok": ok,
        "status": "passed" if ok else "failed",
        "provider": str(native_probe.get("provider", "")),
        "native_runtime": bool(native_probe.get("native_runtime", False)),
        "batch_descriptor_parity_ok": descriptor_ids_match,
        "batch_tensor_parity_ok": max_batch_diff == 0.0,
        "loss_parity_ok": loss_diff <= 1e-7,
        "reference_unchanged_after_shadow_mutation": reference_unchanged,
        "native_shadow_mutated_clone_only": shadow_mutated,
        "native_shadow_updates_original": False,
        "reference_ids": [str(item.get("id", "")) for item in reference_descriptors],
        "native_ids": [str(item.get("id", "")) for item in native_descriptors],
        "max_batch_diff": max_batch_diff,
        "loss_reference": round(float(loss_reference.item()), 8),
        "loss_shadow": round(float(loss_shadow.item()), 8),
        "loss_diff": round(loss_diff, 10),
        "training_path_enabled": False,
        "blocked_reasons": [] if ok else ["native_data_pipeline_e2e_shadow_failed"],
    }


def _batch_tensor(descriptors: Sequence[Mapping[str, Any]], *, torch: Any) -> Any:
    rows: list[list[float]] = []
    for index, item in enumerate(descriptors):
        width = float(item.get("width", 0) or 0)
        height = float(item.get("height", 0) or 0)
        bucket = str(item.get("bucket", "") or "")
        bucket_hash = float(sum(ord(ch) for ch in bucket) % 997)
        rows.append([float(index), width / 1024.0, height / 1024.0, bucket_hash / 997.0])
    return torch.tensor(rows, dtype=torch.float32)


def _synthetic_loss(batch: Any, *, torch: Any) -> Any:
    weights = torch.tensor([0.25, 0.5, -0.125, 0.75], dtype=torch.float32)
    return (batch * weights).sum()


def _validations(semantic: Mapping[str, Any], shadow: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "p6j_semantic_h2d_ready",
            bool(semantic.get("semantic_h2d_matrix_ready", False)),
            "native_data_pipeline_semantic_h2d_missing",
        ),
        _validation(
            "batch_descriptor_parity",
            bool(shadow.get("batch_descriptor_parity_ok", False)),
            "native_data_pipeline_e2e_descriptor_parity_failed",
        ),
        _validation(
            "shadow_loss_parity",
            bool(shadow.get("loss_parity_ok", False)),
            "native_data_pipeline_e2e_loss_parity_failed",
        ),
        _validation(
            "native_shadow_does_not_update_original",
            not bool(shadow.get("native_shadow_updates_original", True))
            and bool(shadow.get("reference_unchanged_after_shadow_mutation", False)),
            "native_data_pipeline_e2e_shadow_updated_original",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(semantic.get("runtime_dispatch_ready", True))
            and not bool(semantic.get("native_dispatch_allowed", True))
            and not bool(semantic.get("training_path_enabled", True)),
            "native_data_pipeline_e2e_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(semantic.get("training_path_enabled", True))
            and not bool(semantic.get("default_behavior_changed", True)),
            "native_data_pipeline_e2e_shadow_changed_default_behavior",
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


__all__ = ["build_native_data_pipeline_e2e_shadow_scorecard"]
