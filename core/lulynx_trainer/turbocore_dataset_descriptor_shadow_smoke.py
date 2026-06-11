"""Smoke probe for debug-only native descriptor shadow parity."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.dataset_loader import CaptionDataset  # noqa: E402
from core.turbocore_dataset_descriptor_shadow import (  # noqa: E402
    build_caption_dataset_descriptor_manifest,
    build_dataset_sampler_order_reference,
    create_caption_dataset_shadow_session,
    run_caption_dataset_descriptor_shadow_probe,
    run_caption_dataset_shadow_lifecycle_probe,
    run_native_dataset_shadow_lifecycle_from_handles,
    validate_native_dataset_sampler_order_shadow,
    validate_native_dataset_staging_plan_sampler_order_from_reference,
)
from core.turbocore_dataset_staging import _load_native_dataset_staging_handle_api  # noqa: E402
from core.turbocore_dataset_staging_session import (  # noqa: E402
    create_native_dataset_descriptor_session,
    destroy_native_dataset_descriptor_session,
    validate_native_dataset_descriptor_session_parity,
)


def _write_sample(root: Path, name: str, size: tuple[int, int], caption: str) -> None:
    Image.new("RGB", size, color=(32, 96, 160)).save(root / f"{name}.png")
    (root / f"{name}.txt").write_text(caption, encoding="utf-8")


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_descriptor_shadow_") as tmp:
        root = Path(tmp)
        _write_sample(root, "img_0001", (512, 768), "one")
        _write_sample(root, "img_0002", (768, 512), "two")
        _write_sample(root, "img_0003", (640, 640), "three")
        dataset = CaptionDataset(
            str(root),
            resolution=512,
            caption_extension=".txt",
            enable_bucket=True,
            min_bucket_reso=512,
            max_bucket_reso=768,
            bucket_reso_steps=128,
            bucket_custom_resos="512x768,768x512,640x640",
            shuffle_caption=False,
        )
        probe = run_caption_dataset_descriptor_shadow_probe(
            dataset,
            batch_size=2,
            drop_last=False,
            prefetch_depth=4,
            chunk_size=2,
        )
        assert probe["ok"] is True, probe
        assert probe["descriptor_count"] == 3, probe
        assert probe["batch_count"] == 2, probe
        assert probe["sample_descriptors_owned"] is True, probe
        assert probe["debug_only"] is True, probe
        assert probe["shadow_run"] is True, probe
        assert probe["training_path_enabled"] is False, probe
        assert probe["parity_probe"]["ok"] is True, probe
        assert probe["parity_probe"]["mismatch_count"] == 0, probe
        assert probe["parity_probe"]["checksum_fast_path"] is True, probe
        assert probe["descriptor_preview"][0]["id"] == "img_0001", probe

        manifest = build_caption_dataset_descriptor_manifest(dataset)
        session_id = create_native_dataset_descriptor_session(
            manifest,
            batch_size=2,
            drop_last=False,
            prefetch_depth=4,
            chunk_size=2,
        )
        try:
            mutated = json.loads(json.dumps(manifest, ensure_ascii=False))
            mutated["samples"][1]["bucket"] = "999x999"
            mismatch = validate_native_dataset_descriptor_session_parity(
                session_id,
                mutated,
                max_mismatches=2,
            )
        finally:
            destroy_native_dataset_descriptor_session(session_id)
        assert mismatch["ok"] is False, mismatch
        assert mismatch["checksum_fast_path"] is False, mismatch
        assert mismatch["exact_compare"] is True, mismatch
        assert mismatch["mismatch_count"] == 1, mismatch
        assert mismatch["mismatches"][0]["index"] == 1, mismatch
        assert mismatch["training_path_enabled"] is False, mismatch

        sequential_order = validate_native_dataset_sampler_order_shadow(
            sample_count=len(dataset),
            batch_size=2,
            drop_last=False,
            shuffle=False,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
        )
        assert sequential_order["ok"] is True, sequential_order
        assert sequential_order["sampler_kind"] == "sequential_range_v1", sequential_order
        assert sequential_order["checksum_match"] is True, sequential_order
        assert sequential_order["preview_match"] is True, sequential_order
        assert sequential_order["debug_only"] is True, sequential_order
        assert sequential_order["training_path_enabled"] is False, sequential_order

        shuffle_order = validate_native_dataset_sampler_order_shadow(
            sample_count=len(dataset),
            batch_size=2,
            drop_last=False,
            shuffle=True,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
        )
        assert shuffle_order["ok"] is True, shuffle_order
        assert shuffle_order["sampler_kind"] == "fisher_yates_splitmix64_v1", shuffle_order
        assert shuffle_order["checksum_match"] is True, shuffle_order
        assert shuffle_order["preview_match"] is True, shuffle_order
        assert shuffle_order["lazy_affine_equivalent"] is False, shuffle_order
        assert shuffle_order["shadow_run"] is True, shuffle_order
        assert shuffle_order["training_path_enabled"] is False, shuffle_order

        cached_shuffle_order = validate_native_dataset_sampler_order_shadow(
            sample_count=len(dataset),
            batch_size=2,
            drop_last=False,
            shuffle=True,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
        )
        assert cached_shuffle_order["ok"] is True, cached_shuffle_order
        assert cached_shuffle_order["reference_cache_hit"] is True, cached_shuffle_order
        assert cached_shuffle_order["checksum_match"] is True, cached_shuffle_order
        assert cached_shuffle_order["training_path_enabled"] is False, cached_shuffle_order

        native = _load_native_dataset_staging_handle_api()
        reference = build_dataset_sampler_order_reference(
            sample_count=len(dataset),
            batch_size=2,
            drop_last=False,
            shuffle=True,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
        )
        plan_id = int(native.create_dataset_staging_plan(len(dataset), 2, False, True, 123, 4, 2))
        try:
            plan_order = validate_native_dataset_staging_plan_sampler_order_from_reference(
                plan_id,
                reference,
            )
        finally:
            native.destroy_dataset_staging_plan(plan_id)
        assert plan_order["ok"] is True, plan_order
        assert plan_order["uses_existing_native_plan"] is True, plan_order
        assert plan_order["checksum_match"] is True, plan_order
        assert plan_order["preview_match"] is True, plan_order
        assert plan_order["training_path_enabled"] is False, plan_order

        lifecycle = run_caption_dataset_shadow_lifecycle_probe(
            dataset,
            batch_size=2,
            drop_last=False,
            shuffle=True,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
            worker_count=2,
        )
        assert lifecycle["ok"] is True, lifecycle
        assert lifecycle["provider"] == "native_dataset_shadow_lifecycle", lifecycle
        assert lifecycle["descriptor_parity_ok"] is True, lifecycle
        assert lifecycle["sampler_order_ok"] is True, lifecycle
        assert lifecycle["worker_preview_ok"] is True, lifecycle
        assert lifecycle["sample_count_match"] is True, lifecycle
        assert lifecycle["batch_shape_match"] is True, lifecycle
        assert lifecycle["uses_existing_native_plan"] is True, lifecycle
        assert lifecycle["uses_existing_descriptor_session"] is True, lifecycle
        assert lifecycle["sample_descriptors_owned"] is True, lifecycle
        assert lifecycle["worker_results_owned"] is True, lifecycle
        assert lifecycle["debug_only"] is True, lifecycle
        assert lifecycle["shadow_run"] is True, lifecycle
        assert lifecycle["training_path_enabled"] is False, lifecycle
        assert lifecycle["descriptor_shadow"]["checksum_fast_path"] is True, lifecycle
        assert lifecycle["sampler_shadow"]["uses_existing_native_plan"] is True, lifecycle
        assert lifecycle["worker_probe"]["result_preview_returned"] is True, lifecycle
        assert lifecycle["worker_probe"]["result_preview"][0]["descriptor_preview"][0]["id"], lifecycle

        session_id = create_native_dataset_descriptor_session(
            manifest,
            batch_size=2,
            drop_last=False,
            prefetch_depth=4,
            chunk_size=2,
        )
        plan_id = int(native.create_dataset_staging_plan(len(dataset), 2, False, True, 123, 4, 2))
        try:
            bad_reference = dict(reference)
            bad_reference["index_checksum"] = int(reference.get("index_checksum", 0) or 0) ^ 1
            lifecycle_mismatch = run_native_dataset_shadow_lifecycle_from_handles(
                session_id=session_id,
                plan_id=plan_id,
                reference_manifest=manifest,
                sampler_reference=bad_reference,
                worker_count=2,
                queue_depth=4,
                max_batches_per_submit=2,
            )
        finally:
            native.destroy_dataset_staging_plan(plan_id)
            destroy_native_dataset_descriptor_session(session_id)
        assert lifecycle_mismatch["ok"] is False, lifecycle_mismatch
        assert lifecycle_mismatch["descriptor_parity_ok"] is True, lifecycle_mismatch
        assert lifecycle_mismatch["sampler_order_ok"] is False, lifecycle_mismatch
        assert lifecycle_mismatch["worker_preview_ok"] is True, lifecycle_mismatch
        assert lifecycle_mismatch["sampler_shadow"]["checksum_match"] is False, lifecycle_mismatch
        assert lifecycle_mismatch["training_path_enabled"] is False, lifecycle_mismatch

        with create_caption_dataset_shadow_session(
            dataset,
            batch_size=2,
            drop_last=False,
            shuffle=True,
            seed=123,
            prefetch_depth=4,
            chunk_size=2,
        ) as shadow_session:
            session_stats = shadow_session.stats()
            assert session_stats["ok"] is True, session_stats
            assert session_stats["run_count"] == 0, session_stats
            first_lifecycle = shadow_session.run_lifecycle(worker_count=2)
            second_lifecycle = shadow_session.run_lifecycle(worker_count=2)
            assert first_lifecycle["ok"] is True, first_lifecycle
            assert second_lifecycle["ok"] is True, second_lifecycle
            assert first_lifecycle["session_id"] == second_lifecycle["session_id"], second_lifecycle
            assert first_lifecycle["plan_id"] == second_lifecycle["plan_id"], second_lifecycle
            assert first_lifecycle["session_reused"] is False, first_lifecycle
            assert second_lifecycle["session_reused"] is True, second_lifecycle
            assert second_lifecycle["shadow_session"]["run_count"] == 2, second_lifecycle
            mutated_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
            mutated_manifest["samples"][0]["bucket"] = "999x999"
            rebuild_guard = shadow_session.run_lifecycle(
                reference_manifest=mutated_manifest,
                worker_count=2,
            )
            assert rebuild_guard["ok"] is False, rebuild_guard
            assert rebuild_guard["reason"] == "reference_manifest_changed", rebuild_guard
            assert rebuild_guard["requires_rebuild"] is True, rebuild_guard
            assert rebuild_guard["manifest_guard"]["requires_rebuild"] is True, rebuild_guard
            assert rebuild_guard["training_path_enabled"] is False, rebuild_guard
        assert shadow_session.stats()["closed"] is True, shadow_session.stats()
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_descriptor_shadow_smoke",
            "ok": True,
            "training_path_enabled": False,
            "shadow_probe": probe,
            "mismatch_probe": mismatch,
            "sequential_order_probe": sequential_order,
            "shuffle_order_probe": shuffle_order,
            "cached_shuffle_order_probe": cached_shuffle_order,
            "plan_order_probe": plan_order,
            "shadow_lifecycle_probe": lifecycle,
            "shadow_lifecycle_mismatch_probe": lifecycle_mismatch,
            "shadow_session_first_probe": first_lifecycle,
            "shadow_session_second_probe": second_lifecycle,
            "shadow_session_rebuild_guard": rebuild_guard,
        }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
