"""Smoke checks for dataset backend strategy resolution."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for root in (REPO_ROOT, BACKEND_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

try:
    from .data_backend_resolver import (
        discover_webdataset_shards,
        normalize_data_backend,
        resolve_data_backend,
    )
except ImportError:  # pragma: no cover - standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_backend_resolver import (
        discover_webdataset_shards,
        normalize_data_backend,
        resolve_data_backend,
    )

from backend.core.configs import UnifiedTrainingConfig
from backend.core.lulynx_trainer.config_adapter import ConfigAdapter


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard_a = root / "000000.tar"
        shard_b = root / "000001.tar.gz"
        non_shard = root / "notes.txt"
        shard_a.write_bytes(b"")
        shard_b.write_bytes(b"")
        non_shard.write_text("not a shard", encoding="utf-8")

        assert normalize_data_backend("wds") == "webdataset"
        assert normalize_data_backend("image-folder") == "caption"
        assert normalize_data_backend("unknown value") == "auto"
        assert UnifiedTrainingConfig.from_dict({"data_backend": "wds"}).data_backend == "webdataset"
        assert ConfigAdapter.from_frontend_dict({"data_backend": "tar"}).data_backend == "webdataset"

        plan = discover_webdataset_shards(root)
        assert plan.shard_count == 2
        assert len(plan.shards) == 2
        assert all(str(path).endswith((".tar", ".tar.gz")) for path in plan.shards)

        auto_decision = resolve_data_backend(
            "auto",
            data_dir=root,
            package_availability={"webdataset": True, "nvidia.dali": False},
        )
        assert auto_decision.resolved_backend == "caption"
        assert auto_decision.webdataset_available is True
        assert auto_decision.webdataset_shards.shard_count == 2
        assert "CaptionDataset" in " ".join(auto_decision.notes)

        explicit_ok = resolve_data_backend(
            "webdataset",
            data_dir=root,
            package_availability={"webdataset": True, "nvidia.dali": False},
        )
        assert explicit_ok.resolved_backend == "webdataset"
        assert explicit_ok.experimental is True
        assert explicit_ok.training_integration == "materialized_captiondataset_bridge"

        missing_package = resolve_data_backend(
            "webdataset",
            data_dir=root,
            package_availability={"webdataset": False},
        )
        assert missing_package.resolved_backend == "webdataset"
        assert missing_package.training_integration == "materialized_captiondataset_bridge"
        assert "built-in" in " ".join(missing_package.notes)

        empty_dir = root / "empty"
        empty_dir.mkdir()
        missing_shards = resolve_data_backend(
            "webdataset",
            data_dir=empty_dir,
            package_availability={"webdataset": True},
        )
        assert missing_shards.resolved_backend == "caption"
        assert "no .tar" in missing_shards.fallback_reason

        dali = resolve_data_backend(
            "dali",
            data_dir=root,
            package_availability={"webdataset": True, "nvidia.dali": False},
        )
        assert dali.resolved_backend == "caption"
        assert "future" in dali.fallback_reason

        profile = explicit_ok.as_dict()
        assert profile["requested_backend"] == "webdataset"
        assert profile["webdataset_shards"]["shard_count"] == 2

    print("data_backend_resolver_smoke: ok")


if __name__ == "__main__":
    main()
