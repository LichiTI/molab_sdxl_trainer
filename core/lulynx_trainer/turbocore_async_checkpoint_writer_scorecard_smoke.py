"""Smoke checks for P6G async checkpoint writer scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_async_checkpoint_writer_scorecard import (  # noqa: E402
    build_async_checkpoint_writer_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_async_checkpoint_writer_scorecard(size_bytes=1024 * 1024)
    proof = report["proof"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert proof["completed_job_count"] == proof["submitted_job_count"], proof
    assert proof["atomic_commit_ok"] is True, proof
    assert proof["parity_ok"] is True, proof
    assert proof["tmp_leftovers"] == [], proof
    return {
        "schema_version": 1,
        "probe": "turbocore_async_checkpoint_writer_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
