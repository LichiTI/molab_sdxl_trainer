"""Smoke checks for P6 checkpoint artifact native streaming scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_checkpoint_artifact_streaming_scorecard import (  # noqa: E402
    build_checkpoint_artifact_streaming_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_checkpoint_artifact_streaming_scorecard(
        size_bytes=4 * 1024 * 1024,
        buffer_bytes=1024 * 1024,
    )
    assert report["ok"] is True, report
    assert report["default_behavior_changed"] is False, report
    if report["capabilities"]["native_entrypoint"]:
        assert report["promotion_ready"] is True, report
        assert report["parity"]["parity_ok"] is True, report
        assert report["benchmark"]["ok"] is True, report
        assert float(report["benchmark"]["native_stream_copy_ms"]) > 0.0, report
    else:
        assert "checkpoint_artifact_streaming_entrypoint_missing" in report["blocked_reasons"], report
    return {
        "schema_version": 1,
        "probe": "turbocore_checkpoint_artifact_streaming_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "scorecard": report,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
