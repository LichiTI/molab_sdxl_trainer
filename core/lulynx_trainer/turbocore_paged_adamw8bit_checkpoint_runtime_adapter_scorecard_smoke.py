"""Smoke checks for PagedAdamW8bit checkpoint runtime adapter proof."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_paged_adamw8bit_checkpoint_runtime_adapter_scorecard import (  # noqa: E402
    ADAPTER_RUNTIME_KIND,
    build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_checkpoint_runtime_adapter_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["adapter_runtime_kind"] == ADAPTER_RUNTIME_KIND, report
    assert report["checkpoint_adapter_runtime_proof_ready"] is True, report
    assert report["checkpoint_adapter_runtime_ready"] is True, report
    assert report["training_checkpoint_integration_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    live = report["live_probe"]
    if torch.cuda.is_available():
        assert live["status"] == "passed", live
        assert live["runtime_envelope_roundtrip_passed"] is True, live
        assert live["runtime_import_restores_live_buffers"] is True, live
        assert live["resume_probe_passed"] is True, live
        assert live["max_resume_diff"] <= 1e-5, live
    else:
        assert live["status"] == "skipped", live
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_checkpoint_runtime_adapter_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
