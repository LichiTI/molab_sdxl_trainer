"""Smoke checks for KahanAdamW8bit checkpoint/resume adapter."""

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

from core.turbocore_kahan_adamw8bit_checkpoint_resume_adapter_scorecard import (  # noqa: E402
    ADAPTER_RUNTIME_KIND,
    build_kahan_adamw8bit_checkpoint_resume_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_kahan_adamw8bit_checkpoint_resume_adapter_scorecard(run_live_probe=True)
    assert report["ok"] is True, report
    assert report["adapter_runtime_kind"] == ADAPTER_RUNTIME_KIND, report
    assert report["checkpoint_resume_adapter_ready"] is True, report
    assert report["training_checkpoint_integration_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    live = report["live_probe"]
    assert live["status"] == "passed", live
    assert live["runtime_envelope_roundtrip_passed"] is True, live
    assert live["kahan_comp_fp32_restored"] is True, live
    assert live["resume_probe_passed"] is True, live
    assert live["quantized_state_mismatch_count"] == 0, live
    return {
        "schema_version": 1,
        "probe": "turbocore_kahan_adamw8bit_checkpoint_resume_adapter_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
