"""Smoke checks for schedule-free plugin checkpoint adapter proof."""

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

from core.turbocore_plugin_schedulefree_checkpoint_adapter_scorecard import (  # noqa: E402
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_schedulefree_checkpoint_adapter_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_checkpoint_adapter_scorecard()
    cases = {str(case["optimizer_name"]): case for case in report["cases"]}
    assert report["ok"] is True, report
    assert report["checkpoint_adapter_proof_ready"] is True, report
    assert report["runtime_adapter_enabled"] is False, report
    assert report["training_checkpoint_integration_enabled"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert set(cases) == set(TARGET_PLUGIN_OPTIMIZERS), cases
    for name, case in cases.items():
        assert case["ok"] is True, case
        assert case["pack_unpack_probe"] is True, case
        assert case["resume_parity"] is True, case
        assert case["max_resume_diff"] <= case["tolerance"], case
        summary = case["envelope_summary"]
        assert summary["train_mode_restored_by_group"] is True, (name, summary)
        assert summary["state_entry_count"] >= 1, (name, summary)
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_checkpoint_adapter_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
