"""Smoke checks for adaptive LR optimizer state-machine scorecard."""

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

from core.turbocore_adaptive_lr_optimizer_state_machine_scorecard import (  # noqa: E402
    ADAPTIVE_LR_OPTIMIZERS,
    build_adaptive_lr_optimizer_state_machine_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adaptive_lr_optimizer_state_machine_scorecard()
    cases = {str(case["case"]): case for case in report["cases"]}
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["ok"] is True, report
    assert report["state_machine_reference_ready"] is True, report
    assert report["adaptive_family_classified"] is True, report
    assert report["scheduler_coupling_constrained"] is True, report
    assert report["adamw_kernel_reuse_blocked"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert len(rows) == len(ADAPTIVE_LR_OPTIMIZERS), rows
    assert all(row["adamw_kernel_compatible"] is False for row in rows.values()), rows
    auto = cases["auto_prodigy_state_roundtrip"]
    assert auto["ok"] is True, auto
    assert auto["after_step"]["has_required_param_state"] is True, auto
    assert auto["after_step"]["has_required_global_state"] is True, auto
    assert auto["after_eval"]["eval_mode"] is True, auto
    assert auto["after_eval"]["has_train_weight_stash"] is True, auto
    assert auto["after_train"]["eval_mode"] is False, auto
    assert auto["max_resume_diff"] <= auto["tolerance"], auto
    guard = cases["adamw_kernel_reuse_guard"]
    assert guard["ok"] is True, guard
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_optimizer_state_machine_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
