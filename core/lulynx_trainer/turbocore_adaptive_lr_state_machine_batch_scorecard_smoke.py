"""Smoke checks for the adaptive-LR state-machine batch scorecard."""

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

from core.configs import OptimizerType  # noqa: E402
from core.turbocore_adaptive_lr_state_machine_batch_scorecard import (  # noqa: E402
    TARGET_OPTIMIZERS,
    build_adaptive_lr_state_machine_batch_scorecard,
)


EXPECTED_OPTIMIZERS = {
    OptimizerType.AUTO_PRODIGY.value,
    OptimizerType.PRODIGY.value,
    OptimizerType.DADAPTATION.value,
    OptimizerType.DADAPT_ADAM_PREPRINT.value,
    OptimizerType.DADAPT_ADAGRAD.value,
    OptimizerType.DADAPT_ADAM.value,
    OptimizerType.DADAPT_ADAN.value,
    OptimizerType.DADAPT_ADAN_IP.value,
    OptimizerType.DADAPT_LION.value,
    OptimizerType.DADAPT_SGD.value,
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE.value,
}


def run_smoke() -> dict[str, Any]:
    report = build_adaptive_lr_state_machine_batch_scorecard(write_artifact=True)
    rows = report["rows"]
    row_by_optimizer = {str(row["optimizer_type"]): row for row in rows}

    assert report["ok"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_ready"] is False, report
    assert set(row_by_optimizer) == EXPECTED_OPTIMIZERS, row_by_optimizer
    assert len(TARGET_OPTIMIZERS) == 11, TARGET_OPTIMIZERS
    assert report["summary"]["target_count"] == 11, report
    assert report["summary"]["state_machine_reference_ready_count"] == 11, report
    assert report["summary"]["state_machine_abi_spec_ready_count"] == 11, report
    assert report["summary"]["dynamic_lr_scalar_state_spec_ready_count"] == 11, report
    assert report["summary"]["d_estimator_global_state_spec_ready_count"] == 11, report
    assert report["summary"]["per_step_quality_guard_spec_ready_count"] == 11, report
    assert report["summary"]["resume_scope_spec_ready_count"] == 11, report
    assert report["summary"]["native_kernel_preconditions_spec_ready_count"] == 11, report
    assert report["summary"]["native_ready_count"] == 0, report
    assert report["summary"]["training_path_enabled_count"] == 0, report
    assert report["summary"]["native_dispatch_allowed_count"] == 0, report

    for row in rows:
        assert row["family"] in {"adaptive_lr_prodigy", "adaptive_lr_dadapt"}, row
        assert row["state_machine_status"] == "reference_ready_report_only", row
        assert row["state_machine_reference_ready"] is True, row
        assert row["batch_reference_ready"] is True, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["native_ready"] is False, row
        assert row["default_behavior_changed"] is False, row
        assert row["next_gate"], row
        assert row["blocked_reasons"], row

    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_state_machine_batch_scorecard_smoke",
        "ok": True,
        "artifact": "temp/turbocore_optimizer/turbocore_adaptive_lr_state_machine_batch_scorecard.json",
        "summary": report["summary"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
