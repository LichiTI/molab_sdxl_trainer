"""Smoke checks for built-in adaptive-LR replay matrix scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_adaptive_lr_state_machine_batch_scorecard import (  # noqa: E402
    TARGET_OPTIMIZERS,
    build_adaptive_lr_state_machine_batch_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_matrix_scorecard import (  # noqa: E402
    MATRIX_KIND,
    build_adaptive_lr_state_machine_replay_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    batch = build_adaptive_lr_state_machine_batch_scorecard()
    payload = build_adaptive_lr_state_machine_replay_matrix_scorecard(batch_report=batch)
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    summary = payload["summary"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_state_machine_replay_matrix_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["state_machine_replay_matrix_ready"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["report_only"] is True, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["native_kernel_ready"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert set(rows) == {optimizer.value for optimizer in TARGET_OPTIMIZERS}, rows
    assert summary["target_count"] == len(TARGET_OPTIMIZERS), summary
    assert summary["state_machine_replay_matrix_artifact_ready_count"] == len(TARGET_OPTIMIZERS), summary
    assert summary["state_machine_replay_matrix_implementation_ready_count"] == 0, summary
    assert summary["state_machine_replay_case_planned_count"] == len(TARGET_OPTIMIZERS) * 6, summary
    assert summary["state_machine_replay_resume_case_planned_count"] == len(TARGET_OPTIMIZERS) * 4, summary
    assert summary["state_machine_abi_implementation_ready_count"] == 0, summary
    assert summary["native_kernel_preconditions_implementation_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary

    for optimizer in TARGET_OPTIMIZERS:
        row = rows[optimizer.value]
        matrix = row["state_machine_replay_matrix_artifact"]
        assert row["state_machine_status"] == "replay_matrix_artifact_ready", row
        assert row["state_machine_reference_ready"] is True, row
        assert row["state_machine_abi_spec_ready"] is True, row
        assert row["state_machine_abi_implementation_ready"] is False, row
        assert row["state_machine_replay_matrix_artifact_ready"] is True, row
        assert row["state_machine_replay_matrix_implementation_ready"] is False, row
        assert row["native_route"] == "none_report_only", row
        assert row["training_path_enabled"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert matrix["artifact_kind"] == MATRIX_KIND, matrix
        assert matrix["spec_ready"] is True, matrix
        assert matrix["implementation_ready"] is False, matrix
        assert matrix["evidence_status"] == "planned_report_only", matrix
        assert "dynamic_lr_scalar_recomputed_from_saved_state" in matrix["replay_cases"], matrix
        assert "resume_next_step_matches_python_reference" in matrix["resume_replay_cases"], matrix
        assert "owner_release_hold" in matrix["blocked_until"], matrix

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_state_machine_replay_matrix_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_state_machine_replay_matrix_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
