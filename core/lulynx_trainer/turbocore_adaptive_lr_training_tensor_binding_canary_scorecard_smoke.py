"""Smoke checks for adaptive-LR live tensor binding canary evidence."""

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

from core.turbocore_adaptive_lr_cuda_kernel_implementation_scorecard import (  # noqa: E402
    build_adaptive_lr_cuda_kernel_implementation_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES  # noqa: E402
from core.turbocore_adaptive_lr_training_tensor_binding_canary_scorecard import (  # noqa: E402
    ENTRYPOINT,
    build_adaptive_lr_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    cuda_impl = build_adaptive_lr_cuda_kernel_implementation_scorecard(workspace_root=REPO_ROOT)
    payload = build_adaptive_lr_training_tensor_binding_canary_scorecard(
        cuda_implementation_report=cuda_impl,
        workspace_root=REPO_ROOT,
    )
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    summary = payload["summary"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_training_tensor_binding_canary_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["training_tensor_binding_canary_ready"] is True, payload
    assert payload["entrypoint"] == ENTRYPOINT, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["runtime_canary_ready"] is False, payload
    assert payload["runtime_canary_hit"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert set(rows) == {case.optimizer.value for case in TARGET_CASES}, rows
    assert summary["target_count"] == len(TARGET_CASES), summary
    assert summary["training_tensor_binding_canary_ready_count"] == len(TARGET_CASES), summary
    assert summary["training_tensor_binding_parity_ready_count"] == len(TARGET_CASES), summary
    assert summary["kernel_executed_count"] == len(TARGET_CASES), summary
    assert summary["family_case_count"] == 2, summary
    assert summary["family_passed_case_count"] == 2, summary
    assert summary["runtime_canary_ready_count"] == 0, summary
    assert summary["runtime_canary_hit_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary

    for case in TARGET_CASES:
        row = rows[case.optimizer.value]
        assert row["state_machine_status"] == "training_tensor_binding_canary_ready", row
        assert row["cuda_kernel_implementation_ready"] is True, row
        assert row["training_tensor_binding_canary_ready"] is True, row
        assert row["training_tensor_binding_parity_ready"] is True, row
        assert row["kernel_executed"] is True, row
        assert row["training_path_enabled"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_training_tensor_binding_canary_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_training_tensor_binding_canary_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
