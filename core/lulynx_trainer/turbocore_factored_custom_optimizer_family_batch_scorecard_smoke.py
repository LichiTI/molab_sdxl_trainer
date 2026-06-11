"""Smoke checks for built-in factored/custom optimizer family batch."""

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

from core.turbocore_factored_custom_optimizer_family_batch_scorecard import (  # noqa: E402
    build_factored_custom_optimizer_family_batch_scorecard,
)

ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    report = build_factored_custom_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT, write_artifact=True)
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["ok"] is True, report
    assert report["factored_custom_family_batch_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert set(rows) == {"adafactor", "Automagic++", "AnimaFactoredAdamW"}, rows
    for row in rows.values():
        assert row["batch_status"] == "factored_custom_dispatch_integration_review_ready", row
        assert row["native_scratch_kernel_ready"] is True, row
        assert row["training_tensor_binding_canary_ready"] is True, row
        assert row["runtime_dispatch_adapter_shadow_ready"] is True, row
        assert row["training_loop_canary_ready"] is True, row
        assert row["e2e_shadow_matrix_ready"] is True, row
        assert row["canary_rollout_policy_ready"] is True, row
        assert row["dispatch_integration_review_ready"] is True, row
        assert row["training_path_enabled"] is False, row
        assert row["default_behavior_changed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["unsafe_reasons"] == [], row
    summary = report["summary"]
    assert summary["optimizer_count"] == 3, summary
    assert summary["native_scratch_kernel_ready_count"] == 3, summary
    assert summary["training_tensor_binding_canary_ready_count"] == 3, summary
    assert summary["runtime_dispatch_adapter_shadow_ready_count"] == 3, summary
    assert summary["training_loop_canary_ready_count"] == 3, summary
    assert summary["e2e_shadow_matrix_ready_count"] == 3, summary
    assert summary["canary_rollout_policy_ready_count"] == 3, summary
    assert summary["dispatch_integration_review_ready_count"] == 3, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_factored_custom_optimizer_family_batch_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
