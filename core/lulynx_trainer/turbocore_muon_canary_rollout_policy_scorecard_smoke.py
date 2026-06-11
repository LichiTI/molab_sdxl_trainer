"""Smoke checks for Muon default-off canary rollout policy."""

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

from core.turbocore_muon_canary_rollout_policy_scorecard import (  # noqa: E402
    build_muon_canary_rollout_policy_scorecard,
)
from core.turbocore_muon_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_muon_e2e_shadow_matrix_scorecard,
)
from core.turbocore_muon_training_loop_canary_scorecard import (  # noqa: E402
    build_muon_training_loop_canary_scorecard,
)
from core.turbocore_muon_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_muon_training_tensor_binding_canary_scorecard,
)
from core.turbocore_muon_native_scratch_kernel_scorecard import (  # noqa: E402
    build_muon_native_scratch_kernel_scorecard,
)
from core.turbocore_muon_model_shape_aware_family_batch_scorecard import (  # noqa: E402
    build_muon_model_shape_aware_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    family = build_muon_model_shape_aware_family_batch_scorecard()
    scratch = build_muon_native_scratch_kernel_scorecard(muon_model_shape_report=family)
    tensor_binding = build_muon_training_tensor_binding_canary_scorecard(
        native_scratch_report=scratch,
        workspace_root=REPO_ROOT,
    )
    training_loop = build_muon_training_loop_canary_scorecard(
        training_tensor_binding_report=tensor_binding,
    )
    shadow = build_muon_e2e_shadow_matrix_scorecard(training_loop_canary_report=training_loop)
    report = build_muon_canary_rollout_policy_scorecard(
        shadow_matrix_report=shadow,
        write_artifact=True,
    )
    policy = report["policy"]
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_muon_canary_rollout_policy_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["canary_rollout_policy_ready"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["canary_auto_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["product_native_ready"] is False, report
    assert policy["optimizer_types"] == ["Muon"], policy
    assert policy["explicit_opt_in_required"] is True, policy
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["max_canary_fraction_default"] == 0.0, policy
    assert summary["canary_rollout_policy_ready_count"] == 1, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_muon_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
