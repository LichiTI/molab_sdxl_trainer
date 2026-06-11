"""Smoke checks for adaptive-LR canary rollout policy evidence."""

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

from core.turbocore_adaptive_lr_canary_rollout_policy_scorecard import (  # noqa: E402
    POLICY_KIND,
    build_adaptive_lr_canary_rollout_policy_scorecard,
)
from core.turbocore_adaptive_lr_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adaptive_lr_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES  # noqa: E402


def run_smoke() -> dict[str, Any]:
    shadow = build_adaptive_lr_e2e_shadow_matrix_scorecard()
    payload = build_adaptive_lr_canary_rollout_policy_scorecard(shadow_matrix_report=shadow)
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    summary = payload["summary"]
    policy = payload["policy"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_canary_rollout_policy_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["canary_rollout_policy_ready"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["canary_auto_enabled"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["product_native_ready"] is False, payload
    assert payload["policy_kind"] == POLICY_KIND, payload
    assert payload["fallback_backend_authoritative"] is True, payload
    assert set(rows) == {case.optimizer.value for case in TARGET_CASES}, rows

    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["canary_auto_enabled"] is False, policy
    assert policy["explicit_opt_in_required"] is True, policy
    assert policy["manual_review_required"] is True, policy
    assert policy["max_canary_fraction_default"] == 0.0, policy
    assert policy["runtime_dispatch_ready"] is False, policy
    assert policy["native_dispatch_allowed"] is False, policy
    assert policy["training_path_enabled"] is False, policy
    assert policy["product_native_ready"] is False, policy
    assert policy["rollback_policy"]["fallback_authoritative"] is True, policy
    assert policy["rollback_policy"]["rollback_on_nonfinite"] is True, policy
    assert policy["rollback_policy"]["rollback_on_parity_failure"] is True, policy
    assert policy["rollback_policy"]["rollback_on_state_machine_guard_failure"] is True, policy

    assert summary["target_count"] == len(TARGET_CASES), summary
    assert summary["canary_rollout_policy_ready"] is True, summary
    assert summary["canary_rollout_policy_ready_count"] == len(TARGET_CASES), summary
    assert summary["canary_enabled_by_default"] is False, summary
    assert summary["explicit_opt_in_required"] is True, summary
    assert summary["max_canary_fraction_default"] == 0.0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary

    for row in rows.values():
        assert row["canary_rollout_policy_ready"] is True, row
        assert row["e2e_shadow_matrix_ready"] is True, row
        assert row["manual_review_required"] is True, row
        assert row["canary_auto_enabled"] is False, row
        assert row["canary_enabled_by_default"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["product_native_ready"] is False, row

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_canary_rollout_policy_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
