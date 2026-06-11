"""Smoke checks for AdamW variant canary rollout policy."""

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

from core.turbocore_adamw_variant_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adamw_variant_canary_rollout_policy_scorecard,
)
from core.turbocore_adamw_variant_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adamw_variant_e2e_shadow_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    shadow = build_adamw_variant_e2e_shadow_matrix_scorecard(include_live_canaries=True)
    report = build_adamw_variant_canary_rollout_policy_scorecard(shadow_matrix_report=shadow)
    policy = report["policy"]
    expected = {
        "AdamW8bit",
        "PagedAdamW",
        "PagedAdamW32bit",
        "PagedAdamW8bit",
        "KahanAdamW8bit",
        "AdamWScheduleFree",
    }
    assert report["scorecard"] == "turbocore_adamw_variant_canary_rollout_policy_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["canary_rollout_policy_ready"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["canary_auto_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert policy["explicit_opt_in_required"] is True, policy
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["max_canary_fraction_default"] == 0.0, policy
    assert set(policy["optimizer_types"]) == expected, policy
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(report: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adamw_variant_canary_rollout_policy_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
