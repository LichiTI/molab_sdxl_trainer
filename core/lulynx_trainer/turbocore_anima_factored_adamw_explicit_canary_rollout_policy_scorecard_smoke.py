"""Smoke checks for AnimaFactoredAdamW P29 explicit canary rollout policy."""

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

from core.turbocore_anima_factored_adamw_explicit_canary_rollout_policy_scorecard import (  # noqa: E402
    P28_AUDIT_BUILDER,
    build_anima_factored_adamw_explicit_canary_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_anima_factored_adamw_explicit_canary_rollout_policy_scorecard()
    policy = report["policy"]
    summary = report["summary"]
    rollback = policy["rollback_policy"]
    assert report["ok"] is True, report
    assert report["report_only"] is True, report
    assert report["explicit_canary_policy_ready"] is True, report
    assert report["canary_auto_enabled"] is False, report
    assert report["manual_review_required"] is True, report
    assert report["fallback_rollback_ready"] is True, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["default_behavior_unchanged"] is True, report
    assert report["p28_dependency"]["required_builder"] == P28_AUDIT_BUILDER, report
    assert report["p28_dependency"]["builder_name_recorded"] is True, report
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["canary_auto_enabled"] is False, policy
    assert policy["manual_review_required"] is True, policy
    assert policy["blocked_modes_until_review"] == ["canary", "auto"], policy
    assert rollback["fallback_backend"] == "python_anima_factored_adamw", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    assert summary["canary_auto_enabled"] is False, summary
    assert summary["manual_review_required"] is True, summary
    assert summary["fallback_rollback_ready"] is True, summary
    assert summary["runtime_dispatch_not_enabled"] is True, summary
    assert summary["default_behavior_unchanged"] is True, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_anima_factored_adamw_explicit_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
