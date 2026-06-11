"""Smoke checks for quantized simple dispatch integration review scorecard."""

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

from core.turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_quantized_rollout_policy_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    rollout = build_simple_optimizer_quantized_rollout_policy_scorecard()
    assert rollout["canary_rollout_policy_ready"] is True, rollout
    payload = build_simple_optimizer_quantized_dispatch_integration_review_scorecard(
        rollout_policy_report=rollout
    )
    review = payload["review_package"]
    hooks = review["runtime_hook_contract"]
    dispatch = review["dispatch_contract"]
    assert payload["scorecard"] == "turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is True, payload
    assert payload["review_gate_ready"] is True, payload
    assert payload["dispatch_integration_review"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["canary_auto_blocked_until_review"] is True, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["summary"]["optimizer_count"] == 3, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert hooks["training_executor_factory"].endswith("build_simple_quantized_optimizer_training_executor"), hooks
    assert hooks["optimizer_state_sync_hook"].endswith("sync_optimizer_state_to_pytorch"), hooks
    assert dispatch["runtime_dispatch_enabled_by_this_gate"] is False, dispatch
    assert dispatch["requires_manual_review_approval"] is True, dispatch
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
