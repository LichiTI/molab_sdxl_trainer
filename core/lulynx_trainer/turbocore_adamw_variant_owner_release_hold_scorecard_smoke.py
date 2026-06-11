"""Smoke checks for AdamW variant owner/release hold scorecard."""

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
from core.turbocore_adamw_variant_dispatch_integration_review_scorecard import (  # noqa: E402
    build_adamw_variant_dispatch_integration_review_scorecard,
)
from core.turbocore_adamw_variant_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adamw_variant_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adamw_variant_family_batch_scorecard import (  # noqa: E402
    build_adamw_variant_family_batch_scorecard,
)
from core.turbocore_adamw_variant_owner_release_hold_scorecard import (  # noqa: E402
    build_adamw_variant_owner_release_hold_scorecard,
)
from core.turbocore_adamw_variant_product_training_canary_scorecard import (  # noqa: E402
    build_adamw_variant_product_training_canary_scorecard,
)


TARGETS = [
    "AdamW8bit",
    "AdamWScheduleFree",
    "KahanAdamW8bit",
    "PagedAdamW",
    "PagedAdamW32bit",
    "PagedAdamW8bit",
]


def run_smoke() -> dict[str, Any]:
    _refresh_review_artifacts()
    family = build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=True)
    product_canary = build_adamw_variant_product_training_canary_scorecard(family_batch_report=family)
    assert product_canary["representative_product_training_canary_ready"] is True, product_canary
    payload = build_adamw_variant_owner_release_hold_scorecard(
        product_training_canary_report=product_canary,
        workspace_root=REPO_ROOT,
    )
    hold = payload["hold_manifest"]
    frozen = hold["frozen_product_boundaries"]
    assert payload["scorecard"] == "turbocore_adamw_variant_owner_release_hold_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["owner_release_hold_ready"] is True, payload
    assert payload["representative_product_training_canary_ready"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["owner_approval_recorded"] is False, payload
    assert payload["release_approval_recorded"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["summary"]["optimizer_count"] == 6, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert hold["approval_state"] == "pending_owner_and_release_approval", hold
    assert hold["optimizer_types"] == TARGETS, hold
    assert hold["allowed_initial_modes"] == ["off", "observe"], hold
    assert hold["blocked_modes_until_approval"] == ["canary", "auto"], hold
    assert all(value is False for value in frozen.values()), frozen
    assert "adamw_variant_owner_approval_missing" in payload["promotion_blockers"], payload
    assert "adamw_variant_release_approval_missing" in payload["promotion_blockers"], payload
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_owner_release_hold_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _refresh_review_artifacts() -> None:
    e2e = build_adamw_variant_e2e_shadow_matrix_scorecard(include_live_canaries=True)
    rollout = build_adamw_variant_canary_rollout_policy_scorecard(shadow_matrix_report=e2e)
    review = build_adamw_variant_dispatch_integration_review_scorecard(rollout_policy_report=rollout)
    _write_named_artifact("turbocore_adamw_variant_e2e_shadow_matrix_scorecard.json", e2e)
    _write_named_artifact("turbocore_adamw_variant_canary_rollout_policy_scorecard.json", rollout)
    _write_named_artifact("turbocore_adamw_variant_dispatch_integration_review_scorecard.json", review)


def _write_real_artifact(payload: dict[str, Any]) -> None:
    _write_named_artifact("turbocore_adamw_variant_owner_release_hold_scorecard.json", payload)


def _write_named_artifact(filename: str, payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
