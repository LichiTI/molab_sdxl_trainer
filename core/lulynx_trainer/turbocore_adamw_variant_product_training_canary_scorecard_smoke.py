"""Smoke checks for AdamW variant representative product-training canary."""

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
from core.turbocore_adamw_variant_product_training_canary_scorecard import (  # noqa: E402
    build_adamw_variant_product_training_canary_scorecard,
)


TARGETS = (
    "AdamW8bit",
    "PagedAdamW",
    "PagedAdamW32bit",
    "PagedAdamW8bit",
    "KahanAdamW8bit",
    "AdamWScheduleFree",
)


def run_smoke() -> dict[str, Any]:
    _refresh_review_artifacts()
    family = build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=True)
    payload = build_adamw_variant_product_training_canary_scorecard(family_batch_report=family)
    assert payload["scorecard"] == "turbocore_adamw_variant_product_training_canary_scorecard_v0", payload
    assert payload["representative_product_training_canary_ready"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["owner_approval_recorded"] is False, payload
    assert payload["release_approval_recorded"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["summary"]["target_optimizer_count"] == len(TARGETS), payload
    assert payload["summary"]["representative_product_training_canary_ready_count"] == len(TARGETS), payload
    assert payload["summary"]["ready_required_family_gate_count"] == 4, payload
    assert payload["summary"]["native_canary_stage_evidence_ready_count"] == 6, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    assert set(rows) == set(TARGETS), rows
    for name in TARGETS:
        row = rows[name]
        assert row["canary_status"] == "representative_product_training_canary_ready", row
        assert row["state_reference_ready"] is True, row
        assert row["native_canary_manifest_ready"] is True, row
        assert row["training_loop_canary_ready"] is True, row
        assert row["native_canary_stage_ready"] is True, row
        assert row["product_native_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
    assert payload["blocked_reasons"] == [], payload
    _write_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_product_training_canary_scorecard_smoke",
        "ok": True,
        "artifact": "temp/turbocore_optimizer/turbocore_adamw_variant_product_training_canary_scorecard.json",
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


def _write_artifact(payload: dict[str, Any]) -> None:
    _write_named_artifact("turbocore_adamw_variant_product_training_canary_scorecard.json", payload)


def _write_named_artifact(filename: str, payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
