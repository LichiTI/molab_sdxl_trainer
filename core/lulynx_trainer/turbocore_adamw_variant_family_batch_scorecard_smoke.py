"""Smoke checks for AdamW variant family batch scorecard."""

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

from core.turbocore_adamw_variant_family_batch_scorecard import (  # noqa: E402
    build_adamw_variant_family_batch_scorecard,
)
from core.turbocore_adamw_variant_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adamw_variant_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adamw_variant_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adamw_variant_canary_rollout_policy_scorecard,
)
from core.turbocore_adamw_variant_dispatch_integration_review_scorecard import (  # noqa: E402
    build_adamw_variant_dispatch_integration_review_scorecard,
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
    payload = build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=True)
    assert payload["scorecard"] == "turbocore_adamw_variant_family_batch_scorecard_v0", payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["summary"]["target_count"] == len(TARGETS), payload
    assert payload["summary"]["native_ready_count"] == 6, payload
    assert payload["summary"]["native_canary_stage_evidence_ready_count"] == 6, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert payload["native_ready_count_policy"]["counts_exact_adamw"] is False, payload
    assert payload["native_ready_count_policy"]["counts_product_default_dispatch"] is False, payload
    assert payload["summary"]["exact_adamw_included"] is False, payload
    assert payload["summary"]["e2e_shadow_matrix_ready"] is True, payload
    assert payload["summary"]["canary_rollout_policy_ready"] is True, payload
    assert payload["summary"]["dispatch_integration_review_ready"] is True, payload
    assert payload["e2e_shadow_matrix"]["e2e_shadow_matrix_ready"] is True, payload
    assert payload["canary_rollout_policy"]["canary_rollout_policy_ready"] is True, payload
    assert payload["dispatch_integration_review"]["review_gate_ready"] is True, payload
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    assert set(rows) == set(TARGETS), rows
    assert rows["AdamWScheduleFree"]["batch_status"] == "native_canary_ready", rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["native_ready"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["stage_ready"]["schedule_free_scratch_formula_canary"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["stage_ready"]["schedule_free_native_scratch_kernel"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["stage_ready"]["schedule_free_runtime_canary"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["stage_ready"]["schedule_free_training_loop_canary"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["training_loop_canary_ready"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamWScheduleFree"]["native_canary_manifest_present"] is True, rows["AdamWScheduleFree"]
    assert rows["AdamW8bit"]["native_canary_manifest_present"] is True, rows["AdamW8bit"]
    assert rows["PagedAdamW"]["native_canary_manifest_present"] is True, rows["PagedAdamW"]
    assert rows["PagedAdamW32bit"]["native_canary_manifest_present"] is True, rows["PagedAdamW32bit"]
    assert rows["PagedAdamW8bit"]["native_canary_manifest_present"] is True, rows["PagedAdamW8bit"]
    assert rows["KahanAdamW8bit"]["native_canary_manifest_present"] is True, rows["KahanAdamW8bit"]
    assert rows["AdamW8bit"]["training_loop_canary_ready"] is True, rows["AdamW8bit"]
    assert rows["PagedAdamW"]["training_loop_canary_ready"] is True, rows["PagedAdamW"]
    assert rows["PagedAdamW32bit"]["training_loop_canary_ready"] is True, rows["PagedAdamW32bit"]
    assert rows["PagedAdamW8bit"]["training_loop_canary_ready"] is True, rows["PagedAdamW8bit"]
    assert rows["KahanAdamW8bit"]["training_loop_canary_ready"] is True, rows["KahanAdamW8bit"]
    assert rows["AdamW8bit"]["batch_status"] == "native_canary_ready", rows["AdamW8bit"]
    assert rows["PagedAdamW"]["batch_status"] == "native_canary_ready", rows["PagedAdamW"]
    assert rows["PagedAdamW32bit"]["batch_status"] == "native_canary_ready", rows["PagedAdamW32bit"]
    assert rows["PagedAdamW8bit"]["batch_status"] == "native_canary_ready", rows["PagedAdamW8bit"]
    assert rows["KahanAdamW8bit"]["batch_status"] == "native_canary_ready", rows["KahanAdamW8bit"]
    assert rows["AdamWScheduleFree"]["batch_status"] == "native_canary_ready", rows["AdamWScheduleFree"]
    assert payload["summary"]["native_canary_manifest_count"] == 6, payload
    assert payload["summary"]["state_reference_ready_count"] == 6, payload
    assert payload["summary"]["native_canary_manifest_ready_count"] == 6, payload
    assert payload["summary"]["training_loop_canary_ready_count"] == 6, payload
    assert payload["summary"]["schedule_free_native_abi_ready_count"] == 1, payload
    assert payload["summary"]["schedule_free_scratch_formula_canary_ready_count"] == 1, payload
    assert payload["summary"]["schedule_free_native_scratch_kernel_ready_count"] == 1, payload
    assert payload["summary"]["schedule_free_runtime_canary_manifest_ready_count"] == 1, payload
    assert payload["summary"]["schedule_free_training_loop_canary_manifest_ready_count"] == 1, payload
    assert payload["summary"]["pending_count"] == 0, payload
    _write_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_family_batch_scorecard_smoke",
        "ok": True,
        "artifact": "temp/turbocore_optimizer/turbocore_adamw_variant_family_batch_scorecard.json",
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adamw_variant_family_batch_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_review_artifacts() -> None:
    e2e = build_adamw_variant_e2e_shadow_matrix_scorecard(include_live_canaries=True)
    rollout = build_adamw_variant_canary_rollout_policy_scorecard(shadow_matrix_report=e2e)
    review = build_adamw_variant_dispatch_integration_review_scorecard(rollout_policy_report=rollout)
    _write_named_artifact("turbocore_adamw_variant_e2e_shadow_matrix_scorecard.json", e2e)
    _write_named_artifact("turbocore_adamw_variant_canary_rollout_policy_scorecard.json", rollout)
    _write_named_artifact("turbocore_adamw_variant_dispatch_integration_review_scorecard.json", review)


def _write_named_artifact(filename: str, payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
