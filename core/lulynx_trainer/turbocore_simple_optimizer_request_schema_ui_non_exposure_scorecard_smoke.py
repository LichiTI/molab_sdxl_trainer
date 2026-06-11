"""Smoke checks for simple optimizer request/schema/UI non-exposure evidence."""

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

from core.turbocore_simple_optimizer_family_batch_scorecard import (  # noqa: E402
    build_simple_optimizer_family_batch_scorecard,
)
from core.turbocore_simple_optimizer_owner_release_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_owner_release_hold_scorecard,
)
from core.turbocore_simple_optimizer_product_training_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_product_training_canary_scorecard,
)
from core.turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_owner_approval_hold_scorecard,
)
from core.turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    AUDIT_KIND,
    EXPECTED_OPTIMIZERS,
    build_simple_optimizer_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_owner_release_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_owner_release_hold_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    simple_batch = build_simple_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    product_canary = build_simple_optimizer_product_training_canary_scorecard(
        family_batch_report=simple_batch,
        workspace_root=REPO_ROOT,
    )
    owner_hold = build_simple_optimizer_owner_release_hold_scorecard(
        product_training_canary_report=product_canary,
        workspace_root=REPO_ROOT,
    )
    quantized_review = build_simple_optimizer_quantized_dispatch_integration_review_scorecard()
    quantized_hold = build_simple_optimizer_quantized_owner_approval_hold_scorecard(
        dispatch_review_report=quantized_review
    )
    schedulefree_rollout = build_simple_optimizer_schedulefree_rollout_policy_scorecard()
    schedulefree_review = build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard(
        rollout_policy_report=schedulefree_rollout
    )
    schedulefree_hold = build_simple_optimizer_schedulefree_owner_release_hold_scorecard(
        dispatch_review_report=schedulefree_review
    )
    payload = build_simple_optimizer_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=owner_hold,
        quantized_owner_approval_hold_report=quantized_hold,
        schedulefree_owner_release_hold_report=schedulefree_hold,
        workspace_root=REPO_ROOT,
    )
    summary = payload["summary"]
    boundary = payload["boundary_inventory"]
    findings = payload["boundary_findings"]

    assert payload["scorecard"] == "turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["request_schema_ui_non_exposure_ready"] is True, payload
    assert payload["owner_release_hold_ready"] is True, payload
    assert payload["quantized_owner_approval_hold_ready"] is True, payload
    assert payload["schedulefree_owner_release_hold_ready"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["owner_approval_recorded"] is False, payload
    assert payload["release_approval_recorded"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["request_adapter_enabled"] is False, payload
    assert payload["backend_router_registered"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["audit_kind"] == AUDIT_KIND, payload
    assert payload["target_optimizer_types"] == list(EXPECTED_OPTIMIZERS), payload

    assert boundary["present_paths"], boundary
    assert summary["present_boundary_path_count"] == len(boundary["present_paths"]), summary
    assert summary["scanned_file_count"] == len(findings["scanned_files"]), summary
    assert summary["scanned_file_count"] > 0, summary
    assert findings["forbidden_token_hits"] == [], findings
    assert summary["forbidden_token_hit_count"] == 0, summary
    assert summary["optimizer_count"] == 7, summary
    assert summary["request_fields_emitted"] is False, summary
    assert summary["schema_exposure_allowed"] is False, summary
    assert summary["ui_exposure_allowed"] is False, summary
    assert summary["runtime_dispatch_ready"] is False, summary
    assert summary["native_dispatch_allowed"] is False, summary
    assert summary["training_path_enabled"] is False, summary
    assert summary["product_native_ready_count"] == 0, summary

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
