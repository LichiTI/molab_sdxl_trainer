"""Smoke checks for adaptive-LR request/schema/UI non-exposure evidence."""

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

from core.turbocore_adaptive_lr_owner_release_hold_scorecard import (  # noqa: E402
    build_adaptive_lr_owner_release_hold_scorecard,
)
from core.turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    AUDIT_KIND,
    build_adaptive_lr_request_schema_ui_non_exposure_scorecard,
)


def run_smoke() -> dict[str, Any]:
    hold = build_adaptive_lr_owner_release_hold_scorecard()
    payload = build_adaptive_lr_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=hold,
        workspace_root=REPO_ROOT,
    )
    summary = payload["summary"]
    boundary = payload["boundary_inventory"]
    findings = payload["boundary_findings"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["request_schema_ui_non_exposure_ready"] is True, payload
    assert payload["owner_release_hold_ready"] is True, payload
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

    assert boundary["present_paths"], boundary
    assert summary["present_boundary_path_count"] == len(boundary["present_paths"]), summary
    assert summary["scanned_file_count"] == len(findings["scanned_files"]), summary
    assert summary["scanned_file_count"] > 0, summary
    assert findings["forbidden_token_hits"] == [], findings
    assert summary["forbidden_token_hit_count"] == 0, summary
    assert summary["optimizer_count"] == 11, summary
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
        "probe": "turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
