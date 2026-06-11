"""Smoke checks for factored/custom owner/release hold scorecard."""

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

from core.turbocore_factored_custom_optimizer_family_batch_scorecard import (  # noqa: E402
    build_factored_custom_optimizer_family_batch_scorecard,
)
from core.turbocore_factored_custom_owner_release_hold_scorecard import (  # noqa: E402
    EXPECTED_OPTIMIZERS,
    build_factored_custom_owner_release_hold_scorecard,
)


def run_smoke() -> dict[str, Any]:
    batch = build_factored_custom_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    report = build_factored_custom_owner_release_hold_scorecard(family_batch_report=batch, write_artifact=True)
    hold = report["hold_manifest"]
    artifact_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_factored_custom_owner_release_hold_scorecard.json"
    assert report["ok"] is True, report
    assert report["owner_release_hold_ready"] is True, report
    assert report["family_batch_ready"] is True, report
    assert report["owner_approval_recorded"] is False, report
    assert report["release_approval_recorded"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert set(hold["optimizer_types"]) == EXPECTED_OPTIMIZERS, hold
    assert hold["approval_state"] == "pending_owner_and_release_approval", hold
    assert hold["allowed_initial_modes"] == ["off", "observe"], hold
    assert hold["blocked_modes_until_approval"] == ["canary", "auto"], hold
    assert all(value is False for value in hold["frozen_product_boundaries"].values()), hold
    assert artifact_path.exists(), artifact_path
    return {
        "schema_version": 1,
        "probe": "turbocore_factored_custom_owner_release_hold_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
