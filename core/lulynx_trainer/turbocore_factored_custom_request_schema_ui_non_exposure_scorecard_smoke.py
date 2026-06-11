"""Smoke checks for factored/custom request/schema/UI non-exposure audit."""

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
    build_factored_custom_owner_release_hold_scorecard,
)
from core.turbocore_factored_custom_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_factored_custom_request_schema_ui_non_exposure_scorecard,
)


def run_smoke() -> dict[str, Any]:
    batch = build_factored_custom_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    hold = build_factored_custom_owner_release_hold_scorecard(family_batch_report=batch)
    report = build_factored_custom_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=hold,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["request_schema_ui_non_exposure_ready"] is True, report
    assert report["owner_release_hold_ready"] is True, report
    assert report["owner_approval_recorded"] is False, report
    assert report["release_approval_recorded"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["request_adapter_enabled"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert summary["optimizer_count"] == 3, summary
    assert summary["present_boundary_path_count"] > 0, summary
    assert summary["scanned_file_count"] > 0, summary
    assert summary["forbidden_token_hit_count"] == 0, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_factored_custom_request_schema_ui_non_exposure_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
