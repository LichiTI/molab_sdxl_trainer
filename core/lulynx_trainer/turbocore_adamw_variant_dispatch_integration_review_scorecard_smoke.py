"""Smoke checks for AdamW variant dispatch integration review package."""

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

from core.turbocore_adamw_variant_dispatch_integration_review_scorecard import (  # noqa: E402
    build_adamw_variant_dispatch_integration_review_scorecard,
)


EXPECTED = {
    "AdamW8bit",
    "PagedAdamW",
    "PagedAdamW32bit",
    "PagedAdamW8bit",
    "KahanAdamW8bit",
    "AdamWScheduleFree",
}


def run_smoke() -> dict[str, Any]:
    report = build_adamw_variant_dispatch_integration_review_scorecard()
    review = report["review_package"]
    assert report["scorecard"] == "turbocore_adamw_variant_dispatch_integration_review_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["dispatch_integration_review"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert set(review["optimizer_types"]) == EXPECTED, review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(report: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adamw_variant_dispatch_integration_review_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
