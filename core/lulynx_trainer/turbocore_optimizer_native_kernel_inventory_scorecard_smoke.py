"""Batch smoke for the static TurboCore optimizer kernel inventory."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_native_kernel_inventory_scorecard.json"
)


def run_smoke(*, rebuild_artifact: bool = False) -> dict[str, Any]:
    report = _artifact_or_build(rebuild_artifact=rebuild_artifact)
    summary = report["summary"]

    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design.md", report
    assert report["ok"] is True, report
    assert report["static_inventory_only"] is True, report
    assert report["cuda_executed"] is False, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert summary["plugin_optimizer_count"] == 124, report
    assert summary["family_count"] == 10, report
    assert summary["kernel_source_present_count"] == 124, report
    assert summary["rust_probe_present_count"] == 124, report
    assert summary["product_native_ready_count"] == 0, report
    assert summary["training_path_enabled_count"] == 0, report
    assert summary["native_dispatch_allowed_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_native_kernel_inventory_scorecard_smoke",
        "ok": True,
        "artifact_mode": "rebuild" if rebuild_artifact else "artifact_first",
        "roadmap": report["roadmap"],
        "summary": summary,
    }


def _artifact_or_build(*, rebuild_artifact: bool) -> dict[str, Any]:
    if not rebuild_artifact and ARTIFACT.exists():
        try:
            report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
            if isinstance(report, dict) and report.get("ok") is True:
                return report
        except (OSError, json.JSONDecodeError):
            pass
    module = importlib.import_module("core.turbocore_optimizer_native_kernel_inventory_scorecard")
    return module.build_optimizer_native_kernel_inventory_scorecard(write_artifact=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-artifact", action="store_true")
    args = parser.parse_args(argv)
    payload = run_smoke(rebuild_artifact=bool(args.rebuild_artifact))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
