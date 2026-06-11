"""Smoke checks for simple optimizer variant native ABI scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_variant_native_abi_scorecard import (  # noqa: E402
    build_simple_optimizer_variant_native_abi_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_variant_native_abi_scorecard()
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["scorecard"] == "turbocore_simple_optimizer_variant_native_abi_scorecard_v0", report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["summary"]["target_optimizer_count"] == 5, report
    assert report["summary"]["quantized_native_abi_spec_ready_count"] == 3, report
    assert report["summary"]["schedule_free_native_abi_spec_ready_count"] == 2, report
    assert report["summary"]["native_abi_spec_ready_count"] == 5, report
    assert report["summary"]["formula_parity_matrix_artifact_ready_count"] == 5, report
    assert report["summary"]["formula_parity_matrix_implementation_ready_count"] == 0, report
    assert report["summary"]["resume_parity_matrix_artifact_ready_count"] == 5, report
    assert report["summary"]["resume_parity_matrix_implementation_ready_count"] == 0, report
    assert report["summary"]["native_kernel_ready_count"] == 0, report
    assert rows["Lion8bit"]["variant_status"] == "native_abi_spec_ready", rows["Lion8bit"]
    assert rows["PagedLion8bit"]["variant_status"] == "native_abi_spec_ready", rows["PagedLion8bit"]
    assert rows["SGDNesterov8bit"]["variant_status"] == "native_abi_spec_ready", rows["SGDNesterov8bit"]
    assert rows["RAdamScheduleFree"]["variant_kind"] == "schedule_free_state_machine", rows["RAdamScheduleFree"]
    assert rows["SGDScheduleFree"]["variant_status"] == "native_abi_spec_ready", rows["SGDScheduleFree"]
    assert all(row["native_kernel_ready"] is False for row in rows.values()), rows
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_variant_native_abi_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_variant_native_abi_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
