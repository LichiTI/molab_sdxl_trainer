"""Smoke checks for V2-P7 simple optimizer ABI scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_abi_scorecard import build_simple_optimizer_abi_scorecard  # noqa: E402
from core.turbocore_simple_optimizer_reference_scorecard import (  # noqa: E402
    build_simple_optimizer_reference_scorecard,
)


def run_smoke() -> dict[str, Any]:
    reference = build_simple_optimizer_reference_scorecard(dtype_cases=("float32",))
    report = build_simple_optimizer_abi_scorecard(reference_report=reference)
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["ok"] is True, report
    assert report["first_abi_stage_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["summary"]["abi_ready_optimizer_count"] == 2, report
    assert rows["Lion"]["optimizer_kind"] == "lion", rows["Lion"]
    assert rows["Lion"]["abi_status"] == "optimizer_kind_contract_ready", rows["Lion"]
    assert rows["SGDNesterov"]["optimizer_kind"] == "sgd_nesterov", rows["SGDNesterov"]
    assert rows["SGDNesterov"]["abi_status"] == "optimizer_kind_contract_ready", rows["SGDNesterov"]
    assert rows["Lion8bit"]["abi_status"] == "quantized_state_layout_pending", rows["Lion8bit"]
    assert rows["PagedLion8bit"]["abi_status"] == "paged_quantized_state_layout_pending", rows["PagedLion8bit"]
    assert rows["SGDNesterov8bit"]["abi_status"] == "quantized_momentum_layout_pending", rows["SGDNesterov8bit"]
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_abi_scorecard_smoke",
        "ok": True,
        "first_abi_stage_ready": report["first_abi_stage_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
