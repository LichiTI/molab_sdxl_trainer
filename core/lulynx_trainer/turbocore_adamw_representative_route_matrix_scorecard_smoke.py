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

from core.turbocore_adamw_representative_route_matrix_scorecard import (  # noqa: E402
    build_adamw_representative_route_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_representative_route_matrix_scorecard(steps=4)
    assert report["ok"] is True, report
    assert report["representative_route_matrix_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert report["native_backend"] == "rust_cuda_adamw_v0", report
    assert report["summary"]["route_row_count"] == 3, report
    assert report["summary"]["route_row_ready_count"] == 3, report
    assert report["summary"]["baseline_native_steps"] == 0, report
    assert report["summary"]["canary_native_steps_count"] > 0, report
    assert report["summary"]["canary_owner_backend"] == "rust_cuda_adamw_v0", report
    assert report["blocked_reasons"] == [], report
    artifact = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adamw_representative_route_matrix_scorecard.json"
    loaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert loaded["representative_route_matrix_ready"] is True, loaded
    assert loaded["native_dispatch_allowed"] is False, loaded
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_representative_route_matrix_scorecard_smoke",
        "ok": True,
        "route_row_ready_count": report["summary"]["route_row_ready_count"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
