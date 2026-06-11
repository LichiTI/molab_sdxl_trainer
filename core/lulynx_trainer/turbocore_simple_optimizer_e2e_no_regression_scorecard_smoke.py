"""Smoke checks for V2-P7 simple optimizer e2e no-regression scorecard."""

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

from core.turbocore_simple_optimizer_e2e_no_regression_scorecard import (  # noqa: E402
    build_simple_optimizer_e2e_no_regression_scorecard,
)
from core.turbocore_simple_optimizer_runtime_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_runtime_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    runtime = build_simple_optimizer_runtime_canary_scorecard()
    assert runtime["runtime_canary_ready"] is True, runtime
    payload = build_simple_optimizer_e2e_no_regression_scorecard(runtime_canary_report=runtime)
    assert payload["scorecard"] == "turbocore_simple_optimizer_e2e_no_regression_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["e2e_no_regression_ready"] is True, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["summary"]["passed_case_count"] == 2, payload
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_e2e_no_regression_scorecard_smoke",
        "ok": True,
        "summary": payload["summary"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
