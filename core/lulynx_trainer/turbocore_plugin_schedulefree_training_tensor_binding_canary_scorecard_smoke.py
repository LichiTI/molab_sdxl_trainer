"""Smoke checks for schedule-free plugin training tensor binding canary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_plugin_schedulefree_training_tensor_binding_canary_scorecard import (  # noqa: E402
    BINDING_SCHEMA,
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_schedulefree_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_training_tensor_binding_canary_scorecard()
    cases = {str(case["optimizer_name"]): case for case in report["cases"]}
    assert report["ok"] is True, report
    assert report["training_tensor_binding_canary_ready"] is True, report
    assert report["checkpoint_adapter_proof_ready"] is True, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert set(cases) == set(TARGET_PLUGIN_OPTIMIZERS), cases
    for name, case in cases.items():
        assert case["ok"] is True, case
        assert case["binding_request_shape_ready"] is True, case
        assert case["non_mutating_binding_probe"] is True, case
        assert case["e2e_no_regression"] is True, case
        assert case["max_binding_param_diff"] == 0.0, case
        assert case["max_e2e_param_diff"] <= case["tolerance"], case
        request = case["binding_request"]
        assert request["schema"] == BINDING_SCHEMA, (name, request)
        assert request["readiness"]["request_shape_ready"] is True, (name, request)
        assert request["pointer_exported"] is False, (name, request)
        assert request["training_path_enabled"] is False, (name, request)
        assert request["native_dispatch_allowed"] is False, (name, request)
        assert {"param", "grad"}.issubset(set(request["reported_roles"])), (name, request)
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_training_tensor_binding_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
