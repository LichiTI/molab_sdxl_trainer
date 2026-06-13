"""Smoke for bridge-created selected plugin TrainingLoop native canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_plugin_bridge_training_loop_canary_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_bridge_training_loop_canary_scorecard,
)


EXPECTED_OPTIMIZERS = {
    "a2grad",
    "adahessian",
    "adafactor",
    "adadelta",
    "adabelief",
    "adabound",
    "adagc",
    "adai",
    "adalite",
    "adan",
    "adanorm",
    "adopt",
    "ademamix",
    "adapnm",
    "adasmooth",
    "adashift",
    "adammini",
    "adago",
    "adamuon",
    "adatam",
    "alig",
    "alice",
    "aida",
    "amos",
    "ano",
    "apollo",
    "apollodqn",
    "avagrad",
    "bcos",
    "came",
    "conda",
    "demo",
    "diffgrad",
    "distributedmuon",
    "emofact",
    "emonavi",
    "emolynx",
    "fira",
    "focus",
    "ftrl",
    "galore",
    "grams",
    "kate",
    "kron",
    "laprop",
    "lorarite",
    "mars",
    "msvag",
    "muon",
    "pnm",
    "racs",
    "rose",
    "sgdsai",
    "scion",
    "scionlight",
    "scalableshampoo",
    "shampoo",
    "simplifiedademamix",
    "sm3",
    "soap",
    "spam",
    "sophiah",
    "spectralsphere",
    "splus",
    "srmm",
    "stablespam",
    "swats",
    "tam",
}


def run_smoke() -> dict[str, Any]:
    report = build_plugin_bridge_training_loop_canary_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["roadmap"] == ROADMAP, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["ok"] is True, report
    assert summary["case_count"] == len(EXPECTED_OPTIMIZERS), report
    assert summary["native_step_count"] == len(EXPECTED_OPTIMIZERS), report
    assert summary["native_kernel_launch_count"] == len(EXPECTED_OPTIMIZERS), report
    assert summary["training_executor_called_count"] == len(EXPECTED_OPTIMIZERS), report
    assert summary["skip_pytorch_count"] == len(EXPECTED_OPTIMIZERS), report
    cases = {str(case["selected_optimizer_name"]): case for case in report["cases"]}
    assert set(cases) == EXPECTED_OPTIMIZERS, cases
    for name, case in cases.items():
        assert case["ok"] is True, case
        assert case["native_step_executed"] is True, case
        assert case["native_kernel_launched"] is True, case
        assert case["training_executor_called"] is True, case
        assert case["training_executor_ok"] is True, case
        assert case["should_call_pytorch_optimizer_step"] is False, case
        assert case["executor_optimizer_kind"] == name, case
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_bridge_training_loop_canary_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
