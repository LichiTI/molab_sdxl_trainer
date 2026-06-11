# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for benchmark_suite.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.benchmark_suite",
    os.path.join(_HERE, "benchmark_suite.py"),
)
_suite = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.benchmark_suite"] = _suite
_spec.loader.exec_module(_suite)


def test_preset_loads_quick_suite():
    root = Path(__file__).resolve().parents[3]
    preset = _suite.SuitePreset.from_path(root / "devtools" / "benchmark_presets" / "quick_training_benchmark.json")
    assert preset.suite_id == "quick_training_benchmark"
    assert len(preset.cases) >= 2
    assert preset.cases[0].id == "step_fn_microbench"
    print("PASS: quick benchmark preset loads")


def test_dry_run_report_is_ok():
    root = Path(__file__).resolve().parents[3]
    preset = _suite.SuitePreset.from_path(root / "devtools" / "benchmark_presets" / "quick_training_benchmark.json")
    runner = _suite.BenchmarkSuiteRunner(preset, root=root)
    report = runner.run(dry_run=True)
    assert report["ok"] is True
    assert report["case_count"] == len(preset.cases)
    assert all(case["skipped"] for case in report["cases"])
    print("PASS: dry-run suite report is ok")


def test_json_summary_extracts_metrics():
    payload = {
        "benchmark": "demo",
        "device": "cpu",
        "results": [
            {
                "backend": "a",
                "success": True,
                "forward_ms": 1.25,
                "peak_allocated_mb": 0.0,
            }
        ],
    }
    summary = _suite._summarize_payload(payload)
    assert summary["benchmark"] == "demo"
    assert summary["results_count"] == 1
    assert summary["successful_results"] == 1
    assert summary["metrics"][0]["forward_ms"] == 1.25
    print("PASS: JSON payload metrics are summarized")


def test_json_summary_extracts_run_metrics_and_losses():
    payload = {
        "benchmark": {"family": "anima", "steps": 2},
        "results": [
            {
                "family": "anima",
                "adapter": "lora",
                "ok": True,
                "resolved_steps": 2,
                "duration_seconds": 3.5,
                "log_tail": ["step 1 Loss: 1.5", "step 2 Loss: 1.0"],
            }
        ],
        "runs": {
            "standard": {
                "success": True,
                "steps_completed": 2,
                "mean_step_ms": 12.5,
                "peak_vram_mb": 128.0,
                "final_loss": 1.0,
            }
        },
    }
    summary = _suite._summarize_payload(payload)
    assert summary["metrics"][0]["loss_last"] == 1.0
    assert summary["run_metrics"]["standard"]["mean_step_ms"] == 12.5
    print("PASS: training run metrics and losses are summarized")


def test_research_preset_gates_longrun_cases():
    root = Path(__file__).resolve().parents[3]
    preset = _suite.SuitePreset.from_path(root / "devtools" / "benchmark_presets" / "training_research_benchmark.json")
    runner = _suite.BenchmarkSuiteRunner(preset, root=root)
    default_report = runner.run(dry_run=True)
    assert any(item["skip_reason"] == "gated_by_tag" for item in default_report["skipped_cases"])
    assert not any("longrun" in case["tags"] for case in default_report["cases"])
    longrun_report = runner.run(dry_run=True, include_tags=["longrun"])
    assert any("longrun" in case["tags"] for case in longrun_report["cases"])
    print("PASS: research preset gates long-run cases by tag")


def test_parse_json_text_from_stdout():
    parsed = _suite._parse_json_text("header\n[{\"ok\": true}]\nfooter")
    assert isinstance(parsed, list)
    assert parsed[0]["ok"] is True
    print("PASS: stdout JSON payload can be extracted")


def test_main_writes_dry_run_report():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "suite_report.json"
        rc = _suite.main([
            "--preset",
            str(root / "devtools" / "benchmark_presets" / "quick_training_benchmark.json"),
            "--out",
            str(out),
            "--dry-run",
        ])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["ok"] is True
        assert data["suite_id"] == "quick_training_benchmark"
    print("PASS: CLI writes dry-run report")


if __name__ == "__main__":
    test_preset_loads_quick_suite()
    test_dry_run_report_is_ok()
    test_json_summary_extracts_metrics()
    test_json_summary_extracts_run_metrics_and_losses()
    test_research_preset_gates_longrun_cases()
    test_parse_json_text_from_stdout()
    test_main_writes_dry_run_report()
    print("\nAll benchmark suite smoke tests passed!")
