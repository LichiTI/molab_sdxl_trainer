"""Smoke checks for native WebDataset promotion scorecard."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_webdataset_promotion_scorecard import (  # noqa: E402
    build_webdataset_materializer_promotion_scorecard,
)


def test_header_validation_gain_does_not_promote_default() -> None:
    report = build_webdataset_materializer_promotion_scorecard(
        native_default_enabled=True,
        validation_mode="header",
        benchmark_report={"header_speedup_vs_python": 1.3, "native_tar_passes": 1},
    )
    reasons = set(report["promotion_blockers"])
    assert report["scorecard"] == "native_webdataset_materializer_promotion_scorecard_v0", report
    assert report["header_validation_has_benchmark_gain"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert "native_webdataset_header_validation_weaker_than_pil_verify" in reasons, report
    assert "pil_verify_equivalent_native_default_not_promoted" in reasons, report


def test_pil_mode_still_requires_explicit_default_promotion() -> None:
    report = build_webdataset_materializer_promotion_scorecard(
        native_default_enabled=False,
        validation_mode="pil",
        benchmark_report={"pil_verify_equivalent": True, "pil_verify_speedup_vs_python": 1.0},
    )
    reasons = set(report["promotion_blockers"])
    assert report["pil_verify_equivalent_mode_default"] is True, report
    assert "native_webdataset_default_disabled" in reasons, report
    assert "pil_verify_equivalent_native_default_not_promoted" not in reasons, report
    assert report["default_training_path_ready"] is False, report


def test_pil_equivalent_native_default_promotes_training_path() -> None:
    report = build_webdataset_materializer_promotion_scorecard(
        native_default_enabled=True,
        validation_mode="pil",
        benchmark_report={
            "pil_verify_equivalent": True,
            "pil_verify_speedup_vs_python": 1.02,
            "native_tar_passes": 1,
        },
    )
    assert report["promotion_ready"] is True, report
    assert report["default_training_path_ready"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["promotion_blockers"] == [], report


def main() -> int:
    test_header_validation_gain_does_not_promote_default()
    test_pil_mode_still_requires_explicit_default_promotion()
    test_pil_equivalent_native_default_promotes_training_path()
    print("webdataset_promotion_scorecard_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
