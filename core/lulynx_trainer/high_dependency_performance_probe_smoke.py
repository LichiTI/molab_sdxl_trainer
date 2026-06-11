# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test the high dependency performance probe."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.lulynx_trainer.high_dependency_performance_probe import build_report  # noqa: E402


def main() -> int:
    report = build_report(
        Namespace(
            images=2,
            image_size=32,
            decode_repeats=1,
            device="auto",
            compile_iters=1,
            compile_warmup=0,
            skip_compile=True,
        )
    )
    assert report["ok"] is True, report
    assert report["non_invasive"] is True, report
    assert "capabilities" in report, report
    assert "fp8_transformer_engine" in report, report
    assert "data_decode" in report, report
    assert report["torch_compile_reduce_overhead"]["skipped"] is True, report
    backends = report["data_decode"]["backends"]
    assert backends["pil"]["ok"] is True, backends
    assert backends["pil_lru"]["ok"] is True, backends
    print("high_dependency_performance_probe_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
