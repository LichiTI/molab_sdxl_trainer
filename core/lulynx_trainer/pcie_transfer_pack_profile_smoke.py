"""Smoke test for PCIe transfer pack profiling."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.pcie_transfer_pack_profile import _profile_case  # noqa: E402

import torch


def main() -> int:
    payload = _profile_case(16, 32, source_dtype=torch.float32, iters=1, seed=7)
    assert payload["shape"] == {"rows": 16, "cols": 32}
    assert payload["formats"]["raw_fp16"]["available"] is True
    assert payload["formats"]["raw_fp16"]["full_pack_ms"] >= 0.0
    assert "fp8_e4m3" in payload["formats"]
    print("pcie_transfer_pack_profile_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
