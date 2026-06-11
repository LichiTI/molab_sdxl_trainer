# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test TransformerEngine FP8 capability profile."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.fp8_te_profile import (  # noqa: E402
    build_fp8_te_profile,
    normalize_precision_experiment,
)


def main() -> int:
    assert normalize_precision_experiment("fp8") == "fp8_te"
    assert normalize_precision_experiment("TransformerEngine") == "fp8_te"
    assert normalize_precision_experiment("unknown") == "bf16"

    default_profile = build_fp8_te_profile(SimpleNamespace()).as_dict()
    assert default_profile["requested"] is False, default_profile
    assert default_profile["resolved"] == "bf16", default_profile
    assert "cuda_available" in default_profile["capabilities"], default_profile

    requested_profile = build_fp8_te_profile(SimpleNamespace(precision_experiment="fp8_te")).as_dict()
    assert requested_profile["requested"] is True, requested_profile
    assert requested_profile["resolved"] in {"bf16", "fp8_te"}, requested_profile
    assert "transformer_engine_available" in requested_profile["capabilities"], requested_profile
    if requested_profile["resolved"] == "bf16":
        assert requested_profile["fallback_reason"], requested_profile
    else:
        assert requested_profile["available"] is True, requested_profile

    legacy_flag = build_fp8_te_profile(SimpleNamespace(fp8_te_enabled=True)).as_dict()
    assert legacy_flag["requested"] is True, legacy_flag

    print("fp8_te_profile_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
