"""Smoke checks for PagedAdamW8bit runtime dispatch adapter shadow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard import (  # noqa: E402
    build_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard(
        run_live_probe=True,
        require_live_matrix=bool(torch.cuda.is_available()),
    )
    route = report["adapter_route"]
    envelope = report["adapter_envelope"]
    assert report["ok"] is True, report
    assert report["runtime_dispatch_adapter_shadow_ready"] is True, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_call_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert route["decision"] == "shadow_adapter_prepared_fallback_authoritative", route
    assert envelope["training_update_authority"] == "python_bitsandbytes", envelope
    assert envelope["native_update_authority"] == "none", envelope
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_runtime_dispatch_adapter_shadow_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
