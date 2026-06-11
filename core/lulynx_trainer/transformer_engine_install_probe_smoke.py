# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for TransformerEngine installation probe."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.lulynx_trainer.transformer_engine_install_probe import main


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_smoke() -> int:
    with tempfile.TemporaryDirectory(prefix="te_probe_smoke_") as tmp:
        out = Path(tmp) / "te_probe.json"
        exit_code = main(["--skip-pip", "--out", str(out)])
        assert out.exists(), "probe output missing"
        payload = _load(out)
        assert payload.get("probe") == "transformer_engine_install_probe", payload
        assert "environment" in payload, payload
        assert "decision" in payload, payload
        assert payload.get("commands") == [], payload
        # skip-pip 路径通常不会是可直接可用；这里只验证结构。
        assert exit_code in {0, 1}, exit_code
    print("transformer_engine_install_probe_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke())
