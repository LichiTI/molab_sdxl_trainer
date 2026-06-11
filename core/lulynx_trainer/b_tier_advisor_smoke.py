"""Smoke tests for B-tier research-only advisor reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_advisor_module():
    module_path = Path(__file__).resolve().with_name("training_advisor.py")
    spec = importlib.util.spec_from_file_location("b_tier_advisor_smoke_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    advisor = _load_advisor_module()
    cfg = SimpleNamespace(
        model_arch="anima",
        train_data_dir="",
        hutchinson_auto_freeze=True,
        hutchinson_freeze_ratio=0.3,
        lulynx_hutchinson_probes=16,
        pcgrad_enabled=True,
        pcgrad_conflict_threshold=-0.05,
        pcgrad_reduction="sum",
        lulynx_geometric_lock=True,
        lulynx_proj_dim=192,
        lulynx_manifold_sparse_freq=4,
        lulynx_anchor_layers="mid, out",
        lulynx_ghost_replay=True,
        lulynx_ghost_path="H:/tmp/missing-fingerprint.lulynx",
        lulynx_ghost_interval=80,
        lulynx_ghost_weight=0.1,
    )

    b_tier = advisor.inspect_b_tier_features(cfg)
    assert b_tier["status"] == "mixed"
    assert b_tier["modules"]["hutchinson_scan"]["requested"] is True
    assert b_tier["modules"]["hutchinson_scan"]["num_probes"] == 16
    assert b_tier["modules"]["pcgrad"]["requested"] is True
    assert b_tier["modules"]["pcgrad"]["train_chain_wired"] is True
    assert b_tier["modules"]["pcgrad"]["ui_exposed"] is True
    assert b_tier["modules"]["pcgrad"]["status"] == "manual_experimental"
    assert b_tier["modules"]["ghost_replay"]["fingerprint_configured"] is True
    assert b_tier["modules"]["ghost_replay"]["fingerprint_exists"] is False
    assert b_tier["modules"]["ghost_replay"]["fingerprint_status"] == "missing"
    assert b_tier["modules"]["manifold_constraint"]["status"] == "manual_experimental"
    assert b_tier["modules"]["manifold_constraint"]["train_chain_wired"] is True
    assert b_tier["modules"]["manifold_constraint"]["orchestrator_removed"] is False
    assert b_tier["modules"]["manifold_constraint"]["anchor_layers"] == ["mid", "out"]

    report = advisor.build_training_advisor_report(cfg, available_vram_gb=24.0).to_dict()
    codes = {item["code"] for item in report["findings"]}
    assert "hutchinson_enabled_experimental" in codes
    assert "pcgrad_enabled_experimental" in codes
    assert "ghost_replay_missing_fingerprint" in codes
    assert "manifold_constraint_enabled_experimental" in codes
    assert report["summary"]["b_tier_modules"]["ghost_replay"] is True

    print("b_tier_advisor_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
