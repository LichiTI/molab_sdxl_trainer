"""Smoke tests for Coreset reporting and classification settings."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.training_components.coreset import CoresetManager


def main() -> int:
    manager = CoresetManager(
        easy_weight=0.25,
        hard_weight=2.5,
        toxic_weight=0.0,
        auto_classify_after=1,
        easy_threshold=0.01,
        hard_loss_threshold=1.0,
        toxic_std_threshold=0.5,
        min_history=5,
    )

    for loss in [0.5, 0.4, 0.3, 0.2, 0.1]:
        manager.update_sample("easy.png", loss)
    for loss in [1.2, 1.205, 1.198, 1.202, 1.201]:
        manager.update_sample("hard.png", loss)
    for loss in [0.1, 1.8, 0.2, 1.7, 0.15]:
        manager.update_sample("toxic.png", loss)
    for loss in [0.5, 0.51, 0.5, 0.49, 0.5]:
        manager.update_sample("normal.png", loss)

    manager.on_epoch_end()
    report = manager.to_report(top_k=3)
    cats = report["summary"]["categories"]
    assert cats["easy"] == 1, cats
    assert cats["hard"] == 1, cats
    assert cats["toxic"] == 1, cats
    assert manager.get_weight("easy.png") == 0.25
    assert manager.get_weight("hard.png") == 2.5
    assert manager.get_weight("toxic.png") == 0.0
    assert report["top_hard"][0]["filename"] == "hard.png"
    assert report["top_toxic"][0]["filename"] == "toxic.png"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coreset_report.json"
        saved = manager.save_report(str(path), top_k=2)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert saved["summary"] == loaded["summary"]
        assert "samples" in loaded

    print("coreset_report_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
