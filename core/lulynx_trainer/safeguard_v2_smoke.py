"""Smoke tests for SafeGuard v2 report-first bad-sample handling."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def _load_safe_guard_module():
    module_path = Path(__file__).resolve().with_name("safe_guard.py")
    project_root = module_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location(
        "core.lulynx_trainer.safe_guard_smoke_target",
        module_path,
        submodule_search_locations=[],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core.lulynx_trainer"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prime_loss_history(sg) -> None:
    for step in range(10):
        action = sg.check(step=step, loss=1.0, lr=1e-4, filenames=[])
        assert action.value == "continue"


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_report_mode_does_not_move(module, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    sample = tmp / "sample.png"
    caption = tmp / "sample.txt"
    sample.write_bytes(b"x")
    caption.write_text("tag", encoding="utf-8")
    report = tmp / "events.jsonl"
    sg = module.TrainingSafeGuard(
        module.SafeGuardConfig(
            enable_bad_sample_culling=True,
            bad_sample_mode="report",
            bad_sample_report_path=str(report),
            loss_spike_threshold=2.0,
        )
    )
    _prime_loss_history(sg)
    action = sg.check(step=10, loss=10.0, lr=1e-4, filenames=[str(sample)])
    assert action.value == "reduce_lr"
    assert sample.exists(), "report mode must not move image"
    assert caption.exists(), "report mode must not move caption"
    events = _read_jsonl(report)
    assert len(events) == 1
    assert events[0]["reason"] == "loss_spike"
    assert events[0]["mode"] == "report"
    assert events[0]["samples"] == [str(sample)]
    stats = sg.get_stats()
    assert stats["bad_sample_event_count"] == 1


def test_move_mode_quarantines(module, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    sample = tmp / "move.png"
    caption = tmp / "move.txt"
    sample.write_bytes(b"x")
    caption.write_text("tag", encoding="utf-8")
    quarantine = tmp / "quarantine"
    report = tmp / "move_events.jsonl"
    culled = []
    sg = module.TrainingSafeGuard(
        module.SafeGuardConfig(
            enable_bad_sample_culling=True,
            bad_sample_mode="move",
            quarantine_dir=str(quarantine),
            bad_sample_report_path=str(report),
            loss_spike_threshold=2.0,
            on_cull_samples=lambda names: culled.extend(names),
        )
    )
    _prime_loss_history(sg)
    action = sg.check(step=10, loss=10.0, lr=1e-4, filenames=[str(sample)])
    assert action.value == "reduce_lr"
    assert not sample.exists(), "move mode should move image"
    assert (quarantine / "move.png").exists()
    assert (quarantine / "move.txt").exists()
    assert culled == [str(sample)]
    events = _read_jsonl(report)
    assert events[0]["mode"] == "move"


def main() -> int:
    module = _load_safe_guard_module()
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        test_report_mode_does_not_move(module, root / "report")
        test_move_mode_quarantines(module, root / "move")
    print("safeguard_v2_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
