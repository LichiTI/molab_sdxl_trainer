# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test trainer cooldown and GPU power-limit helpers."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.trainer import LulynxTrainer


def _trainer(root: Path) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(root),
        output_name="cooldown_smoke",
        cooldown_every_n_epochs=1,
        cooldown_minutes=0.05,
        cooldown_until_temp=65,
        cooldown_poll_seconds=7,
        gpu_power_limit_w=250,
    )
    trainer._should_stop = False
    trainer._gpu_power_limit_attempted = False
    trainer._log_lines: list[str] = []
    trainer._log = trainer._log_lines.append
    return trainer


def main() -> int:
    root = Path("H:/tmp/lulynx_cooldown_smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    trainer = _trainer(root)

    power_calls: list[tuple[str, ...]] = []
    sleep_calls: list[float] = []

    trainer._hidden_subprocess_kwargs = lambda: {}
    trainer._sleep_for_cooldown = lambda seconds: sleep_calls.append(float(seconds))

    def _fake_run(cmd, **kwargs):
        power_calls.append(tuple(str(part) for part in cmd))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    import subprocess
    original_run = subprocess.run
    subprocess.run = _fake_run  # type: ignore[assignment]
    try:
        trainer._apply_gpu_power_limit_if_requested()
        trainer._apply_gpu_power_limit_if_requested()
    finally:
        subprocess.run = original_run  # type: ignore[assignment]

    assert power_calls == [("nvidia-smi", "-pl", "250")], power_calls
    assert any("GPU power limit applied: 250W" in line for line in trainer._log_lines), trainer._log_lines

    trainer._log_lines.clear()
    trainer._read_gpu_temperature_c = lambda: None
    trainer._maybe_cooldown_after_epoch(0, 3)
    assert sleep_calls == [3.0], sleep_calls
    assert any("falling back to time-based cooldown" in line for line in trainer._log_lines), trainer._log_lines

    trainer._log_lines.clear()
    sleep_calls.clear()
    temps = iter([72, 69, 64])
    trainer._read_gpu_temperature_c = lambda: next(temps)
    trainer._maybe_cooldown_after_epoch(0, 3)
    assert sleep_calls == [7.0, 7.0], sleep_calls
    assert any("Cooldown complete: GPU temperature is 64C" in line for line in trainer._log_lines), trainer._log_lines

    print("Cooldown smoke passed: power-limit application and epoch cooldown behavior are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
