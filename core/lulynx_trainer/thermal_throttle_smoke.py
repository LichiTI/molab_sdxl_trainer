# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test the step-boundary GPU duty-cycle / temperature governor.

Uses injected fake clock/sleep/temperature so it runs without a GPU:
* manual duty cycle sleeps the right share of measured busy time,
* default config (duty=1.0, target=0) produces zero behaviour change,
* the temperature PI loop converges duty downward when hot and recovers
  when cool, staying put inside the deadband,
* the trainer mixin factory returns None when both knobs are off.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.trainer_thermal import (
    GpuDutyCycleThrottler,
    TrainerThermalMixin,
)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def busy(self, seconds: float) -> None:
        self.now += seconds


def _make(ft: FakeTime, **kwargs) -> GpuDutyCycleThrottler:
    kwargs.setdefault("log", None)
    return GpuDutyCycleThrottler(sleep_fn=ft.sleep, clock=ft.clock, **kwargs)


def test_manual_duty_sleeps_proportionally() -> None:
    ft = FakeTime()
    throttler = _make(ft, duty_cycle=0.5)
    assert throttler.enabled
    assert throttler.on_step_boundary() is None  # first boundary only arms the clock
    ft.busy(1.0)
    report = throttler.on_step_boundary()
    # duty 0.5 -> sleep == busy
    assert abs(report["sleep_s"] - 1.0) < 1e-6, report
    assert abs(report["busy_s"] - 1.0) < 1e-6, report
    ft.busy(0.5)
    report = throttler.on_step_boundary()
    assert abs(report["sleep_s"] - 0.5) < 1e-6, report
    # duty 0.8 -> sleep = busy * 0.25
    ft2 = FakeTime()
    t2 = _make(ft2, duty_cycle=0.8)
    t2.on_step_boundary()
    ft2.busy(2.0)
    report = t2.on_step_boundary()
    assert abs(report["sleep_s"] - 0.5) < 1e-6, report


def test_disabled_throttler_never_sleeps() -> None:
    ft = FakeTime()
    throttler = _make(ft, duty_cycle=1.0, target_temp_c=0)
    assert not throttler.enabled
    throttler.on_step_boundary()
    for _ in range(5):
        ft.busy(1.0)
        report = throttler.on_step_boundary()
        assert report["sleep_s"] == 0.0
    assert ft.slept == []


def test_sleep_is_capped() -> None:
    ft = FakeTime()
    throttler = _make(ft, duty_cycle=0.2, max_sleep_s=3.0)
    throttler.on_step_boundary()
    ft.busy(10.0)  # uncapped would sleep 40s
    report = throttler.on_step_boundary()
    assert abs(report["sleep_s"] - 3.0) < 1e-6, report


def test_temperature_pid_converges_and_recovers() -> None:
    ft = FakeTime()
    temp_box = {"value": 90}
    throttler = _make(
        ft,
        target_temp_c=75,
        read_temperature_c=lambda: temp_box["value"],
        temp_poll_interval_s=0.0,  # poll every boundary in the fake timeline
    )
    throttler.on_step_boundary()
    # hot: duty must fall below 1.0 and keep falling toward min_duty
    duties = []
    for _ in range(10):
        ft.busy(1.0)
        duties.append(throttler.on_step_boundary()["duty"])
    assert duties[0] < 1.0
    assert duties[-1] <= duties[0]
    assert duties[-1] >= throttler.min_duty - 1e-9

    # cool well below target: duty must recover upward
    temp_box["value"] = 60
    recovered = []
    for _ in range(40):
        ft.busy(1.0)
        recovered.append(throttler.on_step_boundary()["duty"])
    assert recovered[-1] > duties[-1]

    # inside deadband: duty holds steady
    temp_box["value"] = 75
    ft.busy(1.0)
    base = throttler.on_step_boundary()["duty"]
    for _ in range(5):
        ft.busy(1.0)
        assert abs(throttler.on_step_boundary()["duty"] - base) < 1e-9


def test_manual_duty_is_ceiling_for_pid() -> None:
    ft = FakeTime()
    throttler = _make(
        ft,
        duty_cycle=0.6,
        target_temp_c=75,
        read_temperature_c=lambda: 50,  # cold -> PID wants 1.0
        temp_poll_interval_s=0.0,
    )
    throttler.on_step_boundary()
    ft.busy(1.0)
    report = throttler.on_step_boundary()
    assert abs(report["duty"] - 0.6) < 1e-9, report


class _FactoryHost(TrainerThermalMixin):
    def __init__(self, config) -> None:
        self.config = config
        self.logs: list[str] = []

    def _log(self, message: str) -> None:
        self.logs.append(message)


def test_mixin_factory_off_by_default() -> None:
    host = _FactoryHost(SimpleNamespace(gpu_duty_cycle=1.0, gpu_target_temp_c=0))
    assert host._create_thermal_throttler() is None

    host = _FactoryHost(SimpleNamespace(gpu_duty_cycle=0.7, gpu_target_temp_c=0))
    throttler = host._create_thermal_throttler()
    assert throttler is not None and throttler.enabled
    assert any("GPU throttle enabled" in line for line in host.logs)

    host = _FactoryHost(SimpleNamespace(gpu_duty_cycle=1.0, gpu_target_temp_c=75))
    throttler = host._create_thermal_throttler()
    assert throttler is not None and throttler.target_temp_c == 75


def main() -> int:
    test_manual_duty_sleeps_proportionally()
    test_disabled_throttler_never_sleeps()
    test_sleep_is_capped()
    test_temperature_pid_converges_and_recovers()
    test_manual_duty_is_ceiling_for_pid()
    test_mixin_factory_off_by_default()
    print("thermal_throttle_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
