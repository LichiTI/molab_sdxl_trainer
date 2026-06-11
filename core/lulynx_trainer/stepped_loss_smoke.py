# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for stepped loss schedule."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_sl = _import_module(
    "stepped_loss",
    os.path.join(_HERE, "stepped_loss.py"),
)
SteppedLossSchedule = _sl.SteppedLossSchedule


def test_resolve_at_boundaries():
    schedule_json = (
        '[{"step":0,"loss_type":"l2","weight":1.0},'
        '{"step":500,"loss_type":"huber","weight":0.5}]'
    )
    sched = SteppedLossSchedule(schedule_json)
    assert sched.resolve(0) == ("l2", 1.0), f"Expected ('l2', 1.0) at step 0, got {sched.resolve(0)}"
    assert sched.resolve(499) == ("l2", 1.0), f"Expected ('l2', 1.0) at step 499, got {sched.resolve(499)}"
    assert sched.resolve(500) == ("huber", 0.5), f"Expected ('huber', 0.5) at step 500, got {sched.resolve(500)}"
    assert sched.resolve(1000) == ("huber", 0.5), f"Expected ('huber', 0.5) at step 1000, got {sched.resolve(1000)}"
    print("PASS: test_resolve_at_boundaries")


def test_empty_schedule_fallback():
    for value in ("", "[]", "  "):
        sched = SteppedLossSchedule(value)
        assert not sched.enabled, f"Expected schedule to be disabled for input {value!r}"
        assert sched.resolve(0) == ("l2", 1.0), (
            f"Expected fallback ('l2', 1.0) for input {value!r}, got {sched.resolve(0)}"
        )
    print("PASS: test_empty_schedule_fallback")


def test_invalid_json_graceful():
    for value in ("not json at all", "{broken:", "null", "42", "true"):
        try:
            sched = SteppedLossSchedule(value)
            result = sched.resolve(0)
        except Exception as exc:
            raise AssertionError(f"Should not raise for input {value!r}, got {exc!r}") from exc
        assert result == ("l2", 1.0), (
            f"Expected fallback ('l2', 1.0) for invalid input {value!r}, got {result}"
        )
    print("PASS: test_invalid_json_graceful")


def test_single_entry():
    schedule_json = '[{"step":0,"loss_type":"l1","weight":2.0}]'
    sched = SteppedLossSchedule(schedule_json)
    assert sched.enabled, "Expected schedule to be enabled"
    for step in (0, 1, 100, 9999):
        result = sched.resolve(step)
        assert result == ("l1", 2.0), f"Expected ('l1', 2.0) at step {step}, got {result}"
    print("PASS: test_single_entry")


def test_multi_step_schedule():
    schedule_json = (
        '[{"step":0,"loss_type":"l2","weight":1.0},'
        '{"step":200,"loss_type":"smooth_l1","weight":0.8},'
        '{"step":800,"loss_type":"huber","weight":0.4}]'
    )
    sched = SteppedLossSchedule(schedule_json)
    assert sched.enabled, "Expected schedule to be enabled"

    # Before first boundary (step 0 is the first, so step 0 and above use it)
    assert sched.resolve(0) == ("l2", 1.0), f"Got {sched.resolve(0)}"
    assert sched.resolve(199) == ("l2", 1.0), f"Got {sched.resolve(199)}"

    # At and after second boundary
    assert sched.resolve(200) == ("smooth_l1", 0.8), f"Got {sched.resolve(200)}"
    assert sched.resolve(799) == ("smooth_l1", 0.8), f"Got {sched.resolve(799)}"

    # At and after third boundary
    assert sched.resolve(800) == ("huber", 0.4), f"Got {sched.resolve(800)}"
    assert sched.resolve(9999) == ("huber", 0.4), f"Got {sched.resolve(9999)}"

    print("PASS: test_multi_step_schedule")


if __name__ == "__main__":
    test_resolve_at_boundaries()
    test_empty_schedule_fallback()
    test_invalid_json_graceful()
    test_single_entry()
    test_multi_step_schedule()
    print("\nAll stepped loss smoke tests passed!")
