# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""GPU thermal / power-limit runtime helpers for :class:`LulynxTrainer`.

Extracted verbatim from ``trainer.py`` as a mixin to keep the trainer file
navigable. Behaviour is unchanged: these methods still run as bound methods of
the trainer instance (same ``self`` semantics, same call sites).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GpuDutyCycleThrottler:
    """Step-boundary GPU duty-cycle / temperature governor.

    The mechanism behind "GPU utilization" is simply idle time between kernel
    launches, so the cleanest place to manufacture idle deliberately is the
    step boundary: after each optimizer step the throttler measures how long
    the GPU was busy since the previous boundary and sleeps
    ``busy * (1 - duty) / duty`` so average power tracks ``duty``. Data-loader
    prefetch workers are asynchronous and unaffected; the hot path (forward/
    backward/optimizer) is never fragmented, so torch.compile / CUDA graphs
    stay intact.

    Two stacked modes:
      * manual ``duty_cycle`` < 1.0 — fixed duty.
      * ``target_temp_c`` > 0 — a PI controller polls GPU temperature and
        adjusts an automatic duty within ``[min_duty, 1.0]`` to hold the
        target (deadband keeps it from oscillating). The effective duty is
        ``min(manual, automatic)``.

    All timing/sleep/temperature dependencies are injectable for testing.
    """

    def __init__(
        self,
        *,
        duty_cycle: float = 1.0,
        target_temp_c: int = 0,
        read_temperature_c: Optional[Any] = None,
        log: Optional[Any] = None,
        min_duty: float = 0.2,
        temp_poll_interval_s: float = 5.0,
        temp_deadband_c: float = 2.0,
        kp: float = 0.04,
        ki: float = 0.004,
        max_sleep_s: float = 30.0,
        log_interval_s: float = 30.0,
        sleep_fn: Any = time.sleep,
        clock: Any = time.perf_counter,
    ) -> None:
        self.duty_cycle = min(max(float(duty_cycle), 0.05), 1.0)
        self.target_temp_c = max(int(target_temp_c), 0)
        self.min_duty = min(max(float(min_duty), 0.05), 1.0)
        self.temp_poll_interval_s = max(float(temp_poll_interval_s), 0.5)
        self.temp_deadband_c = max(float(temp_deadband_c), 0.0)
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_sleep_s = max(float(max_sleep_s), 0.0)
        self.log_interval_s = max(float(log_interval_s), 0.0)
        self._read_temperature_c = read_temperature_c
        self._log = log
        self._sleep = sleep_fn
        self._clock = clock

        self._auto_duty = 1.0
        self._integral = 0.0
        self._last_boundary: Optional[float] = None
        self._last_temp_poll = float("-inf")
        self._last_temp: Optional[int] = None
        self._last_log = float("-inf")
        self._last_report: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.duty_cycle < 1.0 or self.target_temp_c > 0

    def on_step_boundary(self) -> Optional[Dict[str, Any]]:
        """Account busy time since the previous boundary and sleep its duty share."""
        now = self._clock()
        if self._last_boundary is None:
            self._last_boundary = now
            return None
        busy_s = max(now - self._last_boundary, 0.0)
        duty = self._effective_duty(now)
        sleep_s = 0.0
        if duty < 1.0 and busy_s > 0.0:
            sleep_s = min(busy_s * (1.0 - duty) / duty, self.max_sleep_s)
            if sleep_s >= 0.001:
                self._sleep(sleep_s)
            else:
                sleep_s = 0.0
        self._last_boundary = self._clock()
        self._last_report = {
            "duty": round(duty, 4),
            "busy_s": round(busy_s, 4),
            "sleep_s": round(sleep_s, 4),
            "temp_c": self._last_temp,
            "mode": "temp_pid" if self.target_temp_c > 0 else "manual",
        }
        self._maybe_log(now)
        return dict(self._last_report)

    @property
    def last_report(self) -> Dict[str, Any]:
        return dict(self._last_report)

    def _effective_duty(self, now: float) -> float:
        if self.target_temp_c <= 0:
            return self.duty_cycle
        if now - self._last_temp_poll >= self.temp_poll_interval_s:
            self._last_temp_poll = now
            temp = self._read_temperature_c() if self._read_temperature_c is not None else None
            if temp is not None:
                self._last_temp = int(temp)
                error = float(self._last_temp - self.target_temp_c)
                if abs(error) > self.temp_deadband_c:
                    # PI update only outside the deadband; clamped integral
                    # bounds the windup to +-(ki * 200) duty contribution.
                    self._integral = min(max(self._integral + error, -200.0), 200.0)
                    self._auto_duty = 1.0 - self.kp * error - self.ki * self._integral
                self._auto_duty = min(max(self._auto_duty, self.min_duty), 1.0)
        return min(self.duty_cycle, self._auto_duty)

    def _maybe_log(self, now: float) -> None:
        if self._log is None:
            return
        rep = self._last_report
        if rep.get("sleep_s", 0.0) <= 0.0:
            return
        if now - self._last_log < self.log_interval_s:
            return
        self._last_log = now
        temp_part = f", GPU {rep['temp_c']}C" if rep.get("temp_c") is not None else ""
        self._log(
            f"[thermal] duty {rep['duty']:.2f}: step busy {rep['busy_s'] * 1000:.0f}ms, "
            f"slept {rep['sleep_s'] * 1000:.0f}ms{temp_part}"
        )


class TrainerThermalMixin:
    """nvidia-smi backed GPU temperature polling, power-limit and epoch cooldown."""

    @staticmethod
    def _hidden_subprocess_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if creationflags:
                kwargs["creationflags"] = creationflags
        return kwargs

    def _read_gpu_temperature_c(self) -> Optional[int]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                **self._hidden_subprocess_kwargs(),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception as e:
            logger.debug("GPU temperature query failed: %s", e)
            return None

        if result.returncode != 0:
            return None

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None

        try:
            return int(float(lines[0].split(",")[0].strip()))
        except (TypeError, ValueError, IndexError):
            return None

    def _sleep_for_cooldown(self, seconds: float) -> None:
        time.sleep(max(float(seconds), 0.0))

    def _apply_gpu_power_limit_if_requested(self) -> None:
        if getattr(self, "_gpu_power_limit_attempted", False):
            return

        self._gpu_power_limit_attempted = True
        watts = max(int(getattr(self.config, "gpu_power_limit_w", 0) or 0), 0)
        if watts <= 0:
            return

        try:
            result = subprocess.run(
                ["nvidia-smi", "-pl", str(watts)],
                capture_output=True,
                text=True,
                timeout=10,
                **self._hidden_subprocess_kwargs(),
            )
        except FileNotFoundError:
            self._log("GPU power limit requested but nvidia-smi is not available; skipping.")
            return
        except subprocess.TimeoutExpired:
            self._log(f"GPU power limit request timed out for {watts}W; skipping.")
            return
        except Exception as e:
            self._log(f"GPU power limit request failed: {e}")
            return

        if result.returncode == 0:
            self._log(f"GPU power limit applied: {watts}W")
        else:
            stderr = (result.stderr or "").strip()
            if stderr:
                self._log(f"GPU power limit request failed: {stderr}")
            else:
                self._log(f"GPU power limit request failed with exit code {result.returncode}")

    def _create_thermal_throttler(self) -> Optional[GpuDutyCycleThrottler]:
        """Build the step-level duty-cycle/temperature governor, or None when off."""
        duty = float(getattr(self.config, "gpu_duty_cycle", 1.0) or 1.0)
        target = max(int(getattr(self.config, "gpu_target_temp_c", 0) or 0), 0)
        if duty >= 1.0 and target <= 0:
            return None
        throttler = GpuDutyCycleThrottler(
            duty_cycle=duty,
            target_temp_c=target,
            read_temperature_c=self._read_gpu_temperature_c,
            log=self._log,
        )
        self._log(
            "GPU throttle enabled: "
            f"duty_cycle={throttler.duty_cycle:.2f}"
            + (f", target_temp={target}C" if target > 0 else "")
        )
        return throttler

    def _apply_gpu_clock_lock_if_requested(self) -> None:
        if getattr(self, "_gpu_clock_lock_attempted", False):
            return

        self._gpu_clock_lock_attempted = True
        mhz = max(int(getattr(self.config, "gpu_lock_clocks_mhz", 0) or 0), 0)
        if mhz <= 0:
            return

        try:
            result = subprocess.run(
                ["nvidia-smi", "-lgc", f"0,{mhz}"],
                capture_output=True,
                text=True,
                timeout=10,
                **self._hidden_subprocess_kwargs(),
            )
        except FileNotFoundError:
            self._log("GPU clock lock requested but nvidia-smi is not available; skipping.")
            return
        except subprocess.TimeoutExpired:
            self._log(f"GPU clock lock request timed out for {mhz}MHz; skipping.")
            return
        except Exception as e:
            self._log(f"GPU clock lock request failed: {e}")
            return

        if result.returncode == 0:
            self._gpu_clock_lock_applied = True
            self._log(
                f"GPU core clock locked to <= {mhz}MHz (lower clock -> lower voltage; "
                "more heat reduction per lost FLOP than duty-cycling)"
            )
        else:
            stderr = (result.stderr or "").strip()
            detail = stderr or f"exit code {result.returncode}"
            self._log(
                f"GPU clock lock failed ({detail}); usually requires admin rights. "
                "Falling back to duty-cycle/temperature throttling if configured."
            )

    def _reset_gpu_clock_lock_if_applied(self) -> None:
        if not getattr(self, "_gpu_clock_lock_applied", False):
            return
        self._gpu_clock_lock_applied = False
        try:
            result = subprocess.run(
                ["nvidia-smi", "-rgc"],
                capture_output=True,
                text=True,
                timeout=10,
                **self._hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                self._log("GPU core clock lock reset.")
            else:
                stderr = (result.stderr or "").strip()
                self._log(f"GPU clock lock reset failed: {stderr or result.returncode}")
        except Exception as e:
            self._log(f"GPU clock lock reset failed: {e}")

    def _maybe_cooldown_after_epoch(self, epoch: int, total_epochs: int) -> None:
        if self._should_stop or (epoch + 1) >= max(int(total_epochs), 1):
            return

        every_n = max(int(getattr(self.config, "cooldown_every_n_epochs", 0) or 0), 0)
        if every_n <= 0 or (epoch + 1) % every_n != 0:
            return

        cooldown_minutes = max(float(getattr(self.config, "cooldown_minutes", 0) or 0), 0.0)
        cooldown_seconds = cooldown_minutes * 60.0
        target_temp = max(int(getattr(self.config, "cooldown_until_temp", 0) or 0), 0)
        poll_seconds = max(int(getattr(self.config, "cooldown_poll_seconds", 30) or 30), 1)

        if target_temp > 0:
            self._log(
                f"Cooldown after epoch {epoch + 1}: waiting for GPU temperature <= {target_temp}C "
                f"(poll every {poll_seconds}s)"
            )
            while not self._should_stop:
                current_temp = self._read_gpu_temperature_c()
                if current_temp is None:
                    if cooldown_seconds > 0:
                        self._log(
                            "GPU temperature unavailable; falling back to time-based cooldown "
                            f"for {int(cooldown_seconds)}s"
                        )
                        self._sleep_for_cooldown(cooldown_seconds)
                    else:
                        self._log("GPU temperature unavailable; skipping temperature-based cooldown.")
                    return

                if current_temp <= target_temp:
                    self._log(f"Cooldown complete: GPU temperature is {current_temp}C")
                    return

                self._log(
                    f"GPU temperature {current_temp}C is above target {target_temp}C; "
                    f"sleeping {poll_seconds}s"
                )
                self._sleep_for_cooldown(float(poll_seconds))
            return

        if cooldown_seconds > 0:
            self._log(f"Cooldown after epoch {epoch + 1}: sleeping for {int(cooldown_seconds)}s")
            self._sleep_for_cooldown(cooldown_seconds)


__all__ = ["GpuDutyCycleThrottler", "TrainerThermalMixin"]
