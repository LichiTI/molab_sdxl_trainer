"""Small OS sleep guard used by launcher-started training workers."""

from __future__ import annotations

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


class SleepGuard:
    """Prevents system sleep while a worker process is actively training.

    The guard is intentionally opt-in through an environment variable set by
    the launcher. Non-Windows platforms are a no-op for now.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = str(os.environ.get("LULYNX_PREVENT_SLEEP_DURING_TRAINING") or "").strip() == "1"
        self.enabled = bool(enabled)
        self.active = False

    def __enter__(self) -> "SleepGuard":
        if not self.enabled:
            return self
        if sys.platform != "win32":
            logger.info("Sleep prevention requested, but no platform guard is available for %s", sys.platform)
            return self
        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
        if result == 0:
            logger.warning("Failed to enable Windows sleep prevention")
        else:
            self.active = True
            logger.info("Windows sleep prevention enabled for training worker")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if not self.active or sys.platform != "win32":
            return
        result = ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)  # type: ignore[attr-defined]
        if result == 0:
            logger.warning("Failed to restore Windows sleep policy")
        else:
            logger.info("Windows sleep prevention released")
        self.active = False
