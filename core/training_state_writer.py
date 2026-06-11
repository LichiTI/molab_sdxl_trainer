# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Training state writer — writes .runs/<run_id>/state.json.

Injected into entry_train.py to report training progress to disk.
The file is the authoritative source of truth for training status,
read by routers/training.py and the frontend.

state.json schema:
{
    "run_id": str,
    "pid": int,
    "status": "running" | "completed" | "failed" | "orphaned",
    "current_step": int,
    "current_epoch": int,
    "total_epochs": int,
    "total_steps": int,
    "last_loss": float,
    "last_lr": float,
    "started_at": str (ISO 8601),
    "updated_at": str (ISO 8601),
    "error": str | null,
    "execution_profile_id": str | null,
    "requested_execution_core": str | null,
    "effective_execution_core": str | null,
    "requested_attention_backend": str | null,
    "resolved_attention_backend": str | null,
    "applied_attention_backend": str | null,
    "fallback_reason": str | null,
    "turbocore_fallback_reason": str | null
    "runtime_features": dict | null,
}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TrainingStateWriter:
    """Writes training state to .runs/<run_id>/state.json.

    Usage in entry_train.py:
        writer = TrainingStateWriter(run_dir)
        writer.start(pid=1234, total_epochs=10)
        # On each progress event:
        writer.update(step=100, epoch=1, loss=0.5, lr=1e-4)
        # On completion:
        writer.complete()
        # On failure:
        writer.fail("error message")
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._state_file = self._run_dir / "state.json"
        self._state: dict = {}
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        pid: int,
        total_epochs: int = 0,
        total_steps: int = 0,
        run_id: str = "",
        execution_profile_id: str = "",
        requested_execution_core: str = "",
        effective_execution_core: str = "",
        requested_attention: str = "",
        resolved_attention: str = "",
        memory_optimization: Optional[dict[str, Any]] = None,
        runtime_features: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write initial state on training start."""
        now = datetime.now(timezone.utc).isoformat()
        self._state = {
            "run_id": run_id,
            "pid": pid,
            "status": "running",
            "current_step": 0,
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "total_steps": total_steps,
            "last_loss": 0.0,
            "last_lr": 0.0,
            "started_at": now,
            "updated_at": now,
            "error": None,
            "execution_profile_id": execution_profile_id or None,
            "requested_execution_core": requested_execution_core or None,
            "effective_execution_core": effective_execution_core or None,
            "requested_attention_backend": requested_attention or None,
            "resolved_attention_backend": resolved_attention or None,
            "applied_attention_backend": None,
            "fallback_reason": None,
            "turbocore_fallback_reason": None,
            "memory_optimization": memory_optimization or {"enabled": False},
            "runtime_features": runtime_features or {},
        }
        self._flush()

    def update(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
        requested_execution_core: Optional[str] = None,
        effective_execution_core: Optional[str] = None,
        requested_attention: Optional[str] = None,
        resolved_attention: Optional[str] = None,
        applied_attention: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        turbocore_fallback_reason: Optional[str] = None,
        execution_profile_id: Optional[str] = None,
        memory_optimization: Optional[dict[str, Any]] = None,
        runtime_features: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update state on each progress event."""
        if step is not None:
            self._state["current_step"] = step
        if epoch is not None:
            self._state["current_epoch"] = epoch
        if loss is not None:
            self._state["last_loss"] = round(loss, 6)
        if lr is not None:
            self._state["last_lr"] = lr
        if total_epochs is not None:
            self._state["total_epochs"] = total_epochs
        if total_steps is not None:
            self._state["total_steps"] = total_steps
        if requested_execution_core is not None:
            self._state["requested_execution_core"] = requested_execution_core
        if effective_execution_core is not None:
            self._state["effective_execution_core"] = effective_execution_core
        if requested_attention is not None:
            self._state["requested_attention_backend"] = requested_attention
        if resolved_attention is not None:
            self._state["resolved_attention_backend"] = resolved_attention
        if applied_attention is not None:
            self._state["applied_attention_backend"] = applied_attention
        if fallback_reason is not None:
            self._state["fallback_reason"] = fallback_reason
        if turbocore_fallback_reason is not None:
            self._state["turbocore_fallback_reason"] = turbocore_fallback_reason
        if execution_profile_id is not None:
            self._state["execution_profile_id"] = execution_profile_id
        if memory_optimization is not None:
            self._state["memory_optimization"] = memory_optimization
        if runtime_features is not None:
            self._state["runtime_features"] = runtime_features
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def complete(self) -> None:
        """Mark training as completed."""
        self._state["status"] = "completed"
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def fail(self, error: str) -> None:
        """Mark training as failed."""
        self._state["status"] = "failed"
        self._state["error"] = error
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def _flush(self) -> None:
        """Atomic write to state.json."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._run_dir),
                prefix=".state_",
                suffix=".json.tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, str(self._state_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Failed to write state.json: %s", exc)
