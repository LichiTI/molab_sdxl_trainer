"""Structured training event writer for ``.runs/<run_id>/events.jsonl``.

This is a lightweight, append-only event stream owned by the trainer. It does
not change training behavior; it only records structured runtime events so
launchers and diagnostic tools can consume them without regex-parsing stdout.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingEventWriter:
    """Append-only JSONL writer for structured training events."""

    def __init__(self, run_dir: str | Path, filename: str = "events.jsonl") -> None:
        self._run_dir = Path(run_dir)
        self._path = self._run_dir / filename
        self._lock = threading.Lock()
        self._default_context: Dict[str, Any] = {}
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def set_default_context(self, **context: Any) -> None:
        """Set top-level fields automatically merged into future events."""
        merged = dict(self._default_context)
        for key, value in context.items():
            if value is None:
                continue
            merged[key] = value
        self._default_context = merged

    def emit(self, payload: Dict[str, Any]) -> None:
        """Append *payload* as one JSON line.

        Failures are logged at debug/warning level and never propagated into the
        training path.
        """
        record = dict(self._default_context)
        record.update(payload)
        record.setdefault("schema_version", 1)
        record.setdefault("timestamp", _utc_now())
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to append structured training event: %s", exc)

    def start(
        self,
        *,
        run_id: str,
        pid: int,
        training_type: str,
        model_type: str,
        total_steps: int,
        total_epochs: int,
        execution_profile_id: str = "",
        schema_id: str = "",
        requested_execution_core: str = "",
        effective_execution_core: str = "",
        requested_attention_backend: str = "",
        resolved_attention_backend: str = "",
    ) -> None:
        self.set_default_context(
            run_id=str(run_id or ""),
            runtime_id=str(execution_profile_id or ""),
            task_type=str(training_type or ""),
            model_type=str(model_type or ""),
            execution_profile_id=str(execution_profile_id or ""),
            schema_id=str(schema_id or ""),
        )
        self.emit(
            {
                "event_type": "run_start",
                "summary": "training run started",
                "severity": "info",
                "data": {
                    "pid": int(pid),
                    "training_type": str(training_type or ""),
                    "model_type": str(model_type or ""),
                    "total_steps": int(total_steps or 0),
                    "total_epochs": int(total_epochs or 0),
                    "execution_profile_id": str(execution_profile_id or ""),
                    "schema_id": str(schema_id or ""),
                    "requested_execution_core": str(requested_execution_core or ""),
                    "effective_execution_core": str(effective_execution_core or ""),
                    "requested_attention_backend": str(requested_attention_backend or ""),
                    "resolved_attention_backend": str(resolved_attention_backend or ""),
                },
            }
        )

    def complete(self, *, run_id: str, final_step: int = 0, final_epoch: int = 0) -> None:
        self.emit(
            {
                "event_type": "run_end",
                "summary": "training run completed",
                "severity": "info",
                "step": int(final_step or 0),
                "epoch": int(final_epoch or 0),
                "data": {"status": "completed"},
            }
        )

    def fail(
        self,
        *,
        run_id: str,
        error: str,
        final_step: int = 0,
        final_epoch: int = 0,
    ) -> None:
        self.emit(
            {
                "event_type": "run_end",
                "summary": str(error or "training run failed"),
                "severity": "error",
                "step": int(final_step or 0),
                "epoch": int(final_epoch or 0),
                "data": {"status": "failed", "error": str(error or "")},
            }
        )

    def step(
        self,
        *,
        run_id: str,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        total_steps: int = 0,
        total_epochs: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {
            "loss": float(loss),
            "lr": float(lr),
            "total_steps": int(total_steps or 0),
            "total_epochs": int(total_epochs or 0),
        }
        if extra:
            data.update(extra)
        self.emit(
            {
                "event_type": "step",
                "summary": f"loss={float(loss):.6f}",
                "severity": "info",
                "step": int(step),
                "epoch": int(epoch),
                "data": data,
            }
        )

    def epoch(
        self,
        *,
        run_id: str,
        epoch: int,
        avg_loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {"avg_loss": float(avg_loss)}
        if extra:
            data.update(extra)
        self.emit(
            {
                "event_type": "epoch",
                "summary": f"epoch {int(epoch) + 1} avg_loss={float(avg_loss):.6f}",
                "severity": "info",
                "epoch": int(epoch),
                "data": data,
            }
        )

    def checkpoint(
        self,
        *,
        run_id: str,
        path: str,
        epoch: int,
        step: int = 0,
        final: bool = False,
    ) -> None:
        self.emit(
            {
                "event_type": "checkpoint",
                "summary": f"saved: {path}",
                "severity": "info",
                "step": int(step or 0),
                "epoch": int(epoch or 0),
                "data": {"path": str(path), "final": bool(final)},
            }
        )
