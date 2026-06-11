from typing import List, Dict, Any, Optional
import threading
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TelemetryStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TelemetryStore, cls).__new__(cls)
                    cls._instance._init_store()
        return cls._instance

    def _init_store(self):
        # Format: { step: { 'drift': float, 'rank': float, 'residual': float, 'layer': str } }
        self.mn_lora_history: Dict[int, List[Dict[str, Any]]] = {}
        # Also keep a simplified global average history for plotting
        # Format: [ { step, drift, rank, residual } ]
        self.global_trajectory: List[Dict[str, Any]] = []
        # Max history size to prevent memory leak
        self.max_history_size = 5000

    def clear(self):
        with self._lock:
            self.mn_lora_history.clear()
            self.global_trajectory.clear()

    def add_mn_lora_data(self, step: int, layer: str, drift: float, rank: float, residual: float = 0.0):
        with self._lock:
            if step not in self.mn_lora_history:
                self.mn_lora_history[step] = []
            
            entry = {
                "layer": layer,
                "drift": drift,
                "rank": rank,
                "residual": residual
            }
            self.mn_lora_history[step].append(entry)

            # Update global trajectory (average of all layers for this step)
            # This is a bit inefficient (recomputing avg each time a layer reports),
            # but usually all layers report in a burst.
            # We can optimize by computing avg only when requested or lazily.
            # For now, let's just append to history if step is new, or update last.
            
            # Simple approach: Recompute average for this step
            items = self.mn_lora_history[step]
            avg_drift = sum(x['drift'] for x in items) / len(items)
            avg_rank = sum(x['rank'] for x in items) / len(items)
            avg_residual = sum(x['residual'] for x in items) / len(items)
            
            # Check if we already have an entry for this step in global_trajectory
            idx = -1
            for i, p in enumerate(self.global_trajectory):
                if p['step'] == step:
                    idx = i
                    break
            
            point = {
                "step": step,
                "drift": avg_drift,
                "rank": avg_rank,
                "residual": avg_residual
            }
            
            if idx >= 0:
                self.global_trajectory[idx] = point
            else:
                self.global_trajectory.append(point)
                # Keep sorted by step
                self.global_trajectory.sort(key=lambda x: x['step'])
                
            # Memory Pruning
            if len(self.global_trajectory) > self.max_history_size:
                # Remove oldest 10%
                prune_count = int(self.max_history_size * 0.1)
                self.global_trajectory = self.global_trajectory[prune_count:]
                
                # Cleanup detailed history map too (optional but recommended)
                # This is harder since it's a dict by step.
                # Let's just keep strict global limit for safety.
                min_step = self.global_trajectory[0]['step']
                keys_to_remove = [k for k in self.mn_lora_history if k < min_step]
                for k in keys_to_remove:
                    del self.mn_lora_history[k]

    def get_trajectory(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.global_trajectory)

# Global instance
telemetry_store = TelemetryStore()


class FileTelemetryReader:
    """Reads training telemetry from .runs/<run_id>/ files.

    Data sources:
    - state.json — progress/status (written by TrainingStateWriter)
    - output.log — console output (written by detached subprocess)
    - launch_request.json — launch context (written at spawn time)
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self._runs_dir = Path(runs_dir)

    def get_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read state.json for a run."""
        state_file = self._runs_dir / run_id / "state.json"
        if not state_file.is_file():
            return None
        try:
            return json.loads(state_file.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            logger.warning("Failed to read state for %s: %s", run_id, exc)
            return None

    def get_log_tail(self, run_id: str, max_lines: int = 200) -> List[str]:
        """Read the last N lines of output.log for a run."""
        log_file = self._runs_dir / run_id / "output.log"
        if not log_file.is_file():
            return []
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-max_lines:]
        except Exception as exc:
            logger.warning("Failed to read log for %s: %s", run_id, exc)
            return []

    def get_log_since(self, run_id: str, offset: int = 0) -> tuple[List[str], int]:
        """Read output.log lines starting from byte offset.

        Returns (lines, new_offset) for incremental reading.
        Offset is a true byte offset (binary seek).
        """
        log_file = self._runs_dir / run_id / "output.log"
        if not log_file.is_file():
            return [], offset
        try:
            with open(log_file, "rb") as f:
                f.seek(offset)
                data = f.read()
                new_offset = f.tell()
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            return lines, new_offset
        except Exception as exc:
            logger.warning("Failed to read log for %s: %s", run_id, exc)
            return [], offset

    def get_structured_events(self, run_id: str, max_events: int = 200) -> List[Dict[str, Any]]:
        """Read structured events from ``events.jsonl`` for a run."""
        events_file = self._runs_dir / run_id / "events.jsonl"
        if not events_file.is_file():
            return []
        try:
            lines = events_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            logger.warning("Failed to read events for %s: %s", run_id, exc)
            return []
        events: List[Dict[str, Any]] = []
        for line in lines[-max_events:]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _read_run_json(self, run_id: str, filename: str) -> Optional[Dict[str, Any]]:
        json_file = self._runs_dir / run_id / filename
        if not json_file.is_file():
            return None
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def get_launch_request(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read launch_request.json for a run."""
        return self._read_run_json(run_id, "launch_request.json")

    def get_config_snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read config_snapshot.json for a run."""
        return self._read_run_json(run_id, "config_snapshot.json")

    def get_config_lock(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read config_lock.json for a run."""
        return self._read_run_json(run_id, "config_lock.json")

    def list_runs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all runs, optionally filtered by status."""
        runs = []
        if not self._runs_dir.exists():
            return runs
        for run_dir in self._runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            state = self.get_state(run_dir.name)
            if state is None:
                continue
            if status_filter and state.get("status") != status_filter:
                continue
            runs.append({
                "run_id": run_dir.name,
                **state,
            })
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    def get_active_runs(self) -> List[Dict[str, Any]]:
        """List runs with status='running'."""
        return self.list_runs(status_filter="running")


# Global file telemetry reader (reads from .runs/ relative to backend root)
_file_reader: Optional[FileTelemetryReader] = None


def get_file_telemetry_reader(runs_dir: str | Path | None = None) -> FileTelemetryReader:
    """Get or create the global FileTelemetryReader."""
    global _file_reader
    if _file_reader is None or runs_dir is not None:
        if runs_dir is None:
            runs_dir = Path(__file__).parent.parent / ".runs"
        _file_reader = FileTelemetryReader(runs_dir)
    return _file_reader
