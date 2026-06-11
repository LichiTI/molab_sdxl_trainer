import json
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SimpleSignal:
    def __init__(self):
        self._listeners = []
    def connect(self, slot):
        if slot not in self._listeners: self._listeners.append(slot)
    def emit(self, *args):
        for listener in self._listeners: listener(*args)

class TrainingDataReader:
    """Headless-compatible log reader replacing the legacy QObject version."""
    def __init__(self):
        self.new_data = SimpleSignal()
        self.training_started = SimpleSignal()
        self.training_finished = SimpleSignal()
        self._file_path: Optional[Path] = None
        self._last_position = 0
        self._history: List[Dict] = []
        self._running = False
        self._thread = None

    def start(self, file_path: str, poll_interval_ms: int=100):
        self._file_path = Path(file_path)
        self._last_position = 0
        self._history.clear()
        if self._file_path.exists():
            self._last_position = self._file_path.stat().st_size
        
        self.training_started.emit()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(poll_interval_ms / 1000.0,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.training_finished.emit()

    def _run_loop(self, interval_sec):
        while self._running:
            self._read_new_data()
            time.sleep(interval_sec)

    def _read_new_data(self):
        if not self._file_path or not self._file_path.exists():
            return
        try:
            current_size = self._file_path.stat().st_size
            if current_size > self._last_position:
                with open(self._file_path, 'r', encoding='utf-8') as f:
                    f.seek(self._last_position)
                    new_content = f.read()
                    self._last_position = f.tell()
                for line in new_content.strip().split('\n'):
                    if line:
                        try:
                            record = json.loads(line)
                            self._history.append(record)
                            self.new_data.emit(record)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f'[TrainingDataReader] Error reading file: {e}')

    def get_history(self) -> List[Dict]:
        return self._history.copy()

    def get_latest(self) -> Optional[Dict]:
        return self._history[-1] if self._history else None

class TrainingStatsCalculator:
    @staticmethod
    def calculate_throughput(history: List[Dict], window: int=10) -> float:
        if len(history) < 2: return 0.0
        recent = history[-window:]
        if len(recent) < 2: return 0.0
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(recent[0]['timestamp'])
            t2 = datetime.fromisoformat(recent[-1]['timestamp'])
            dt = (t2 - t1).total_seconds()
            if dt > 0:
                steps = recent[-1]['step'] - recent[0]['step']
                return steps / dt
        except Exception: pass
        return 0.0

    @staticmethod
    def calculate_efficiency(history: List[Dict], window: int=10) -> float:
        if len(history) < 2: return 0.0
        recent = history[-window:]
        throughput = TrainingStatsCalculator.calculate_throughput(recent)
        avg_power = sum((r.get('power_w', 200) for r in recent)) / len(recent)
        if avg_power > 0: return throughput / avg_power
        return 0.0

    @staticmethod
    def calculate_extraction_rate(history: List[Dict], window: int=10) -> float:
        if len(history) < 2: return 0.0
        recent = history[-window:]
        throughput = TrainingStatsCalculator.calculate_throughput(recent)
        avg_vram = sum((r.get('vram_gb', 8) for r in recent)) / len(recent)
        if avg_vram > 0: return throughput / avg_vram
        return 0.0

    @staticmethod
    def get_average_health(history: List[Dict], window: int=10) -> float:
        if not history: return 0.0
        recent = history[-window:]
        scores = []
        for record in recent:
            svd_data = record.get('svd', [])
            for layer in svd_data:
                if 'health_score' in layer:
                    scores.append(layer['health_score'])
        return sum(scores) / len(scores) if scores else 0.0
