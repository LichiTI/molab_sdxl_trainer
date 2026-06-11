import os
import time
import tempfile
import shutil
import logging
from enum import Enum
from typing import List, Any, Optional, Callable, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Event
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class ExecutionMode(Enum):
    TEMP_FILE = 'temp_file'
    MEMORY_SINGLE = 'memory_single'
    MEMORY_LIMITED = 'memory_limited'
    MEMORY_DYNAMIC = 'memory_dynamic'

@dataclass
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.TEMP_FILE
    memory_limit_gb: float = 4.0
    preload_threads: int = 2
    memory_reserve_percent: float = 0.2
    temp_dir: Optional[str] = None
    cleanup_temp: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> 'ExecutionConfig':
        mode_str = data.get('mode', 'temp_file')
        mode = ExecutionMode(mode_str) if mode_str in [e.value for e in ExecutionMode] else ExecutionMode.TEMP_FILE
        return cls(mode=mode, memory_limit_gb=data.get('memory_limit_gb', 4.0), preload_threads=data.get('preload_threads', 2), memory_reserve_percent=data.get('memory_reserve_percent', 0.2), temp_dir=data.get('temp_dir'), cleanup_temp=data.get('cleanup_temp', True))

    def to_dict(self) -> dict:
        return {'mode': self.mode.value, 'memory_limit_gb': self.memory_limit_gb, 'preload_threads': self.preload_threads, 'memory_reserve_percent': self.memory_reserve_percent, 'temp_dir': self.temp_dir, 'cleanup_temp': self.cleanup_temp}

class MemoryManager:
    DEFAULT_IMAGE_SIZE_MB = 10

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = logging.getLogger('MemoryManager')

    def get_available_memory_gb(self) -> float:
        if not PSUTIL_AVAILABLE:
            return 4.0
        try:
            mem = psutil.virtual_memory()
            return mem.available / 1024 ** 3
        except Exception:
            return 4.0

    def get_total_memory_gb(self) -> float:
        if not PSUTIL_AVAILABLE:
            return 16.0
        try:
            mem = psutil.virtual_memory()
            return mem.total / 1024 ** 3
        except Exception:
            return 16.0

    def calculate_batch_size(self, image_avg_size_mb: float=None) -> int:
        if image_avg_size_mb is None:
            image_avg_size_mb = self.DEFAULT_IMAGE_SIZE_MB
        mode = self.config.mode
        if mode == ExecutionMode.MEMORY_SINGLE:
            return 1
        elif mode == ExecutionMode.MEMORY_LIMITED:
            limit_mb = self.config.memory_limit_gb * 1024
            batch_size = int(limit_mb / image_avg_size_mb)
            return max(1, batch_size)
        elif mode == ExecutionMode.MEMORY_DYNAMIC:
            available_gb = self.get_available_memory_gb()
            usable_gb = available_gb * (1 - self.config.memory_reserve_percent)
            usable_mb = usable_gb * 1024
            batch_size = int(usable_mb / image_avg_size_mb)
            return max(1, batch_size)
        else:
            return 1

    def estimate_image_size(self, image: Image.Image) -> float:
        width, height = image.size
        channels = len(image.getbands())
        bytes_per_pixel = 1
        size_bytes = width * height * channels * bytes_per_pixel
        return size_bytes / (1024 * 1024)

class PreloadPool:

    def __init__(self, num_threads: int=2, max_queue_size: int=10):
        self.num_threads = num_threads
        self.max_queue_size = max_queue_size
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.stop_event = Event()
        self.threads: List[Thread] = []
        self.logger = logging.getLogger('PreloadPool')

    def start(self, file_paths: List[Path], loader_func: Callable[[Path], Any]):
        self.stop_event.clear()

        def worker():
            for path in file_paths:
                if self.stop_event.is_set():
                    break
                try:
                    data = loader_func(path)
                    self.queue.put((path, data))
                except Exception as e:
                    self.logger.error(f'Failed to preload {path}: {e}')
                    self.queue.put((path, None))
        chunk_size = (len(file_paths) + self.num_threads - 1) // self.num_threads
        for i in range(self.num_threads):
            start = i * chunk_size
            end = min(start + chunk_size, len(file_paths))
            chunk = file_paths[start:end]
            if chunk:
                t = Thread(target=lambda c=chunk: worker(), daemon=True)
                t.start()
                self.threads.append(t)

    def get_next(self, timeout: float=5.0) -> Optional[Tuple[Path, Any]]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

class TempFileManager:

    def __init__(self, base_dir: Optional[str]=None):
        self.logger = logging.getLogger('TempFileManager')
        if base_dir:
            self.temp_dir = Path(base_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self._created_by_us = False
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='lulynx_flow_'))
            self._created_by_us = True
        self.step_dirs: List[Path] = []

    def create_step_dir(self, step_index: int) -> Path:
        step_dir = self.temp_dir / f'step_{step_index:03d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        self.step_dirs.append(step_dir)
        return step_dir

    def cleanup(self, keep_last: bool=False):
        try:
            if keep_last and self.step_dirs:
                dirs_to_remove = self.step_dirs[:-1]
            else:
                dirs_to_remove = self.step_dirs
            for d in dirs_to_remove:
                if d.exists():
                    shutil.rmtree(d)
            if self._created_by_us and (not keep_last):
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f'Cleanup failed: {e}')

    def save_image(self, image: Image.Image, step_dir: Path, filename: str) -> Path:
        # [SECURITY] Sanitize filename
        safe_name = Path(filename).stem
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in (' ', '-', '_')).strip() or "image"
        
        save_path = step_dir / f'{safe_name}.png'
        image.save(save_path)
        return save_path

    def save_metadata(self, metadata: dict, step_dir: Path, filename: str) -> Path:
        import json
        save_path = step_dir / f'{Path(filename).stem}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return save_path

def stream_images(file_paths: List[Path], batch_size: int=1, preload_pool: Optional[PreloadPool]=None) -> Generator[List[Tuple[Path, Image.Image]], None, None]:
    batch = []
    for path in file_paths:
        try:
            if preload_pool:
                result = preload_pool.get_next()
                if result and result[1] is not None:
                    batch.append(result)
                else:
                    img = Image.open(path).convert('RGB')
                    batch.append((path, img))
            else:
                img = Image.open(path).convert('RGB')
                batch.append((path, img))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        except Exception as e:
            logging.error(f'Failed to load image {path}: {e}')
    if batch:
        yield batch