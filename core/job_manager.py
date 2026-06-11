import uuid
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger("JobManager")

class SimpleSignal:
    """A simple signal implementation to replace pyqtSignal for headless environments."""
    def __init__(self):
        self._listeners = []

    def connect(self, slot, connection_type=None):
        if slot not in self._listeners:
            self._listeners.append(slot)

    def emit(self, *args):
        for listener in self._listeners:
            try:
                listener(*args)
            except Exception as e:
                logger.error(f"Error in signal listener: {e}")

class JobStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

class JobType(Enum):
    TRAINING = 'training'
    TAGGING = 'tagging'
    SCORING = 'scoring'
    CROPPING = 'cropping'
    TAG_ANALYSIS = 'tag_analysis'
    TAG_BATCH_EDIT = 'tag_batch_edit'
    TAG_RETAG = 'tag_retag'
    TAG_SUGGESTIONS_REFRESH = 'tag_suggestions_refresh'
    GENERIC = 'generic'

@dataclass
class Job:
    type: JobType = JobType.GENERIC
    name: str = ''
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    error: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_progress(self, completed: int, total: Optional[int]=None):
        self.completed_items = completed
        if total is not None:
            self.total_items = total
            if self.total_items > 0:
                self.progress = self.completed_items / self.total_items
            else:
                self.progress = 0.0

class JobWorker(threading.Thread):
    def __init__(self, job: Job, worker_func: Callable, args: tuple=(), kwargs: dict=None):
        super().__init__(daemon=True)
        self.progress_updated = SimpleSignal()
        self.job_completed = SimpleSignal()
        self.job = job
        self.worker_func = worker_func
        self.args = args
        self.kwargs = kwargs or {}
        self._cancelled = False

    def run(self):
        try:
            self.kwargs['progress_callback'] = self._on_progress
            self.kwargs['cancel_check'] = lambda: self._cancelled
            result = self.worker_func(*self.args, **self.kwargs)
            if not self._cancelled:
                self.job_completed.emit(self.job.id, True, '')
        except Exception as e:
            self.job_completed.emit(self.job.id, False, str(e))

    def _on_progress(self, completed: int, total: int):
        self.progress_updated.emit(self.job.id, completed, total)

    def cancel(self):
        self._cancelled = True

class JobManager:
    """Manages background job execution with queue and worker limit.
    
    Standardized to remove PyQt6 dependency for V2 Headless Backend.
    """
    def __init__(self, storage_dir: Optional[Path]=None, max_workers: int = 2):
        self.job_submitted = SimpleSignal()
        self.job_started = SimpleSignal()
        self.job_progress = SimpleSignal()
        self.job_finished = SimpleSignal()
        
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._workers: Dict[str, JobWorker] = {}
        self._max_workers = max_workers
        self._pending_queue: List[tuple] = []
        self._storage_dir = storage_dir or Path(__file__).parent.parent / 'data'
        self._storage_dir.mkdir(exist_ok=True)
        self._history_file = self._storage_dir / 'job_history.json'
        self._load_history()

    def submit(self, job: Job, worker_func: Callable=None, args: tuple=(), kwargs: dict=None, auto_start: bool=True) -> str:
        with self._lock:
            self._jobs[job.id] = job
        self.job_submitted.emit(job.id)
        if worker_func and auto_start:
            self._start_job(job, worker_func, args, kwargs or {})
        return job.id

    def _start_job(self, job: Job, worker_func: Callable, args: tuple, kwargs: dict):
        with self._lock:
            if len(self._workers) >= self._max_workers:
                self._pending_queue.append((job, worker_func, args, kwargs))
                logger.info(f'[JobManager] 任务 {job.id} 已加入队列 (当前运行: {len(self._workers)}/{self._max_workers})')
                return
        
        self._actually_start_job(job, worker_func, args, kwargs)
    
    def _actually_start_job(self, job: Job, worker_func: Callable, args: tuple, kwargs: dict):
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        worker = JobWorker(job, worker_func, args, kwargs)
        worker.progress_updated.connect(self._on_worker_progress)
        worker.job_completed.connect(self._on_worker_completed)
        
        with self._lock:
            self._workers[job.id] = worker
        
        self.job_started.emit(job.id)
        worker.start()
    
    def _start_next_pending(self):
        job_data = None
        with self._lock:
            if self._pending_queue and len(self._workers) < self._max_workers:
                job_data = self._pending_queue.pop(0)
        
        if job_data:
            job, worker_func, args, kwargs = job_data
            logger.info(f'[JobManager] 从队列启动任务 {job.id}')
            self._actually_start_job(job, worker_func, args, kwargs)

    def _on_worker_progress(self, job_id: str, completed: int, total: int):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.update_progress(completed, total)
                self.job_progress.emit(job_id, job.progress)

    def _on_worker_completed(self, job_id: str, success: bool, error: str):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.finished_at = datetime.now()
                if job.status == JobStatus.CANCELLED:
                    pass
                elif success:
                    job.status = JobStatus.COMPLETED
                    job.progress = 1.0
                    self.job_finished.emit(job_id, success, error)
                else:
                    job.status = JobStatus.FAILED
                    job.error = error
                    self.job_finished.emit(job_id, success, error)
                if job_id in self._workers:
                    del self._workers[job_id]
        self._save_history()
        self._start_next_pending()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            job = self._jobs[job_id]
            if job.status != JobStatus.RUNNING:
                return False
            if job_id in self._workers:
                self._workers[job_id].cancel()
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now()
        self.job_finished.emit(job_id, False, '用户取消')
        return True

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_running_jobs(self) -> List[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

    def get_all_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def clear_completed(self):
        with self._lock:
            completed_statuses = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            self._jobs = {k: v for k, v in self._jobs.items() if v.status not in completed_statuses}
        self._save_history()

    def _load_history(self):
        if not self._history_file.exists():
            return
        try:
            with open(self._history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with self._lock:
                for job_dict in data.get('jobs', []):
                    job = self._dict_to_job(job_dict)
                    if job:
                        self._jobs[job.id] = job
            logger.info(f'[JobManager] 已加载 {len(self._jobs)} 条历史任务记录')
        except Exception as e:
            logger.error(f'[JobManager] 加载历史任务失败: {e}')

    MAX_HISTORY_SIZE = 50

    def _save_history(self):
        try:
            with self._lock:
                finished_jobs = [j for j in self._jobs.values() if j.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}]
            finished_jobs.sort(key=lambda j: j.finished_at or j.created_at, reverse=True)
            recent_jobs = finished_jobs[:self.MAX_HISTORY_SIZE]
            data = {'version': '1.0', 'updated_at': datetime.now().isoformat(), 'jobs': [self._job_to_dict(j) for j in recent_jobs]}
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f'[JobManager] 保存历史任务失败: {e}')

    def _job_to_dict(self, job: Job) -> dict:
        return {'id': job.id, 'type': job.type.value if isinstance(job.type, JobType) else job.type, 'name': job.name, 'status': job.status.value if isinstance(job.status, JobStatus) else job.status, 'progress': job.progress, 'total_items': job.total_items, 'completed_items': job.completed_items, 'error': job.error, 'created_at': job.created_at.isoformat() if job.created_at else None, 'started_at': job.started_at.isoformat() if job.started_at else None, 'finished_at': job.finished_at.isoformat() if job.finished_at else None, 'metadata': job.metadata}

    def _dict_to_job(self, d: dict) -> Optional[Job]:
        try:
            job = Job(type=JobType(d['type']) if d.get('type') else JobType.GENERIC, name=d.get('name', ''), status=JobStatus(d['status']) if d.get('status') else JobStatus.COMPLETED, progress=d.get('progress', 0), total_items=d.get('total_items', 0), completed_items=d.get('completed_items', 0), error=d.get('error'), metadata=d.get('metadata', {}))
            job.id = d.get('id', job.id)
            if d.get('created_at'):
                job.created_at = datetime.fromisoformat(d['created_at'])
            if d.get('started_at'):
                job.started_at = datetime.fromisoformat(d['started_at'])
            if d.get('finished_at'):
                job.finished_at = datetime.fromisoformat(d['finished_at'])
            return job
        except Exception as e:
            logger.warning(f'[JobManager] 解析任务记录失败: {e}')
            return None
