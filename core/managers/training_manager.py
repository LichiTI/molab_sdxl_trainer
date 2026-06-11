
import subprocess
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging
logger = logging.getLogger("TrainingManager")

# Windows detached process flags
_DETACHED_FLAGS = (
    getattr(subprocess, "CREATE_DETACHED_PROCESS", getattr(subprocess, "DETACHED_PROCESS", 0))
    | subprocess.CREATE_NEW_PROCESS_GROUP
    | subprocess.CREATE_NO_WINDOW
)


class TrainingProcessManager:
    """
    Manages the lifecycle of training processes.
    Uses detached spawn so training survives launcher/WebUI restarts.

    On Windows: CREATE_DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    stdout/stderr are redirected to .runs/<run_id>/output.log (file, not PIPE).
    """
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        # Unified PID path: data/training.pid (Persistent over logs/)
        self._pid_file = Path(__file__).parent.parent / "data" / "training.pid"
        if not self._pid_file.parent.exists():
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    def get_process(self) -> Optional[subprocess.Popen]:
        """Backward-compatible accessor for current process."""
        return self._process

    def clear_pid(self):
        """Public helper for clearing persisted PID state."""
        self._clear_pid()

    def start_worker(
        self,
        cmd: list,
        env: dict,
        cwd: str,
        config_name: str,
        run_dir: Optional[str] = None,
        bufsize: int = 1,
    ) -> subprocess.Popen:
        """
        Start a training process as a detached (independent) process.

        On Windows, uses CREATE_DETACHED_PROCESS so the child has no
        parent-child relationship with the launcher — training survives
        launcher restarts.

        stdout/stderr are redirected to <run_dir>/output.log when
        *run_dir* is provided; otherwise they go to DEVNULL.

        Returns the Popen object (caller can read .pid immediately).
        """
        if self.is_running:
            raise RuntimeError("Training is already running")

        # Ensure data dir exists
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine stdout/stderr destination
        if run_dir:
            run_path = Path(run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            log_path = run_path / "output.log"
            log_file = open(log_path, "a", encoding="utf-8", errors="replace")
        else:
            log_file = subprocess.DEVNULL

        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = _DETACHED_FLAGS

            self._process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=bufsize,
                creationflags=creationflags,
            )

            self._save_pid(self._process.pid, config_name)
            return self._process

        except Exception as e:
            # Cleanup if startup fails
            self._process = None
            if run_dir and log_file is not subprocess.DEVNULL:
                try:
                    log_file.close()
                except Exception:
                    pass
            raise e

    def stop_worker(self) -> Dict[str, str]:
        """
        Safe termination of the process tree.
        Supports Windows (taskkill), Linux/Unix (SIGTERM/SIGKILL).
        """
        if not self._process:
            last_pid = self.load_last_pid()
            if last_pid is None or not self.attach_orphaned_process(int(last_pid)):
                self._clear_pid()
                return {"status": "no_process"}

        pid = self._process.pid
        status_msg = "stopped"

        try:
            import psutil
            parent = psutil.Process(pid)

            # Verify identity to avoid killing innocent processes if PID recycled
            cmdline = ' '.join(parent.cmdline())
            is_training_process = (
                'python' in cmdline.lower() and
                ('entry_train.py' in cmdline or 'entry_lr_find.py' in cmdline)
            )

            if not is_training_process:
                self._process = None
                return {"status": "error", "message": "Process validation failed: Not a training process"}

            children = parent.children(recursive=True)

            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()

            gone, alive = psutil.wait_procs([parent] + children, timeout=5)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

        except ImportError:
            # Platform-native fallback without psutil
            if sys.platform == 'win32':
                # Windows: Use taskkill /T to kill entire process tree
                try:
                    subprocess.call(
                        ['taskkill', '/F', '/T', '/PID', str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception:
                    if self._process: self._process.kill()
            else:
                # Unix/Linux/Mac: Kill entire process group
                import signal
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        if self._process: self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    if self._process: self._process.kill()
        except Exception:
             if self._process: self._process.kill()

        self._process = None
        self._clear_pid()
        return {"status": "stopped", "pid": str(pid)}

    def attach_orphaned_process(self, pid: int) -> bool:
        """
        Attempt to re-attach to a running process from a PID file.
        (Advanced feature for future recovery)
        """
        try:
            import psutil

            if not psutil.pid_exists(pid):
                return False

            proc = psutil.Process(pid)
            if not self.is_training_pid(pid):
                return False

            class OrphanProcess:
                def __init__(self, process: "psutil.Process"):
                    self.pid = process.pid
                    self._proc = process

                def poll(self):
                    return None if self._proc.is_running() else 0

                def wait(self, timeout=None):
                    return self._proc.wait(timeout)

            self._process = OrphanProcess(proc)
            return True
        except Exception:
            return False

    def discover_running_sessions(self, runs_dir: Path) -> List[Dict]:
        """Scan .runs/ for running sessions on startup.

        For each run with state.json showing status="running", check
        if the PID is still alive. Returns list of discovered sessions.
        """
        sessions = []
        if not runs_dir.exists():
            return sessions

        for run_dir in runs_dir.iterdir():
            state_file = run_dir / "state.json"
            if not state_file.is_file():
                continue
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                if state.get("status") != "running":
                    continue
                pid = state.get("pid", 0)
                if pid and self._is_pid_alive(pid):
                    sessions.append({
                        "run_id": run_dir.name,
                        "pid": pid,
                        "state": state,
                        "run_dir": str(run_dir),
                    })
                else:
                    # Mark as orphaned
                    state["status"] = "orphaned"
                    state_file.write_text(
                        json.dumps(state, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.warning("Failed to read state from %s: %s", run_dir, exc)

        return sessions

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a PID is alive (cross-platform)."""
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
            except Exception:
                pass
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def _save_pid(self, pid: int, config_name: str):
        """Save PID to disk"""
        try:
            import datetime
            self._pid_file.write_text(json.dumps({
                "pid": pid,
                "config_name": config_name,
                "started_at": datetime.datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"[TrainingManager] Failed to save PID: {e}")

    def _clear_pid(self):
        """Remove PID file"""
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
        except Exception:
            pass

    def load_last_pid(self) -> Optional[int]:
        """Read last known PID"""
        try:
            if self._pid_file.exists():
                data = json.loads(self._pid_file.read_text())
                return data.get("pid")
        except Exception:
            pass
        return None

    @staticmethod
    def is_training_pid(pid: int) -> bool:
        try:
            import psutil
            if not psutil.pid_exists(pid):
                return False
            cmdline = ' '.join(psutil.Process(pid).cmdline())
            return 'python' in cmdline.lower() and ('entry_train.py' in cmdline or 'entry_lr_find.py' in cmdline)
        except Exception:
            return False

    def cleanup_after_exit(self):
        """Cleanup PID file if process has exited naturally"""
        if self._process and self._process.poll() is not None:
            self._clear_pid()
            self._process = None


# Global Singleton
training_process_manager = TrainingProcessManager()
