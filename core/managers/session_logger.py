import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from collections import deque
import threading

class LogType(str, Enum):
    CONFIG = 'CONFIG'
    FLOW = 'FLOW'
    TASK = 'TASK'
    PLUGIN = 'PLUGIN'
    SYSTEM = 'SYSTEM'

class LogEntry:

    def __init__(self, log_type: LogType, message: str, data: Optional[Dict]=None):
        self.timestamp = datetime.now()
        self.log_type = log_type
        self.message = message
        self.data = data or {}

    def to_dict(self) -> Dict:
        return {'timestamp': self.timestamp.isoformat(), 'type': self.log_type.value, 'message': self.message, 'data': self.data}

    def format(self) -> str:
        icons = {LogType.CONFIG: '⚙️', LogType.FLOW: '🔗', LogType.TASK: '📋', LogType.PLUGIN: '🔌', LogType.SYSTEM: '💻'}
        icon = icons.get(self.log_type, '📝')
        time_str = self.timestamp.strftime('%H:%M:%S')
        return f'[{time_str}] {icon} [{self.log_type.value}] {self.message}'

class SessionLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_entries: int=1000):
        if getattr(self, '_initialized', False):
            return
        self.logger = logging.getLogger("SessionLogger")
        self.entries: deque = deque(maxlen=max_entries)
        self.callbacks: List[callable] = []
        self._lock = threading.Lock()
        self._initialized = True
        self.log(LogType.SYSTEM, '会话开始')

    def log(self, log_type: LogType, message: str, data: Optional[Dict]=None):
        entry = LogEntry(log_type, message, data)
        with self._lock:
            self.entries.append(entry)
        self.logger.info(entry.format())
        # Callbacks might be slow, so maybe keep them out of lock? 
        # But callbacks list needs protection. Copying list is safer.
        with self._lock:
            callbacks_copy = list(self.callbacks)
            
        for callback in callbacks_copy:
            try:
                callback(entry)
            except Exception as e:
                self.logger.error(f'Log callback error: {e}')

    def log_config_change(self, key: str, old_value: Any, new_value: Any):
        self.log(LogType.CONFIG, f'{key}: {old_value} → {new_value}', {'key': key, 'old': old_value, 'new': new_value})

    def log_task_start(self, task_name: str, image_count: int):
        self.log(LogType.TASK, f'开始处理 "{task_name}" ({image_count} 张)', {'task_name': task_name, 'image_count': image_count})

    def log_task_complete(self, success: int, error: int, skipped: int):
        self.log(LogType.TASK, f'完成: 成功 {success}, 失败 {error}, 跳过 {skipped}', {'success': success, 'error': error, 'skipped': skipped})

    def log_flow_change(self, action: str, node_name: str):
        self.log(LogType.FLOW, f'{action}: {node_name}', {'action': action, 'node': node_name})

    def log_plugin_event(self, action: str, plugin_name: str):
        self.log(LogType.PLUGIN, f'{action}: {plugin_name}', {'action': action, 'plugin': plugin_name})

    def add_callback(self, callback: callable):
        with self._lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: callable):
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def get_entries(self, log_type: Optional[LogType]=None, limit: int=100) -> List[LogEntry]:
        entries = list(self.entries)
        if log_type:
            entries = [e for e in entries if e.log_type == log_type]
        return entries[-limit:]

    def get_formatted_logs(self, log_type: Optional[LogType]=None, limit: int=100) -> str:
        entries = self.get_entries(log_type, limit)
        return '\n'.join((e.format() for e in entries))

    def export_to_file(self, filepath: str):
        entries = [e.to_dict() for e in self.entries]
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to export log: {e}")

    def clear(self):
        self.entries.clear()
        self.log(LogType.SYSTEM, '日志已清空')
session_logger = SessionLogger()

def log_config(key: str, old_value: Any, new_value: Any):
    session_logger.log_config_change(key, old_value, new_value)

def log_task_start(name: str, count: int):
    session_logger.log_task_start(name, count)

def log_task_complete(success: int, error: int, skipped: int):
    session_logger.log_task_complete(success, error, skipped)