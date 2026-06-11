import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
from contextlib import contextmanager

class TaskHistoryDB:

    def __init__(self, db_path: str='task_history.db'):
        self.db_path = db_path
        self.logger = logging.getLogger("TaskHistoryDB")
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30.0)

    @contextmanager
    def _connection(self):
        """AUDIT FIX: Provide a context manager for database connections to ensure they are always closed."""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("\n        CREATE TABLE IF NOT EXISTS task_history (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            task_name TEXT,\n            memo TEXT DEFAULT '',\n            start_time TIMESTAMP,\n            end_time TIMESTAMP,\n            duration_seconds REAL DEFAULT 0,\n            total_images INTEGER DEFAULT 0,\n            success_count INTEGER DEFAULT 0,\n            error_count INTEGER DEFAULT 0,\n            skipped_count INTEGER DEFAULT 0,\n            config_snapshot TEXT DEFAULT '{}',\n            flow_definition TEXT DEFAULT '{}',\n            error_log TEXT DEFAULT '[]',\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n        ")
            cursor.execute('\n        CREATE TABLE IF NOT EXISTS task_metrics (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            task_id INTEGER REFERENCES task_history(id) ON DELETE CASCADE,\n            timestamp TIMESTAMP,\n            images_processed INTEGER,\n            images_per_minute REAL,\n            memory_usage_mb REAL,\n            vram_usage_mb REAL\n        )\n        ')
            cursor.execute('\n        CREATE TABLE IF NOT EXISTS task_tag_stats (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            task_id INTEGER REFERENCES task_history(id) ON DELETE CASCADE,\n            tag TEXT,\n            count INTEGER\n        )\n        ')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_created ON task_history(created_at)')
            conn.commit()

    def create_task(self, task_name: str, config_snapshot: Dict, flow_definition: Optional[Dict]=None, memo: str='') -> int:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('\n        INSERT INTO task_history (task_name, memo, start_time, config_snapshot, flow_definition)\n        VALUES (?, ?, ?, ?, ?)\n        ', (task_name, memo, datetime.now().isoformat(), json.dumps(config_snapshot, ensure_ascii=False), json.dumps(flow_definition or {}, ensure_ascii=False)))
            task_id = cursor.lastrowid
            conn.commit()
            self.logger.info(f'Created task history record: {task_id}')
            return task_id

    def update_task_progress(self, task_id: int, total_images: int, success_count: int, error_count: int, skipped_count: int):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('\n        UPDATE task_history \n        SET total_images = ?, success_count = ?, error_count = ?, skipped_count = ?\n        WHERE id = ?\n        ', (total_images, success_count, error_count, skipped_count, task_id))
            conn.commit()

    def complete_task(self, task_id: int, success_count: int, error_count: int, skipped_count: int, error_log: Optional[List[Dict]]=None):
        with self._connection() as conn:
            cursor = conn.cursor()
            end_time = datetime.now()
            cursor.execute('SELECT start_time FROM task_history WHERE id = ?', (task_id,))
            row = cursor.fetchone()
            duration = 0
            if row:
                start_time = datetime.fromisoformat(row[0])
                duration = (end_time - start_time).total_seconds()
            cursor.execute('\n        UPDATE task_history \n        SET end_time = ?, duration_seconds = ?, \n            success_count = ?, error_count = ?, skipped_count = ?,\n            error_log = ?\n        WHERE id = ?\n        ', (end_time.isoformat(), duration, success_count, error_count, skipped_count, json.dumps(error_log or [], ensure_ascii=False), task_id))
            conn.commit()
            self.logger.info(f'Task {task_id} completed in {duration:.1f}s')

    def add_metric(self, task_id: int, images_processed: int, images_per_minute: float, memory_usage_mb: float=0, vram_usage_mb: float=0):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('\n        INSERT INTO task_metrics (task_id, timestamp, images_processed, images_per_minute, memory_usage_mb, vram_usage_mb)\n        VALUES (?, ?, ?, ?, ?, ?)\n        ', (task_id, datetime.now().isoformat(), images_processed, images_per_minute, memory_usage_mb, vram_usage_mb))
            conn.commit()

    def add_tag_stats(self, task_id: int, tag_counts: Dict[str, int]):
        with self._connection() as conn:
            cursor = conn.cursor()
            for tag, count in tag_counts.items():
                cursor.execute('\n            INSERT INTO task_tag_stats (task_id, tag, count) VALUES (?, ?, ?)\n            ', (task_id, tag, count))
            conn.commit()

    def get_task(self, task_id: int) -> Optional[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM task_history WHERE id = ?', (task_id,))
            row = cursor.fetchone()
            if row:
                task = dict(row)
                try:
                    task['config_snapshot'] = json.loads(task['config_snapshot'])
                except (json.JSONDecodeError, TypeError):
                    task['config_snapshot'] = {}
                
                try:
                    task['flow_definition'] = json.loads(task['flow_definition'])
                except (json.JSONDecodeError, TypeError):
                    task['flow_definition'] = {}
                
                try:
                    task['error_log'] = json.loads(task['error_log'])
                except (json.JSONDecodeError, TypeError):
                    task['error_log'] = []

                cursor.execute('SELECT * FROM task_metrics WHERE task_id = ? ORDER BY timestamp', (task_id,))
                task['metrics'] = [dict(r) for r in cursor.fetchall()]
                cursor.execute('SELECT tag, count FROM task_tag_stats WHERE task_id = ? ORDER BY count DESC LIMIT 50', (task_id,))
                task['top_tags'] = [(r['tag'], r['count']) for r in cursor.fetchall()]
                return task
            return None

    def get_recent_tasks(self, limit: int=20) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('\n        SELECT id, task_name, memo, start_time, end_time, duration_seconds,\n               total_images, success_count, error_count, skipped_count, flow_definition\n        FROM task_history\n        ORDER BY created_at DESC\n        LIMIT ?\n        ', (limit,))
            tasks = []
            for row in cursor.fetchall():
                task = dict(row)
                task['flow_definition'] = json.loads(task['flow_definition'])
                tasks.append(task)
            return tasks

    def export_config(self, task_id: int, filepath: str):
        task = self.get_task(task_id)
        if task:
            export_data = {'task_name': task['task_name'], 'config': task['config_snapshot'], 'flow': task['flow_definition'], 'exported_at': datetime.now().isoformat()}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f'Exported config to {filepath}')

    def get_statistics(self) -> Dict:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('\n        SELECT \n            COUNT(*) as total_tasks,\n            COALESCE(SUM(total_images), 0) as total_images,\n            COALESCE(SUM(success_count), 0) as total_success,\n            COALESCE(SUM(duration_seconds), 0) as total_duration\n        FROM task_history\n        ')
            row = cursor.fetchone()
            stats = {'total_tasks': row[0], 'total_images_processed': row[1], 'total_success': row[2], 'total_duration_hours': row[3] / 3600 if row[3] else 0}
            return stats

    def delete_task(self, task_id: int):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM task_metrics WHERE task_id = ?', (task_id,))
            cursor.execute('DELETE FROM task_tag_stats WHERE task_id = ?', (task_id,))
            cursor.execute('DELETE FROM task_history WHERE id = ?', (task_id,))
            conn.commit()
            self.logger.info(f'Deleted task {task_id}')

    def _batch_delete_tasks(self, cursor: sqlite3.Cursor, task_ids: List[int]) -> int:
        if not task_ids:
            return 0
        
        # Split into chunks of 999 to respect SQLite variable limit
        chunk_size = 999
        deleted_count = 0
        for i in range(0, len(task_ids), chunk_size):
            chunk = task_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            
            cursor.execute(f'DELETE FROM task_metrics WHERE task_id IN ({placeholders})', chunk)
            cursor.execute(f'DELETE FROM task_tag_stats WHERE task_id IN ({placeholders})', chunk)
            cursor.execute(f'DELETE FROM task_history WHERE id IN ({placeholders})', chunk)
            deleted_count += len(chunk)
            
        return deleted_count

    def cleanup(self, max_records: int=None, max_days: int=None) -> int:
        with self._connection() as conn:
            cursor = conn.cursor()
            deleted_count = 0
            
            task_ids_to_delete = set()
            
            if max_records and max_records > 0:
                cursor.execute('''
            SELECT id FROM task_history 
            ORDER BY created_at DESC 
            LIMIT -1 OFFSET ?
            ''', (max_records,))
                task_ids_to_delete.update(row[0] for row in cursor.fetchall())

            if max_days and max_days > 0:
                cutoff_date = (datetime.now() - timedelta(days=max_days)).isoformat()
                cursor.execute('''
            SELECT id FROM task_history 
            WHERE created_at < ?
            ''', (cutoff_date,))
                task_ids_to_delete.update(row[0] for row in cursor.fetchall())

            if task_ids_to_delete:
                deleted_count = self._batch_delete_tasks(cursor, list(task_ids_to_delete))
            
            conn.commit()
            if deleted_count > 0:
                self.logger.info(f'Cleaned up {deleted_count} old task records')
            return deleted_count
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    db = TaskHistoryDB('test_history.db')
    task_id = db.create_task('Furry 角色训练集', config_snapshot={'wd14_threshold': 0.35, 'cropper_mode': 'bucket'}, memo='第三次尝试')
    db.add_metric(task_id, 100, 12.5, 1024, 4096)
    db.add_metric(task_id, 200, 14.2, 1100, 4200)
    db.complete_task(task_id, success_count=180, error_count=5, skipped_count=15)
    db.add_tag_stats(task_id, {'1girl': 150, 'smile': 120, 'looking_at_viewer': 100})
    task = db.get_task(task_id)
    logging.info(json.dumps(task, indent=2, default=str))
    logging.info('\n=== 全局统计 ===')
    logging.info(db.get_statistics())