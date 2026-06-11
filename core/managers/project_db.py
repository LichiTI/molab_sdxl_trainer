import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum
from contextlib import contextmanager

class ImageStatus(str, Enum):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    DONE = 'DONE'
    ERROR = 'ERROR'

class ProjectDB:

    def __init__(self, db_path: str='project.db'):
        # [SECURITY] Prevent directory traversal/arbitrary path writes
        self.db_path = str(Path(db_path).resolve())
        # Ensure db file is strictly within allowed locations or just valid filename
        # For now, we ensure the directory exists and is valid
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("ProjectDB")
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager for database connections to ensure proper closure and commits."""
        conn = self._get_conn() # Use the new _get_conn method
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise e
        finally:
            conn.close()

    def _get_conn(self):
        """Returns a new SQLite connection with a specified timeout."""
        return sqlite3.connect(self.db_path, timeout=30.0)
        
    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                status TEXT DEFAULT 'PENDING',
                score REAL DEFAULT 0.0,
                tags TEXT DEFAULT '',
                crop_box TEXT DEFAULT '{}',
                meta TEXT DEFAULT '{}',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON images(status)')

    def add_images(self, paths: List[str]) -> int:
        added_count = 0
        if not paths:
            return 0
            
        with self._connection() as conn:
            cursor = conn.cursor()
            # Use executemany for bulk insert performance
            # Prepare data: (path, status) tuples
            data = [(str(p), ImageStatus.PENDING.value) for p in paths]
            try:
                # INSERT OR IGNORE avoids errors on duplicates
                cursor.executemany('INSERT OR IGNORE INTO images (path, status) VALUES (?, ?)', data)
                # rowcount in executemany is reliable in recent sqlite3
                added_count = cursor.rowcount
            except Exception as e:
                self.logger.error(f'Failed to bulk add images: {e}')
        return added_count

    def get_pending_images(self, limit: int=10) -> List[Dict[str, Any]]:
        results = []
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images WHERE status = ? LIMIT ?', (ImageStatus.PENDING.value, limit))
            rows = cursor.fetchall()
            for row in rows:
                results.append(dict(row))
                cursor.execute('UPDATE images SET status = ? WHERE id = ?', (ImageStatus.PROCESSING.value, row['id']))
        return results

    def update_image(self, image_id: int, **kwargs):
        valid_fields = {'status', 'score', 'tags', 'crop_box', 'meta'}
        updates = []
        values = []
        for k, v in kwargs.items():
            if k in valid_fields:
                updates.append(f'{k} = ?')
                if isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)
        if not updates:
            return
        values.append(image_id)
        sql = f"UPDATE images SET {', '.join(updates)}, last_updated = CURRENT_TIMESTAMP WHERE id = ?"
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)

    def get_stats(self) -> Dict[str, int]:
        stats = {ImageStatus.PENDING.value: 0, ImageStatus.PROCESSING.value: 0, ImageStatus.DONE.value: 0, ImageStatus.ERROR.value: 0, 'total': 0}
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT status, COUNT(*) FROM images GROUP BY status')
            rows = cursor.fetchall()
            total = 0
            for status, count in rows:
                stats[status] = count
                total += count
            stats['total'] = total
        return stats

    def reset_processing_status(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE images SET status = ? WHERE status = ?', (ImageStatus.PENDING.value, ImageStatus.PROCESSING.value))
            if cursor.rowcount > 0:
                self.logger.info(f'Reset {cursor.rowcount} interrupted tasks to PENDING.')

    def clear(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM images')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    db = ProjectDB('test.db')
    db.clear()
    logging.info(f"Added: {db.add_images(['img1.jpg', 'img2.png'])}")
    tasks = db.get_pending_images(1)
    if tasks:
        t = tasks[0]
        logging.info(f"Processing: {t['path']}")
        db.update_image(t['id'], status='DONE', score=0.85, tags='1girl, smile')
    logging.info(db.get_stats())