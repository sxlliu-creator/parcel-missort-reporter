"""
事件缓冲队列：基于 SQLite 的本地持久化缓冲
断网时事件不丢失，恢复后自动补发
"""
import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

from ..detector.engine import DetectionEvent


class EventBuffer:
    """
    基于 SQLite 的事件缓冲队列
    线程安全，支持 FIFO 出队
    """

    def __init__(self, db_path: str, max_size: int = 10000):
        self.db_path = Path(db_path)
        self.max_size = max_size
        self._lock = threading.Lock()

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status
                ON events(status, created_at)
            """)
            conn.commit()

    def push(self, event: DetectionEvent) -> bool:
        """入队一个事件，返回是否成功"""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    # 检查队列大小
                    count = conn.execute(
                        "SELECT COUNT(*) FROM events WHERE status='pending'"
                    ).fetchone()[0]
                    if count >= self.max_size:
                        # 删除最旧的
                        conn.execute("""
                            DELETE FROM events
                            WHERE id = (
                                SELECT id FROM events
                                WHERE status='pending'
                                ORDER BY created_at ASC LIMIT 1
                            )
                        """)

                    event_json = json.dumps(asdict(event), ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO events (event_json) VALUES (?)",
                        (event_json,)
                    )
                    conn.commit()
                return True
            except Exception:
                return False

    def pop(self, limit: int = 100) -> List[Dict]:
        """出队一批待发送的事件，标记为 sending"""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT id, event_json FROM events
                    WHERE status='pending'
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (limit,)).fetchall()

                if rows:
                    ids = [r[0] for r in rows]
                    placeholders = ','.join('?' * len(ids))
                    conn.execute(
                        f"UPDATE events SET status='sending' WHERE id IN ({placeholders})",
                        ids
                    )
                    conn.commit()

                return [{"db_id": r[0], **json.loads(r[1])} for r in rows]

    def mark_sent(self, db_ids: List[int]):
        """标记事件已成功发送"""
        if not db_ids:
            return
        with self._lock:
            with self._get_conn() as conn:
                placeholders = ','.join('?' * len(db_ids))
                conn.execute(
                    f"DELETE FROM events WHERE id IN ({placeholders})",
                    db_ids
                )
                conn.commit()

    def mark_failed(self, db_ids: List[int]):
        """标记发送失败，增加重试计数"""
        if not db_ids:
            return
        with self._lock:
            with self._get_conn() as conn:
                placeholders = ','.join('?' * len(db_ids))
                conn.execute(f"""
                    UPDATE events
                    SET status='pending', retry_count=retry_count+1
                    WHERE id IN ({placeholders})
                """, db_ids)
                conn.commit()

    def pending_count(self) -> int:
        """待发送事件数"""
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM events WHERE status='pending'"
            ).fetchone()[0]

    def total_count(self) -> int:
        """总事件数"""
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]
