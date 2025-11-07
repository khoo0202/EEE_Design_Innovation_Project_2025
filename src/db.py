import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Any

class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,           -- display: DD-MM-YYYY HH:MM:SS
                ts_iso TEXT NOT NULL,       -- filter:  YYYY-MM-DD HH:MM:SS
                session_id TEXT NOT NULL,
                session_label TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                predicted TEXT NOT NULL,    -- smoking/vaping/none (model)
                confirmed TEXT NULL,        -- operator: smoking/vaping/NULL
                description TEXT NULL,      -- VQA description, optional
                source_type TEXT NOT NULL,  -- "RTSP" or "FILE"
                image_path TEXT NULL,       -- main snapshot
                evidence_dir TEXT NULL      -- folder holding review_*.jpg etc.
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_ts ON violations(ts_iso);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_sess ON violations(session_id);")
        self._conn.commit()

    def insert_violation(
        self,
        ts_dt: datetime,
        session_id: str,
        session_label: str,
        track_id: int,
        predicted: str,
        description: Optional[str],
        source_type: str,
        image_path: Optional[str],
        evidence_dir: Optional[str],
    ) -> int:
        ts = ts_dt.strftime("%d-%m-%Y %H:%M:%S")
        ts_iso = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        cur = self._conn.execute(
            """
            INSERT INTO violations
            (ts, ts_iso, session_id, session_label, track_id, predicted, confirmed,
             description, source_type, image_path, evidence_dir)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (ts, ts_iso, session_id, session_label, track_id, predicted,
             description or "", source_type, image_path or "", evidence_dir or "")
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_confirmed(self, violation_id: int, label: Optional[str]):
        if label is None:
            self._conn.execute("UPDATE violations SET confirmed=NULL WHERE id=?", (violation_id,))
        else:
            self._conn.execute("UPDATE violations SET confirmed=? WHERE id=?", (label, violation_id))
        self._conn.commit()

    def delete_violation(self, violation_id: int):
        self._conn.execute("DELETE FROM violations WHERE id=?", (violation_id,))
        self._conn.commit()

    def list_violations(
        self,
        date_from_ddmmyyyy: Optional[str],
        date_to_ddmmyyyy: Optional[str]
    ) -> List[Tuple]:
        where = []
        args: list[str] = []

        if date_from_ddmmyyyy:
            # start-of-day
            dt_iso = datetime.strptime(date_from_ddmmyyyy, "%d-%m-%Y").strftime("%Y-%m-%d 00:00:00")
            where.append("ts_iso >= ?")
            args.append(dt_iso)

        if date_to_ddmmyyyy:
            # end-of-day
            dt_iso = datetime.strptime(date_to_ddmmyyyy, "%d-%m-%Y").strftime("%Y-%m-%d 23:59:59")
            where.append("ts_iso <= ?")
            args.append(dt_iso)

        cond = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self._conn.execute(
            f"""
            SELECT id, ts, session_label, track_id, predicted, confirmed,
                   description, source_type, image_path, evidence_dir
            FROM violations
            {cond}
            ORDER BY ts_iso DESC
            """,
            tuple(args)
        ).fetchall()
        return rows

    def get_violation(self, violation_id: int) -> Optional[Tuple[Any, ...]]:
        row = self._conn.execute(
            """
            SELECT id, ts, session_label, track_id, predicted, confirmed,
                   description, source_type, image_path, evidence_dir,
                   session_id
            FROM violations
            WHERE id=?
            """,
            (violation_id,)
        ).fetchone()
        return row

    # which track_ids have already been logged/confirmed etc.
    def all_for_session(self, session_id: str) -> List[Tuple]:
        rows = self._conn.execute(
            """
            SELECT track_id, predicted, confirmed
            FROM violations
            WHERE session_id=?
            """,
            (session_id,)
        ).fetchall()
        return rows

    def latest_session(self) -> Optional[Tuple[str, str]]:
        row = self._conn.execute(
            """
            SELECT session_id, session_label, MAX(ts_iso)
            FROM violations
            GROUP BY session_id
            ORDER BY MAX(ts_iso) DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        return (row[0], row[1])

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
