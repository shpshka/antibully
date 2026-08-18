from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


class EventLog:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.db_path = directory / "events.sqlite3"
        self.jsonl_path = directory / "events.jsonl"
        self.snapshots = directory / "snapshots"
        self.snapshots.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL, kind TEXT NOT NULL,
                severity TEXT NOT NULL, confidence REAL NOT NULL,
                message TEXT NOT NULL, snapshot TEXT, metadata TEXT NOT NULL
                )"""
            )

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def add(
        self,
        kind: str,
        severity: str,
        confidence: float,
        message: str,
        snapshot: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        item = {
            "created_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            "severity": severity,
            "confidence": round(float(confidence), 3),
            "message": message,
            "snapshot": snapshot,
            "metadata": metadata or {},
        }
        with self._lock:
            with self._connect() as db:
                cur = db.execute(
                    "INSERT INTO events(created_at,kind,severity,confidence,message,snapshot,metadata) VALUES(?,?,?,?,?,?,?)",
                    (
                        item["created_at"],
                        kind,
                        severity,
                        item["confidence"],
                        message,
                        snapshot,
                        json.dumps(item["metadata"], ensure_ascii=False),
                    ),
                )
                item["id"] = cur.lastrowid
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def recent(self, limit: int = 100, since: str | None = None) -> list[dict]:
        with self._connect() as db:
            if since:
                rows = db.execute(
                    "SELECT * FROM events WHERE created_at >= ? ORDER BY id DESC LIMIT ?",
                    (since, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            result.append(item)
        return result

    def clear(self) -> int:
        """Remove journal records while leaving evidence files recoverable."""
        with self._lock:
            with self._connect() as db:
                count = int(db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
                db.execute("DELETE FROM events")
            self.jsonl_path.write_text("", encoding="utf-8")
        return count

    def csv_text(self, limit: int = 5000) -> str:
        rows = self.recent(limit)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            (
                "id",
                "created_at",
                "kind",
                "severity",
                "confidence",
                "message",
                "snapshot",
                "metadata",
            )
        )
        for item in rows:
            writer.writerow(
                (
                    item["id"],
                    item["created_at"],
                    item["kind"],
                    item["severity"],
                    item["confidence"],
                    item["message"],
                    item["snapshot"],
                    json.dumps(item["metadata"], ensure_ascii=False),
                )
            )
        return output.getvalue()
