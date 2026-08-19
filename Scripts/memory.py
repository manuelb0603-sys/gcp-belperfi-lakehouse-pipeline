"""SQLite-backed memory helpers for short-term thread history and long-term facts.

Defaults:
- DB path: agent/memory/memory.sqlite3
- Short-term trim default: 12 entries per thread
- WAL journaling enabled for concurrency

Usage:
    from Scripts import memory
    memory.init_db()
    thread_id = memory.get_or_create_thread('email', 'alice@example.com')
    memory.save_short_term(thread_id, subject, sender, prompt, html, text)
    memory.upsert_long_fact('balance_trend', 'Balance is rising', source_short_term_id=1)
"""
import os
import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.environ.get("MEMORY_DB_PATH") or os.path.join(
    _PROJECT_ROOT,
    "agent",
    "memory",
    "memory.sqlite3",
)
_DB_PATH: Optional[str] = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def init_db(db_path: Optional[str] = None) -> str:
    """Initialize the SQLite DB and create required tables. Returns resolved db_path."""
    global _DB_PATH
    if db_path:
        _DB_PATH = db_path
    elif _DB_PATH is None:
        _DB_PATH = DEFAULT_DB_PATH

    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                internal_id TEXT UNIQUE NOT NULL,
                channel TEXT NOT NULL,
                external_key_hashed TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS short_term (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_internal_id TEXT NOT NULL,
                subject TEXT,
                sender TEXT,
                prompt TEXT,
                response_html TEXT,
                response_text TEXT,
                tags TEXT,
                meta TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_key TEXT UNIQUE NOT NULL,
                fact_text TEXT NOT NULL,
                source_short_term_id INTEGER,
                weight REAL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.commit()
    finally:
        conn.close()

    return _DB_PATH


def _get_conn():
    if not _DB_PATH:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(_DB_PATH, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def make_internal_id(channel: str, external_key: str) -> str:
    """Return a stable sha256 hex digest for channel+external_key."""
    h = hashlib.sha256(f"{channel}:{external_key}".encode("utf-8"))
    return h.hexdigest()


def get_or_create_thread(channel: str, external_key: str) -> str:
    """Get or create a thread entry and return its internal_id.

    The external_key is hashed before storing to avoid keeping raw PII.
    """
    internal_id = make_internal_id(channel, external_key)
    external_hashed = hashlib.sha256(external_key.encode("utf-8")).hexdigest()
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT internal_id FROM threads WHERE internal_id = ?", (internal_id,))
        row = cur.fetchone()
        if row:
            return internal_id

        conn.execute(
            "INSERT INTO threads (internal_id, channel, external_key_hashed, created_at) VALUES (?,?,?,?)",
            (internal_id, channel, external_hashed, _now_iso()),
        )
        conn.commit()
        return internal_id
    finally:
        conn.close()


def save_short_term(
    thread_internal_id: str,
    subject: Optional[str],
    sender: Optional[str],
    prompt: Optional[str],
    response_html: Optional[str],
    response_text: Optional[str],
    tags: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    max_entries: int = 12,
) -> int:
    """Insert a short-term memory entry and trim to `max_entries` for the thread."""
    conn = _get_conn()
    try:
        tags_j = json.dumps(tags) if tags is not None else None
        meta_j = json.dumps(meta) if meta is not None else None
        cur = conn.execute(
            """
            INSERT INTO short_term
                (thread_internal_id, subject, sender, prompt, response_html, response_text, tags, meta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_internal_id,
                subject,
                sender,
                prompt,
                response_html,
                response_text,
                tags_j,
                meta_j,
                _now_iso(),
            ),
        )
        rowid = cur.lastrowid
        conn.commit()
        trim_short_term(thread_internal_id, max_entries=max_entries)
        return rowid
    finally:
        conn.close()


def get_short_term(thread_internal_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM short_term WHERE thread_internal_id = ? ORDER BY created_at DESC LIMIT ?",
            (thread_internal_id, limit),
        )
        rows = []
        for r in cur.fetchall():
            item = dict(r)
            if item.get("tags"):
                item["tags"] = json.loads(item["tags"])
            if item.get("meta"):
                item["meta"] = json.loads(item["meta"])
            rows.append(item)
        return rows
    finally:
        conn.close()


def trim_short_term(thread_internal_id: str, max_entries: int = 12) -> None:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT id FROM short_term WHERE thread_internal_id = ? ORDER BY created_at DESC",
            (thread_internal_id,),
        )
        ids = [r[0] for r in cur.fetchall()]
        # Keep the newest `max_entries` (ids are ordered newest->oldest)
        to_delete = ids[max_entries:]
        if to_delete:
            q = "DELETE FROM short_term WHERE id IN ({})".format(
                ",".join("?" for _ in to_delete)
            )
            conn.execute(q, to_delete)
            conn.commit()
    finally:
        conn.close()


def upsert_long_fact(fact_key: str, fact_text: str, source_short_term_id: Optional[int] = None, weight: float = 1.0) -> int:
    conn = _get_conn()
    try:
        now = _now_iso()
        conn.execute(
            """
            INSERT INTO long_facts (fact_key, fact_text, source_short_term_id, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_key) DO UPDATE SET
                fact_text = excluded.fact_text,
                source_short_term_id = excluded.source_short_term_id,
                weight = excluded.weight,
                updated_at = excluded.updated_at
            """,
            (fact_key, fact_text, source_short_term_id, weight, now),
        )
        conn.commit()
        cur = conn.execute("SELECT id FROM long_facts WHERE fact_key = ?", (fact_key,))
        row = cur.fetchone()
        return int(row[0])
    finally:
        conn.close()


def get_long_facts(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM long_facts ORDER BY weight DESC, updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
