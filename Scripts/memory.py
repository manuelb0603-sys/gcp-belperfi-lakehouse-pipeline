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
import math
import re
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

        # Existing memory databases predate vector retrieval. Add the nullable JSON
        # embedding column in place so they remain usable without a destructive reset.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(long_facts)")}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE long_facts ADD COLUMN embedding TEXT")
        if "embedding_model" not in columns:
            conn.execute("ALTER TABLE long_facts ADD COLUMN embedding_model TEXT")

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


def _serialize_embedding(embedding: Optional[List[float]]) -> Optional[str]:
    if embedding is None:
        return None
    values = [float(value) for value in embedding]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding must contain finite numeric values")
    return json.dumps(values, separators=(",", ":"))


def upsert_long_fact(
    fact_key: str,
    fact_text: str,
    source_short_term_id: Optional[int] = None,
    weight: float = 1.0,
    embedding: Optional[List[float]] = None,
    embedding_model: Optional[str] = None,
) -> int:
    """Insert or update a durable fact and its optional vector embedding."""
    conn = _get_conn()
    try:
        now = _now_iso()
        embedding_json = _serialize_embedding(embedding)
        if embedding_json is not None and not (embedding_model and embedding_model.strip()):
            raise ValueError("embedding_model is required when storing an embedding")
        conn.execute(
            """
            INSERT INTO long_facts (fact_key, fact_text, source_short_term_id, weight, updated_at, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_key) DO UPDATE SET
                fact_text = excluded.fact_text,
                source_short_term_id = excluded.source_short_term_id,
                weight = excluded.weight,
                updated_at = excluded.updated_at,
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model
            """,
            (fact_key, fact_text, source_short_term_id, weight, now, embedding_json, embedding_model),
        )
        conn.commit()
        cur = conn.execute("SELECT id FROM long_facts WHERE fact_key = ?", (fact_key,))
        row = cur.fetchone()
        return int(row[0])
    finally:
        conn.close()


def get_long_facts(limit: int = 20) -> List[Dict[str, Any]]:
    """Return facts without their potentially large embedding vectors."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT id, fact_key, fact_text, source_short_term_id, weight, updated_at "
            "FROM long_facts ORDER BY weight DESC, updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_long_facts_needing_embedding(embedding_model: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """List facts without vectors from the requested model for lazy backfill/re-embedding."""
    if not embedding_model or not embedding_model.strip():
        raise ValueError("embedding_model must be provided")
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT fact_key, fact_text FROM long_facts "
            "WHERE embedding IS NULL OR embedding_model IS NOT ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (embedding_model, max(0, int(limit))),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def set_long_fact_embedding(fact_key: str, embedding: List[float], embedding_model: str) -> None:
    """Store or refresh one fact's embedding and its producing model name."""
    if not embedding_model or not embedding_model.strip():
        raise ValueError("embedding_model must be provided")
    embedding_json = _serialize_embedding(embedding)
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE long_facts SET embedding = ?, embedding_model = ? WHERE fact_key = ?",
            (embedding_json, embedding_model, fact_key),
        )
        conn.commit()
    finally:
        conn.close()


def search_long_facts(
    query_embedding: List[float], embedding_model: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Rank same-model vectors by cosine similarity; vector values stay internal."""
    if not embedding_model or not embedding_model.strip():
        raise ValueError("embedding_model must be provided")
    query = [float(value) for value in query_embedding]
    if not query or not all(math.isfinite(value) for value in query):
        raise ValueError("Query embedding must contain finite numeric values")
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        raise ValueError("Query embedding must not be a zero vector")

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact_key, fact_text, source_short_term_id, weight, updated_at, embedding, embedding_model "
            "FROM long_facts WHERE embedding IS NOT NULL AND embedding_model = ?",
            (embedding_model,),
        ).fetchall()
    finally:
        conn.close()

    ranked = []
    for row in rows:
        try:
            vector = json.loads(row["embedding"])
            if not isinstance(vector, list) or len(vector) != len(query):
                continue
            vector = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in vector):
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if vector_norm == 0:
                continue
            score = sum(a * b for a, b in zip(query, vector)) / (query_norm * vector_norm)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        item = {key: row[key] for key in (
            "id", "fact_key", "fact_text", "source_short_term_id", "weight", "updated_at", "embedding_model"
        )}
        item["similarity"] = score
        ranked.append(item)

    ranked.sort(key=lambda item: (item["similarity"], item["weight"] or 0), reverse=True)
    return ranked[:max(0, int(limit))]


_FACT_QUERY_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "with", "would", "user",
}


def _fact_search_tokens(text: str) -> List[str]:
    return [
        token for token in re.findall(r"[^\W_]+", text.casefold())
        if len(token) > 1 and token not in _FACT_QUERY_STOPWORDS
    ]


def search_long_facts_bm25(
    query_text: str,
    limit: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    """Rank fact keys and text with BM25 using an exact scan of the small fact set."""
    query_tokens = set(_fact_search_tokens(query_text))
    if not query_tokens or limit <= 0:
        return []
    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError("BM25 parameters require k1 >= 0 and 0 <= b <= 1")

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact_key, fact_text, source_short_term_id, weight, updated_at "
            "FROM long_facts"
        ).fetchall()
    finally:
        conn.close()

    documents = []
    document_frequency = {token: 0 for token in query_tokens}
    total_length = 0
    for row in rows:
        # Include the key as searchable text, so concise labels such as
        # "travel_goal" work just like their spaced natural-language form.
        tokens = _fact_search_tokens(f'{row["fact_key"] or ""} {row["fact_text"] or ""}')
        frequencies: Dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        documents.append((row, tokens, frequencies))
        total_length += len(tokens)
        for token in query_tokens & frequencies.keys():
            document_frequency[token] += 1

    document_count = len(documents)
    if not document_count:
        return []
    average_length = total_length / document_count
    ranked = []
    for row, tokens, frequencies in documents:
        matched_tokens = query_tokens & frequencies.keys()
        if not matched_tokens:
            continue
        score = 0.0
        for token in matched_tokens:
            term_frequency = frequencies[token]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            length_normalizer = k1 * (
                1 - b + b * len(tokens) / average_length
            ) if average_length else 0.0
            score += inverse_document_frequency * (
                term_frequency * (k1 + 1) / (term_frequency + length_normalizer)
            )
        item = {key: row[key] for key in (
            "id", "fact_key", "fact_text", "source_short_term_id", "weight", "updated_at"
        )}
        item["bm25_score"] = score
        item["matched_terms"] = sorted(matched_tokens)
        ranked.append(item)

    ranked.sort(
        key=lambda item: (item["bm25_score"], item["weight"] or 0, item["updated_at"]),
        reverse=True,
    )
    return ranked[:limit]


def combine_fact_search_results(
    vector_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    limit: int = 5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge dense and lexical rankings with reciprocal rank fusion (RRF)."""
    combined: Dict[Any, Dict[str, Any]] = {}
    for source, results in (("vector", vector_results), ("bm25", lexical_results)):
        for rank, result in enumerate(results, start=1):
            identity = result.get("id", result.get("fact_key"))
            if identity is None:
                continue
            item = combined.setdefault(identity, dict(result))
            item["retrieval_score"] = item.get("retrieval_score", 0.0) + 1.0 / (rrf_k + rank)
            sources = item.setdefault("retrieval_sources", [])
            if source not in sources:
                sources.append(source)
            for key, value in result.items():
                if key not in item or key in {"similarity", "lexical_score", "matched_terms"}:
                    item[key] = value

    ranked = sorted(
        combined.values(),
        key=lambda item: (
            item["retrieval_score"], item.get("weight") or 0, item.get("updated_at") or ""
        ),
        reverse=True,
    )
    return ranked[:max(0, int(limit))]
