"""Finance Q&A with selected-report retrieval and historical long-term facts."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Any

from .memory_search import search_long_facts
from .ollama import embed_texts, generate_text
from .settings import PROJECT_ROOT


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup or "")
    return "\n".join(parser.parts)


def _chunks(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class ReportContextIndex:
    """Small bounded cache of ephemeral vector indexes for selected report content."""

    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[list[str], list[list[float]]]] = OrderedDict()

    def search(
        self,
        question: str,
        report_html: str,
        report_data: dict[str, Any],
        ollama_url: str,
        embedding_model: str,
        timeout: int,
        limit: int = 5,
    ) -> list[str]:
        source_text = (
            "SELECTED-PERIOD FINANCIAL DATA (authoritative):\n"
            + json.dumps(report_data, ensure_ascii=False, sort_keys=True, default=str)
            + "\n\nGENERATED FINANCE SUMMARY:\n"
            + _html_to_text(report_html)
        )
        key = hashlib.sha256(f"{embedding_model}\0{source_text}".encode("utf-8")).hexdigest()
        entry = self._entries.get(key)
        if entry is None:
            pieces = _chunks(source_text)
            if not pieces:
                return []
            vectors = embed_texts(pieces, ollama_url, embedding_model, timeout)
            entry = (pieces, vectors)
            self._entries[key] = entry
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)

        query_vector = embed_text(question, ollama_url, embedding_model, timeout)
        pieces, vectors = entry
        ranked = sorted(
            zip(pieces, vectors),
            key=lambda item: _cosine_similarity(query_vector, item[1]),
            reverse=True,
        )
        return [piece for piece, _ in ranked[:max(1, limit)]]


_REPORT_INDEX = ReportContextIndex()


def _memory_module(config: dict[str, Any]):
    """Initialize the existing SQLite memory module for the configured schema."""
    import memory

    schema = str(config.get("memory_schema", "memory"))
    if not re.fullmatch(r"[A-Za-z0-9_]+", schema):
        raise ValueError("memory_schema may contain only letters, numbers, and underscores")
    database_path = PROJECT_ROOT / "agent" / "memory" / f"{schema}.sqlite3"
    memory.init_db(str(database_path))
    return memory


def answer_finance_question(
    question: str,
    report_context: dict[str, Any] | None,
    report_html: str,
    config: dict[str, Any],
    ollama_url: str,
    model_name: str,
    memory_module: Any = None,
) -> str:
    """Retrieve selected-report evidence and historical long facts, answer, then remember QA."""
    question = (question or "").strip()
    if not question:
        return "Enter a question about the selected financial summary."
    if not report_context:
        return "Load a year and month summary before asking a question."

    timeout = int(config.get("ollama_timeout", 600))
    embedding_model = config.get("embedding_model_name", "nomic-embed-text")
    report_limit = min(8, max(1, int(config.get("qa_report_context_chunks", 5))))
    memory_limit = min(10, max(1, int(config.get("qa_long_fact_limit", 5))))
    try:
        report_passages = _REPORT_INDEX.search(
            question,
            report_html,
            report_context,
            ollama_url,
            embedding_model,
            timeout,
            limit=report_limit,
        )
    except Exception as exc:
        # If report-vector search is unavailable, retain the selected-period data directly
        # rather than losing the authoritative source from the answer context.
        print(f"Selected-report vector search unavailable; including report data directly: {exc}")
        report_passages = [json.dumps(report_context, ensure_ascii=False, default=str)[:6000]]

    memory_module = memory_module or _memory_module(config)
    try:
        long_facts, retrieval_method = search_long_facts(
            question,
            memory_module,
            ollama_url,
            embedding_model,
            timeout,
            limit=memory_limit,
        )
    except Exception as exc:
        print(f"Long-term finance memory unavailable: {exc}")
        long_facts, retrieval_method = [], "unavailable"

    prompt = f"""You are a careful personal-finance assistant. Answer the user's question concisely using the selected-period report passages and finance data below. Selected-period BigQuery figures are the sole authority for current amounts, trends, and months. Do not invent transactions or explanations. Clearly distinguish historical preferences/goals from current financial facts. Historical memory can add context but cannot change the selected-period numbers. If the available evidence does not answer the question, say so.

SELECTED PERIOD REPORT EVIDENCE (authoritative current context):
{json.dumps(report_passages, ensure_ascii=False, indent=2)}

SELECTED PERIOD FINANCE SNAPSHOT (authoritative):
{json.dumps({"selected_month": report_context.get("selected_month"), "snapshot": report_context.get("snapshot")}, ensure_ascii=False, indent=2, default=str)}

HISTORICAL LONG-TERM FACTS (supplemental only; retrieval={retrieval_method}):
{json.dumps(long_facts, ensure_ascii=False, indent=2, default=str)}

USER QUESTION:
{question}
"""
    try:
        answer = generate_text(prompt, ollama_url, model_name, timeout=timeout)
    except Exception as exc:
        return f"Could not get an answer from Ollama: {exc}"

    try:
        selected_period = str(report_context.get("selected_month", "unknown"))
        thread_key = f"finance-qa:{config.get('memory_schema', 'memory')}"
        thread_id = memory_module.get_or_create_thread("finance_qa", thread_key)
        memory_module.save_short_term(
            thread_internal_id=thread_id,
            subject=f"Finance Q&A — {selected_period}",
            sender="finance-summary-ui",
            prompt=question,
            response_html=None,
            response_text=answer,
            tags=["finance_qa", selected_period],
            meta={"selected_month": selected_period},
        )
    except Exception as exc:
        # Persistence is helpful for future context, but must never discard a valid answer.
        print(f"Could not save finance Q&A exchange to memory: {exc}")
    return answer
