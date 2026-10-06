"""Report memory after an email is sent: the short-term report row and extracted long-term facts.

Used by both report modes. The memory is a finance.memory.MemoryStore (duck-typed; None skips
it). Facts are extracted with the configured fact model through Ollama (failures leave no facts)
and embedded with the embedding model; an embedding failure stores the fact without a vector so
retrieval can backfill it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .email_delivery import report_subject
from .legacy_report import query_ollama
from .ollama import embed_text
from .settings import ReportSettings
from .status_log import log_status


def open_report_memory(memory_schema: str = "memory") -> Any:
    """Open the SQLite MemoryStore for the selected schema (MEMORY_DB_PATH overrides it); None on failure."""
    try:
        from .memory import open_memory

        store = open_memory(memory_schema)
        log_status(f"Using memory schema '{memory_schema}' at {store.db_path}")
        return store
    except Exception as exc:
        print(f"Memory initialization failed: {exc}")
        return None


def extract_json_payload(payload: str) -> list[dict[str, Any]]:
    """Pull a JSON list out of a noisy model response so memory facts can be extracted reliably."""
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*?\}\s*\]", payload, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def remember_report(memory_module: Any, sender: str, receivers: list[str], receiver_header: str, prompt: str, report_text: str, model_name: str, ollama_url: str, config: dict | ReportSettings, report_plain_text: str | None = None) -> None:
    """Persist the report in the memory store and extract reusable long facts for future runs.

    memory_module is a MemoryStore (or None to skip); config is a config dict or ReportSettings.

    report_text is the report HTML. When report_plain_text is supplied (agent mode), it is
    stored as the readable text and used for fact extraction instead of raw HTML.
    """
    if memory_module is None:
        return

    try:
        log_status("Starting memory persistence for this report")
        thread_id = memory_module.get_or_create_thread("email", receiver_header or sender or "email:unknown")
        # Keep the complete report available to the next run for UI and goal continuity.
        has_plain_text = bool(report_plain_text and report_plain_text.strip())
        plain_text = report_plain_text if has_plain_text else (report_text or "")

        st_id = memory_module.save_short_term(
            thread_internal_id=thread_id,
            subject=report_subject(),
            sender=sender,
            prompt=prompt,
            response_html=report_text,
            response_text=plain_text,
            tags=["weekly_report"],
            meta={"recipients": receivers},
        )
        log_status(f"Saved short-term memory row with thread id={thread_id}, short_term id={st_id}")

        settings = ReportSettings.coerce(config)
        fact_model = settings.fact_model_name or model_name
        embedding_model = settings.embedding_model_name
        embedding_timeout = settings.ollama_timeout
        fact_prompt = (
            f"Extract a JSON array of concise facts from the following {'plain-text' if has_plain_text else 'HTML'} report. "
            "Each item must be an object with keys 'fact_key', 'fact_text', and optional 'weight'.\n\n"
            f"Report:\n{plain_text}\n\nReturn ONLY valid JSON."
        )
        log_status(f"Extracting facts from completed report using model '{fact_model}'")
        fact_res = query_ollama(fact_prompt, ollama_url, fact_model, stream=False)
        facts = extract_json_payload(fact_res)
        log_status(f"Fact extraction returned {len(facts)} candidate fact(s)")
        # Each extracted fact is stored as a reusable long-term memory item so future runs
        # can inherit patterns and goals without relying on the previous raw numbers.
        for fact in facts:
            try:
                fk = str(fact.get("fact_key") or fact.get("key") or "").strip()
                ft = str(fact.get("fact_text") or fact.get("text") or "").strip()
                if not fk or not ft:
                    continue
                w = float(fact.get("weight", 1.0)) if fact.get("weight") is not None else 1.0
                try:
                    embedding = embed_text(ft, ollama_url, embedding_model, embedding_timeout)
                except Exception as embedding_exc:
                    log_status(f"Could not embed long fact '{fk}'; it will be backfilled on retrieval: {embedding_exc}")
                    embedding = None
                memory_module.upsert_long_fact(
                    fk,
                    ft,
                    source_short_term_id=st_id,
                    weight=w,
                    embedding=embedding,
                    embedding_model=embedding_model if embedding is not None else None,
                )
            except Exception as fact_exc:
                log_status(f"Could not persist long fact: {fact_exc}")
                continue
        log_status("Completed memory persistence and fact extraction")
    except Exception as exc:
        print(f"Memory logging failed: {exc}")
