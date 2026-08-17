"""Simple test harness for Scripts/memory.py

Run as: python Scripts/memory_test.py
"""
import os
import importlib.util
from pathlib import Path

# Import Scripts/memory.py directly by path to avoid import/package path issues
module_path = Path(__file__).with_name("memory.py")
spec = importlib.util.spec_from_file_location("memory", str(module_path))
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


def main():
    db_path = os.path.join("agent", "memory", "memory_test.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)

    print("Initializing DB at:", memory.init_db(db_path))

    thread = memory.get_or_create_thread("email", "alice@example.com")
    print("Thread internal id:", thread)

    # Insert 15 short-term entries to test trimming (default max_entries=12)
    for i in range(15):
        sid = memory.save_short_term(
            thread,
            subject=f"Report {i}",
            sender="alice@example.com",
            prompt=f"prompt {i}",
            response_html=f"<p>resp {i}</p>",
            response_text=f"resp {i}",
            tags=["test"],
            meta={"n": i},
            max_entries=12,
        )
        print("Inserted short_term id", sid)

    rows = memory.get_short_term(thread, limit=20)
    print("Short-term rows (most recent first):", len(rows))
    for r in rows[:3]:
        print(r["subject"], r["created_at"]) if isinstance(r, dict) else print(r)

    # Upsert some long facts
    fid = memory.upsert_long_fact("balance_trend", "Balance is rising", source_short_term_id=rows[0]["id"], weight=2.0)
    print("Upserted long_fact id", fid)

    facts = memory.get_long_facts()
    print("Long facts:", facts)


if __name__ == "__main__":
    main()
