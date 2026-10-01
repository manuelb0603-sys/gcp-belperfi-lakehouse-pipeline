import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import email_weekly_report
import memory


class TestLongFactVectorMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "memory.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_migrates_existing_long_facts_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE long_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_key TEXT UNIQUE NOT NULL,
                fact_text TEXT NOT NULL,
                source_short_term_id INTEGER,
                weight REAL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO long_facts (fact_key, fact_text, weight, updated_at) VALUES (?, ?, ?, ?)",
            ("old_fact", "Legacy fact", 1.0, "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        conn.close()

        memory.init_db(self.db_path)

        self.assertEqual(memory.get_long_facts_needing_embedding("embed-v1"), [
            {"fact_key": "old_fact", "fact_text": "Legacy fact"}
        ])
        migrated = sqlite3.connect(self.db_path)
        try:
            columns = {row[1] for row in migrated.execute("PRAGMA table_info(long_facts)")}
        finally:
            migrated.close()
        self.assertIn("embedding", columns)
        self.assertIn("embedding_model", columns)

    def test_vector_search_ranks_by_cosine_and_hides_embedding(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact(
            "goal", "Prefers saving toward a house", embedding=[1.0, 0.0], embedding_model="embed-v1"
        )
        memory.upsert_long_fact(
            "style", "Likes brief weekly summaries", embedding=[0.0, 1.0], embedding_model="embed-v1"
        )

        results = memory.search_long_facts([1.0, 0.0], "embed-v1", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fact_key"], "goal")
        self.assertAlmostEqual(results[0]["similarity"], 1.0)
        self.assertNotIn("embedding", results[0])
        self.assertEqual(results[0]["embedding_model"], "embed-v1")

    def test_updating_fact_without_vector_invalidates_stale_embedding(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact(
            "goal", "Old text", embedding=[1.0, 0.0], embedding_model="embed-v1"
        )

        memory.upsert_long_fact("goal", "Revised text")

        self.assertEqual(memory.get_long_facts_needing_embedding("embed-v1"), [
            {"fact_key": "goal", "fact_text": "Revised text"}
        ])

    def test_search_ignores_dimension_mismatch(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact(
            "wrong_size", "Wrong vector size", embedding=[1.0, 0.0, 0.0], embedding_model="embed-v1"
        )

        self.assertEqual(memory.search_long_facts([1.0, 0.0], "embed-v1"), [])

    def test_search_and_backfill_only_use_requested_embedding_model(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact(
            "fact", "Example fact", embedding=[1.0, 0.0], embedding_model="embed-v1"
        )

        self.assertEqual(memory.search_long_facts([1.0, 0.0], "embed-v2"), [])
        self.assertEqual(memory.get_long_facts_needing_embedding("embed-v2"), [
            {"fact_key": "fact", "fact_text": "Example fact"}
        ])

    def test_bm25_finds_exact_terms_and_prefers_fact_key_match(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact("travel_goal", "Prefers saving for a trip", weight=1.0)
        memory.upsert_long_fact("misc", "The travel budget is a recurring concern", weight=1.0)
        memory.upsert_long_fact("dining", "Enjoys trying new restaurants", weight=1.0)

        results = memory.search_long_facts_bm25("travel goal", limit=5)

        self.assertEqual([item["fact_key"] for item in results[:2]], ["travel_goal", "misc"])
        self.assertEqual(results[0]["matched_terms"], ["goal", "travel"])
        self.assertGreater(results[0]["bm25_score"], results[1]["bm25_score"])
        self.assertNotIn("dining", [item["fact_key"] for item in results])

    def test_hybrid_fusion_deduplicates_and_rewards_cross_retriever_match(self):
        vector_results = [
            {"id": 1, "fact_key": "semantic", "fact_text": "Semantic result", "similarity": 0.95},
            {"id": 2, "fact_key": "overlap", "fact_text": "Shared result", "similarity": 0.80},
        ]
        lexical_results = [
            {"id": 2, "fact_key": "overlap", "fact_text": "Shared result", "bm25_score": 2.0},
            {"id": 3, "fact_key": "keyword", "fact_text": "Keyword result", "bm25_score": 1.0},
        ]

        results = memory.combine_fact_search_results(
            vector_results, lexical_results, limit=3, rrf_k=0
        )

        self.assertEqual([item["id"] for item in results], [2, 1, 3])
        self.assertEqual(results[0]["retrieval_sources"], ["vector", "bm25"])
        self.assertEqual(results[0]["similarity"], 0.80)
        self.assertEqual(results[0]["bm25_score"], 2.0)

    def test_bm25_search_skips_queries_without_meaningful_tokens(self):
        memory.init_db(self.db_path)
        memory.upsert_long_fact("goal", "Save for travel")

        self.assertEqual(memory.search_long_facts_bm25("what is it"), [])


class TestOllamaEmbeddingClient(unittest.TestCase):
    def test_embedding_uses_ollama_embed_endpoint(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"embeddings": [[0.25, 0.75]]}

        class FakeRequests:
            def __init__(self):
                self.url = None
                self.payload = None

            def post(self, url, json, timeout):
                self.url = url
                self.payload = json
                return FakeResponse()

        fake_requests = FakeRequests()
        with patch.dict(sys.modules, {"requests": fake_requests}):
            vector = email_weekly_report._ollama_embedding(
                "user savings goal", "http://localhost:11434/api/generate", "nomic-embed-text", 12
            )

        self.assertEqual(fake_requests.url, "http://localhost:11434/api/embed")
        self.assertEqual(fake_requests.payload, {
            "model": "nomic-embed-text",
            "input": "user savings goal",
        })
        self.assertEqual(vector, [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
