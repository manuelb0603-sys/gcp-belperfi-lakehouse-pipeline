import unittest
from datetime import date
from unittest import mock

import pandas as pd

from pipeline import generate_mock_cd
from tests.chat_test_fixtures import TODAY, raw_rows
from finance.chat import data
from finance.settings import ReportSettings


class FakeJob:
    def __init__(self, rows):
        self.rows = rows

    def result(self):
        return iter(self.rows)


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, sql, job_config=None):
        self.calls.append((sql, job_config))
        return FakeJob(self.rows)


def settings(**extra):
    return ReportSettings.from_config({"project_name": "demo-project", **extra})


class TestFrame(unittest.TestCase):
    def setUp(self):
        data.refresh()
        self.addCleanup(data.refresh)

    def test_load_builds_typed_frame_from_rows(self):
        client = FakeClient(raw_rows())
        frame = data.load_transactions(settings(), date(2026, 6, 1), client=client)
        sql, job_config = client.calls[0]
        self.assertIn("`demo-project.gold.fact_transactions`", sql)
        self.assertIn("transaction_date >= @start", sql)
        parameter = job_config.query_parameters[0]
        self.assertEqual((parameter.name, parameter.type_, parameter.value), ("start", "DATE", date(2026, 6, 1)))

        self.assertEqual(len(frame), len(raw_rows()))
        self.assertEqual(str(frame["transaction_date"].dtype), "datetime64[ns]")
        self.assertEqual(frame["amount"].dtype, float)
        self.assertEqual(str(frame["month"].dtype), "period[M]")
        self.assertTrue(frame["weekday"].cat.ordered)
        self.assertEqual(list(frame["weekday"].cat.categories), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        first = frame.iloc[1]
        self.assertEqual((first["merchant"], first["amount"], first["spend"], str(first["month"]), first["weekday"]),
                         ("Wholefds Mkt", -100.0, 100.0, "2026-07", "Wed"))
        self.assertEqual(frame.loc[frame["description"].str.startswith("AUTOPAY"), "merchant"].iloc[0], "Card Payment")

    def test_default_start_is_24_months_back(self):
        self.assertEqual(data.default_start(TODAY), date(2024, 10, 1))
        client = FakeClient([])
        with mock.patch.object(data, "default_start", return_value=date(2024, 10, 1)):
            frame = data.load_transactions(settings(), client=client)
        self.assertEqual(client.calls[0][1].query_parameters[0].value, date(2024, 10, 1))
        self.assertTrue(frame.empty)
        self.assertIn("merchant", frame.columns)

    def test_merchant_aliases_from_settings(self):
        frame = data.load_transactions(settings(merchant_aliases={"WHOLEFDS MKT": "Whole Foods"}), "2026-06-01",
                                       client=FakeClient(raw_rows()))
        self.assertIn("Whole Foods", frame["merchant"].tolist())

    def test_missing_project_is_an_error(self):
        with self.assertRaises(ValueError):
            data.load_transactions(ReportSettings.from_config({}), client=FakeClient([]))


class TestCache(unittest.TestCase):
    def setUp(self):
        data.refresh()
        self.addCleanup(data.refresh)
        self.now = [1000.0]
        patcher = mock.patch.object(data, "clock", lambda: self.now[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ttl_and_refresh(self):
        client = FakeClient(raw_rows())
        s = settings(chat_data_cache_seconds=60)
        first = data.load_transactions(s, "2026-06-01", client=client)
        first.loc[0, "merchant"] = "mutated"  # callers get copies
        self.now[0] += 59
        second = data.load_transactions(s, "2026-06-01", client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertNotEqual(second.loc[0, "merchant"], "mutated")
        data.load_transactions(s, "2026-07-01", client=client)  # different start -> new query
        self.assertEqual(len(client.calls), 2)
        self.now[0] += 2
        data.load_transactions(s, "2026-06-01", client=client)  # expired
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(data.refresh("demo-project"), 2)
        data.load_transactions(s, "2026-06-01", client=client)
        self.assertEqual(len(client.calls), 4)

    def test_zero_ttl_disables_cache(self):
        client = FakeClient(raw_rows())
        s = settings(chat_data_cache_seconds=0)
        data.load_transactions(s, "2026-06-01", client=client)
        data.load_transactions(s, "2026-06-01", client=client)
        self.assertEqual(len(client.calls), 2)

    def test_settings_defaults_and_validation(self):
        s = ReportSettings.from_config({})
        self.assertEqual((s.chat_data_cache_seconds, s.merchant_aliases), (900, {}))
        for bad in ({"chat_data_cache_seconds": -1}, {"chat_data_cache_seconds": "soon"}, {"merchant_aliases": ["x"]}):
            with self.assertRaises(ValueError):
                ReportSettings.from_config(bad)


class TestCatalog(unittest.TestCase):
    def test_catalog(self):
        frame = data.frame_from_rows(raw_rows())
        catalog = data.build_catalog(frame)
        self.assertEqual(catalog["date_range"], {"min": "2026-06-15", "max": "2026-10-03", "months": 5})
        self.assertEqual(catalog["issuers"], ["Amex", "Chase"])
        self.assertEqual(catalog["categories"]["Groceries"], ["Needs"])
        self.assertNotIn("Card Payment", catalog["merchants"])  # payments are not analysis rows
        self.assertEqual(catalog["merchants"][:3], ["Amazon", "Wholefds Mkt", "Sushi Zen"])
        self.assertEqual(catalog["transactions"], 32)
        self.assertIn("net_spend", catalog["metrics"])
        self.assertLess(len(str(catalog)), 4000)

    def test_empty_catalog(self):
        catalog = data.build_catalog(data.frame_from_rows([]))
        self.assertEqual(catalog["date_range"]["months"], 0)
        self.assertEqual(catalog["merchants"], [])


class TestMockGenerator(unittest.TestCase):
    def test_output_shape_and_conventions(self):
        chase = generate_mock_cd.generate_chase_transactions(200, seed=7, today=TODAY)
        amex = generate_mock_cd.generate_amex_transactions(200, seed=7, today=TODAY)
        self.assertEqual(list(chase.columns), generate_mock_cd.CHASE_COLUMNS)
        self.assertEqual(list(amex.columns), generate_mock_cd.AMEX_COLUMNS)
        self.assertEqual((len(chase), len(amex)), (200, 200))
        dates = pd.to_datetime(chase["Transaction Date"], format="%m/%d/%Y")
        self.assertTrue(((dates >= pd.Timestamp("2025-10-05")) & (dates <= pd.Timestamp(TODAY))).all())
        pd.to_datetime(amex["Date"], format="%m/%d/%Y")
        sales = chase[chase["Type"] == "Sale"]
        self.assertTrue((sales["Amount"] < 0).all())
        payments = chase[chase["Type"] == "Payment"]
        self.assertTrue((payments["Category"] == "").all() and (payments["Amount"] > 0).all())
        self.assertTrue((amex.loc[amex["Category"] != "", "Amount"] != 0).all())
        self.assertTrue(amex["Description"].str.contains(r"\d{3}-\d{3}-\d{4}\s+[A-Z]{2}$").any())
        self.assertTrue(chase["Description"].str.startswith(("SQ *", "TST* ")).any())

    def test_merchants_recur(self):
        from finance.chat.merchants import normalize_merchant

        chase = generate_mock_cd.generate_chase_transactions(300, seed=3, today=TODAY)
        counts = chase["Description"].map(normalize_merchant).value_counts()
        self.assertGreaterEqual(counts.iloc[0], 10)
        self.assertLess(counts.size, 150)

    def test_seed_is_reproducible(self):
        a = generate_mock_cd.generate_amex_transactions(20, seed=11, today=TODAY)
        b = generate_mock_cd.generate_amex_transactions(20, seed=11, today=TODAY)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
