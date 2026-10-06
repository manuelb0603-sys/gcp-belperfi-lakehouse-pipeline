import json
import unittest
from datetime import date
from decimal import Decimal

from tests.chat_test_fixtures import TODAY, fixture_frame
from finance.chat.analysis_spec import AnalysisSpec
from finance.chat.charts import ACCENT, OTHER_COLOR, PALETTE, SEQUENTIAL, build_figure, color_map
from finance.chat.data import frame_from_rows
from finance.chat.executor import AnalysisContext, run_analysis

FRAME = fixture_frame()


def figure(frame=FRAME, **spec):
    result = run_analysis(frame, AnalysisSpec(**spec), AnalysisContext(today=TODAY))
    return result, build_figure(result)


class TestFigures(unittest.TestCase):
    def test_line_by_month(self):
        result, fig = figure(group_by=["month"], period="last_quarter")
        self.assertEqual([trace.type for trace in fig.data], ["scatter"])
        self.assertEqual(list(fig.data[0].x), ["2026-07", "2026-08", "2026-09"])
        self.assertEqual(fig.data[0].line.color, ACCENT)
        self.assertEqual(fig.layout.title.text, "Net spend by month · Jul–Sep 2026 (last quarter)")
        self.assertEqual(fig.layout.yaxis.tickprefix, "$")
        self.assertEqual(fig.layout.yaxis.title.text, "Net spend ($)")
        self.assertIn("$%{y:,.2f}", fig.data[0].hovertemplate)

    def test_horizontal_bar_by_category(self):
        _, fig = figure(group_by=["category"], period="last_quarter")
        self.assertEqual(fig.data[0].type, "bar")
        self.assertEqual(fig.data[0].orientation, "h")
        self.assertEqual(fig.layout.xaxis.tickprefix, "$")
        self.assertEqual(list(fig.layout.yaxis.categoryarray)[-1], "Groceries")  # largest on top

    def test_stacked_bar_month_by_category_uses_stable_colors(self):
        _, fig = figure(metric="gross_spend", group_by=["month", "category"], period="last_quarter")
        self.assertEqual(fig.layout.barmode, "stack")
        self.assertEqual(len(fig.data), 6)
        colors = {trace.name: trace.marker.color for trace in fig.data}
        self.assertEqual(colors["Groceries"], PALETTE[0])
        self.assertEqual(colors["Food &amp; Drink"], PALETTE[1])
        self.assertEqual(len(set(colors.values())), 6)

    def test_grouped_bar_and_comparison(self):
        _, fig = figure(group_by=["category", "issuer"], period="last_quarter", chart="grouped_bar")
        self.assertEqual(fig.layout.barmode, "group")
        _, fig = figure(group_by=["category"], period="last_month", compare_to="previous_period")
        self.assertEqual([trace.name for trace in fig.data], ["Net spend", "August 2026 (previous period)"])

    def test_heatmap_scales(self):
        result, fig = figure(metric="transaction_count", group_by=["month", "weekday"], period="last_quarter")
        self.assertEqual(result.chart, "heatmap")
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertEqual(list(fig.data[0].y), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        self.assertEqual(list(fig.data[0].x), ["2026-07", "2026-08", "2026-09"])
        scale = [color for _, color in fig.layout.coloraxis.colorscale]
        self.assertEqual(scale, [color for _, color in SEQUENTIAL])
        self.assertIsNone(fig.layout.coloraxis.cmid)
        _, fig = figure(metric="net_spend", group_by=["month", "category"], period="last_quarter", chart="heatmap")
        self.assertEqual(fig.layout.coloraxis.cmid, 0)  # Payments credit is negative -> diverging

    def test_pie_and_treemap(self):
        _, fig = figure(metric="share_of_total", group_by=["issuer"], period="last_quarter")
        self.assertEqual(fig.data[0].type, "pie")
        self.assertIn("%{value:,.1f}%", fig.data[0].hovertemplate)
        _, fig = figure(metric="gross_spend", group_by=["parent_group", "category"], period="last_quarter", chart="treemap")
        self.assertEqual(fig.data[0].type, "treemap")

    def test_table_and_none_return_none(self):
        self.assertIsNone(figure(period="last_quarter")[1])
        self.assertIsNone(figure(group_by=["category"], period="last_quarter", chart="table")[1])
        self.assertIsNone(figure(group_by=["category"], period="last_quarter", chart="none")[1])

    def test_count_axis_and_custom_title(self):
        _, fig = figure(metric="transaction_count", group_by=["weekday"], period="last_month", title="Busy days")
        self.assertEqual(fig.layout.title.text, "Busy days")
        self.assertEqual(fig.data[0].orientation, None)  # weekday stays vertical
        self.assertEqual(fig.layout.yaxis.tickformat, ",d")

    def test_pie_with_integer_day_labels_renders(self):
        result = run_analysis(FRAME, AnalysisSpec(metric="net_spend", group_by=["day_of_month"], chart="bar", period="all"), AnalysisContext(today=TODAY))
        result.chart = "pie"
        fig = build_figure(result)
        self.assertEqual(fig.data[0].type, "pie")
        self.assertEqual(len(fig.data[0].marker.colors), len(fig.data[0].labels))
        self.assertNotIn(OTHER_COLOR, fig.data[0].marker.colors[:3])


class TestEscaping(unittest.TestCase):
    def test_every_label_is_html_escaped(self):
        rows = [
            {"transaction_date": date(2026, 9, 3), "issuer": "Chase", "description": "SHOP <b>X</b>",
             "amount": Decimal("-20"), "transaction_type": "Sale", "category": "<img src=x onerror=alert(1)>", "parent_group": "Wants"},
            {"transaction_date": date(2026, 9, 4), "issuer": "Amex & <i>Co</i>", "description": "CAFE",
             "amount": Decimal("-5"), "transaction_type": "Sale", "category": "Food & Drink", "parent_group": "Wants"},
        ]
        frame = frame_from_rows(rows)
        for spec in (
            {"group_by": ["category"]},
            {"group_by": ["month", "category"], "chart": "stacked_bar"},
            {"group_by": ["issuer", "category"], "chart": "heatmap"},
            {"group_by": ["category"], "chart": "pie"},
            {"group_by": ["month", "issuer"], "chart": "line", "title": "<script>x</script>"},
        ):
            _, fig = figure(frame, period="last_month", **spec)
            text = json.dumps(json.loads(fig.to_json()), ensure_ascii=False)  # undo plotly's \u003c escapes
            self.assertNotIn("<img", text, spec)
            self.assertNotIn("<i>", text, spec)
            self.assertNotIn("<script>", text, spec)
        _, fig = figure(frame, period="last_month", group_by=["category"])
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", list(fig.data[0].y))

    def test_color_map(self):
        colors = color_map(["Shopping", "Other", "Groceries", "<b>"])
        self.assertEqual(colors["Other"], OTHER_COLOR)
        self.assertEqual(colors["Groceries"], PALETTE[0])
        self.assertIn("&lt;b&gt;", colors)
        self.assertEqual(color_map(["Groceries"])["Groceries"], color_map(["Travel", "Groceries"])["Groceries"])


if __name__ == "__main__":
    unittest.main()
