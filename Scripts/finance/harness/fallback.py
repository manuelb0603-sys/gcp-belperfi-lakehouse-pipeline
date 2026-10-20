"""Deterministic fallback report used when the run ends without an acceptable author report."""

from __future__ import annotations

from typing import Any

from ..report_models import ReportSnapshot


def build_fallback_report(snapshot: ReportSnapshot, reason: str = "Model response missing required report_html output.") -> dict[str, Any]:
    """Create a deterministic fallback report if the model never emits a valid final document."""
    latest = float(snapshot.latest_total_net_expense)
    previous = float(snapshot.previous_total_net_expense or latest)
    delta = ((latest - previous) / previous * 100) if previous else 0.0
    trend = "up" if delta >= 0 else "down"
    plain_text = (
        f"Weekly Financial Progress Report\n"
        f"As of {snapshot.as_of}. Latest month: {snapshot.latest_month}. "
        f"Net expenses were ${latest:,.2f} versus ${previous:,.2f} last month ({delta:+.1f}% {trend}). "
        f"Disposable cash is ${snapshot.disposable_cash:,.2f}."
    )
    html = (
        "<html><body><h1>Weekly Financial Progress</h1>"
        f"<p><strong>Fallback report:</strong> {reason}</p>"
        f"<p>Latest month: {snapshot.latest_month}. Net expenses: ${latest:,.2f}</p>"
        f"<p>Previous month: {snapshot.previous_month or 'n/a'}. Net expenses: ${previous:,.2f}</p>"
        f"<p>Month-over-month change: {delta:+.1f}% ({trend})</p>"
        f"<p>Disposable cash: ${snapshot.disposable_cash:,.2f}</p>"
        "</body></html>"
    )
    return {"report_html": html, "plain_text": plain_text}
