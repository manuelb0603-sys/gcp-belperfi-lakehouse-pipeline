"""Shared hand-built transaction fixture for the finance chat tests (gold sign convention)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

TODAY = date(2026, 10, 5)

# (date, issuer, description, amount, transaction_type, category, parent_group)
ROWS = [
    # June 2026
    ("2026-06-15", "Chase", "WHOLEFDS MKT #10234", "-70.00", "Sale", "Groceries", "Needs"),
    # July 2026
    ("2026-07-01", "Chase", "WHOLEFDS MKT #10234", "-100.00", "Sale", "Groceries", "Needs"),
    ("2026-07-03", "Amex", "TRADER JOE S #552         800-555-0199        CA", "-50.00", "Sale", "Groceries", "Needs"),
    ("2026-07-05", "Chase", "SQ *BLUE BOTTLE COFFEE", "-6.50", "Sale", "Food & Drink", "Wants"),
    ("2026-07-10", "Amex", "TST* SUSHI ZEN", "-45.00", "Sale", "Food & Drink", "Wants"),
    ("2026-07-12", "Chase", "AMZN Mktp US*2K3LM1", "-80.00", "Sale", "Shopping", "Wants"),
    ("2026-07-15", "Chase", "AMZN Mktp US*9Q8RT", "20.00", "Return", "Shopping", "Wants"),
    ("2026-07-20", "Amex", "SHELL OIL 57444", "-40.00", "Sale", "Automotive", "Needs"),
    ("2026-07-25", "Chase", "AUTOMATIC PAYMENT - THANK", "500.00", "Payment", "Payments", "Payments & Others"),
    ("2026-07-28", "Amex", "AUTOPAY PAYMENT - THANK YOU", "300.00", "Payment", "Payments", "Payments & Others"),
    # August 2026
    ("2026-08-01", "Chase", "WHOLEFDS MKT #10234", "-120.00", "Sale", "Groceries", "Needs"),
    ("2026-08-04", "Amex", "TRADER JOE S #552         800-555-0199        CA", "-30.00", "Sale", "Groceries", "Needs"),
    ("2026-08-08", "Chase", "COSTCO WHSE #0123", "-25.00", "Sale", "Groceries", "Needs"),
    ("2026-08-09", "Chase", "SQ *BLUE BOTTLE COFFEE", "-12.00", "Sale", "Food & Drink", "Wants"),
    ("2026-08-14", "Amex", "TST* SUSHI ZEN", "-60.00", "Sale", "Food & Drink", "Wants"),
    ("2026-08-16", "Chase", "STARBUCKS STORE 12345", "-8.50", "Sale", "Food & Drink", "Wants"),
    ("2026-08-18", "Chase", "AMAZON.COM*AB12CD", "-200.00", "Sale", "Shopping", "Wants"),
    ("2026-08-20", "Amex", "Platinum Shopping Credit", "15.00", "Return", "Payments", "Payments & Others"),
    ("2026-08-22", "Amex", "SHELL OIL 57444", "-45.00", "Sale", "Automotive", "Needs"),
    ("2026-08-25", "Chase", "PURCHASE INTEREST CHARGE", "-5.00", "Adjustment", "Other", "Payments & Others"),
    ("2026-08-26", "Chase", "AUTOMATIC PAYMENT - THANK", "600.00", "Payment", "Payments", "Payments & Others"),
    # September 2026
    ("2026-09-01", "Chase", "WHOLEFDS MKT #10234", "-90.00", "Sale", "Groceries", "Needs"),
    ("2026-09-02", "Amex", "TRADER JOE S #552         800-555-0199        CA", "-40.00", "Sale", "Groceries", "Needs"),
    ("2026-09-05", "Chase", "COSTCO WHSE #0123", "-35.00", "Sale", "Groceries", "Needs"),
    ("2026-09-06", "Chase", "SQ *BLUE BOTTLE COFFEE", "-7.00", "Sale", "Food & Drink", "Wants"),
    ("2026-09-11", "Amex", "TST* SUSHI ZEN", "-50.00", "Sale", "Food & Drink", "Wants"),
    ("2026-09-13", "Chase", "STARBUCKS STORE 12345", "-9.00", "Sale", "Food & Drink", "Wants"),
    ("2026-09-14", "Chase", "STARBUCKS STORE 12345", "-4.00", "Sale", "Food & Drink", "Wants"),
    ("2026-09-15", "Chase", "AMZN Mktp US*7Y6TR", "-150.00", "Sale", "Shopping", "Wants"),
    ("2026-09-18", "Chase", "TARGET 00012345", "-60.00", "Sale", "Shopping", "Wants"),
    ("2026-09-21", "Chase", "TARGET 00012345", "60.00", "Return", "Shopping", "Wants"),
    ("2026-09-22", "Amex", "SHELL OIL 57444", "-55.00", "Sale", "Automotive", "Needs"),
    ("2026-09-26", "Amex", "CHEVRON 0012", "-30.00", "Sale", "Automotive", "Needs"),
    ("2026-09-27", "Chase", "AUTOMATIC PAYMENT - THANK", "700.00", "Payment", "Payments", "Payments & Others"),
    ("2026-09-30", "Amex", "AUTOPAY PAYMENT - THANK YOU", "400.00", "Payment", "Payments", "Payments & Others"),
    # October 2026 (month to date)
    ("2026-10-02", "Chase", "STARBUCKS STORE 12345", "-10.00", "Sale", "Food & Drink", "Wants"),
    ("2026-10-03", "Amex", "TST* SUSHI ZEN", "-35.00", "Sale", "Food & Drink", "Wants"),
]


def raw_rows() -> list[dict]:
    """Rows as the BigQuery client returns them: Decimal amounts and date objects."""
    keys = ("transaction_date", "issuer", "description", "amount", "transaction_type", "category", "parent_group")
    rows = []
    for values in ROWS:
        row = dict(zip(keys, values))
        row["transaction_date"] = date.fromisoformat(row["transaction_date"])
        row["amount"] = Decimal(row["amount"])
        rows.append(row)
    return rows


def fixture_frame():
    from finance.chat.data import frame_from_rows

    return frame_from_rows(raw_rows())
