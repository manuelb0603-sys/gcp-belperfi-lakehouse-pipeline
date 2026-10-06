"""Shared hand-built transaction fixture for the finance chat tests (gold sign convention)."""

from __future__ import annotations

from datetime import date, timedelta
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


# ----- profile fixtures -----

CATEGORY_GROUPS = (
    ("Groceries", "Needs"), ("Food & Drink", "Wants"), ("Shopping", "Wants"), ("Automotive", "Needs"),
    ("Bills & Utilities", "Needs"), ("Health & Wellness", "Needs"), ("Travel", "Wants"), ("Entertainment", "Wants"),
    ("Professional & Business Services", "Payments & Others"), ("Gifts & Donations", "Wants"), ("Other", "Payments & Others"),
)
_SURNAMES = (
    "Adams Baker Carter Davis Ellis Foster Garcia Harris Irwin Jensen Keller Lopez Morris Nolan Owens Parker Quinn "
    "Reyes Sanders Turner Underwood Vargas Walker Young Zimmerman Brooks Collins Dawson Fleming Grant Hughes Ingram "
    "Jordan Knight Lawson Mason Norris Ortiz Porter Russell"
).split()
_SUFFIXES = ("LLC", "Inc", "Group", "PLC", "Ltd", "and Sons")

SPARSE_TODAY = date(2026, 10, 6)
SPARSE_START, SPARSE_END = date(2025, 8, 20), date(2026, 8, 19)
SPARSE_MERCHANTS = 531


def _company_names(count: int, rng) -> list[str]:
    names = [f"{a} {b} {suffix}" for a in _SURNAMES for b in _SURNAMES if a != b for suffix in _SUFFIXES[:1]]
    names += [f"{a}-{b} {suffix}" for a in _SURNAMES for b in _SURNAMES if a != b for suffix in _SUFFIXES[1:2]]
    rng.shuffle(names)
    return names[:count]


def _row(day, issuer, description, amount, kind, category, group) -> dict:
    return {"transaction_date": day, "issuer": issuer, "description": description, "amount": Decimal(f"{amount:.2f}"),
            "transaction_type": kind, "category": category, "parent_group": group}


def sparse_merchant_rows(seed: int = 42) -> list[dict]:
    """Shaped like the mock gold data: 600 purchases/returns from 2025-08-20 to 2026-08-19 at 531 nearly
    unique (Faker-style company) merchants, plus monthly card payments."""
    import random

    rng = random.Random(seed)
    names = _company_names(SPARSE_MERCHANTS, rng)
    merchants = names + [rng.choice(names) for _ in range(600 - len(names))]
    rng.shuffle(merchants)
    span = (SPARSE_END - SPARSE_START).days
    rows = []
    for index, name in enumerate(merchants):
        day = SPARSE_START + timedelta(days=0 if index == 0 else span if index == 1 else rng.randint(0, span))
        category, group = CATEGORY_GROUPS[index % len(CATEGORY_GROUPS)]
        returned = index % 30 == 29
        amount = rng.uniform(5, 300) * (1 if returned else -1)
        rows.append(_row(day, rng.choice(("Amex", "Chase")), name.upper(), amount, "Return" if returned else "Sale",
                         category, group))
    month = date(2025, 9, 1)
    while month <= SPARSE_END:
        rows.append(_row(month + timedelta(days=4), "Chase", "AUTOMATIC PAYMENT - THANK", 900, "Payment",
                         "Payments", "Payments & Others"))
        month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    return rows


REPEATING_TODAY = date(2026, 10, 6)
REPEATING_START, REPEATING_END = date(2026, 3, 1), date(2026, 10, 3)
REPEATING_MERCHANTS = (
    ("WHOLEFDS MKT #10234", "Groceries", "Needs"), ("TRADER JOE S #552", "Groceries", "Needs"),
    ("SAFEWAY #1188", "Groceries", "Needs"), ("STARBUCKS STORE 12345", "Food & Drink", "Wants"),
    ("SQ *BLUE BOTTLE COFFEE", "Food & Drink", "Wants"), ("TST* SUSHI ZEN", "Food & Drink", "Wants"),
    ("CHIPOTLE 0456", "Food & Drink", "Wants"), ("AMZN Mktp US*2K3LM1", "Shopping", "Wants"),
    ("TARGET 00012345", "Shopping", "Wants"), ("SHELL OIL 57444", "Automotive", "Needs"),
    ("CHEVRON 0012", "Automotive", "Needs"), ("COMCAST CABLE", "Bills & Utilities", "Needs"),
    ("CVS PHARMACY #0789", "Health & Wellness", "Needs"), ("NETFLIX.COM", "Entertainment", "Wants"),
)


def repeating_merchant_rows(seed: int = 7) -> list[dict]:
    """Realistic recurring merchants: 14 merchants, about 20 visits each, 2026-03-01 to 2026-10-03."""
    import random

    rng = random.Random(seed)
    span = (REPEATING_END - REPEATING_START).days
    rows = []
    for index in range(280):
        description, category, group = REPEATING_MERCHANTS[index % len(REPEATING_MERCHANTS)]
        day = REPEATING_START + timedelta(days=0 if index == 0 else span if index == 1 else rng.randint(0, span))
        rows.append(_row(day, "Amex" if index % 3 else "Chase", description, -rng.uniform(4, 180), "Sale", category, group))
    return rows


def frame_of(rows):
    from finance.chat.data import frame_from_rows

    return frame_from_rows(rows)
