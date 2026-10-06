"""Generate mock Chase and Amex card CSVs with realistic, recurring merchants.

Merchants come from a fixed pool per category (8-15 each) with frequency weights,
weekday patterns, a few store locations each, and issuer-style descriptor noise
(store numbers, ``SQ *``/``TST*`` processor prefixes, padded Amex descriptions with
phone and state), so merchant and weekday analysis in the finance chat is meaningful.
Output columns and formats match the real issuer exports.

Run from any folder; the CSVs are written to the current directory:
    ..\\.venv\\Scripts\\python.exe Scripts\\pipeline\\generate_mock_cd.py [--records N] [--seed S]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from faker import Faker

CHASE_COLUMNS = ['Transaction Date', 'Post Date', 'Description', 'Category', 'Type', 'Amount', 'Memo']
AMEX_COLUMNS = [
    'Date', 'Description', 'Card Member', 'Account #', 'Amount',
    'Extended Details', 'Appears On Your Statement As', 'Address',
    'City/State', 'Zip Code', 'Country', 'Reference', 'Category',
]

# Relative weekday weights, Monday..Sunday.
WEEKDAYS_EVEN = (1, 1, 1, 1, 1, 1, 1)
WEEKDAYS_WEEKEND = (0.6, 0.6, 0.7, 0.8, 1.3, 1.8, 1.5)
WEEKDAYS_EVENINGS_OUT = (0.5, 0.6, 0.8, 1.0, 1.8, 2.0, 1.0)
WEEKDAYS_COMMUTE = (1.4, 1.3, 1.3, 1.3, 1.2, 0.5, 0.4)


@dataclass(frozen=True)
class Merchant:
    name: str
    low: float
    high: float
    weight: float = 1.0  # relative visit frequency within the pool
    prefix: str = ''  # processor prefix, e.g. 'SQ *' or 'TST* '
    stores: int = 1  # number of locations (each gets its own store number)


@dataclass(frozen=True)
class Pool:
    merchants: tuple[Merchant, ...]
    weekdays: tuple[float, ...] = WEEKDAYS_EVEN
    weight: float = 1.0  # relative share of all purchases


def _m(name, low, high, weight=1.0, prefix='', stores=1):
    return Merchant(name, low, high, weight, prefix, stores)


POOLS: dict[str, Pool] = {
    'groceries': Pool((
        _m('WHOLEFDS MKT', 35, 180, 3, stores=2), _m('TRADER JOE S', 25, 120, 4, stores=2),
        _m('SAFEWAY', 20, 150, 2, stores=2), _m('COSTCO WHSE', 80, 350, 1.5),
        _m('KROGER', 20, 140, 1.5), _m('SPROUTS FARMERS MKT', 15, 90, 1),
        _m('ALDI', 15, 80, 1), _m('H MART', 20, 110, 0.8), _m('FARMERS MARKET STAND', 8, 40, 0.5, prefix='SQ *'),
    ), WEEKDAYS_WEEKEND, 3.0),
    'dining': Pool((
        _m('SUSHI ZEN', 30, 95, 2, prefix='TST* '), _m('RAMEN HOUSE', 18, 45, 2, prefix='TST* '),
        _m('CHIPOTLE', 10, 25, 2.5, stores=3), _m('SWEETGREEN', 12, 22, 2, stores=2),
        _m('OLIVE GARDEN', 35, 90, 0.8), _m('TACO TRUCK EL GUERO', 9, 24, 1, prefix='SQ *'),
        _m('PHO SAIGON', 14, 35, 1.2, prefix='TST* '), _m('SHAKE SHACK', 12, 30, 1.2, stores=2),
        _m('THE CHEESECAKE FACTORY', 40, 120, 0.5), _m('IZAKAYA MOGU', 45, 140, 0.7, prefix='TST* '),
        _m('DOORDASH', 20, 60, 1.5),
    ), WEEKDAYS_EVENINGS_OUT, 2.2),
    'coffee': Pool((
        _m('STARBUCKS', 4, 12, 5, stores=4), _m('BLUE BOTTLE COFFEE', 5, 14, 2, prefix='SQ *'),
        _m('PEETS COFFEE', 4, 11, 2, stores=2), _m('DUTCH BROS', 4, 10, 1.5),
        _m('LOCAL BEAN CAFE', 4, 12, 1.5, prefix='SQ *'), _m('BOBA GUYS', 5, 9, 1.2, prefix='SQ *'),
        _m('THE TAP ROOM', 15, 60, 0.8, prefix='TST* '), _m('DUNKIN', 3, 9, 1),
    ), WEEKDAYS_COMMUTE, 2.0),
    'online': Pool((
        _m('AMZN MKTP US', 10, 220, 6), _m('AMAZON.COM', 15, 260, 3), _m('ETSY.COM', 12, 80, 1),
        _m('TARGET.COM', 15, 150, 1.5), _m('BEST BUY.COM', 30, 400, 0.6), _m('CHEWY.COM', 25, 90, 1),
        _m('UNIQLO USA', 20, 120, 0.8), _m('APPLE.COM/BILL', 1, 15, 2), _m('WAYFAIR', 40, 350, 0.4),
    ), WEEKDAYS_EVEN, 2.0),
    'general': Pool((
        _m('TARGET', 15, 160, 3, stores=2), _m('WALMART', 10, 140, 2, stores=2), _m('IKEA', 30, 300, 0.5),
        _m('HOME DEPOT', 15, 260, 1.2, stores=2), _m('LOWES', 15, 220, 0.8), _m('CVS PHARMACY', 5, 60, 1.5, stores=3),
        _m('WALGREENS', 5, 50, 1.2, stores=2), _m('MARSHALLS', 20, 120, 0.7), _m('REI', 25, 250, 0.4),
        _m('DICKS SPORTING GOODS', 20, 200, 0.4), _m('STAPLES', 8, 90, 0.5),
    ), WEEKDAYS_WEEKEND, 1.8),
    'fuel': Pool((
        _m('SHELL OIL', 30, 75, 3, stores=3), _m('CHEVRON', 30, 80, 2.5, stores=2), _m('ARCO', 25, 65, 2, stores=2),
        _m('COSTCO GAS', 35, 70, 1.5), _m('76 STATION', 30, 70, 1), _m('VALERO', 28, 68, 1),
        _m('JIFFY LUBE', 60, 120, 0.3), _m('CITY PARKING', 5, 30, 1, prefix='PY *'),
    ), WEEKDAYS_COMMUTE, 1.6),
    'transit': Pool((
        _m('UBER *TRIP', 9, 45, 3), _m('LYFT *RIDE', 9, 40, 2), _m('CLIPPER TRANSIT', 5, 40, 1.5),
        _m('BART', 3, 12, 1), _m('YELLOW CAB', 12, 50, 0.5), _m('AMTRAK', 30, 120, 0.3),
        _m('LIME *RIDE', 3, 10, 0.8), _m('SPOTHERO', 8, 30, 0.6),
    ), WEEKDAYS_COMMUTE, 0.9),
    'travel': Pool((
        _m('DELTA AIR LINES', 120, 650, 1), _m('UNITED AIRLINES', 120, 600, 1), _m('ALASKA AIRLINES', 90, 400, 1),
        _m('SOUTHWEST AIRLINES', 80, 350, 1), _m('MARRIOTT HOTELS', 150, 450, 0.8), _m('AIRBNB', 120, 600, 0.8),
        _m('HILTON HOTELS', 140, 420, 0.6), _m('HERTZ RENT-A-CAR', 80, 300, 0.4),
    ), WEEKDAYS_EVEN, 0.35),
    'bills': Pool((
        _m('PG&E WEB ONLINE', 60, 180, 1), _m('COMCAST XFINITY', 70, 95, 1), _m('VERIZON WIRELESS', 85, 130, 1),
        _m('NETFLIX.COM', 15.49, 15.49, 1), _m('SPOTIFY USA', 11.99, 11.99, 1), _m('CITY WATER UTIL', 35, 80, 1),
        _m('GOOGLE *STORAGE', 2.99, 2.99, 1), _m('T-MOBILE', 50, 90, 1),
    ), WEEKDAYS_EVEN, 0.4),
    'health': Pool((
        _m('CVS PHARMACY', 8, 60, 2, stores=2), _m('WALGREENS', 8, 50, 1.5, stores=2), _m('KAISER PERMANENTE', 20, 150, 1),
        _m('24 HOUR FITNESS', 39.99, 39.99, 1), _m('ONE MEDICAL', 15, 90, 0.5), _m('DENTAL CARE CTR', 40, 250, 0.3),
        _m('YOGA WORKS', 20, 30, 1, prefix='SQ *'), _m('GNC', 15, 70, 0.5),
    ), WEEKDAYS_EVEN, 0.8),
    'gifts': Pool((
        _m('RED CROSS', 25, 100, 1), _m('GOFUNDME', 20, 100, 1), _m('1-800-FLOWERS', 40, 120, 0.8),
        _m('HALLMARK', 5, 30, 1), _m('UNICEF USA', 25, 80, 0.6), _m('LOCAL FOOD BANK', 20, 75, 0.8),
        _m('GIFT SHOP BOUTIQUE', 15, 90, 1, prefix='SQ *'), _m('EDIBLE ARRANGEMENTS', 40, 110, 0.5),
    ), WEEKDAYS_WEEKEND, 0.3),
    'personal': Pool((
        _m('GREAT CLIPS', 18, 30, 1), _m('SUPERCUTS', 20, 35, 0.8), _m('ULTA BEAUTY', 15, 90, 1),
        _m('SEPHORA', 20, 120, 0.8), _m('NAIL SPA', 25, 60, 1, prefix='SQ *'), _m('DRY CLEANERS', 15, 45, 1, prefix='SQ *'),
        _m('BARBER SHOP', 25, 45, 1, prefix='SQ *'), _m('MASSAGE ENVY', 60, 110, 0.4),
    ), WEEKDAYS_WEEKEND, 0.5),
    'office': Pool((
        _m('STAPLES', 8, 90, 2), _m('OFFICE DEPOT', 8, 80, 1.5), _m('FEDEX OFFICE', 5, 60, 1),
        _m('UPS STORE', 8, 45, 1), _m('ZOOM.US', 15.99, 15.99, 1), _m('ADOBE', 22.99, 22.99, 1),
        _m('DROPBOX', 11.99, 11.99, 0.8), _m('VISTAPRINT', 15, 70, 0.5),
    ), WEEKDAYS_COMMUTE, 0.4),
}

CHASE_CATEGORIES = {
    'Groceries': 'groceries', 'Shopping': 'online', 'Automotive': 'fuel', 'Health & Wellness': 'health',
    'Food & Drink': 'dining', 'Travel': 'travel', 'Bills & Utilities': 'bills', 'Gifts & Donations': 'gifts',
    'Personal': 'personal',
}
CHASE_EXTRA = {'Food & Drink': 'coffee', 'Shopping': 'general'}  # second pool sharing the category

AMEX_CATEGORIES = {
    'Merchandise & Supplies-Internet Purchase': 'online',
    'Merchandise & Supplies-General Merchandise': 'general',
    'Groceries': 'groceries',
    'Restaurant-Bar & Cafe': 'coffee',
    'Restaurant-Restaurants': 'dining',
    'Business Services-Office Supplies': 'office',
    'Communications-Mobile Telecom': 'bills',
    'Transportation-Taxis & Coach': 'transit',
    'Transportation-Airlines': 'travel',
    'Automotive-Fuel': 'fuel',
    'Merchandise & Supplies-Home Improvement': 'general',
    'Health & Wellness': 'health',
    'Merchandise & Supplies-Sporting Goods': 'general',
}

CITIES = [('SAN FRANCISCO', 'CA'), ('OAKLAND', 'CA'), ('SEATTLE', 'WA'), ('PORTLAND', 'OR'), ('AUSTIN', 'TX'),
          ('DENVER', 'CO'), ('NEW YORK', 'NY'), ('CHICAGO', 'IL')]


class MockGenerator:
    """Deterministic (seeded) mock transaction generator."""

    def __init__(self, seed: int | None = None, today: date | None = None):
        self.rng = random.Random(seed)
        self.fake = Faker()
        if seed is not None:
            self.fake.seed_instance(seed)
        self.today = today or date.today()
        self.start = self.today - timedelta(days=365)
        self._locations: dict[tuple[str, int], dict] = {}

    # ----- shared helpers -----

    def location(self, merchant: Merchant, store: int) -> dict:
        """Fixed store number, phone, city, and state per merchant location."""
        key = (merchant.name, store)
        if key not in self._locations:
            city, state = self.rng.choice(CITIES)
            self._locations[key] = {
                'number': self.rng.choice([f'#{self.rng.randint(100, 9999)}', f'{self.rng.randint(1000, 99999):05d}', '']),
                'phone': f'{self.rng.randint(200, 999)}-{self.rng.randint(200, 999)}-{self.rng.randint(1000, 9999)}',
                'city': city,
                'state': state,
                'address': self.fake.street_address().upper(),
                'zip': self.fake.zipcode(),
            }
        return self._locations[key]

    def pick_date(self, weekdays: tuple[float, ...]) -> date:
        """A date in the last year, weighted by weekday (rejection sampling)."""
        top = max(weekdays)
        span = (self.today - self.start).days
        while True:
            day = self.start + timedelta(days=self.rng.randint(0, span))
            if self.rng.random() * top <= weekdays[day.weekday()]:
                return day

    def pick_merchant(self, pool: Pool) -> Merchant:
        return self.rng.choices(pool.merchants, weights=[m.weight for m in pool.merchants])[0]

    def amount(self, merchant: Merchant) -> float:
        if merchant.low == merchant.high:
            return round(merchant.low, 2)
        # Most visits cluster toward the low end of the merchant's range.
        return round(merchant.low + (merchant.high - merchant.low) * self.rng.random() ** 1.6, 2)

    def chase_descriptor(self, merchant: Merchant, store: int) -> str:
        loc = self.location(merchant, store)
        text = f'{merchant.prefix}{merchant.name}'
        if loc['number'] and not merchant.prefix:
            text += f" {loc['number']}"
        if self.rng.random() < 0.3:
            text += f" {loc['city']} {loc['state']}"
        return text[:40]

    def amex_descriptor(self, merchant: Merchant, store: int) -> tuple[str, dict]:
        loc = self.location(merchant, store)
        name = f'{merchant.prefix}{merchant.name}'
        if loc['number'] and not merchant.prefix and self.rng.random() < 0.6:
            name += f" {loc['number']}"
        return f"{name[:24]:<24}{loc['phone']:>16}        {loc['state']}", loc

    # ----- issuers -----

    def chase(self, num_records: int) -> pd.DataFrame:
        records = []
        categories = list(CHASE_CATEGORIES)
        weights = [POOLS[CHASE_CATEGORIES[c]].weight + (POOLS[CHASE_EXTRA[c]].weight if c in CHASE_EXTRA else 0) for c in categories]
        payments = max(1, num_records // 25)
        for index in range(num_records):
            if index < payments:
                trans_date = self.pick_date(WEEKDAYS_EVEN)
                records.append(self._chase_record(trans_date, 'AUTOMATIC PAYMENT - THANK', '', 'Payment',
                                                  round(self.rng.uniform(500.0, 3000.0), 2)))
                continue
            category = self.rng.choices(categories, weights=weights)[0]
            pool_name = CHASE_CATEGORIES[category]
            if category in CHASE_EXTRA and self.rng.random() < 0.5:
                pool_name = CHASE_EXTRA[category]
            pool = POOLS[pool_name]
            merchant = self.pick_merchant(pool)
            store = self.rng.randrange(merchant.stores)
            trans_date = self.pick_date(pool.weekdays)
            txn_type = self.rng.choices(['Sale', 'Return'], weights=[0.96, 0.04])[0]
            amount = self.amount(merchant)
            # Native Chase CSVs list sales as negative values and returns as positive.
            amount = -amount if txn_type == 'Sale' else amount
            records.append(self._chase_record(trans_date, self.chase_descriptor(merchant, store), category, txn_type, amount))
        records.sort(key=lambda r: pd.Timestamp(r['Transaction Date']), reverse=True)
        return pd.DataFrame(records, columns=CHASE_COLUMNS)

    def _chase_record(self, trans_date: date, description: str, category: str, txn_type: str, amount: float) -> dict:
        post_date = trans_date + timedelta(days=self.rng.randint(0, 2))
        return {
            'Transaction Date': trans_date.strftime('%m/%d/%Y'),
            'Post Date': post_date.strftime('%m/%d/%Y'),
            'Description': description,
            'Category': category,  # Chase leaves the category blank on payments
            'Type': txn_type,
            'Amount': amount,
            'Memo': '',
        }

    def amex(self, num_records: int) -> pd.DataFrame:
        records = []
        categories = list(AMEX_CATEGORIES)
        weights = [POOLS[AMEX_CATEGORIES[c]].weight for c in categories]
        card_member = self.fake.name().upper()
        account = f'-{self.rng.randint(10000, 99999)}'
        for _ in range(num_records):
            category = self.rng.choices(categories, weights=weights)[0]
            pool = POOLS[AMEX_CATEGORIES[category]]
            merchant = self.pick_merchant(pool)
            store = self.rng.randrange(merchant.stores)
            trans_date = self.pick_date(pool.weekdays)
            description, loc = self.amex_descriptor(merchant, store)
            amount = self.amount(merchant)  # Amex purchases are positive; credits and payments negative.
            extended = f"{self.rng.randint(10000000000, 99999999999)} {loc['phone']}\n{merchant.name}\n{loc['phone']}\n{loc['state']}"
            address, city, state, zip_code = loc['address'], loc['city'], loc['state'], loc['zip']

            roll = self.rng.random()
            if roll < 0.04:
                amount = -round(self.rng.uniform(500.0, 3000.0), 2)
                description = extended = 'AUTOPAY PAYMENT - THANK YOU'
                category = address = city = state = zip_code = ''
            elif roll < 0.10:
                amount = -amount
                description = f'Platinum {merchant.name.title()} Credit'
                extended = f'{merchant.name}\n{description}'
                category = 'Fees & Adjustments-Fees & Adjustments'

            records.append({
                'Date': trans_date.strftime('%m/%d/%Y'),
                'Description': description,
                'Card Member': card_member,
                'Account #': account,
                'Amount': amount,
                'Extended Details': extended,
                'Appears On Your Statement As': description,
                'Address': address,
                'City/State': f'{city}\n{state}',
                'Zip Code': zip_code,
                'Country': 'UNITED STATES',
                'Reference': f"'{self.rng.randint(10**17, 10**18 - 1)}",
                'Category': category,
            })
        records.sort(key=lambda r: pd.Timestamp(r['Date']), reverse=True)
        return pd.DataFrame(records, columns=AMEX_COLUMNS)


def generate_chase_transactions(num_records, seed=None, today=None):
    """Mock Chase export (sales negative, returns and payments positive)."""
    return MockGenerator(seed, today).chase(num_records)


def generate_amex_transactions(num_records, seed=None, today=None):
    """Mock Amex export (purchases positive, credits and payments negative)."""
    return MockGenerator(seed, today).amex(num_records)


# About 50 combined card transactions per month across both issuers.
MOCK_RECORDS_PER_ISSUER = 300


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--records', type=int, default=MOCK_RECORDS_PER_ISSUER, help='rows per issuer')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducible output')
    args = parser.parse_args(argv)
    generator = MockGenerator(args.seed)
    generator.chase(args.records).to_csv('mock_chase_transactions.csv', index=False)
    generator.amex(args.records).to_csv('mock_amex_transactions.csv', index=False)


if __name__ == '__main__':
    main()
