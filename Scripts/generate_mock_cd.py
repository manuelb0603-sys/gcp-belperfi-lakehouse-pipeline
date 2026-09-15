import pandas as pd
from faker import Faker 
import random
from datetime import timedelta

fake = Faker()

def generate_chase_transactions(num_records):
    records = []
    
    # Valid Chase categories that directly map to your dim_category.sqlx logic
    categories = [
        'Groceries', 'Shopping', 'Automotive', 'Health & Wellness', 
        'Food & Drink', 'Travel', 'Bills & Utilities', 
        'Gifts & Donations', 'Personal', 'Payment'
    ]
    
    for _ in range(num_records):
        # Generate dates with the Post Date trailing 0-2 days behind Transaction Date
        trans_date = fake.date_between(start_date='-1y', end_date='today')
        post_date = trans_date + timedelta(days=random.randint(0, 2))
        
        category = random.choice(categories)
        
        # Replicate Chase's Type and Amount logic (Sales are negative, Returns/Payments are positive)
        if category == 'Payment':
            txn_type = 'Payment'
            amount = round(random.uniform(500.0, 3000.0), 2)
            description = 'AUTOMATIC PAYMENT - THANK'
            category = '' # Chase often leaves the category blank on payments
        else:
            txn_type = random.choices(['Sale', 'Return'], weights=[0.95, 0.05])[0]
            amount = round(random.uniform(1.0, 500.0), 2)
            
            if txn_type == 'Sale':
                amount = -amount # Native Chase CSVs list sales as negative values
                
            description = fake.company().upper() # Uppercase to mimic merchant terminals

        records.append({
            'Transaction Date': trans_date.strftime('%m/%d/%Y'),
            'Post Date': post_date.strftime('%m/%d/%Y'),
            'Description': description,
            'Category': category,
            'Type': txn_type,
            'Amount': amount,
            'Memo': ''
        })
        
    return pd.DataFrame(records)


def generate_amex_transactions(num_records):
    """Generate random Amex transactions with Faker-generated details."""
    profiles = [
        ('Merchandise & Supplies-Internet Purchase', 15, 300),
        ('Merchandise & Supplies-General Merchandise', 10, 250),
        ('Groceries', 15, 350),
        ('Restaurant-Bar & Cafe', 5, 75),
        ('Restaurant-Restaurants', 8, 100),
        ('Business Services-Office Supplies', 8, 125),
        ('Communications-Mobile Telecom', 25, 125),
        ('Transportation-Taxis & Coach', 10, 100),
        ('Transportation-Airlines', 80, 650),
        ('Automotive-Fuel', 25, 100),
        ('Merchandise & Supplies-Home Improvement', 20, 400),
        ('Health & Wellness', 8, 150),
        ('Merchandise & Supplies-Sporting Goods', 25, 250),
    ]
    columns = [
        'Date', 'Description', 'Card Member', 'Account #', 'Amount',
        'Extended Details', 'Appears On Your Statement As', 'Address',
        'City/State', 'Zip Code', 'Country', 'Reference', 'Category',
    ]
    records = []

    for _ in range(num_records):
        trans_date = fake.date_between(start_date='-1y', end_date='today')
        category, minimum, maximum = random.choice(profiles)
        merchant = fake.company().upper()
        amount = round(random.uniform(minimum, maximum), 2)
        address = fake.street_address().upper()
        city = fake.city().upper()
        state = fake.state_abbr()
        zip_code = fake.zipcode()
        phone = f'{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}'
        description = f'{merchant:<24}{phone:>16}        {state}'
        extended_details = (
            f'{random.randint(10000000000, 99999999999)} {phone}\n'
            f'{merchant}\n{phone}\n{state}'
        )

        # Amex purchases are positive; credits and payments are negative.
        payment_roll = random.random()
        if payment_roll < 0.04:
            amount = -round(random.uniform(500.0, 3000.0), 2)
            description = 'AUTOPAY PAYMENT - THANK YOU'
            extended_details = description
            category = ''
            address = ''
            city = ''
            state = ''
            zip_code = ''
        elif payment_roll < 0.12:
            amount = -amount
            description = f'Platinum {merchant.title()} Credit'
            extended_details = f'{merchant}\n{description}'
            category = 'Fees & Adjustments-Fees & Adjustments'

        records.append({
            'Date': trans_date.strftime('%m/%d/%Y'),
            'Description': description,
            'Card Member': fake.name().upper(),
            'Account #': f'-{random.randint(10000, 99999)}',
            'Amount': amount,
            'Extended Details': extended_details,
            'Appears On Your Statement As': description,
            'Address': address,
            'City/State': f'{city}\n{state}',
            'Zip Code': zip_code,
            'Country': 'UNITED STATES',
            'Reference': f"'{random.randint(10**17, 10**18 - 1)}",
            'Category': category,
        })

    return pd.DataFrame(records, columns=columns)

# Generate approximately 50 combined card transactions per month across both issuers.
MOCK_RECORDS_PER_ISSUER = 300

chase_df = generate_chase_transactions(MOCK_RECORDS_PER_ISSUER)
chase_df.to_csv('mock_chase_transactions.csv', index=False)
amex_df = generate_amex_transactions(MOCK_RECORDS_PER_ISSUER)
amex_df.to_csv('mock_amex_transactions.csv', index=False)