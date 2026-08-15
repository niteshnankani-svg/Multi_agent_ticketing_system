import pandas as pd

df = pd.read_csv('data/raw/tickets_en.csv')

QUEUE_MAP = {
    "Billing and Payments": "finance",
    "Sales and Pre-Sales": "finance",
    "Returns and Exchanges": "finance",
    "Human Resources": "internal",
    "General Inquiry": "general",
}

BACKEND_TAGS = {"network", "outage", "incident", "disruption", "crash", "bug",
                 "software", "hardware", "maintenance", "security", "breach",
                 "virus", "encryption", "login", "recovery", "integration"}
FINANCE_TAGS = {"billing", "payment"}

def categorize(row):
    queue = row['queue']
    if queue in QUEUE_MAP:
        return QUEUE_MAP[queue]
    tags = {str(row.get(f'tag_{i}', '')).lower() for i in range(1, 9)}
    if tags & FINANCE_TAGS:
        return "finance"
    if tags & BACKEND_TAGS:
        return "backend"
    return "general"

df['category'] = df.apply(categorize, axis=1)

# fill missing subject/body with empty string, not the text "nan"
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['text'] = (df['subject'] + '. ' + df['body']).str.strip()

# drop rows where we ended up with basically nothing to read
df = df[df['text'].str.len() > 10]

print(df['category'].value_counts())
print('rows after cleaning:', len(df))

df.to_csv('data/raw/tickets_categorized.csv', index=False)
print("saved: data/raw/tickets_categorized.csv")
