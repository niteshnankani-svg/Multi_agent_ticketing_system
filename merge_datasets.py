import pandas as pd

# ---- file 1: bueck tickets ----
df1 = pd.read_csv('data/raw/tickets_categorized.csv')
df1 = df1[['text', 'category']].copy()
df1['source'] = 'bueck'

# ---- file 2: IT service tickets ----
df2 = pd.read_csv('data/raw/it_service.csv')

IT_MAP = {
    'HR Support': 'internal',
    'Hardware': 'backend',
    'Access': 'backend',
    'Storage': 'backend',
    'Administrative rights': 'backend',
    'Purchase': 'finance',
    'Internal Project': 'general',
    'Miscellaneous': 'general',
}

df2['category'] = df2['Topic_group'].map(IT_MAP)
df2['text'] = df2['Document']
df2 = df2[['text', 'category']].copy()
df2['source'] = 'it_service'

# ---- file 3: banking77, all finance ----
df3 = pd.read_csv('data/raw/banking77_train.csv')
df3['category'] = 'finance'
df3 = df3[['text', 'category']].copy()
df3['source'] = 'banking77'

# ---- combine everything ----
combined = pd.concat([df1, df2, df3], ignore_index=True)
combined = combined[combined['text'].str.len() > 10]

print('=== combined totals ===')
print(combined['category'].value_counts())
print()
print('=== by source ===')
print(combined.groupby(['source', 'category']).size())
print()
print('total rows:', len(combined))

combined.to_csv('data/raw/all_categorized.csv', index=False)
print("saved: data/raw/all_categorized.csv")
