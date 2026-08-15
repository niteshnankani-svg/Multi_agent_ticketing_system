import pandas as pd

# the original file still has subject/body/answer separately
orig = pd.read_csv('data/raw/tickets_en.csv')
orig['text'] = (orig['subject'].fillna('') + '. ' + orig['body'].fillna('')).str.strip()
orig_lookup = orig[['text', 'answer']].dropna(subset=['answer'])
orig_lookup = orig_lookup.drop_duplicates(subset=['text'])

# our current train/val/test only have text + category, no answer
train = pd.read_csv('data/raw/train.csv')

merged = train.merge(orig_lookup, on='text', how='left')

print('train rows:', len(train))
print('rows with a recovered answer:', merged['answer'].notna().sum())
print()
print(merged[merged['answer'].notna()][['category']].value_counts())

merged.to_csv('data/raw/train_with_answers.csv', index=False)
print('\nsaved: data/raw/train_with_answers.csv')
