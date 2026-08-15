import pandas as pd

df = pd.read_csv('data/raw/train.csv')

# find non-finance rows containing money words (the hard negatives)
money = df['text'].str.contains('payment|paycheck|payroll|salary|invoice|billing',
                                 case=False, na=False)
hard_negatives = df[money & (df['category'] != 'finance')]
print("hard negatives found:", len(hard_negatives))

# duplicate them 4x and add back
boosted = pd.concat([df] + [hard_negatives] * 4, ignore_index=True)
boosted = boosted.sample(frac=1, random_state=42)  # shuffle

print(boosted['category'].value_counts())
boosted.to_csv('data/raw/train_boosted.csv', index=False)
print("saved: data/raw/train_boosted.csv")
