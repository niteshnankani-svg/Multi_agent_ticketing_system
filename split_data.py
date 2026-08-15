import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw/all_categorized.csv')

train, temp = train_test_split(
    df, test_size=0.30, stratify=df['category'], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp['category'], random_state=42
)

train.to_csv('data/raw/train.csv', index=False)
val.to_csv('data/raw/val.csv', index=False)
test.to_csv('data/raw/test.csv', index=False)

print('train:', len(train), dict(train['category'].value_counts()))
print('val:  ', len(val), dict(val['category'].value_counts()))
print('test: ', len(test), dict(test['category'].value_counts()))
