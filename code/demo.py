import pandas as pd

df = pd.read_csv('../data/weread_books_cleaned.csv')

data = df['recommendation']
print(data.describe())
print(data.isnull().sum())
