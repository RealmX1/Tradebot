import pandas as pd

df = pd.read_csv('fred_series.csv')

# print the name column, one by one
for name in df['name']:
    print(name)

