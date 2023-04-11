import pandas as pd

df = pd.read_csv('../data/csv/bar_set_20230101_20230410.csv', index_col = ['symbols', 'timestamps'])