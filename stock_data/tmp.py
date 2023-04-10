import pandas as pd

# # sample data
# df = pd.DataFrame({
#     'symbol': ['A', 'A', 'B', 'B'],
#     'timestamp': ['2022-01-01 09:30:00', '2022-01-01 10:00:00', '2022-01-01 09:30:00', '2022-01-01 10:00:00'],
#     'value1': [1, 2, 3, 4],
#     'value2': [5, 6, 7, 8]
# })

# print(df)

# # pivot table
# pt = pd.pivot_table(df, index='timestamp', columns='symbol', values=['value1', 'value2'])

# # flatten the multi-level column index
# pt.columns = [f'{col[1]}_{col[0]}' for col in pt.columns]

# print(pt)

from sklearn.preprocessing import StandardScaler



# df = pd.read_csv('data/csv/bar_set_20230403_20230404_test.csv', index_col = ['symbols', 'timestamps'])
# print(df.head(10))
# df_head = df.head(10)

# # Instantiate a StandardScaler object
# scaler = StandardScaler()

# # Standardize the DataFrame
# df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# # print(scaler)
# # print(df_standardized.head(10))
# df_head2 = df_standardized.head(10)
# df_head_standardized = pd.DataFrame(scaler.fit_transform(df_head), columns=df_head.columns)
# df_head_standardized2 = pd.DataFrame(scaler.fit_transform(df_head2), columns=df_head2.columns)
# print(df_head_standardized)
# print(df_head_standardized2)