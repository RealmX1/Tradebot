import pandas as pd

# Read the CSV file
data_path = "cdl_test.csv"
df = pd.read_csv(data_path, index_col = ['symbol', 'timestamp'])

# Sort the dataframe by the second index
df_sorted = df.sort_index(level=1)

# Split the dataframe into two parts
percentage = 0.5
n_rows = len(df_sorted)
split_row = int(percentage * n_rows)

df_1 = df_sorted.iloc[:split_row, :]
print(df_1.head(5))
df_2 = df_sorted.iloc[split_row:, :]
print(df_2.head(5))

# Save the two parts into separate CSV files
df_1.to_csv('cdl_test_1.csv', index=True, index_label=['symbol', 'timestamp'])
df_2.to_csv('cdl_test_2.csv', index=True, index_label=['symbol', 'timestamp'])