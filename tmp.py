# import os

# from AI.model_structure_param import *

# time_str = '20200101_20200701'
# symbol = 'MSFT'
# data_type = '16feature0'
# data_name = f'bar_set_{time_str}_{symbol}_{data_type}_RAW'
# model_name = f'last_model_{config_name}'


# log_pth_template = f'../TradebotGraph/{data_name}--{model_name}_{{}}.txt'
# pic_pth_template = f'../TradebotGraph/{data_name}--{model_name}_{{i_th_attempt}}_{{block_col_name}}.png'

# i = 0
# while True:
#     i += 1
#     pic_pth_template_2 = pic_pth_template.format(i_th_attempt = i, block_col_name = '{}')

#     if not os.path.exists(complete_log_pth := log_pth_template.format(i)): break
# print(i)
# print(complete_log_pth)
# print(pic_pth_template_2)
# print(pic_pth_template_2.format('N/A'))

import os
import csv
import datetime

import os
import csv
import datetime

# Define the CSV file path
csv_file_path = "log/back_test_log.csv"

# Define the header (column names)
header = ['test_time', 'model_pth', 'data_path', 'blocked_col', 'symbol', 'account_value', 'account_growth', 'stock_growth', 'pct_growth_diff', 'interval_per_trade', 'long_count', 'profitable_long_count', 'mean_long_profit_pct', 'occupancy_rate']

# Define a sample row as a dictionary
row_dict = {
    'test_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_pth': 'sample_model_path',
    'data_path': 'sample_data_path',
    'symbol': 'MSFT',
    'account_value': 1000,
    'account_growth': 10,
    'stock_growth': 5,
    'pct_growth_diff': 5,
    'interval_per_trade': 2,
    'long_count': 10,
    'profitable_long_count': 5,
    'mean_long_profit_pct': 1,
    'occupancy_rate': 0.5
}

# Check if the CSV file exists
# if not os.path.exists(csv_file_path):
#     # If not, create the file with the header
#     with open(csv_file_path, 'w', newline='') as csvfile:
#         csv_writer = csv.DictWriter(csvfile, fieldnames=header)
#         csv_writer.writeheader()

# # Append the new row to the CSV file
# with open(csv_file_path, 'a', newline='') as csvfile:
#     csv_writer = csv.DictWriter(csvfile, fieldnames=header)
#     csv_writer.writerow(row_dict)

import pandas as pd

# Create a sample DataFrame
data = {'close': [110, 120, 130, 140, 150],
        'high': [115, 125, 135, 145, 155],
        'low': [105, 115, 125, 135, 145]}
df = pd.DataFrame(data)

# Shift 'high' and 'low' columns up one row
df['shifted_high'] = df['high'].shift(-1)
df['shifted_low'] = df['low'].shift(-1)
print(df)

# Create 'fall' and 'rise' columns based on conditions
df['fall'] = df['shifted_high'] < df['close']
df['rise'] = df['shifted_low'] > df['close']

# Drop the temporary 'shifted_high' and 'shifted_low' columns
df.drop(['shifted_high', 'shifted_low'], axis=1, inplace=True)

print(df)