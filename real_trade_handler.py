# read trade history csv and split the data into gorups by minute mark

import pandas as pd
import time


trade_data_pth = 'data/csv/trade_set_20200101_20200201_raw.csv'
trade_df = pd.read_csv(trade_data_pth, index_col = ['symbol'])

bar_data_pth = 'data/csv/bar_set_20200101_20200201_MSFT_16feature0.csv'
bar_df = pd.read_csv(bar_data_pth, index_col = ['symbol'])

# # convert the timestamp column to datetime objects
# trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
# bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])

# # round the timestamp to the nearest minute
# trade_df['rounded_timestamp'] = trade_df['timestamp'].dt.round('1min')
# bar_df['rounded_timestamp'] = bar_df['timestamp'].dt.round('1min')

# set the index to the new rounded timestamp column
trade_df.set_index('rounded_timestamp', inplace=True)
bar_df.set_index('rounded_timestamp', inplace=True)

# iterate over the rows of the bar_df dataframe
start_time = time.time()
ma_time = 0

for index, bar_row in bar_df.iterrows():
    pass
    
    # print('bar row: ', bar_row['timestamp'])
    # # get the rounded timestamp value
    # rounded_timestamp = index
    
    # # filter the corresponding rows from the trade_df dataframe
    # trade_rows = trade_df[trade_df.index == rounded_timestamp]
    # for trade_row in trade_rows.iterrows():
    #     pass
    #     # print('trade row: ', trade_row[0])
    
    # # perform the necessary operations on the trade_rows
    # # ...
    # # print('time elapsed: ', time.time() - start_time)
    # ma_time = ma_time * 0.9 + (time.time() - start_time) * 0.1
    # row_per_sec = 1 / (ma_time)
    # print('row per sec: ', row_per_sec)
    # start_time = time.time()