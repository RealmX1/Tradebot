import pandas as pd
import numpy as np


API_KEY = "AKOZFEX5F94X2SD7HQOQ"
SECRET_KEY =  "3aNqjtbPlkJv09NicPgYFXC3KUhNOR16JGGdiLet"
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



# df = pd.read_csv('data/csv/bar_set_20230401_20230403_test.csv', index_col = ['symbols'])
# print(df['timestamps'])
# print(df['timestamps'][0])
# print(type(df['timestamps'][0]))
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
# names = ['AAPL','GOOG','MSFT','SPY','TSLA']
# means = np.zeros((6,))
# stds = np.zeros((6,))
# for name in names:
#     df = pd.read_csv(f'data/csv/bar_set_huge_20180101_20230410_{name}_indicator.csv', index_col = ['symbols', 'timestamps'])
#     last_three_cols = df.iloc[:, -6:]
#     last_three_cols_mean = last_three_cols.mean()
#     means += last_three_cols_mean
#     last_three_cols_std = last_three_cols.std()
#     stds += last_three_cols_std

# means /= len(names)
# stds /= len(names)

# print("means: ", means)
# '''
# ADX_14    30.171159
# DMP_14    32.843816
# DMN_14    32.276572
# '''
# print("stds: ", stds)
# '''
# ADX_14    16.460923
# DMP_14    18.971341
# DMN_14    18.399032
# '''

# df = pd.read_csv('data/csv/bar_set_test_AAPL_indicator.csv', index_col = ['symbols', 'timestamps'])
# import pandas as pd
# import pytz
# from dateutil.parser import parse

# # First, make sure the index is timezone aware (in UTC)
# df['timestamps_col'] = pd.to_datetime(df.index.get_level_values(1))
# # df['timestamps_col'] = pd.to_datetime(df['timestamps_col'], utc=True)
# # print(df.head(5))
# # print(df.index.get_level_values(1))
# # Set the timestamps_col column as the index of the DataFrame

# # Convert the index to Eastern timezone
# df['edt_time'] = df['timestamps_col'].dt.tz_convert('US/Eastern')
# # print(df.head(5))

# # Extract the time of day (in hours) as a new column
# df['edt_hour'] = df['edt_time'].dt.hour + df['edt_time'].dt.minute / 60
# # print(df.head(5))

# # Create a new column with the time of day scaled from 0.0 (9:30 am) to 1.0 (1:00 pm)
# start_hour, end_hour = 9.5, 16.0
# df['edt_scaled'] = (df['edt_hour'] - start_hour) / (end_hour - start_hour)
# df.drop(columns=['timestamps_col', 'edt_time', 'edt_hour'], inplace=True)
# df['is_core_time'] = ((df['edt_scaled'] >= 0) & (df['edt_scaled'] <= 1)).astype(int)
# print(df.head(5))

# df.to_csv('data/csv/bar_set_test_AAPL_indicator_edt.csv')


##############################################
import scipy as sp
import pickle
import time
import copy


from datetime import datetime, timedelta
from typing import Dict

from alpaca.data import Trade, Snapshot, Quote, Bar
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
    StockLatestBarRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Exchange, DataFeed
from alpaca.data.models import BarSet, QuoteSet, TradeSet
import pytz

stock_client = StockHistoricalDataClient(API_KEY,  SECRET_KEY)

def get_bars(symbol_or_symbols, timeframe, start, end, limit):
    # Test single symbol request
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol_or_symbols, timeframe=timeframe, start=start, end=end, limit=limit, adjustment="all", feed = DataFeed.IEX
    )

    print("Start request")
    bar_set = stock_client.get_stock_bars(request_params=request)
    print("End request")

    return bar_set

symbol = 'BABA'
timeframe = TimeFrame.Minute
start = datetime(2022, 4, 13)
end = None
limit = None

bar_set = get_bars(symbol, timeframe, start, end, limit)
df = bar_set.df

print(df.index.levels[1].dtype)
df.to_csv('data/csv/today.csv')
df = pd.read_csv('data/csv/today.csv')
print(df)
edt = pytz.timezone('US/Eastern')
now = datetime.now(edt)
print(type(now))
print(now)