import pandas as pd
import numpy as np


API_KEY = "AKOZFEX5F94X2SD7HQOQ"
SECRET_KEY =  "3aNqjtbPlkJv09NicPgYFXC3KUhNOR16JGGdiLet"


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


import ast

# Your string representation of a list
list_str = "['close', 'vwap', 'EMA_14', 'DEMA_10', 'TEMA_10', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0', 'RSI_14', 'CCI_14_0.015', 'ADX_14', 'edt_dayofweek', 'edt_scaled', 'is_core_time']"

# Convert it to a list
result_list = ast.literal_eval(list_str)

print(result_list)

# stock_client = StockHistoricalDataClient(API_KEY,  SECRET_KEY)

# def get_bars(symbol_or_symbols, timeframe, start, end, limit):
#     # Test single symbol request
    
#     request = StockBarsRequest(
#         symbol_or_symbols=symbol_or_symbols, timeframe=timeframe, start=start, end=end, limit=limit, adjustment="all", feed = DataFeed.IEX
#     )

#     print("Start request")
#     bar_set = stock_client.get_stock_bars(request_params=request)
#     print("End request")

#     return bar_set

# symbol = 'BABA'
# timeframe = TimeFrame.Minute
# start = datetime(2022, 4, 13)
# end = None
# limit = None

# bar_set = get_bars(symbol, timeframe, start, end, limit)
# df = bar_set.df

# print(df.index.levels[1].dtype)
# df.to_csv('data/csv/today.csv')
# df = pd.read_csv('data/csv/today.csv')
# print(df)
# edt = pytz.timezone('US/Eastern')
# now = datetime.now(edt)
# print(type(now))
# print(now)



# import pandas_ta as ta
from alpaca.data.enums import Exchange, DataFeed
print(DataFeed.IEX.value)
print(DataFeed.SIP.value == 'iex')
# df = pd.read_csv('../data/csv/trade_set_20200101_20200201_raw.csv', index_col = ['symbol', 'timestamp'])
# print(df.columns.tolist())
# df = df.drop(['exchange', 'id', 'conditions', 'tape'], axis = 1)
# print(df.columns.tolist())
# df_sorted = df.sort_index(level='timestamp')
# df.to_csv('../data/csv/trade_set_20200101_20200201_raw.csv', index=True, index_label=['symbol', 'timestamp'])