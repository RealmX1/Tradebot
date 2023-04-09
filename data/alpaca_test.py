# api = tradeapi.REST('<APCA-API-KEY-ID>', '<APCA-API-SECRET-KEY>', api_version=
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import time
import copy

pd.set_option('display.max_rows', None)
api_key = "PKGNSI31E7XI9ACCSSVZ"
secret_key =  "yhupKUckY5vAbP7UOrkB26v4X4Gb9cdffo39V4OM"



### test alpaca-py ###
from datetime import datetime
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


    

def test_get_bars(symbol_or_symbols, timeframe, start, limit):
    # Test single symbol request

    stock_client = StockHistoricalDataClient("PKGNSI31E7XI9ACCSSVZ",  "yhupKUckY5vAbP7UOrkB26v4X4Gb9cdffo39V4OM")
    _start_in_url = start.isoformat("T") + "Z"
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol_or_symbols, timeframe=timeframe, start=start, limit=limit
    )

    print("Start request")
    bar_set = stock_client.get_stock_bars(request_params=request)
    print("End request")

    return bar_set



def complete_timestamp_with_all_symbols(df):
    # fill each timestamp with all symbols -- if no data, fill with -1
    start_time = time.time()
    print("start creating missing rows...")
    unique_symbols = df.index.get_level_values(0).unique().tolist()     # unique symbols
    unique_timestamps = df.index.get_level_values(1).unique().tolist()  # unique timestamps
    idx = pd.MultiIndex.from_product([unique_symbols, unique_timestamps])
    full = df.reindex(index=idx, fill_value=-1)
    print("missing rows added: ", full.shape[0] - df.shape[0])
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return full

# this function assumes that "full" has symbol as first index and timestamp as second index; and that each timestamp has all symbols
def infer_missing_data(full,symbol_num):
    # fill missing data with previous row (same symbol, previous timestamp)
    start_time = time.time()
    print("start filling missing data...")
    full['is_new_row'] = 0
    df_inferred = full.sort_index(level=1).copy()
    # note that after the sort, the original second index (timestamp) is now the first index
    for a, (i, row) in enumerate(df_inferred.iterrows()):
        if -1.0 in row.values:
            symbol, timestamp = i
            # mask = (df_inferred.index.get_level_values(0) == symbol) & (df_inferred.index.get_level_values(1) < timestamp)
            # prev_indexes = df_inferred.loc[mask].index
            if a < symbol_num:
                print("no previous index (with same symbol) exist for ", i)
                continue
            prev_row = df_inferred.iloc[a-symbol_num]
            new_row = [prev_row[3]]*4       # open, high, low, close = prev_close
            new_row += [0.0]*2                # volume & trade_count = 0
            new_row += [prev_row[6]]   # vwap = prev_vwap
            new_row += [1]                  # is_new_row = 1
            df_inferred.loc[i] = new_row
        if (a+1) % 1000 == 0:
            print(f"processed {a+1} rows")
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return df_inferred

def main():
    symbol = ["AAPL","MSFT","TSLA","GOOG"]
    symbol_num = len(symbol)
    timeframe = TimeFrame.Minute
    start = datetime(2022, 7, 27)
    limit = None

    start_time = time.time()
    print("Start getting bars")
    # bar_set = test_get_bars(symbol, timeframe, start, limit)
    with open('bar_set.pkl', 'rb') as f:
        bar_set = pickle.load(f)
    print(f'completed in {time.time()-start_time:.2f} seconds')

    df = bar_set.df
    print("raw data shape: ", df.shape)

    # with open('bar_set.pkl', 'wb') as f:
    #     pickle.dump(bar_set, f)
    

    full = complete_timestamp_with_all_symbols(df)

    df_inferred = infer_missing_data(full,symbol_num)
        
    # start_time = time.time()
    # print("start saving to csv...")
    # df_inferred.to_csv('df_inferred.csv', index=True)
    # print(f'completed in {time.time()-start_time:.2f} seconds')

    # start_time = time.time()
    # print("start reading df_inferred.csv...")
    # df = pd.read_csv('df_inferred.csv', index_col=[0,1])
    # print(f'completed in {time.time()-start_time:.2f} seconds')


    # Remove some rows from the beginning of the dataframe;
    # These rows are removed to clean "TIME VOID" that is generated at start of the dataset during the filling of absent time.
    # print(df.head(50)) 
    # this number might need to be adjusted for another dataset
    df = df.drop(df.index[:44])
    print(df.head(20)) 
    df.to_csv('tmp.csv', index=True)

if __name__ == "__main__":
    main()



### test alpaca alpaca-trade-api ###