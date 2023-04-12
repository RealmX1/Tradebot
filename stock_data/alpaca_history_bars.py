# api = tradeapi.REST('<APCA-API-KEY-ID>', '<APCA-API-SECRET-KEY>', api_version=
import pandas as pd
import numpy as np
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




pd.set_option('display.max_rows', None)
API_KEY = "PKGNSI31E7XI9ACCSSVZ"
SECRET_KEY =  "yhupKUckY5vAbP7UOrkB26v4X4Gb9cdffo39V4OM"
data_source = DataFeed.IEX

stock_client = StockHistoricalDataClient(API_KEY,  SECRET_KEY)
    

def get_bars(symbol_or_symbols, timeframe, start, end, limit):
    # Test single symbol request
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol_or_symbols, timeframe=timeframe, start=start, end=end, limit=limit, adjustment="all", feed = data_source
    )

    print("Start request")
    bar_set = stock_client.get_stock_bars(request_params=request)
    print("End request")

    return bar_set

def get_quotes(symbol_or_symbols, start, limit):
    pass



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
    print("full shape: ", full.shape)
    # full['is_new_row'] = 0 # for debugging purpose only
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
            # new_row += [1]                  # is_new_row = 1
            df_inferred.loc[i] = new_row
        if (a+1) % 10000 == 0:
            print(f"processed {a+1} rows")
    print(f'completed in {time.time()-start_time:.2f} seconds')

    # df_inferred = df_inferred.drop(columns=df.columns[-1]) # drop the last column (is_new_row)

    return df_inferred

def concat_symbols(df):
    start_time = time.time()
    print("Concatenating each row for symbols")
    df_concat = pd.pivot_table(df, index='timestamps', columns='symbols')
    print(df_concat.head(10))
    df_concat.columns = [f"{col[1]}_{col[0]}" for col in df_concat.columns.values] # Flatten multi-level column names
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return df_concat

# saves pkl to 
def get_and_process_bars(symbol, timeframe, start, end, limit, time_str, download=False):
    symbol_num = len(symbol)
    start_time = time.time()
    pkl_path = f'data/pkl/bar_set_{time_str}_baba.pkl'
    csv_path = f'data/csv/bar_set_{time_str}_baba.csv'

    if download:
        print("Start getting bars")
        bar_set = get_bars(symbol, timeframe, start, end, limit)

        with open(pkl_path, 'wb') as f:
            pickle.dump(bar_set, f)
        print(f'completed in {time.time()-start_time:.2f} seconds')
    else:
        start_time = time.time()
        print("Start loading bars")
        with open(pkl_path, 'rb') as f:
            bar_set = pickle.load(f)
        print(f'completed in {time.time()-start_time:.2f} seconds')
        
    df = bar_set.df
    
    print("raw data shape: ", df.shape)
    df = complete_timestamp_with_all_symbols(df)
    df = infer_missing_data(df,symbol_num)
    # print(df)

    start_time = time.time()
    print("start saving to csv...")
    df.to_csv(csv_path, index=True, index_label=['symbols', 'timestamps']) 
    # note that the index_label is necessary; if not specified, the index name will not be saved
    print(f'completed in {time.time()-start_time:.2f} seconds')

    # Remove some rows from the beginning of the dataframe;
    # These rows are removed to clean "TIME VOID" that is generated at start of the dataset during the filling of absent time.
    # start_time = time.time()
    # print("start concatinating symbols...")
    # df = pd.read_csv(csv_path, index_col = ['symbols', 'timestamps'])
    # print(df.head(1000))
    # print(f'completed in {time.time()-start_time:.2f} seconds')

    # !!! If more features need to be calculated and added, do it here !!! After concatinating the symbols, it will be much harder.
    # !!! noramlization probably should be done here as well

    # turn 2-level index into 1 level index -- keep timestamp index, and concatenate each row for symbols
    # df = pd.read_csv('df_done.csv', index_col = ['symbols', 'timestamps'])
    # df_concat = concat_symbols(df)

    # print(df_concat.head(10))
    
    # df_concat.to_csv('concatinated_APPL_4symbol_20220201.csv', index=True, index_label=['timestamps'])

# assumes taht df_bars is in the order of timestamp, symbol
def combine_bars(df_bars):
    start_time = time.time()
    print("Combining multiple df...")
    concatenated_df = pd.concat(df_bars)
    # Remove rows where both indices are the same
    filtered_df = concatenated_df[~(concatenated_df.index.get_level_values(0) == concatenated_df.index.get_level_values(1))]
    # the "==" operation creates a boolean mask that selects redundent rows, and 
    print(f'completed in {time.time()-start_time:.2f} seconds')
    return filtered_df

def get_and_process_quotes(symbol, timeframe, start, limit):
    pass

def caluclate_indicators():
    pass

def add_symbol():
    pass

def get_latest_bars(symbol_or_symbols):
    request = StockLatestBarRequest(
        symbol_or_symbols=symbol_or_symbols, feed = data_source.IEX
    )

    print("Start request")
    bar_set = stock_client.get_stock_latest_bar(request_params=request)
    print("End request")

    return bar_set

def main():
    # barset = get_latest_bars("BABA")
    # print(barset)
    symbol = ["BABA"] #["AAPL","MSFT","TSLA","GOOG","SPY"]
    symbol_num = len(symbol)
    timeframe = TimeFrame.Minute
    start = datetime(2023,1,1)
    end = None #datetime(2023,4,12)
    limit = None

    # now we are using adjusted dataset.
    time_str = "20230101_20230412" # "20200101_20210101" # "20230403_20230404_test" "20210101_20220727" "20220727_20230406" 
    get_and_process_bars(symbol, timeframe, start, end, limit, time_str, download=False)

    start_time = time.time()
    print("Start getting multiple bars file")
    df_strs = []
    for i in range(20, 23):
        start = datetime(2000+i,1,1)
        end = datetime(2000+i+1,1,1)
        time_str = f"20{i}0101_20{i+1}0101"
        df_strs.append(time_str)
        get_and_process_bars(symbol, timeframe, start, end, limit, time_str, download=False)
    print(f'All files downloaded and processed in {time.time()-start_time:.2f} seconds')

    df_strs.append("20230101_20230412")
    print(df_strs)
    

    # combine multiple csv files into one
    start_time = time.time()
    print("Reading multiple processed csv...")
    dfs = []
    for str in df_strs:
        csv_path = f'data/csv/bar_set_{str}_baba.csv'
        df = pd.read_csv(csv_path, index_col = ['symbols', 'timestamps'])
        dfs.append(df)
    print(f'completed in {time.time()-start_time:.2f} seconds')
    df = combine_bars(dfs)

    df = infer_missing_data(df,symbol_num)

    start_time = time.time()
    print("start saving to csv...")
    df.to_csv('data/csv/bar_set_huge_20180101_20230412.csv', index=True, index_label=['symbols', 'timestamps'])
    print(f'completed in {time.time()-start_time:.2f} seconds')

    # df = pd.read_csv('data/csv/bar_set_huge_20180101_20230410.csv', index_col = ['symbols', 'timestamps'])
    # df = df.drop(df.index[:144])
    # print(df.head(12))


    # need to re-download prices adjusted for splits; experimetn with the parameter
    # https://forum.alpaca.markets/t/getbars-adjustment/5056

    





if __name__ == "__main__":
    main()



### test alpaca alpaca-trade-api ###