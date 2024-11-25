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
API_KEY = "AKOZFEX5F94X2SD7HQOQ"
SECRET_KEY =  "3aNqjtbPlkJv09NicPgYFXC3KUhNOR16JGGdiLet"
data_source = DataFeed.IEX

stock_client = StockHistoricalDataClient(API_KEY,  SECRET_KEY)

pre = "bar_set"
post = "raw"
    

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
    df_concat = pd.pivot_table(df, index='timestamp', columns='symbol')
    print(df_concat.head(10))
    df_concat.columns = [f"{col[1]}_{col[0]}" for col in df_concat.columns.values] # Flatten multi-level column names
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return df_concat

# saves pkl to 
def get_and_process_bars(symbols, timeframe, start, end, limit = None, download=False, pre = pre, post = pre):
    symbol_num = len(symbols)
    start_time = time.time()
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    time_str = f'{start_str}_{end_str}'
    pkl_path = f'data/pkl/{pre}_{time_str}_{post}.pkl'
    csv_path = f'data/csv/{pre}_{time_str}_{post}.csv'

    if download:
        print("Start getting bars")
        bar_set = get_bars(symbols, timeframe, start, end, limit)

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

    start_time = time.time()
    print("start saving to csv...")
    df.to_csv(csv_path, index=True, index_label=['symbol', 'timestamp']) 
    # note that the index_label is necessary; if not specified, the index name will not be saved
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return df

    # Remove some rows from the beginning of the dataframe;
    # These rows are removed to clean "TIME VOID" that is generated at start of the dataset during the filling of absent time.
    # start_time = time.time()
    # print("start concatinating symbols...")
    # df = pd.read_csv(csv_path, index_col = ['symbol', 'timestamp'])
    # print(df.head(1000))
    # print(f'completed in {time.time()-start_time:.2f} seconds')

    # !!! If more features need to be calculated and added, do it here !!! After concatinating the symbols, it will be much harder.
    # !!! noramlization probably should be done here as well

    # turn 2-level index into 1 level index -- keep timestamp index, and concatenate each row for symbols
    # df = pd.read_csv('df_done.csv', index_col = ['symbol', 'timestamp'])
    # df_concat = concat_symbols(df)

    # print(df_concat.head(10))
    
    # df_concat.to_csv('concatinated_APPL_4symbol_20220201.csv', index=True, index_label=['timestamp'])

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXassumes that df_bars is in the order of timestamp, symbol
# UNTESTED
def read_raw_bars(time_strs):
    start_time = time.time()
    print("Reading multiple processed csv...")
    dfs = []
    for time_str in time_strs:
        csv_path = f'data/csv/bar_set_{time_str}_raw.csv'
        df = pd.read_csv(csv_path, index_col = ['symbol', 'timestamp'])
        dfs.append(df)
    print(f'completed in {time.time()-start_time:.2f} seconds')
    return dfs
# UNTESTED
def combine_bars(df_bars, symbol_num):
    start_time = time.time()
    print("Combining multiple df...")
    combined_df = pd.concat(df_bars)
    # Remove rows where both indices are the same
    combined_df[~combined_df.index.duplicated()]

    # combined_df.drop_duplicates(subset, keep='last', inplace=True)
    combined_df = complete_timestamp_with_all_symbols(combined_df)
    symbol_num = combined_df.index.get_level_values(0).nunique() # number of unique first index
    combined_df = infer_missing_data(combined_df,symbol_num)
    # the "==" operation creates a boolean mask that selects redundent rows, and 
    print(f'completed in {time.time()-start_time:.2f} seconds')
    return combined_df

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

def last_week_bars(symbols, timeframe = TimeFrame.Minute):
    # get the last week of data
    end = datetime.now()
    start = (end - timedelta(days=end.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    df = get_and_process_bars(symbols, timeframe, start, end, download = True, pre = 'last_week_' + pre)
    return df
    # return get_and_process_bars(symbols, timeframe, start, end, None)

# UNTESTED
def get_load_of_bars(symbols, timeframe, start, end, limit = None, download = False):
    raw_start = start
    raw_end = end
    
    limit = None
    time_strs = []

    start_time = time.time()
    print("Start getting multiple bars file")
    
    for i in range(raw_start.year, raw_end.year + 1):
        if (i == raw_start.year):
            start = raw_start
        else:
            start = datetime(i,1,1)
        if (i == raw_end.year):
            end = raw_end
        else:
            end = datetime(i+1,1,1)

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        time_str = f"{start_str}_{end_str}"
        print('getting data with time_str: ', time_str)
        time_strs.append(time_str)
        get_and_process_bars(symbols, timeframe, start, end, limit, download=download)
    print(f'All files downloaded and processed in {time.time()-start_time:.2f} seconds')

    print("time_strs: ", time_strs)

    # combine multiple csv files into one
    dfs = read_raw_bars(time_strs)

    df = combine_bars(dfs, len(symbols))

    start_time = time.time()
    print("start saving to csv...")
    start_str = raw_start.strftime("%Y%m%d")
    end_str = raw_end.strftime("%Y%m%d")
    df.to_csv(f'data/csv/bar_set_huge_{start_str}_{end_str}_raw.csv', index=True, index_label=['symbol', 'timestamp'])
    print(f'completed in {time.time()-start_time:.2f} seconds')


def main():
    # barset = get_latest_bars("BABA")
    # print(barset)
    symbols = ["BABA"] #["AAPL","MSFT","TSLA","GOOG","SPY"]
    timeframe = TimeFrame.Minute
    start = datetime(2022,1,1)
    end = datetime.now() 

    # get_load_of_bars(symbols, timeframe, start, end, limit = None, download=False)

    last_week_bars(symbols, timeframe = TimeFrame.Minute)

    
    


    # need to re-download prices adjusted for splits; experimetn with the parameter
    # https://forum.alpaca.markets/t/getbars-adjustment/5056

    





if __name__ == "__main__":
    main()



### test alpaca alpaca-trade-api ###