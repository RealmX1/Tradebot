# api = tradeapi.REST('<APCA-API-KEY-ID>', '<APCA-API-SECRET-KEY>', api_version=
import pandas as pd
import numpy as np
import scipy as sp
import time
# import copy
# import threading
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# from string import Template

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

# Read the API key and secret from environment variables and store them in variables.
def parse_keys(filename):
    keys = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                keys[key.strip()] = value.strip()
    return keys

keys = parse_keys("alpaca_api/_alpaca_api_key.key")

API_KEY = keys.get('API_KEY')
SECRET_KEY = keys.get('SECRET_KEY')
#############################################




pd.set_option('display.max_rows', None)
data_source = DataFeed.IEX # DataFeed. is default/free; SIP & OTC are available through paid subscription
data_source_str = str(data_source)[-3:]

stock_client = StockHistoricalDataClient(API_KEY,  SECRET_KEY)

# pre = 'bar_set'
post = 'raw'
default_data_pth = 'data' #currently, launch by default from the project main folder

save_template = "{data_pth}/{data_folder}/{pre}_{time_str}_{symbol_str}_{timeframe_str}_{post}_{data_source_str}.csv"


def get_bars(symbol_lst, timeframe, start, end, limit, adjustment = 'all'):    
    request = StockBarsRequest(
        symbol_or_symbols=symbol_lst, 
        timeframe=timeframe, 
        start=start, 
        end=end, 
        limit=limit, 
        adjustment=adjustment, 
        feed = data_source
    )
    
    print('Start request')
    bar_set = stock_client.get_stock_bars(request_params=request)
    print('End request')

    return bar_set

def get_trades(symbol_or_symbol_lst, timeframe, start, end, limit, adjustment = None):    
    request = StockTradesRequest(
        symbol_or_symbols=symbol_or_symbol_lst, start=start, end=end, limit=limit, feed = data_source
    )

    print('Start request')
    bar_set = stock_client.get_stock_trades(request_params=request)
    print('End request')

    return bar_set

def get_quotes(symbol_or_symbol_lst, start, limit):
    pass





# Since for each recorded time not necessarily all symbols have trade data. This 
# no need to use on trade data... for now.
def complete_timestamp_with_all_symbol_lst(df):
    # fill each timestamp with all symbol_lst -- if no data, fill with -1
    start_time = time.time()
    print('start creating missing rows...')
    unique_symbol_lst = df.index.get_level_values(0).unique().tolist()     # unique symbol_lst
    unique_timestamps = df.index.get_level_values(1).unique().tolist()  # unique timestamps
    idx = pd.MultiIndex.from_product([unique_symbol_lst, unique_timestamps])
    # print(idx)
    full = df.reindex(index=idx, fill_value=-1)
    print('missing rows added: ', full.shape[0] - df.shape[0])
    print(f'completed in {time.time()-start_time:.2f} seconds')

    return full

# this function assumes that 'full' has symbol as first index and timestamp as second index; and that each timestamp has all symbol_lst
'''
Fill the NAN/-1 values in the dataframe, mostly created by the expansion for each timestamp to include all symbols in symbol_lst.
    Expected Input:
        full: a pandas dataframe where EACH TIMESTAMP HAS ALL SYMBOLS IN SYMBOL_LST, has multi-level index (symbol, timestamp)
        symbol_num: number of symbols in the symbol_lst

'''
# DON'T USE IT ON TRADE DATA.
def infer_missing_data(full,symbol_num):
    # fill missing data with previous row (same symbol, previous timestamp)
    start_time = time.time()
    print('start filling missing data...')
    print('full shape: ', full.shape)
    # full['is_new_row'] = 0 # for debugging purpose only
    df_inferred = full.sort_index(level=1).copy()
    # note that after the sort, the original second index (timestamp) is now the first index
    for a, (i, row) in enumerate(df_inferred.iterrows()):
        if -1.0 in row.values:
            symbol, timestamp = i
            # mask = (df_inferred.index.get_level_values(0) == symbol) & (df_inferred.index.get_level_values(1) < timestamp)
            if a < symbol_num:
                print('no previous index (with same symbol) exist for ', i)
                continue
            prev_row  = df_inferred.iloc[a-symbol_num]
            new_row   = [prev_row[3]]*4        # open, high, low, close = prev_close
            new_row  += [0.0]*2                # volume & trade_count = 0
            new_row  += [prev_row[6]]   # vwap                        = prev_vwap
            df_inferred.loc[i] = new_row
        if (a+1) % 100000 == 0:
            print(f'processed {a+1} rows')
    print(f'missing data filled in {time.time()-start_time:.2f} seconds')

    # df_inferred = df_inferred.drop(columns=df.columns[-1]) # drop the last column (is_new_row)

    return df_inferred

def concat_symbol_lst(df):
    start_time = time.time()
    print('Concatenating each row for symbol_lst')
    df_concat = pd.pivot_table(df, index='timestamp', columns='symbol')
    print(df_concat.head(10))
    df_concat.columns = [f'{col[1]}_{col[0]}' for col in df_concat.columns.values] # Flatten multi-level column names
    print(f'concatination completed in {time.time()-start_time:.2f} seconds')

    return df_concat

    # Remove some rows from the beginning of the dataframe;
    # These rows are removed to clean 'TIME VOID' that is generated at start of the dataset during the filling of absent time.
    # start_time = time.time()
    # print('start concatinating symbol_lst...')
    # df = pd.read_csv(csv_path, index_col = ['symbol', 'timestamp'])
    # print(df.head(1000))
    # print(f'completed in {time.time()-start_time:.2f} seconds')

    # !!! If more features need to be calculated and added, do it here !!! After concatinating the symbol_lst, it will be much harder.
    # !!! noramlization probably should be done here as well

    # turn 2-level index into 1 level index -- keep timestamp index, and concatenate each row for symbol_lst
    # df = pd.read_csv('df_done.csv', index_col = ['symbol', 'timestamp'])
    # df_concat = concat_symbol_lst(df)

    # print(df_concat.head(10))
    
    # df_concat.to_csv('concatinated_APPL_4symbol_20220201.csv', index=True, index_label=['timestamp'])

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXassumes that df_bars is in the order of timestamp, symbol
# UNTESTED
def combine_bars(df_bars, type = 'bars'):
    start_time = time.time()
    print('Combining multiple df...')

    combined_df = pd.concat(df_bars)
    # Remove rows where both indices are the same
    duplicated = combined_df.index.duplicated()

    combined_df = combined_df[~duplicated]
    # print (combined_df)

    # combined_df.drop_duplicates(subset, keep='last', inplace=True)
    
    if type == 'bars':
        combined_df = complete_timestamp_with_all_symbol_lst(combined_df)
        symbol_num = combined_df.index.get_level_values(0).nunique() # number of unique first index
        combined_df = infer_missing_data(combined_df,symbol_num)
    # the '==' operation creates a boolean mask that selects redundent rows, and 
    print(f'multiple df combined in {time.time()-start_time:.2f} seconds')
    return combined_df

# TODO: what is the default behavior of this?
def get_latest_bars(symbol_or_symbol_lst):
    request = StockLatestBarRequest(
        symbol_or_symbol_lst=symbol_or_symbol_lst, feed = data_source
    )

    print('Start request')
    bar_set = stock_client.get_stock_latest_bar(request_params=request)
    print('End request')

    return bar_set

def last_week_bars(symbol_lst, timeframe = TimeFrame.Minute, data_pth = default_data_pth, download = True):
    # get the last week of data
    end = datetime.now()
    start = (end - timedelta(days=end.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
    df = get_and_process_data(symbol_lst, timeframe, start, end, limit = None, download= download, pre = '', post = post, data_pth = data_pth, type = 'bars', adjustment ='all')

    return df


def get_and_process_quotes(symbol, timeframe, start, limit):
    pass

############################################################################################################
def format_save_pth(symbol, timeframe, start, end, 
                    pre      = '', post = post,
                    data_pth = default_data_pth,
                    type     = 'bars',
                    data_folder   = 'raw'):

    # symbol_str = symbol
    start_str  = start.strftime('%Y%m%d')
    end_str    = end.strftime('%Y%m%d')
    time_str   = f'{start_str}_{end_str}'

    csv_pth = save_template.format(data_pth = data_pth, data_folder = data_folder, data_type = '{data_type}', pre = pre, symbol_str = symbol, time_str = time_str, timeframe_str = timeframe.value, post = post, data_source_str = data_source_str)
    return csv_pth

'''
Get data for a single symbol from start to end in a single attempt TODO: autometically determine what new data to download; instead of downloading everything, only download the new dataXXXXXXXXXXXXXXXXXXXX Trerrible idea? 

    Expected Input:
        Request Parameters: 
            symbol (use [symbol] in request)
            timeframe
            start: datetime object
            end  : datetime object
            limit: None by default
        Path Parameters:
            pre               : prefix of the csv file name. Default "". This is before "bar_set" or "trade_set", should have a trailing "_" if not empty
            post              : postfix of the csv file name. Default "raw". This is after the timeframe, should have a leading "_" if not empty
            data_pth          : path to the data folder
        download          : if true download new data, otherwise read from existing csv
        adjustment        : 'all', 'raw', 'split_only', 'dividend_only', 'none' # FIXME: look up alpaca API for exact possible inputs
        type              : 'bars' or 'trades' # TODO: add 'quotes'
    
    Expected Result:
        1) return: 
            a pandas dataframe with ALL SYMBOLS INCLUDED, has multi-level index (symbol, timestamp), and with the following features/columns: 
                symbol, timestamp,                    open, high, low,  close,  volume,   trade_count, vwap
                ARBB,   2023-04-05 15: 14: 00+00: 00, 3.6,  3.75, 3.55, 3.6,    297401.0, 213.0,       3.615044
                ARBB,   2023-04-05 15: 15: 00+00: 00, 3.6,  3.85, 3.55, 3.85,   63472.0,  336.0,       3.708484
                ARBB,   2023-04-05 15: 16: 00+00: 00, 3.8,  3.9,  3.75, 3.7925, 44114.0,  231.0,       3.835715
        2) FEATURE REMOVED: a saved .csv file of the returned pandas dataframe, according to the save_template
            This feature is now removed; all file-saving is now done in the get-load of data section.
    
    Useage:
        df = get_and_process_data(symbol_lst, timeframe, start, end, limit = None, download=False, pre = '', post = post, data_pth = default_data_pth, type = 'bars', adjustment ='all')

    
    Steps/Process:
        1. Format csv_pth
        2.a. If download is True, get data from Alpaca API, and save to csv_pth
        2.b. If download is False, read from csv_pth
'''
def get_and_process_data(symbol, timeframe, start, end, limit = None, adjustment ='all',
                         pre      = 'bar_set', post = post,
                         data_pth = default_data_pth,
                         type     = 'bars',
                         get_data = get_bars,
                         overwrite = False):
    
    print("Getting Data for symbol:", symbol)
    csv_pth = format_save_pth(symbol, timeframe, start, end,
                            pre      = pre,      post = post,
                            data_pth = data_pth,
                            type     = type)
    
    if not overwrite:
        if os.path.exists(csv_pth):
            print(f'{csv_pth} already exists')
            return pd.read_csv(csv_pth, index_col = ['symbol', 'timestamp'])

    print("csv_pth: ", csv_pth)

    if os.path.exists(csv_pth) == False:
        print(f'Downloading {type} for {symbol}:')
        dataset = get_data([symbol], timeframe, start, end, limit, adjustment = adjustment)

        # DEBUG

        df = dataset.df
        if type == 'trades':
            print(df.columns.tolist())
            df.drop(['exchange', 'id', 'conditions', 'tape'], axis = 1, inplace=True)
            print(df.columns.tolist())

        start_time = time.time()
        print(f'start saving to csv at: {csv_pth}')
        df.to_csv(csv_pth, index=True, index_label=['symbol', 'timestamp'])                       # note that the index_label is necessary; if not specified, the index name will not be saved
        print(f'csv saving completed in {time.time()-start_time:.2f} seconds\n')
    else:
        start_time = time.time()
        print(f'reading from csv at: {csv_pth}')
        df = pd.read_csv(csv_pth, index_col = ['symbol', 'timestamp'])                            # note how index_label is used to save, and index_col is used to read... bizarre ain't it?
        print(f'csv reading completed in {time.time()-start_time:.2f} seconds\n')

    return df


'''
helper function for splitting up the entire time period into smaller chunks (so that download doesn't fail)
and protect agaisnt download that does fail half way--don't need to completely restart, as some data is already received and saved

TODO: maybe try multi-threaded request?
'''
def get_next_end(start, raw_end, type = 'bars'):
    if type == 'bars':
        end = start.replace(day=1) + relativedelta(years=+1)
    elif type == 'trades':
        if (start.day < 16):
            end = start.replace(day=16)
        else:
            end = start.replace(day=1) + relativedelta(months=+1)
    
    if end > raw_end:
        end = raw_end
    return end

'''
Get data for a single symbol from start to end in a single attempt TODO: autometically determine what new data to download; instead of downloading everything, only download the new dataXXXXXXXXXXXXXXXXXXXX Trerrible idea? 

    Expected Input:
        dfs: a list of different time_sessions; each time_session is a list of dataframes for each symbol
    
    Expected Result:
        1) return: 
            a pandas dataframe with ALL SYMBOLS INCLUDED, has multi-level index (symbol, timestamp), sorted by timestamp, and with the following features/columns: 
                symbol, timestamp,                    open, high, low,  close,  volume,   trade_count, vwap
                ARBB,   2023-04-05 15: 14: 00+00: 00, 3.6,  3.75, 3.55, 3.6,    297401.0, 213.0,       3.615044
                ARBB,   2023-04-05 15: 15: 00+00: 00, 3.6,  3.85, 3.55, 3.85,   63472.0,  336.0,       3.708484
                ARBB,   2023-04-05 15: 16: 00+00: 00, 3.8,  3.9,  3.75, 3.7925, 44114.0,  231.0,       3.835715
        2) FEATURE REMOVED: a saved .csv file of the returned pandas dataframe, according to the save_template
            This feature is now removed; all file-saving is now done in the get-load of data section.
    
    Useage:
        df = get_and_process_data(symbol_lst, timeframe, start, end, limit = None, download=False, pre = '', post = post, data_pth = default_data_pth, type = 'bars', adjustment ='all')

    
    Steps/Process:
        1. Format csv_pth
        2.a. If download is True, get data from Alpaca API, and save to csv_pth
        2.b. If download is False, read from csv_pth
'''
            
    

'''
# for long timeframe and large symbol num, it might be necessary to split up the request, thus this function is needed TODO: autometically split up the request, and use this as default

    Expected Input:
        Request Parameters: Same as get_and_process_data
    
    Expected Result:
''' 


def get_load_of_data(symbol_lst, timeframe, start, end, limit = None, adjustment = 'all',
                     pre      = '', post = post,
                     data_pth = default_data_pth,
                     type     = 'bars',
                     combine  = False,
                     overwrite = False):

    get_data = get_bars
    if type == 'bars':
        get_data = get_bars
        pre = pre + 'bar_set'
    elif type == 'trades':
        get_data = get_trades
        pre = pre + 'trade_set'

    
    raw_start = start
    raw_end = end
    
    limit = None
    time_strs = []

    start_time = time.time()
    print('Start getting multiple bars file')

    start = raw_start
    if type == 'bars':
        start = start.replace(day=1, month = 1) # start from the first day of the month
    # dfs = []
    if start >= raw_end:
        raise Warning(f'start time {start} is later than end time {raw_end}')
    for symbol in symbol_lst:
        current_start = start  # Assuming initial_start is defined as the starting point
        symbol_dfs = []

        while current_start < raw_end:
            end = get_next_end(current_start, raw_end, type=type)

            start_str = current_start.strftime('%Y%m%d')
            end_str = end.strftime('%Y%m%d')
            time_str = f'{start_str}_{end_str}'
            print('getting data for time session: ', time_str)

            if not time_str in time_strs:
                time_strs.append(time_str)

            df = get_and_process_data(symbol, timeframe, current_start, end, limit, adjustment, 
                                    pre=pre, post=post, data_pth=data_pth, type=type, get_data=get_data, overwrite=overwrite)
            
            if combine:
                symbol_dfs.append(df)

            current_start = end
        
        if combine:
            symbol_df = pd.concat(symbol_dfs)
            
            
            csv_pth = format_save_pth(symbol, timeframe, start, end,
                                        pre      = pre,      post = post,
                                        data_pth = data_pth,
                                        type     = type,
                                        data_folder = 'raw')
            
            print('saving combined df to: ', csv_pth)
            if not overwrite:
                if os.path.exists(csv_pth):
                    print(f'{csv_pth} already exists')
                    continue
            symbol_df.to_csv(csv_pth, index=True, index_label=['symbol', 'timestamp'])
            print(f'saved.')



    print(f'All files downloaded and processed in {time.time()-start_time:.2f} seconds')

    print('time_strs: ', time_strs)

    # return dfs

def main():

    # get_load_of_bars(symbol_lst, timeframe, start, end, limit = None, download=False)

    # symbol_lst = ['PDD', 'DQ', 'ARBB', 'JYD', 'MGIH', 'NVDA']
    symbol_lst = ["MSFT", "AAPL"]
    timeframe = TimeFrame.Minute
    start = datetime(2023,1,1) # 2020-01-01 is wednesday
    end = datetime(2023,2,1) 

    print('start getting data\n')
    dfs = get_load_of_data(symbol_lst, timeframe, start, end, limit = None, adjustment = 'all',
                            pre = '', post = post, 
                            type = 'bars',
                            combine = True)
    

    # need to re-download prices adjusted for splits; experiment with the parameter
    # https://forum.alpaca.markets/t/getbars-adjustment/5056

if __name__ == '__main__':
    main()



### test alpaca alpaca-trade-api ###