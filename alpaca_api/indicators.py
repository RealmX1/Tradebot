import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import json
import matplotlib.pyplot as plt
import os
import ast
from datetime import datetime

from indicator_param import *
from alpaca_history import get_load_of_data
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Exchange, DataFeed
# pd.set_option('display.max_columns', None)

'''
indicators to calculate: 
    - moving average
    - bollinger bands
    - rsi
'''

# Define function to apply to rolling window
def func(x): 
    return x.sum()

def add_time_embedding(df): 
    # Create a column from the timestamp index (second index), make sure the index is timezone aware (in UTC);
    df.loc[:, '_timestamps_col'] = pd.to_datetime(df.index.get_level_values(1), utc=True)

    # Convert the index to Eastern timezone
    df.loc[:, '_edt_time'] = df['_timestamps_col'].dt.tz_convert('US/Eastern')

    # Extract the time of day (in hours) as a new column
    df.loc[:, '_edt_hour']      = df['_edt_time'].dt.hour + df['_edt_time'].dt.minute / 60
    df.loc[:, 'edt_dayofweek'] = df['_edt_time'].dt.dayofweek

    # Create a new column with the time of day scaled from 0.0 (9:30 am) to 1.0 (1:00 pm)
    start_hour, end_hour       = 9.5, 16.0
    df.loc[:, 'edt_scaled']    = (df['_edt_hour'] - start_hour) / (end_hour - start_hour)
    df.loc[:, 'is_core_time']  = ((df['edt_scaled'] >= 0) & (df['edt_scaled'] <= 1)).astype(int)

    df.drop(columns=['_timestamps_col', '_edt_time', '_edt_hour'], inplace=True)

def append_indicators(df, mock_trade = True): # note that the mocktrade version removes the final row -- becuase it becomes NaN in rise and fall
    df.ta.ema(append=True, length = 14)
    df.ta.dema(append=True)
    df.ta.tema(append=True)
    df.ta.bbands(append=True)
    df.ta.rsi(append=True)
    df.ta.cci(append=True)
    # df.ta.dmi(append=True) # not working
    # Add ADX
    df.ta.adx(append=True)

    add_time_embedding(df) # very inefficient compared to pandas_ta indicators;
    # # the previous indicators in total used 0.025 seconds on a week's data, this one took 0.065 seconds

    mock_trade_df = None
    df.dropna(inplace = True)
    if mock_trade: 
        df['next_high'] = df['high'].shift(-1)
        df['next_fall'] = df['low'].shift(-1)
        df.dropna(inplace = True)
        df_reset      = df.reset_index()
        new_df        = df_reset[['symbol','timestamp', 'next_high', 'next_fall']]
        mock_trade_df = pd.melt(new_df, id_vars=['symbol','timestamp'], value_vars=['next_high', 'next_fall'], var_name='price_type', value_name='price')
        mock_trade_df.set_index('timestamp', inplace=True)
        mock_trade_df.sort_index(inplace=True)
        df.drop(columns=['next_high', 'next_fall'], inplace=True)

    columns_2_drop_lst =    ['open', 'high', 'low', 'volume', 'trade_count', \
                            f"DMN_{IndicatorParam.ADX.value['length']}", f"DMP_{IndicatorParam.ADX.value['length']}"]
    # next_high and next_fall columns can be used to approximate minute level trade action. 
    # will delete them after creating new dataframe and saving it.

    # for column in columns_2_drop_lst: 
    # if column in df.columns         : 
    # df.dropna(inplace = True)?
    df.drop(columns=columns_2_drop_lst, inplace=True)
    feature_list = list(df.columns)

    # print("col_list: ", feature_list)
    return feature_list, mock_trade_df


""" 
    This function is used to check if the feature_lst already exists in feature_hist_df; 
    if so use the id of the existing record; if not, create a new record
    this is to avoid saving the same feature set multiple times and mixing them up.
"""
def log_feature_record(feature_lst, feature_hist_df):
    start_time2 = time.time()
    feature_set  = set(feature_lst)
    feature_num  = len(feature_lst)
    exist_record = False
    id           = 0

    for index, record in feature_hist_df.iterrows(): 
        print("type of feature list is:", type(record['feature_lst']))
        record_feature_set = set(ast.literal_eval(record['feature_lst']))
        record_feature_num = int(record['feature_num'])
        record_id          = int(record['id'])

        if record_feature_num == feature_num:
            exist_record = False
            id           = max(id, record_id)
        
        if record_feature_set == feature_set:
            exist_record = True
            id           = record_id
            print('found existing feature_hist')
            break

    # If no existing record found, insert a new row
    if not exist_record: 
        current_timestamp = datetime.now()
        new_row           = {'feature_num': feature_num, 'timestamp': current_timestamp, 'feature_lst': str(feature_lst), 'id': id}
        feature_hist_df   = feature_hist_df._append(new_row, ignore_index=True)

    feature_record = f'{feature_num}feature{id}' # later used to construct save_pth

    return feature_record, feature_hist_df

def main(): 
    timeframe = TimeFrame.Minute
    
    training = True # HYPERPARAMETER
    pre  = "bar_set"
    post = "raw"

    start = datetime(2020, 1, 1)
    end   = datetime(2023, 1, 1)
    time_str = start.strftime('%Y%m%d') + '_' + end.strftime('%Y%m%d')  # test = '20230101_20230701'; train = '20200101_20230101'
    symbols     = ['NVDA','AAPL','MSFT','GOOG','TSLA']
    # Download all required data using alpaca_history's get_loads_of_data
    get_load_of_data(symbols, timeframe, start, end, limit = None, adjustment = 'all',
                    pre      = '', post = post,
                    type     = 'bars',
                    combine  = True)

    # Load feature history record
    features_hist_pth = 'features_hist.csv'
    if os.path.exists(features_hist_pth): 
        feature_hist_df = pd.read_csv(features_hist_pth)
    else: 
        feature_hist_df = pd.DataFrame(columns=['feature_num', 'timestamp','feature_lst', 'id'])


    # Generate save path and input path
    
    timeframe         = TimeFrame.Minute.value
    data_source = DataFeed.IEX # DataFeed. is default/free; SIP & OTC are available through paid subscription
    data_source_str = str(data_source)[-3:]
    
    input_pth_template = 'data/raw_combine/{pre}_{time_str}_{symbol}_{timeframe}_{post}_{data_source}.csv'.format(pre = pre, time_str = time_str, timeframe = timeframe, post = post, data_source = data_source_str,
                                                                                                        symbol = '{symbol}')
    save_pth_template  = 'data/{purpose}/{pre}_{time_str}_{symbol}_{timeframe}_{feature_record}_{data_source}.csv'.format(pre = pre, time_str = time_str, timeframe = timeframe, data_source = data_source_str,
                                                                                                          purpose = '{purpose}', symbol = '{symbol}', feature_record = '{feature_record}')
    
    if training == True:
        save_pth_template = save_pth_template.format(purpose = 'train', symbol = '{symbol}', feature_record = '{feature_record}')
        mock_trade        = False
    else: 
        save_pth_template = save_pth_template.format(purpose = 'test', symbol = '{symbol}', feature_record = '{feature_record}')
        mock_trade        = True
    
    for symbol in symbols: 
        input_path                  = input_pth_template.format(pre = 'bar_set', symbol = symbol, time_str = time_str, timeframe = timeframe, post = 'raw')
        indicated_save_pth_template = save_pth_template.format(pre = 'bar_set', symbol = symbol, time_str = time_str, timeframe = timeframe, feature_record = '{feature_record}')
        df                          = pd.read_csv(input_path, index_col = ['symbol', 'timestamp'])
        print(df.shape)


        total_calculation_time = 0
        total_csv_saving_time  = 0
        # Create a new dataframe for each group
        start_time = time.time()
        print('start calculating indicators...')

        feature_lst = []
        
        # groups = df.groupby('symbol')
        # for symbol, df in groups: 
        # symbol  : the symbol of the group (in this case, the unique values in 'index_1')
        start_time2 = time.time()
        
        feature_lst, mock_trade_df = append_indicators(df, mock_trade = mock_trade)
        
        calculation_time        = time.time() - start_time2
        total_calculation_time += calculation_time
        print(f'finished calculating indicators for {symbol} in {calculation_time} seconds')

        # Check if the feature_lst already exists in feature_hist_df; 
        #   if so use the id of the existing record; if not, create a new record
        # this is to avoid saving the same feature set multiple times and mixing them up.
        feature_record, feature_hist_df = log_feature_record(feature_lst, feature_hist_df)
        
        save_pth  = indicated_save_pth_template.format(feature_record = feature_record)
        df.to_csv(save_pth, index=True, index_label=['symbol', 'timestamp'])
        print('start saving to: ', save_pth)
        if mock_trade: 
            assert 2*df.shape[0] == mock_trade_df.shape[0]
            mock_trade_path       = f'../data/csv/test/mock_trade_{time_str}_{symbol}.csv'
            mock_trade_df.to_csv(mock_trade_path)
        csv_saving_time        = time.time() - start_time2
        total_csv_saving_time += csv_saving_time
        print(f'finished calculating indicators for {symbol} in {csv_saving_time:4.2f} seconds')
        # df.to_csv(f'data/csv/test.csv', index=True, index_label=['symbol', 'timestamp'])

    print(f'finished calculating indicators for all symbols in {time.time() - start_time} seconds')
    print(feature_lst)

    # data = df.values
    # plot close_price
    # plt.plot(data[:,3])
    # plt.show()
    
    # print(feature_hist_dict)
    feature_hist_df.to_csv(features_hist_pth, index=False)

if __name__ == '__main__':
    main()