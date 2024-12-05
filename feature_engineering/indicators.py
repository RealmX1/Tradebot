import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import os
import ast
from datetime import datetime

from alpaca_api.alpaca_history import get_load_of_data, default_data_pth
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Exchange, DataFeed
# pd.set_option('display.max_columns', None)

from stock_indicators import Quote
from stock_indicators.indicators import get_alma, get_atr, get_bollinger_bands, get_rsi, get_adx
import pathlib

'''
indicators to calculate: 
    - moving average
    - bollinger bands
    - rsi
'''

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

def df_to_quotes(df):
    """Convert DataFrame to list of Quote objects"""
    return [
        Quote(d, o, h, l, c, v) 
        for d, o, h, l, c, v 
        in zip(df["_timestamps_col"], df['open'], df['high'], 
               df['low'], df['close'], df['volume'])
    ]

def append_indicators(df):
    """Add technical indicators using stock-indicators package"""
    
    df.loc[:, '_timestamps_col'] = pd.to_datetime(df.index.get_level_values(1), utc=True)
    
    quotes = df_to_quotes(df)
    
    # Calculate indicators
    alma_results = get_alma(quotes, lookback_periods=14)
    bb_results = get_bollinger_bands(quotes, lookback_periods=20)
    rsi_results = get_rsi(quotes, lookback_periods=14)
    atr_results = get_atr(quotes, lookback_periods=14)
    adx_results = get_adx(quotes, lookback_periods=14)
    
    # Add results to dataframe
    df['ALMA_14'] = [r.alma for r in alma_results]
    
    df['BB_SMA_20'] = [r.sma for r in bb_results]
    df['BB_UPPER_20'] = [r.upper_band for r in bb_results]
    df['BB_LOWER_20'] = [r.lower_band for r in bb_results]
    df['BB_WIDTH_20'] = [r.width for r in bb_results]
    df['BB_Z_SCORE_20'] = [r.z_score for r in bb_results]
    
    df['RSI_14'] = [r.rsi for r in rsi_results]
    
    df['ATR_14'] = [r.atr for r in atr_results]
    df['ATRP_14'] = [r.atrp for r in atr_results]
    
    df['ADX_14'] = [r.adx for r in adx_results]
    
    add_time_embedding(df)

    
    # Create mock trade data
    mock_trade_df = None
    df.dropna(inplace=True)
    
    df['next_high'] = df['high'].shift(-1)
    df['next_fall'] = df['low'].shift(-1)
    df.dropna(inplace=True)
    df_reset = df.reset_index()
    new_df = df_reset[['symbol', 'timestamp', 'next_high', 'next_fall']]
    mock_trade_df = pd.melt(new_df, id_vars=['symbol', 'timestamp'], 
                            value_vars=['next_high', 'next_fall'], 
                            var_name='price_type', value_name='price')
    mock_trade_df.set_index('timestamp', inplace=True)
    mock_trade_df.sort_index(inplace=True)
    df.drop(columns=['next_high', 'next_fall'], inplace=True)

    columns_2_drop_lst = ['open', 'high', 'low', 'volume', 'trade_count']
    df.drop(columns=columns_2_drop_lst, inplace=True)
    feature_list = list(df.columns)

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

class PathConfig:
    def __init__(self, base_dir=default_data_pth, purpose=None):
        self.base_dir = pathlib.Path(base_dir)
        self.purpose = purpose
        
    def get_raw_path(self, symbol, timeframe, time_str, data_source):
        """Get path for raw data files"""
        path = self.base_dir / 'raw' / f'bar_set_{time_str}_{symbol}_{timeframe}_raw_{data_source}.csv'
        # create the directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_processed_path(self, symbol, timeframe, time_str, feature_record, data_source):
        """Get path for processed data files"""
        path = self.base_dir / self.purpose / f'{time_str}_{symbol}_{timeframe}_{feature_record}_{data_source}.csv'
        # create the directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_mock_trade_path(self, symbol, timeframe, time_str):
        """Get path for mock trade files"""
        path = self.base_dir / self.purpose / 'mock_trade' / f'{time_str}_{symbol}_{timeframe}.csv'
        # create the directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
symbols = [
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK.B", "GOOG", "AVGO",
    "JPM", "LLY", "UNH", "V", "XOM", "COST", "MA", "HD", "PG", "WMT",
    "NFLX", "JNJ", "CRM", "BAC", "ABBV", "ORCL", "CVX", "MRK", "WFC", "KO",
    "CSCO", "ADBE", "AMD", "NOW", "ACN", "LIN", "PEP", "IBM", "DIS", "MCD",
    "PM", "TMO", "ABT", "GE", "ISRG", "CAT", "GS", "INTU", "QCOM", "TXN"
]

def indicate(symbol_lst=symbols, training=True, timeframe = TimeFrame.Day):
    paths = PathConfig(purpose='train' if training else 'test')
    
    if training:
        start = datetime(2015, 1, 1)
        end = datetime(2023, 1, 1)
    else:
        start = datetime(2023, 1, 1)
        end = datetime(2023, 7, 1)

    time_str = start.strftime('%Y%m%d') + '_' + end.strftime('%Y%m%d')
    
    # Download data
    get_load_of_data(symbol_lst, timeframe, start, end, limit=None, adjustment='all',
                    pre='', post='raw', type='bars', combine=True)

    # Load feature history
    features_hist_pth = pathlib.Path('feature_engineering/features_hist.csv')
    if features_hist_pth.exists():
        feature_hist_df = pd.read_csv(features_hist_pth)
    else:
        feature_hist_df = pd.DataFrame(columns=['feature_num', 'timestamp', 'feature_lst', 'id'])

    data_source = str(DataFeed.IEX)[-3:]
    
    for symbol in symbol_lst:
        input_path = paths.get_raw_path(symbol, timeframe.value, time_str, data_source)
        print(f"Loading data for {symbol} from {input_path}")
        df = pd.read_csv(input_path, index_col=['symbol', 'timestamp'])
        
        feature_lst, mock_trade_df = append_indicators(df)
        
        feature_record, feature_hist_df = log_feature_record(feature_lst, feature_hist_df)
        
        save_path = paths.get_processed_path(
            symbol, timeframe.value, time_str, feature_record, data_source
        )
        
        mock_trade_path = paths.get_mock_trade_path(symbol, timeframe.value, time_str)
        mock_trade_df.to_csv(mock_trade_path)
            
        df.to_csv(save_path, index=True, index_label=['symbol', 'timestamp'])

    feature_hist_df.to_csv(features_hist_pth, index=False)

def main():    
    indicate()

if __name__ == '__main__':
    main()