import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import json
import matplotlib.pyplot as plt

from indicator_param import *
# pd.set_option('display.max_columns', None)

'''
indicators to calculate:
    - moving average
    - bollinger bands
    - rsi
'''

# window_size = 10

pre = "bar_set"
post = "raw"

# Define function to apply to rolling window
def func(x):
    return x.sum()

def add_time_embedding(df):
    # First, make sure the index is timezone aware (in UTC)
    df['timestamps_col'] = pd.to_datetime(df.index.get_level_values(1))
#     /mnt/c/Users/zhang/OneDrive/desktop/Tradebot/alpaca_api/indicators.py:29: FutureWarning: Inferring datetime64[ns, UTC] from data containing strings is deprecated and will be removed in a future version. To retain the old behavior explicitly pass Series(data, dtype=datetime64[ns, UTC])
#   df['timestamps_col'] = pd.to_datetime(df.index.get_level_values(1))
    # df['timestamps_col'] = pd.to_datetime(df['timestamps_col'], utc=True)
    # print(df.head(5))
    # print(df.index.get_level_values(1))
    # Set the timestamps_col column as the index of the DataFrame

    # Convert the index to Eastern timezone
    df['edt_time'] = df['timestamps_col'].dt.tz_convert('US/Eastern')
    # print(df.head(5))

    # Extract the time of day (in hours) as a new column
    df['edt_hour'] = df['edt_time'].dt.hour + df['edt_time'].dt.minute / 60
    df['edt_dayofweek'] = df['edt_time'].dt.dayofweek
    # print(df.head(5))

    # Create a new column with the time of day scaled from 0.0 (9:30 am) to 1.0 (1:00 pm)
    start_hour, end_hour = 9.5, 16.0
    df['edt_scaled'] = (df['edt_hour'] - start_hour) / (end_hour - start_hour)
    df['is_core_time'] = ((df['edt_scaled'] >= 0) & (df['edt_scaled'] <= 1)).astype(int)

    df.drop(columns=['timestamps_col', 'edt_time', 'edt_hour'], inplace=True)

def append_indicators(df):
    # Add EMA
    df.ta.ema(append=True, length = 14)
    # Add DEMA
    df.ta.dema(append=True)
    # Add TEMA
    df.ta.tema(append=True)
    # Add Bollinger Bands
    df.ta.bbands(append=True)
    # Add RSI
    df.ta.rsi(append=True)
    # Add CCI
    df.ta.cci(append=True)
    # # Add DI+ and DI-
    # df.ta.dmi(append=True) # not working
    # Add ADX
    df.ta.adx(append=True)

    add_time_embedding(df) # very inefficient compared to pandas_ta indicators;
    # # the previous indicators in total used 0.025 seconds on a week's data, this one took 0.065 seconds

    # df.ta.macd(append=True)

    df.dropna(inplace = True)
    columns_2_drop_lst = ['open', 'high', 'low', 'volume', 'trade_count', \
                     f"DMN_{IndicatorParam.ADX.value['length']}", f"DMP_{IndicatorParam.ADX.value['length']}"]
    
    for column in columns_2_drop_lst:
        if column in df.columns:
            df.drop(columns=column, inplace=True)
    col_list = list(df.columns)
    
    return col_list



def main():
    features_hist_pth = 'features_hist.json'
    with open(features_hist_pth, 'r') as f:
        feature_hist_dict = json.load(f)
    # print(feature_hist_dict)
    
    # time_str = '20200101_20230417'
    # input_path = f'../data/csv/bar_set_huge_{time_str}_raw.csv'
    time_str = '20200101_20200701'
    input_path = f'../data/csv/{pre}_{time_str}_{post}.csv'
    
    df = pd.read_csv(input_path, index_col = ['symbol', 'timestamp'])
    # df = pd.read_csv('data/csv/test_ bar_set_20230101_20230412_baba.csv', index_col = ['symbol', 'timestamp'])
    # df = df.drop(df.index[:144])
    print(df.shape)

    # create column for new indicators

    # Apply rolling window function to create new column
    # df['C'] = df.rolling(window=3).apply(func)


    groups = df.groupby('symbol')

    total_calculation_time = 0
    total_csv_saving_time = 0
    # Create a new dataframe for each group
    start_time = time.time()
    print('start calculating indicators...')

    col_lst = []
    col_num_str = ''
    for name, df in groups:
        start_time2 = time.time()
        # name: the name of the group (in this case, the unique values in 'index_1')
        # group_df: the dataframe containing the group data
        
        # Do something with the group dataframe, for example:
        print(f'Group {name}:')
        
        col_lst = append_indicators(df)
        col_num_str = str(len(col_lst))
        
        calculation_time = time.time() - start_time2
        total_calculation_time += calculation_time
        print(f'finished calculating indicators for {name} in {calculation_time} seconds')
        start_time2 = time.time()

        if not (col_num_str in feature_hist_dict):
            feature_hist_dict[col_num_str] = [col_lst]
        else:
            exist = False
            for lst in feature_hist_dict[col_num_str]:
                if set(lst) == set(col_lst):
                    exist = True
            
            if not exist: 
                feature_hist_dict[col_num_str].append(col_lst)

        id = feature_hist_dict[col_num_str].index(col_lst)
        data_type = f'{col_num_str}feature{id}' # later used to construct save_path

        save_path = f'../data/csv/bar_set_{time_str}_{name}_{data_type}.csv'
        print('start saving to: ', save_path)
        df.to_csv(save_path, index=True, index_label=['symbol', 'timestamp'])
        csv_saving_time = time.time() - start_time2
        total_csv_saving_time += csv_saving_time
        print(f'finished calculating indicators for {name} in {csv_saving_time:4.2f} seconds')
        # df.to_csv(f'data/csv/test.csv', index=True, index_label=['symbol', 'timestamp'])

    print(f'finished calculating indicators for all symbols in {time.time() - start_time} seconds')
    print(col_lst)
    print(f'col num: {col_num_str}')

    # data = df.values
    # plot close_price
    # plt.plot(data[:,3])
    # plt.show()
    
    print(feature_hist_dict)
    with open(features_hist_pth, 'w') as f:
        json.dump(feature_hist_dict, f)

if __name__ == '__main__':
    main()