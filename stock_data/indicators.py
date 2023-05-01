import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', None)

'''
indicators to calculate:
    - moving average
    - bollinger bands
    - rsi
'''

window_size = 10

# Define function to apply to rolling window
def func(x):
    return x.sum()

def add_time_embedding(df):
    # First, make sure the index is timezone aware (in UTC)
    df['timestamps_col'] = pd.to_datetime(df.index.get_level_values(1))
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

def append_indicators(df_raw):
    # Add EMA
    df_raw.ta.ema(append=True)
    # Add DEMA
    df_raw.ta.dema(append=True)
    # Add TEMA
    df_raw.ta.tema(append=True)
    # Add Bollinger Bands
    df_raw.ta.bbands(append=True)
    # Add RSI
    df_raw.ta.rsi(append=True)
    # Add CCI
    df_raw.ta.cci(append=True)
    # # Add DI+ and DI-
    # df.ta.dmi(append=True) # not working
    # Add ADX
    df_raw.ta.adx(append=True)

    add_time_embedding(df_raw) # very inefficient compared to pandas_ta indicators;
    # # the previous indicators in total used 0.025 seconds on a week's data, this one took 0.065 seconds

    # df_raw.ta.macd(append=True)

    df = df_raw.dropna()

    return df

def main():
    
    # time_str = '20200101_20230417'
    # input_path = f'../data/csv/bar_set_huge_{time_str}_raw.csv'
    time_str = '20230418_20230501'
    input_path = f'../data/csv/bar_set_{time_str}_raw.csv'
    
    data_type = '23feature' # later used to construct save_path
    df = pd.read_csv(input_path, index_col = ['symbol', 'timestamp'])
    # df = pd.read_csv('data/csv/test_ bar_set_20230101_20230412_baba.csv', index_col = ['symbol', 'timestamp'])
    # df = df.drop(df.index[:144])
    print(df.shape)

    # create column for new indicators

    # Apply rolling window function to create new column
    # df['C'] = df.rolling(window=3).apply(func)


    groups = df.groupby('symbol')
    columns = []

    total_calculation_time = 0
    total_csv_saving_time = 0
    # Create a new dataframe for each group
    start_time = time.time()
    print('start calculating indicators...')
    for name, df in groups:
        start_time2 = time.time()
        print(df.head(5))
        # name: the name of the group (in this case, the unique values in 'index_1')
        # group_df: the dataframe containing the group data
        
        # Do something with the group dataframe, for example:
        print(f'Group {name}:')
        
        df = append_indicators(df)

        print(df.head(5))
        # columns = list(df.columns)
        # print(columns)
        calculation_time = time.time() - start_time2
        total_calculation_time += calculation_time
        print(f'finished calculating indicators for {name} in {calculation_time} seconds')
        start_time2 = time.time()
        print('start saving csv...')
        save_path = f'../data/csv/bar_set_huge_{time_str}_{name}_{data_type}.csv'
        df.to_csv(save_path, index=True, index_label=['symbol', 'timestamp'])
        csv_saving_time = time.time() - start_time2
        total_csv_saving_time += csv_saving_time
        print(f'finished calculating indicators for {name} in {csv_saving_time} seconds')
        # df.to_csv(f'data/csv/test.csv', index=True, index_label=['symbol', 'timestamp'])
    print(f'finished calculating indicators for all symbols in {time.time() - start_time} seconds')

    data = df.values
    # plot close_price
    plt.plot(data[:,3])
    plt.show()


    # ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'EMA_10', 'DEMA_10', 'TEMA_10', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0', 'RSI_14', 'CCI_14_0.015', 'ADX_14', 'DMP_14', 'DMN_14']
    # normalize_method: 0: no normalization, 1: normalize using close, 2: normalize itself, 3: custom normalization (fixed value)
    normalize_method = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3]

if __name__ == '__main__':
    main()