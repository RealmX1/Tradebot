import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', None)

"""
indicators to calculate:
    - moving average
    - bollinger bands
    - rsi
""" 

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
    # print(df.head(5))

    # Create a new column with the time of day scaled from 0.0 (9:30 am) to 1.0 (1:00 pm)
    start_hour, end_hour = 9.5, 16.0
    df['edt_scaled'] = (df['edt_hour'] - start_hour) / (end_hour - start_hour)
    df['is_core_time'] = ((df['edt_scaled'] >= 0) & (df['edt_scaled'] <= 1)).astype(int)

    df.drop(columns=['timestamps_col', 'edt_time', 'edt_hour'], inplace=True)

def main():
    df = pd.read_csv('data/csv/bar_set_huge_20200101_20230412_baba.csv', index_col = ['symbols', 'timestamps'])
    # df = df.drop(df.index[:144])
    print(df.shape)

    # create column for new indicators

    # Apply rolling window function to create new column
    # df['C'] = df.rolling(window=3).apply(func)


    groups = df.groupby('symbols')
    columns = []

    # Create a new dataframe for each group
    start_time = time.time()
    print("start calculating indicators...")
    for name, df in groups:
        print(df.head(5))
        # name: the name of the group (in this case, the unique values in 'index_1')
        # group_df: the dataframe containing the group data
        
        # Do something with the group dataframe, for example:
        print(f"Group {name}:")
        # Add EMA
        df.ta.ema(append=True)
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

        add_time_embedding(df)

        df = df.dropna()

        print(df.head(5))
        # columns = list(df.columns)
        # print(columns)
        
        df.to_csv(f'data/csv/bar_set_huge_20200101_20230412_{name}_indicator.csv', index=True, index_label=['symbols', 'timestamps'])
    print(f"finished calculating indicators in {time.time() - start_time} seconds")

    data = df.values
    # plot close_price
    plt.plot(data[:,3])
    plt.show()


    # ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'EMA_10', 'DEMA_10', 'TEMA_10', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0', 'RSI_14', 'CCI_14_0.015', 'ADX_14', 'DMP_14', 'DMN_14']
    # normalize_method: 0: no normalization, 1: normalize using close, 2: normalize itself, 3: custom normalization (fixed value)
    normalize_method = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3]

if __name__ == "__main__":
    main()