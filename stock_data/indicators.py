import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

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

def main():
    df = pd.read_csv('data/csv/bar_set_huge_20180101_20230410.csv', index_col = ['symbols', 'timestamps'])
    df = df.drop(df.index[:144])
    print(df.shape)
    print(df.head(12))

    # create column for new indicators

    # Apply rolling window function to create new column
    # df['C'] = df.rolling(window=3).apply(func)


    groups = df.groupby('symbols')
    columns = []

    # Create a new dataframe for each group
    start_time = time.time()
    print("start calculating indicators...")
    for name, df in groups:
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


        df = df.dropna()

        # print(df.head(12))
        # columns = list(df.columns)
        # print(columns)
        
        df.to_csv(f'data/csv/bar_set_huge_20180101_20230410_{name}_indicator.csv', index=True, index_label=['symbols', 'timestamps'])
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