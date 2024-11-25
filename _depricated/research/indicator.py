import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import io

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
    df['edt_dayofweek'] = df['edt_time'].dt.dayofweek
    # print(df.head(5))

    # Create a new column with the time of day scaled from 0.0 (9:30 am) to 1.0 (1:00 pm)
    start_hour, end_hour = 9.5, 16.0
    df['edt_scaled'] = (df['edt_hour'] - start_hour) / (end_hour - start_hour)
    df['is_core_time'] = ((df['edt_scaled'] >= 0) & (df['edt_scaled'] <= 1)).astype(int)

    df.drop(columns=['timestamps_col', 'edt_time', 'edt_hour'], inplace=True)

def append_indicators(df_raw):
    df_raw.ta.cdl_pattern(name="all", append=True)
    # # Add EMA
    # df_raw.ta.ema(append=True)
    # # Add DEMA
    # df_raw.ta.dema(append=True)
    # # Add TEMA
    # df_raw.ta.tema(append=True)
    # # Add Bollinger Bands
    # df_raw.ta.bbands(append=True)
    # # Add RSI
    # df_raw.ta.rsi(append=True)
    # # Add CCI
    # df_raw.ta.cci(append=True)
    # # # Add DI+ and DI-
    # # df.ta.dmi(append=True) # not working
    # # Add ADX
    # df_raw.ta.adx(append=True)

    add_time_embedding(df_raw) # very inefficient compared to pandas_ta indicators;
    # the previous indicators in total used 0.025 seconds on a week's data, this one took 0.065 seconds

    df = df_raw.dropna()
    # print(df.columns)
    return df

@contextmanager
def suppress_print():
    """Suppresses print statements within a with block"""
    old_stdout = sys.stdout
    sys.stdout = None
    try:
        yield
    finally:
        sys.stdout = old_stdout

def analyze(df, col_name, trend_reversal, trend_window):

    labeled_rows = trend_reversal[df[col_name] != 0.0]
    true_positive = labeled_rows.sum()
    false_positive = len(labeled_rows) - true_positive

    non_labeled_rows = trend_reversal[df[col_name] == 0.0]
    false_negative = non_labeled_rows.sum() # false negative is when the model predicts no reversal but there is a reversal
    true_negative = len(non_labeled_rows) - false_negative # true_negative is when the model predicts no reversal and there is no reversal
    
    
    print(true_positive, true_negative, false_positive, false_negative)
    precision = true_positive / (true_positive + false_positive)
    print(f"naive {trend_window} min trend reversal pred precision: {precision*100:.2f}%") 
    # recall = true_positive / (true_positive + false_negative)
    # print(f"naive {trend_window} min trend reversal pred recall: {recall*100:.2f}%")
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    print(f"naive {trend_window} min trend reversal pred acc: {accuracy*100:.2f}%")
    reversal_percentage = trend_reversal.sum()/len(trend_reversal)
    print(f"naive {trend_window} min trend reversal raw percentage: {reversal_percentage*100:.2f}%") # raw percision is the same as naive percision (random guessing)
    total_reversal = trend_reversal.sum()
    print(f"total reversal: {total_reversal:.2f}")

    # count = len(df[df[col_name] != 0.0])
    # print(f"Number of rows where {col_name} isn't 0.0:", count)
    if col_name == 'CDL_DOJI_10_0.1':
        total_label = (df[col_name]/100).sum()
    elif col_name == 'CDL_2CROWS':
        # print(df['CDL_2CROWS'].unique()) # [0. -100.]
        total_label = df[col_name].sum()
    else:
        print(df['CDL_2CROWS'].unique())
        total_label = df[col_name].sum()
    

    print(f"total {col_name} sign: {total_label:.2f}")


def analyze_indicators(df):
    print("\nStarting naive analysis of doji candle reversal...")

    analysis_window = 10
    # doji_reversal_pred_accs = []
    # total_reversal_rates = []

    col_names = ['CDL_DOJI_10_0.1', 'CDL_2CROWS', 'CDL_3BLACKCROWS']
    for col_name in col_names:
        count = len(df[df[col_name] != 0.0])
        print(f"Number of rows where {col_name} isn't 0.0:", count)

    for trend_window in range (1, analysis_window+1):
        print(f"analysing naive {trend_window} min trend reversal...")
        
        # calculate rolling mean of the previous 5 rows and the next 'trend_window' rows
        df['prev_mean'] = df['close'].rolling(trend_window, min_periods=1).mean().shift(1)
        # print(prev_mean)
        df['next_mean'] = df['close'].rolling(trend_window, min_periods=1).mean().shift(-trend_window)
        # print(next_mean)
        # cols = df.loc[:, ['next_mean', 'prev_mean', 'close']]
        # print(cols.head(20))

        now_from_prev = df['close'] - df['prev_mean']
        next_from_now = -df['close'] + df['next_mean'] 
        # print(now_from_prev.head(20))
        # print(next_from_now.head(20))
        now_from_prev_sign = np.sign(now_from_prev)
        next_from_now_sign = np.sign(next_from_now)
        # print(now_from_prev_sign.shape)

        
        # print(((now_from_prev_sign != 0) & (next_from_now_sign != 0)).head(20))

        trend_reversal = (now_from_prev_sign != 0) & (next_from_now_sign != 0) & (now_from_prev_sign != next_from_now_sign)
        bearish_reversal = (now_from_prev_sign == 1) & (next_from_now_sign == -1)
        bullish_reversal = (now_from_prev_sign == -1) & (next_from_now_sign == 1)
        # print(trend_reversal)
        # print(trend_reversal.shape)
        # analyze(df, 'CDL_DOJI_10_0.1', trend_reversal, trend_window) # 0.5% accuracy increase compared to random guessing
        # analyze(df, 'CDL_2CROWS', bearish_reversal, trend_window) # almost confirmed to be useless.
        # analyze(df, 'CDL_3BLACKCROWS', bearish_reversal, trend_window) # almost confirmed to be useless.
        # analyze(df, 'CDL_3INSIDE', bearish_reversal, trend_window) # 2~3% increase in accuracy compared to random guessing
        analyze(df, 'CDL_3LINESTRIKE', bearish_reversal, trend_window) # 2~3% increase in accuracy compared to random guessing
        
        # analyze(df, )




        bullish_reversal = (now_from_prev_sign == 1) & (next_from_now_sign == -1)
        bearish_reversal = (now_from_prev_sign == -1) & (next_from_now_sign == 1)


        # doji_reversal_pred_acc = doji_reversal.sum() / len(doji_reversal)
        # print (f"doji candle naive {trend_window} min trend reversal pred acc: {doji_reversal_pred_acc:.4f}")
        
        # total_reversal_rate = trend_reversal.sum() / len(trend_reversal)
        # print (f"total reversal rate, i.e., accuracy of randomly guessing reversal: {total_reversal_rate:.4f}")
        # print (f"{trend_window} min doji candle reversal prediction accuracy analysis done.\n")

        # doji_reversal_pred_accs.append(doji_reversal_pred_acc)
        # total_reversal_rates.append(total_reversal_rate)

    # x = range(len(doji_reversal_pred_accs))
    # assert len(doji_reversal_pred_accs) == len(total_reversal_rates), "length problem"

    # plt.plot(x, doji_reversal_pred_accs, label = "doji candle reversal prediction accuracy")
    # plt.plot(x, total_reversal_rates, label = "total reversal rate")
    # plt.xlabel('trend calcalculation window')
    # plt.legend()
    # plt.show()



def main():
    time_str = '20200101_20230417'
    df = pd.read_csv(f'data/bar_set_huge_{time_str}_raw.csv', index_col = ['symbol', 'timestamp'])
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
    print("start calculating indicators...")
    for name, df in groups:
        start_time2 = time.time()
        # print(df.head(5))
        # name: the name of the group (in this case, the unique values in 'index_1')
        # group_df: the dataframe containing the group data
        
        # Do something with the group dataframe, for example:
        print(f"Group {name}:")
        
        df = append_indicators(df)
    #   Index(['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap',
    #    'CDL_2CROWS', 'CDL_3BLACKCROWS', 'CDL_3INSIDE', 'CDL_3LINESTRIKE',
    #    'CDL_3OUTSIDE', 'CDL_3STARSINSOUTH', 'CDL_3WHITESOLDIERS',
    #    'CDL_ABANDONEDBABY', 'CDL_ADVANCEBLOCK', 'CDL_BELTHOLD',
    #    'CDL_BREAKAWAY', 'CDL_CLOSINGMARUBOZU', 'CDL_CONCEALBABYSWALL',
    #    'CDL_COUNTERATTACK', 'CDL_DARKCLOUDCOVER', 'CDL_DOJI_10_0.1',
    #    'CDL_DOJISTAR', 'CDL_DRAGONFLYDOJI', 'CDL_ENGULFING',
    #    'CDL_EVENINGDOJISTAR', 'CDL_EVENINGSTAR', 'CDL_GAPSIDESIDEWHITE',
    #    'CDL_GRAVESTONEDOJI', 'CDL_HAMMER', 'CDL_HANGINGMAN', 'CDL_HARAMI',
    #    'CDL_HARAMICROSS', 'CDL_HIGHWAVE', 'CDL_HIKKAKE', 'CDL_HIKKAKEMOD',
    #    'CDL_HOMINGPIGEON', 'CDL_IDENTICAL3CROWS', 'CDL_INNECK', 'CDL_INSIDE',
    #    'CDL_INVERTEDHAMMER', 'CDL_KICKING', 'CDL_KICKINGBYLENGTH',
    #    'CDL_LADDERBOTTOM', 'CDL_LONGLEGGEDDOJI', 'CDL_LONGLINE',
    #    'CDL_MARUBOZU', 'CDL_MATCHINGLOW', 'CDL_MATHOLD', 'CDL_MORNINGDOJISTAR',
    #    'CDL_MORNINGSTAR', 'CDL_ONNECK', 'CDL_PIERCING', 'CDL_RICKSHAWMAN',
    #    'CDL_RISEFALL3METHODS', 'CDL_SEPARATINGLINES', 'CDL_SHOOTINGSTAR',
    #    'CDL_SHORTLINE', 'CDL_SPINNINGTOP', 'CDL_STALLEDPATTERN',
    #    'CDL_STICKSANDWICH', 'CDL_TAKURI', 'CDL_TASUKIGAP', 'CDL_THRUSTING',
    #    'CDL_TRISTAR', 'CDL_UNIQUE3RIVER', 'CDL_UPSIDEGAP2CROWS',
    #    'CDL_XSIDEGAP3METHODS', 'edt_dayofweek', 'edt_scaled', 'is_core_time'],
    #   dtype='object')

        calculation_time = time.time() - start_time2
        total_calculation_time += calculation_time
        print(df.shape)
        print(f"finished calculating indicators for {name} in {calculation_time} seconds")
        # start_time2 = time.time()
        # print("start saving csv...")
        # df.to_csv(f'data/bar_set_huge_{time_str}_{name}_indicator.csv', index=True, index_label=['symbol', 'timestamp'])
        # csv_saving_time = time.time() - start_time2
        # total_csv_saving_time += csv_saving_time
        # print(f"finished calculating indicators for {name} in {csv_saving_time} seconds")
        # df.to_csv(f'data/csv/test.csv', index=True, index_label=['symbol', 'timestamp'])
    print(f"finished calculating indicators for all symbols in {time.time() - start_time} seconds")
    
    # with suppress_print():
    analyze_indicators(df)

    data = df.values
    df = df.reset_index(level=0, drop=True)
    df.index = pd.to_datetime(df.index)
    # print(df)
    # plot close_price
    fig, ax1 = plt.subplots()

    # for index, row in df.iterrows():
    #     if row['CDL_DOJI_10_0.1']!=0.0:
    #         print(f"Row index: {index}")
    #         print(f"Values: {row['CDL_DOJI_10_0.1']}")

    # price_hist = df['close'].values[:1000]
    # np_doji = df.filter(like='DOJI').values[:1000]
    # ax1.plot(price_hist, label = 'price')

    # buy_time = [i for i, x in enumerate(np_doji) if x != 0]
    # buy_price = [p for x, p in zip(np_doji,price_hist) if x != 0]
    # ax1.scatter(buy_time, buy_price, marker = '^', label = 'doji', )







    # import mplfinance as mpf

    # # Define the data to plot

    # # Define the kwargs for the candlestick plot
    # kwargs = dict(type='candle', volume=True, figratio=(16,10), figscale=1.5)

    # # Plot the data
    # mpf.plot(df.head(500), **kwargs)

    # # ax2 = ax1.twinx()
    
    
    # # ax2.scatter(df.filter(like='DOJI').values[:1000])

    # plt.show()

if __name__ == "__main__":
    main()