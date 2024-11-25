import os
import sys
sys.path.append('AI')  # add the path to my_project
sys.path.append('alpaca_api') 
sys.path.append('sim')


from alpaca_history import *
from indicators import *
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from alpaca.data.timeframe import TimeFrame

sp100_info_pth = 'data/sp100_info.csv'


# save_template = "{data_pth}/{data_type}/{pre}_{symbol_str}_{time_str}_{timeframe_str}_{post}.{data_type}"

def get_sp100_symbols():
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    sp100_table = pd.read_html(url, attrs={"class": "wikitable sortable"}, header=0)[0]
    sp100_symbols_lst = sp100_table['Symbol'].tolist()
    print(sp100_symbols_lst)

    return sp100_symbols_lst

def update_sp100_info():
    sp100_symbols_lst = get_sp100_symbols()
    info_list = []
    for symbol in sp100_symbols_lst:
        print(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        info['Symbol'] = symbol
        info_list.append(info)

    # Create a DataFrame from the list of dictionaries
    info_df = pd.DataFrame(info_list)

    # Save the DataFrame to a CSV file
    info_df.to_csv(sp100_info_pth, index=False)

def get_pth_template(symbol, start, end):
    pth_template_2 = 'data/{type}/{purpose}/bar_set_{symbol}_{time_str}_raw.{type}'

    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    time_str = f"{start_str}_{end_str}"
    pth_template = pth_template_2.format(symbol=symbol, time_str=time_str, type = '{type}', purpose = '{purpose}')

    return pth_template

def remove_existing_data(purpose, symbol_lst, timeframe, start, end):
    result_lst = []
    for symbol in symbol_lst:
        csv_pth = get_pth_template(symbol, start, end).format(purpose = purpose, type='csv')
        if not os.path.exists(csv_pth):
            result_lst.append(symbol)
    
    existing_symbols = list(set(symbol_lst) - set(result_lst))
    print(f"existing_symbols: {existing_symbols}")
    
    return result_lst

def main():
    # update_sp100_info()
    timeframe = TimeFrame.Minute
    train_start = datetime(2020, 1, 1)
    train_end = datetime(2023, 1, 1)
    train_symbol_lst = []

    test_start = datetime(2023, 1, 1)
    test_end = datetime(2023, 5, 1)
    test_symbol_lst = ['DQ', 'PDD','AAPL','VZ']

    sp100_info_df = pd.read_csv(sp100_info_pth, index_col = ['symbol'])
    print(sp100_info_df.columns.tolist())
    sp100_info_df = sp100_info_df[['overallRisk', 'trailingPE', 'earningsGrowth', 'profitMargins', 'marketCap', 'averageDailyVolume10Day', 'currentPrice']]
    sp100_info_df.dropna(inplace = True)
    sp100_info_df.drop(sp100_info_df[sp100_info_df['trailingPE'] < 0].index, inplace = True)
    sp100_info_df.drop(sp100_info_df[sp100_info_df['trailingPE'] > 50].index, inplace = True)
    
    sp100_info_df['volume*price'] = sp100_info_df['averageDailyVolume10Day'] * sp100_info_df['currentPrice']
    # sp100_info_df[''] = sp100_info_df[sp100_info_df['trailingPE'] > 0]
    
    
    sp100_info_df.sort_values('volume*price' ,ascending = False, inplace=True)
    print(sp100_info_df.head(20).index.tolist())
    train_symbol_lst = sp100_info_df.head(20).index.tolist()
    print("train symbol lst: ",train_symbol_lst)

    # remove the pkl that already exist.
    train_symbol_lst = remove_existing_data('training', train_symbol_lst, timeframe, train_start, train_end)



    symbols = ['AAPL', 'TSLA', 'NVDA', 'PDD', 'DQ', 'ARBB', 'VZ', 'JYD', 'MGIH']
        
    df = pd.read_csv(input_path, index_col = ['symbol', 'timestamp'])

    download = True
    # for symbol in symbols:
    #     get_and_process_data([symbol], timeframe, test_start, test_end, limit = None, download = download, type = 'bars', data_pth = 'data')
        # get_load_of_data([symbol], timeframe, train_start, train_end, limit = None, download = download, type = 'bars', data_pth = 'data')
    
    
    for symbol in symbols:
        start_str = test_start.strftime('%Y%m%d')
        end_str = test_end.strftime('%Y%m%d')
        time_str = f"{start_str}_{end_str}"
        symbol = 'AAPL'
        timeframe = TimeFrame.Minute.value
        input_path = f'../data/csv/testing/bar_set_{symbol}_{time_str}_{timeframe}_raw.csv'

        df = pd.read_csv(input_path, index_col = ['symbol', 'timestamp'])
        df, mock_trade_df = append_indicators(df)
        
    

    # need better way to pick stock!

    # sp100_symbols_lst = sp100_info_df.index.tolist()
    # for symbol in sp100_symbols_lst:
    #     stock = yf.Ticker(symbol)
    #     print(stock.info)
    #     break


if __name__ == '__main__':
    main()