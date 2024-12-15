import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import pathlib
default_data_pth = 'data'
save_template = "{data_pth}/{data_folder}/{pre}_{time_str}_{symbol_str}_{timeframe_str}_{post}_{data_source_str}.csv"

def format_save_pth(symbol, timeframe, start, end, 
                   pre='', post='raw',
                   data_pth=default_data_pth,
                   data_folder='raw',
                   data_source_str='yf'):
    """Format the save path similar to alpaca_history"""
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    time_str = f'{start_str}_{end_str}'

    csv_pth = save_template.format(
        data_pth=data_pth,
        data_folder=data_folder,
        pre=pre,
        symbol_str=symbol,
        time_str=time_str,
        timeframe_str=timeframe,
        post=post,
        data_source_str=data_source_str
    )
    return csv_pth

def convert_timeframe(timeframe):
    """Convert alpaca timeframe to yfinance interval"""
    # Map common timeframes. Add more as needed
    timeframe_map = {
        "1Day": "1d",
        "1Hour": "1h",
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "30Min": "30m",
        "Day": "1d",
        "Hour": "1h",
        "Minute": "1m"
    }
    return timeframe_map.get(str(timeframe), "1d")

def get_and_process_data(symbol, timeframe, start, end, pre='', post='raw', 
                        data_pth=default_data_pth, overwrite=False):
    """Get data for a single symbol and save to CSV"""
    
    csv_pth = format_save_pth(symbol, timeframe, start, end,
                             pre=pre, post=post,
                             data_pth=data_pth)
    
    # Create directory if it doesn't exist
    pathlib.Path(csv_pth).parent.mkdir(parents=True, exist_ok=True)
    
    if not overwrite and os.path.exists(csv_pth):
        print(f'{csv_pth} already exists')
        return pd.read_csv(csv_pth, index_col=['symbol', 'timestamp'])

    print(f"Downloading data for {symbol}")
    
    # Convert datetime to string format for yfinance
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    # Get data from yfinance
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        interval=convert_timeframe(timeframe),
        start=start_str,
        end=end_str
    )
    
    # Rename columns to match alpaca format
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Add required columns
    df['trade_count'] = 0  # YFinance doesn't provide this
    df['vwap'] = df['close']  # Using close as a simple approximation
    
    # Reset index to create timestamp column
    df = df.reset_index()
    df['symbol'] = symbol
    
    # Set multi-index like alpaca
    df.set_index(['symbol', 'Date'], inplace=True)
    df.index.names = ['symbol', 'timestamp']
    
    # Save to CSV
    print(f'Saving to {csv_pth}')
    df.to_csv(csv_pth, index=True, index_label=['symbol', 'timestamp'])
    
    return df

def get_load_of_data(symbol_lst, timeframe, start, end, limit=None, adjustment='all',
                     pre='', post='raw',
                     data_pth=default_data_pth,
                     type='bars',
                     combine=False,
                     overwrite=False):
    """Main function that mimics alpaca's get_load_of_data"""
    
    if type != 'bars':
        raise ValueError("Only 'bars' type is supported for YFinance data")
    
    pre = pre + 'bar_set' if type == 'bars' else pre
    
    print('Start getting multiple bars file')
    
    for symbol in symbol_lst:
        df = get_and_process_data(
            symbol, 
            timeframe, 
            start, 
            end, 
            pre=pre,
            post=post,
            data_pth=data_pth,
            overwrite=overwrite
        )
        
        if combine:
            # Save combined file
            csv_pth = format_save_pth(
                symbol, 
                timeframe, 
                start, 
                end,
                pre=pre,
                post=post,
                data_pth=data_pth,
                data_folder='raw'
            )
            
            if not overwrite and os.path.exists(csv_pth):
                print(f'{csv_pth} already exists')
                continue
                
            df.to_csv(csv_pth, index=True, index_label=['symbol', 'timestamp'])
            print(f'Saved combined data for {symbol}')

def main():
    # Example usage
    symbol_lst = ["MSFT", "AAPL"]
    timeframe = "Day"
    start = datetime(2023, 1, 1)
    end = datetime(2023, 2, 1)
    
    get_load_of_data(
        symbol_lst,
        timeframe,
        start,
        end,
        combine=True
    )

if __name__ == '__main__':
    main() 