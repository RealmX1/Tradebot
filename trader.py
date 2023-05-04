import pandas as pd
import torch
import numpy as np
import asyncio
import threading
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from alpaca.data import Trade, Snapshot, Quote, Bar, BarSet, QuoteSet, TradeSet
from alpaca.data.timeframe import TimeFrame
import datetime
import msgpack

from AI.S2S import *
from AI.sim import *
from AI.data_utils import *
from AI.model_structure_param import *
from alpaca_api.indicators import append_indicators
from alpaca_api.alpaca_history_bars import last_week_bars
from alpaca_api.alpaca_trading import *
from alpaca_api.alpaca_api_param import * # API_KEY, SECRET_KEY


# stream parmeters
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
symbols = ['AAPL','BABA','TSLA']
BAR_MAPPING = {
    "t": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "n": "trade_count",
    "vw": "vwap",
}
columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
raw_data = True

# data parameters
df = None
stream_df = None
data_path = 'data/'
df_csv_path = "data/df.csv"
stream_csv_path = "data/csv/stream.csv"

rows_buffer = [] # buffer for processing rows.
current_time_stamp = None
last_data_update_time = time.time()

# model parameters
model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
model_pth = f'../model/model_{config_name}.pt'
model.load_state_dict(torch.load(model_pth))
model.eval()
teacher_forcing_ratio = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

async def on_receive_bar(bar): # even with multiple subscritions, bars come in one by one.
    # how to handle aftermarket, when not all symbol bars are received?
    # maybe not doing data processing in this handler
    global last_data_update_time
    last_data_update_time = time.time()

    symbol = bar['S']
    print(f"received new bars:")
    print(bar)
    handler_start_time = time.time()
    mapped_bar = {
        BAR_MAPPING[key]: val for key, val in bar.items() if key in BAR_MAPPING
    }
    print(mapped_bar)
    dt = mapped_bar['timestamp'].to_datetime()
    formatted = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    del mapped_bar['timestamp']

    new_row = pd.DataFrame(mapped_bar, index=pd.MultiIndex.from_tuples([(symbol, formatted)],
                                                       names=['symbol', 'timestamp']))
    print(new_row)
    global rows_buffer, current_time_stamp
    if current_time_stamp == None:
        print("buffer is empty")
        current_time_stamp = dt        

    if dt != current_time_stamp: 
        assert rows_buffer == [], "Warning: buffer should be empty!"
        current_time_stamp = dt
    rows_buffer.append(new_row)


    

    print(f'Data processing completed {time.time() - handler_start_time} seconds after receiving bar.')

        # stream_df_indicated  = append_indicators(stream_df)
        # print(stream_df_indicated.tail(5))

        # y_pred = model()



    # stream_df.to_csv(stream_csv_path)
def my_function():
    
    print("Thread executing...")
    global last_data_update_time, rows_buffer, symbols

    no_more_data = time.time() - last_data_update_time > 5 and len(rows_buffer) != 0
    # note that when not all symbols are refreshed, it doesn't quite matter, 
    # as the policy will make the same prediction for those not updated symbols -- i.e., policy will not do anything.
    collection_complete = len(rows_buffer) == len(symbols)

    if no_more_data or collection_complete:
        print("buffer is full, start processing.")

        global stream_df
        if stream_df is None:
            global df
            stream_df = df.tail(200)
        stream_df = pd.concat([stream_df] + rows_buffer)
        rows_buffer = []
        print(stream_df.tail(5))

        
        append_indicators(stream_df)
        print(stream_df.tail(5))


        # start making prediction
        global model
        with torch.no_grad():
            predictions = {}
            groups = stream_df.groupby('symbol')
            for name, df_single_symbol in groups:
                x_np = normalize_data(df_single_symbol.tail(hist_window))
                x_batch = torch.tensor(x_np).view(1, hist_window, input_size).float().to(device)
                print(df_single_symbol.tail(3))

                print("x_batch.shape: ", x_batch.shape)
            
                y_pred = model(x_batch, None, teacher_forcing_ratio)

                # should I have policy make one decision for every symbol, or one decision for all symbols?
                # all symbols.
                # then I should put prediction of all symbols together and feed to policy


            
                


        
        alpaca_account = trading_client.get_account()


        if alpaca_account.trading_blocked:
            print('Account is currently restricted from trading.')
        # market_order_data = create_limit_order(symbol = "AAPL", price = price, qty = 1, order_side = OrderSide.BUY, tif = TimeInForce.DAY)

        # Check how much money we can use to open new positions.
        print('${} is available as buying power.'.format(alpaca_account.buying_power))
    # Add your desired function code here

def thread_function():
    while True:
        t = threading.Thread(target=my_function)
        t.start()
        time.sleep(1)


def main():
    global df
    df = last_week_bars(symbols, dp = data_path, download = False)

    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data, feed=DataFeed.SIP)
    
    for symbol in symbols:
        stream.subscribe_bars(on_receive_bar, symbol)
        # subscribe_quote
        print(f'subscribed to {symbol}')

    thread = threading.Thread(target=thread_function)
    thread.start()

    stream.run()


    print("is this printed?")

if __name__ == "__main__":
    main()