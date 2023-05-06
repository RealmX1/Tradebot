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
import time
import pytz
est_tz = pytz.timezone('US/Eastern')

import sys
sys.path.append('AI')  # add the path to my_project
sys.path.append('alpaca_api') 
sys.path.append('sim')

from S2S import *
from data_utils import *
from model_structure_param import *
from indicators import append_indicators
from alpaca_history import last_week_bars
from alpaca_trade import *
from alpaca_api_param import * # API_KEY, SECRET_KEY
from policy import *
from account import *



# stream parmeters
trading_client = TradingClient(PAPER_API_KEY, PAPER_SECRET_KEY, paper=True)
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
wait_time = time.time()
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
model_pth = f'model/last_model_{config_name}.pt'
model.load_state_dict(torch.load(model_pth))
model.eval()
teacher_forcing_ratio = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# my_func parameters
decision_making_flag = False
weights = np.ones((prediction_window,1))
for i in range(1, prediction_window):
    weights[i][0] = weights[i-1][0] * 1.5
policy = AlpacaSimpleLong(trading_client)

async def on_receive_bar(bar): # even with multiple subscritions, bars come in one by one.
    # how to handle aftermarket, when not all symbol bars are received?
    # maybe not doing data processing in this handler
    global last_data_update_time
    last_data_update_time = time.time()

    symbol = bar['S']
    print(f"received new bars:")
    # print(bar)
    handler_start_time = time.time()
    mapped_bar = {
        BAR_MAPPING[key]: val for key, val in bar.items() if key in BAR_MAPPING
    }
    # print(mapped_bar)
    dt = mapped_bar['timestamp'].to_datetime()
    print(dt)
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


    

    print(f'Data processing completed {time.time() - handler_start_time:5.2f} seconds after receiving bar.')
    print('')

        # stream_df_indicated  = append_indicators(stream_df)
        # print(stream_df_indicated.tail(5))

        # y_pred = model()



    # stream_df.to_csv(stream_csv_path)
def account_update(): 
    
    pass
    # update status to policy when account status changes.

def my_function():
    global decision_making_flag
    if decision_making_flag:
        return
    
    global last_data_update_time, rows_buffer, symbols, wait_time
    # print('')
    print(f"Thread Waiting...{time.time() - wait_time:4.2f}", end = '\r')

    no_more_data = time.time() - last_data_update_time > 3 and len(rows_buffer) != 0
    # note that when not all symbols are refreshed, it doesn't quite matter, 
    # as the policy will make the same prediction for those not updated symbols -- i.e., policy will not do anything.
    collection_complete = len(rows_buffer) == len(symbols)

    if no_more_data or collection_complete:
        decision_making_flag = True
        if no_more_data:
            print("no more data, start processing.")
        else:
            print("buffer is full, start processing.")

        global stream_df
        if stream_df is None:
            global df
            stream_df = df.tail(200)
            
        stream_df = pd.concat([stream_df] + rows_buffer)
        print(stream_df.shape)
        rows_buffer = []

        symbol_num = len(symbols)
        
        df = stream_df.tail(3*symbol_num*hist_window).copy()
        col_lst = append_indicators(df)
        col_num = len(col_lst)
        close_idx = col_lst.index('close')
        print('close_idx: ', close_idx)

        # start making prediction
        
        global model
        with torch.no_grad():
            np2i_dict = {}
            
            x_batch_buffer = []
            col_names = df.columns.str
            groups = df.groupby('symbol')
            for symbol, df_single_symbol in groups:
                np2i_dict = norm_param_2_idx(col_names).copy()
                
                # print('dict: ',np2i_dict)
                x_df = df_single_symbol.tail(hist_window).copy()
                timestamp, x_normalized_np = normalize_data(x_df, np2i_dict)
                timestamp.reshape(1,-1,1) # z_continuous makes it
                # print(timestamp)
                x_batch = x_normalized_np.reshape(1, hist_window, input_size)


                col_idx_set = set(range(col_num))
                not_close_batch_norm_lst = list(col_idx_set - set(np2i_dict[NormParam.CloseBatch]))

                x_batch = batch_norm(x_batch, not_close_batch_norm_lst, close_idx)
                
                # print('x_batch shape: ', x_batch.shape)
                # (1, hist_window, feature_num)

                x_batch_buffer.append((symbol, timestamp, x_batch))
                
            # should I have policy make one decision for every symbol, or one decision for all symbols?
            # all symbols.
            # then I should put prediction of all symbols together and feed to policy
            
            col_idx_set = set(range(col_num))
            not_close_batch_norm_lst = list(col_idx_set - set(np2i_dict[NormParam.CloseBatch]))
            for symbol, timestamp, hist in x_batch_buffer:
                x_batch = torch.tensor(hist, dtype = torch.float32).to(device)
            
                y_pred = model(x_batch, None, teacher_forcing_ratio = 0.0)
                predictions = y_pred.clone().detach().cpu().numpy()
                global weights
                weighted_prediction = np.matmul(predictions, weights)/np.sum(weights) # prediction 1,5; weights 5,1
                weighted_prediction = weighted_prediction[0,0]
                policy.process(symbol, hist, weighted_prediction, col_lst)

                price = hist[0,-1,close_idx]
                print('price: ', price)
                alpaca_account = trading_client.get_account()
                decision = policy.decide()

                print('decision: ', decision)

        


            
                


        
        


        if alpaca_account.trading_blocked:
            print('Account is currently restricted from trading.')
        # market_order_data = create_limit_order(symbol = "AAPL", price = price, qty = 1, order_side = OrderSide.BUY, tif = TimeInForce.DAY)

        # Check how much money we can use to open new positions.
        print(f'${alpaca_account.buying_power} is available as buying power.')

        decision_making_flag = False
        wait_time = time.time()
    # Add your desired function code here


def thread_function():
    global policy
    while True:
        t = threading.Thread(target=my_function)
        t.start()
        time.sleep(.5)
        policy.update_account_status()


def main():
    global df
    df = last_week_bars(symbols, dp = data_path, download = False)
    df.sort_index(level=1, inplace=True)

    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data, feed=DataFeed.SIP)
    
    for symbol in symbols:
        stream.subscribe_bars(on_receive_bar, symbol)
        # subscribe_quote
        print(f'subscribed to {symbol}')

    thread = threading.Thread(target=thread_function)
    thread.start()

    stream.run()

    thread.join()


    print("is this printed?")

if __name__ == "__main__":
    main()