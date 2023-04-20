API_KEY = "AKOZFEX5F94X2SD7HQOQ"
SECRET_KEY =  "3aNqjtbPlkJv09NicPgYFXC3KUhNOR16JGGdiLet"

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
from stock_data.indicators import append_indicators
from stock_data.alpaca_history_bars import last_week_bars


# stream parmeters
symbols = ['AAPL']
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
df_csv_path = "data/df.csv"
stream_csv_path = "data/csv/stream.csv"

# model parameters
model = None
feature_num         = input_size = 23 # Number of features (i.e. columns) in the CSV file -- the time feature is removed.
hidden_size         = 200    # Number of neurons in the hidden layer of the LSTM
num_layers          = 4     # Number of layers in the LSTM
output_size         = 1     # Number of output values (closing price 1~10min from now)
prediction_window   = 5
hist_window         = 100 # using how much data from the past to make prediction?
data_prep_window    = hist_window + prediction_window # +ouput_size becuase we need to keep 10 for calculating loss


learning_rate   = 0.0001
batch_size      = 1
train_percent   = 0
num_epochs      = 0
dropout         = 0.1
teacher_forcing_ratio = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# policy parameters
policy = NaiveLong()
account = Account(100000, ['AAPL'])

async def on_receive_bar(bar):
    handler_start_time = time.time()
    mapped_bar = {
        BAR_MAPPING[key]: val for key, val in bar.items() if key in BAR_MAPPING
    }
    print(mapped_bar["timestamp"])
    dt = mapped_bar['timestamp'].to_datetime()
    formatted = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    print(dt)
    print(formatted)
    del mapped_bar['timestamp']

    new_row = pd.DataFrame(mapped_bar, index=pd.MultiIndex.from_tuples([(symbols[0], formatted)],
                                                       names=['symbol', 'timestamp']))
    global df
    global stream_df
    if stream_df is None:
        stream_df = df.tail(200)
    stream_df = pd.concat([stream_df, new_row])
    

    print(f'prediction completed {time.time() - handler_start_time} seconds after receiving bar.')

    stream_df_indicated  = append_indicators(stream_df)
    print(stream_df_indicated.tail(5))

    y_pred = model()



    stream_df.to_csv(stream_csv_path)



    
    
    
if __name__ == "__main__":    
    model_pth = 'AI/lstm_updown_S2S_attention.pt'
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    model.load_state_dict(torch.load(model_pth))

    df = last_week_bars(symbols)
    print(df.tail(5))




    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data) #, feed=DataFeed.IEX)
    for symbol in symbols:
        stream.subscribe_bars(on_receive_bar, symbol)
    stream.run()
    print("is this printed?")