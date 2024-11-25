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
import datetime
import msgpack

from S2S import *
from sim import *
from alpaca_history_bars import last_week_bars


# stream parmeters
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

# trade parameters


# model parameters
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

async def bar_handler(bar):
    mapped_bar = {
        BAR_MAPPING[key]: val for key, val in bar.items() if key in BAR_MAPPING
    }
    print(mapped_bar)
    dt = mapped_bar['timestamp'].to_datetime()
    print(dt)
    print(type(dt))
    formatted = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    print(formatted)


    
    
    
if __name__ == "__main__":    
    model_pth = 'lstm_updown_S2S_attention.pt'
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)
    model.load_state_dict(torch.load(model_pth))

    symbols = ['AAPL']

    df = last_week_bars(symbols)




    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data, feed=DataFeed.IEX)
    stream.run()