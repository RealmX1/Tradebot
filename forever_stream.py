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
from datetime import datetime, timedelta
import msgpack
import sys
import csv

sys.path.append('AI')  # add the path to my_project
sys.path.append('alpaca_api')
sys.path.append('sim')

from sim import *
from AI.data_utils import *
from alpaca_api.indicators import append_indicators
from alpaca_api.alpaca_history import get_bars
from symbols import *

# stream parmeters
BAR_MAPPING = {
    "S": "symbol",
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
header = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
raw_data = True

# data parameters
df = None
stream_df = None
df_csv_path = "data/df.csv"
stream_csv_path = "data/forever_stream.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# policy parameters

async def on_receive_bar(bar):
    start_time = time.time()
    mapped_bar = {
        BAR_MAPPING[key]: val for key, val in bar.items() if key in BAR_MAPPING
    }
    print(mapped_bar["timestamp"])
    dt = mapped_bar['timestamp'].to_datetime()
    formatted = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    mapped_bar['timestamp'] = formatted
    print('bar: ', mapped_bar)
    
    with open(stream_csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header)
        csv_writer.writerow(mapped_bar)



    
    
    
if __name__ == "__main__":
    tf = TimeFrame.Minute

    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data, feed=DataFeed.SIP)
    for symbol in symbols:
        print('subscribing to: ', symbol)
        stream.subscribe_bars(on_receive_bar, symbol)
    stream.run()