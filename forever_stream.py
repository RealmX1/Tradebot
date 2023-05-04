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

from AI.S2S import *
from AI.sim import *
from AI.data_utils import *
from AI.model_structure_param import *
from stock_data.indicators import append_indicators
from stock_data.alpaca_history_bars import get_bars


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
stream_csv_path = "data/forever_stream.csv"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# policy parameters
policy = SimpleLongShort(allow_short = False)
account = Account(100000, ['AAPL'])

async def on_receive_bar(bar):
    start_time = time.time()
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
    # global df
    # df = pd.concat([df, new_row])

    global stream_df
    if stream_df is None:
        stream_df = df.tail(200).copy()
    stream_df = pd.concat([stream_df, new_row])

    print("done after: ", time.time() - start_time)


    # stream_df_indicated  = append_indicators(stream_df)
    print(stream_df.tail(5))

    stream_df.to_csv(stream_csv_path)



    
    
    
if __name__ == "__main__":
    tf = TimeFrame.Minute

    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data, feed=DataFeed.SIP)
    for symbol in symbols:
        stream.subscribe_bars(on_receive_bar, symbol)
    stream.run()