API_KEY = "AKOZFEX5F94X2SD7HQOQ"
SECRET_KEY =  "3aNqjtbPlkJv09NicPgYFXC3KUhNOR16JGGdiLet"

# from alpaca.data.live import StockDataStream


# wss_client = StockDataStream(API_KEY, SECRET_KEY)

# # async handler
# async def quote_data_handler(data):
#     # quote data will arrive here
#     print(data)

# wss_client.subscribe_quotes(quote_data_handler, "SPY")

# wss_client.run()
import pandas as pd
import asyncio
import threading
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from alpaca.data import Trade, Snapshot, Quote, Bar, BarSet, QuoteSet, TradeSet
import datetime
import msgpack
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


new_data_flag = True
bar_list = []
start_of_stream_flag = True
start = None

raw_data = True
columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']

# index = pd.MultiIndex.from_product([symbols, timestamps], names=['symbol', 'timestamp'])

# df = pd.DataFrame(columns=columns, index=index)

def get_bars(symbol_or_symbols, timeframe, start, end, limit):
    pass

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



    # TODO: add new row to df_raw (including last week data till now); calculate using indicator function.

    
    # mock_mapped_bar = {'open': 96.055, 'high': 96.055, 'low': 96.055, 'close': 96.055, 'volume': 100, 'timestamp': Timestamp(seconds=1681411020, nanoseconds=0), 'trade_count': 1, 'vwap': 96.055}
    


    # if type(raw_bar) != dict:
    #     print("raw_bar is not dict!!!", print(type(raw_bar)))
    # else:
    # raw_data = Bar('BABA', mapped_bar)
    # print("raw_bar: ", raw_bar)
    # print("timestamp: ", raw_bar['t'])
    # print("timestamp type: ", type(raw_bar['t']))
    # bar_list.append(raw_bar)

    # data = {"BABA": bar_list}

    # global start_of_stream_flag
    # if start_of_stream_flag:
    #     start_of_stream_flag = False
    #     # start = mapped_bar["start"]
    #     print("start_of_stream!")

    # print("bar_set: ", BarSet(data).df)
    # new_data_flag = True
    pass

async def trade_handler(raw_trade, Timeframe):
    # calcualte customized aggregate according to timeframe.
    pass

async def quote_handler(raw_quote):
    # print("new quote: ", raw_quote)
    pass

def main():
    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=raw_data)  # , data_feed=DataFeed.IEX)
    # stream = CryptoDataStream(api_key = API_KEY, secret_key = SECRET_KEY)

    symbols = ['AAPL']

    for symbol in symbols:
        stream.subscribe_bars(bar_handler, symbol)
    # stream.subscribe_quotes(quote_handler, "BABA")
    # stream.subscribe_trades(trade_handler, "BABA")
    print("Start streaming")
    stream.run()
    

if __name__ == "__main__":
    main()