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

import asyncio
import threading
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from alpaca.data import Trade, Snapshot, Quote, Bar, BarSet, QuoteSet, TradeSet
from alpaca.data.mappings import BAR_MAPPING

from alpaca.common.types import RawData

new_data_flag = True
bar_list = []
start_of_stream_flag = True
start = None


async def handler(raw_bar):
    print("raw_bar: ", raw_bar)
    if type(raw_bar) != dict:
        print("raw_bar is not dict!!!", print(type(raw_bar)))
    else:
        mapped_bar = {
                BAR_MAPPING[key]: val for key, val in raw_bar.items() if key in BAR_MAPPING
        } 
    raw_data = RawData('BABA', mapped_bar)
    print("mapped_bar: ", bar_list)
    global start_of_stream_flag
    if start_of_stream_flag:
        start_of_stream_flag = False
        # start = mapped_bar["start"]
        print("start_of_stream!")
    bar_list.append(raw_bar)
    print("bar_set: ", BarSet(bar_list).df)
    new_data_flag = True

def main():
    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, raw_data=True,)
    # stream = CryptoDataStream(api_key = API_KEY, secret_key = SECRET_KEY)

    stream.subscribe_bars(handler, "AAPL")
    print("Start streaming")
    stream.run()
    

if __name__ == "__main__":
    main()