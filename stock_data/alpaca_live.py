API_KEY = "PKGNSI31E7XI9ACCSSVZ"
SECRET_KEY =  "yhupKUckY5vAbP7UOrkB26v4X4Gb9cdffo39V4OM"

# from alpaca.data.live import StockDataStream


# wss_client = StockDataStream(API_KEY, SECRET_KEY)

# # async handler
# async def quote_data_handler(data):
#     # quote data will arrive here
#     print(data)

# wss_client.subscribe_quotes(quote_data_handler, "SPY")

# wss_client.run()

import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

async def handler(data):
    print(data)

def main():
    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY, feed=DataFeed.IEX)

    stream.subscribe_bars(handler, "SPY", "1Min")
    stream.run()

if __name__ == "__main__":
    main()