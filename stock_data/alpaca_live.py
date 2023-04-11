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
import threading
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed

async def handler(data):
    print("received data: ", data)
    

def main():
    stream = StockDataStream(api_key = API_KEY, secret_key = SECRET_KEY)
    # stream = CryptoDataStream(api_key = API_KEY, secret_key = SECRET_KEY)

    stream.subscribe_bars(handler, "BABA")
    print("Start streaming")
    stream.run()
    

if __name__ == "__main__":
    main()