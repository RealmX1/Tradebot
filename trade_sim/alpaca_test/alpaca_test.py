import alpaca_trade_api as tradeapi
import pandas as pd
import time

# Replace these with your own Alpaca API keys
API_KEY = "PKPTGB2VKG39X1Q4W43K"
SECRET_KEY = "pdEPRpFhpD687llTFKNv22lbya5XJ1oZ4zJhBGWQ"

# Initialize the Alpaca API client
api = tradeapi.REST(API_KEY, SECRET_KEY, api_version='v2')

# Define a list of symbols to track
symbols = ["NVDA", "AAPL", "GOOGL", "BABA", "MSFT"]

# Define an empty DataFrame to store the data
columns = ["symbol", "timestamp", "bid_price", "bid_size", "ask_price", "ask_size"]
df = pd.DataFrame(columns=columns)

# Define a function to handle streaming updates for each stock
def handle_bar(conn, channel, data):
    # Extract the relevant data from the streaming message
    symbol = data.symbol
    timestamp = data.timestamp
    bid_price = data.bid.price
    bid_size = data.bid.size
    ask_price = data.ask.price
    ask_size = data.ask.size

    # Add the data to the DataFrame
    df.loc[len(df)] = [symbol, timestamp, bid_price, bid_size, ask_price, ask_size]

# Subscribe to the streaming data for each stock
for symbol in symbols:
    api.polygon.subscribe_ticker(symbol, handle_bar)

# Wait for a few seconds to collect data
time.sleep(5)

# Print the DataFrame
print(df.head())