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
from alpaca.trading.requests import GetAssetsRequest

import datetime
import msgpack

from AI.S2S import *
from AI.sim import *
from AI.data_utils import *
from AI.model_structure_param import *
from alpaca_api.indicators import append_indicators
from alpaca_api.alpaca_history_bars import last_week_bars
from alpaca_api.alpaca_trading import * # trading client created.

# df = last_week_bars(symbols, dp = 'data/')
df = pd.read_csv('data/csv/last_week_bar_set_20230424_20230503_raw.csv', index_col = ['symbol', 'timestamp'])
col_names = df.columns
model = Seq2Seq(input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device).to(device)

arr = np.ones(prediction_window)
for i in range(1, prediction_window):
    arr[i] = arr[i-1] * weight_decay
weights = arr.reshape(prediction_window,1)

policy = AlpacaSimpleLongShort() 
symbols = ['AAPL']
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# print(indicated.tail(hist_window))



# check if the account has unfinished/just finished trade; 

def handler(new_data = None):
    for symbol in symbols:

    stream_df = df.tail(hist_window+1000)
    indicated = append_indicators(stream_df)

    x_np = normalize_data(indicated.tail(hist_window))
    x_batch = torch.tensor(x_np).view(1, hist_window, input_size).float().to(device)
    print(x_batch.shape)

    y_pred = model(x_batch, None, teacher_forcing_ratio=0)

    prediction = y_pred.clone().detach().cpu()
    weighted_prediction = (prediction * weights).sum() / weights.sum()
    
    alpaca_account = trading_client.get_account()
    decision = policy.decide('AAPL', x_batch.clone().detach().cpu(), price, weighted_prediction, col_names, alpaca_account)

    if decision[0] == 'b':
        order_side = OrderSide.BUY
    elif decision[0] == 's':
        order_side = OrderSide.SELL
    else:
        return #???
    qty = decision[1]
    create_limit_order(symbol, qty, price, order_side, tif = TimeInForce.DAY, stop = None) # probably should ask policy to provide symbol.; maybe policy should be multiple decision tuples -- one for each symbol.



def main():    

    alpaca_account = trading_client.get_account()


    if alpaca_account.trading_blocked:
        print('Account is currently restricted from trading.')
    # market_order_data = create_limit_order(symbol = "AAPL", price = price, qty = 1, order_side = OrderSide.BUY, tif = TimeInForce.DAY)

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(alpaca_account.buying_power))


if __name__ == "__main__":
    main()

# how to receive signals when an order is executed?